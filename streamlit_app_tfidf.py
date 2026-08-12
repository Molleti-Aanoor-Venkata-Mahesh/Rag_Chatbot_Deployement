"""
NovaTech RAG Chatbot — Streamlit UI (TF-IDF version)
-----------------------------------------------------
This version replaces ChromaDB with scikit-learn TF-IDF search.

WHY: ChromaDB downloads a 79MB ONNX embedding model at runtime,
     which exceeds the 512MB RAM limit on free cloud tiers and causes
     a crash loop. TF-IDF needs no model download, uses ~5MB RAM,
     and starts in under 2 seconds — making it suitable for free deployment.

WHAT IS TF-IDF:
  TF-IDF (Term Frequency - Inverse Document Frequency) is a classic
  information retrieval technique. It finds the most relevant chunks by
  scoring how often your query words appear in each chunk (TF) while
  down-weighting words that appear in every chunk anyway (IDF).

  ChromaDB uses neural embeddings — it understands meaning and synonyms.
  TF-IDF uses exact word matching — it finds chunks with the same words.

  For 120 chunks of company documents, TF-IDF works very well because:
  - Users ask about specific policies using the same words in the documents
  - The documents are not ambiguous or metaphorical
  - Speed and reliability matter more than semantic nuance here

Run locally:  streamlit run streamlit_app_tfidf.py
Deploy:       This file is what the Dockerfile points to for production
"""

import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ──────────────────────────────────────
# CONFIG
# ──────────────────────────────────────
load_dotenv()
GROQ_API_KEY = os.getenv("groq_api_key")
DATA_FOLDER  = os.path.join(os.path.dirname(__file__), "_data")

# ──────────────────────────────────────
# PAGE SETUP
# ──────────────────────────────────────
st.set_page_config(
    page_title="NovaTech Assistant",
    page_icon="💬",
    layout="centered"
)

st.title("NovaTech Internal Assistant")
st.caption("Ask anything about company policies, products, or procedures.")
st.divider()

# ──────────────────────────────────────
# RAG SETUP — TF-IDF version
# @st.cache_resource runs once per session and keeps the index in memory
# ──────────────────────────────────────
@st.cache_resource
def load_rag():
    """
    Read all .txt files, chunk by paragraph, build a TF-IDF index.

    Returns:
        chunks     — list of all paragraph strings
        sources    — list of source filenames (one per chunk, same order)
        vectorizer — fitted TfidfVectorizer
        tfidf_matrix — the document-term matrix (chunks x vocab)
        client     — Groq API client
    """

    if not os.path.exists(DATA_FOLDER):
        st.error(f"_data/ folder not found at: {DATA_FOLDER}")
        st.stop()

    chunks  = []
    sources = []

    # Read and chunk every .txt file in _data/
    for filename in sorted(os.listdir(DATA_FOLDER)):
        if not filename.endswith(".txt"):
            continue

        with open(os.path.join(DATA_FOLDER, filename), "r") as f:
            text = f.read()

        # Split into paragraphs — same chunking strategy as the ChromaDB version
        for para in text.strip().split("\n\n"):
            para = para.strip()

            # Skip very short fragments and section dividers
            if len(para) < 50:
                continue
            if para.startswith("===="):
                continue

            chunks.append(para)
            sources.append(filename)

    # Build TF-IDF index over all chunks
    # This converts every chunk into a vector of word importance scores
    vectorizer  = TfidfVectorizer(
        stop_words="english",   # ignore common words like "the", "is", "at"
        ngram_range=(1, 2),     # consider single words AND two-word phrases
        max_features=10000      # cap vocabulary size for memory efficiency
    )
    tfidf_matrix = vectorizer.fit_transform(chunks)

    # Groq client — OpenAI-compatible API
    client = OpenAI(
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1"
    )

    return chunks, sources, vectorizer, tfidf_matrix, client


# Load once — show spinner while indexing
with st.spinner("Loading and indexing company documents..."):
    chunks, sources, vectorizer, tfidf_matrix, groq_client = load_rag()

st.success(f"Ready — {len(chunks)} document chunks indexed.", icon="✅")
st.divider()

# ──────────────────────────────────────
# CHAT HISTORY
# ──────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ──────────────────────────────────────
# RAG FUNCTION — TF-IDF retrieval
# ──────────────────────────────────────
def ask_rag(question: str, top_k: int = 3) -> dict:
    """
    Given a question:
    1. Convert the question to a TF-IDF vector
    2. Compute cosine similarity against all chunk vectors
    3. Pick the top_k most similar chunks
    4. Send those chunks as context to Groq LLM
    5. Return answer + sources + raw chunks

    Cosine similarity measures the angle between two vectors.
    A score of 1.0 = identical direction (very relevant).
    A score of 0.0 = perpendicular (no shared words at all).
    """

    # Step 1 — vectorise the question using the same vocabulary
    question_vec = vectorizer.transform([question])

    # Step 2 — compute similarity between question and every chunk
    similarities = cosine_similarity(question_vec, tfidf_matrix).flatten()

    # Step 3 — get indices of top_k highest similarity scores
    top_indices = np.argsort(similarities)[::-1][:top_k]

    retrieved_chunks   = [chunks[i]  for i in top_indices]
    retrieved_sources  = [sources[i] for i in top_indices]
    retrieved_scores   = [float(similarities[i]) for i in top_indices]

    # Step 4 — build prompt with retrieved context
    context = "\n\n---\n\n".join(retrieved_chunks)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful NovaTech company assistant. "
                "Answer questions using ONLY the provided context. "
                "If the context does not contain enough information, "
                "say 'I don't have enough information to answer this.' "
                "Be concise and direct."
            )
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}"
        }
    ]

    # Step 5 — generate answer via Groq
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.2
    )

    answer = response.choices[0].message.content

    return {
        "answer":  answer,
        "sources": list(dict.fromkeys(retrieved_sources)),  # deduplicated, order preserved
        "chunks":  list(zip(retrieved_chunks, retrieved_sources, retrieved_scores))
        # each item: (chunk_text, source_file, similarity_score)
    }

# ──────────────────────────────────────
# RENDER CHAT HISTORY
# ──────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant" and msg.get("sources"):
            st.caption(f"Sources: {', '.join(msg['sources'])}")

            with st.expander("View retrieved document chunks"):
                for i, (chunk_text, source_file, score) in enumerate(msg["chunks"], 1):
                    st.markdown(f"**Chunk {i} — `{source_file}`** (similarity: {score:.3f})")
                    st.info(chunk_text)

# ──────────────────────────────────────
# CHAT INPUT
# ──────────────────────────────────────
question = st.chat_input("Ask a question about NovaTech policies or products...")

if question:

    with st.chat_message("user"):
        st.markdown(question)

    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer..."):
            result = ask_rag(question)

        st.markdown(result["answer"])
        st.caption(f"Sources: {', '.join(result['sources'])}")

        # Dropdown showing retrieved chunks with their similarity scores
        with st.expander("View retrieved document chunks"):
            for i, (chunk_text, source_file, score) in enumerate(result["chunks"], 1):
                st.markdown(f"**Chunk {i} — `{source_file}`** (similarity: {score:.3f})")
                st.info(chunk_text)

    st.session_state.messages.append({
        "role":    "assistant",
        "content": result["answer"],
        "sources": result["sources"],
        "chunks":  result["chunks"]
    })
