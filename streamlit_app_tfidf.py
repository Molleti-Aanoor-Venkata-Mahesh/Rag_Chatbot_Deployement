"""
NovaTech RAG Chatbot — Streamlit UI (TF-IDF version)
=====================================================

PURPOSE
-------
This is the production/deployment version of the chatbot UI.
It replaces ChromaDB with scikit-learn TF-IDF search — a lightweight
alternative that works on free cloud tiers with limited RAM.

WHEN TO USE THIS FILE
---------------------
- Deploying to any free cloud platform (Render, Railway, Fly.io)
- When RAM is limited (< 1GB available)
- When fast startup is important (no model download required)

WHY NOT CHROMADB FOR DEPLOYMENT
---------------------------------
ChromaDB automatically downloads the all-MiniLM-L6-v2 ONNX model (79MB)
on first use and loads it into ~300MB of RAM. On free cloud tiers with
512MB total RAM, this leaves no room for Streamlit + the app itself,
causing an out-of-memory crash loop.

TF-IDF uses zero external models, ~5MB RAM, and starts in under 2 seconds.

HOW TO RUN LOCALLY
------------------
    streamlit run streamlit_app_tfidf.py
    Open: http://localhost:8501

WHAT IS TF-IDF
--------------
TF-IDF = Term Frequency — Inverse Document Frequency

It is a classical information retrieval technique from the 1970s that still
works remarkably well for structured documents like HR policies and manuals.

  TF (Term Frequency):
    How often does the query word appear in this chunk?
    A chunk about "leave policy" that mentions "leave" 5 times scores higher
    than one that mentions it once.

  IDF (Inverse Document Frequency):
    How rare is this word across ALL chunks?
    "the", "is", "and" appear everywhere — they tell us nothing.
    "probation", "maternity", "severance" are rare — they are informative.

  TF-IDF score = TF × IDF
    High score = word appears often in THIS chunk AND rarely in other chunks
    = the chunk is specifically about this topic

  Cosine Similarity:
    Each chunk is represented as a vector of TF-IDF scores (one per word).
    The query is converted to the same kind of vector.
    Cosine similarity measures the angle between the query vector and each
    chunk vector. Small angle = similar topic = relevant chunk.

CHROMADB vs TF-IDF COMPARISON
------------------------------
  Feature               ChromaDB (semantic)      TF-IDF (keyword)
  ──────────────────────────────────────────────────────────────────
  Understands synonyms  Yes ("WFH" ≈ "remote")  No (must match exactly)
  RAM usage             ~300MB                   ~5MB
  Startup time          30-60s (model download)  < 2s
  Works offline         No (downloads model)     Yes
  Best for              General text, ambiguous  Structured docs, policies
  Free deployment       Crashes                  Works perfectly

For NovaTech company documents (HR policy, engineering standards, etc.),
TF-IDF works excellently because users naturally use the same terminology
as the documents themselves.

ARCHITECTURE
------------
    User types question
          │
          ▼
    streamlit_app_tfidf.py
          │
          ├── load_rag() [cached] ──► reads _data/*.txt
          │                       ──► chunks into paragraphs
          │                       ──► TfidfVectorizer.fit_transform()
          │                           builds document-term matrix
          │
          ├── ask_rag(question)
          │       │
          │       ├── vectorizer.transform(question) ──► query vector
          │       ├── cosine_similarity(query, matrix) ──► scores
          │       ├── argsort → top 3 indices ──► retrieved chunks
          │       └── Groq API call ──► LLaMA 3.3 70B generates answer
          │
          └── Streamlit renders answer + sources + chunk dropdown
              (with similarity scores shown for each chunk)

DEPENDENCIES
------------
- streamlit         : web UI framework
- scikit-learn      : TfidfVectorizer + cosine_similarity
- numpy             : argsort for ranking chunks by similarity score
- openai            : Groq API client (OpenAI-compatible)
- python-dotenv     : loads .env for local development
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
# RAG SETUP — TF-IDF indexing
# ──────────────────────────────────────

@st.cache_resource
def load_rag():
    """
    Read all .txt files, chunk by paragraph, and build a TF-IDF index.

    @st.cache_resource ensures this function runs only ONCE per app session.
    Without caching, the entire index would be rebuilt on every message,
    which would be slow and wasteful.

    TF-IDF indexing process:
        1. Read every .txt file in _data/
        2. Split each file into paragraphs (split on blank lines)
        3. Filter out very short fragments and section dividers
        4. Pass all paragraphs to TfidfVectorizer.fit_transform():
           - fit()      : learn the vocabulary from all chunks
           - transform(): convert each chunk to a TF-IDF vector
        5. The result is a sparse matrix: rows=chunks, cols=vocab words
           Each cell = TF-IDF score for that word in that chunk

    TfidfVectorizer parameters explained:
        stop_words="english"  : removes "the", "is", "at", etc.
                                These words appear everywhere and add noise
        ngram_range=(1, 2)    : consider both single words AND word pairs
                                e.g. "work from home" generates:
                                  unigrams: "work", "home"
                                  bigrams:  "work from", "from home"
                                Bigrams improve matching of multi-word concepts
        max_features=10000    : keep only the 10,000 most informative terms
                                Caps memory usage without affecting quality

    Returns:
        chunks       : list[str] — all paragraph strings
        sources      : list[str] — source filename for each chunk (same order)
        vectorizer   : fitted TfidfVectorizer (used to transform query later)
        tfidf_matrix : sparse matrix (n_chunks × n_vocab_terms)
        client       : Groq API client
    """

    if not os.path.exists(DATA_FOLDER):
        st.error(f"_data/ folder not found at: {DATA_FOLDER}")
        st.stop()

    chunks  = []   # paragraph text
    sources = []   # corresponding source filename

    for filename in sorted(os.listdir(DATA_FOLDER)):
        if not filename.endswith(".txt"):
            continue

        with open(os.path.join(DATA_FOLDER, filename), "r") as f:
            text = f.read()

        # Split on double newline = paragraph boundary.
        # Same chunking strategy as streamlit_app.py (ChromaDB version)
        # so both versions are directly comparable.
        for para in text.strip().split("\n\n"):
            para = para.strip()

            if len(para) < 50:        # skip headers, single lines
                continue
            if para.startswith("===="):  # skip "=====" dividers
                continue

            chunks.append(para)
            sources.append(filename)

    # Build the TF-IDF document-term matrix.
    # fit_transform() does fit() + transform() in one step.
    # After this, vectorizer "knows" the vocabulary of the entire document set.
    vectorizer   = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=10000
    )
    tfidf_matrix = vectorizer.fit_transform(chunks)
    # tfidf_matrix.shape = (n_chunks, n_vocab_terms)
    # e.g. (120, 8547) for our document set

    # Groq client — same as ChromaDB version, just a different retrieval layer
    client = OpenAI(
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1"
    )

    return chunks, sources, vectorizer, tfidf_matrix, client


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
    Execute the full RAG pipeline using TF-IDF retrieval.

    The overall flow is identical to the ChromaDB version:
    Retrieve → Augment → Generate

    The only difference is HOW retrieval works:
    - ChromaDB: neural embeddings (semantic meaning)
    - TF-IDF:   word frequency scores (keyword matching)

    Args:
        question : the user's natural language question
        top_k    : number of chunks to retrieve (default: 3)

    Returns:
        dict with keys:
            "answer"  : str — the LLM-generated answer
            "sources" : list[str] — deduplicated source filenames
            "chunks"  : list[tuple] — [(chunk_text, source_file, score), ...]
                        The score is the cosine similarity (0.0 to 1.0)

    Step 1 — Vectorise the question:
        vectorizer.transform([question]) converts the question into a TF-IDF
        vector using the SAME vocabulary learned during indexing.
        Words in the question that weren't in any document get score 0.

    Step 2 — Compute cosine similarity:
        cosine_similarity(query_vec, tfidf_matrix) returns a score for every
        chunk. Score 1.0 = perfectly similar, 0.0 = no shared words.
        .flatten() converts the (1, n_chunks) matrix to a 1D array.

    Step 3 — Rank and select:
        np.argsort(similarities) returns indices that would sort the array
        ascending. [::-1] reverses to descending. [:top_k] takes the first 3.
        Result: indices of the 3 most relevant chunks.

    Step 4 — Build augmented prompt:
        Same as ChromaDB version — inject retrieved chunks as context.

    Step 5 — Generate:
        Same Groq API call as ChromaDB version.
    """

    # Step 1 — convert question to TF-IDF vector
    # transform() (not fit_transform) — uses the existing vocabulary, not a new one
    question_vec = vectorizer.transform([question])

    # Step 2 — score every chunk against the question
    # Result shape: (1, n_chunks) — one row (the query) vs all chunks
    similarities = cosine_similarity(question_vec, tfidf_matrix).flatten()

    # Step 3 — rank chunks by similarity, take top_k
    # argsort returns indices that sort ascending → reverse → slice
    top_indices = np.argsort(similarities)[::-1][:top_k]

    retrieved_chunks  = [chunks[i]           for i in top_indices]
    retrieved_sources = [sources[i]          for i in top_indices]
    retrieved_scores  = [float(similarities[i]) for i in top_indices]

    # Step 4 — build augmented prompt
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
        # dict.fromkeys preserves insertion order while deduplicating
        # (unlike set() which loses ordering)
        "sources": list(dict.fromkeys(retrieved_sources)),
        # Each tuple: (chunk_text, source_filename, similarity_score)
        # The score is shown in the UI so users can see retrieval confidence
        "chunks":  list(zip(retrieved_chunks, retrieved_sources, retrieved_scores))
    }

# ──────────────────────────────────────
# RENDER CHAT HISTORY
# ──────────────────────────────────────

# Redraw all previous messages on every re-run
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant" and msg.get("sources"):
            st.caption(f"Sources: {', '.join(msg['sources'])}")

            # Collapsed chunk viewer — shows text AND similarity score
            # The score teaches students how confident the retrieval was
            # Score > 0.3 = strong match | Score < 0.05 = weak/no match
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
