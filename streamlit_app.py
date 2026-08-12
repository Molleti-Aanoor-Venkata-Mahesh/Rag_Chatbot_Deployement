"""
NovaTech RAG Chatbot — Streamlit UI
Self-contained: RAG logic lives here, no FastAPI needed.

Run with: streamlit run streamlit_app.py
"""

import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
import chromadb

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
# RAG SETUP — cached so it only runs once per session
# st.cache_resource keeps the ChromaDB collection alive
# across re-runs (every time a user types something,
# Streamlit re-runs the script — without caching this
# would re-index the documents on every single message)
# ──────────────────────────────────────
@st.cache_resource
def load_rag():
    """Index all .txt files and return (chroma_collection, openai_client)."""

    # Validate data folder exists before trying to read it
    if not os.path.exists(DATA_FOLDER):
        st.error(f"_data/ folder not found at: {DATA_FOLDER}")
        st.stop()

    # Build Groq client (OpenAI-compatible)
    client = OpenAI(
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1"
    )

    # Build in-memory ChromaDB collection
    chroma = chromadb.EphemeralClient()

    try:
        chroma.delete_collection("docs")
    except Exception:
        pass

    collection = chroma.create_collection("docs")

    # Read and chunk every .txt file in _data/
    all_chunks    = []
    all_ids       = []
    all_metadatas = []
    chunk_id      = 0

    for filename in sorted(os.listdir(DATA_FOLDER)):
        if not filename.endswith(".txt"):
            continue

        with open(os.path.join(DATA_FOLDER, filename), "r") as f:
            text = f.read()

        # Split into paragraphs — simple but effective chunking strategy
        for para in text.strip().split("\n\n"):
            para = para.strip()

            # Skip very short fragments and section dividers
            if len(para) < 50:
                continue
            if para.startswith("===="):
                continue

            all_chunks.append(para)
            all_ids.append(f"chunk_{chunk_id}")
            all_metadatas.append({"source": filename, "text": para})
            chunk_id += 1

    collection.add(
        documents=all_chunks,
        ids=all_ids,
        metadatas=all_metadatas
    )

    return collection, client, chunk_id


# Load once — Streamlit shows a spinner while indexing
with st.spinner("Loading and indexing company documents..."):
    collection, groq_client, total_chunks = load_rag()

# Subtle confirmation that indexing worked
st.success(f"Ready — {total_chunks} document chunks indexed.", icon="✅")
st.divider()

# ──────────────────────────────────────
# CHAT HISTORY
# st.session_state persists values across re-runs
# within the same browser session
# ──────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role", "content", "sources", "chunks"}

# ──────────────────────────────────────
# RAG FUNCTION
# ──────────────────────────────────────
def ask_rag(question: str) -> dict:
    """
    Given a question:
    1. Retrieve the top 3 most relevant chunks from ChromaDB
    2. Build a prompt with those chunks as context
    3. Call Groq LLM to generate an answer
    Returns: {"answer": str, "sources": list, "chunks": list}
    """

    # Step 1 — retrieve
    results = collection.query(query_texts=[question], n_results=3)

    chunks   = results["documents"][0]       # list of 3 text chunks
    sources  = [m["source"] for m in results["metadatas"][0]]  # filenames

    # Step 2 — build prompt
    context = "\n\n---\n\n".join(chunks)

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

    # Step 3 — generate
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.2
    )

    answer = response.choices[0].message.content

    return {
        "answer":  answer,
        "sources": list(set(sources)),       # deduplicated filenames
        "chunks":  list(zip(chunks, sources)) # [(chunk_text, source_file), ...]
    }

# ──────────────────────────────────────
# RENDER CHAT HISTORY
# ──────────────────────────────────────
for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Only assistant messages have sources and chunks
        if msg["role"] == "assistant" and msg.get("sources"):

            # Source files pill
            st.caption(f"Sources: {', '.join(msg['sources'])}")

            # Retrieved chunks — collapsed by default so they don't clutter
            with st.expander("View retrieved document chunks"):
                for i, (chunk_text, source_file) in enumerate(msg["chunks"], 1):
                    st.markdown(f"**Chunk {i} — `{source_file}`**")
                    st.info(chunk_text)

# ──────────────────────────────────────
# CHAT INPUT
# ──────────────────────────────────────
question = st.chat_input("Ask a question about NovaTech policies or products...")

if question:

    # Show user message immediately
    with st.chat_message("user"):
        st.markdown(question)

    # Save to history
    st.session_state.messages.append({"role": "user", "content": question})

    # Call RAG and show response
    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer..."):
            result = ask_rag(question)

        st.markdown(result["answer"])
        st.caption(f"Sources: {', '.join(result['sources'])}")

        # Dropdown showing the exact chunks that were retrieved
        with st.expander("View retrieved document chunks"):
            for i, (chunk_text, source_file) in enumerate(result["chunks"], 1):
                st.markdown(f"**Chunk {i} — `{source_file}`**")
                st.info(chunk_text)

    # Save assistant response to history
    st.session_state.messages.append({
        "role":    "assistant",
        "content": result["answer"],
        "sources": result["sources"],
        "chunks":  result["chunks"]
    })
