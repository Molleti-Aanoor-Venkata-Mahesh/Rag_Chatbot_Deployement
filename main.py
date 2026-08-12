"""
NovaTech RAG Chatbot — FastAPI Server
The simplest possible API that wraps our RAG system.

Run with: uvicorn main:app --reload
Test at:  http://localhost:8000/docs
"""

import os
import uuid
import logging
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import chromadb

# ──────────────────────────────────────
# CONFIG
# ──────────────────────────────────────
load_dotenv()
GROQ_API_KEY = os.getenv("groq_api_key")

# ──────────────────────────────────────
# LOGGING
# ──────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

# ──────────────────────────────────────
# PYDANTIC SCHEMAS (request & response shape)
# ──────────────────────────────────────
class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    request_id: str

# ──────────────────────────────────────
# RAG SETUP (runs once when server starts)
# ──────────────────────────────────────
client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")

# EphemeralClient = in-memory only (no persistence needed — we re-index on every start)
chroma_client = chromadb.EphemeralClient()

def load_and_index_documents():
    """Load all .txt files from _data/ and store in ChromaDB."""
    # _data/ sits next to main.py at the repo root
    data_folder = os.path.join(os.path.dirname(__file__), "_data")
    
    if not os.path.exists(data_folder):
        raise RuntimeError(f"_data/ folder not found at: {data_folder}")
    
    try:
        chroma_client.delete_collection("docs")
    except:
        pass
    
    collection = chroma_client.create_collection(name="docs")
    
    all_chunks = []
    all_ids = []
    all_metadatas = []
    chunk_id = 0
    
    for filename in os.listdir(data_folder):
        if not filename.endswith(".txt"):
            continue
        
        filepath = os.path.join(data_folder, filename)
        with open(filepath, "r") as f:
            text = f.read()
        
        # Chunk by paragraph
        paragraphs = text.strip().split("\n\n")
        for para in paragraphs:
            para = para.strip()
            if len(para) < 50:
                continue
            if para.startswith("===="):
                continue
            
            all_chunks.append(para)
            all_ids.append(f"chunk_{chunk_id}")
            all_metadatas.append({"source": filename})
            chunk_id += 1
    
    collection.add(documents=all_chunks, ids=all_ids, metadatas=all_metadatas)
    logger.info(f"Indexed {chunk_id} chunks from {data_folder}")
    return collection

collection = load_and_index_documents()

# ──────────────────────────────────────
# RAG FUNCTION
# ──────────────────────────────────────
def ask_rag(question):
    """Retrieve relevant chunks and generate answer."""
    # Retrieve
    results = collection.query(query_texts=[question], n_results=3)
    chunks = results["documents"][0]
    sources = [m["source"] for m in results["metadatas"][0]]
    
    # Build prompt
    context = "\n\n".join(chunks)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful company assistant. Answer questions using ONLY "
                "the provided context. If the context doesn't contain the answer, "
                "say 'I don't have enough information to answer this.' Be concise."
            )
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}"
        }
    ]
    
    # Generate
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.2
    )
    
    return response.choices[0].message.content, list(set(sources))

# ──────────────────────────────────────
# FASTAPI APP
# ──────────────────────────────────────
app = FastAPI(title="NovaTech RAG Chatbot", version="1.0")

@app.get("/")
def home():
    """Health check — confirms the server is running."""
    return {"status": "running", "message": "NovaTech RAG Chatbot is live!"}

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Ask a question about company documents."""
    request_id = str(uuid.uuid4())[:8]
    
    logger.info(f"[{request_id}] Question: {request.question}")
    
    answer, sources = ask_rag(request.question)
    
    logger.info(f"[{request_id}] Sources: {sources}")
    
    return ChatResponse(
        answer=answer,
        sources=sources,
        request_id=request_id
    )
