
FROM python:3.11-slim

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy the RAG chatbot application
COPY rag_app.py .

# Copy company documents
COPY _data/ ./_data/

# Start Streamlit application
CMD ["streamlit", "run", "rag_app.py", "--server.port=10000", "--server.address=0.0.0.0"]
