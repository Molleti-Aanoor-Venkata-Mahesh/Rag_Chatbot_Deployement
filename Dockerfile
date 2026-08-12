FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (cached layer — only rebuilds if requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY streamlit_app.py .
COPY main.py .

# Copy the company documents the RAG system searches through
COPY _data/ ./_data/

# 0.0.0.0 makes Streamlit accessible outside the container (required for Render)
# Port 10000 is Render's default
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=10000", "--server.address=0.0.0.0"]
