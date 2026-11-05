FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY streamlit_app_tfidf.py .

COPY _data/ ./_data/

CMD ["streamlit", "run", "streamlit_app_tfidf.py", "--server.port=10000", "--server.address=0.0.0.0"]