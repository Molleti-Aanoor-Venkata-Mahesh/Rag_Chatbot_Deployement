# NovaTech RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about company
documents. Ask it about HR policies, product details, engineering standards, or
onboarding procedures — it searches the actual documents and generates a grounded answer.

Built with: FastAPI, Streamlit, ChromaDB, Groq (LLaMA 3.3 70B), Docker.

---

## What is in this repo

```
novatech-rag-chatbot/
├── streamlit_app.py     # Option A: Streamlit chat UI (recommended for demos)
├── main.py              # Option B: FastAPI backend API
├── Dockerfile           # Container definition (runs Streamlit by default)
├── requirements.txt     # Pinned Python dependencies
├── .env                 # Local secrets — never committed (gitignored)
├── .gitignore
└── _data/               # Company documents the RAG system searches
    ├── company_hr_policy.txt
    ├── engineering_standards.txt
    ├── onboarding_guide.txt
    ├── product_knowledge_base.txt
    └── security_policy.txt
```

---

## How it works

```
User types a question
        |
        v
ChromaDB searches _data/ documents
(finds top 3 most relevant paragraphs)
        |
        v
Those paragraphs are sent to Groq LLM as context
        |
        v
LLM generates a grounded answer
        |
        v
Answer + source documents + raw chunks displayed to user
```

Documents are chunked by paragraph at startup and stored in an in-memory ChromaDB
collection. No external database needed — everything resets on each restart.

---

## Prerequisites

- Python 3.11+
- A Groq API key — free at https://console.groq.com
- Git

---

## Local Setup

### Step 1: Clone the repo

```bash
git clone https://github.com/Ch-Balaji/novatech-rag-chatbot_demo.git
cd novatech-rag-chatbot_demo
```

### Step 2: Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Mac / Linux
# .venv\Scripts\activate         # Windows
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Add your Groq API key

Create a `.env` file in the project root:

```
groq_api_key=gsk_xxxxxxxxxxxxxxxxxxxxxxxx
```

Get your key at https://console.groq.com — it is free to sign up.

---

## Running Locally

### Option A: Streamlit UI (recommended)

Gives you a proper chat interface in the browser with conversation history
and a dropdown to inspect the retrieved document chunks.

```bash
streamlit run streamlit_app.py
```

Open: http://localhost:8501

You will see:
- A green confirmation that documents were indexed
- A chat input at the bottom
- Answers with source filenames and a collapsible "View retrieved chunks" section

### Option B: FastAPI backend only

Runs a pure JSON API with no visual interface. Use this to understand how a
backend API works, or if you want to build your own frontend later.

```bash
uvicorn main:app --reload --port 8000
```

Open: http://localhost:8000/docs

The `/docs` page is FastAPI's built-in interactive UI (Swagger). To test:
1. Click `POST /chat`
2. Click `Try it out`
3. Change the question in the request body
4. Click `Execute`
5. See the JSON response below

To call it from Python:

```python
import requests

response = requests.post(
    "http://localhost:8000/chat",
    json={"question": "What is the work from home policy?"}
)
print(response.json())
```

---

## Deploying to Render (Free)

Render is a cloud platform that runs Docker containers for free (with some limits).
The Dockerfile in this repo is already configured for Render.

### Step 1: Push your code to GitHub

```bash
# Inside the project folder, with venv active:

git init
git branch -m main
git add .
git commit -m "Initial deploy: NovaTech RAG chatbot with Streamlit UI"

# Set remote with your GitHub PAT for authentication
# Replace YOUR_USERNAME and YOUR_PAT with your actual values
git remote add origin https://YOUR_USERNAME:YOUR_PAT@github.com/YOUR_USERNAME/novatech-rag-chatbot_demo.git

git push -u origin main
```

Getting a GitHub PAT (Personal Access Token):
1. Go to https://github.com/settings/tokens/new
2. Note: `novatech-deploy`
3. Expiration: 90 days
4. Scopes: tick `repo`
5. Click Generate token — copy it immediately

### Step 2: Create a Render account

Go to https://render.com and sign up (free — no credit card needed for basic use).

### Step 3: Create a new Web Service

1. Click `New` → `Web Service`
2. Connect your GitHub account when prompted
3. Select the `novatech-rag-chatbot_demo` repository
4. Fill in the settings:

| Setting | Value |
|---------|-------|
| Name | `novatech-rag-chatbot` |
| Region | Frankfurt (or closest to you) |
| Branch | `main` |
| Runtime | `Docker` |
| Instance Type | `Free` |

5. Click `Create Web Service`

### Step 4: Add your Groq API key as a secret

Never put API keys in code or in the repo. Render injects them as environment variables.

1. On your service page, click `Environment`
2. Click `Add Environment Variable`
3. Key: `groq_api_key`
4. Value: your actual Groq API key (`gsk_...`)
5. Click `Save Changes`

Render will automatically redeploy with the secret available.

### Step 5: Wait for the build (~3-5 minutes)

Render will:
1. Pull your code from GitHub
2. Build the Docker image (installs all packages from `requirements.txt`)
3. Start the container running `streamlit run streamlit_app.py`

Watch the build logs on the Render dashboard. You will see:
```
Indexed 120 chunks from /app/_data
You can now view your Streamlit app in your browser.
```

### Step 6: Access your live app

Once deployed, your app is live at:
```
https://novatech-rag-chatbot.onrender.com
```

Share this URL — anyone can open it and chat with the NovaTech assistant.

### Updating the app

Every time you push to GitHub, Render automatically rebuilds and redeploys:

```bash
# Make your changes, then:
git add .
git commit -m "Update: describe what changed"
git push
```

---

## Troubleshooting

| Problem | Likely cause | Fix |
|---------|-------------|-----|
| `Error loading ASGI app` | Running uvicorn from the wrong folder | `cd` into the project folder first |
| `_data/ folder not found` | Running from wrong directory | Make sure `_data/` is next to the script |
| `groq_api_key` not found | `.env` file missing or wrong key name | Check `.env` has `groq_api_key=gsk_...` |
| Push rejected (auth failed) | PAT expired or wrong scope | Create a new classic PAT with `repo` scope |
| Render build fails | Check Render logs tab | Usually a missing package or wrong port |
| App loads but answers are wrong | LLM answering from memory, not docs | Check `_data/` files were committed to GitHub |
| Cold start on Render (30s delay) | Free tier spins down after 15min idle | Normal behaviour — first request is slow |

---

## Key concepts covered

| Concept | Where to see it |
|---------|----------------|
| RAG pipeline | `load_rag()` in `streamlit_app.py` |
| ChromaDB vector search | `collection.query()` in `ask_rag()` |
| Groq API call | `groq_client.chat.completions.create()` |
| Streamlit session state | `st.session_state.messages` |
| Streamlit caching | `@st.cache_resource` on `load_rag()` |
| FastAPI request/response schema | `ChatRequest` / `ChatResponse` in `main.py` |
| Docker containerisation | `Dockerfile` |
| Environment secrets | `.env` locally, Render dashboard in production |

---

## Notes on the free Render tier

- The app **spins down after 15 minutes of inactivity** — the first request after idle takes ~30 seconds to wake up
- Free tier has 512MB RAM — sufficient for this app
- To keep it always awake, upgrade to Render's paid tier or use an uptime monitor like https://uptimerobot.com to ping it every 10 minutes
