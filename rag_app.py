
import os
import chromadb
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv


# 1. Streamlit page configuration

st.set_page_config(
    page_title="Mahesh RAG Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("Welcome to Mahesh RAG Chatbot")
st.write("Ask questions about company documents.")


# 2. Connect to Ollama
load_dotenv()
llm = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY")
)


# 3. Load documents into ChromaDB

@st.cache_resource
def load_chroma_collection():

    db = chromadb.EphemeralClient()

    collection = db.create_collection(
        name="company_docs"
    )

    DATA_FOLDER = "_data"

    documents = []
    ids = []
    metadatas = []

    for filename in os.listdir(DATA_FOLDER):

        if filename.endswith(".txt"):

            file_path = os.path.join(DATA_FOLDER, filename)

            with open(file_path, "r", encoding="utf-8") as file:

                content = file.read()

                paragraphs = content.split("\n\n")

                for index, paragraph in enumerate(paragraphs):

                    paragraph = paragraph.strip()

                    if len(paragraph) < 50:
                        continue

                    if paragraph.startswith("==="):
                        continue

                    documents.append(paragraph)

                    ids.append(f"{filename}_{index}")

                    metadatas.append({
                        "source": filename
                    })

    if documents:

        collection.add(
            documents=documents,
            ids=ids,
            metadatas=metadatas
        )

    return collection


# 4. Initialize ChromaDB collection

collection = load_chroma_collection()

st.success(f"Documents loaded: {collection.count()}")


# 5. Retrieve documents and generate an answer

def ask_question(question):

    results = collection.query(
        query_texts=[question],
        n_results=min(3, collection.count())
    )

    retrieved_docs = results["documents"][0]

    retrieved_sources = results["metadatas"][0]

    if not retrieved_docs:
        return "I don't know.", []

    context = "\n\n".join(retrieved_docs)

    prompt = f"""
You are a company information assistant.

Instructions:
1. Answer using only the provided context.
2. Give clear and simple answers.
3. Do not invent information.
4. If the answer is unavailable, say "I don't know."

Context:
{context}

Question:
{question}

Answer:
"""

    response = llm.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    answer = response.choices[0].message.content

    sources = list(set(
        item["source"] for item in retrieved_sources
    ))

    return answer, sources


# 6. Chatbot user interface

question = st.text_input(
    "Ask your company chatbot:"
)

if st.button("Submit"):

    if question.strip():

        with st.spinner("Searching documents and generating answer..."):

            answer, sources = ask_question(question)

        st.subheader("Chatbot Answer")

        st.write(answer)

        if sources:

            st.subheader("Sources")

            for source in sources:
                st.write(f"- {source}")

    else:

        st.warning("Please enter a question.")
