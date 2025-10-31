import os
import streamlit as st
from dotenv import load_dotenv
from src.vectordb import VectorDB
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatGoogleGenerativeAI

# Load API Key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
if not api_key:
    st.error("🚨 Please set your GOOGLE_API_KEY in Streamlit Secrets before running.")
    st.stop()

# Initialize LLM (Gemini)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)

# Initialize Vector DB and auto-load datasets
st.set_page_config(page_title="Africa RAG Assistant 🌍", layout="wide")
st.title("🌍 Africa RAG Assistant (ReadyTensor Project 1)")
st.write("An AI assistant trained on African datasets — built with Gemini API and ChromaDB.")

vdb = VectorDB()
if not vdb.has_data():
    with st.spinner("🧠 Loading African datasets from Hugging Face... (first time only)"):
        vdb.ingest_african_datasets()
    st.success("✅ African datasets successfully loaded into vector DB!")

# Chat interface
query = st.text_input("💬 Ask about Africa:", placeholder="e.g. Explain ECOWAS role in West Africa")
if query:
    with st.spinner("🔎 Retrieving context..."):
        results = vdb.search(query, n_results=4)
        context = "\n".join(results["documents"])

    prompt_template = ChatPromptTemplate.from_template(
        "You are an expert on African development and culture. "
        "Use the following context to answer clearly and factually.\n\nContext:\n{context}\n\nQuestion:\n{question}"
    )
    prompt = prompt_template.format(context=context, question=query)

    with st.spinner("🤖 Thinking..."):
        response = llm.predict(prompt)

    st.markdown("### 🧠 Answer")
    st.write(response)
    st.markdown("---")
    st.markdown("### 📚 Context Used")
    st.text(context)
