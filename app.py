import streamlit as st
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from src.vectordb import VectorDB

# Load environment variables
load_dotenv()

# Streamlit page setup
st.set_page_config(page_title="Africa Knowledge RAG", page_icon="🌍", layout="wide")
st.title("🌍 Africa Knowledge RAG - v1")
st.markdown("### A retrieval-augmented AI assistant for African data sources.")

# Load your Google API Key
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("❌ GOOGLE_API_KEY not found in environment variables.")
    st.stop()

# Initialize model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)

# Initialize vector database
try:
    vectordb = VectorDB()
    if not vectordb.has_data():
        with st.spinner("Loading African datasets into vector store..."):
            vectordb.ingest_african_datasets()
except Exception as e:
    st.error(f"Error initializing vector database: {e}")
    st.stop()

# --- Persistent Chat State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar: Chat history
with st.sidebar:
    st.markdown("### 💬 Chat History")
    for i, chat in enumerate(st.session_state.chat_history):
        st.markdown(f"**You:** {chat['user']}")
        st.markdown(f"**AI:** {chat['bot']}")
        st.markdown("---")

# --- Main Chat Section ---
st.markdown("#### Ask your question about Africa 🌍")
query = st.text_input("Enter your question:")
send_btn = st.button("Ask")

if send_btn and query:
    with st.spinner("Thinking..."):
        try:
            # Retrieve context
            docs = vectordb.search(query, n_results=3)
            context = "\n\n".join(docs["documents"])

            # Build prompt
            prompt = ChatPromptTemplate.from_template("""
            You are an AI assistant specialized in African research, history, and data.
            Use the provided context to answer clearly, concisely, and accurately.

            Context:
            {context}

            Question:
            {question}
            """)

            chain_input = {"context": context, "question": query}
            formatted_prompt = prompt.format(**chain_input)
            response = llm.invoke(formatted_prompt)

            answer = response.content

            # Store in session
            st.session_state.chat_history.append({"user": query, "bot": answer})

            # Display
            st.success("✅ Answer:")
            st.markdown(f"<div style='height:300px; overflow-y:auto; padding:10px; background-color:#f8f9fa; border-radius:8px;'>{answer}</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"⚠️ Error: {e}")

# Footer
st.markdown("---")
st.caption("Developed with ❤️ for the Africa RAG Project v1 (AAIDC-ready)")
