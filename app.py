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
st.markdown(
    """
    <style>
    body {background-color: #0e1117; color: white;}
    .main {padding: 2rem;}
    .stTextInput > div > div > input {
        background-color: #1c1f26;
        color: white;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #007bff;
        color: white;
        border-radius: 10px;
    }
    .chat-box {
        max-height: 400px;
        overflow-y: auto;
        background-color: #1e1e1e;
        padding: 15px;
        border-radius: 10px;
        margin-top: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Header
st.title("🌍 Africa RAG Project - v1")
st.caption("A retrieval-augmented AI model for African data sources.")

# Persistent chat history
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Load Google API Key
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("❌ GOOGLE_API_KEY not found in environment variables.")
    st.stop()

# Initialize Model
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-latest",
    google_api_key=api_key,
    api_version="v1"
)

# Load VectorDB
try:
    vectordb = VectorDB()
    if not vectordb.has_data():
        with st.spinner("Building vector database from African datasets..."):
            vectordb.ingest_african_datasets()
except Exception as e:
    st.error(f"Error loading vector database: {e}")
    st.stop()

# User input
query = st.text_input("Ask your question about Africa 🌍:")

if st.button("Search") and query:
    with st.spinner("Thinking..."):
        try:
            docs = vectordb.similarity_search(query, n_results=3)
            context = "\n\n".join([d["page_content"] for d in docs])

            prompt = ChatPromptTemplate.from_template("""
            You are an AI assistant specialized in African research and cultural context.
            Use the provided context to answer concisely and accurately.

            Context:
            {context}

            Question:
            {question}
            """)

            formatted_prompt = prompt.format(context=context, question=query)
            response = llm.invoke(formatted_prompt)

            # Add to chat history
            st.session_state.chat_history.append({"user": query, "bot": response.content})

        except Exception as e:
            st.error(f"An error occurred: {e}")

# Chat display
if st.session_state.chat_history:
    st.markdown("<div class='chat-box'>", unsafe_allow_html=True)
    for chat in reversed(st.session_state.chat_history[-10:]):  # last 10 only
        st.markdown(f"🧑‍💻 **You:** {chat['user']}")
        st.markdown(f"🤖 **AI:** {chat['bot']}")
        st.markdown("---")
    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption("Developed with ❤️ for the Africa RAG Project v1")
