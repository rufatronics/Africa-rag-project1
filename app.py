import streamlit as st
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from src.vectordb import VectorDB  # your local vector store helper

# Load environment variables
load_dotenv()

# Streamlit page setup
st.set_page_config(page_title="Africa Knowledge RAG", page_icon="🌍")
st.title("🌍 Africa RAG Project - v1")
st.write("A retrieval-augmented AI model for African data sources.")

# Load your Google API Key
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("❌ GOOGLE_API_KEY not found in environment variables.")
    st.stop()

# Initialize Model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)

# Load VectorDB
try:
    vectordb = VectorDB()
except Exception as e:
    st.error(f"Error loading vector database: {e}")
    st.stop()

# User input
query = st.text_input("Ask your question about Africa 🌍:")

if st.button("Search") and query:
    with st.spinner("Thinking..."):
        try:
            # Retrieve documents
            docs = vectordb.similarity_search(query, k=3)
            context = "\n\n".join([d.page_content for d in docs])

            # Build prompt
            prompt = ChatPromptTemplate.from_template(
                """
                You are an AI assistant specialized in African research and context.
                Use the provided context to answer concisely and clearly.

                Context:
                {context}

                Question:
                {question}
                """
            )

            chain_input = {"context": context, "question": query}
            formatted_prompt = prompt.format(**chain_input)
            response = llm.invoke(formatted_prompt)

            st.success("✅ Answer:")
            st.write(response.content)

        except Exception as e:
            st.error(f"An error occurred: {e}")

# Footer
st.markdown("---")
st.caption("Developed with ❤️ for the Africa RAG Project v1")
