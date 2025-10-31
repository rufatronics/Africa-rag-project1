# 🌍 Africa RAG Assistant — ReadyTensor Project 1

A fully cloud-deployable Retrieval-Augmented Generation (RAG) assistant powered by **Gemini API** and **Hugging Face African datasets**.

## 🚀 Features
- Automatically loads 3 African datasets (NER, stories, news)
- Uses ChromaDB for contextual retrieval
- Responds using Gemini 1.5 Flash via LangChain
- Streamlit UI (upload optional, ask questions directly)

## 🧩 File Structure

africa-rag-project1/ ├── app.py ├── src/ │   └── vectordb.py ├── requirements.txt ├── .gitignore └── README.md

## ⚙️ Setup (Cloud-Only)
1. Create a GitHub repo (public).
2. Add all files above.
3. Deploy to [Streamlit Cloud](https://share.streamlit.io)
4. In **Settings → Secrets**, add:

GOOGLE_API_KEY = "your_gemini_api_key_here"

5. Click **Deploy App**.

## 💡 Example Prompts
- “Explain ECOWAS’s peacekeeping mission in West Africa.”
- “Summarize renewable projects in Kenya.”
- “Tell me a folktale from AfricanStories dataset.”

## 🧱 Datasets
- [MasakhaneNER](https://huggingface.co/datasets/masakhane/masakhane-ner)
- [AfricanStories](https://huggingface.co/datasets/joelito/AfricanStories)
- [Africa-News-Dataset](https://huggingface.co/datasets/Kelechi/Africa-News-Dataset)
