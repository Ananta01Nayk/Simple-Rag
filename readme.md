# 📄 PDF-based RAG System using Streamlit & LangChain

This project is a **Retrieval-Augmented Generation (RAG)** system that allows users to upload PDF documents and ask questions about their contents. It combines the power of **LangChain**, **FAISS**, and **Streamlit** to deliver contextual answers using language models like OpenAI's GPT.

---

## 🚀 Key Features

- Upload any `.pdf` file through the UI
- Automatically splits and embeds PDF content
- Stores chunks in a FAISS vector database
- Retrieves relevant information using similarity search
- Generates answers using an LLM via LangChain
- Clean and interactive UI built with Streamlit

---

## 🧠 Tech Stack

- **LangChain** – For chaining retriever + LLM
- **OpenAI Embeddings** – To embed text chunks
- **FAISS** – For efficient vector similarity search
- **PyPDFLoader** – To load and process PDF files
- **Streamlit** – For building the web interface

---

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/pdf-rag-app.git

cd pdf-rag-app

###2. Create a Virtual Environment (Recommended)
python -m venv venv

source venv/bin/activate  # Windows: venv\Scripts\activate
###3. Install Dependencies
pip install -r requirements.txt
