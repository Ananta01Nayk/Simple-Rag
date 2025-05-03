import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import tempfile
import os

st.set_page_config(page_title="RAG PDF QA", layout="wide")
st.title("📄 PDF-based RAG System with LangChain + Streamlit")

# --- Sidebar: Upload and API Key ---
st.sidebar.header("Upload PDF and Configure")

pdf_file = st.sidebar.file_uploader("NIPS-2017-attention-is-all-you-need-Paper.pdf", type=["pdf"])

openai_api_key = st.sidebar.text_input("Enter your OpenAI API key", type="password")

if pdf_file and openai_api_key:
    # Save PDF to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        tmp_pdf.write(pdf_file.read())
        tmp_path = tmp_pdf.name

    # Load and split PDF
    loader = PyPDFLoader(tmp_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    # Create embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Create Retriever
    retriever = vectorstore.as_retriever()

    # LLM
    llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    # --- Main Chat Section ---
    st.subheader("Ask questions about your PDF")

    user_query = st.text_input("Enter your question here:")

    if user_query:
        with st.spinner("Thinking..."):
            result = qa_chain.run(user_query)
        st.success("Answer:")
        st.write(result)

    # Clean up temp file
    os.remove(tmp_path)
else:
    st.warning("Please upload a PDF and provide your OpenAI API key to start.")
