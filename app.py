# app.py

import streamlit as st
import os

from src.loader import load_pdfs
from src.chunker import chunk_documents
from src.embeddings import get_embeddings
from src.vectorstore import create_vectorstore
from src.llm import load_llm
from src.rag_chain import build_rag_chain
from config import PDF_DIRECTORY, TOP_K


st.set_page_config(page_title="Multi-PDF RAG Assistant", layout="wide")
st.title("📚 Multi-PDF RAG Assistant")


@st.cache_resource
def initialize_system():
    documents = load_pdfs()
    chunks = chunk_documents(documents)
    embeddings = get_embeddings()
    vectorstore = create_vectorstore(chunks, embeddings)
    llm = load_llm()
    return vectorstore, llm


# Initialize everything once
vectorstore, llm = initialize_system()

# Get list of resume PDFs
resume_files = [f for f in os.listdir(PDF_DIRECTORY) if f.endswith(".pdf")]

if not resume_files:
    st.warning("No PDFs found in data/pdfs folder.")
    st.stop()

# Resume selection dropdown
selected_resume = st.selectbox("Select Resume:", resume_files)

# Create metadata-filtered retriever
retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": TOP_K,
        "filter": {"source": selected_resume}
    }
)

# Build RAG chain dynamically for selected resume
rag_chain = build_rag_chain(retriever, llm)

# User input
user_input = st.text_input("Ask a question about the selected resume:")

if user_input:
    with st.spinner("Thinking..."):
        response = rag_chain.invoke(user_input)

    st.markdown("### Answer:")
    st.write(response)
