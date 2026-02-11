from langchain_community.vectorstores import Chroma
from config import CHROMA_PERSIST_DIR

def create_vectorstore(chunks, embeddings):
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIR
    )
    vectorstore.persist()
    return vectorstore
