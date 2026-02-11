from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def build_rag_chain(retriever, llm):

    prompt = ChatPromptTemplate.from_template("""
You are an AI assistant.

Answer briefly in 2-3 lines only.
Do not repeat the context.
If answer not found say: Not available in documents.

Context:
{context}

Question:
{question}

Short Answer:
""")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain
