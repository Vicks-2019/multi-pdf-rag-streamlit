# src/loader.py

import os
import pdfplumber
from langchain_core.documents import Document
from config import PDF_DIRECTORY


def load_pdfs():
    documents = []

    if not os.path.exists(PDF_DIRECTORY):
        raise FileNotFoundError(
            f"PDF directory not found: {PDF_DIRECTORY}. Please create it and add PDFs."
        )

    for filename in os.listdir(PDF_DIRECTORY):
        if filename.endswith(".pdf"):
            path = os.path.join(PDF_DIRECTORY, filename)

            with pdfplumber.open(path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    text = page.extract_text()

                    if text:
                        documents.append(
                            Document(
                                page_content=text,
                                metadata={
                                    "source": filename,  # Full file name
                                    "resume_name": filename.replace(".pdf", ""),  # Clean name
                                    "page": page_num + 1
                                }
                            )
                        )

    return documents
