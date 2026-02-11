from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
from config import MODEL_NAME

def load_llm():
    pipe = pipeline(
        "text-generation",
        model=MODEL_NAME,
        max_new_tokens=200,
        temperature=0.2,
        device_map="auto"
    )

    return HuggingFacePipeline(pipeline=pipe)

