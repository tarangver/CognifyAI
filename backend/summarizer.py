from backend.utils import call_groq_model
from backend.chunker import chunk_text

def generate_summary(doc_text: str, max_words: int = 150) -> str:
    prompt = (
        "Summarize the following research document in less than "
        f"{max_words} words:\n\n{doc_text}"
    )
    return call_groq_model(prompt)

