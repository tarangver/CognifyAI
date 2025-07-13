from backend.chunker import chunk_text
from backend.utils import call_groq_model

def generate_questions(doc_text: str):
    chunks = chunk_text(doc_text)
    context = " ".join(chunks[:2])  # use first two chunks for basic logic Qs

    prompt = (
        "Generate 3 logical, comprehension-based questions from the text below:\n\n"
        f"{context}\n\nReturn each question on a new line."
    )

    response = call_groq_model(prompt)
    questions = [q.strip("- ").strip() for q in response.strip().split("\n") if q.strip()]
    return questions[:3]
