from backend.chunker import chunk_text
from backend.utils import call_groq_model, find_best_chunk

def answer_question(doc_text: str, user_question: str):
    chunks = chunk_text(doc_text)
    relevant_chunk = find_best_chunk(user_question, chunks)

    prompt = (
        f"Using the context below, answer the question.\n\n"
        f"Context:\n{relevant_chunk}\n\n"
        f"Question: {user_question}\n\n"
        f"Also explain which part of the context supports your answer."
    )

    full_answer = call_groq_model(prompt)
    if "Justification:" in full_answer:
        answer, justification = full_answer.split("Justification:", 1)
    else:
        answer, justification = full_answer, "Answer derived from relevant chunk."

    return answer.strip(), justification.strip()
