from backend.utils import call_groq_model, find_best_chunk

def evaluate_answers(doc_text: str, questions: list, user_answers: list):
    results = []
    for question, user_answer in zip(questions, user_answers):
        relevant_chunk = find_best_chunk(question, doc_text.split("\n\n"))

        prompt = (
            f"Evaluate the user's answer to the question based on the document context.\n\n"
            f"Context:\n{relevant_chunk}\n\n"
            f"Question: {question}\n"
            f"User's Answer: {user_answer}\n\n"
            f"Is the answer correct or partially correct? Justify clearly."
        )

        eval_response = call_groq_model(prompt)
        if "Justification:" in eval_response:
            evaluation, justification = eval_response.split("Justification:", 1)
        else:
            evaluation, justification = eval_response, "Based on best-matching content."

        results.append((evaluation.strip(), justification.strip()))
    return results
