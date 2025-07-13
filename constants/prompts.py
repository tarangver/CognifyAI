SUMMARY_PROMPT_TEMPLATE = """
Summarize the following research paper or document in no more than {max_words} words.

Text:
{document}
"""

QA_PROMPT_TEMPLATE = """
Using the context below, answer the user's question.

Context:
{context}

Question:
{question}

Answer concisely and provide a justification from the context.
"""

CHALLENGE_QUESTION_GENERATOR_TEMPLATE = """
Generate 3 logic-based or comprehension-focused questions based on the following document context.

Context:
{context}

Return each question on a new line.
"""

EVALUATION_PROMPT_TEMPLATE = """
Evaluate the user's answer based on the provided document context.

Context:
{context}

Question:
{question}

User's Answer:
{user_answer}

Respond if the answer is correct or partially correct. Then provide a clear justification.
"""
