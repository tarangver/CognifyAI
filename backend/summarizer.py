from backend.utils import call_groq_model
from backend.chunker import chunk_text

def generate_summary(doc_text: str, max_words: int = 150) -> str:
    chunks = chunk_text(doc_text, max_tokens=500)[:3]  # Limit to ~1500 tokens total

    summaries = []
    for chunk in chunks:
        prompt = (
            f"Summarize the following document segment in less than {max_words} words:\n\n{chunk}"
        )
        try:
            summary = call_groq_model(prompt)
            summaries.append(summary)
        except Exception as e:
            summaries.append("⚠️ Failed to summarize this section.")

    # Combine partial summaries into one
    final_summary = "\n\n".join(summaries)
    return final_summary.strip()
