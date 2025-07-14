import os
import openai
from dotenv import load_dotenv
import numpy as np
import tiktoken
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

load_dotenv()

# Groq API Key and Model
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-70b-8192")

# Load tokenizer
enc = tiktoken.get_encoding("cl100k_base")

def count_tokens(text):
    return len(enc.encode(text))

def call_groq_model(prompt: str, max_tokens=512) -> str:
    import requests
    import json

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful research assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": max_tokens
    }

    response = requests.post(url, headers=headers, json=data)

    # Handle failed request or API error
    if response.status_code != 200:
        raise Exception(f"Groq API Error {response.status_code}: {response.text}")

    response_json = response.json()

    if "choices" not in response_json:
        raise Exception(f"Unexpected API response format:\n{json.dumps(response_json, indent=2)}")

    return response_json["choices"][0]["message"]["content"]

def find_best_chunk(question: str, chunks: list) -> str:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")
    question_embedding = model.encode([question])
    chunk_embeddings = model.encode(chunks)

    scores = cosine_similarity(question_embedding, chunk_embeddings)[0]
    best_idx = int(np.argmax(scores))
    return chunks[best_idx]
