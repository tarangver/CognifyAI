import os
import json
import requests
import openai
from dotenv import load_dotenv
import numpy as np
import tiktoken
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

load_dotenv()

# Groq API Key and Model (default updated to recommended replacement)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# default to a supported model (change in .env if you want another default)
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# Optional OpenAI fallback (set ENABLE_OPENAI_FALLBACK=true and OPENAI_API_KEY if you want)
ENABLE_OPENAI_FALLBACK = os.getenv("ENABLE_OPENAI_FALLBACK", "false").lower() in ("1", "true", "yes")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # change if you prefer another OpenAI model

if ENABLE_OPENAI_FALLBACK and OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

# Load tokenizer
enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(enc.encode(text))
# def count_tokens(text):
#     return len(enc.encode(text))


def call_groq_model(prompt: str, max_tokens=512, model_override: str = None, temperature: float = 0.2) -> str:
    """
    Call Groq's OpenAI-compatible endpoint. If the configured model is decommissioned,
    this function raises a helpful Exception. Optionally falls back to OpenAI if enabled.

    - prompt: the user prompt (string)
    - max_tokens: max output tokens
    - model_override: if provided, use this model id instead of environment GROQ_MODEL
    """

    model_to_use = model_override or os.getenv("GROQ_MODEL", GROQ_MODEL)

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.getenv('GROQ_API_KEY', GROQ_API_KEY)}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model_to_use,
        "messages": [
            {"role": "system", "content": "You are a helpful research assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
    except requests.exceptions.RequestException as e:
        # network-level failure: attempt fallback if enabled
        if ENABLE_OPENAI_FALLBACK and OPENAI_API_KEY:
            return _call_openai_fallback(prompt, max_tokens=max_tokens, temperature=temperature)
        raise Exception(f"Network error while calling Groq: {str(e)}")

    # parse response
    if resp.status_code == 200:
        try:
            rj = resp.json()
        except Exception:
            raise Exception(f"Groq returned non-json response: {resp.text}")

        # support both typical OpenAI-compatible shape and Groq's outputs if different
        # try standard "choices" first
        if "choices" in rj and len(rj["choices"]) > 0:
            choice = rj["choices"][0]
            # Chat-style content
            if "message" in choice and "content" in choice["message"]:
                return choice["message"]["content"]
            # older style text field
            if "text" in choice:
                return choice["text"]
        # fallback — try Groq outputs path
        if "outputs" in rj and isinstance(rj["outputs"], list) and len(rj["outputs"]) > 0:
            first = rj["outputs"][0]
            # often content is list of dicts with 'text'
            content = first.get("content")
            if isinstance(content, list):
                pieces = []
                for c in content:
                    if isinstance(c, dict) and "text" in c:
                        pieces.append(c["text"])
                if pieces:
                    return " ".join(pieces)
            # fallback to stringified output
            return json.dumps(first)
        # last fallback — try top-level text
        if "text" in rj:
            return rj["text"]

        # If we couldn't find text, return the whole json (as string)
        return json.dumps(rj)

    else:
        # handle error responses
        try:
            err_json = resp.json()
        except Exception:
            err_json = {"error_text": resp.text}

        # check for decommission/code hints
        err_msg = ""
        err_code = None
        if isinstance(err_json, dict):
            err_msg = err_json.get("error", {}).get("message") or err_json.get("error") or json.dumps(err_json)
            err_code = err_json.get("error", {}).get("code")

        # If model_decommissioned detected, provide clear actionable message
        if err_code == "model_decommissioned" or ("decommission" in (err_msg or "").lower()):
            hint = (
                f"Groq model appears decommissioned or unsupported (model: {model_to_use}). "
                "Update GROQ_MODEL to a supported model id (e.g., 'llama-3.3-70b-versatile') "
                "or enable OpenAI fallback by setting ENABLE_OPENAI_FALLBACK=true and providing OPENAI_API_KEY."
            )
            # If fallback available, attempt it
            if ENABLE_OPENAI_FALLBACK and OPENAI_API_KEY:
                try:
                    return _call_openai_fallback(prompt, max_tokens=max_tokens, temperature=temperature)
                except Exception:
                    # If fallback fails, raise the clearer decommission error
                    raise Exception(f"Groq API Error {resp.status_code}: {err_msg} — {hint}")
            else:
                raise Exception(f"Groq API Error {resp.status_code}: {err_msg} — {hint}")

        # For other errors, optionally try fallback
        if ENABLE_OPENAI_FALLBACK and OPENAI_API_KEY:
            try:
                return _call_openai_fallback(prompt, max_tokens=max_tokens, temperature=temperature)
            except Exception:
                pass

        # otherwise raise
        raise Exception(f"Groq API Error {resp.status_code}: {err_msg}")


def _call_openai_fallback(prompt: str, max_tokens=512, temperature: float = 0.2) -> str:
    """
    Simple OpenAI ChatCompletion fallback if enabled.
    """
    if not OPENAI_API_KEY:
        raise Exception("OpenAI fallback requested but OPENAI_API_KEY is not set.")

    # Using OpenAI's ChatCompletion
    try:
        response = openai.ChatCompletion.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful research assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        # Expect the standard structure
        if "choices" in response and len(response["choices"]) > 0:
            return response["choices"][0]["message"]["content"]
        # fallback
        return str(response)
    except Exception as e:
        raise Exception(f"OpenAI fallback failed: {str(e)}")

# Embedding and retrieval helpers
_EMBEDDING_CACHE = {}

def embed_texts(texts, model_name: str = "all-MiniLM-L6-v2"):
    """
    Compute embeddings for a list of texts using sentence-transformers.
    Uses in-memory cache keyed by text content.
    Returns numpy array of shape (len(texts), dim)
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    to_compute = []
    idx_map = []
    results = []

    for i, t in enumerate(texts):
        if t in _EMBEDDING_CACHE:
            results.append(_EMBEDDING_CACHE[t])
        else:
            to_compute.append(t)
            idx_map.append(i)
            results.append(None)

    if to_compute:
        embs = model.encode(to_compute, show_progress_bar=False)
        # place computed embeddings into results and cache them
        j = 0
        for i in range(len(results)):
            if results[i] is None:
                emb = embs[j]
                _EMBEDDING_CACHE[texts[i]] = emb
                results[i] = emb
                j += 1

    # convert to numpy array
    return np.vstack(results)

def find_top_k_chunks(question: str, chunks: list, k: int = 3):
    """
    Given question (str) and chunks (list of dicts or str), return top-k chunk indices with similarity scores.
    `chunks` may be list of dicts with 'text' key (from chunker.split_text_into_chunks) or list of strings.
    Returns list of tuples: [(idx, score), ...] sorted by score desc.
    """
    # Normalize chunk texts
    if len(chunks) == 0:
        return []

    if isinstance(chunks[0], dict):
        texts = [c["text"] for c in chunks]
    else:
        texts = list(chunks)

    # Embed texts and question
    try:
        text_embs = embed_texts(texts)
    except Exception as e:
        # fallback: simple heuristic - use length-based score
        scores = [len(t) for t in texts]
        ranked = sorted(list(enumerate(scores)), key=lambda x: x[1], reverse=True)
        return [(i, float(s)) for i, s in ranked[:k]]

    # embed question
    try:
        from sentence_transformers import SentenceTransformer
        q_model = SentenceTransformer("all-MiniLM-L6-v2")
        q_emb = q_model.encode([question])[0]
    except Exception:
        # If embedding model fails, fallback to cosine with mean of text emb
        q_emb = np.mean(text_embs, axis=0)

    # compute cosine similarities
    sims = cosine_similarity([q_emb], text_embs)[0]
    ranked_idx = np.argsort(-sims)[:k]
    return [(int(i), float(sims[int(i)])) for i in ranked_idx]

# backward-compatible helper used elsewhere in your repo
def find_best_chunk(question: str, chunks: list) -> str:
    top = find_top_k_chunks(question, chunks, k=1)
    if not top:
        return "" if not chunks else (chunks[0]["text"] if isinstance(chunks[0], dict) else chunks[0])
    idx = top[0][0]
    if isinstance(chunks[0], dict):
        return chunks[idx]["text"]
    else:
        return chunks[idx]
