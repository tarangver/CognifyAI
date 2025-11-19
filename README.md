# 🧠 CognifyAI - GenAI Research Summarization Assistant

CognifyAI is a lightweight AI-powered assistant designed to help learners understand **small-sized research documents**, articles, and study material. It generates concise summaries, answers contextual questions, and tests comprehension using logic-based AI questions.

> ⚠️ **Important:** This version of CognifyAI is optimized for **small documents only** (short PDFs, articles, notes, 1–3 pages max). It does *not* reliably handle large research papers or long academic PDFs.

---

## 🔗 Live App

👉 **Launch CognifyAI:** [https://cognifyai-ez.streamlit.app/](https://cognifyai-ez.streamlit.app/)

---

## 🚀 Features (Current Capabilities)

### ✔️ 1. **Document Upload (PDF/TXT)**

Upload small research notes, articles, or pages for quick insight.

### ✔️ 2. **Auto‑Summarization**

* Creates a short, clear summary (≤150 words).
* Works best for small documents.
* Powered by **Groq Llama 3.1 (8B instant)**.

### ✔️ 3. **Ask Anything – Contextual Q&A**

Ask any question related to the uploaded document.
The AI scans the most relevant chunk and provides:

* A direct answer
* A short justification referencing the content

### ✔️ 4. **Challenge Me – AI-Generated Quiz**

CognifyAI creates:

* 3 logic-based questions
* Evaluates your answers
* Gives feedback + explanation

### ✔️ 5. **Semantic Chunk Matching**

Uses MiniLM-based embeddings to match your question with the most relevant section of the document.

### ✔️ 6. **Fast LLM Responses**

Integrated with **Groq’s ultra-fast inference API**.

---

## 🧰 Tech Stack

| Layer         | Technology                       |
|---------------|----------------------------------|
| UI            | Streamlit                        |
| Backend       | Python                           |
| LLM           | Groq API (llama-3.1-8b-instant)  |
| Embeddings    | sentence-transformers (MiniLM)   |
| PDF Parsing   | PyMuPDF                          |
| Environment   | python-dotenv, virtualenv        |

---

## 📁 Folder Structure

```
CognifyAI/
├── app.py                     # Streamlit main app
├── .env                       # API keys
├── requirements.txt           # Dependencies
├── README.md
│
├── backend/
│   ├── parser.py              # PDF/TXT parsing
│   ├── summarizer.py          # Summarization logic
│   ├── qa_engine.py           # Ask Anything logic
│   ├── challenge_gen.py       # Challenge Me logic
│   ├── evaluator.py           # Answer evaluation logic
│   ├── chunker.py             # Document chunking
│   └── utils.py               # API calls, similarity, helper functions
│
└── constants/
    └── prompts.py             # Prompt templates
```

---

## 🔧 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/tarangver/CognifyAI.git
cd CognifyAI
```

### 2. Install Required Packages

```bash
pip install -r requirements.txt
```

### 3. Add your `.env` file

Create a `.env` file if not already in the root folder with this content:

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.1-8b-instant
```

> ✅ You can replace `llama-3.1-8b-instant` with other Groq-supported models.

---

## ▶️ Running the App

```bash
streamlit run app.py
```

It will open in your browser at:
[http://localhost:8501](http://localhost:8501)

---

## 🧪 How to Use CognifyAI

1. Upload a small PDF/TXT file.
2. Read the auto-generated summary.
3. Pick a mode:

   * **Ask Anything** → Ask content-based questions
   * **Challenge Me** → Test your understanding
4. Get answers with justification.

---

## ⚠️ Document Limitations (Important)

CognifyAI (current version):

### ❌ Cannot handle:

* Long research papers (8–50 pages)
* High‑token PDFs
* Full books / long chapters
* Heavy academic literature

### ✔️ Works Best With:

* Small articles
* Single-page research summaries
* Handwritten notes (converted to text)
* Short academic paragraphs

Reason:
Groq’s free tier has **strict TPM (Tokens Per Minute)** and **input size limits**, which cause failures for long documents.

---

## ⚙️ Available Models (Groq)

You may use any of the following Groq models:

- "MetaLlama 3.1 8B": `llama-3.1-8b-instant`
- "MetaLlama 3.3 70B": `llama-3.3-70b-versatile`
- "OpenAI GPT OSS 120B": `openai/gpt-oss-120b`
- "OpenAI GPT OSS 20B": `openai/gpt-oss-20b`

Make sure your key supports the one you're using.

---

## 🧠 How It Works

- Text is extracted from PDF/TXT → chunked manually
- For QA, chunks are ranked using semantic similarity (MiniLM embeddings)
- Groq API is used for:
  - Auto-summary
  - Answering user questions
  - Generating logical questions
  - Evaluating user answers with justification

---

## 🔮 Future Enhancements (Planned)

These will be added in the next major upgrade:

### 🧠 1. **Large Document Support (Full RAG Pipeline)**

* Token-aware chunking
* Vector store retrieval
* Multi-pass summarization
* Long-context LLM support

### 🔄 2. **Multi-Provider LLM Support (OpenAI + Google + Groq)**

* Smart routing: Summaries → Gemini, Q&A → GPT‑4o, Speed → Groq

### 📄 3. **Downloadable Outputs**

* Export summary as PDF/TXT
* Export quiz results

### 📌 4. **Paragraph-Level Citations**

Highlight exact snippets used to answer.

### 💬 5. **Conversation Memory**

Allow follow-up questions on document context.

### 🎨 6. **Modern UI Upgrade**

New layout, animations, and dark mode.

---

## 👨‍💻 Author

Built with ❤️ and curiosity by **Tarang Verma**

* GitHub: [https://github.com/tarangver](https://github.com/tarangver)
* LinkedIn: [https://www.linkedin.com/in/verma-tarang/](https://www.linkedin.com/in/verma-tarang/)

---

## 🛡️ License

This project is open source and available under the [MIT License](LICENSE).
