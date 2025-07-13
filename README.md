# 🧠 CognifyAI - GenAI Research Summarization Assistant

CognifyAI is a GenAI-powered assistant that reads structured documents (PDF/TXT), generates intelligent summaries, allows users to ask contextual questions, and evaluates comprehension through logic-based Q&A.

## 🔗 Live App: **Launch [CognifyAI](https://cognifyai-ez.streamlit.app/)**

## 🚀 Features

- 📄 **Upload Document** (PDF/TXT)
- 🧠 **Auto-Summarization** (≤150 words)
- 💬 **Ask Anything**: Free-form Q&A based on document context
- 🧪 **Challenge Me**: Generates 3 logic-based questions and evaluates your answers
- ✅ **Justified Answers**: All responses cite back to original content
- 🔍 **Semantic Chunk Matching** using sentence transformers
- 🔐 **Groq API** integration with `llama-3.1-8b-instant` model

---

## 🧰 Tech Stack

| Layer         | Technology                      |
|---------------|----------------------------------|
| UI            | Streamlit                        |
| Backend       | Python                           |
| LLM           | Groq API (llama-3.1-8b-instant)           |
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

Create a file if not already created named `.env` in the root folder with this content:

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.1-8b-instant
```

> ✅ You can replace `llama-3.1-8b-instant` with other Groq-supported models like `llama3-70b-8192` or `gemma-7b-it`.

---

## ▶️ Running the App

```bash
streamlit run app.py
```

It will open in your browser at:
[http://localhost:8501](http://localhost:8501)

---

## 🧪 How to Use

1. Upload a PDF or TXT file via the sidebar.
2. View a concise summary (≤150 words).
3. Choose:
   - **Ask Anything** – Type questions based on the document
   - **Challenge Me** – Let the AI generate 3 questions and test your comprehension
4. Receive AI answers + source-based justifications

---

## ⚙️ Available Models (Groq)

You may use any of the following Groq models:

- `gemma-7b-it`
- `llama3-70b-8192`
- `llama-3.1-8b-instant`

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

## 📌 TODO... Improvements

- [ ] Answer snippet highlighting
- [ ] File download (summary, quiz result)
- [ ] Conversation memory (contextual follow-ups)
- [ ] Switchable models via dropdown
- [ ] Add citations by paragraph number

---

## 👨‍💻 Author



Made with ❤️ by **TARANG VERMA** as part of the **EZ GenAI Internship Task**.

---

## 🛡️ License

This project is open source and available under the [MIT License](LICENSE).
