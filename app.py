import streamlit as st
from backend import parser, summarizer, qa_engine, challenge_gen, evaluator
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# App config
st.set_page_config(page_title="CognifyAI", layout="wide")

st.title("🧠 CognifyAI - Smart Learning Companion")

# ---------------------------
# Model options (display -> id)
# ---------------------------
MODEL_OPTIONS = {
    "MetaLlama 3.1 8B": "llama-3.1-8b-instant",
    "MetaLlama 3.3 70B": "llama-3.3-70b-versatile",
    "OpenAI GPT OSS 120B": "openai/gpt-oss-120b",
    "OpenAI GPT OSS 20B": "openai/gpt-oss-20b",
}

# Determine current default model id (from env or fallback)
default_model_id = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# Find a friendly label for the default if available, otherwise use first option
default_label = None
for label, mid in MODEL_OPTIONS.items():
    if mid == default_model_id:
        default_label = label
        break
if default_label is None:
    # If env model not in options, create a label and add to MODEL_OPTIONS for runtime selection
    default_label = f"Custom ({default_model_id})"
    MODEL_OPTIONS[default_label] = default_model_id

# ---------------------------
# Sidebar: Model selector (moved above upload)
# ---------------------------
st.sidebar.header("🧭 Choose Model")
selected_label = st.sidebar.selectbox(
    "Select model:",
    options=list(MODEL_OPTIONS.keys()),
    index=list(MODEL_OPTIONS.keys()).index(default_label)
)
selected_model_id = MODEL_OPTIONS[selected_label]

# Store and expose runtime override so backend picks it up
if "GROQ_MODEL" not in st.session_state:
    st.session_state["GROQ_MODEL"] = default_model_id

if selected_model_id != st.session_state["GROQ_MODEL"]:
    st.session_state["GROQ_MODEL"] = selected_model_id
    os.environ["GROQ_MODEL"] = selected_model_id  # backend utils reads this at call time

# Simple warning if user picks a clearly deprecated id (heuristic)
if "8192" in st.session_state["GROQ_MODEL"] or "depre" in st.session_state["GROQ_MODEL"].lower():
    st.sidebar.warning("Selected model may be deprecated or unsupported. Try a different option if calls fail.")

# ---------------------------
# Sidebar: Upload Document (moved below model selector)
# ---------------------------
st.sidebar.header("📤 Upload Document")
uploaded_file = st.sidebar.file_uploader("Choose a PDF or TXT file", type=["pdf", "txt"])

st.sidebar.header("🎛️ Interaction Mode")
mode = st.sidebar.radio("Choose a mode:", ["Ask Anything", "Challenge Me"])

# Session state to store parsed content
if "doc_text" not in st.session_state:
    st.session_state.doc_text = ""
if "summary" not in st.session_state:
    st.session_state.summary = ""

# Upload handling
if uploaded_file:
    file_ext = uploaded_file.name.split(".")[-1].lower()
    if file_ext == "pdf":
        text = parser.parse_pdf(uploaded_file)
    elif file_ext == "txt":
        text = parser.parse_txt(uploaded_file)
    else:
        st.error("Unsupported file type. Please upload a PDF or TXT file.")
        st.stop()

    st.session_state.doc_text = text

    if len(text.split()) > 7500:
        st.warning("⚠️ Large document detected. Only the first part will be summarized due to token limits.")

    # Generate summary inside try/except so app doesn't crash on API errors
    try:
        # ensure runtime model override is visible to backend by setting env var
        os.environ["GROQ_MODEL"] = st.session_state["GROQ_MODEL"]
        st.session_state.summary = summarizer.generate_summary(text)
    except Exception as e:
        st.session_state.summary = ""
        st.error("⚠️ Failed to generate summary. See details below.")
        st.code(str(e))

    st.success("✅ Document parsed successfully!")

    # Display summary
    st.subheader("🔎 Auto Summary")
    if st.session_state.summary:
        st.write(st.session_state.summary)
    else:
        st.info("Summary not available due to API error. Try changing the model in the sidebar or enable OpenAI fallback in your .env.")

    # Interaction mode UI
    if mode == "Ask Anything":
        st.subheader("💬 Ask Anything")
        question = st.text_input("Enter your question based on the document:")
        if question:
            try:
                # ensure backend sees the runtime model override
                os.environ["GROQ_MODEL"] = st.session_state["GROQ_MODEL"]
                response, justification = qa_engine.answer_question(st.session_state.doc_text, question)
                st.markdown(f"**Answer:** {response}")
                st.markdown(f"🧾 *Justification:* {justification}")
            except Exception as e:
                st.error("⚠️ Failed to answer question. See details below.")
                st.code(str(e))

    elif mode == "Challenge Me":
        st.subheader("🎯 Challenge Yourself!")
        if st.button("Generate Questions"):
            try:
                os.environ["GROQ_MODEL"] = st.session_state["GROQ_MODEL"]
                questions = challenge_gen.generate_questions(st.session_state.doc_text)
                st.session_state.challenge_qs = questions
                st.session_state.user_answers = [""] * len(questions)
            except Exception as e:
                st.error("⚠️ Failed to generate questions. See details below.")
                st.code(str(e))

        if "challenge_qs" in st.session_state:
            for i, q in enumerate(st.session_state.challenge_qs):
                st.markdown(f"**Q{i+1}: {q}**")
                st.session_state.user_answers[i] = st.text_input(f"Your answer to Q{i+1}:", key=f"ua_{i}")

            if st.button("Evaluate Answers"):
                try:
                    os.environ["GROQ_MODEL"] = st.session_state["GROQ_MODEL"]
                    results = evaluator.evaluate_answers(
                        st.session_state.doc_text,
                        st.session_state.challenge_qs,
                        st.session_state.user_answers
                    )
                    for i, (eval_text, justification) in enumerate(results):
                        st.markdown(f"✅ **Evaluation for Q{i+1}:** {eval_text}")
                        st.markdown(f"🧾 *Justification:* {justification}")
                except Exception as e:
                    st.error("⚠️ Failed to evaluate answers. See details below.")
                    st.code(str(e))
