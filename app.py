import streamlit as st
from backend import parser, summarizer, qa_engine, challenge_gen, evaluator
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# App config
st.set_page_config(page_title="CognifyAI - Research Summarizer", layout="wide")

st.title("🧠 CognifyAI - Smart Research Companion")

# Session state to store parsed content
if "doc_text" not in st.session_state:
    st.session_state.doc_text = ""
if "summary" not in st.session_state:
    st.session_state.summary = ""

# Upload section
st.sidebar.header("📤 Upload Document")
uploaded_file = st.sidebar.file_uploader("Choose a PDF or TXT file", type=["pdf", "txt"])

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
    st.session_state.summary = summarizer.generate_summary(text)

    st.success("✅ Document parsed successfully!")

    # Display summary
    st.subheader("🔎 Auto Summary")
    st.write(st.session_state.summary)

    # Select interaction mode
    st.sidebar.header("🎛️ Interaction Mode")
    mode = st.sidebar.radio("Choose a mode:", ["Ask Anything", "Challenge Me"])

    if mode == "Ask Anything":
        st.subheader("💬 Ask Anything")
        question = st.text_input("Enter your question based on the document:")
        if question:
            response, justification = qa_engine.answer_question(st.session_state.doc_text, question)
            st.markdown(f"**Answer:** {response}")
            st.markdown(f"🧾 *Justification:* {justification}")

    elif mode == "Challenge Me":
        st.subheader("🎯 Challenge Yourself!")
        if st.button("Generate Questions"):
            questions = challenge_gen.generate_questions(st.session_state.doc_text)
            st.session_state.challenge_qs = questions
            st.session_state.user_answers = [""] * len(questions)

        if "challenge_qs" in st.session_state:
            for i, q in enumerate(st.session_state.challenge_qs):
                st.markdown(f"**Q{i+1}: {q}**")
                st.session_state.user_answers[i] = st.text_input(f"Your answer to Q{i+1}:", key=f"ua_{i}")

            if st.button("Evaluate Answers"):
                results = evaluator.evaluate_answers(
                    st.session_state.doc_text,
                    st.session_state.challenge_qs,
                    st.session_state.user_answers
                )
                for i, (eval_text, justification) in enumerate(results):
                    st.markdown(f"✅ **Evaluation for Q{i+1}:** {eval_text}")
                    st.markdown(f"🧾 *Justification:* {justification}")
