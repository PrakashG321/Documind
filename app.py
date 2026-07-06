import os

import streamlit as st
from dotenv import load_dotenv
from query import build_pipeline, answer_question

# Local dev: read the key from a .env file into environment variables.
load_dotenv()

# Cloud (Streamlit Community Cloud): there is no .env file. The key lives in
# Streamlit's secrets. Copy it into the environment so ChatGoogleGenerativeAI,
# which reads GOOGLE_API_KEY from the environment, can find it. Wrapped in
# try/except because st.secrets raises when no secrets file exists (local dev).
try:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
except Exception:
    pass

st.set_page_config(
    page_title="DocuMind — Laravel Docs Assistant",
    page_icon="📚",
    layout="centered",
)

# --- lightweight styling ---
st.markdown(
    """
    <style>
    .block-container { max-width: 760px; padding-top: 3rem; }
    div.stButton > button, div.stFormSubmitButton > button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_cached():
    return build_pipeline()


store, llm = load_cached()

st.title("📚 DocuMind")
st.caption(
    "Ask a question about Laravel — answers are grounded in the official "
    "Laravel 13 documentation, with sources. Not the model's memory."
)

with st.expander("💡 Example questions"):
    st.markdown(
        "- How do I define a single-action controller?\n"
        "- What is middleware and how do I register it?\n"
        "- How do I validate an incoming form request?"
    )

with st.form("ask"):
    question = st.text_input(
        "Your question",
        placeholder="e.g. How do I define a single-action controller?",
    )
    submitted = st.form_submit_button("Ask")

if submitted and question:
    with st.spinner("Searching the Laravel docs…"):
        answer = answer_question(store, llm, question)
    st.markdown("### Answer")
    with st.container(border=True):
        st.markdown(answer)

st.divider()
st.caption("Built with LangChain · Chroma · HuggingFace embeddings · Google Gemini · Streamlit")
