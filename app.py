import streamlit as st
from ui_component import (
    qa_ui,
    summarization_ui,
    translation_ui,
    sentiment_analysis_ui)
from core_agent import (
    load_qa_pipeline,
    load_summarization_pipeline,
    load_sentiment_pipeline
)

st.set_page_config(
    page_title="AI Text Processing Suite",
    page_icon= ":magic_wand:",
    initial_sidebar_state="auto",
    layout="wide"
)

# --- Load Pipelines ---
qa_pipeline = load_qa_pipeline()
summarization_pipeline = load_summarization_pipeline()
sentiment_pipeline = load_sentiment_pipeline()

# --- Main Streamlit App ---
st.title("AI Text Processing Suite")
st.markdown("Explore various AI text processing capabilities.")

tab1, tab2, tab3, tab4 = st.tabs(["Question Answering", "Text Summarization", "Text Translation", "Sentiment Analysis"])

with tab1:
    qa_ui(qa_pipeline)
with tab2:
    summarization_ui(summarization_pipeline)
with tab3:
    translation_ui()
with tab4:
    sentiment_analysis_ui(sentiment_pipeline)

# --- Footer ---
st.markdown("----")
st.markdown(
    "<p style='text-align: center; color: #888;'>Built with Streamlit & Hugging Face Transformers</p>",
    unsafe_allow_html=True,
)