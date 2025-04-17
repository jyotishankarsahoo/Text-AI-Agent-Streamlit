from transformers import pipeline
import streamlit as st

@st.cache_resource
def load_qa_pipeline():
    return pipeline(task="question-answering",
                    model="bert-large-uncased-whole-word-masking-finetuned-squad")
@st.cache_resource
def load_translation_pipeline(model_name: str):
    return pipeline(task="translation",
                    model=model_name)
@st.cache_resource
def load_summarization_pipeline():
    return pipeline(task="summarization",
                    model="facebook/bart-large-cnn")
@st.cache_resource
def load_sentiment_pipeline():
    return pipeline(task="sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english")

def run_question_answering(qa_pipeline: pipeline, context: str, question: str):
    if context and question:
        return qa_pipeline(question, context)
    else:
        return None

def run_translation(translation_pipeline: pipeline, text: str):
    if text:
        return translation_pipeline(text)[0]["translation_text"]
    else:
        return None

def run_text_summarization(summarization_pipeline, text: str, max_length: int, min_length: int):
    if text:
        return summarization_pipeline(text, max_length, min_length, do_sample=False)[0]["summary_text"]
    else:
        return None

def run_sentiment_analysis(sentiment_pipeline, text: str):
    if text:
        return sentiment_pipeline(text)[0]
    else:
        return None