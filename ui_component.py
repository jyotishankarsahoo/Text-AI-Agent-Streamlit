import streamlit as st

from core_agent import (
    run_question_answering,
    load_translation_pipeline,
    run_translation,
    run_text_summarization,
    run_sentiment_analysis
)


def qa_ui(qa_pipeline):
    st.title("Question Answering")
    qa_file_uploader = st.file_uploader(label="Upload a text file for context (optional)",
                                        type= ["txt"],
                                        key="qa_fileuploader")

    qa_context = st.text_area("Or enter the context here:",
                              height=200,
                              key="qa_text_area")

    qa_question = st.text_input("Enter question here:",
                                key="qa_question")

    if qa_file_uploader:
        try:
            qa_context = qa_file_uploader.read().decode("utf-8")
            st.success(f"File: {qa_file_uploader.name} with Size: {qa_file_uploader.size} uploaded successfully for question answering")
        except UnicodeDecodeError:
            st.error("Error decoding the uploaded file for Q&A, Please ensure it's a valid UTF-8 text file. ")

    if st.button("Get Answer", key="qa_button"):
        if qa_context and qa_question:
            with st.spinner("Processing..."):
                result_qa = run_question_answering(qa_pipeline, qa_context, qa_question)
                if result_qa:
                    st.subheader("Answer:")
                    st.write(f"{result_qa}")
        else:
            st.warning("Please provide context and a question", icon="⚠️")

def summarization_ui(summarization_pipeline):
    st.title("Text Summarization")
    uploaded_file_summary = st.file_uploader(label="Upload a text file to summarize (optional)",
                                        type=["txt"],
                                        key="summarization_fileuploader")
    text_to_summarize = st.text_area("Enter the text to summarize here:", height=300, key="summary_text_area")
    if uploaded_file_summary is not None:
        try:
            text_to_summarize = uploaded_file_summary.read().decode("utf-8")
            st.success("File uploaded for Summarization!")
        except UnicodeDecodeError:
            st.error("Error decoding the uploaded file for summarization. Please ensure it's a valid UTF-8 text file.")

    if st.button("Summarize", key="summarize_button"):
        with st.spinner("Summarizing Text..."):
            input_length = len(summarization_pipeline.tokenizer.encode(text_to_summarize))
            suggested_max_length = int(input_length * 0.7)  # Example: aim for ~60% of input length
            actual_max_length = min(130, suggested_max_length)  # Take the smaller value
            summary_result = run_text_summarization(summarization_pipeline,
                                                    text_to_summarize,
                                                    actual_max_length,
                                                    50)
            if summary_result:
                st.subheader("Summary")
                st.markdown(f"{summary_result}")

def translation_ui():
    st.header("Language Translation")
    source_language = st.selectbox("Source Language:", ["English", "French", "German", "Spanish"], key="translation_source")
    target_language = st.selectbox("Target Language:", ["French", "English", "German", "Spanish"], key="translation_target")
    text_to_translate = st.text_area("Enter text to translate:", height=150, key="translation_text_area")

    if st.button("Translate", key="translate_button"):
        if text_to_translate and source_language and target_language:
            model_name = get_translation_model(source_language, target_language)
            if model_name:
                translator = load_translation_pipeline(model_name)
                with st.spinner(f"Translating From {source_language} to {target_language}..."):
                    try:
                        translation_result = run_translation(translator, text_to_translate)
                        st.subheader("Translation:")
                        st.write(f"{translation_result}")
                    except Exception as e:
                        st.error(f"Translation Error: {e}")
            else:
                st.warning("Translation From {source_language} to {target_language} is not directly supported by pre-loaded model")
        else:
            st.warning("Please select source and target languages and enter text to translate.")

def sentiment_analysis_ui(sentiment_pipeline):
    st.header("Sentiment Analysis")
    text_to_analyze = st.text_area("Enter text to analyze text", height=200, key="sentiment_input_text")
    if st.button("Analyze Sentiment", key="sentiment_button"):
        if text_to_analyze:
            with st.spinner("Analyzing sentiment..."):
                sentiment_result = run_sentiment_analysis(sentiment_pipeline, text_to_analyze)
                if sentiment_result:
                    st.subheader("Sentiment Result:")
                    sentiment_label = sentiment_result['label']
                    score = sentiment_result['score']
                    if sentiment_label == "POSITIVE":
                        color = "green"
                    elif sentiment_label == "NEGATIVE":
                        color = "red"
                    else:
                        color = "gray"
                    st.markdown(
                        f"- **Sentiment**: <span style='color:{color}; font-weight: bold;'>{sentiment_label}</span> (Confidence: {score:.2f})",
                        unsafe_allow_html=True,
                    )
        else:
            st.warning("Please enter text to analyze.")

def get_translation_model(source: str, target: str):
    language_codes = {
        "English": "en",
        "French": "fr",
        "German": "de",
        "Spanish": "es"
    }
    source_code = language_codes.get(source)
    target_code = language_codes.get(target)
    if source_code and target_code and source_code != target_code:
        return f"Helsinki-NLP/opus-mt-{source_code}-{target_code}"
    else:
        return None