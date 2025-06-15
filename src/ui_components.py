# src/ui_components.py
import streamlit as st
from src import database, document_processor, azure_services

def render_sidebar():
    with st.sidebar:
        st.header("Azure AI Assistant")
        
        if st.button("➕ New Chat", use_container_width=True):
            st.session_state.clear()
            st.rerun()

        st.subheader("Previous Chats")
        chats = database.get_chats()
        for chat_id, title, _ in chats:
            if st.button(title, key=f"chat_{chat_id}", use_container_width=True):
                st.session_state.current_chat_id = chat_id
                st.session_state.messages = database.get_messages(chat_id)
                st.session_state.vector_store = None # Clear vector store when switching chats
                st.rerun()
        
        st.divider()
        st.subheader("Document Q&A (RAG)")
        uploaded_docs = st.file_uploader(
            "Upload PDF or TXT", type=['pdf', 'txt'], accept_multiple_files=True
        )
        if uploaded_docs and not st.session_state.get('vector_store'):
            raw_text = document_processor.get_text_from_files(uploaded_docs)
            if raw_text:
                text_chunks = document_processor.get_text_chunks(raw_text)
                st.session_state.vector_store = document_processor.create_vector_store(text_chunks)
                st.success("Documents ready for Q&A!")

        st.divider()
        st.subheader("Image Analysis")
        uploaded_image = st.file_uploader("Upload an Image", type=['jpg', 'png', 'jpeg'])
        if uploaded_image:
            st.image(uploaded_image)
            if st.button("Analyze Image", use_container_width=True):
                with st.spinner("Analyzing..."):
                    analysis = azure_services.analyze_image(uploaded_image)
                    if "error" in analysis:
                        st.error(f"Error: {analysis['error']}")
                    else:
                        st.success(f"**Description:** {analysis['description']}")
                        st.write("**Tags:** " + ", ".join(analysis['tags']))

        st.divider()
        st.subheader("Audio Tools")
        st.session_state.tts_enabled = st.toggle("Enable Text-to-Speech", value=False)
        uploaded_audio = st.file_uploader("Upload Audio File", type=['wav', 'mp3'])
        if uploaded_audio:
            transcribed_text = azure_services.transcribe_audio_file(uploaded_audio)
            st.session_state.prompt_from_audio = transcribed_text

def render_chat_messages():
    for message in st.session_state.get("messages", []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
