# app.py
import streamlit as st
from dotenv import load_dotenv
from src import database, azure_services, ui_components

# --- Page Config and Initialization ---
st.set_page_config(page_title="Azure AI Chat", layout="centered")
load_dotenv()
database.init_db()

# Initialize session state if not already done
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- UI Rendering ---
ui_components.render_sidebar()
st.title("Azure AI Chat Assistant")
ui_components.render_chat_messages()

# --- Chat Input Logic ---
# Use a key to manage the input widget's state
prompt_input = st.session_state.get("prompt_from_audio", "")
if prompt := st.chat_input(placeholder="Ask a question or use the mic...", args=[prompt_input]):
    # If starting a new chat, create it in the database
    if st.session_state.current_chat_id is None:
        new_title = prompt[:40] + "..."
        st.session_state.current_chat_id = database.add_chat(new_title)
        st.rerun()

    # Append and save user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    database.add_message(st.session_state.current_chat_id, "user", prompt)
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            stream = azure_services.get_chat_completion(
                st.session_state.messages,
                st.session_state.get("vector_store")
            )
            response_content = st.write_stream(stream)
    
    # Append and save assistant message
    st.session_state.messages.append({"role": "assistant", "content": response_content})
    database.add_message(st.session_state.current_chat_id, "assistant", response_content)

    # Text-to-Speech if enabled
    if st.session_state.get("tts_enabled", False):
        with st.spinner("Synthesizing audio..."):
            audio_data = azure_services.synthesize_text_to_speech(response_content)
            if audio_data:
                st.audio(audio_data, format="audio/wav")
    
    # Clear audio-based prompt and rerun
    if "prompt_from_audio" in st.session_state:
        del st.session_state.prompt_from_audio
    st.rerun()

# Microphone button logic outside chat_input
if st.button("🎤", key="mic_button"):
    transcribed_text = azure_services.transcribe_audio_from_mic()
    if "Error" not in transcribed_text and transcribed_text:
        st.session_state.prompt_from_audio = transcribed_text
        st.rerun()
    else:
        st.error("Failed to transcribe. Please try again.")

