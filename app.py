import streamlit as st
from dotenv import load_dotenv
import os
from ingestion import (
    save_uploaded_file,
    save_uploaded_image,
    extract_text_from_txt,
    extract_text_from_pdf,
    extract_text_from_image,
    is_image,
)
from embedding_retrieval import chunk_text, build_embedding_index, get_top_chunks
from PIL import Image
from openai import AzureOpenAI

load_dotenv()
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
GPT_DEPLOYMENT = os.getenv("GPT_MODEL", "gpt-4-1106-preview")

client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version="2024-02-01",
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
)

st.set_page_config(page_title="ChatGPT+ Semantic File Q&A", layout="wide")

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "doc_chunks" not in st.session_state:
    st.session_state["doc_chunks"] = []
if "doc_embeddings" not in st.session_state:
    st.session_state["doc_embeddings"] = None

st.markdown("""
    <style>
    .chat-bubble-user {
        background-color: #cce7ff; border-radius:18px; padding:10px 16px; margin:8px 0; text-align:right; float:right; max-width:70%; clear:both;
    }
    .chat-bubble-assistant {
        background-color: #f0f0f0; border-radius:18px; padding:10px 16px; margin:8px 0; text-align:left; float:left; max-width:70%; clear:both;
    }
    .chat-bubble-ocr {
        background-color: #fff9c4; border-radius:14px; padding:10px 16px; margin:8px 0; text-align:left; float:left; font-size:0.93em; max-width:70%; clear:both;
    }
    .msg-clear { clear: both; }
    </style>
""", unsafe_allow_html=True)

st.title("💬 ChatGPT+ Semantic File Q&A")

# --- Chat Box, Upload inside chat (like ChatGPT) ---
st.markdown("**Ask a question, or upload a file/image with the 📎 icon.**")

col1, col2 = st.columns([10,1])
with col1:
    user_prompt = st.text_input("Your message...", key="chat_input", label_visibility="collapsed")
with col2:
    uploaded_in_chat = st.file_uploader("📎", type=["txt","pdf","jpg","jpeg","png"], label_visibility="collapsed", key="upload_in_chat")

# --- Handle uploads ---
if uploaded_in_chat is not None:
    filename = uploaded_in_chat.name
    ext = filename.lower().split(".")[-1]
    if is_image(filename):
        file_path = save_uploaded_image(uploaded_in_chat, filename)
        ocr_text = extract_text_from_image(file_path)
        st.session_state.chat_history.append(
            ("user", f"[Uploaded Image: {filename}]")
        )
        st.session_state.chat_history.append(
            ("ocr", f"OCR result for **{filename}**:\n\n{ocr_text}")
        )
        # Add to semantic context as a chunk
        st.session_state.doc_chunks.append(ocr_text)
    elif ext == "txt":
        file_path = save_uploaded_file(uploaded_in_chat)
        txt = extract_text_from_txt(file_path)
        st.session_state.chat_history.append(("user", f"[Uploaded TXT: {filename}]"))
        st.session_state.chat_history.append(("ocr", f"TXT extracted from **{filename}**:\n\n{txt[:1200]}"))
        # Chunk and store for retrieval
        for chunk in chunk_text(txt):
            st.session_state.doc_chunks.append(chunk)
    elif ext == "pdf":
        file_path = save_uploaded_file(uploaded_in_chat)
        txt, ocr_msgs = extract_text_from_pdf(file_path)
        st.session_state.chat_history.append(("user", f"[Uploaded PDF: {filename}]"))
        st.session_state.chat_history.append(("ocr", f"Text extracted from **{filename}**:\n\n{txt[:1200]}"))
        # Show OCR for images in PDF (as chat message)
        for msg in ocr_msgs:
            st.session_state.chat_history.append(("ocr", msg))
        for chunk in chunk_text(txt):
            st.session_state.doc_chunks.append(chunk)
    else:
        st.session_state.chat_history.append(("assistant", f"Unsupported file type: {filename}"))

# --- Build/Update Embeddings Index ---
if st.session_state.doc_chunks:
    st.session_state.doc_embeddings = build_embedding_index(st.session_state.doc_chunks)

# --- Handle Chat ---
if st.button("Send", use_container_width=True) and user_prompt.strip():
    st.session_state.chat_history.append(("user", user_prompt))

    # Semantic retrieval if context exists
    retrieved_chunks = []
    if st.session_state.doc_embeddings is not None and st.session_state.doc_chunks:
        retrieved_chunks = get_top_chunks(user_prompt, st.session_state.doc_chunks, st.session_state.doc_embeddings, top_k=4)

    messages = [{"role": "system", "content": "You are a helpful assistant. If there is context, use it to answer the user's question."}]
    if retrieved_chunks:
        context_text = "\n\n".join(retrieved_chunks)
        messages.append({"role": "system", "content": f"Context:\n{context_text[:3000]}"})
    messages.append({"role": "user", "content": user_prompt})

    with st.spinner("Thinking..."):
        response = client.chat.completions.create(
            model=GPT_DEPLOYMENT,
            messages=messages,
            max_tokens=512,
        )
        answer = response.choices[0].message.content
    st.session_state.chat_history.append(("assistant", answer))

# --- Render chat bubbles (bottom up, like real chat) ---
for role, msg in st.session_state.chat_history:
    if role == "user":
        st.markdown(f'<div class="chat-bubble-user">{msg}</div><div class="msg-clear"></div>', unsafe_allow_html=True)
    elif role == "assistant":
        st.markdown(f'<div class="chat-bubble-assistant">{msg}</div><div class="msg-clear"></div>', unsafe_allow_html=True)
    elif role == "ocr":
        st.markdown(f'<div class="chat-bubble-ocr">{msg}</div><div class="msg-clear"></div>', unsafe_allow_html=True)
