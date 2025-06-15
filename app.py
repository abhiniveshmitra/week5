import streamlit as st
from db import init_db, new_chat, get_chats, get_messages, add_message
from ingestion import (
    save_uploaded_file,
    save_uploaded_image,
    extract_text_from_txt,
    extract_text_from_pdf,
    extract_text_from_image,
    is_image
)
from embedding_retrieval import chunk_text, build_embedding_index, get_top_chunks
from PIL import Image
from openai import AzureOpenAI
from speech_services import azure_tts, azure_stt
import uuid
import base64
from dotenv import load_dotenv
import os

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
init_db()

# ---- Sidebar: Chats ----
st.sidebar.title("Chats")
chats = get_chats()
if st.sidebar.button("➕  New Chat"):
    chat_id = new_chat()
    st.session_state["chat_id"] = chat_id
    st.experimental_rerun()
if "chat_id" not in st.session_state:
    st.session_state["chat_id"] = chats[0][0] if chats else new_chat()

for cid, name in chats:
    label = name if name else f"Chat {cid}"
    if st.sidebar.button(label, key=f"chat_{cid}"):
        st.session_state["chat_id"] = cid
        st.experimental_rerun()

chat_id = st.session_state["chat_id"]
messages = get_messages(chat_id)
doc_chunks = []
for m in messages:
    if m["type"] == "text" and m["role"] == "user" and m["content"].startswith("[Uploaded "):
        continue
    if m["type"] == "text" and m["role"] == "user":
        for chunk in chunk_text(m["content"]):
            doc_chunks.append(chunk)
    if m["type"] == "image" and m["role"] == "user":
        img_path = f"data/uploads/{uuid.uuid4().hex}.png"
        with open(img_path, "wb") as f:
            f.write(base64.b64decode(m["content"]))
        ocr = extract_text_from_image(img_path)
        doc_chunks.append(ocr)

if doc_chunks:
    doc_embeddings = build_embedding_index(doc_chunks)
else:
    doc_embeddings = None

# ---- Chat bubbles area ----
st.markdown("""
<style>
.main { background: #f7f7fa;}
.chat-main { max-width: 660px; margin: 0 auto; min-height: 600px; padding-top: 20px; }
.bubble-user {
    background: #cce7ff; border-radius:20px; padding:10px 18px; margin:8px 0;
    max-width:72%; float: right; clear: both; text-align: right; font-size: 1.13em;
}
.bubble-assistant {
    background: #eee; border-radius:20px; padding:10px 18px; margin:8px 0;
    max-width:72%; float: left; clear: both; text-align: left; font-size: 1.13em;
}
.bubble-img {
    border-radius: 14px; border: 2px solid #cce7ff; margin:10px 0; float: right; max-width:220px;
    display: block;
}
.msg-clear { clear: both; }
.stTextInput > div > div > input {
    font-size: 1.13em;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="chat-main">', unsafe_allow_html=True)
for entry in messages:
    role = entry["role"]
    if entry["type"] == "text":
        if role == "user":
            st.markdown(f'<div class="bubble-user">{entry["content"]}</div><div class="msg-clear"></div>', unsafe_allow_html=True)
        else:
            msg_id = str(uuid.uuid4())
            col_a, col_b = st.columns([20,1])
            with col_a:
                st.markdown(f'<div class="bubble-assistant">{entry["content"]}</div><div class="msg-clear"></div>', unsafe_allow_html=True)
            with col_b:
                if st.button("🔊", key="listen_"+msg_id, help="Listen"):
                    audio_bytes = azure_tts(entry["content"])
                    if audio_bytes:
                        st.audio(audio_bytes, format="audio/mp3")
    elif entry["type"] == "image":
        st.markdown(f'<img src="data:image/png;base64,{entry["content"]}" class="bubble-img"><div class="msg-clear"></div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# --- Chat Input: text, upload, paste, voice input ---
c1, c2, c3 = st.columns([7,1,3])
with c1:
    # If voice was just recognized, auto-fill input box
    voice_default = st.session_state.pop('voice_input', '') if 'voice_input' in st.session_state else ''
    user_prompt = st.text_input("Type your message...", value=voice_default, key=str(uuid.uuid4()), label_visibility="collapsed")
with c2:
    uploaded_file = st.file_uploader("📎", type=["txt", "pdf", "jpg", "jpeg", "png"], label_visibility="collapsed", key=str(uuid.uuid4()))
with c3:
    pasted_img = st.camera_input("Paste or capture image", key=str(uuid.uuid4()))

# Voice input (record then convert to text)
audio_in = st.audio_recorder("🎤 Record voice (click, speak, click again to stop)", format="wav")
if audio_in is not None:
    audio_bytes = audio_in['audio']
    if audio_bytes:
        transcript = azure_stt(audio_bytes)
        if transcript:
            st.session_state['voice_input'] = transcript
            st.success(f"You said: {transcript}")
            st.experimental_rerun()

if uploaded_file is not None:
    filename = uploaded_file.name
    ext = filename.lower().split(".")[-1]
    if is_image(filename):
        file_path = save_uploaded_image(uploaded_file, filename)
        with open(file_path, "rb") as imgf:
            b64 = base64.b64encode(imgf.read()).decode()
        add_message(chat_id, "user", "image", b64)
        st.experimental_rerun()
    elif ext == "txt":
        file_path = save_uploaded_file(uploaded_file)
        txt = extract_text_from_txt(file_path)
        add_message(chat_id, "user", "text", f"[Uploaded TXT: {filename}]")
        for chunk in chunk_text(txt):
            add_message(chat_id, "user", "text", chunk)
        st.experimental_rerun()
    elif ext == "pdf":
        file_path = save_uploaded_file(uploaded_file)
        txt = extract_text_from_pdf(file_path)
        add_message(chat_id, "user", "text", f"[Uploaded PDF: {filename}]")
        for chunk in chunk_text(txt):
            add_message(chat_id, "user", "text", chunk)
        st.experimental_rerun()
    else:
        add_message(chat_id, "assistant", "text", f"Unsupported file type: {filename}")
        st.experimental_rerun()

if pasted_img is not None:
    img = Image.open(pasted_img)
    temp_img_path = f"data/uploads/pasted_{uuid.uuid4().hex}.png"
    img.save(temp_img_path)
    with open(temp_img_path, "rb") as imgf:
        b64 = base64.b64encode(imgf.read()).decode()
    add_message(chat_id, "user", "image", b64)
    st.experimental_rerun()

if st.button("Send", use_container_width=True) and user_prompt.strip():
    add_message(chat_id, "user", "text", user_prompt)
    retrieved_chunks = []
    if doc_embeddings is not None and doc_chunks:
        retrieved_chunks = get_top_chunks(user_prompt, doc_chunks, doc_embeddings, top_k=4)
    messages_for_llm = [{"role": "system", "content": "You are a helpful assistant. Use any context if relevant."}]
    if retrieved_chunks:
        context_text = "\n\n".join(retrieved_chunks)
        messages_for_llm.append({"role": "system", "content": f"Context:\n{context_text[:3000]}"})
    messages_for_llm.append({"role": "user", "content": user_prompt})
    with st.spinner("Thinking..."):
        response = client.chat.completions.create(
            model=GPT_DEPLOYMENT,
            messages=messages_for_llm,
            max_tokens=512,
        )
        answer = response.choices[0].message.content
    add_message(chat_id, "assistant", "text", answer)
    st.experimental_rerun()
