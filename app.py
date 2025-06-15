import streamlit as st
from dotenv import load_dotenv
import os
from ingestion import (
    save_uploaded_file,
    save_pasted_image,
    extract_text_from_pdf,
    extract_text_from_txt,
    list_uploaded_files,
)
from vision_ocr import azure_ocr_image
from PIL import Image
from openai import AzureOpenAI

# --- Load env variables ---
load_dotenv()
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
GPT_DEPLOYMENT = os.getenv("GPT_MODEL", "gpt-4-1106-preview")

# --- AzureOpenAI Client ---
client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version="2024-02-01",
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
)

st.set_page_config(page_title="Modular Chat + OCR Upload", layout="wide")
st.markdown("""
    <style>
    .chat-bubble-user {
        background-color: #d2f1ff; border-radius:18px; padding:10px 16px; margin:8px 0; text-align:right; float:right; max-width:70%; clear:both;
    }
    .chat-bubble-assistant {
        background-color: #f0f0f0; border-radius:18px; padding:10px 16px; margin:8px 0; text-align:left; float:left; max-width:70%; clear:both;
    }
    .msg-clear { clear: both; }
    </style>
""", unsafe_allow_html=True)

st.title("🗂️ Modular Chat + OCR File/Screenshot Upload")

# --- File Upload (txt/pdf/images) ---
uploaded_file = st.file_uploader(
    "Upload a file (.txt, .pdf, .jpg, .png)", type=["txt", "pdf", "jpg", "jpeg", "png"]
)
uploaded_text, ocr_text = "", ""
if uploaded_file is not None:
    file_path = save_uploaded_file(uploaded_file)
    st.success(f"File uploaded: {uploaded_file.name}")
    if uploaded_file.name.lower().endswith(".pdf"):
        uploaded_text = extract_text_from_pdf(file_path)
        st.markdown("**Extracted text from PDF:**")
        st.write(uploaded_text[:1000])
    elif uploaded_file.name.lower().endswith(".txt"):
        uploaded_text = extract_text_from_txt(file_path)
        st.markdown("**Text file content:**")
        st.write(uploaded_text[:1000])
    elif uploaded_file.name.lower().endswith((".jpg", ".jpeg", ".png")):
        st.markdown("**Image Preview:**")
        st.image(file_path, width=300)
        with st.spinner("Running OCR..."):
            ocr_text = azure_ocr_image(file_path)
        st.markdown("**Extracted text from image (OCR):**")
        st.write(ocr_text[:1000])

# --- Screenshot/Paste Image (via camera/clipboard) ---
st.markdown("---\n**Paste an image below (Ctrl+V or drag/capture):**")
pasted_img = st.camera_input("Paste or capture image")
pasted_ocr_text = ""
if pasted_img is not None:
    image = Image.open(pasted_img)
    img_path = save_pasted_image(image, name="pasted_image.png")
    st.success("Image pasted and saved!")
    st.image(img_path, width=300)
    with st.spinner("Running OCR..."):
        pasted_ocr_text = azure_ocr_image(img_path)
    st.markdown("**Extracted text from pasted image (OCR):**")
    st.write(pasted_ocr_text[:1000])

# --- List Uploaded Files ---
st.markdown("---\n### Uploaded Files")
files = list_uploaded_files()
if files:
    for f in files:
        st.write(f.name)
else:
    st.info("No files uploaded yet.")

# --- Modular Chat Section (ChatGPT-style bubbles) ---
st.markdown("---")
st.header("💬 Chat")

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

user_prompt = st.text_input("Type your question (with or without uploaded files):", key="chat_input")

if st.button("Ask"):
    # Gather all context (text from txt/pdf, ocr from images)
    context = ""
    for f in files:
        fname = str(f).lower()
        if fname.endswith(".pdf"):
            context += extract_text_from_pdf(str(f))[:2000]
        elif fname.endswith(".txt"):
            context += extract_text_from_txt(str(f))[:2000]
        elif fname.endswith((".jpg", ".jpeg", ".png")):
            context += azure_ocr_image(str(f))[:2000]
    if ocr_text:
        context += ocr_text[:2000]
    if pasted_ocr_text:
        context += pasted_ocr_text[:2000]

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
    ]
    if context.strip():
        messages.append({"role": "system", "content": f"Context: {context[:3000]}"})
    messages.append({"role": "user", "content": user_prompt})

    with st.spinner("Thinking..."):
        response = client.chat.completions.create(
            model=GPT_DEPLOYMENT,
            messages=messages,
            max_tokens=512,
        )
        answer = response.choices[0].message.content
    st.session_state["chat_history"].append(("user", user_prompt))
    st.session_state["chat_history"].append(("assistant", answer))

# --- Show chat history with "bubbles" ---
for role, msg in st.session_state["chat_history"]:
    if role == "user":
        st.markdown(f'<div class="chat-bubble-user">{msg}</div><div class="msg-clear"></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-bubble-assistant">{msg}</div><div class="msg-clear"></div>', unsafe_allow_html=True)
