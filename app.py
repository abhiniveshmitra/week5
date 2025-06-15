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
from PIL import Image
from openai import AzureOpenAI

# --- Load environment variables ---
load_dotenv()
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
GPT_DEPLOYMENT = os.getenv("GPT_MODEL", "gpt-4-1106-preview")

# --- Create the AzureOpenAI client ---
client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version="2024-02-01",
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
)

st.set_page_config(page_title="Modular Q&A Upload Demo", layout="wide")
st.title("🗂️ Modular Chat + File/Screenshot Upload")

# --- File Upload (txt/pdf/images) ---
uploaded_file = st.file_uploader(
    "Upload a file (.txt, .pdf, .jpg, .png)", type=["txt", "pdf", "jpg", "jpeg", "png"]
)
if uploaded_file is not None:
    file_path = save_uploaded_file(uploaded_file)
    st.success(f"File uploaded: {uploaded_file.name}")
    if uploaded_file.name.lower().endswith(".pdf"):
        st.markdown("**Extracted text from PDF:**")
        st.write(extract_text_from_pdf(file_path)[:1000])
    elif uploaded_file.name.lower().endswith(".txt"):
        st.markdown("**Text file content:**")
        st.write(extract_text_from_txt(file_path)[:1000])
    elif uploaded_file.name.lower().endswith((".jpg", ".jpeg", ".png")):
        st.markdown("**Image Preview:**")
        st.image(file_path, width=300)

# --- Screenshot/Paste Image (via camera/clipboard) ---
st.markdown("---\n**Paste an image below (Ctrl+V or drag/capture):**")
pasted_img = st.camera_input("Paste or capture image")
if pasted_img is not None:
    image = Image.open(pasted_img)
    img_path = save_pasted_image(image, name="pasted_image.png")
    st.success("Image pasted and saved!")
    st.image(img_path, width=300)

# --- List Uploaded Files ---
st.markdown("---\n### Uploaded Files")
files = list_uploaded_files()
if files:
    for f in files:
        st.write(f.name)
else:
    st.info("No files uploaded yet.")

# --- Basic Chat Section (contextual if files, otherwise general) ---
st.markdown("---")
st.header("💬 Chat")

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

user_prompt = st.text_input("Type your question (with or without uploaded files):", key="chat_input")

if st.button("Ask"):
    # Collect all uploaded files' text as context
    context = ""
    for f in files:
        fname = str(f).lower()
        if fname.endswith(".pdf"):
            context += extract_text_from_pdf(str(f))[:2000]
        elif fname.endswith(".txt"):
            context += extract_text_from_txt(str(f))[:2000]
        # (Add image OCR here in future)

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
    st.session_state["chat_history"].append(("User", user_prompt))
    st.session_state["chat_history"].append(("Assistant", answer))

# --- Show chat history ---
for role, msg in st.session_state["chat_history"]:
    st.markdown(f"**{role}:** {msg}")

