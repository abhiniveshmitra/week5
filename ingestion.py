from dotenv import load_dotenv
import os
from pathlib import Path
from PyPDF2 import PdfReader
from PIL import Image

load_dotenv()

UPLOAD_DIR = "data/uploads"

def ensure_upload_dir():
    os.makedirs(UPLOAD_DIR, exist_ok=True)

def save_uploaded_file(uploaded_file):
    ensure_upload_dir()
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def extract_text_from_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def save_pasted_image(image_data, name="pasted_image.png"):
    ensure_upload_dir()
    file_path = os.path.join(UPLOAD_DIR, name)
    image_data.save(file_path)
    return file_path

def list_uploaded_files():
    ensure_upload_dir()
    return list(Path(UPLOAD_DIR).glob("*"))
