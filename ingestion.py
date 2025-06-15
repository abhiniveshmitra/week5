from dotenv import load_dotenv
import os
from pathlib import Path
from PyPDF2 import PdfReader
from PIL import Image
import tempfile
from vision_ocr import azure_ocr_image

load_dotenv()

UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def save_uploaded_file(uploaded_file):
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def save_uploaded_image(uploaded_file, name):
    file_path = os.path.join(UPLOAD_DIR, name)
    image = Image.open(uploaded_file)
    image.save(file_path)
    return file_path

def extract_text_from_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    all_text = []
    ocr_msgs = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            all_text.append(text)
        # OCR on images in PDF page (if images exist)
        if '/XObject' in page['/Resources']:
            xObject = page['/Resources']['/XObject'].get_object()
            for obj in xObject:
                if xObject[obj]['/Subtype'] == '/Image':
                    data = xObject[obj]._data
                    # Save image to temp file for OCR
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_img:
                        tmp_img.write(data)
                        tmp_img.flush()
                        ocr_result = azure_ocr_image(tmp_img.name)
                        ocr_msgs.append(f"**[OCR] PDF page {i+1}, image:**\n{ocr_result}")
                        os.unlink(tmp_img.name)
    text = "\n".join(all_text)
    return text, ocr_msgs

def extract_text_from_image(file_path):
    return azure_ocr_image(file_path)

def is_image(filename):
    return filename.lower().endswith((".jpg", ".jpeg", ".png"))
