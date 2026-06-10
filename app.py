from fastapi import FastAPI, UploadFile, File
from pypdf import PdfReader
import os
from pdf_reader import extract_text_from_pdf

app = FastAPI()

UPLOAD_DIR = "uploads"

os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
async def root():
    return{"message":"hello fastapi"}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    text, pages = extract_text_from_pdf(file_path)

    return {
        "filename": file.filename,
        "pages": pages,
        "characters": len(text),
        "preview": text[:500]
    }