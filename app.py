from fastapi import FastAPI, UploadFile, File
from pypdf import PdfReader
import os
from pdf_reader import extract_text_from_pdf
from chunker import create_chunks
from embeddings import generate_embedding,generate_embeddings
from vector_store import store_chunks

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
    chunks = create_chunks(
    text,
    chunk_size=500,
    overlap=100
    )
    vectors = generate_embeddings(chunks)
    store_chunks(
    chunks,
    vectors
    )
    return {
        "filename": file.filename,
        "pages": pages,
        "characters": len(text),
        "chunks": len(chunks),
        "embedding_dimension": len(vectors[0]),
        "first_chunk": chunks[0] if chunks else ""
    }





