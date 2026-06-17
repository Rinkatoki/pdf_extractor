from fastapi import FastAPI, UploadFile, File
from pypdf import PdfReader
from pydantic import BaseModel

import os
from pdf_reader import extract_text_from_pdf
from chunker import create_chunks
from embeddings import generate_embedding,generate_embeddings
from vector_store import store_chunks,search_chunks
from qa import generate_answer

app = FastAPI()

UPLOAD_DIR = "uploads"

os.makedirs(UPLOAD_DIR, exist_ok=True)


class QuestionRequest(BaseModel):
    question: str


@app.get("/")
async def root():
    return{"message":"hello fastapi"}


@app.post("/ask")
def ask_question(request: QuestionRequest):

    query_embedding = generate_embedding(
        request.question
    )

    results = search_chunks(
        query_embedding,
        n_results=3
    )
    documents = results.get("documents", [])

    if not documents or not documents[0]:
        return {
            "answer": "No relevant information found."
    }

    retrieved_chunks = documents[0]

    answer = generate_answer(
        request.question,
        retrieved_chunks
    )

    return {
        "question": request.question,
        "results": answer,
        "sources": results["metadatas"][0]
    }

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
    vectors,
    file.filename
    )
    return {
        "filename": file.filename,
        "pages": pages,
        "characters": len(text),
        "chunks": len(chunks),
        "embedding_dimension": len(vectors[0]),
        "first_chunk": chunks[0] if chunks else ""
    }





