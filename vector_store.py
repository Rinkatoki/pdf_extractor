import chromadb
import uuid

client = chromadb.PersistentClient(path="chroma_db")

collection = client.get_or_create_collection(
    name="pdf_chunks"
)


def store_chunks(chunks, embeddings,filename):

    ids = [
        str(uuid.uuid4())
        for _ in chunks
    ]

    collection.add(
    ids=ids,
    documents=chunks,
    embeddings=embeddings,
    metadatas=[
        {
            "filename": filename,
            "chunk_index": i
        }
        for i in range(len(chunks))
    ]
)


def search_chunks(query_embedding, n_results=3):

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    return results