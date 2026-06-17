import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

genai.configure(
    api_key=os.getenv("api_key")
)

model = genai.GenerativeModel(
    "gemini-3.1-flash-lite-preview"
)


def generate_answer(question, retrieved_chunks):

    context = "\n\n".join(retrieved_chunks)

    prompt = f"""
You are a helpful PDF Question Answering Assistant.

Answer ONLY using the provided context.

If the answer is not present in the context, reply:
"I could not find the answer in the uploaded PDF."

Context:
{context}

Question:
{question}

Answer:
"""

    response = model.generate_content(prompt)

    return response.text