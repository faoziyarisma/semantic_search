from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from fastapi import HTTPException
from transformers import pipeline
import services.helper_service as helper_service
from google import genai
import os
from dotenv import load_dotenv

load_dotenv()

# Embedding model
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Vector database
vectorstore = Chroma(persist_directory="database", embedding_function=embedding)

# Summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# QA model
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

# Gemini client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
model = client.models


def search(query: str, k: int = 5):
    """
    Retrieve top-k documents with relevance scores
    and clean the text content
    """
    results = vectorstore.similarity_search_with_relevance_scores(query, k=k)

    cleaned_results = []

    for doc, score in results:
        if doc.page_content:
            doc.page_content = helper_service.clean_text(doc.page_content)

        cleaned_results.append((doc, score))

    return cleaned_results


def search_display(query: str, k: int = 5):
    """
    Return formatted search results
    """
    try:
        results = search(query, k)

        response = [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": float(score),
            }
            for doc, score in results
        ]

        return {
            "query": query,
            "total_results": len(response),
            "results": response,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def generate_summary(query: str, k: int = 5):
    """
    Generate summary from retrieved documents
    """
    try:
        results = search(query, k)

        if not results:
            return "No relevant information found."

        text = " ".join([doc.page_content for doc, _ in results])

        text = text[:3000]

        summary = summarizer(text, max_length=150, min_length=50, do_sample=False)

        return summary[0]["summary_text"]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def search_summary(query: str, k: int = 5):
    try:
        summary = generate_summary(query, k)

        return {"query": query, "summary": summary}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def generate_answer(query: str, k: int = 5):
    """
    Extractive QA using HuggingFace QA model
    """
    try:
        results = search(query, k)

        if not results:
            return "Informasi tidak ditemukan."

        contexts = [doc.page_content.strip() for doc, _ in results if doc.page_content]

        contexts = contexts[:3]

        best_answer = ""
        best_score = 0

        for context in contexts:
            context = context[:2000]

            result = qa_pipeline(question=query, context=context)

            if result["score"] > best_score:
                best_score = result["score"]
                best_answer = result["answer"]

        if not best_answer.strip():
            return contexts[0][:200] + "..."

        return best_answer

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def search_answer(query: str, k: int = 5):
    try:
        answer = generate_answer(query, k)

        return {"query": query, "answer": answer}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def gemini_answer(query: str, k: int = 5):
    """
    RAG answer using Gemini
    """
    try:
        results = search(query, k)

        if not results:
            return {"query": query, "answer": "Informasi tidak ditemukan."}

        contexts = [doc.page_content.strip() for doc, _ in results if doc.page_content]

        contexts = contexts[:3]

        answer = generate_rag_summary(query, contexts)

        return {"query": query, "answer": answer}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def generate_rag_summary(query: str, contexts: list):
    """
    Generate natural answer using Gemini
    """

    context_text = "\n".join(contexts)

    prompt = f"""
Anda adalah asisten yang menjelaskan informasi dengan bahasa sederhana.

Pertanyaan pengguna:
{query}

Informasi dari dokumen:
{context_text}

Tugas:
1. Jelaskan jawaban dari pertanyaan pengguna.
2. Gunakan bahasa Indonesia yang mudah dipahami.
3. Buat ringkasan 3-5 kalimat.
4. Fokus menjawab pertanyaan pengguna.

Jawaban:
"""

    response = model.generate_content(model="gemini-2.5-flash", contents=prompt)

    return response.text
