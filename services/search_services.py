from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from fastapi import HTTPException
from transformers import pipeline


embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = Chroma(persist_directory="database", embedding_function=embedding)

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")


def search(query: str, k: int = 5):
    """
    Retrieve top-k documents with relevance scores
    """
    return vectorstore.similarity_search_with_relevance_scores(query, k=k)


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

        # gabungkan semua content
        text = " ".join([doc.page_content for doc, _ in results])

        # limit panjang teks agar aman untuk model
        text = text[:3000]

        summary = summarizer(text, max_length=150, min_length=50, do_sample=False)

        return summary[0]["summary_text"]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def search_summary(query: str, k: int = 5):
    """
    API-friendly summary response
    """
    try:
        summary = generate_summary(query, k)

        return {"query": query, "summary": summary}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def generate_answer(query: str, k: int = 5):
    results = search(query, k)

    if not results:
        return "Informasi tidak ditemukan."

    # gabungkan context
    context = " ".join([doc.page_content for doc, _ in results])

    context = context[:4000]

    result = qa_pipeline(question=query, context=context)

    return result["answer"]


def search_answer(query: str, k: int = 5):
    try:
        answer = generate_answer(query, k)

        return {"query": query, "answer": answer}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
