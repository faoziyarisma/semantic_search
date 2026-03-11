from fastapi import FastAPI
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

app = FastAPI()

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = Chroma(persist_directory="database", embedding_function=embedding)


@app.get("/")
def read_root():
    return {"message": "Welcome to the Semantic Search API!"}


@app.get("/search")
def search(query: str):
    results = vectorstore.similarity_search(query, k=5)
    return {"query": query, "results": [r.page_content for r in results]}


@app.get("/search_summary")
def search_summary(query: str):
    results = vectorstore.similarity_search(query, k=5)
    return {"query": query, "summary": " ".join([r.page_content for r in results])}
