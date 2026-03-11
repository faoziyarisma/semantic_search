from fastapi import FastAPI

# from services.search_services import search_display
import services.search_services as search_services

app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "Welcome to the Semantic Search API!"}


@app.get("/search")
def search(query: str):
    return search_services.search_display(query)


@app.get("/search_summary")
def search_summary(query: str):
    return search_services.generate_summary(query)
