from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

loader = PyPDFLoader("pdfs/document.pdf")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

chunks = text_splitter.split_documents(documents)

print(len(chunks))

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = Chroma.from_documents(
    documents=chunks, embedding=embedding, persist_directory="database"
)

vectorstore.persist()

results = vectorstore.similarity_search_by_vector(
    embedding.embed_query("apa isi utama dokumen ini?"), k=3
)

for r in results:
    print(r.page_content)
