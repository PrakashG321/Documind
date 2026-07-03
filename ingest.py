from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

path = Path("laravel-docs")

docs = []

excluded_files  = {"readme.md", "documentation.md", "license.md"}

for file in path.glob("**/*.md"):
    if file.name in excluded_files:
        continue
    text = file.read_text(encoding="utf-8")
    docs.append(Document(page_content=text, metadata={"source": file.name}))


chunk_method = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunks = chunk_method.split_documents(docs)

embeddings = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)

chroma = Chroma.from_documents(documents = chunks, embedding = embeddings, persist_directory="./chroma_db")