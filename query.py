from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

embeddings = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)

chroma = Chroma(persist_directory="./chroma_db", embedding_function = embeddings)

results = chroma.similarity_search("what do i create a controller?", k=3)

for result in results:
    print(result.metadata["source"])
    print(result.page_content[:200])
    print("--------------------------------------------------")