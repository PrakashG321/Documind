from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)

chroma = Chroma(persist_directory="./chroma_db", embedding_function = embeddings)

user_prompt = "In laravel 13 , how do i define a single-action controller?"

results = chroma.similarity_search_with_score(user_prompt, k=15)

ragContext = ""
for result, score in results:
    ragContext += result.metadata["source"] + ": " + result.page_content + "\n"

system_prompt = "You are a documentation assistant for laravel 13. You are given a context and a question. Answer the question based on the context along with the source used for the context. If the answer is not in the context, say 'The answer is not in the context.'."
final_prompt = f"{system_prompt}\n\nContext:\n{ragContext}\n\nQuestion: {user_prompt}"
model="gemini-2.5-flash"
llm = ChatGoogleGenerativeAI(model=model)

result = llm.invoke(final_prompt)
print(result.content)