from fastapi import FastAPI, Request
from query import build_pipeline, answer_question
from contextlib import asynccontextmanager
from pydantic import BaseModel

class AskRequest(BaseModel):
    question: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.store, app.state.llm = build_pipeline()
    yield

app = FastAPI(lifespan=lifespan)

@app.post("/ask")
def ask(req: AskRequest, request: Request):
    answer = answer_question(request.app.state.store, request.app.state.llm, req.question)
    return {"answer": answer}