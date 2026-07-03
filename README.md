
# DocuMind — Grounded Q&A for the Laravel Documentation

DocuMind answers plain-English questions about Laravel by retrieving the
relevant parts of Laravel's official documentation and having an LLM answer
**only from those retrieved docs** — with citations — rather than from the
model's own memory. It's a Retrieval-Augmented Generation (RAG) system, not a
generic chatbot: the grounding is the point.

> [write in your own words: 2–3 sentences on *why you built this* — what you
> wanted to learn, and why grounding/RAG specifically.]

## How it works (the RAG pipeline)

1. **Load** — read Laravel's Markdown docs into LangChain `Document` objects,
   tagging each with its source filename (for citations).
2. **Split** — chunk each doc into ~1000-character pieces (150 overlap) with a
   recursive splitter that breaks on natural boundaries (paragraphs/headers).
3. **Embed & store** — embed every chunk with a local sentence-transformer
   (`all-MiniLM-L6-v2`, 384-dim) and persist the vectors in a Chroma vector
   store. Done once, offline (`ingest.py`).
4. **Retrieve** — embed the user's question and fetch the top-k most similar
   chunks from Chroma (`query.py`).
5. **Generate (grounded)** — assemble a prompt with the question + retrieved
   chunks + an instruction to answer *only* from them, then have the LLM
   produce a cited answer.  *(in progress)*
6. **UI** — a minimal Streamlit app wrapping the above.  *(planned)*

## Tech stack (v1)

- Python
- LangChain (`langchain-core`, `langchain-text-splitters`, `langchain-huggingface`, `langchain-chroma`)
- Embeddings: HuggingFace `sentence-transformers/all-MiniLM-L6-v2` (local, free)
- Vector store: Chroma (persisted to disk)
- LLM: *(TBD)*
- UI: Streamlit

## Setup

1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

2. Install dependencies
pip install -r requirements.txt

3. Get the Laravel docs (Laravel 13.x)
git clone --branch 13.x --depth 1 https://github.com/laravel/docs.git laravel-docs

4. Build the vector store (run once)
python ingest.py

5. Query it
python query.py



## Scope

**v1 (this repo):** grounded Q&A over the Laravel docs, single local vector
store, Streamlit UI, and a ~30-question evaluation set measuring answer accuracy.

**Explicitly out of scope for v1:** FastAPI, user accounts, cross-session
memory, agents/tools, multiple documentation sources.

**v2 (planned):** a FastAPI backend so multiple clients can share one
retrieval/LLM service, plus (candidates) better retrieval for broad questions
(re-ranking, parent-document retrieval), and additional doc sources.

## Evaluation

> [write in your own words later: your ~30-question method, and the accuracy
> numbers before/after tuning chunk size or k. This is the part that proves the
> system actually works — save your real results here.]

## Known limitations

- Retrieval returns the *nearest* chunks even for out-of-scope questions; the
  grounding prompt is what prevents hallucinated answers.
- Basic top-k retrieval struggles with broad "explain everything about X"
  questions (a known RAG limitation; deferred to v2).