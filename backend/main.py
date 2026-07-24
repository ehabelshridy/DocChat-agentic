"""
main.py

FastAPI backend for DocChat. Exposes the LangGraph pipeline (graph.py)
as a single POST /chat endpoint, AND serves the frontend (index.html,
style.css, script.js) directly -- one process, one port, no separate
static file server needed.

Run with:
    uvicorn main:app --reload --port 8000

Then open http://127.0.0.1:8000 in a browser.

Prompt Caching visibility
-------------------------
Set LOG_LEVEL=DEBUG in your .env (or shell) to see per-call cache
stats in the terminal:

    LOG_LEVEL=DEBUG uvicorn main:app --reload --port 8000

You will see lines like:
    DEBUG llm: Prompt cache HIT: 1024/1187 input tokens served from cache (saved ~50% on those tokens).
    DEBUG llm: Prompt cache MISS: 1187 input tokens (prefix will be cached for subsequent identical calls).
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from graph import run_query

load_dotenv()

# Configure logging — DEBUG shows prompt cache hits from llm.py
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(levelname)s %(name)s: %(message)s",
)

app = FastAPI(title="DocChat API", version="1.0.0")

# Resolve the frontend folder relative to this file, not relative to
# whatever directory `uvicorn` happens to be launched from. Assumes
# the project layout:
#   docchat/
#     backend/main.py   <- this file
#     frontend/         <- index.html, style.css, script.js
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

if not FRONTEND_DIR.exists():
    raise RuntimeError(
        f"Frontend directory not found at {FRONTEND_DIR}. "
        "Expected layout: docchat/backend/main.py and docchat/frontend/."
    )


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


class Source(BaseModel):
    source_file: str
    section: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    relevance_retries: int
    generation_retries: int


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        result = run_query(request.question)
    except Exception as e:
        import traceback
        traceback.print_exc()   # prints full traceback to the uvicorn terminal
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")

    return ChatResponse(
        answer=result["answer"],
        sources=[Source(**s) for s in result["sources"]],
        relevance_retries=result["retries"]["relevance"],
        generation_retries=result["retries"]["generation"],
    )


# Mounted LAST and at the root path: FastAPI matches routes in the
# order they're declared, so /health and /chat above still take
# priority over the static file lookup. html=True makes "/" resolve to
# index.html automatically.
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
