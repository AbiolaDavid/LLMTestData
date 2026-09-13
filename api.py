import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from weaviate_client import get_client, get_agent   # <-- changed

app = FastAPI()

ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Question(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    collection: Optional[str] = None

class Source(BaseModel):
    text: str = ""
    source: Optional[str] = None
    page: Optional[int] = None
    chunk_id: Optional[str] = None

class Answer(BaseModel):
    answer: str
    sources: List[Source] = []

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.get("/")
def root():
    try:
        ready = get_client().is_ready()
    except Exception:
        ready = False
    return {"status": "ok", "service": "testingdata-llm", "weaviate_ready": ready}

@app.post("/ask", response_model=Answer)
def ask_question(data: Question):
    try:
        agent = get_agent(data.collection)
        resp = agent.ask(data.question)

        answer = getattr(resp, "final_answer", None) or str(resp)

        sources: List[Source] = []
        for s in getattr(resp, "sources", None) or []:
            if isinstance(s, dict):
                sources.append(Source(**{k: s.get(k) for k in ("text","source","page","chunk_id")}))
            else:
                sources.append(Source(
                    text=getattr(s, "text", "") or "",
                    source=getattr(s, "source", None),
                    page=getattr(s, "page", None),
                    chunk_id=getattr(s, "chunk_id", None),
                ))

        return Answer(answer=answer, sources=sources)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"QueryAgent error: {e}")
