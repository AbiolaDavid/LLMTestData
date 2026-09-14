import os
import logging
from typing import List, Optional, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from weaviate_client import get_client, get_agent

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("api")

app = FastAPI()

# ---------- CORS ----------
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Advanced Prompt Engineering ----------
# This prompt uses "Negative Constraints" and "Forced Fallbacks" to stop hallucinations.
STRICT_RAG_INSTRUCTIONS = """You are a strict, highly accurate research assistant. 
Your task is to answer the user's question using ONLY the provided context.
Rules:
1. If the answer is not explicitly stated in the context, you MUST respond EXACTLY with: "I do not have enough information in the provided documents to answer that."
2. Do not use your pre-trained knowledge or outside facts.
3. Do not guess, infer, or make assumptions.
4. Base your answer strictly on the retrieved sources."""

# ---------- Models ----------
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

# ---------- Source Extraction (Robust) ----------
def _extract_sources(resp: Any) -> List[Source]:
    """
    Pull sources out of a QueryAgent response, tolerating multiple shapes.
    """
    raw = (
        getattr(resp, "sources", None)
        or getattr(resp, "objects", None)
        or getattr(resp, "collection_results", None)
        or []
    )
    if not raw and hasattr(resp, "sources"):
        inner = getattr(resp.sources, "objects", None)
        if inner:
            raw = inner

    out: List[Source] = []
    for item in raw:
        try:
            if isinstance(item, dict):
                props = item.get("properties", item)
            elif hasattr(item, "properties") and item.properties is not None:
                props = item.properties
            else:
                props = item

            def pick(*names, default=None):
                for n in names:
                    if isinstance(props, dict) and props.get(n) not in (None, ""):
                        return props[n]
                    v = getattr(props, n, None)
                    if v not in (None, ""):
                        return v
                return default

            src = Source(
                text     = str(pick("text", "content", "body", "chunk", default="") or ""),
                source   = pick("source", "filename", "title", "doc_id"),
                page     = pick("page", "page_number", "page_num"),
                chunk_id = pick("chunk_id", "id", "uuid"),
            )
            if src.text or src.source:
                out.append(src)
        except Exception as e:
            log.warning("Failed to parse source item: %s", e)
            continue
    return out

# ---------- Routes ----------
@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.get("/")
def root():
    try:
        ready = get_client().is_ready()
    except Exception as e:
        log.warning("is_ready check failed: %s", e)
        ready = False
    return {"status": "ok", "service": "testingdata-llm", "weaviate_ready": ready}

@app.post("/ask", response_model=Answer)
def ask_question(data: Question):
    try:
        agent = get_agent(data.collection)
        
        # 1. Inject Strict Prompt Instructions (if supported by the agent SDK version)
        try:
            resp = agent.ask(data.question, instructions=STRICT_RAG_INSTRUCTIONS)
        except TypeError:
            # Fallback if the specific weaviate-agents version doesn't accept 'instructions' in ask()
            resp = agent.ask(data.question)

        answer = getattr(resp, "final_answer", None) or str(resp)
        sources = _extract_sources(resp)

        # 2. CODE-LEVEL GUARDRAIL: The "Empty Context" Override
        # If the agent found 0 relevant sources, the LLM is likely hallucinating.
        # We override the answer to enforce the fallback.
        if not sources:
            log.warning("No sources retrieved. Overriding LLM answer to prevent hallucination.")
            return Answer(
                answer="I do not have enough information in the provided documents to answer that.",
                sources=[]
            )

        # 3. Check if the LLM successfully triggered its own fallback phrase
        fallback_phrases = ["do not have enough information", "i don't know", "not explicitly stated"]
        if any(phrase in answer.lower() for phrase in fallback_phrases):
            return Answer(answer=answer, sources=[]) # Return answer but clear sources if it's a fallback

        return Answer(answer=answer, sources=sources)

    except Exception as e:
        log.exception("QueryAgent failed")
        raise HTTPException(status_code=500, detail=f"QueryAgent error: {e}")
