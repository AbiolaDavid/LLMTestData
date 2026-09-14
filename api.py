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

# ---------- Advanced Prompt: Strict but Comprehension-Aware ----------
# This allows the LLM to synthesize information, but forbids outside knowledge.
STRICT_RAG_INSTRUCTIONS = """You are a precise research assistant. 
Answer the user's question using ONLY the provided context.
Rules:
1. Read the context carefully. You are allowed to synthesize and summarize information that is clearly supported by the text.
2. DO NOT use your pre-trained knowledge or outside facts.
3. If the context truly does not contain the answer, respond EXACTLY with: "I do not have enough information in the provided documents to answer that."
4. Do not guess or make assumptions."""

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

# ---------- Source Extraction (Catch-All & Debuggable) ----------
def _extract_sources(resp: Any) -> List[Source]:
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

            # 1. Try standard names
            text_val = pick("text", "content", "body", "chunk", "passage", "document", default="")
            
            # 2. CATCH-ALL: If standard names fail, find the FIRST long string property in the object
            if not text_val and isinstance(props, dict):
                for key, val in props.items():
                    if isinstance(val, str) and len(val) > 50: # Assume long strings are the document text
                        text_val = val
                        break
            
            src = Source(
                text     = str(text_val or ""),
                source   = pick("source", "filename", "title", "doc_id"),
                page     = pick("page", "page_number", "page_num"),
                chunk_id = pick("chunk_id", "id", "uuid"),
            )
            if src.text or src.source:
                out.append(src)
        except Exception as e:
            log.warning("Failed to parse source item: %s", e)
            continue
    
    # DEBUG LOG: This will show up in your Render logs. 
    # If it says "Extracted 0 sources" or "Total text length: 0", your Weaviate schema names don't match.
    total_text_len = sum(len(s.text) for s in out)
    log.info(f"DEBUG: Extracted {len(out)} sources. Total text length passed to LLM: {total_text_len} chars.")
    
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
        
        # Try passing instructions. If the weaviate-agents version throws a TypeError, it will fall back.
        try:
            resp = agent.ask(data.question, instructions=STRICT_RAG_INSTRUCTIONS)
        except TypeError:
            log.warning("QueryAgent does not support 'instructions' kwarg in this version. Using default.")
            resp = agent.ask(data.question)

        answer = getattr(resp, "final_answer", None) or str(resp)
        sources = _extract_sources(resp)

        # GUARDRAIL: If extraction failed and returned 0 sources, force the fallback.
        if not sources:
            log.warning("CODE GUARDRAIL TRIGGERED: No sources extracted. Forcing fallback to prevent hallucination.")
            return Answer(
                answer="I do not have enough information in the provided documents to answer that.",
                sources=[]
            )

        return Answer(answer=answer, sources=sources)

    except Exception as e:
        log.exception("QueryAgent failed")
        raise HTTPException(status_code=500, detail=f"QueryAgent error: {e}")
