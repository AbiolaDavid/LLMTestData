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

# ---------- Source Extraction (Using EXACT Property Names) ----------
def _extract_sources(resp: Any) -> List[Source]:
    raw = getattr(resp, "sources", None) or []
    
    # DEBUG: Log the raw structure of the sources returned by Weaviate
    log.info(f"DEBUG RAW SOURCES: type={type(raw).__name__}, length={len(raw) if hasattr(raw, '__len__') else 'N/A'}")
    if raw and len(raw) > 0:
        log.info(f"DEBUG FIRST SOURCE RAW: {raw[0]}")

    out: List[Source] = []
    for item in raw:
        try:
            # Handle dict or object
            if isinstance(item, dict):
                props = item.get("properties", item)
            elif hasattr(item, "properties") and item.properties is not None:
                props = item.properties
            else:
                props = item

            # Extract using EXACT property names you confirmed
            def get_val(key):
                if isinstance(props, dict):
                    return props.get(key)
                return getattr(props, key, None)

            text_val = get_val("text")
            source_val = get_val("source")
            page_val = get_val("page")
            chunk_id_val = get_val("chunk_id")

            src = Source(
                text=str(text_val or ""),
                source=str(source_val) if source_val is not None else None,
                page=int(page_val) if page_val is not None else None,
                chunk_id=str(chunk_id_val) if chunk_id_val is not None else None,
            )
            
            if src.text:
                out.append(src)
        except Exception as e:
            log.warning(f"Failed to parse source item: {e}")
            continue
            
    log.info(f"DEBUG EXTRACTED SOURCES COUNT: {len(out)}")
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
        resp = agent.ask(data.question)
        
        # 1. Get the raw answer directly from the LLM/Agent BEFORE any code touches it
        raw_answer = getattr(resp, "final_answer", None) or str(resp)
        log.info(f"DEBUG RAW LLM ANSWER: {raw_answer}")
        
        # 2. Extract sources using exact property names
        sources = _extract_sources(resp)

        # 3. GUARDRAIL TEMPORARILY DISABLED
        # We are returning the raw answer directly to see if the LLM was actually refusing, 
        # or if our previous code was falsely overwriting a correct answer.
        
        return Answer(answer=raw_answer, sources=sources)

    except Exception as e:
        log.exception("QueryAgent failed")
        raise HTTPException(status_code=500, detail=f"QueryAgent error: {e}")
