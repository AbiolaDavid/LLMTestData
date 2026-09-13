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

# ---------- Source extraction ----------
def _extract_sources(resp: Any) -> List[Source]:
    """
    Pull sources out of a QueryAgent response, tolerating multiple shapes:
      - resp.sources / resp.objects / resp.collection_results
      - each item being a dict, an object with .properties, or an object with flat attrs
    Returns only sources that carry real text or a real source label.
    """
    # 1. find the container
    raw = (
        getattr(resp, "sources", None)
        or getattr(resp, "objects", None)
        or getattr(resp, "collection_results", None)
        or []
    )

    # 2. also check nested shape: resp.sources.objects (some versions)
    if not raw and hasattr(resp, "sources"):
        inner = getattr(resp.sources, "objects", None)
        if inner:
            raw = inner

    out: List[Source] = []
    for item in raw:
        try:
            # Case A: plain dict
            if isinstance(item, dict):
                data = item
                props = item.get("properties", item)

            # Case B: object with .properties (Weaviate object)
            elif hasattr(item, "properties") and item.properties is not None:
                props = item.properties
                data = props if isinstance(props, dict) else {}

            # Case C: flat object
            else:
                props = item
                data = None

            # Pull each field, trying multiple common names
            def pick(*names, default=None):
                for n in names:
                    # dict-style
                    if isinstance(props, dict) and props.get(n) not in (None, ""):
                        return props[n]
                    if data and isinstance(data, dict) and data.get(n) not in (None, ""):
                        return data[n]
                    # attr-style
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

            # Only keep meaningful entries
            if src.text or src.source:
                out.append(src)

        except Exception as e:
            log.warning("Failed to parse source item: %s (type=%s)", e, type(item).__name__)
            continue

    return out


def _shape_of(resp: Any) -> str:
    """Small diagnostic string describing the response's top-level shape."""
    attrs = [a for a in dir(resp) if not a.startswith("_")]
    srcs = getattr(resp, "sources", None) or getattr(resp, "objects", None) or []
    first_attrs = []
    try:
        if srcs:
            first = srcs[0]
            first_attrs = [a for a in dir(first) if not a.startswith("_")] if not isinstance(first, dict) else list(first.keys())
    except Exception:
        pass
    return f"resp_attrs={attrs} src_count={len(srcs) if hasattr(srcs, '__len__') else '?'} first_src_attrs={first_attrs}"


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

        answer = getattr(resp, "final_answer", None) or str(resp)
        sources = _extract_sources(resp)

        # If we got an answer but no sources, log the shape once so prod debugging is easy
        if answer and not sources:
            log.warning("No usable sources. shape: %s", _shape_of(resp))

        return Answer(answer=answer, sources=sources)

    except Exception as e:
        log.exception("QueryAgent failed")
        raise HTTPException(status_code=500, detail=f"QueryAgent error: {e}")
