import os
import logging
from fastapi import FastAPI, HTTPException, Query, Depends, Header, Request, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from weaviate_client import (
    get_agent,
    search_collection,
    close_client,
    COLLECTION_NAME,
)
from formatter import format_answer, format_search_results, format_source

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (fail fast)
# ---------------------------------------------------------------------------
API_KEY = os.environ.get("API_KEY")
if not API_KEY:
    raise RuntimeError("API_KEY environment variable is required")

ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv("ALLOWED_ORIGINS", "").split(",")
    if o.strip()
]
if not ALLOWED_ORIGINS:
    # Safe default for local development only
    ALLOWED_ORIGINS = ["http://localhost:3000", "http://localhost:8000"]

# ---------------------------------------------------------------------------
# App & middleware
# ---------------------------------------------------------------------------
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="SOC 101 RAG API",
    description="Query the TestingData Weaviate collection (Introduction to Sociology).",
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------
async def verify_api_key(x_api_key: str = Header(..., alias="X-API-Key")):
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
    return x_api_key

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Natural-language question",
    )
    pretty: bool = Field(True, description="Return a human-readable formatted answer")

class SourceItem(BaseModel):
    text: str | None = None
    source: str | None = None
    module_title: str | None = None
    heading: str | None = None
    page: int | None = None
    chunk_id: str | None = None
    score: float | None = None
    citation: str | None = None

class AskResponse(BaseModel):
    answer: str
    answer_pretty: str | None = None
    sources: list[SourceItem]

class SearchResponse(BaseModel):
    query: str
    collection: str
    results: list[SourceItem]
    results_pretty: str | None = None

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/healthz")
def healthz():
    return {"status": "ok", "collection": COLLECTION_NAME}

@app.post("/ask", response_model=AskResponse, dependencies=[Depends(verify_api_key)])
@limiter.limit("10/minute")
def ask_question(request: Request, req: AskRequest):
    """RAG endpoint: QueryAgent searches TestingData and generates an answer."""
    try:
        agent = get_agent()
        result = agent.ask(req.question)
    except Exception as e:
        logger.error(f"Agent error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while generating the answer.",
        )

    sources: list[SourceItem] = []
    for s in getattr(result, "sources", []) or []:
        props = getattr(s, "properties", None) or {}
        item = SourceItem(
            text=props.get("text"),
            source=props.get("source"),
            module_title=props.get("module_title"),
            heading=props.get("heading"),
            page=props.get("page"),
            chunk_id=props.get("chunk_id"),
        )
        item.citation = format_source(item.model_dump())
        sources.append(item)

    raw_answer = getattr(result, "final_answer", None) or str(result)

    if req.pretty:
        pretty = format_answer(raw_answer, sources, include_sources=True)
        return AskResponse(
            answer=pretty,
            answer_pretty=pretty,
            sources=sources,
        )
    return AskResponse(answer=raw_answer, sources=sources)

@app.get("/search", response_model=SearchResponse, dependencies=[Depends(verify_api_key)])
@limiter.limit("30/minute")
def search(
    request: Request,
    q: str = Query(..., min_length=1, max_length=500, description="Search query"),
    limit: int = Query(8, ge=1, le=30, description="Max number of chunks to return"),
    alpha: float = Query(
        0.7,
        ge=0.0,
        le=1.0,
        description="Hybrid search balance (0=BM25 only, 1=vector only)",
    ),
    pretty: bool = Query(True, description="Also return a human-readable block"),
):
    """Direct hybrid search against the TestingData collection."""
    try:
        hits = search_collection(q, limit=limit, alpha=alpha)
    except Exception as e:
        logger.error(f"Search error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while searching.",
        )

    results = []
    for h in hits:
        item = SourceItem(
            text=h.get("text"),
            source=h.get("source"),
            module_title=h.get("module_title"),
            heading=h.get("heading"),
            page=h.get("page"),
            chunk_id=h.get("chunk_id"),
            score=h.get("score"),
        )
        item.citation = format_source(item.model_dump())
        results.append(item)

    pretty_block = format_search_results(hits) if pretty else None
    return SearchResponse(
        query=q,
        collection=COLLECTION_NAME,
        results=results,
        results_pretty=pretty_block,
    )

@app.on_event("shutdown")
def _shutdown():
    close_client()
