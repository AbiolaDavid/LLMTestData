import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Depends, Header, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from weaviate_client import (
    search_collection,
    close_client,
    COLLECTION_NAME,
)
from formatter import format_answer, format_search_results, format_source
from langchain_pipeline import (
    build_dual_response_chain,
    get_llm,
    COHERE_MODEL,
    COHERE_BASE_URL,
)

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

# When set to "1", /ask and /search surface the actual exception in the HTTP
# 500 body. Useful while debugging; never enable in production.
DEBUG_ERRORS = os.getenv("DEBUG_ERRORS") == "1"

ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv("ALLOWED_ORIGINS", "").split(",")
    if o.strip()
]
if not ALLOWED_ORIGINS:
    ALLOWED_ORIGINS = ["http://localhost:3000", "http://localhost:8000"]


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
def _llm_startup_check() -> None:
    """Ping the LangChain LLM once at boot so misconfig is visible early."""
    llm = get_llm()
    if llm is None:
        logger.warning("LLM disabled — external expansion will return fallback text")
        return
    try:
        llm.invoke("ping")
        logger.info("LangChain LLM OK (model=%s, base=%s)", COHERE_MODEL, COHERE_BASE_URL)
    except Exception as e:
        logger.error("LLM startup check failed: %s", e, exc_info=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _llm_startup_check()
    yield
    close_client()


# ---------------------------------------------------------------------------
# App & middleware
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Federal University of Lafia (FULafia) AI RAG API",
    description=(
        "Welcome to Federal University of Lafia (FULafia) AI RAG API. LangChain-orchestrated dual-response "
        "pipeline combining internal Weaviate retrieval with external Cohere expansion."
    ),
    lifespan=lifespan,
)
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
        ..., min_length=1, max_length=5000, description="Natural-language question"
    )
    pretty: bool = Field(True, description="Return a human-readable formatted answer")
    department: str | None = Field(None, description="User department context")
    level: str | None = Field(None, description="User level context (e.g. 100, 200, 300)")


class SourceItem(BaseModel):
    text: str | None = None
    source: str | None = None
    module_title: str | None = None
    heading: str | None = None
    page: int | str | None = None  # tolerate "Page 12" / "12" / 12
    chunk_id: str | None = None
    score: float | None = None
    citation: str | None = None


class ResponseMetadata(BaseModel):
    department: str | None = Field(None, description="User department context")
    level: str | None = Field(None, description="User level context (e.g. 100, 200, 300)")


class AskResponse(BaseModel):
    answer: str
    answer_pretty: str | None = None
    sources: list[SourceItem]
    metadata: ResponseMetadata | None = None


class SearchResponse(BaseModel):
    query: str
    collection: str
    results: list[SourceItem]
    results_pretty: str | None = None
    metadata: ResponseMetadata | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _error_detail(exc: Exception, default: str) -> str:
    """Return a detailed error message only when DEBUG_ERRORS=1."""
    if DEBUG_ERRORS:
        return f"{type(exc).__name__}: {exc}"
    return default


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    return {"status": "ok", "service": "Federal University of Lafia (FULafia) AI RAG API System", "orchestrator": "langchain"}


@app.get("/healthz")
def healthz():
    llm = get_llm()
    return {
        "status": "ok",
        "collection": COLLECTION_NAME,
        "orchestrator": "langchain",
        "llm_enabled": llm is not None,
        "llm_model": COHERE_MODEL if llm else None,
        "llm_base_url": COHERE_BASE_URL if llm else None,
    }


@app.post("/ask", response_model=AskResponse, dependencies=[Depends(verify_api_key)])
def ask_question(req: AskRequest):
    """
    LangChain LCEL pipeline:
      1. Retrieve relevant chunks via hybrid search (with dept/level filters).
      2. Generate grounded internal answer with source-grounded prompt.
      3. Generate complementary external answer via Cohere.
      4. Aggregate into dual-response payload for the frontend.
    """
    try:
        chain = build_dual_response_chain(
            department=req.department,
            level=req.level,
        )
        result = chain.invoke({"question": req.question})
    except Exception as e:
        logger.error("Chain invocation error: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=_error_detail(
                e, "An internal error occurred while generating the answer."
            ),
        )

    internal_answer = result.get("internal_answer", "")
    external_answer = result.get("external_answer", "")
    docs = result.get("docs", []) or []

    # Build sources with citations
    sources: list[SourceItem] = []
    for d in docs:
        item = SourceItem(
            text=d.get("text"),
            source=d.get("source"),
            module_title=d.get("module_title"),
            heading=d.get("heading"),
            page=d.get("page"),
            chunk_id=d.get("chunk_id"),
            score=d.get("score"),
        )
        item.citation = format_source(item.model_dump())
        sources.append(item)

    # Aggregate dual response
    final_dual_response = (
        f"**Internal Response:**\n{internal_answer}\n\n"
        f"---\n\n"
        f"**External Source:**\n{external_answer}"
    )

    response_meta = ResponseMetadata(
        department=req.department,
        level=req.level,
    )

    if req.pretty:
        pretty_internal = format_answer(internal_answer, sources, include_sources=True)
        pretty_dual_response = (
            f"{pretty_internal}\n\n"
            f"---\n\n"
            f"**External Source:**\n{external_answer}"
        )
        return AskResponse(
            answer=final_dual_response,
            answer_pretty=pretty_dual_response,
            sources=sources,
            metadata=response_meta,
        )

    return AskResponse(
        answer=final_dual_response,
        sources=sources,
        metadata=response_meta,
    )


@app.get("/search", response_model=SearchResponse, dependencies=[Depends(verify_api_key)])
def search(
    q: str = Query(..., min_length=1, max_length=500, description="Search query"),
    limit: int = Query(8, ge=1, le=30, description="Max number of chunks to return"),
    alpha: float = Query(
        0.7,
        ge=0.0,
        le=1.0,
        description="Hybrid search balance (0=BM25 only, 1=vector only)",
    ),
    department: str | None = Query(None, description="Department metadata filter"),
    level: str | None = Query(None, description="Level metadata filter"),
    pretty: bool = Query(True, description="Also return a human-readable block"),
):
    """Direct hybrid search against the collection with optional filters."""
    try:
        hits = search_collection(
            q, limit=limit, alpha=alpha, department=department, level=level
        )
    except Exception as e:
        logger.error("Search error: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=_error_detail(e, "An internal error occurred while searching."),
        )

    results: list[SourceItem] = []
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
    response_meta = ResponseMetadata(
        department=department,
        level=level,
    )
    return SearchResponse(
        query=q,
        collection=COLLECTION_NAME,
        results=results,
        results_pretty=pretty_block,
        metadata=response_meta,
    )
