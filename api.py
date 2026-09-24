import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query, Depends, Header, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import cohere  # For Cohere compatibility endpoint

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

COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
if not COHERE_API_KEY:
    logger.warning("COHERE_API_KEY not set. External LLM expansion will be disabled.")

# Model choice (V2 API). Override via env var if you want to swap models later.
COHERE_MODEL = os.environ.get("COHERE_MODEL", "command-r-plus-08-2024")

# Initialize Native Cohere ClientV2
cohere_client = cohere.ClientV2(api_key=COHERE_API_KEY) if COHERE_API_KEY else None

ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv("ALLOWED_ORIGINS", "").split(",")
    if o.strip()
]
if not ALLOWED_ORIGINS:
    ALLOWED_ORIGINS = ["http://localhost:3000", "http://localhost:8000"]


# ---------------------------------------------------------------------------
# Cohere health check (run once at startup so failures are visible early)
# ---------------------------------------------------------------------------
def _cohere_startup_check() -> None:
    """Ping Cohere once at boot so bad keys / bad model names surface immediately."""
    global cohere_client
    if not cohere_client:
        return
    try:
        cohere_client.chat(
            model=COHERE_MODEL,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=5,
        )
        logger.info("Cohere V2 client OK (model=%s)", COHERE_MODEL)
    except Exception as e:
        logger.error(
            "Cohere startup check failed (model=%s): %s. "
            "Disabling external expansion for this process.",
            COHERE_MODEL,
            e,
            exc_info=True,
        )
        cohere_client = None  # disable so `if cohere_client` short-circuits


# ---------------------------------------------------------------------------
# Lifespan (replaces deprecated @app.on_event)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: verify Cohere is reachable, Weaviate client is lazy-initialized.
    _cohere_startup_check()
    yield
    # Shutdown: close the Weaviate client cleanly.
    close_client()


# ---------------------------------------------------------------------------
# App & middleware
# ---------------------------------------------------------------------------
app = FastAPI(
    title="SOC 101 RAG API",
    description="Query the TestingData Weaviate collection (Introduction to Sociology).",
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
        ...,
        min_length=1,
        max_length=5000,
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
# Helpers
# ---------------------------------------------------------------------------
def _extract_cohere_text(response) -> str:
    """
    Robustly pull the text out of a Cohere V2 chat response.
    Falls back gracefully if the shape is unexpected.
    """
    try:
        blocks = response.message.content
        text = next(
            (
                getattr(b, "text", None)
                for b in blocks
                if getattr(b, "text", None)
            ),
            None,
        )
        if text:
            return text.strip()
    except Exception:
        logger.error("Unexpected Cohere response shape: %r", response, exc_info=True)
    return "External source returned no content."


def _call_cohere_external(question: str, internal_answer: str, is_internal_empty: bool) -> str:
    """
    Call Cohere V2 chat and return the external-supplement text.
    Uses `preamble=` for the system prompt (works across V2 SDK versions).
    """
    if not cohere_client:
        return "External expansion temporarily unavailable."

    if is_internal_empty:
        preamble = (
            "You are an expert academic assistant. The internal knowledge base could not "
            "find an answer to the user's question. Provide a comprehensive, well-structured "
            "answer based entirely on your general pre-trained academic knowledge."
        )
        user_prompt = f"Original Question: {question}"
    else:
        preamble = (
            "You are an expert academic supplement. A user asked a question, and an internal "
            "knowledge base provided a grounded answer. Your task is to provide a related, "
            "broader, or complementary perspective using your general pre-trained knowledge. "
            "Provide real-world examples, broader sociological context, or related theories. "
            "CRITICAL RULES: 1) DO NOT repeat the internal answer. 2) DO NOT cite the internal sources. "
            "3) Provide new, additive value only."
        )
        user_prompt = (
            f"Original Question: {question}\n\n"
            f"Internal Knowledge Base Answer: {internal_answer}"
        )

    try:
        response = cohere_client.chat(
            model=COHERE_MODEL,
            preamble=preamble,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return _extract_cohere_text(response)
    except Exception as e:
        logger.error("External LLM (Cohere) error: %s", e, exc_info=True)
        # Fail gracefully: internal answer is still returned.
        return "External expansion temporarily unavailable."


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    return {"status": "ok", "service": "Fulafia AI RAG API"}


@app.get("/healthz")
def healthz():
    return {
        "status": "ok",
        "collection": COLLECTION_NAME,
        "cohere_enabled": cohere_client is not None,
        "cohere_model": COHERE_MODEL if cohere_client else None,
    }


@app.post("/ask", response_model=AskResponse, dependencies=[Depends(verify_api_key)])
def ask_question(req: AskRequest):
    """
    RAG endpoint:
    1. QueryAgent searches TestingData (Internal).
    2. Cohere expands on the answer using general knowledge (External).
    3. Responses are aggregated and separated by a horizontal line.
    """
    # --- STEP 1: Execute Internal Weaviate Query ---
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

    internal_answer = getattr(result, "final_answer", None) or str(result)

    # Detect if the internal agent failed to find an answer
    is_internal_empty = "not in the provided materials" in internal_answer.lower()

    # --- STEP 2: Generate External Response via Cohere (V2) ---
    external_answer = _call_cohere_external(
        question=req.question,
        internal_answer=internal_answer,
        is_internal_empty=is_internal_empty,
    )

    # --- STEP 3: Application-Layer Aggregation ---
    final_dual_response = (
        f"**Internal Response:**\n{internal_answer}\n\n"
        f"---\n\n"
        f"**External Source:**\n{external_answer}"
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
        )

    return AskResponse(
        answer=final_dual_response,
        sources=sources,
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
    return SearchResponse(
        query=q,
        collection=COLLECTION_NAME,
        results=results,
        results_pretty=pretty_block,
    )
