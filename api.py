import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query, Depends, Header, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import OpenAI  # OpenAI-compatible client for Cohere

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

COHERE_API_KEY = (os.environ.get("COHERE_API_KEY") or "").strip()
if not COHERE_API_KEY:
    logger.warning("COHERE_API_KEY not set. External LLM expansion will be disabled.")

# Cohere OpenAI-compatible endpoint & model (overridable via env vars)
COHERE_BASE_URL = os.environ.get(
    "COHERE_BASE_URL", "https://api.cohere.ai/compatibility/v1"
)
COHERE_MODEL = os.environ.get("COHERE_MODEL", "command-a-03-2025")

# Initialize OpenAI-compatible client pointing at Cohere
cohere_client = (
    OpenAI(base_url=COHERE_BASE_URL, api_key=COHERE_API_KEY)
    if COHERE_API_KEY
    else None
)

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
        cohere_client.chat.completions.create(
            model=COHERE_MODEL,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=5,
        )
        logger.info("Cohere (OpenAI-compat) client OK (model=%s)", COHERE_MODEL)
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
# Lifespan
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
    title="FULafia RAG API",
    description="Welcome to FULafia's RAG API. Combining internal knowledge base search with external LLM expansion.",
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
    department: str | None = Field(None, description="User department context")
    level: str | None = Field(None, description="User level context (e.g. 100, 200, 300)")


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
def _extract_openai_text(response) -> str:
    """Robustly pull text out of an OpenAI-style chat completion response."""
    try:
        choices = getattr(response, "choices", None) or []
        if choices:
            content = getattr(choices[0].message, "content", None)
            if content:
                return content.strip()
    except Exception:
        logger.error("Unexpected Cohere response shape: %r", response, exc_info=True)
    return "External source returned no content."


def _call_cohere_external(
    question: str,
    internal_answer: str,
    is_internal_empty: bool,
    department: str | None = None,
    level: str | None = None,
) -> str:
    """Call Cohere via OpenAI-compatible endpoint with optional student context."""
    if not cohere_client:
        return "External expansion temporarily unavailable."

    context_details = []
    if department:
        context_details.append(f"Department: {department}")
    if level:
        context_details.append(f"Level: {level}")
    
    user_context = f" [Student Info: {', '.join(context_details)}]" if context_details else ""

    if is_internal_empty:
        system_prompt = (
           f"You are an expert academic assistant and pedagogical coach.{user_context} "
    "The internal knowledge base could not find a relevant answer to the user's question. "
    "Your task is to provide a comprehensive, accurate, and well-structured response "
    "drawn solely from your pre-trained knowledge.\n\n"
    "Core guidelines:\n"
    "1. Tailor depth, vocabulary, examples, and complexity precisely to the student's academic level and field of study.\n"
    "2. Structure the answer clearly (e.g., definition → explanation → examples → implications/common pitfalls → summary).\n"
    "3. Use precise academic language while remaining accessible; include concrete examples, analogies, or step-by-step reasoning where helpful.\n"
    "4. Explicitly note any limitations, uncertainties, or areas where knowledge may be incomplete or contested.\n"
    "5. Promote deeper learning: briefly highlight key concepts, common misconceptions, and suggest related questions or further exploration when appropriate.\n"
    "6. Maintain a professional, supportive, and neutral academic tone. Avoid unnecessary greetings, flattery, or filler."
        )
        user_prompt = f"Original Question: {question}"
    else:
        system_prompt = (
            f"You are an expert academic supplement and pedagogical coach.{user_context} "
    "A user asked a question and an internal knowledge base has already provided a grounded, primary answer. "
    "Your sole task is to deliver a related, broader, complementary, or deeper perspective that adds genuine new value "
    "suited to the student's level and field of study.\n\n"
    "CRITICAL RULES (strictly enforce):\n"
    "1. Scan the internal answer for any greetings, pleasantries, or opening/closing formalities and never repeat them.\n"
    "2. DO NOT contradict, rephrase, summarize, or restate any part of the internal answer.\n"
    "3. DO NOT repeat or closely paraphrase the internal answer's content, explanations, examples, or conclusions.\n"
    "4. DO NOT cite, reference, or allude to internal sources, the knowledge base, or the existence of a prior answer.\n"
    "5. Provide only new, additive value: alternative angles, broader context, real-world applications, historical/theoretical connections, "
    "common pitfalls, advanced nuances, cross-disciplinary links, study strategies, or further questions for reflection.\n"
    "6. Match the student's academic level and field in tone, terminology, and depth.\n"
    "7. Keep the response focused, well-structured, and concise while remaining academically rigorous.\n"
    
        )
        user_prompt = (
            f"Original Question: {question}\n\n"
            f"Internal Knowledge Base Answer: {internal_answer}"
        )

    try:
        response = cohere_client.chat.completions.create(
            model=COHERE_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return _extract_openai_text(response)
    except Exception as e:
        logger.error("External LLM (Cohere) error: %s", e, exc_info=True)
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
        "cohere_base_url": COHERE_BASE_URL if cohere_client else None,
    }


@app.post("/ask", response_model=AskResponse, dependencies=[Depends(verify_api_key)])
def ask_question(req: AskRequest):
    """
    RAG endpoint:
    1. QueryAgent searches TestingData (Internal) filtered/guided by department and level context.
    2. Cohere expands on the answer using contextualized general knowledge (External).
    3. Responses are aggregated and returned.
    """
    # --- STEP 1: Execute Internal Weaviate Query ---
    try:
        agent = get_agent(department=req.department, level=req.level)
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
    is_internal_empty = "not in the provided materials" in internal_answer.lower()

    # --- STEP 2: Generate External Response via Cohere ---
    external_answer = _call_cohere_external(
        question=req.question,
        internal_answer=internal_answer,
        is_internal_empty=is_internal_empty,
        department=req.department,
        level=req.level,
    )

    # --- STEP 3: Aggregation ---
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
    department: str | None = Query(None, description="Department metadata filter"),
    level: str | None = Query(None, description="Level metadata filter"),
    pretty: bool = Query(True, description="Also return a human-readable block"),
):
    """Direct hybrid search against the TestingData collection with optional filters."""
    try:
        hits = search_collection(q, limit=limit, alpha=alpha, department=department, level=level)
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
