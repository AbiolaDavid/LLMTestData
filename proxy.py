import os
import logging
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
UPSTREAM_URL = os.environ.get(
    "UPSTREAM_URL", "https://llmtestdata.onrender.com"
).rstrip("/")

UPSTREAM_API_KEY = os.environ.get("UPSTREAM_API_KEY")
if not UPSTREAM_API_KEY:
    raise RuntimeError("UPSTREAM_API_KEY environment variable is required")

ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:3000,http://localhost:5173,http://127.0.0.1:5500",
    ).split(",")
    if o.strip()
]


# ---------------------------------------------------------------------------
# HTTP client (shared connection pool)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=10.0, read=240.0, write=30.0, pool=10.0),
    )
    yield
    await app.state.http.aclose()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="ABIOLA AI Proxy", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class AskRequest(BaseModel):
    # Updated max_length to 5000 to match backend api_level_department.py
    question: str = Field(..., min_length=1, max_length=5000)
    pretty: bool = True
    department: str | None = Field(None, description="User department context")
    level: str | None = Field(None, description="User level context (e.g. 100, 200, 300)")


class SourceItem(BaseModel):
    text: str | None = None
    source: str | None = None
    course_code: str | None = None
    course_title: str | None = None
    module_title: str | None = None
    lecturer_name: str | None = None
    department: str | None = None
    level: str | None = None
    heading: str | None = None
    subheading: str | None = None
    heading_and_subheading: str | None = None
    page: int | str | None = None
    chunk_id: str | None = None
    score: float | None = None
    keywords: list[str] | None = None
    citation: str | list[str] | None = None


class ResponseMetadata(BaseModel):
    department: str | None = Field(None, description="User department context")
    level: str | None = Field(None, description="User level context (e.g. 100, 200, 300)")


class AskResponse(BaseModel):
    answer: str
    answer_pretty: str | None = None
    sources: list[SourceItem]
    keywords_for_further_reading: list[str] = Field(default_factory=list)
    metadata: ResponseMetadata | None = None


class SearchResponse(BaseModel):
    query: str
    collection: str
    results: list[SourceItem]
    results_pretty: str | None = None
    metadata: ResponseMetadata | None = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    return {"status": "ok", "service": "ABIOLA AI Proxy", "upstream": UPSTREAM_URL}


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.post("/api/ask", response_model=AskResponse)
async def proxy_ask(req: AskRequest, request: Request):
    """Forward the question to the upstream Render API with the secret key."""
    url = f"{UPSTREAM_URL}/ask"
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": UPSTREAM_API_KEY,
    }
    payload = {
        "question": req.question,
        "pretty": req.pretty,
        "department": req.department,
        "level": req.level,
    }

    try:
        r = await request.app.state.http.post(url, json=payload, headers=headers)
    except httpx.TimeoutException:
        logger.error("Upstream timeout")
        raise HTTPException(504, "Upstream timed out")
    except httpx.RequestError as e:
        logger.error(f"Upstream connection error: {e}")
        raise HTTPException(502, "Cannot reach upstream service")

    if r.status_code >= 400:
        try:
            detail = r.json().get("detail", r.text)
        except Exception:
            detail = r.text
        logger.warning(f"Upstream returned {r.status_code}: {detail}")
        raise HTTPException(r.status_code, detail)

    return r.json()


@app.get("/api/search", response_model=SearchResponse)
async def proxy_search(
    request: Request,
    q: str = Query(..., min_length=1, max_length=500),
    limit: int = Query(8, ge=1, le=30),
    alpha: float = Query(0.7, ge=0.0, le=1.0),
    department: str | None = Query(None),
    level: str | None = Query(None),
    pretty: bool = Query(True),
):
    """Proxy the /search endpoint, forwarding department and level filters."""
    url = f"{UPSTREAM_URL}/search"
    headers = {"X-API-Key": UPSTREAM_API_KEY}
    params: dict[str, Any] = {
        "q": q,
        "limit": limit,
        "alpha": alpha,
        "pretty": pretty,
    }
    if department:
        params["department"] = department
    if level:
        params["level"] = level

    try:
        r = await request.app.state.http.get(url, params=params, headers=headers)
    except httpx.TimeoutException:
        logger.error("Upstream timeout")
        raise HTTPException(504, "Upstream timed out")
    except httpx.RequestError as e:
        logger.error(f"Upstream connection error: {e}")
        raise HTTPException(502, "Cannot reach upstream service")

    if r.status_code >= 400:
        try:
            detail = r.json().get("detail", r.text)
        except Exception:
            detail = r.text
        raise HTTPException(r.status_code, detail)

    return r.json()
