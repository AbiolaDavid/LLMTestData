import os
import logging
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
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
    question: str = Field(..., min_length=1, max_length=2000)
    pretty: bool = False


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    return {"status": "ok", "service": "ABIOLA AI Proxy", "upstream": UPSTREAM_URL}


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.post("/api/ask")
async def proxy_ask(req: AskRequest, request: Request):
    """Forward the question to the upstream Render API with the secret key."""
    url = f"{UPSTREAM_URL}/ask"
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": UPSTREAM_API_KEY,
    }
    payload = {"question": req.question, "pretty": req.pretty}

    try:
        r = await request.app.state.http.post(url, json=payload, headers=headers)
    except httpx.TimeoutException:
        logger.error("Upstream timeout")
        raise HTTPException(504, "Upstream timed out")
    except httpx.RequestError as e:
        logger.error(f"Upstream connection error: {e}")
        raise HTTPException(502, "Cannot reach upstream service")

    if r.status_code >= 400:
        # Pass through the upstream error shape where possible
        try:
            detail = r.json().get("detail", r.text)
        except Exception:
            detail = r.text
        logger.warning(f"Upstream returned {r.status_code}: {detail}")
        raise HTTPException(r.status_code, detail)

    return r.json()


@app.get("/api/search")
async def proxy_search(request: Request, q: str, limit: int = 8, alpha: float = 0.7):
    """Optional: proxy the /search endpoint too."""
    url = f"{UPSTREAM_URL}/search"
    headers = {"X-API-Key": UPSTREAM_API_KEY}
    params = {"q": q, "limit": limit, "alpha": alpha, "pretty": True}

    try:
        r = await request.app.state.http.get(url, params=params, headers=headers)
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