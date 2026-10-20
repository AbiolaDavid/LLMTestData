import os
import logging
from typing import Any

import weaviate
from weaviate.classes.init import Auth, AdditionalConfig, Timeout
from weaviate.classes.query import MetadataQuery, Filter
from weaviate.agents.query import QueryAgent
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

COLLECTION_NAME = os.getenv("WEAVIATE_COLLECTION", "TestingData")

_client: weaviate.WeaviateClient | None = None

# Candidate property names that may hold the chunk body text. Different
# ingestion pipelines use different names, so we auto-detect per collection.
_TEXT_CANDIDATES = (
    "text",
    "content",
    "body",
    "page_content",
    "chunk_text",
    "chunk",
    "document",
)

_WANTED_PROPS = [
    *_TEXT_CANDIDATES,
    "source",
    "course_code",
    "course_title",
    "module_title",
    "lecturer_name",
    "heading",
    "subheading",
    "page",
    "chunk_id",
    "department",
    "level",
    "keywords",
]

# Cache of property names per collection so we never request or filter on
# properties that don't exist in the schema.
_PROPS_CACHE: dict[str, list[str]] = {}


# ---------------------------------------------------------------------------
# Client / collection helpers
# ---------------------------------------------------------------------------
def get_client() -> weaviate.WeaviateClient:
    """Return a connected Weaviate Cloud client (singleton)."""
    global _client
    if _client is None or not _client.is_ready():
        url = os.getenv("WEAVIATE_URL")
        key = os.getenv("WEAVIATE_API_KEY")
        if not url or not key:
            raise RuntimeError("WEAVIATE_URL and WEAVIATE_API_KEY must be set")

        headers: dict[str, str] = {}
        if hf := os.getenv("HUGGINGFACE_API_KEY"):
            headers["X-HuggingFace-Api-Key"] = hf
        if openai := os.getenv("OPENAI_API_KEY"):
            headers["X-OpenAI-Api-Key"] = openai
        if cohere := os.getenv("COHERE_API_KEY"):
            headers["X-Cohere-Api-Key"] = cohere

        try:
            _client = weaviate.connect_to_weaviate_cloud(
                cluster_url=url,
                auth_credentials=Auth.api_key(key),
                headers=headers or None,
                additional_config=AdditionalConfig(
                    timeout=Timeout(init=60, query=240, insert=240)
                ),
            )
            logger.info("Weaviate client connected successfully")
        except Exception as e:
            logger.error("Failed to connect to Weaviate: %s", e, exc_info=True)
            raise RuntimeError("Unable to connect to Weaviate") from e

    return _client


def get_collection(collection_name: str | None = None):
    """Return the named collection (defaults to TestingData)."""
    name = collection_name or COLLECTION_NAME
    client = get_client()
    if not client.collections.exists(name):
        raise RuntimeError(f"Collection '{name}' does not exist in Weaviate")
    return client.collections.get(name)


def _available_props(collection, name: str) -> list[str]:
    """Return the schema property names for the collection (cached)."""
    if name not in _PROPS_CACHE:
        cfg = collection.config.get()
        _PROPS_CACHE[name] = [p.name for p in cfg.properties]
        logger.info("Collection '%s' schema props: %s", name, _PROPS_CACHE[name])
    return _PROPS_CACHE[name]


def _get_text(doc: dict[str, Any]) -> str:
    """Return the first non-empty text field from a document, or ''."""
    for k in _TEXT_CANDIDATES:
        v = doc.get(k)
        if v:
            if isinstance(v, str):
                s = v.strip()
            else:
                s = str(v).strip()
            if s:
                return s
    return ""


def _value_variants(value: str) -> list[str]:
    """Generate common casing variants so Weaviate's case-sensitive equality
    can still match values that were normalized differently during ingestion."""
    v = value.strip()
    seen: dict[str, None] = {}
    for candidate in (v, v.lower(), v.upper(), v.title(), v.capitalize()):
        seen[candidate] = None
    return list(seen.keys())


def _build_eq_filter(prop: str, value: str) -> Filter:
    variants = _value_variants(value)
    if len(variants) == 1:
        return Filter.by_property(prop).equal(variants[0])
    return Filter.any_of([Filter.by_property(prop).equal(v) for v in variants])


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------
def search_collection(
    query: str,
    *,
    limit: int = 12,
    alpha: float = 0.5,
    collection_name: str | None = None,
    department: str | None = None,
    level: str | None = None,
    min_score: float | None = None,
) -> list[dict[str, Any]]:
    """Hybrid search with optional department / level filtering.

    Raises:
        ValueError: if a filter was requested on a property that does not
            exist in the collection schema. This is intentionally loud —
            silently dropping filters leads to misleading cross-department
            answers.
    """
    name = collection_name or COLLECTION_NAME
    collection = get_collection(name)

    props = _available_props(collection, name)
    return_props = [p for p in _WANTED_PROPS if p in props] or None
    query_props = [p for p in _TEXT_CANDIDATES if p in props] or None

    conditions: list[Filter] = []
    if department:
        if "department" not in props:
            raise ValueError(
                "Cannot filter by 'department': property is not present in "
                f"collection '{name}'. Available: {props}"
            )
        conditions.append(_build_eq_filter("department", department))
    if level:
        if "level" not in props:
            raise ValueError(
                "Cannot filter by 'level': property is not present in "
                f"collection '{name}'. Available: {props}"
            )
        conditions.append(_build_eq_filter("level", level))

    if not conditions:
        filters: Filter | None = None
    elif len(conditions) == 1:
        filters = conditions[0]
    else:
        filters = Filter.all_of(conditions)

    response = collection.query.hybrid(
        query=query,
        alpha=alpha,
        limit=limit,
        filters=filters,
        query_properties=query_props,
        return_metadata=MetadataQuery(score=True),
        return_properties=return_props,
    )

    results: list[dict[str, Any]] = []
    dropped_empty = 0
    dropped_score = 0

    for obj in response.objects:
        item = dict(obj.properties)
        score = obj.metadata.score if obj.metadata else None
        if score is not None:
            item["score"] = score

        if not _get_text(item):
            dropped_empty += 1
            logger.warning(
                "Dropping hit with empty text: uuid=%s course=%s props=%s",
                obj.uuid,
                item.get("course_code"),
                list(item.keys()),
            )
            continue

        if min_score is not None and score is not None and score < min_score:
            dropped_score += 1
            continue

        results.append(item)

    logger.info(
        "hybrid search q=%r dept=%s level=%s alpha=%.2f min_score=%s -> "
        "%d raw hits, %d kept (empty=%d, below_score=%d)",
        query,
        department,
        level,
        alpha,
        min_score,
        len(response.objects),
        len(results),
        dropped_empty,
        dropped_score,
    )
    return results


# ---------------------------------------------------------------------------
# Debug helpers (exposed via /debug/* endpoints)
# ---------------------------------------------------------------------------
def debug_schema(collection_name: str | None = None) -> dict[str, Any]:
    """Return the collection schema so callers can confirm property names."""
    name = collection_name or COLLECTION_NAME
    collection = get_collection(name)
    cfg = collection.config.get()
    return {
        "collection": name,
        "vectorizer": cfg.vectorizer,
        "vectorizer_config": str(getattr(cfg, "vectorizer_config", None)),
        "properties": [
            {"name": p.name, "type": p.data_type} for p in cfg.properties
        ],
        "text_candidates_present": [p for p in _TEXT_CANDIDATES
                                    if any(q.name == p for q in cfg.properties)],
    }


def debug_search(
    query: str,
    *,
    limit: int = 3,
    alpha: float = 0.5,
    collection_name: str | None = None,
    department: str | None = None,
    level: str | None = None,
) -> dict[str, Any]:
    """Raw hybrid search returning ALL properties so callers can verify that
    retrieved documents actually carry body text."""
    name = collection_name or COLLECTION_NAME
    collection = get_collection(name)
    props = _available_props(collection, name)

    conditions: list[Filter] = []
    if department and "department" in props:
        conditions.append(_build_eq_filter("department", department))
    if level and "level" in props:
        conditions.append(_build_eq_filter("level", level))

    if not conditions:
        filters: Filter | None = None
    elif len(conditions) == 1:
        filters = conditions[0]
    else:
        filters = Filter.all_of(conditions)

    resp = collection.query.hybrid(
        query=query,
        alpha=alpha,
        limit=limit,
        filters=filters,
        return_metadata=MetadataQuery(score=True),
        return_properties=["*"],
    )

    hits: list[dict[str, Any]] = []
    for obj in resp.objects:
        raw = dict(obj.properties)
        truncated: dict[str, Any] = {}
        for k, v in raw.items():
            if isinstance(v, str) and len(v) > 300:
                truncated[k] = v[:300] + "…"
            else:
                truncated[k] = v
        hits.append(
            {
                "uuid": str(obj.uuid),
                "score": obj.metadata.score if obj.metadata else None,
                "text_prop": next(
                    (k for k in _TEXT_CANDIDATES if raw.get(k)), None
                ),
                "text_len": len(_get_text(raw)),
                "props": truncated,
            }
        )

    return {
        "collection": name,
        "schema_props": props,
        "query": query,
        "alpha": alpha,
        "filters_applied": {
            "department": department if department and "department" in props else None,
            "level": level if level and "level" in props else None,
        },
        "hits": hits,
    }


# ---------------------------------------------------------------------------
# QueryAgent (kept for future use; not wired into the HTTP API)
# ---------------------------------------------------------------------------
def get_agent(
    collection_name: str | None = None,
    department: str | None = None,
    level: str | None = None,
) -> QueryAgent:
    """Return a QueryAgent dynamically contextualized with student context.

    NOTE: This agent is NOT used by the current LangChain pipeline. The
    LangChain chain in `langchain_pipeline.py` is the active orchestrator.
    Kept for future use.
    """
    name = collection_name or COLLECTION_NAME
    get_collection(name)

    context_parts: list[str] = []
    if department:
        context_parts.append(f"Department: {department}")
    if level:
        context_parts.append(f"Level: {level}")

    user_context = (
        f" Student Profile: [{', '.join(context_parts)}]."
        if context_parts
        else ""
    )

    system_prompt = (
        f"Role: You are an expert multi-course academic teaching assistant."
        f"{user_context}\n\n"
        "Retrieval & Knowledge Base Guidelines:\n"
        "1. Answer exclusively from the retrieved knowledge base materials provided in context. "
        "Do not draw on outside knowledge.\n"
        "2. Identify the correct course or subject dynamically from the retrieved chunks.\n"
        "3. Synthesize a coherent, student-friendly answer across modules.\n"
        "4. If information is missing, state clearly: "
        "'The answer to this question is not present in the provided materials.'\n"
        "5. If the question is vague, ask a brief clarifying question before answering.\n\n"
        "Output Formatting:\n"
        "- Use Markdown with section headings (##, ###), bullet points, and bold terms.\n"
        "- Do NOT emit citation blocks or keyword lists; the API layer appends "
        "structured citations and keywords automatically."
    )

    agent = QueryAgent(
        client=get_client(),
        collections=[name],
        system_prompt=system_prompt,
    )
    logger.info(
        "QueryAgent initialized for collection '%s' with context: %s",
        name,
        context_parts,
    )
    return agent


def close_client() -> None:
    """Close the Weaviate client."""
    global _client
    if _client is not None:
        try:
            _client.close()
            logger.info("Weaviate client closed")
        except Exception as e:
            logger.warning("Error while closing Weaviate client: %s", e)
        _client = None
