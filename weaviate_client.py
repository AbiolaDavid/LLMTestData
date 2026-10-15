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
            logger.error(f"Failed to connect to Weaviate: {e}", exc_info=True)
            raise RuntimeError("Unable to connect to Weaviate") from e

    return _client


def get_collection(collection_name: str | None = None):
    """Return the named collection (defaults to TestingData)."""
    name = collection_name or COLLECTION_NAME
    client = get_client()
    if not client.collections.exists(name):
        raise RuntimeError(f"Collection '{name}' does not exist in Weaviate")
    return client.collections.get(name)


def search_collection(
    query: str,
    *,
    limit: int = 30,
    alpha: float = 0.7,
    collection_name: str | None = None,
    department: str | None = None,
    level: str | None = None,
) -> list[dict[str, Any]]:
    """Hybrid search against the collection with optional department and level metadata filtering."""
    collection = get_collection(collection_name)

    # Build dynamic metadata filters
    filters = None
    conditions = []

    if department:
        conditions.append(Filter.by_property("department").equal(department))
    if level:
        conditions.append(Filter.by_property("level").equal(level))

    if len(conditions) == 1:
        filters = conditions[0]
    elif len(conditions) > 1:
        filters = Filter.all_of(conditions)

    response = collection.query.hybrid(
        query=query,
        alpha=alpha,
        limit=limit,
        filters=filters,
        return_metadata=MetadataQuery(score=True),
        return_properties=[
            "text",
            "source",
            "module_title",
            "heading",
            "page",
            "chunk_id",
            "department",
            "level",
        ],
    )

    results: list[dict[str, Any]] = []
    for obj in response.objects:
        item = dict(obj.properties)
        if obj.metadata and obj.metadata.score is not None:
            item["score"] = obj.metadata.score
        results.append(item)
    return results


def get_agent(
    collection_name: str | None = None,
    department: str | None = None,
    level: str | None = None,
) -> QueryAgent:
    """Return a QueryAgent dynamically contextualized with user department and level context."""
    name = collection_name or COLLECTION_NAME
    get_collection(name)  # validates collection exists

    # Build student profile context string
    context_parts = []
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
        f'role:"You are a multi-course academic teaching assistant.{user_context} '
        'Task:"Answer using only the retrieved course material. '
        "Identify the relevant course/subject from context; do not assume one course. "
        "Tailor explanations appropriately to the student's level and department context when relevant. "
        "Be clear, accurate, and student-friendly. "
        "Synthesize relevant sources. "
        "Cite source, module_title, Lecturer Name, heading, Sub-heading, and page when available. "
        "Never invent facts, citations, or page numbers. "
        "If the context is insufficient, say the answer is not in the provided materials. "
        'If the question is ambiguous, ask for clarification." '
        'Keywords:"list relevant keywords from the question in your answer at the end of the response."'
    )

    # Instantiated per request to maintain specific system prompt context per student
    agent = QueryAgent(
        client=get_client(),
        collections=[name],
        system_prompt=system_prompt,
    )
    logger.info(f"QueryAgent initialized for collection '{name}' with context: {context_parts}")
    return agent


def close_client() -> None:
    """Close the Weaviate client."""
    global _client
    if _client is not None:
        try:
            _client.close()
            logger.info("Weaviate client closed")
        except Exception as e:
            logger.warning(f"Error while closing Weaviate client: {e}")
        _client = None
