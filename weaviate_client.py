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

# Cache of property names per collection so we never request or filter on
# properties that don't exist in the schema.
_PROPS_CACHE: dict[str, list[str]] = {}

_WANTED_PROPS = [
    "text",
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


def _available_props(collection, name: str) -> list[str]:
    """Return the schema property names for the collection (cached)."""
    if name not in _PROPS_CACHE:
        cfg = collection.config.get()
        _PROPS_CACHE[name] = [p.name for p in cfg.properties]
        logger.info("Collection '%s' schema props: %s", name, _PROPS_CACHE[name])
    return _PROPS_CACHE[name]


def search_collection(
    query: str,
    *,
    limit: int = 30,
    alpha: float = 0.7,
    collection_name: str | None = None,
    department: str | None = None,
    level: str | None = None,
) -> list[dict[str, Any]]:
    """Hybrid search with optional department / level filtering."""
    name = collection_name or COLLECTION_NAME
    collection = get_collection(name)

    props = _available_props(collection, name)
    return_props = [p for p in _WANTED_PROPS if p in props]

    conditions = []
    if department:
        if "department" in props:
            conditions.append(Filter.by_property("department").equal(department))
        else:
            logger.warning(
                "Department filter '%s' ignored — 'department' not in schema",
                department,
            )
    if level:
        if "level" in props:
            conditions.append(Filter.by_property("level").equal(level))
        else:
            logger.warning(
                "Level filter '%s' ignored — 'level' not in schema", level
            )

    filters = None
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
        return_properties=return_props,
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
    """Return a QueryAgent dynamically contextualized with student context."""
    name = collection_name or COLLECTION_NAME
    get_collection(name)

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
        f'Role: "You are an expert multi-course academic teaching assistant.\n{user_context}\n\n'
        "Retrieval & Knowledge Base Guidelines:\n"
        "1. Information Source: Answer queries exclusively using the retrieved knowledge base materials provided in the context. Do not draw on outside knowledge.\n"
        "2. Multi-Course Context Awareness: Identify the correct course or subject from the context chunks dynamically.\n"
        "3. Contextual Retrieval & Synthesis: Synthesize a coherent, student-friendly answer across modules.\n"
        "4. Handling Missing Information: If information is missing, state clearly: 'The answer to this question is not present in the provided materials.' Do not guess.\n"
        "5. Clarification: If the question is vague, ask a brief clarifying question before answering.\n\n"
        "Output Formatting Guidelines:\n"
        "- Professional Presentation: Use Markdown formatting with section headings (##, ###), bullet points, and bold terms.\n"
        "- Citation Format Requirement: Always include response citations in the following exact list format at the end of your citation sections:\n"
        "  - Course Code: [Course Code]\n"
        "  - Course Title: [Course Title]\n"
        "  - Department: [Department]\n"
        "  - Level: [Level]\n"
        "  - Lecturer Name: [Lecturer Name]\n"
        "  - Heading & Subheading: [Heading] > [Subheading]\n\n"
        "Keywords for Further Reading Section:\n"
        "- Include a dedicated section listing key topics and suggested terms strictly in list format:\n"
        "  ### Keywords for Further Reading\n"
        "  - [Keyword 1]\n"
        "  - [Keyword 2]\n"
        "  - [Keyword 3]\""
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
            logger.warning(f"Error while closing Weaviate client: {e}")
        _client = None
