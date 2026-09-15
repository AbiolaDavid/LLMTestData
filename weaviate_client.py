import os
from typing import Any

import weaviate
from weaviate.classes.init import Auth, AdditionalConfig, Timeout
from weaviate.classes.query import MetadataQuery
from weaviate.agents.query import QueryAgent
from dotenv import load_dotenv

load_dotenv()

COLLECTION_NAME = os.getenv("WEAVIATE_COLLECTION", "TestingData")

_client: weaviate.WeaviateClient | None = None
_agent: QueryAgent | None = None
_agent_collection: str | None = None


def get_client() -> weaviate.WeaviateClient:
    """Return a connected Weaviate Cloud client (singleton)."""
    global _client
    if _client is None or not _client.is_ready():
        url = os.getenv("WEAVIATE_URL")
        key = os.getenv("WEAVIATE_API_KEY")
        if not url or not key:
            raise RuntimeError("WEAVIATE_URL and WEAVIATE_API_KEY must be set")

        headers: dict[str, str] = {}
        hf = os.getenv("HUGGINGFACE_API_KEY")
        if hf:
            headers["X-HuggingFace-Api-Key"] = hf
        openai = os.getenv("OPENAI_API_KEY")
        if openai:
            headers["X-OpenAI-Api-Key"] = openai
        cohere = os.getenv("COHERE_API_KEY")
        if cohere:
            headers["X-Cohere-Api-Key"] = cohere

        _client = weaviate.connect_to_weaviate_cloud(
            cluster_url=url,
            auth_credentials=Auth.api_key(key),
            headers=headers or None,
            additional_config=AdditionalConfig(
                timeout=Timeout(init=60, query=240, insert=240)
            ),
        )
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
    limit: int = 8,
    alpha: float = 0.7,
    collection_name: str | None = None,
) -> list[dict[str, Any]]:
    """Hybrid search against the collection; returns plain dicts + score."""
    collection = get_collection(collection_name)
    response = collection.query.hybrid(
        query=query,
        alpha=alpha,
        limit=limit,
        return_metadata=MetadataQuery(score=True),
        return_properties=[
            "text",
            "source",
            "module_title",
            "heading",
            "page",
            "chunk_id",
        ],
    )

    results: list[dict[str, Any]] = []
    for obj in response.objects:
        item = dict(obj.properties)
        if obj.metadata and obj.metadata.score is not None:
            item["score"] = obj.metadata.score
        results.append(item)
    return results


def get_agent(collection_name: str | None = None) -> QueryAgent:
    """Return a QueryAgent bound to the given collection (cached)."""
    global _agent, _agent_collection
    name = collection_name or COLLECTION_NAME
    if _agent is None or _agent_collection != name:
        get_collection(name)
        _agent = QueryAgent(
            client=get_client(),
            collections=[name],
            system_prompt=(
                "You are a helpful teaching assistant for an Introduction to Sociology "
                "course. Answer only from the provided course material. Cite the module "
                "title, heading and page when possible. If the answer is not in the "
                "material, say you do not know."
            ),
        )
        _agent_collection = name
    return _agent


def close_client() -> None:
    """Close the Weaviate client and clear cached agent."""
    global _client, _agent, _agent_collection
    if _client is not None:
        _client.close()
        _client = None
    _agent = None
    _agent_collection = None
