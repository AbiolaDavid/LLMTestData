import os
import weaviate
from weaviate.classes.init import Auth, AdditionalConfig, Timeout
from weaviate.agents.query import QueryAgent
from dotenv import load_dotenv

load_dotenv()

COLLECTION_NAME = os.getenv("WEAVIATE_COLLECTION", "TestingData")

_client = None
_agent = None
_agent_collection = None

def get_client() -> weaviate.WeaviateClient:
    global _client
    if _client is None or not _client.is_ready():
        url = os.getenv("WEAVIATE_URL")
        key = os.getenv("WEAVIATE_API_KEY")
        if not url or not key:
            raise RuntimeError("WEAVIATE_URL and WEAVIATE_API_KEY must be set")
        _client = weaviate.connect_to_weaviate_cloud(
            cluster_url=url,
            auth_credentials=Auth.api_key(key),
            additional_config=AdditionalConfig(
                timeout=Timeout(init=60, query=240, insert=240)
            ),
        )
    return _client

def get_agent(collection_name: str | None = None) -> QueryAgent:
    global _agent, _agent_collection
    name = collection_name or COLLECTION_NAME
    if _agent is None or _agent_collection != name:
        _agent = QueryAgent(client=get_client(), collections=[name])
        _agent_collection = name
    return _agent

def close_client():
    global _client
    if _client is not None:
        _client.close()
        _client = None
