# langchain_pipeline.py
# LangChain LCEL orchestration for ABIOLA AI RAG pipeline.
#
# Responsibilities:
#   - Wrap Weaviate hybrid retrieval (search_collection) as a Runnable
#   - Build the dual-response chain: internal (grounded) + external (Cohere)
#   - Preserve student department/level context in prompts
#   - Gracefully degrade when the LLM is unavailable

import os
import logging
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnablePassthrough,
    RunnableBranch,
)
from langchain_openai import ChatOpenAI

from weaviate_client import search_collection

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (mirrors api.py env vars)
# ---------------------------------------------------------------------------
COHERE_API_KEY = (os.environ.get("COHERE_API_KEY") or "").strip()
COHERE_BASE_URL = os.environ.get(
    "COHERE_BASE_URL", "https://api.cohere.ai/compatibility/v1"
)
COHERE_MODEL = os.environ.get("COHERE_MODEL", "command-a-03-2025")

_llm: ChatOpenAI | None = None


def get_llm() -> ChatOpenAI | None:
    """Singleton LLM via OpenAI-compatible Cohere endpoint."""
    global _llm
    if _llm is None:
        if not COHERE_API_KEY:
            logger.warning("COHERE_API_KEY not set — external expansion disabled")
            return None
        _llm = ChatOpenAI(
            model=COHERE_MODEL,
            api_key=COHERE_API_KEY,
            base_url=COHERE_BASE_URL,
            temperature=0.2,
        )
    return _llm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _format_docs(docs: list[dict[str, Any]]) -> str:
    if not docs:
        return "(no relevant documents found)"
    parts = []
    for i, d in enumerate(docs, 1):
        parts.append(
            f"[{i}] Source: {d.get('source', '?')} | "
            f"Module: {d.get('module_title', '?')} | "
            f"Heading: {d.get('heading', '?')} | "
            f"Page: {d.get('page', '?')}\n"
            f"{d.get('text', '')}"
        )
    return "\n\n".join(parts)


def _is_internal_empty(x: dict[str, Any]) -> bool:
    return "not in the provided materials" in x["internal_answer"].lower()


def _safe(runnable: Runnable, fallback_message: str) -> Runnable:
    """Wrap a runnable so LLM/network failures don't break the whole request."""
    return runnable.with_fallbacks([RunnableLambda(lambda _: fallback_message)])


# ---------------------------------------------------------------------------
# Chain factory
# ---------------------------------------------------------------------------
def build_dual_response_chain(
    department: str | None = None,
    level: str | None = None,
    limit: int = 8,
    alpha: float = 0.7,
) -> Runnable:
    """
    Build a per-request LCEL chain.

    Input:  {"question": str}
    Output: {
        "question": str,
        "docs": list[dict],        # raw Weaviate hits (for source citations)
        "context": str,
        "internal_answer": str,
        "external_answer": str,
    }
    """
    llm = get_llm()

    # ---- Student profile context for prompts ----
    context_parts: list[str] = []
    if department:
        context_parts.append(f"Department: {department}")
    if level:
        context_parts.append(f"Level: {level}")
    student_context = (
        f" Student Profile: [{', '.join(context_parts)}]." if context_parts else ""
    )

    # ---- Retriever (wraps existing hybrid search, preserving alpha + filters) ----
    # Wrapped in try/except so a Weaviate failure degrades to an empty result
    # set (and the "not in provided materials" branch of the prompts) instead
    # of surfacing as an opaque 500 from /ask.
    def _retrieve(inputs: dict[str, Any]) -> list[dict[str, Any]]:
        try:
            return search_collection(
                inputs["question"],
                limit=limit,
                alpha=alpha,
                department=department,
                level=level,
            )
        except Exception as e:
            logger.error(
                "Retrieval failed for question %r: %s",
                inputs.get("question"),
                e,
                exc_info=True,
            )
            return []

    retriever = RunnableLambda(_retrieve)

    # ---- Prompts ----
    internal_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"You are a multi-course academic teaching assistant.{student_context} "
                "Answer using only the retrieved course material. "
                "Identify the relevant course/subject from context; do not assume one course. "
                "Tailor explanations to the student's level and department. "
                "Cite source, module_title, heading, and page when available. "
                "Never invent facts, citations, or page numbers. "
                "If the context is insufficient, say the answer is not in the provided materials.",
            ),
            ("human", "Retrieved Materials:\n{context}\n\nQuestion: {question}"),
        ]
    )

    external_normal_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"You are an expert academic supplement.{student_context} "
                "An internal knowledge base provided a grounded answer. "
                "Provide a related, broader, or complementary perspective. "
                "CRITICAL RULES: 1) DO NOT repeat the internal answer. "
                "2) DO NOT cite internal sources. 3) Provide new, additive value only.",
            ),
            (
                "human",
                "Original Question: {question}\n\nInternal Answer: {internal_answer}",
            ),
        ]
    )

    external_empty_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"You are an expert academic assistant.{student_context} "
                "The internal knowledge base could not find an answer. "
                "Provide a comprehensive, well-structured answer based on pre-trained knowledge.",
            ),
            ("human", "Original Question: {question}"),
        ]
    )

    # ---- Internal step ----
    if llm is None:
        internal_generate: Runnable = RunnableLambda(
            lambda _: "Internal answer generation is currently unavailable."
        )
    else:
        internal_generate = _safe(
            RunnableLambda(
                lambda x: {"context": x["context"], "question": x["question"]}
            )
            | internal_prompt
            | llm
            | StrOutputParser(),
            "Internal answer generation failed.",
        )

    internal_step = RunnablePassthrough.assign(
        context=RunnableLambda(lambda x: _format_docs(x["docs"]))
    ) | RunnablePassthrough.assign(internal_answer=internal_generate)

    # ---- External step ----
    if llm is None:
        external_step = RunnablePassthrough.assign(
            external_answer=RunnableLambda(
                lambda _: "External expansion temporarily unavailable."
            )
        )
    else:
        external_normal_generate = _safe(
            RunnableLambda(
                lambda x: {
                    "question": x["question"],
                    "internal_answer": x["internal_answer"],
                }
            )
            | external_normal_prompt
            | llm
            | StrOutputParser(),
            "External expansion temporarily unavailable.",
        )

        external_empty_generate = _safe(
            RunnableLambda(lambda x: {"question": x["question"]})
            | external_empty_prompt
            | llm
            | StrOutputParser(),
            "External expansion temporarily unavailable.",
        )

        external_generate = RunnableBranch(
            (RunnableLambda(_is_internal_empty), external_empty_generate),
            external_normal_generate,
        )

        external_step = RunnablePassthrough.assign(external_answer=external_generate)

    # ---- Full chain ----
    return (
        RunnablePassthrough.assign(docs=retriever)
        | internal_step
        | external_step
    )
