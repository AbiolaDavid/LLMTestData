# langchain_pipeline.py
# LangChain LCEL orchestration for ABIOLA AI RAG pipeline.

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


def _format_docs(docs: list[dict[str, Any]]) -> str:
    if not docs:
        return "(no relevant documents found)"
    parts = []
    for i, d in enumerate(docs, 1):
        heading = d.get("heading") or ""
        subheading = d.get("subheading") or ""
        combined_heading = (
            f"{heading} > {subheading}".strip(" >")
            if heading or subheading
            else d.get("heading_and_subheading", "N/A")
        )

        parts.append(
            f"[{i}] Course Code: {d.get('course_code', 'N/A')} | "
            f"Course Title: {d.get('course_title') or d.get('module_title', 'N/A')} | "
            f"Department: {d.get('department', 'N/A')} | "
            f"Level: {d.get('level', 'N/A')} | "
            f"Lecturer: {d.get('lecturer_name', 'N/A')} | "
            f"Heading & Subheading: {combined_heading} | "
            f"Page: {d.get('page', 'N/A')}\n"
            f"Text: {d.get('text', '')}"
        )
    return "\n\n".join(parts)


def _is_internal_empty(x: dict[str, Any]) -> bool:
    return "not in the provided materials" in x["internal_answer"].lower()


def _safe(runnable: Runnable, fallback_message: str) -> Runnable:
    """Wrap a runnable so LLM/network failures don't break the whole request."""
    return runnable.with_fallbacks([RunnableLambda(lambda _: fallback_message)])


def build_dual_response_chain(
    department: str | None = None,
    level: str | None = None,
    limit: int = 8,
    alpha: float = 0.7,
) -> Runnable:
    """Build a per-request LCEL chain with explicit metadata and keyword placeholders."""
    llm = get_llm()

    context_parts: list[str] = []
    if department:
        context_parts.append(f"Department: {department}")
    if level:
        context_parts.append(f"Level: {level}")
    student_context = (
        f" Student Profile: [{', '.join(context_parts)}]." if context_parts else ""
    )

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

    # System prompt integrating metadata placeholders and list-formatted keywords
    internal_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"You are a multi-course academic teaching assistant.{student_context}\n\n"
                "INSTRUCTIONS:\n"
                "1. Answer using ONLY the retrieved course materials. Do not draw on external knowledge.\n"
                "2. Identify the relevant course or subject dynamically.\n"
                "3. If the answer is missing, state clearly: 'The answer to this question is not present in the provided materials.'\n"
                "4. Always include metadata citations directly in your response in list format as follows:\n"
                "   - Course Code: {{course_code}}\n"
                "   - Course Title: {{course_title}}\n"
                "   - Department: {{department}}\n"
                "   - Level: {{level}}\n"
                "   - Lecturer Name: {{lecturer_name}}\n"
                "   - Heading & Subheading: {{heading_and_subheading}}\n\n"
                "5. Always end your response with a dedicated section listing relevant keywords strictly in list format:\n"
                "   ### Keywords for Further Reading\n"
                "   - {{keyword_1}}\n"
                "   - {{keyword_2}}\n"
                "   - {{keyword_3}}",
            ),
            ("human", "Retrieved Materials:\n{context}\n\nQuestion: {question}"),
        ]
    )

    external_normal_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"You are an expert academic supplement.{student_context}\n"
                "An internal knowledge base provided a grounded answer. "
                "Provide a related, broader, or complementary perspective.\n"
                "CRITICAL RULES:\n"
                "1) DO NOT repeat the internal answer.\n"
                "2) DO NOT cite internal sources.\n"
                "3) Provide new, additive value only.\n"
                "4) Include relevant keywords for further reading strictly in list format at the end:\n"
                "   ### Keywords for Further Reading\n"
                "   - {{keyword_1}}\n"
                "   - {{keyword_2}}\n"
                "   - {{keyword_3}}",
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
                f"You are an expert academic assistant.{student_context}\n"
                "The internal knowledge base could not find an answer. "
                "Provide a comprehensive, well-structured answer based on pre-trained knowledge.\n"
                "Always include relevant keywords for further reading strictly in list format at the end:\n"
                "### Keywords for Further Reading\n"
                "- {{keyword_1}}\n"
                "- {{keyword_2}}\n"
                "- {{keyword_3}}",
            ),
            ("human", "Original Question: {question}"),
        ]
    )

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

    return (
        RunnablePassthrough.assign(docs=retriever)
        | internal_step
        | external_step
    )
