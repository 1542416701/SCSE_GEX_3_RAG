#################### AI USE DECLARATION ####################
# AI assistance was used for the development of this file,
# using only free / open-source models — no paid AI service:
#
# AI models used:
#   - Local (Ollama, http://127.0.0.1:11434):
#       Qwen3 family (1.7B / 4B / 8B) for daily coding;
#       qwen3:0.6b as the runtime model of this RAG (the 8B
#       model is NOT used at runtime);
#       nomic-embed-text via Ollama for embeddings.
#   - Initial testing (mandatory): Qwen Flash on Alibaba
#     (limited trial access).
#   - Final testing (mandatory): Qwen3.5-Plus on Alibaba
#     (limited trial access); no other model used for tests.
#
# Compliance: no agent frameworks (OpenClaw, Dify, MetaGPT,
# etc.); this is plain LangChain code. Every AI-assisted
# segment is marked with an [AI-...] tag.
# Tags in this file: [AI-SETUP] [AI-PROMPT] [AI-CHAIN] [AI-MAIN]
#
# Why AI: one clean, consistent framework; helper reuse to
# avoid duplicate code; edge cases (missing facts, retriever
# failures, grounding) handled in one pass.
# Principle: keep it simple — the minimum code needed.
# I reviewed, adjusted and verified all AI-drafted code; the
# usage is documented in the Project Report.
############################################################


"""
Deterministic retrieve-then-answer (RAG) pipeline.

This pipeline always grounds the answer in retrieved policy excerpts. It
does not rely on a small model's ability to plan tool calls, so it is the
reliable entry point for demonstrating grounded answers.

Crucially, when a requested fact is absent from the knowledge base -- such
as an IT Service Desk phone number -- the model is instructed to state that
the detail is not specified instead of inventing one.
"""

import os
import sys

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from rag_tool import get_retriever


# [AI-SETUP] Model / endpoint configuration
LLM_MODEL = "qwen3:0.6b"
OLLAMA_BASE_URL = os.environ.get(
    "OLLAMA_BASE_URL", "http://127.0.0.1:11434"
)

# [AI-PROMPT] System rules enforcing grounding (never invent facts)
SYSTEM_RULES = (
    "You are a university IT support assistant. "
    "Answer using ONLY the policy excerpts provided below. "
    "If the exact requested information is not present in the excerpts, "
    "clearly say that it is not specified or not available. "
    "Never invent phone numbers, email addresses, or other contact details. "
    "Keep the answer concise."
)

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_RULES + "\n\nPolicy excerpts:\n{context}"),
        ("human", "{question}"),
    ]
)


# [AI-CHAIN] Format retrieved documents into one context block
def _format_docs(docs):
    blocks = []
    for doc in docs:
        policy_id = doc.metadata.get("policy_id", "?")
        blocks.append(
            "[policy {0}] {1}".format(policy_id, doc.page_content)
        )
    return "\n\n".join(blocks)


# [AI-CHAIN] LCEL composition: retrieval -> prompt -> model -> parser
def build_rag_chain():
    """Build the context-retrieval + answer generation chain."""
    llm = ChatOllama(
        model=LLM_MODEL, temperature=0, base_url=OLLAMA_BASE_URL
    )
    retriever = get_retriever()

    chain = (
        {
            "context": retriever | _format_docs,
            "question": RunnablePassthrough(),
        }
        | PROMPT
        | llm
        | StrOutputParser()
    )
    return chain


# [AI-CHAIN] Answer a single question using the RAG pipeline
def ask(question):
    """Answer a single question using the RAG pipeline."""
    return build_rag_chain().invoke(question)


# [AI-MAIN] Entry point; only runs when executed directly
if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or (
        "What exact phone number should I call for the "
        "University IT Service Desk?"
    )

    print("QUESTION:", question)
    print("ANSWER:", ask(question))
