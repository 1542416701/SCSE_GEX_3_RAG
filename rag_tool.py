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
# Tags in this file: [AI-SETUP] [AI-VECTOR] [AI-TOOL]
#
# Why AI: one clean, consistent framework; helper reuse to
# avoid duplicate code; edge cases (missing facts, retriever
# failures, grounding) handled in one pass.
# Principle: keep it simple — the minimum code needed.
# I reviewed, adjusted and verified all AI-drafted code; the
# usage is documented in the Project Report.
############################################################


"""
RAG retrieval tool for the university IT support knowledge base.

Loads the policy file through the given ``policy_loader``, embeds every
policy section into a persistent Chroma vector store, and exposes a
LangChain tool (``policy_search``) that an agent can call to retrieve the
official policy excerpts.
"""

import os
import time

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools import tool

from policy_loader import load_policy_documents


# ----------------------------------------------------------------------
# [AI-SETUP] Configuration: paths, collection, embedding model, top-k
# ----------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
POLICY_FILE = os.path.join(HERE, "university_it_policies.txt")
PERSIST_DIR = os.path.join(HERE, "chroma_db")

COLLECTION_NAME = "university_it_policies"
EMBEDDING_MODEL = "nomic-embed-text"
OLLAMA_BASE_URL = os.environ.get(
    "OLLAMA_BASE_URL", "http://127.0.0.1:11434"
)
TOP_K = 4
EMBED_BATCH_SIZE = 32

# Cached singleton so the vector store is built only once per process.
_vectorstore = None


# ----------------------------------------------------------------------
# [AI-VECTOR] Vector store / retriever
# ----------------------------------------------------------------------

def _add_with_retry(vectorstore, documents, attempts=3):
    """Add one batch of documents, retrying if the local runner restarts."""
    for attempt in range(attempts):
        try:
            vectorstore.add_documents(documents)
            return
        except Exception:
            if attempt == attempts - 1:
                raise
            # A crashed llama-server is relaunched by ollama; wait for it.
            time.sleep(5 * (attempt + 1))


def get_vectorstore():
    """Return the Chroma vector store, building it on first use."""
    global _vectorstore

    if _vectorstore is None:
        embeddings = OllamaEmbeddings(
            model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL
        )

        # Opening the persistent directory reuses previously stored vectors.
        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=PERSIST_DIR,
        )

        # First run: the collection is empty, so populate it from the file.
        stored = vectorstore.get(include=[])
        if len(stored["ids"]) == 0:
            documents = load_policy_documents(POLICY_FILE)
            total = len(documents)
            # Embed in small batches: sending ~1200 sections in one request can
            # overload / crash the local llama-server runner.
            for start in range(0, total, EMBED_BATCH_SIZE):
                batch = documents[start:start + EMBED_BATCH_SIZE]
                _add_with_retry(vectorstore, batch)
                print(
                    "Indexed {0}/{1} policy sections".format(
                        min(start + EMBED_BATCH_SIZE, total), total
                    )
                )

        _vectorstore = vectorstore

    return _vectorstore


def get_retriever():
    """Return a Chroma retriever over the policy sections."""
    return get_vectorstore().as_retriever(search_kwargs={"k": TOP_K})


# ----------------------------------------------------------------------
# [AI-TOOL] Agent tool
# ----------------------------------------------------------------------

@tool("policy_search")
def policy_search(query: str) -> str:
    """Search the official university IT support policy knowledge base.

    Use this for any question about IT services, accounts, passwords,
    printing, VPN, email, storage, phishing, or contacting the IT Service
    Desk. Returns the relevant policy excerpts. The excerpts may not
    contain every requested detail (for example, no phone number is
    provided in this knowledge base).
    """
    docs = get_retriever().invoke(query)

    if not docs:
        return "No relevant policy entry was found in the knowledge base."

    sections = []
    for doc in docs:
        policy_id = doc.metadata.get("policy_id", "?")
        topic = doc.metadata.get("topic", "")
        sections.append(
            "[policy {0} | {1}]\n{2}".format(
                policy_id, topic, doc.page_content
            )
        )

    return "\n\n".join(sections)
