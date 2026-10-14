#################### AI USE DECLARATION ####################
# AI tool: Qwen, run locally through Ollama, for framework
# drafting; all code was reviewed, adjusted and verified by me.
#
# AI-assisted segments (tags inline) and methods used:
#   [AI-SETUP]  configuration — model name, endpoint, paths
#   [AI-PROMPT] prompt design — system rules enforcing grounding
#   [AI-AGENT]  agent construction — bind model to policy_search
#   [AI-MAIN]   entry guard — if __name__ == "__main__"
#
# Why AI was used:
#   - one clean, consistent framework (readable and presentable);
#   - avoid repetitive code through shared helpers;
#   - edge cases (missing facts, retriever failures) handled in
#     one pass.
# Guiding principle: keep it simple — the minimum code needed.
# Course compliance: only Qwen models (local Ollama for dev;
# Qwen Flash for initial test, Qwen3.5-Plus for final test);
# no agent frameworks (OpenClaw / Dify / MetaGPT).
# Author responsibility: verified by running check_data, rag.py
# and agent.py — grounded answers, no invented phone number.
############################################################


"""
Agent that answers university IT questions by searching the policy
knowledge base through the ``policy_search`` retrieval tool.

The agent runs the qwen3:0.6b model locally through Ollama and grounds its
answers in the retrieved policy excerpts. If a requested detail (such as an
IT Service Desk phone number) is not present in the knowledge base, the
agent states that it is not specified instead of inventing it.
"""

import os
import sys

from langchain_ollama import ChatOllama
from langchain.agents import create_agent

from rag_tool import policy_search


# [AI-SETUP] Model / endpoint configuration
LLM_MODEL = "qwen3:0.6b"
OLLAMA_BASE_URL = os.environ.get(
    "OLLAMA_BASE_URL", "http://127.0.0.1:11434"
)

# [AI-PROMPT] System prompt enforcing grounding (never invent facts)
SYSTEM_PROMPT = (
    "You are a university IT support assistant. "
    "Use the policy_search tool to find the official policy before answering. "
    "Base your answer strictly on the tool results. "
    "If a requested detail (such as an IT Service Desk phone number) is not "
    "present in the results, say that it is not specified. "
    "Never invent phone numbers or other contact details. "
    "Keep the answer concise."
)


# [AI-AGENT] Tool-calling agent construction
def build_agent():
    """Build the tool-calling agent backed by qwen3:0.6b."""
    llm = ChatOllama(
        model=LLM_MODEL, temperature=0, base_url=OLLAMA_BASE_URL
    )
    return create_agent(
        llm,
        [policy_search],
        system_prompt=SYSTEM_PROMPT,
    )


# [AI-MAIN] Entry point; only runs when executed directly
def main():
    question = " ".join(sys.argv[1:]) or (
        "What exact phone number should I call for the "
        "University IT Service Desk?"
    )

    agent = build_agent()
    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]}
    )

    print("QUESTION:", question)
    print("ANSWER:", result["messages"][-1].content)


if __name__ == "__main__":
    main()
