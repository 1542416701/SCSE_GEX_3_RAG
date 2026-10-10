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


LLM_MODEL = "qwen3:0.6b"
OLLAMA_BASE_URL = os.environ.get(
    "OLLAMA_BASE_URL", "http://127.0.0.1:11434"
)

SYSTEM_PROMPT = (
    "You are a university IT support assistant. "
    "Use the policy_search tool to find the official policy before answering. "
    "Base your answer strictly on the tool results. "
    "If a requested detail (such as an IT Service Desk phone number) is not "
    "present in the results, say that it is not specified. "
    "Never invent phone numbers or other contact details. "
    "Keep the answer concise."
)


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
