#################### AI USE DECLARATION ####################
# [AI-CHECK] This data-integrity verification script was
# drafted with AI assistance (Qwen via Ollama, local; free
# model, no paid service), then reviewed and run by me.
# It uses no Ollama model or Chroma — only Python checks.
# AI tools used across the project:
#   Qwen3 family (1.7B / 4B / 8B) via Ollama for daily
#   coding; Qwen Flash (initial test) and Qwen3.5-Plus
#   (final test) on Alibaba are the mandatory test models.
# No agent frameworks (OpenClaw / Dify / MetaGPT) used.
############################################################

"""Data integrity check; no Ollama or Chroma required."""
import json
import re
from pathlib import Path
from collections import Counter

root = Path(__file__).resolve().parent
original = [json.loads(line) for line in (root / "original_domain_corpus_clean.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
text = (root / "university_it_policies.txt").read_text(encoding="utf-8")
pattern = re.compile(r"\[POLICY id=(doc-\d+) topic=([a-z_]+)\]\s*\n(.*?)\n\[END POLICY\]", re.DOTALL)
converted = pattern.findall(text)
assert len(original) == len(converted), (len(original), len(converted))
for row, (doc_id, topic, body) in zip(original, converted):
    assert (row["id"], row["topic"], row["text"].strip()) == (doc_id, topic, body.strip())
assert len(set(row["id"] for row in original)) == len(original)
print(f"PASS: {len(original)} original records reproduced exactly, with IDs and topics.")
print("Topics:", dict(Counter(row["topic"] for row in original)))
print("No verified Service Desk phone number is supplied by this dataset.")
