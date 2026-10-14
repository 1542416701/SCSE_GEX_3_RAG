#################### AI USE DECLARATION ####################
# AI tool: Qwen, run locally through Ollama.
# [AI-CHECK] This verification script was AI-drafted, then
# reviewed and run by me; it uses no model or Chroma.
#
# Why AI: produce the integrity checks in one pass and avoid
# hand-written repetitive assertions.
# Course compliance: only Qwen models (local Ollama; Qwen Flash
# initial test, Qwen3.5-Plus final test); no agent frameworks.
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
