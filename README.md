# NimbusNote · Retrieval-first Q&A

> A transparent, dependency-free RAG mini-project over the supplied NimbusNote documentation.

This bot retrieves evidence before it answers. It is deliberately **not** a thin wrapper around an LLM: every answer is formed from sentences in the highest-ranked local document sections.

## Why it is RAG

```text
Question
   │
   ▼
Markdown sections ──► TF–IDF + cosine ranking ──► top matching passages
                                                       │
                                                       ▼
                                              extractive grounded answer
```

The command line output makes the boundary visible: `RETRIEVED PASSAGES` (with file, section, and score) is always shown before `ANSWER`.

## Quick start

Python 3.10+ is the only requirement.

```powershell
python app.py "Why can't I upload an image?"
python app.py "What happens when I downgrade?" --top-k 2
```

Example output:

```text
RETRIEVED PASSAGES
[1] 03-troubleshooting.md — "I can't upload an image" (score: 0.592)
...

ANSWER
Image attachments are a Pro and Team plan feature only.
```

## Design notes

| Stage | Implementation | Guardrail |
| --- | --- | --- |
| Ingest | Heading-level chunks from `01-*`, `02-*`, and `03-*` Markdown documents | Operational files such as this README are never indexed. |
| Retrieve | Local TF–IDF vectors and cosine similarity | Section headings receive a modest metadata boost. |
| Answer | Extractive sentence selection from retrieved chunks | Unknown questions return an explicit “not found” response. |

## Verify

```powershell
python -m unittest -v
```

The test suite covers relevant retrieval, grounded answers, missing-information handling, and invalid result limits.

## Project structure

```text
app.py          # CLI: show evidence, then answer
rag.py          # parsing, ranking, and grounded answer selection
test_rag.py     # standard-library test suite
0*-*.md         # the fixed NimbusNote knowledge base
```
