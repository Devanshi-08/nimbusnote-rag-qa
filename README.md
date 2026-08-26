# NimbusNote retrieval-first Q&A bot

This is a self-contained RAG mini-project over the three included NimbusNote documents. It does **not** send the question straight to a model: it first indexes local Markdown sections, ranks them with TF–IDF cosine similarity, exposes the matched passages and their scores, and only then creates an extractive answer from that evidence.

## Run it

Requires Python 3.10+ and no packages:

```powershell
python app.py "Why can't I upload an image?"
python app.py "What happens when I downgrade?" --top-k 2
```

The CLI prints `RETRIEVED PASSAGES` before `ANSWER`, so the retrieval step is observable and inspectable.

## Verify it

```powershell
python -m unittest -v
```

## How it works

1. `load_chunks` parses each Markdown file into heading-level chunks and retains the filename and heading as metadata.
2. `Retriever` tokenizes each chunk, builds a local TF–IDF index, and ranks chunks by cosine similarity to the question.
3. `answer` selects relevant sentences solely from those retrieved chunks. If none overlap with the question, it explicitly says that the answer is absent instead of inventing one.

The compact, dependency-free implementation makes the key RAG boundary easy to audit in `rag.py`.
