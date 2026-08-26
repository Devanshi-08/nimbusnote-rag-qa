"""A small, transparent retrieval-augmented Q&A engine.

The engine deliberately keeps retrieval and answering separate: answers are built
only from sentences in the retrieved Markdown sections.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import log, sqrt
from pathlib import Path
import re

WORD_RE = re.compile(r"[a-z0-9]+(?:\.[a-z0-9]+)?")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'])")
STOP_WORDS = {
    "a", "an", "are", "can", "does", "for", "happen", "how", "i", "is",
    "it", "my", "of", "the", "to", "what", "when", "why", "with",
    "nimbusnote",
}


@dataclass(frozen=True)
class Chunk:
    source: str
    heading: str
    text: str

    @property
    def label(self) -> str:
        return f"{self.source} — {self.heading}"


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float


def tokens(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def query_tokens(text: str) -> list[str]:
    """Discard question words, so generic wording does not become evidence."""
    return [token for token in tokens(text) if token not in STOP_WORDS]


def load_chunks(documents_dir: Path) -> list[Chunk]:
    """Split each Markdown document into H2 sections, preserving its source."""
    chunks: list[Chunk] = []
    # The supplied knowledge base is intentionally limited to numbered source
    # documents. This prevents operational files such as README.md from being
    # accidentally treated as answer evidence.
    for path in sorted(documents_dir.glob("[0-9][0-9]-*.md")):
        title = path.stem
        heading = title
        body: list[str] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("# "):
                title = line[2:].strip()
            elif line.startswith("## "):
                if body:
                    chunks.append(Chunk(path.name, heading, "\n".join(body).strip()))
                heading, body = line[3:].strip(), []
            else:
                body.append(line)
        if body:
            chunks.append(Chunk(path.name, heading, "\n".join(body).strip()))
    return [chunk for chunk in chunks if chunk.text]


class Retriever:
    def __init__(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("No Markdown sections found to index.")
        self.chunks = chunks
        self.doc_terms = [Counter(tokens(f"{c.heading} {c.text}")) for c in chunks]
        document_frequency = Counter({term: 0 for terms in self.doc_terms for term in terms})
        for terms in self.doc_terms:
            for term in terms:
                document_frequency[term] += 1
        count = len(chunks)
        self.idf = {term: log((count + 1) / (frequency + 1)) + 1 for term, frequency in document_frequency.items()}
        self.vectors = [self._vector(terms) for terms in self.doc_terms]

    def _vector(self, term_counts: Counter[str]) -> dict[str, float]:
        raw = {term: count * self.idf.get(term, 0.0) for term, count in term_counts.items()}
        magnitude = sqrt(sum(value * value for value in raw.values()))
        return {term: value / magnitude for term, value in raw.items()} if magnitude else {}

    def search(self, question: str, limit: int = 3) -> list[SearchResult]:
        query = self._vector(Counter(query_tokens(question)))
        ranked = [
            SearchResult(chunk, sum(query.get(term, 0.0) * value for term, value in vector.items()))
            for chunk, vector in zip(self.chunks, self.vectors)
        ]
        return sorted(ranked, key=lambda item: item.score, reverse=True)[:limit]


def answer(question: str, results: list[SearchResult]) -> str:
    """Return the most relevant evidence sentences; never invent unsupported facts."""
    if not results or results[0].score == 0:
        return "I couldn't find an answer in the indexed NimbusNote documents."
    query_terms = set(query_tokens(question))
    candidates: list[tuple[float, str]] = []
    for result in results:
        for sentence in SENTENCE_RE.split(re.sub(r"\s+", " ", result.chunk.text)):
            overlap = len(query_terms.intersection(tokens(sentence)))
            if overlap:
                candidates.append((result.score * overlap, sentence.strip()))
    selected = [sentence for _, sentence in sorted(candidates, reverse=True)[:2]]
    return " ".join(selected) or results[0].chunk.text.replace("\n", " ")
