"""Command-line interface for the NimbusNote retrieval-first Q&A bot."""

from __future__ import annotations

import argparse
from pathlib import Path

from rag import Retriever, answer, load_chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a question grounded in the local NimbusNote docs.")
    parser.add_argument("question", nargs="+", help="Question to ask")
    parser.add_argument("--top-k", type=int, default=3, help="Number of sections to retrieve (default: 3)")
    parser.add_argument("--docs", type=Path, default=Path(__file__).parent, help="Directory containing Markdown documents")
    args = parser.parse_args()

    question = " ".join(args.question)
    results = Retriever(load_chunks(args.docs)).search(question, args.top_k)
    print("RETRIEVED PASSAGES")
    for index, result in enumerate(results, 1):
        print(f"\n[{index}] {result.chunk.label} (score: {result.score:.3f})")
        print(result.chunk.text)
    print(f"\nANSWER\n{answer(question, results)}")


if __name__ == "__main__":
    main()
