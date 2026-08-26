from pathlib import Path
import unittest

from rag import Retriever, answer, load_chunks


DOCS = Path(__file__).parent


class RagTests(unittest.TestCase):
    def test_image_question_retrieves_troubleshooting_section(self):
        results = Retriever(load_chunks(DOCS)).search("Why can't I upload an image?", limit=1)
        self.assertEqual(results[0].chunk.source, "03-troubleshooting.md")
        self.assertEqual(results[0].chunk.heading, '"I can\'t upload an image"')

    def test_answer_is_grounded_in_retrieved_text(self):
        results = Retriever(load_chunks(DOCS)).search("How often does background sync happen?", limit=1)
        response = answer("How often does background sync happen?", results)
        self.assertIn("5 minutes", response)

    def test_unknown_question_does_not_hallucinate(self):
        results = Retriever(load_chunks(DOCS)).search("What colour is the NimbusNote logo?", limit=3)
        self.assertTrue(answer("What colour is the NimbusNote logo?", results).startswith("I couldn't find"))

    def test_invalid_result_limit_is_rejected(self):
        with self.assertRaises(ValueError):
            Retriever(load_chunks(DOCS)).search("sync", limit=0)
