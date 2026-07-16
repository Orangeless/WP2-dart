import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import config
import dataio
import resolvers


class QueryPhraseTests(unittest.TestCase):
    def test_prefers_dataset_question_over_property_id(self):
        inst = {"subject": "Martin Lambie-Nairn", "relation": "P166",
                "question": "Which award did Martin Lambie-Nairn receive?"}
        self.assertEqual(dataio.query_phrase(inst),
                         "Which award did Martin Lambie-Nairn receive?")

    def test_synthesises_from_readable_relation(self):
        inst = {"subject": "Marie Curie", "relation": "field of work"}
        self.assertEqual(dataio.query_phrase(inst),
                         "What is the field of work of Marie Curie?")

    def test_rejects_bare_property_id_with_no_question(self):
        with self.assertRaises(ValueError):
            dataio.query_phrase({"subject": "X", "relation": "P166"})

    def test_property_id_never_reaches_the_prompt(self):
        seen = {}

        def fake_chat(prompt, model, **kwargs):
            seen["prompt"] = prompt
            return '{"object": "Example", "chosen_source": 1}'

        with patch.object(resolvers, "chat", side_effect=fake_chat):
            resolvers.llm_judge("Martin Lambie-Nairn",
                                "Which award did Martin Lambie-Nairn receive?",
                                [], "model", "1", [])

        self.assertIn("Which award did Martin Lambie-Nairn receive?", seen["prompt"])
        self.assertNotIn("P166", seen["prompt"])


class LlmAdjudicationPromptTests(unittest.TestCase):
    def test_disable_abstain_prompt_forbids_null(self):
        original = config.ALLOW_ABSTAIN
        config.ALLOW_ABSTAIN = False
        try:
            seen = {}

            def fake_chat(prompt, model, **kwargs):
                seen["prompt"] = prompt
                return '{"object": "Example"}'

            with patch.object(resolvers, "chat", side_effect=fake_chat):
                resolvers._llm_adjudicate("Subject", "relation", [], "model", "1", [], False)
        finally:
            config.ALLOW_ABSTAIN = original

        self.assertIn("Do not return null", seen["prompt"])
        self.assertNotIn("<value or null>", seen["prompt"])

    def test_llm_judge_plain_does_not_request_provenance(self):
        """llm_judge (no provenence) should NOT ask for chosen_source."""
        seen = {}

        def fake_chat(prompt, model, **kwargs):
            seen["prompt"] = prompt
            return '{"object": "Example"}'

        with patch.object(resolvers, "chat", side_effect=fake_chat):
            resolvers.llm_judge("Subject", "relation", [], "model", "1", [])

        self.assertNotIn("chosen_source", seen["prompt"])

    def test_llm_judge_provenance_requests_source_selection(self):
        """llm_judge_provenance SHOULD ask for chosen_source."""
        seen = {}

        def fake_chat(prompt, model, **kwargs):
            seen["prompt"] = prompt
            return '{"object": "Example", "chosen_source": 1}'

        with patch.object(resolvers, "chat", side_effect=fake_chat):
            resolvers.llm_judge_provenance("Subject", "relation", [], "model", "1", [])

        self.assertIn("chosen_source", seen["prompt"])


if __name__ == "__main__":
    unittest.main()
