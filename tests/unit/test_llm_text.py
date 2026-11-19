"""The language-model step: checks on the answer, retries, fallback to rule-based points. No network is used."""

import json
import tempfile
import unittest
from pathlib import Path

from src.reporting import lag_facts, llm_text

ROOT = Path(__file__).resolve().parents[2]
VOCABULARY = lag_facts.load_vocabulary(ROOT / "configs" / "report_vocabulary.yaml")
ROWS, METADATA = lag_facts.load_results(ROOT / "tests" / "fixtures" / "lag_results_example.json")
FACTS = lag_facts.build_facts(ROWS, VOCABULARY, METADATA)


def first(direction, strength=None):
    return next(f for f in FACTS["facts"] if f["direction"] == direction
                and (strength is None or f["evidence"]["strength"] == strength))


def careful_answer():
    good, bad = first("favourable", "suggestive"), first("unfavourable", "suggestive")
    links = FACTS["analysis"]["links_tested"]
    return {
        "overview": f"{links} possible links were tested and a few stood out.",
        "positive_points": [{"text": f"When you spent more time {good['instructor_action']}, {good['student_behaviour']} tended to "
                                     f"{'rise' if good['change'] == 'rose' else 'fall'} about {good['after_seconds']} seconds later. "
                                     "These detections have not been verified by a person.",
                             "fact_ids": [good["id"]], "explanation_ids": []}],
        "negative_points": [{"text": f"In the {bad['after_seconds']} seconds after you were {bad['instructor_action']}, "
                                     f"{bad['student_behaviour']} tended to rise. This may be because the activity changed. "
                                     "The detections have not been verified.",
                             "fact_ids": [bad["id"]], "explanation_ids": ["E1"]}],
    }


class ValidateTests(unittest.TestCase):
    def test_careful_answer_passes(self):
        self.assertEqual(llm_text.validate(careful_answer(), FACTS), [])

    def test_each_kind_of_mistake_is_caught(self):
        answer = careful_answer()
        answer["negative_points"][0]["text"] = ("Long explanations caused students to fall asleep 77 times; the correlation "
                                                "was strong.")
        problems = " | ".join(llm_text.validate(answer, FACTS))
        for expected in ("causal wording", "internal states", "statistics jargon", "numbers not found", "unverified"):
            self.assertIn(expected, problems)

    def test_wrong_direction_and_unknown_ids(self):
        answer = careful_answer()
        answer["positive_points"][0]["fact_ids"] = [first("unfavourable")["id"], "F99"]
        problems = " | ".join(llm_text.validate(answer, FACTS))
        self.assertIn("not 'favourable' facts", problems)
        self.assertIn("unknown fact ids", problems)

    def test_weak_link_needs_a_hedge(self):
        weak = first("unfavourable", "weak")
        answer = careful_answer()
        answer["negative_points"] = [{"text": f"Students' behaviour changed {weak['after_seconds']} seconds after you were "
                                              f"{weak['instructor_action']}. The detections have not been verified.",
                                      "fact_ids": [weak["id"]], "explanation_ids": []}]
        self.assertIn("weak link", " | ".join(llm_text.validate(answer, FACTS)))

    def test_overview_numbers_must_come_from_the_analysis(self):
        answer = careful_answer()
        answer["overview"] = "42 links were tested."
        self.assertIn("overview: numbers not found", " | ".join(llm_text.validate(answer, FACTS)))


class FusionAnswerTests(unittest.TestCase):
    """Answers about the team's fusion layer output, including its event changes (peri-event averages)."""

    def setUp(self):
        rows, metadata = lag_facts.load_results(ROOT / "tests" / "fixtures" / "fusion_example.json")
        self.facts = lag_facts.build_facts(rows, VOCABULARY, metadata)
        self.event = next(f for f in self.facts["facts"] if f["kind"] == "event_change")

    def answer(self, text):
        return {"overview": f"{self.facts['analysis']['links_tested']} possible links were tested.",
                "positive_points": [{"text": text, "fact_ids": [self.event["id"]], "explanation_ids": []}],
                "negative_points": []}

    def test_event_change_answer_passes(self):
        e = self.event
        text = (f"After you {e['instructor_event']}, {e['student_behaviour']} rose by about {e['average_change']} "
                f"{e['unit']} over the next {e['after_seconds']} seconds, compared with the {e['before_seconds']} "
                f"seconds before. This happened {e['times']} times. These detections have not been verified by a person.")
        self.assertEqual(llm_text.validate(self.answer(text), self.facts), [])

    def test_event_change_answer_with_an_invented_number_fails(self):
        e = self.event
        text = (f"After you {e['instructor_event']}, {e['student_behaviour']} rose by about 12 percentage points. "
                "These detections have not been verified.")
        self.assertIn("numbers not found", " | ".join(llm_text.validate(self.answer(text), self.facts)))

    def test_prompt_explains_both_kinds(self):
        self.assertIn("event_change", llm_text.SYSTEM_PROMPT)
        self.assertNotIn("10-second", llm_text.SYSTEM_PROMPT)


class GenerateTests(unittest.TestCase):
    def test_rules_only_writes_report_and_keeps_facts_in_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            points = llm_text.generate(FACTS, directory, use_llm=False)
            self.assertEqual(points["source"], "rules")
            text = (Path(directory) / llm_text.TEXT_FILE).read_text(encoding="utf-8")
            self.assertIn("positive points:", text)
            self.assertIn("negative points:", text)
            self.assertFalse((Path(directory) / llm_text.FACTS_FILE).exists())
            llm_text.generate(FACTS, directory, use_llm=False, keep_facts=True)
            self.assertTrue((Path(directory) / llm_text.FACTS_FILE).exists())

    def test_retry_after_failed_checks_then_cache(self):
        calls = []

        def fake_call(system, user, model_alias):
            calls.append(user)
            if len(calls) == 1:
                bad = careful_answer()
                bad["overview"] = "The lecture caused problems."
                return bad, {"input_tokens": 10, "output_tokens": 5}, None
            return careful_answer(), {"input_tokens": 10, "output_tokens": 5}, None

        with tempfile.TemporaryDirectory() as directory:
            points = llm_text.generate(FACTS, directory, call=fake_call)
            self.assertEqual(points["source"], "llm")
            self.assertEqual(len(calls), 2)
            self.assertIn("failed these checks", calls[1])
            again = llm_text.generate(FACTS, directory, call=fake_call)
            self.assertEqual(len(calls), 2)                     # the cached answer was reused, no new call
            self.assertEqual(again["attempts"][0]["source"], "cache")

    def test_unreachable_api_falls_back_to_rules_without_retrying(self):
        calls = []

        def offline(system, user, model_alias):
            calls.append(1)
            return None, None, "Could not reach the Claude API."

        with tempfile.TemporaryDirectory() as directory:
            points = llm_text.generate(FACTS, directory, call=offline)
            self.assertEqual((points["source"], len(calls)), ("rules", 1))
            self.assertIn("Could not reach", points["attempts"][0]["problems"][0])

    def test_two_failed_answers_keep_the_rule_based_points(self):
        bad = careful_answer()
        bad["positive_points"][0]["text"] += " Students were sleeping."
        with tempfile.TemporaryDirectory() as directory:
            points = llm_text.generate(FACTS, directory, call=lambda s, u, m: (bad, {}, None))
            self.assertEqual(points["source"], "rules")
            self.assertEqual(len(points["attempts"]), llm_text.MAX_ATTEMPTS)

    def test_fake_response_and_saved_points(self):
        with tempfile.TemporaryDirectory() as directory:
            points = llm_text.generate(FACTS, directory, fake_response=careful_answer())
            saved = json.loads((Path(directory) / llm_text.POINTS_FILE).read_text(encoding="utf-8"))
            self.assertEqual(saved["source"], "llm")
            self.assertEqual(saved["positive_points"], points["positive_points"])
            self.assertEqual(saved["model"], "claude-opus-5-5")

    def test_cost_estimate_is_small(self):
        tokens, dollars = llm_text.estimate_cost(llm_text.build_user_message(FACTS), "opus")
        self.assertLess(dollars, 0.10)
        self.assertGreater(tokens, 0)


if __name__ == "__main__":
    unittest.main()
