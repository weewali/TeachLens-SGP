"""The instructor report stage on a lesson folder: comparisons, facts, rule-based points, the checked AI rewrite with
its fallback, and the human review sheets. Made-up data only (tests/fixtures/demo_lesson.py); no network."""
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "fixtures"))

from demo_lesson import make_demo_lesson, windows_table  # noqa: E402
from src.analysis import activity_comparison  # noqa: E402
from src.reporting import feedback, llm_report, review_sheets  # noqa: E402


def response(answer):
    """What call_claude returns for a successful request."""
    usage = SimpleNamespace(input_tokens=2000, output_tokens=500)
    return answer, SimpleNamespace(usage=usage, model="claude-opus-5-5"), None


def good_answer(facts):
    favourable = next(f for f in facts["facts"] if f["kind"] == "hand_raises")
    return {"positive_points": [{"text": f"Students raised their hands {favourable['count']} times, for example at "
                                         "00:05; these hand raises have not been verified by a person.",
                                 "fact_ids": [favourable["id"]], "explanation_ids": []}],
            "negative_points": []}


class ComparisonTests(unittest.TestCase):
    def test_the_planted_difference_is_found_and_counts_as_a_test(self):
        rows = activity_comparison.compare(windows_table())
        facing = {r["measure"]: r for r in rows if r["teacher_feature"] == "T_facing_class"}
        on_task = facing["S_on_task"]
        self.assertGreater(on_task["with"], on_task["without"] + 0.2)
        self.assertTrue(on_task["favourable"] and on_task["suggestive"])
        self.assertEqual((on_task["windows_with"], on_task["windows_without"]), (30, 30))
        self.assertLess(facing["S_phone"]["with"], facing["S_phone"]["without"])
        self.assertTrue(facing["S_phone"]["favourable"])            # less phone use is the good direction
        self.assertTrue(all(r["tests"] == sum(x["p_hac"] is not None for x in rows) for r in rows))
        board = [r for r in rows if r["teacher_feature"] == "T_board_or_screen"]
        self.assertEqual(board, [])          # only 3 board windows: fewer than MIN_WINDOWS, not compared

    def test_activity_of_a_window(self):
        window = activity_comparison.activity_of
        self.assertEqual(window({"teacher_known_seconds": 1}, 5), "out_of_view")
        self.assertEqual(window({"teacher_known_seconds": 5, "T_board_or_screen": 0.4, "T_facing_class": 1}, 5), "board")
        self.assertEqual(window({"teacher_known_seconds": 5, "T_facing_class": 0.7}, 5), "explaining")
        self.assertEqual(window({"teacher_known_seconds": 5, "T_facing_class": 0.2}, 5), "not_facing")

    def test_mean_range(self):
        self.assertIsNone(activity_comparison.mean_range([0.5, 0.6]))
        self.assertEqual(activity_comparison.mean_range([0.4] * 6), (0.4, 0.4))
        low, high = activity_comparison.mean_range([0.5, 0.7, 0.6, 0.8, 0.55, 0.65, 0.75, 0.6])
        self.assertLess(low, 0.65)
        self.assertGreater(high, 0.65)


class FactsAndRulesTests(unittest.TestCase):
    def test_activity_facts_are_added_once_and_renumbered(self):
        rows = activity_comparison.compare(windows_table())
        added = feedback.activity_facts(rows, 5.0)
        self.assertTrue(added and len(added) <= feedback.MAX_ACTIVITY_FACTS)
        first = added[0]
        self.assertEqual((first["kind"], first["evidence"]["strength"]), ("activity_comparison", "suggestive"))
        self.assertEqual(first["instructor_activity"], "the instructor faced the class")
        base = {"facts": [{"id": "F1", "kind": "hand_raises", "direction": "favourable"}]}
        once = feedback.add_activity_facts(base, added)
        twice = feedback.add_activity_facts(once, added)          # a second run replaces, never duplicates
        self.assertEqual(once, twice)
        self.assertEqual([f["id"] for f in twice["facts"]], [f"F{n}" for n in range(1, len(added) + 2)])

    def test_the_report_step_writes_no_rule_based_text(self):
        self.assertFalse(hasattr(feedback, "rule_points"))      # Claude writes the feedback (user decision 2026-11-20)


class AiRewriteTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.lesson = make_demo_lesson(self.temp.name, claude_ok=False)
        self.facts = json.loads((self.lesson / "facts.json").read_text(encoding="utf-8"))

    def tearDown(self):
        self.temp.cleanup()

    def test_the_checks_accept_a_faithful_answer_and_reject_a_bad_one(self):
        self.assertEqual(llm_report.validate(good_answer(self.facts), self.facts), [])
        phone = next(f for f in self.facts["facts"] if f["kind"] == "phone_use")
        bad = {"positive_points": [{"text": "Students were distracted by phones 99% of the time, which caused bored "
                                            "faces.", "fact_ids": [phone["id"]], "explanation_ids": []}],
               "negative_points": []}
        problems = " | ".join(llm_report.validate(bad, self.facts))
        for expected in ("not 'favourable' facts", "numbers not found", "causal wording", "internal states",
                         "does not say so"):
            self.assertIn(expected, problems)

    def test_without_a_key_there_is_no_feedback_only_the_reason(self):
        status = json.loads((self.lesson / "report_status.json").read_text(encoding="utf-8"))
        self.assertEqual((status["source"], status["code"]), ("none", "01"))   # the web app shows only "Code=01"
        self.assertIn("No Claude API key", status["reason"])                    # the reason stays in the file
        self.assertFalse((self.lesson / "final_feedback_report.txt").exists())
        self.assertFalse((self.lesson / "final_feedback_report_rules.txt").exists())
        for table in ("events.csv", "activity_comparisons.csv"):
            self.assertTrue((self.lesson / "tables" / table).exists())

    def test_a_checked_answer_is_used_and_cached(self):
        calls = []

        def claude(system, user, model):
            calls.append(model)
            return response(good_answer(self.facts))
        status = feedback.report_lesson(self.lesson, caller=claude, log=lambda _: None)
        self.assertEqual((status["source"], status["reason"], len(calls)), ("llm", None, 1))
        self.assertIn("have not been verified", (self.lesson / "final_feedback_report.txt").read_text(encoding="utf-8"))
        record = json.loads((self.lesson / "llm_output.json").read_text(encoding="utf-8"))
        self.assertEqual(record["result"]["source"], "llm")

        def must_not_call(*_args):
            raise AssertionError("the cached answer should have been used")
        again = feedback.report_lesson(self.lesson, caller=must_not_call, log=lambda _: None)
        self.assertEqual(again["source"], "llm")

    def test_an_answer_failing_twice_falls_back_with_the_reason(self):
        def claude(system, user, model):
            return response({"positive_points": [{"text": "Students were bored.", "fact_ids": ["F99"],
                                                  "explanation_ids": []}], "negative_points": []})
        (self.lesson / "final_feedback_report.txt").write_text("an earlier run's text", encoding="utf-8")
        status = feedback.report_lesson(self.lesson, caller=claude, log=lambda _: None)
        self.assertEqual((status["source"], status["code"]), ("none", None))    # not a key problem: no code
        self.assertIn("failed the automatic checks 2 time(s)", status["reason"])
        self.assertFalse((self.lesson / "final_feedback_report.txt").exists())   # never an old text as if new
        record = json.loads((self.lesson / "llm_output.json").read_text(encoding="utf-8"))
        self.assertEqual(len(record["attempts"]), 2)

    def test_only_api_key_problems_get_code_01(self):
        def rejected(*_args):
            raise llm_report.RewriteUnavailable("Claude rejected the API key.", llm_report.KEY_PROBLEM)

        def offline(*_args):
            raise llm_report.RewriteUnavailable("Could not reach the Claude API (no internet connection?).")
        lines = []
        status = feedback.report_lesson(self.lesson, caller=rejected, log=lines.append)
        self.assertEqual((status["code"], status["reason"]), ("01", "Claude rejected the API key."))
        self.assertIn("Error, Code=01", lines)                    # the log is shown on the progress page
        self.assertFalse(any("API key" in line for line in lines))
        status = feedback.report_lesson(self.lesson, caller=offline, log=lambda _: None)
        self.assertEqual(status["code"], None)
        self.assertIn("internet", status["reason"])

    def test_error_statuses_are_sorted_into_codes(self):
        def error(status, message="", kind=None, name="APIStatusError"):
            return type(name, (Exception,), {})(message) if status is None else \
                type(name, (Exception,), {"status_code": status, "type": kind, "message": message})(message)
        code = lambda e: llm_report.status_problem(e).code   # noqa: E731
        self.assertEqual(code(error(401, "invalid x-api-key", "authentication_error")), "01")
        self.assertEqual(code(error(403, "not allowed", "permission_error")), "01")
        self.assertEqual(code(error(400, "Your credit balance is too low to access the Anthropic API.",
                                    "invalid_request_error")), "02")
        self.assertEqual(code(error(402, "payment required", "billing_error")), "02")
        self.assertEqual(code(error(429, "limit", "rate_limit_error", "EnforcedSpendLimitReachedError")), "02")
        self.assertEqual(code(error(400, "on hold", "invalid_request_error", "OrganizationOnHoldError")), "02")
        for plain in (error(429, "slow down", "rate_limit_error"), error(400, "bad schema", "invalid_request_error"),
                      error(529, "overloaded", "overloaded_error")):
            self.assertIsNone(code(plain))                      # other problems keep their plain reason
        self.assertIn("rate-limiting", str(llm_report.status_problem(error(429, "slow down", "rate_limit_error"))))


class ReviewSheetTests(unittest.TestCase):
    def test_sheets_are_made_scored_and_mark_the_facts_as_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            lesson = make_demo_lesson(directory)
            rows = review_sheets.sheet_rows(lesson, samples=300)
            kinds = {(r["indicator"], r["positive"]) for r in rows}
            self.assertIn(("device_use", "yes"), kinds)
            self.assertIn(("hand_raise", "yes"), kinds)
            folder = Path(directory) / "validation" / "demo_lecture"
            sheet = review_sheets.make_sheets(lesson, folder, samples=300)   # no review video: rows without images
            with open(sheet, newline="", encoding="utf-8") as f:
                filled = list(csv.DictReader(f))
            for row in filled:   # a reviewer: phones right 3 times in 4, everything else right
                row["human"] = "no" if row["indicator"] == "device_use" and int(row["item"]) % 4 == 0 else "yes"
            with open(sheet, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=review_sheets.SHEET_COLUMNS)
                writer.writeheader()
                writer.writerows(filled)
            with self.assertRaises(FileExistsError):                 # answers are never overwritten
                review_sheets.make_sheets(lesson, folder)
            results = review_sheets.score(sheet)
            self.assertEqual(results["hand_raise"]["precision"], 1.0)
            self.assertLess(results["device_use"]["agreement"], 1.0)
            facts = review_sheets.update_facts(lesson / "facts.json", results)
            self.assertTrue(facts["indicator_validation"]["phone_use"]["checked_against_human_labels"])
            phone = next(f for f in facts["facts"] if f["kind"] == "phone_use")
            precision = facts["indicator_validation"]["phone_use"]["precision"]
            self.assertEqual(llm_report.unverified(phone, facts), ["phone_use"] if precision < 0.7 else [])
        self.assertEqual(review_sheets.wilson(0, 0), None)
        low, high = review_sheets.wilson(9, 10)
        self.assertTrue(0.5 < low < 0.9 < high <= 1.0)


if __name__ == "__main__":
    unittest.main()
