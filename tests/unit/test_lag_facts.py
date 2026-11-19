"""Lagged-analysis results -> plain-language facts: direction, evidence strength, vocabulary, rule-based points."""

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from src.reporting import lag_facts, llm_text

ROOT = Path(__file__).resolve().parents[2]
VOCABULARY = lag_facts.load_vocabulary(ROOT / "configs" / "report_vocabulary.yaml")
EXAMPLE = ROOT / "tests" / "fixtures" / "lag_results_example.json"
FUSION_EXAMPLE = ROOT / "tests" / "fixtures" / "fusion_example.json"


def load_script(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def row(action, behaviour, lag, effect, p, prespecified=False, windows=100):
    return {"instructor_action": action, "student_behaviour": behaviour, "lag_seconds": lag, "effect": effect,
            "p_value": p, "n_windows": windows, "prespecified": prespecified}


class NormalizeTests(unittest.TestCase):
    def test_older_pipeline_column_names_are_accepted(self):
        old = {"teacher_feature": "explanation_ratio", "student_outcome": "phone_rate", "lag_seconds": "20",
               "correlation": "0.3", "p_value": "0.2", "p_controlled": "0.01", "n_windows": "50", "prespecified": "True"}
        normalized = lag_facts.normalize_row(old)
        self.assertEqual((normalized["instructor_action"], normalized["student_behaviour"]), ("explanation_ratio", "phone_rate"))
        self.assertEqual(normalized["p_value"], 0.01)   # the controlled p-value wins
        self.assertTrue(normalized["prespecified"])
        self.assertEqual(normalized["effect_type"], "correlation")

    def test_missing_fields_are_named(self):
        with self.assertRaisesRegex(ValueError, "student_behaviour"):
            lag_facts.normalize_row({"instructor_action": "x", "lag_seconds": 10, "effect": 0.1})
        with self.assertRaisesRegex(ValueError, "must be numbers"):
            lag_facts.normalize_row({"instructor_action": "x", "student_behaviour": "y", "lag_seconds": "soon", "effect": 0.1})
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            lag_facts.normalize_row(row("x", "y", 10, 0.1, 3.0))

    def test_csv_input(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lags.csv"
            path.write_text("teacher_feature,student_outcome,lag_seconds,correlation,p_controlled,n_windows,prespecified\n"
                            "movement_amount,attention_level,10,0.4,0.0001,90,False\n", encoding="utf-8")
            rows, metadata = lag_facts.load_results(path)
            facts = lag_facts.build_facts(rows, VOCABULARY, metadata)
            self.assertEqual(len(facts["facts"]), 1)
            self.assertEqual(facts["facts"][0]["direction"], "favourable")


class BuildFactsTests(unittest.TestCase):
    def test_direction_follows_the_vocabulary(self):
        facts = lag_facts.build_facts([
            row("explanation_ratio", "phone_rate", 20, -0.4, 0.001),       # fewer phones: good news
            row("movement_amount", "attention_level", 10, -0.4, 0.001),    # fewer heads up: bad news
        ], VOCABULARY)["facts"]
        by_behaviour = {f["indicators"][1]: f for f in facts}
        self.assertEqual(by_behaviour["phone_rate"]["direction"], "favourable")
        self.assertEqual(by_behaviour["phone_rate"]["change"], "fell")
        self.assertEqual(by_behaviour["attention_level"]["direction"], "unfavourable")

    def test_every_row_counts_towards_the_correction(self):
        rows = [row("movement_amount", "attention_level", 10, 0.3, 0.012)] + \
               [row("teacher_gesture", "phone_rate", lag, 0.0, 0.9) for lag in (10, 20, 30, 40)]
        facts = lag_facts.build_facts(rows, VOCABULARY)
        self.assertEqual(facts["analysis"]["links_tested"], 5)
        self.assertEqual(facts["facts"][0]["evidence"]["strength"], "weak")   # 0.012 > 0.05 / 5
        alone = lag_facts.build_facts(rows[:1], VOCABULARY)
        self.assertEqual(alone["facts"][0]["evidence"]["strength"], "suggestive")

    def test_prespecified_rows_are_their_own_family(self):
        rows = [row("explanation_ratio", "phone_rate", 20, 0.3, 0.004, prespecified=True)] + \
               [row("teacher_gesture", "phone_rate", lag, 0.0, 0.9) for lag in range(10, 200, 10)]
        fact = lag_facts.build_facts(rows, VOCABULARY)["facts"][0]
        self.assertEqual(fact["evidence"]["strength"], "suggestive")         # judged against 1 test, not 20
        self.assertTrue(fact["prespecified"])

    def test_one_fact_per_pair_at_its_strongest_lag(self):
        rows = [row("explanation_ratio", "phone_rate", lag, effect, p) for lag, effect, p in
                ((10, 0.1, 0.04), (20, 0.3, 0.001), (30, 0.2, 0.03))]
        facts = lag_facts.build_facts(rows, VOCABULARY)["facts"]
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0]["after_seconds"], 20)

    def test_chance_links_and_unknown_indicators_are_left_out(self):
        facts = lag_facts.build_facts([
            row("explanation_ratio", "phone_rate", 10, 0.05, 0.6),
            row("teacher_voice_volume", "attention_level", 10, 0.5, 0.0001),
        ], VOCABULARY)
        self.assertEqual(facts["facts"], [])
        self.assertEqual(facts["not_interpreted"], ["teacher_voice_volume"])

    def test_weak_links_are_capped(self):
        rows = [row(action, "phone_rate", 10, 0.2, 0.04) for action in VOCABULARY["instructor_actions"]]
        facts = lag_facts.build_facts(rows, VOCABULARY, max_weak=3)
        self.assertEqual(len(facts["facts"]), 3)
        self.assertEqual(facts["analysis"]["weak_links_found"], len(rows))

    def test_example_file(self):
        rows, metadata = lag_facts.load_results(EXAMPLE)
        facts = lag_facts.build_facts(rows, VOCABULARY, metadata)
        strengths = [f["evidence"]["strength"] for f in facts["facts"]]
        self.assertEqual(strengths[:3], ["suggestive"] * 3)                  # strongest first
        self.assertEqual(facts["not_interpreted"], ["teacher_voice_volume"])
        self.assertEqual(facts["analysis"]["prespecified_tests"], 9)
        self.assertTrue(all(not v["checked_against_human_labels"] for v in facts["indicator_validation"].values()))
        json.dumps(facts)                                                    # serialisable as is

    def test_validation_results_are_carried_over(self):
        metadata = {"indicator_validation": {"phone_rate": {"checked_against_human_labels": True, "precision": 0.9}}}
        facts = lag_facts.build_facts([row("explanation_ratio", "phone_rate", 20, 0.3, 0.001)], VOCABULARY, metadata)
        self.assertEqual(lag_facts.unverified_indicators(facts["facts"][0], facts), ["explanation_ratio"])


class RuleBasedTests(unittest.TestCase):
    def test_points_follow_direction_and_say_how_sure(self):
        rows, metadata = lag_facts.load_results(EXAMPLE)
        facts = lag_facts.build_facts(rows, VOCABULARY, metadata)
        overview, positive, negative = lag_facts.rule_based_points(facts)
        self.assertIn("15 possible links", overview)
        self.assertTrue(positive and negative)
        self.assertTrue(all(len(points) <= 3 for points in (positive, negative)))
        self.assertIn("unlikely to be chance", positive[0]["text"])
        self.assertIn("not proof", lag_facts.report_note(facts))

    def test_nothing_found(self):
        facts = lag_facts.build_facts([row("explanation_ratio", "phone_rate", 10, 0.01, 0.9)], VOCABULARY)
        overview, positive, negative = lag_facts.rule_based_points(facts)
        self.assertIn("none stood out", overview)
        self.assertEqual((positive, negative), ([], []))


class FusionInputTests(unittest.TestCase):
    """The team's fusion layer output (scripts/fuse_lesson.py, src/analysis/fusion.py + lag_analysis.py)."""

    def test_fusion_report_is_read(self):
        rows, metadata = lag_facts.load_results(FUSION_EXAMPLE)
        doc = json.loads(FUSION_EXAMPLE.read_text(encoding="utf-8"))
        tested_events = [e for e in doc["peri_event"] if "p_random_onsets" in e]
        self.assertEqual(len(rows), len(doc["links"]) + len(tested_events))
        self.assertEqual(metadata["lecture"], {"windows": 150, "window_seconds": 5, "duration_seconds": 750})
        self.assertEqual(metadata["lags_tested_seconds"], [5, 10, 15, 20, 25, 30])
        facts = lag_facts.build_facts(rows, VOCABULARY, metadata)
        self.assertEqual(facts["not_interpreted"], [])
        self.assertEqual(facts["analysis"]["too_rare_to_test"],
                         ["at the board or screen (seen in 1 window)", "went to the board or screen (1 time)"])
        self.assertIn("5-second", facts["glossary"]["link"])
        json.dumps(facts)

    def test_report_agrees_with_the_fusion_findings_on_fresh_team_code(self):
        # A fresh run of the team's analysis (not the saved fixture), so a change in their output shows up here.
        maker = load_script(ROOT / "tests" / "fixtures" / "make_fusion_example.py", "make_fusion_example")
        table, onsets = maker.example_table()
        table = table[["window", "lesson_seconds", *maker.USED]]
        settings = dict(maker.fusion_settings(), permutations=200)
        doc = maker.fusion_report(table, {k: onsets[k] for k in maker.USED_EVENTS}, settings,
                                  maker.feature_metadata(maker.USED))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lesson.fusion.json"
            path.write_text(json.dumps(doc), encoding="utf-8")
            rows, metadata = lag_facts.load_results(path)
        facts = lag_facts.build_facts(rows, VOCABULARY, metadata)
        tone = {"strength": "favourable", "weak_point": "unfavourable"}
        expected = {(f["teacher_feature"], f["student_feature"], tone[f["kind"]])
                    for f in doc["findings"] if f["kind"] in tone}
        reported = {(*f["indicators"], f["direction"]) for f in facts["facts"]
                    if f["evidence"]["strength"] == "suggestive"}
        self.assertEqual(reported, expected)
        self.assertIn(("T_stationary", "S_phone", "unfavourable"), reported)       # planted
        self.assertIn(("T_gesturing", "S_off_target", "favourable"), reported)     # planted
        self.assertEqual(facts["analysis"]["neutral_links_found"],
                         sum(1 for f in doc["findings"] if f["kind"] == "observation"))
        gesturing = next(f for f in facts["facts"] if f["indicators"] == ["T_gesturing", "S_off_target"])
        self.assertEqual((gesturing["after_seconds"], gesturing["change"]), (10, "fell"))

    def test_links_csv_is_accepted(self):
        doc = json.loads(FUSION_EXAMPLE.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lesson.fusion_links.csv"
            with open(path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(doc["links"][0]))
                writer.writeheader()
                writer.writerows(doc["links"])
            rows, metadata = lag_facts.load_results(path)
        facts = lag_facts.build_facts(rows, VOCABULARY, metadata)
        self.assertEqual(facts["analysis"]["links_tested"], len(doc["links"]))
        self.assertTrue(facts["facts"])
        self.assertTrue(all(f["kind"] == "lagged_link" for f in facts["facts"]))

    def test_q_value_decides_how_sure(self):
        self.assertEqual(lag_facts.judge(0.0072, 112, q_value=0.04), "suggestive")
        self.assertEqual(lag_facts.judge(0.0072, 1, q_value=0.29), "weak")         # q decides, not Bonferroni
        self.assertIsNone(lag_facts.judge(0.08, 1, q_value=0.3))

    def test_event_change_sentence_and_notes(self):
        rows = [{"kind": "event_change", "teacher_event": "turn_to_class", "student_feature": "S_hand_raises",
                 "lag_seconds": 20, "before_seconds": 10, "effect": -0.4, "p_value": 0.002, "q_value": 0.02, "events": 3}]
        metadata = {"lecture": {"window_seconds": 5}, "feature_kinds": {"S_hand_raises": "count"}}
        fact = lag_facts.build_facts(rows, VOCABULARY, metadata)["facts"][0]
        self.assertEqual((fact["average_change"], fact["unit"], fact["direction"]), (0.4, "per 5-second window", "unfavourable"))
        self.assertIn("Based on only 3 times", " ".join(fact["evidence"]["notes"]))
        self.assertEqual(lag_facts.rule_sentence(fact, 5),
                         "After the instructor turned to face the class (3 times), the number of hand raises fell by 0.4 "
                         "per 5-second window on average over the next 20 seconds, compared with the 10 seconds before "
                         "(unlikely to be chance).")

    def test_a_link_as_strong_the_other_way_round_is_flagged(self):
        rows = [{"teacher_feature": "T_gesturing", "student_feature": "S_phone", "lag_seconds": 10, "r": 0.4,
                 "r_reverse_max": -0.45, "p_shift": 0.002, "q": 0.01}]
        fact = lag_facts.build_facts(rows, VOCABULARY)["facts"][0]
        self.assertIn("other way round", " ".join(fact["evidence"]["notes"]))

    def test_neutral_behaviours_are_counted_not_listed(self):
        rows = [{"teacher_feature": "T_facing_class", "student_feature": "S_talking", "lag_seconds": 5, "r": 0.8,
                 "p_shift": 0.002, "q": 0.01}]
        facts = lag_facts.build_facts(rows, VOCABULARY)
        self.assertEqual((facts["facts"], facts["analysis"]["neutral_links_found"]), ([], 1))
        overview, positive, negative = lag_facts.rule_based_points(facts)
        self.assertIn("none about good or bad news stood out", overview)
        self.assertIn("so it is not listed", overview)

    def test_vocabulary_covers_the_fusion_layer_with_the_same_directions(self):
        from src.analysis.fusion import FEATURES
        fuse_lesson = load_script(ROOT / "scripts" / "fuse_lesson.py", "fuse_lesson_for_vocabulary")
        for feature in FEATURES:
            if feature.camera == "teacher":
                self.assertIn(feature.name, VOCABULARY["instructor_actions"])
            else:
                self.assertIn(feature.name, VOCABULARY["student_behaviours"])
                self.assertIs(VOCABULARY["student_behaviours"][feature.name]["favourable_when_higher"],
                              lag_facts.DESIRABLE[feature.desirable], feature.name)
        for name in fuse_lesson.TEACHER_EVENTS:
            self.assertIn(name, VOCABULARY["instructor_events"])

    def test_disagreeing_directions_are_refused(self):
        rows = [{"teacher_feature": "T_gesturing", "student_feature": "S_phone", "lag_seconds": 10, "r": -0.4,
                 "p_shift": 0.002, "q": 0.01}]
        with self.assertRaisesRegex(ValueError, "S_phone"):
            lag_facts.build_facts(rows, VOCABULARY, {"desirable": {"S_phone": 1}})   # fusion says more phones is good

    def test_vocabulary_phrases_pass_the_report_word_checks(self):
        # the model is told to copy these words, so they must survive the checks on its answer, and carry no number
        # (every number it writes must come from the facts' values)
        for source in (None, *VOCABULARY["measured_by"]):
            vocabulary = lag_facts.vocabulary_for(VOCABULARY, source)
            for section in lag_facts.SECTIONS:
                for name, entry in vocabulary[section].items():
                    phrase = entry["phrase"]
                    self.assertEqual(llm_text._word_checks(name, phrase, set()), [], (source, name))


class OtherPipelineWordingTests(unittest.TestCase):
    """An analysis measured by main code/Main Code.py, whose rules are narrower than the team's, gets its own words."""

    def test_wording_follows_the_pipeline_that_measured_it(self):
        doc = json.loads(FUSION_EXAMPLE.read_text(encoding="utf-8"))
        rows, metadata = lag_facts.from_fusion(dict(doc, measured_by="main_code"))
        phrases = {f["indicators"][1]: f["student_behaviour"] for f in lag_facts.build_facts(rows, VOCABULARY, metadata)["facts"]}
        team_rows, team_metadata = lag_facts.from_fusion(doc)
        team = {f["indicators"][1]: f["student_behaviour"]
                for f in lag_facts.build_facts(team_rows, VOCABULARY, team_metadata)["facts"]}
        self.assertEqual(phrases["S_phone"], "the share of students with a phone seen near them")
        self.assertIn("in hand", team["S_phone"])
        self.assertEqual(set(phrases), set(team))                       # same facts, same directions: only words differ
        main_code = lag_facts.vocabulary_for(VOCABULARY, "main_code")["student_behaviours"]
        for name, entry in VOCABULARY["student_behaviours"].items():
            self.assertIs(main_code[name]["favourable_when_higher"], entry["favourable_when_higher"], name)
        self.assertIs(lag_facts.vocabulary_for(VOCABULARY, "unknown_pipeline"), VOCABULARY)

    def test_other_wording_may_not_change_good_or_bad_news(self):
        base = (ROOT / "configs" / "report_vocabulary.yaml").read_text(encoding="utf-8")
        cases = {"only give a phrase": "measured_by:\n  x:\n    student_behaviours:\n      S_phone:\n        phrase: phones\n"
                                "        favourable_when_higher: true\n",
                 "to reword": "measured_by:\n  x:\n    student_behaviours:\n      S_unknown:\n        phrase: something\n"}
        for expected, extra in cases.items():
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "vocabulary.yaml"
                path.write_text(base.split("\nmeasured_by:")[0] + "\n" + extra, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, expected):
                    lag_facts.load_vocabulary(path)

    def test_rule_based_points_can_speak_to_the_instructor(self):
        rows = [{"kind": "event_change", "teacher_event": "turn_to_class", "student_feature": "S_hand_raises",
                 "lag_seconds": 20, "before_seconds": 10, "effect": 0.4, "p_value": 0.002, "q_value": 0.02, "events": 6}]
        facts = lag_facts.build_facts(rows, VOCABULARY, {"lecture": {"window_seconds": 5}})
        overview, positive, _ = lag_facts.rule_based_points(facts, subject="you")
        self.assertIn("between what you did", overview)
        self.assertTrue(positive[0]["text"].startswith("After you turned to face the class (6 times)"))
        self.assertIn("between what you did", lag_facts.report_note(facts, "you"))


if __name__ == "__main__":
    unittest.main()
