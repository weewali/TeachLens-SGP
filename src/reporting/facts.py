"""facts.json: the fused lesson as pre-computed facts for the LLM report stage.

The LLM stage (pipeline_creation branch, `Downloaded from prototype/llm_report.py`,
2026-11-11) reads one facts.json (schema_version 1) and is the only thing the
model sees. Each fact has an id, a direction (favourable / unfavourable for
the instructor's feedback), a kind, plain-language fields with every number
the model may quote, and evidence.strength (suggestive / weak / descriptive).
llm_report.py then checks the model's answer in code: numbers must come from
the cited facts, no causal or internal-state words, hedges for weak evidence
and for detections nobody has checked (indicator_validation).

This module writes that schema from the fusion results, so a lesson folder's
facts.json can be passed straight to `llm_report.py --facts <path>`.
Wording follows the project's rule: observable proxies ("on task", "facing
the class"), never attention or engagement as fact.
"""

from typing import Any

import numpy as np

SCHEMA_VERSION = 1
STRETCH_WINDOWS = 6  # 30 s stretches for the highest / lowest on-task facts
# Plain wording for features (instructor activity / student measure).
TEACHER_WORDS = {
    "T_facing_class": "the instructor faced the class",
    "T_looking_at_students": "the instructor looked at the students",
    "T_board_or_screen": "the instructor worked at the board or screen",
    "T_writing": "the instructor wrote on the board",
    "T_pointing": "the instructor pointed at something",
    "T_pointing_at_students": "the instructor pointed toward the students",
    "T_back_to_class": "the instructor had their back to the class",
    "T_reading_notes": "the instructor looked down at notes or a laptop",
    "T_seated": "the instructor was seated",
    "T_at_desk": "the instructor was at the desk",
    "T_gesturing": "the instructor used hand gestures",
    "T_walking": "the instructor walked",
    "T_stationary": "the instructor stood still in one place",
    "T_path": "the instructor moved around the most",
}
STUDENT_WORDS = {
    "S_on_task": "students on task", "S_off_task": "students off task", "S_phone": "students using a phone",
    "S_on_target": "students facing the front", "S_off_target": "students facing away",
    "S_talking": "students talking", "S_head_down": "students with the head down",
    "S_sleep": "students dozing", "S_writing": "students writing", "S_hand_raises": "hand raises",
    "S_answers": "answers after a hand raise", "S_smiling": "students smiling",
    "S_looking_down": "students looking down", "S_reading": "students reading", "S_laptop": "students on a laptop",
    "S_moving": "students moving",
}
# Share of the lesson above which an instructor behaviour becomes a descriptive
# fact, and which way it points (the booklet's effective / ineffective codes).
TEACHER_PROFILE = (
    ("T_facing_class", 0.6, "favourable"), ("T_gesturing", 0.3, "favourable"),
    ("T_walking", 0.15, "favourable"), ("T_seated", 0.5, "unfavourable"),
    ("T_reading_notes", 0.3, "unfavourable"), ("T_back_to_class", 0.3, "unfavourable"),
)
VALIDATION_KEYS = ("instructor_role", "head_up_posture", "hand_raise", "phone_use", "instructor_seated",
                   "student_on_task_state", "instructor_facing", "instructor_gestures")
GLOSSARY = {
    "on task": "A student who faced the current focus (the front, or a classmate who was answering), wrote, or "
               "read, and was not off task. An observable proxy; it is not a measure of attention or learning.",
    "off task": "A student with a phone (seen in the hand, or the head held down for a long time), dozing, "
                "talking toward a neighbour, or facing away from the focus.",
    "students on task (percent)": "The share of the observed student-seconds in which students were on task.",
    "window": "A 5-second slice of the lesson; the two cameras are lined up by their recording clocks to about "
              "1 second.",
    "later": "Student measures are taken in the window after the instructor's behaviour (5 to 10 seconds later), "
             "unless a fact says otherwise.",
    "seated": "From the legs, or, behind the desk, from how low the shoulders are above the desk.",
    "evidence.strength": "suggestive = unlikely to be pure chance after allowing for neighbouring windows being "
                         "alike and for the number of tests; weak = the gap could easily be chance; "
                         "descriptive = a share, a count or a single stretch, not a test.",
    "lesson time": "mm:ss from the start of the stretch both cameras recorded.",
}
ALTERNATIVE_EXPLANATIONS = {
    "E1": "Students may have looked down to read a phone message or notes that the camera could not tell apart.",
    "E2": "The lesson may have been scripted or very short, so the class had little time to change.",
    "E3": "The kind of activity (explanation, seat-work, questions) was not recorded, and it changes how students "
          "sit and look on its own.",
    "E4": "The two cameras are lined up to about 1 second, so very quick reactions cannot be timed.",
    "E5": "A behaviour and a student change can both follow from the lesson plan rather than from each other.",
}
LIMITATIONS = [
    "Behaviour labels come from rules on body keypoints, face landmarks, hand gestures and object boxes, not from "
    "trained behaviour models, and no detection has been checked against human labels yet.",
    "Roles come from the camera: the instructor camera's single person is the instructor; everyone in the student "
    "camera is a student.",
    "Frames are analysed once per second (faces five times per second), so short actions can be missed.",
    "Associations within one lesson, not causes.",
]


def clock(seconds: float) -> str:
    """mm:ss (lesson time)."""
    seconds = max(0, int(round(seconds)))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def pct(value: float | None) -> int | None:
    return None if value is None or not np.isfinite(value) else int(round(100 * value))


def _strength(q: float | None, p: float | None, alpha: float) -> str:
    if q is not None and q <= alpha and p is not None and p < 0.05:
        return "suggestive"
    return "weak"


def _p_text(p: float | None) -> str | None:
    if p is None:
        return None
    return "< 0.01" if p < 0.01 else f"= {p:.2f}"


def _main_activity(row: Any) -> str | None:
    """The instructor behaviour that filled most of a window (or None)."""
    best, share = None, 0.0
    for name in TEACHER_WORDS:
        if name in row and name not in ("T_path", "T_stationary"):
            value = row[name]
            if value is not None and np.isfinite(value) and value > share:
                best, share = name, float(value)
    return TEACHER_WORDS[best] if best and share >= 0.5 else None


def comparison_facts(hypotheses: list[dict[str, Any]], features: dict[str, Any], alpha: float,
                     window_seconds: float, lessons: int | None = None) -> list[dict[str, Any]]:
    """One fact per tested hypothesis with a comparison: the student measure after
    the instructor behaviour versus the other windows."""
    facts = []
    for h in hypotheses:
        comp = h.get("comparison")
        if not comp or comp.get("p_hac") is None:
            continue
        gap = 100 * comp["difference"] if comp.get("difference_within_lessons") is None \
            else 100 * comp["difference_within_lessons"]
        if abs(gap) < 1:
            continue
        desirable = features[h["student_feature"]]["desirable"]
        counts = h["student_feature"] in ("S_hand_raises", "S_answers")
        fact = {
            "direction": "favourable" if desirable * gap > 0 else "unfavourable",
            "kind": "comparison",
            "instructor_activity": TEACHER_WORDS.get(h["teacher_feature"], h["teacher_feature"]),
            "student_measure": STUDENT_WORDS.get(h["student_feature"], h["student_feature"]),
            "measured": f"{int(comp['lag_windows'] * window_seconds)} to "
                        f"{int((comp['lag_windows'] + 1) * window_seconds)} seconds later",
            "windows": comp["windows_with"], "seconds": int(comp["windows_with"] * window_seconds),
            "other_windows": comp["windows_without"],
        }
        if counts:
            fact.update({"per_window_after": round(comp["outcome_with"], 2),
                         "per_window_other": round(comp["outcome_without"], 2)})
        else:
            fact.update({"pct_after": pct(comp["outcome_with"]), "pct_other_windows": pct(comp["outcome_without"]),
                         "gap_points": int(round(abs(gap)))})
        notes = ["The test allows for neighbouring windows being alike."]
        if lessons:
            fact["lessons"] = lessons
            notes.append("Compared within each lesson, over several lessons.")
        if h.get("matches_expectation") is False:
            notes.append("The gap points the other way from what the teaching guide expects.")
        fact["evidence"] = {"strength": _strength(h.get("q"), comp["p_hac"], alpha), "test_p": _p_text(comp["p_hac"]),
                            "notes": notes}
        facts.append(fact)
    order = {"suggestive": 0, "weak": 1}
    facts.sort(key=lambda f: (order[f["evidence"]["strength"]], -f.get("gap_points", 0)))
    return facts


def lesson_comparison_facts(hypotheses: list[dict[str, Any]], features: dict[str, Any]) -> list[dict[str, Any]]:
    """Facts comparing lessons: the student outcome in lessons where the instructor
    behaviour filled most of the time versus the other lessons."""
    facts = []
    for h in hypotheses:
        between = h.get("between_lessons") or {}
        if "outcome_with" not in between:
            continue
        gap = 100 * (between["outcome_with"] - between["outcome_without"])
        if abs(gap) < 1:
            continue
        desirable = features[h["student_feature"]]["desirable"]
        facts.append({
            "direction": "favourable" if desirable * gap > 0 else "unfavourable",
            "kind": "lesson_comparison",
            "instructor_activity": TEACHER_WORDS.get(h["teacher_feature"], h["teacher_feature"])
            + " for most of the lesson",
            "student_measure": STUDENT_WORDS.get(h["student_feature"], h["student_feature"]),
            "lessons_with": len(between["lessons_with"]), "lessons": between["lessons"],
            "pct_in_those_lessons": pct(between["outcome_with"]),
            "pct_in_other_lessons": pct(between["outcome_without"]),
            "gap_points": int(round(abs(gap))),
            "evidence": {"strength": "weak", "test_p": _p_text(between["p_permutation"]),
                         "notes": [f"Only {between['lessons']} lessons; lessons differ in many ways at once."]}})
    facts.sort(key=lambda f: -f["gap_points"])
    return facts


def lesson_facts(table: Any, report: dict[str, Any], *, lesson: str, videos: list[str],
                 student_summary: dict[str, Any], instructor_summary: dict[str, Any]) -> dict[str, Any]:
    """facts.json for one fused lesson (schema_version 1 of the LLM stage)."""
    window = float(report["window_seconds"])
    alpha = float(report["settings"]["alpha"])
    features = report["features"]
    facts = comparison_facts(report.get("hypotheses", []), features, alpha, window)
    n = len(table)
    for name, threshold, direction in TEACHER_PROFILE:
        if name not in table:
            continue
        values = table[name].to_numpy(float)
        known = values[~np.isnan(values)]
        if len(known) < max(3, n // 4):
            continue
        share = float(np.mean(known))
        if share >= threshold:
            facts.append({"direction": direction, "kind": "instructor_behaviour_share",
                          "instructor_activity": TEACHER_WORDS[name], "share_of_lesson_pct": pct(share),
                          "evidence": {"strength": "descriptive",
                                       "notes": ["A share of the lesson time where it could be judged; "
                                                 "the detection has not been checked against human labels."]}})
    if "S_on_task" in table and n >= STRETCH_WINDOWS:
        rolling = table["S_on_task"].rolling(STRETCH_WINDOWS, min_periods=STRETCH_WINDOWS - 1).mean()
        if rolling.notna().any():
            for kind, index, direction in (("highest_on_task_stretch", rolling.idxmax(), "favourable"),
                                           ("lowest_on_task_stretch", rolling.idxmin(), "unfavourable")):
                first = max(0, index - STRETCH_WINDOWS + 1)
                stretch = table.iloc[first:index + 1]
                activities = [a for a in (_main_activity(row) for _, row in stretch.iterrows()) if a]
                fact = {"direction": direction, "kind": kind,
                        "span": f"{clock(stretch['lesson_seconds'].iloc[0])}-"
                                f"{clock(stretch['lesson_seconds'].iloc[-1] + window)}",
                        "stretch_seconds": int(len(stretch) * window),
                        "students_on_task_pct": pct(rolling.iloc[index]),
                        "evidence": {"strength": "descriptive", "notes": ["A single 30-second stretch is noisy."]}}
                if activities:
                    fact["instructor_activity"] = max(set(activities), key=activities.count)
                facts.append(fact)
    raises = student_summary.get("student_hand_raises") or {}
    count = sum((raises.get("episodes_by_max_level") or {}).values())
    if "S_hand_raises" in table:
        times = [clock(t) for t, c in zip(table["lesson_seconds"], table["S_hand_raises"]) if c > 0][:3]
        if count:
            answered = (raises.get("outcomes") or {}).get("answered_likely", 0)
            facts.append({"direction": "favourable", "kind": "hand_raises", "count": count,
                          "answered_likely": answered, "example_times": times,
                          "evidence": {"strength": "descriptive",
                                       "notes": ["Hand raises have not been checked against human labels."]}})
        gaps, last = [], 0.0
        for t, c in zip(table["lesson_seconds"], table["S_hand_raises"]):
            if c > 0:
                gaps.append((t - last, last, t))
                last = t + window
        gaps.append((n * window - last, last, n * window))
        longest = max(gaps)
        if longest[0] >= 60:
            facts.append({"direction": "unfavourable", "kind": "no_participation_stretch",
                          "span": f"{clock(longest[1])}-{clock(longest[2])}",
                          "stretch_seconds": int(longest[0]), "missing": "student hand raise",
                          "evidence": {"strength": "descriptive",
                                       "notes": ["Hand raises the detector missed would make this stretch "
                                                 "look quieter than it was."]}})
    if "S_phone" in table:
        phone = table["S_phone"].to_numpy(float)
        known = phone[~np.isnan(phone)]
        if len(known) and np.mean(known) >= 0.05:
            facts.append({"direction": "unfavourable", "kind": "phone_use",
                          "students_with_phone_pct": pct(float(np.mean(known))),
                          "peak_window": clock(float(table["lesson_seconds"].iloc[int(np.nanargmax(phone))])),
                          "peak_pct": pct(float(np.nanmax(phone))),
                          "evidence": {"strength": "descriptive",
                                       "notes": ["Phones are small; a phone held under the desk is inferred "
                                                 "from a long head-down posture and has not been checked."]}})
    for number, fact in enumerate(facts, start=1):
        fact["id"] = f"F{number}"
    on_task_known = report.get("feature_activity_windows", {}).get("S_on_task")
    count_summary = student_summary.get("visible_student_count") or {}
    return {
        "schema_version": SCHEMA_VERSION,
        "lecture": {"lesson": lesson, "videos": videos, "duration": clock(n * window),
                    "start_real_time": report.get("start_real_time"), "sampling_seconds": 1,
                    "window_seconds": int(window), "windows": n},
        "data_quality": {
            "mode": "two fixed cameras (instructor and students); pose keypoints, face landmarks, hand gestures, "
                    "phone/laptop/book boxes",
            "instructor_visible_pct": pct(instructor_summary.get("teacher_visible_fraction_of_valid_samples")),
            "students_visible_median": count_summary.get("median"),
            "windows_with_student_state": on_task_known,
            "limitations": LIMITATIONS},
        "indicator_validation": {key: {"checked_against_human_labels": False} for key in VALIDATION_KEYS},
        "glossary": GLOSSARY,
        "alternative_explanations": ALTERNATIVE_EXPLANATIONS,
        "facts": [{"id": f["id"], **{k: v for k, v in f.items() if k != "id"}} for f in facts],
        "report_note": ("Note: \"on task\" means facing the focus, writing or reading, and \"off task\" means a phone, "
                        "dozing, chatting or facing away; both are observable proxies, not measures of attention or "
                        "learning. Times are mm:ss from the start of the stretch both cameras recorded. See "
                        "fusion/fusion.json for the statistics and lesson.json for every file."),
    }


def pooled_facts(pooled: dict[str, Any], features: dict[str, Any], window_seconds: float,
                 name: str) -> dict[str, Any]:
    """facts.json for several lessons analysed together (pooled hypotheses)."""
    alpha = float(pooled["settings"]["alpha"])
    lessons = len(pooled["lessons"])
    facts = comparison_facts(pooled["hypotheses"], features, alpha, window_seconds, lessons=lessons)
    facts += lesson_comparison_facts(pooled["hypotheses"], features)
    for number, fact in enumerate(facts, start=1):
        fact["id"] = f"F{number}"
    return {
        "schema_version": SCHEMA_VERSION,
        "lecture": {"lesson": name, "lessons": pooled["lessons"], "window_seconds": int(window_seconds)},
        "data_quality": {"mode": "several lessons, compared within each lesson", "limitations": LIMITATIONS},
        "indicator_validation": {key: {"checked_against_human_labels": False} for key in VALIDATION_KEYS},
        "glossary": GLOSSARY,
        "alternative_explanations": ALTERNATIVE_EXPLANATIONS,
        "facts": [{"id": f["id"], **{k: v for k, v in f.items() if k != "id"}} for f in facts],
        "report_note": "Note: several lessons were compared within each lesson; see pooled.json for the statistics.",
    }
