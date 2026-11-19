"""Turns the fusion layer's lagged-analysis results into plain-language facts for the instructor's report.

Nothing here calls a model. The direction of each link (good or bad news for the instructor), how much weight it
deserves, and the words used for each indicator are all decided in code, so the language model only has to explain
facts that are already correct.

Input (what `load_results` reads):

- The team's fusion layer output, `outputs/reports/<name>.fusion.json` from `scripts/fuse_lesson.py`
  (docs/fusion_lag_analysis.md, 2026-11-18), read by `from_fusion`:
  * each entry of "links" (a teacher feature -> a student feature at its strongest lag: r, p_shift, q) becomes a
    "lagged_link" row;
  * each tested entry of "peri_event" (the average change of a student feature after a teacher action started, versus
    before: mean_change, p_random_onsets, q) becomes an "event_change" row;
  * teacher actions too rare to test, the window length, the lags tried, alpha, each feature's kind and "desirable"
    direction, and the fusion's fixed limitations are kept as metadata.
- Its `<name>.fusion_links.csv` (links only, no lecture details).
- The older provisional formats: a JSON document {"results": [...]}, a JSON list of rows, or a CSV file, with the
  fields below (`COLUMN_ALIASES` lists the accepted names).

    instructor_action   indicator ID of what the instructor did (an event ID for event_change rows)
    student_behaviour   indicator ID of what the students did
    lag_seconds         how long after the instructor action the behaviour was measured (event_change: the span after)
    effect              signed size: positive = the behaviour rose after more of the action / after the action started
    effect_type         "correlation" (size words are added) or anything else (direction only)
    p_value, q_value    the test's p-value and, when the fusion layer gives one, its q-value (Benjamini-Hochberg)
    n_windows           how many windows the test used
    prespecified        true for hypotheses fixed before the data were seen (optional, default false)

How much weight a link gets: with a q-value, "suggestive" means q <= alpha, which is exactly the fusion layer's own
"findings" rule, and "weak" means p < alpha but q > alpha. Without q-values every row counts towards a Bonferroni
limit for its family (pre-registered rows are their own family).
"""
import csv
import hashlib
import json
from pathlib import Path

import yaml

ALPHA = 0.05
MAX_WEAK_FACTS = 6                 # weak links are listed, but only the few strongest, so they cannot crowd the report
FEW_TIMES = 5                      # an event_change resting on fewer starts than this gets a note saying so
STRENGTH_RANK = {"suggestive": 0, "weak": 1, "unknown": 2}
KINDS = ("lagged_link", "event_change")
# accepted names for each field, first match wins (p_controlled is preferred over a plain p_value)
COLUMN_ALIASES = {
    "kind": ("kind",),
    "instructor_action": ("instructor_action", "instructor_indicator", "teacher_feature", "teacher_event"),
    "student_behaviour": ("student_behaviour", "student_behavior", "student_indicator", "student_outcome",
                          "student_feature"),
    "lag_seconds": ("lag_seconds",),
    "before_seconds": ("before_seconds",),
    "effect": ("effect", "correlation", "coefficient", "coef_controlled", "r"),
    "reverse_effect": ("reverse_effect", "r_reverse_max"),
    "effect_type": ("effect_type",),
    "p_value": ("p_controlled", "p_value", "p", "p_shift"),
    "q_value": ("q_value", "q"),
    "n_windows": ("n_windows", "windows"),
    "events": ("events",),
    "prespecified": ("prespecified",),
}
# the fusion layer's "desirable" (src/analysis/fusion.py Feature) -> the vocabulary's favourable_when_higher
DESIRABLE = {1: True, -1: False, 0: None}
ALTERNATIVE_EXPLANATIONS = {
    "E1": "The type of activity (lecture, seat-work, discussion) was not recorded, and it changes student posture on its own.",
    "E2": "Many links were tested, so a few can appear by chance.",
    "E3": "The link may run the other way: the instructor may have changed what they were doing in response to the students.",
    "E4": "The two cameras are lined up to about 1 second, so timings are approximate.",
    "E5": "The detections have not been checked against a person's labels, so some may be wrong.",
    "E6": "The lecture was short or the action was rare, so a few moments can decide the result.",
}


def _plain(number):
    """5.0 -> 5, so a number reads (and is checked) the way a person would write it."""
    return int(number) if isinstance(number, float) and number.is_integer() else number


def glossary(window_seconds=None):
    """The terms the facts use, with the window length of this analysis."""
    slice_ = f"{_plain(window_seconds)}-second" if window_seconds else "short"
    return {
        "link": f"More of an instructor action in one {slice_} window was followed by a change in a student behaviour "
                "a little later. A link in time, not proof that one caused the other.",
        "event_change": "How a student behaviour changed in the seconds after the instructor started an action, "
                        "compared with the seconds just before, averaged over the times it happened. A link in time, "
                        "not proof that one caused the other.",
        "after_seconds": "How long after the instructor action the student behaviour was measured.",
        "windows": f"How many {slice_} slices of the lecture the link was measured on.",
        "times": "How many times the instructor started that action (starts close together count once).",
        "average_change": "How much the behaviour changed on average: in percentage points of the students for "
                          "shares, or in number per window for counts.",
        "evidence.strength": "suggestive = unlikely to be chance even after allowing for the number of links tested; "
                             "weak = could easily be chance.",
        "effect_size": "small / moderate / large: how closely the two moved together (only for lagged links).",
        "prespecified": "true = one of the questions fixed before the analysis was run, not found by searching.",
    }


SECTIONS = ("instructor_actions", "instructor_events", "student_behaviours")


def load_vocabulary(path):
    """The plain-language names for indicator IDs (configs/report_vocabulary.yaml).

    An optional `measured_by` section holds other wording for analyses measured by another pipeline (its fusion
    report says "measured_by"), whose rules are narrower than the team's. It may only change phrases of IDs the
    vocabulary already has, never whether a rise is good news."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    data.setdefault("instructor_events", {})
    data.setdefault("measured_by", {})
    for key in SECTIONS:
        if not isinstance(data.get(key), dict):
            raise ValueError(f"{path}: '{key}' must be a mapping of indicator IDs.")
        for name, entry in data[key].items():
            if not isinstance(entry, dict) or not str(entry.get("phrase") or "").strip():
                raise ValueError(f"{path}: '{key}.{name}' needs a phrase.")
    for name, entry in data["student_behaviours"].items():
        value = entry.get("favourable_when_higher", "missing")
        if not (value is None or isinstance(value, bool)):
            raise ValueError(f"{path}: student behaviour '{name}' needs favourable_when_higher: true, false or null.")
    if not isinstance(data["measured_by"], dict):
        raise ValueError(f"{path}: 'measured_by' must be a mapping of pipeline names.")
    for source, sections in data["measured_by"].items():
        for key, entries in (sections or {}).items():
            if key not in SECTIONS or not isinstance(entries, dict):
                raise ValueError(f"{path}: 'measured_by.{source}.{key}' is not one of {', '.join(SECTIONS)}.")
            for name, entry in entries.items():
                where = f"{path}: 'measured_by.{source}.{key}.{name}'"
                if name not in data[key]:
                    raise ValueError(f"{where} has no entry in '{key}' to reword.")
                if not isinstance(entry, dict) or set(entry) != {"phrase"} or not str(entry["phrase"] or "").strip():
                    raise ValueError(f"{where} may only give a phrase (never whether a rise is good news).")
    return data


def vocabulary_for(vocabulary, measured_by=None):
    """The vocabulary with the wording for the pipeline that measured this analysis (unchanged if it has none)."""
    overrides = (vocabulary.get("measured_by") or {}).get(measured_by) or {}
    if not overrides:
        return vocabulary
    merged = {key: {name: dict(entry) for name, entry in vocabulary.get(key, {}).items()} for key in SECTIONS}
    for key, entries in overrides.items():
        for name, entry in entries.items():
            merged[key][name]["phrase"] = entry["phrase"]
    return {**vocabulary, **merged}


def from_fusion(doc):
    """(rows, metadata) from the fusion layer's <name>.fusion.json (scripts/fuse_lesson.py)."""
    window = float(doc["window_seconds"])
    windows = doc.get("windows")
    settings = doc.get("settings") or {}
    rows = [{"kind": "lagged_link", "instructor_action": link["teacher_feature"],
             "student_behaviour": link["student_feature"], "lag_seconds": link["lag_seconds"], "effect": link["r"],
             "reverse_effect": link.get("r_reverse_max"), "effect_type": "correlation",
             "p_value": link.get("p_shift"), "q_value": link.get("q"), "n_windows": windows}
            for link in doc.get("links") or []]
    too_few_starts = {}
    for test in doc.get("peri_event") or []:
        if "p_random_onsets" not in test:              # "skipped": too few onsets
            too_few_starts[test["teacher_event"]] = test.get("events", 0)
            continue
        span = test.get("curve_relative_windows") or [0]
        rows.append({"kind": "event_change", "instructor_action": test["teacher_event"],
                     "student_behaviour": test["student_feature"], "lag_seconds": max(span) * window,
                     "before_seconds": -min(span) * window, "effect": test["mean_change"],
                     "effect_type": "mean_change", "p_value": test["p_random_onsets"], "q_value": test.get("q"),
                     "events": test["events"], "n_windows": windows})
    features = doc.get("features") or {}
    max_lag = settings.get("max_lag_windows")
    metadata = {
        "source": "fusion",
        "measured_by": doc.get("measured_by"),          # e.g. "main_code" (main code/Main Code.py); None = the team's
        "alpha": settings.get("alpha", ALPHA),
        "lecture": {"windows": windows, "window_seconds": _plain(window),
                    "duration_seconds": _plain(round(windows * window)) if windows else None},
        "lags_tested_seconds": [_plain(round(k * window, 3)) for k in range(1, int(max_lag) + 1)] if max_lag else None,
        "feature_kinds": {name: f.get("kind", "share") for name, f in features.items()},
        "desirable": {name: f.get("desirable", 0) for name, f in features.items() if f.get("camera") == "students"},
        "too_rare": {"actions": dict(doc.get("untested_teacher_features") or {}), "events": too_few_starts},
        "data_quality": {"limitations": list(doc.get("limitations") or [])},
    }
    return rows, metadata


def load_results(path):
    """Reads the lagged-analysis output. Returns (rows, metadata)."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        with open(path, encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f)), {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data, {}
    if isinstance(data, dict) and isinstance(data.get("links"), list) and "window_seconds" in data:
        return from_fusion(data)
    if not isinstance(data, dict) or not isinstance(data.get("results"), list):
        raise ValueError(f"{path}: expected a fusion layer report (<name>.fusion.json), a list of rows, "
                         "or a document with a 'results' list.")
    return data["results"], {k: v for k, v in data.items() if k != "results"}


def _pick(raw, field):
    for name in COLUMN_ALIASES[field]:
        value = raw.get(name)
        if value not in (None, "", "nan", "NaN"):
            return value
    return None


def _number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None   # NaN is not a number here


def normalize_row(raw, index=0):
    """One row in the contract's field names and types; raises ValueError naming what is missing."""
    row = {field: _pick(raw, field) for field in COLUMN_ALIASES}
    for field in ("instructor_action", "student_behaviour"):
        if not row[field]:
            raise ValueError(f"row {index}: missing {field} (accepted names: {', '.join(COLUMN_ALIASES[field])}).")
        row[field] = str(row[field]).strip()
    for field in ("lag_seconds", "before_seconds", "effect", "reverse_effect", "p_value", "q_value", "n_windows", "events"):
        row[field] = _number(row[field])
    if row["lag_seconds"] is None or row["effect"] is None:
        raise ValueError(f"row {index} ({row['instructor_action']} -> {row['student_behaviour']}): lag_seconds and effect must be numbers.")
    for field in ("p_value", "q_value"):
        if row[field] is not None and not 0 <= row[field] <= 1:
            raise ValueError(f"row {index}: {field} must be between 0 and 1.")
    row["kind"] = str(row["kind"] or "lagged_link").strip()
    if row["kind"] not in KINDS:
        raise ValueError(f"row {index}: kind must be one of {', '.join(KINDS)}.")
    default_type = "correlation" if row["kind"] == "lagged_link" else "mean_change"
    row["effect_type"] = str(row["effect_type"] or default_type).strip().lower()
    row["prespecified"] = str(row["prespecified"]).strip().lower() in ("true", "1", "yes")
    for field in ("n_windows", "events"):
        row[field] = int(row[field]) if row[field] is not None else None
    return row


def judge(p_value, n_family, q_value=None, alpha=ALPHA):
    """How much weight a link deserves. With a q-value: the fusion layer's rule (q <= alpha); otherwise a Bonferroni
    limit for the link's family of tests. None = no link worth reporting."""
    if q_value is not None:
        if q_value <= alpha:
            return "suggestive"
        return "weak" if p_value is not None and p_value < alpha else None
    if p_value is None:
        return "unknown"
    if p_value < alpha / max(n_family, 1):
        return "suggestive"
    if p_value < alpha:
        return "weak"
    return None


def size_word(effect, effect_type):
    if effect_type != "correlation":
        return None
    size = abs(effect)
    return "small" if size < 0.3 else "moderate" if size < 0.5 else "large"


def change_size(effect, feature_kind, window_seconds=None):
    """(average change, unit) of an event_change in words a person reads: percentage points for shares."""
    if feature_kind == "count":
        unit = f"per {_plain(window_seconds)}-second window" if window_seconds else "per window"
        return _plain(round(abs(effect), 2)), unit
    if feature_kind in ("share", "episode"):
        return _plain(round(abs(effect) * 100, 1)), "percentage points"
    return _plain(round(abs(effect), 3)), None


def check_directions(behaviours, desirable):
    """Refuse to report when the vocabulary and the fusion layer disagree on whether a rise is good news."""
    wrong = [f"{name} (fusion layer: desirable {value}; vocabulary: favourable_when_higher "
             f"{behaviours[name]['favourable_when_higher']})"
             for name, value in sorted(desirable.items())
             if name in behaviours and DESIRABLE.get(value, "unknown") != behaviours[name]["favourable_when_higher"]]
    if wrong:
        raise ValueError("The fusion layer and configs/report_vocabulary.yaml disagree on whether a rise is good news "
                         f"for: {'; '.join(wrong)}. Fix one of them, so the report cannot call bad news good.")


def build_facts(rows, vocabulary, metadata=None, max_weak=MAX_WEAK_FACTS):
    """The facts document: one fact per instructor action / student behaviour pair with a link worth reporting.

    Every row counts towards the correction, reported or not, and pre-registered rows form their own family. For each
    pair only its strongest lag is kept, so the same link is never reported twice. Behaviours that are neither good
    nor bad news (favourable_when_higher: null) are counted but not turned into facts."""
    metadata = metadata or {}
    vocabulary = vocabulary_for(vocabulary, metadata.get("measured_by"))
    alpha = float(metadata.get("alpha", ALPHA))
    lecture = metadata.get("lecture") or {}
    window = lecture.get("window_seconds")
    kinds = metadata.get("feature_kinds") or {}
    normalized = [normalize_row(raw, i) for i, raw in enumerate(rows)]
    actions, events = vocabulary["instructor_actions"], vocabulary.get("instructor_events", {})
    behaviours = vocabulary["student_behaviours"]
    check_directions(behaviours, metadata.get("desirable") or {})
    families = {flag: sum(1 for r in normalized if r["prespecified"] == flag) for flag in (True, False)}

    def doers(row):
        return events if row["kind"] == "event_change" else actions

    rare = metadata.get("too_rare") or {}
    not_interpreted = {r["instructor_action"] for r in normalized if r["instructor_action"] not in doers(r)} | \
        {r["student_behaviour"] for r in normalized if r["student_behaviour"] not in behaviours} | \
        {name for name in rare.get("actions", {}) if name not in actions} | \
        {name for name in rare.get("events", {}) if name not in events}
    best, neutral = {}, set()
    for row in normalized:
        if row["instructor_action"] not in doers(row) or row["student_behaviour"] not in behaviours:
            continue
        row["strength"] = judge(row["p_value"], families[row["prespecified"]], row["q_value"], alpha)
        if row["strength"] is None:
            continue
        key = (row["kind"], row["instructor_action"], row["student_behaviour"])
        if behaviours[row["student_behaviour"]]["favourable_when_higher"] is None:
            if row["strength"] == "suggestive":
                neutral.add(key)
            continue
        certainty = row["q_value"] if row["q_value"] is not None else row["p_value"]
        rank = (STRENGTH_RANK[row["strength"]], certainty if certainty is not None else 1.0, -abs(row["effect"]))
        if key not in best or rank < best[key][0]:
            best[key] = (rank, row)

    chosen = [row for _, row in sorted(best.values(), key=lambda item: item[0])]
    weak = [row for row in chosen if row["strength"] == "weak"]
    chosen = [row for row in chosen if row["strength"] != "weak"] + weak[:max_weak]

    facts = []
    for n, row in enumerate(chosen, 1):
        behaviour = behaviours[row["student_behaviour"]]
        rose = row["effect"] > 0
        notes = ["One of the questions fixed before the analysis." if row["prespecified"] else
                 "Found by searching many links; treat it as a lead."]
        if row["strength"] == "unknown":
            notes.append("No test result was given for this link.")
        if row["reverse_effect"] is not None and abs(row["reverse_effect"]) >= abs(row["effect"]):
            notes.append("The link is as strong the other way round (the students' change came first), so both may "
                         "follow a shared trend.")
        fact = {"id": f"F{n}", "direction": "favourable" if rose == behaviour["favourable_when_higher"] else "unfavourable",
                "kind": row["kind"]}
        if row["kind"] == "lagged_link":
            fact.update({
                "instructor_action": actions[row["instructor_action"]]["phrase"],
                "student_behaviour": behaviour["phrase"],
                "change": "rose" if rose else "fell",
                "after_seconds": int(round(row["lag_seconds"])),
                "windows": row["n_windows"],
                "effect": round(row["effect"], 2),
                "effect_size": size_word(row["effect"], row["effect_type"]),
            })
        else:
            size, unit = change_size(row["effect"], kinds.get(row["student_behaviour"]), window)
            if row["events"] is not None and row["events"] < FEW_TIMES:
                notes.append(f"Based on only {row['events']} times the action started.")
            fact.update({
                "instructor_event": events[row["instructor_action"]]["phrase"],
                "student_behaviour": behaviour["phrase"],
                "change": "rose" if rose else "fell",
                "times": row["events"],
                "after_seconds": int(round(row["lag_seconds"])),
                "before_seconds": int(round(row["before_seconds"])) if row["before_seconds"] is not None else None,
                "average_change": size,
                "unit": unit,
            })
        fact.update({"prespecified": row["prespecified"], "evidence": {"strength": row["strength"], "notes": notes},
                     "indicators": [row["instructor_action"], row["student_behaviour"]]})
        facts.append(fact)

    lags = metadata.get("lags_tested_seconds") or sorted(
        {int(round(r["lag_seconds"])) for r in normalized if r["kind"] == "lagged_link"})
    too_rare = [f"{actions[name]['phrase']} (seen in {n} window{'' if n == 1 else 's'})"
                for name, n in rare.get("actions", {}).items() if name in actions] + \
               [f"{events[name]['phrase']} ({n} time{'' if n == 1 else 's'})"
                for name, n in rare.get("events", {}).items() if name in events]
    shown = {i for f in facts for i in f["indicators"]}
    validation = {name: {"checked_against_human_labels": False} for name in sorted(shown)}
    validation.update({k: v for k, v in (metadata.get("indicator_validation") or {}).items() if k in shown})
    return {
        "schema_version": 2,
        "lecture": lecture,
        "analysis": {
            "links_tested": len(normalized),
            "prespecified_tests": families[True],
            "suggestive_links": sum(1 for f in facts if f["evidence"]["strength"] == "suggestive"),
            "weak_links_shown": sum(1 for f in facts if f["evidence"]["strength"] == "weak"),
            "weak_links_found": len(weak),
            "neutral_links_found": len(neutral),
            "seconds_after_tested": lags,
            "too_rare_to_test": too_rare,
        },
        "data_quality": {"limitations": list((metadata.get("data_quality") or {}).get("limitations", []))},
        "indicator_validation": validation,
        "glossary": glossary(window),
        "alternative_explanations": ALTERNATIVE_EXPLANATIONS,
        "facts": facts,
        "not_interpreted": sorted(not_interpreted),
    }


def facts_hash(facts_doc):
    """A short, stable fingerprint of the facts (key order does not matter)."""
    return hashlib.sha256(json.dumps(facts_doc, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def unverified_indicators(fact, facts_doc, min_precision=0.7):
    """The detections a fact rests on that no person has checked, or that were checked and found unreliable."""
    shaky = []
    for name in fact.get("indicators", ()):
        entry = facts_doc.get("indicator_validation", {}).get(name, {})
        precision = entry.get("precision", entry.get("accuracy"))
        if not entry.get("checked_against_human_labels") or (precision is not None and precision < min_precision):
            shaky.append(name)
    return shaky


def rule_sentence(fact, window_seconds=None, subject="the instructor"):
    """The plain sentence the report uses for a fact when no language model is involved. subject="you" speaks to the
    instructor, as the web report does."""
    weight = {"suggestive": "unlikely to be chance", "weak": "could easily be chance",
              "unknown": "not tested"}[fact["evidence"]["strength"]]
    if fact["kind"] == "event_change":
        size = f"{fact['average_change']} {fact['unit']}" if fact.get("unit") else str(fact["average_change"])
        times = f" ({fact['times']} time{'' if fact['times'] == 1 else 's'})" if fact.get("times") else ""
        before = f", compared with the {fact['before_seconds']} seconds before" if fact.get("before_seconds") else ""
        return (f"After {subject} {fact['instructor_event']}{times}, {fact['student_behaviour']} {fact['change']} "
                f"by {size} on average over the next {fact['after_seconds']} seconds{before} ({weight}).")
    slices = f" windows of {_plain(window_seconds)} seconds" if window_seconds else " windows"
    windows = f"; {fact['windows']}{slices}" if fact.get("windows") else ""
    change = "rise" if fact["change"] == "rose" else "fall"
    return (f"When {subject} spent more time {fact['instructor_action']}, {fact['student_behaviour']} tended to "
            f"{change} about {fact['after_seconds']} seconds later ({weight}{windows}).")


def rule_based_points(facts_doc, limit=3, subject="the instructor"):
    """(overview, positive, negative) from the facts alone: the report used when the language model is not."""
    analysis = facts_doc["analysis"]
    found = analysis["suggestive_links"] + analysis["weak_links_found"]
    neutral = analysis.get("neutral_links_found", 0)
    were = lambda n: f"{n} {'was' if n == 1 else 'were'}"
    tested = (f"{analysis['links_tested']} possible links between what {subject} did and how students behaved "
              "shortly after were tested")
    overview = (f"{tested}; {were(analysis['suggestive_links'])} unlikely to be chance and "
                f"{analysis['weak_links_found']} could easily be chance." if found else
                f"{tested}, and none about good or bad news stood out from chance." if neutral else
                f"{tested}, and none stood out from chance.")
    if neutral:
        overview += (f" {were(neutral)} unlikely to be chance but about behaviours that are neither good nor bad news, "
                     f"so {'it is' if neutral == 1 else 'they are'} not listed.")
    if analysis.get("too_rare_to_test"):
        overview += f" Too rare to test: {'; '.join(analysis['too_rare_to_test'])}."
    window = (facts_doc.get("lecture") or {}).get("window_seconds")
    points = {"favourable": [], "unfavourable": []}
    for fact in facts_doc["facts"]:      # facts are already ordered strongest first
        if len(points[fact["direction"]]) < limit:
            points[fact["direction"]].append({"text": rule_sentence(fact, window, subject), "fact_ids": [fact["id"]]})
    return overview, points["favourable"], points["unfavourable"]


def report_note(facts_doc, subject="the instructor"):
    """The fixed note printed under the points."""
    lags = facts_doc["analysis"]["seconds_after_tested"]
    span = f"{lags[0]}-{lags[-1]} seconds" if len(lags) > 1 else (f"{lags[0]} seconds" if lags else "a short time")
    unchecked = any(not v.get("checked_against_human_labels") for v in facts_doc["indicator_validation"].values())
    return (f"Note: these are links in time between what {subject} did and how students behaved about "
            f"{span} later; they are not proof that one caused the other. Student behaviours are observed postures "
            "and objects, not measures of attention or learning."
            + (" The detections have not yet been checked against a person's labels." if unchecked else ""))
