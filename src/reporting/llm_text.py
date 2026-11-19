"""Writes the instructor's positive / negative points from the lagged-analysis facts with Claude, then checks them.

The model only sees the facts built by `lag_facts.build_facts` (numbers, direction and evidence strength already
decided in code; no video, tables or names). Everything it writes is checked in code: every number must come from a
cited fact, positive points may only cite favourable facts, no causal or inner-state wording, no statistics jargon,
hedges where the evidence is weak or a detection is unverified. One retry with the problems listed; if that also
fails, or the API cannot be reached, the rule-based points are used instead, so a report is always produced.

The facts themselves are kept in memory; they are written to disk only when asked (keep_facts=True).
"""
import hashlib
import json
import os
import re
from pathlib import Path

from src.reporting import lag_facts

# alias -> (model id, input $/M tokens, output $/M tokens, supports server-side fallbacks). Anthropic list prices;
# Claude Haiku 5.5 has no server-side fallback, so none may be sent to it.
MODELS = {
    "opus": ("claude-opus-5-5", 4.00, 20.00, True),
    "sonnet": ("claude-sonnet-5-5", 2.00, 10.00, True),
    "haiku": ("claude-haiku-5-5", 0.10, 0.50, False),
}
EFFORT = "medium"                  # the default for Opus 5.5 and Haiku 5.5, set explicitly; thinking stays adaptive
PROMPT_VERSION = "lag-3"           # 2: the fusion layer's windows and event changes; 3: "you" (changes the cache key)
MAX_ATTEMPTS = 2                   # the second attempt is told which checks the first one failed
MAX_POINTS = 3
MAX_WORDS = 80
POINTS_FILE, TEXT_FILE, FACTS_FILE = "report_points.json", "report_points.txt", "lag_facts.json"

SYSTEM_PROMPT = """\
You explain the results of a lagged analysis of one recorded lecture to the university instructor who taught it.
You do not see the video. A program cut the lecture into short windows (lecture.window_seconds), measured what the
instructor was doing and how the students behaved in each, and tested two kinds of link, given to you as JSON facts:
- kind "lagged_link": more of an instructor action was followed, after_seconds later, by a change in a student
  behaviour.
- kind "event_change": after the instructor started an action (times = how often), a student behaviour changed on
  average by average_change (in unit) over the next after_seconds, compared with the before_seconds just before.
Write an overview and the "positive points" and "negative points" of the instructor's feedback.

Rules:
1. Use only the facts given. Copy numbers exactly as written and write them as digits. Never calculate a new number.
2. List the fact ids behind each point in fact_ids. A positive point may cite only facts whose direction is
   "favourable"; a negative point only facts whose direction is "unfavourable". Never write ids, or the word "fact",
   in the text.
3. These are links in time, not proof of cause. Write "was followed by", "tended to", "in the seconds after". Never
   write "caused", "led to", "resulted in", "made students" or "due to".
4. Match each point to evidence.strength. "suggestive" means unlikely to be chance. "weak" means it could easily be
   chance: say so plainly. Never present a weak link as a clear one. Keep the cautions in evidence.notes.
5. Where it helps, add one short possible explanation from alternative_explanations, marked as a possibility ("this
   may be because ..."), and list its id in explanation_ids. Never present it as a finding.
6. Describe student behaviours only with the words in the facts and glossary (observed postures and objects). Do not
   write about attention, engagement, boredom, understanding, motivation, confusion, sleep, tiredness or distraction
   as if they were measured. Never judge the instructor as a person.
7. Plain English for an instructor who is not a statistician, 2 or 3 sentences per point, speaking to them as "you".
   No jargon: never write "p-value", "correlation", "coefficient", "regression", "Bonferroni" or "lag".
8. At most 3 positive and 3 negative points. If the facts support fewer, write fewer; never pad. Combine facts that
   point the same way. Put first the points about actions the instructor can change.
9. indicator_validation says whether a person has checked each detection. In every point resting on a detection
   that has not been checked, say it has not been verified.
10. overview: 1 or 2 sentences on how many links were tested and how many stood out, from the analysis section. If
    analysis.too_rare_to_test lists actions, you may say they happened too rarely to test. No judgement of the
    instructor.
"""

POINT_SCHEMA = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "fact_ids": {"type": "array", "items": {"type": "string"}},
        "explanation_ids": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["text", "fact_ids", "explanation_ids"],
    "additionalProperties": False,
}
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "overview": {"type": "string"},
        "positive_points": {"type": "array", "items": POINT_SCHEMA},
        "negative_points": {"type": "array", "items": POINT_SCHEMA},
    },
    "required": ["overview", "positive_points", "negative_points"],
    "additionalProperties": False,
}

TIME_RE = re.compile(r"\b\d{1,2}:\d{2}\b")
NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
CAUSAL_RE = re.compile(r"\b(caus(?:e|es|ed|ing)|led to|leads? to|leading to|result(?:s|ed|ing)? in|due to|as a result|"
                       r"drove|triggered|made (?:the )?students|makes (?:the )?students|proves?|proved)\b", re.I)
INTERNAL_STATE_RE = re.compile(r"\b(attention|attentive|inattentive|engag\w*|bored\w*|understand\w*|understood|motivat\w*|"
                               r"confus\w*|enjoy\w*|asleep|sleep\w*|drows\w*|tired\w*|fatigue\w*|distract\w*)\b", re.I)
JUDGEMENT_RE = re.compile(r"\b(good|bad|poor|great|terrible|ineffective|effective|excellent)\s+(teacher|instructor|teaching|lecturer)\b", re.I)
JARGON_RE = re.compile(r"p-value|\bp\s*[<=]|q-value|\bq\s*[<=]|\bcorrelat\w*|\bcoefficient|\bregression|\bbonferroni|"
                       r"\blag(?:s|ged)?\b", re.I)
LEAK_RE = re.compile(r"\bF\d+\b|\bfacts?\b|\bE\d\b")
HEDGES = ("may", "might", "could", "possibly", "possible", "perhaps", "suggest", "tentative", "weak", "unclear", "chance",
          "uncertain", "not certain", "caution", "cannot tell", "can't tell", "not clear")
UNVERIFIED_RE = re.compile(r"unverified|not (?:been |yet )?(?:verified|checked)|unreliable|no one has checked|"
                           r"has not been checked|have not been checked", re.I)
NUMBER_WORDS = {"two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
                "ten": "10", "eleven": "11", "twelve": "12"}


def numbers_in(text):
    """The numbers and mm:ss times in a piece of text (times count as one token, not as two numbers)."""
    return set(TIME_RE.findall(text)) | set(NUMBER_RE.findall(TIME_RE.sub(" ", text)))


def build_user_message(facts_doc):
    shown = {k: v for k, v in facts_doc.items() if k != "not_interpreted"}
    return "Write the overview and points for this lecture from the facts below.\n\n<facts>\n" + \
        json.dumps(shown, indent=2, sort_keys=True) + "\n</facts>"


def _word_checks(where, text, allowed):
    problems = []
    invented = sorted(numbers_in(text) - allowed)
    if invented:
        problems.append(f"{where}: numbers not found in the cited facts: {invented}.")
    for word, digit in NUMBER_WORDS.items():
        if re.search(rf"\b{word}\b", text, re.I) and digit not in allowed:
            problems.append(f"{where}: the word '{word}' is a number that is not in the cited facts; write numbers as digits from the facts.")
    for label, pattern in (("causal wording", CAUSAL_RE), ("a word about internal states", INTERNAL_STATE_RE),
                           ("a judgement of the instructor", JUDGEMENT_RE), ("statistics jargon", JARGON_RE),
                           ("a fact id or the word 'fact'", LEAK_RE)):
        found = pattern.search(text)
        if found:
            problems.append(f"{where}: contains {label} ('{found.group(0)}').")
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if re.search(r"\bbecause\b", sentence, re.I) and not any(h in sentence.lower() for h in HEDGES):
            problems.append(f"{where}: says 'because' without marking it as a possibility.")
    return problems


def validate(output, facts_doc):
    """Returns a list of problems; an empty list means the answer passed every check."""
    problems = []
    if not isinstance(output, dict):
        return ["The answer is not a JSON object."]
    by_id = {f["id"]: f for f in facts_doc["facts"]}
    explanations = facts_doc.get("alternative_explanations", {})
    context = numbers_in(json.dumps(facts_doc.get("lecture", {}))) | numbers_in(json.dumps(facts_doc["analysis"]))

    overview = output.get("overview")
    if not isinstance(overview, str) or not overview.strip():
        problems.append("overview is missing or empty.")
    else:
        if len(overview.split()) > MAX_WORDS:
            problems.append(f"overview: {len(overview.split())} words; the maximum is {MAX_WORDS}.")
        problems += _word_checks("overview", overview, context)

    for key, direction in (("positive_points", "favourable"), ("negative_points", "unfavourable")):
        points = output.get(key)
        if not isinstance(points, list):
            problems.append(f"{key} is missing or not a list.")
            continue
        if len(points) > MAX_POINTS:
            problems.append(f"{key} has {len(points)} points; the maximum is {MAX_POINTS}.")
        for n, point in enumerate(points, 1):
            where = f"{key} #{n}"
            text, ids = point.get("text", ""), point.get("fact_ids", [])
            if not text.strip():
                problems.append(f"{where}: empty text.")
                continue
            if not ids:
                problems.append(f"{where}: cites no fact ids.")
            unknown = [i for i in ids if i not in by_id]
            if unknown:
                problems.append(f"{where}: cites unknown fact ids {unknown}.")
            wrong_way = [i for i in ids if i in by_id and by_id[i]["direction"] != direction]
            if wrong_way:
                problems.append(f"{where}: cites {wrong_way}, which are not '{direction}' facts.")
            bad_explanations = [e for e in point.get("explanation_ids", []) if e not in explanations]
            if bad_explanations:
                problems.append(f"{where}: unknown explanation ids {bad_explanations}.")
            if len(text.split()) > MAX_WORDS:
                problems.append(f"{where}: {len(text.split())} words; the maximum is {MAX_WORDS}.")

            cited = [by_id[i] for i in ids if i in by_id]
            allowed = set(context)
            for fact in cited:
                allowed |= numbers_in(json.dumps(fact))
            for explanation_id in point.get("explanation_ids", []):
                allowed |= numbers_in(explanations.get(explanation_id, ""))
            problems += _word_checks(where, text, allowed)
            if any(f["evidence"]["strength"] in ("weak", "unknown") for f in cited) and not any(h in text.lower() for h in HEDGES):
                problems.append(f"{where}: rests on a weak link but does not say so.")
            shaky = sorted({name for f in cited for name in lag_facts.unverified_indicators(f, facts_doc)})
            if shaky and not UNVERIFIED_RE.search(text):
                problems.append(f"{where}: rests on detections that are unverified or unreliable ({', '.join(shaky)}) but does not say so.")
    return problems


def call_claude(system, user, model_alias):
    """One request to Claude. Returns (output or None, usage dict, error text or None); never raises for API problems,
    so the caller can fall back to the rule-based points."""
    import anthropic   # imported here so the rule-based path works without the package
    model, _, _, fallbacks = MODELS[model_alias]
    request = dict(model=model, max_tokens=16000, system=system, messages=[{"role": "user", "content": user}],
                   output_config={"effort": EFFORT, "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}})
    try:
        client = anthropic.Anthropic()
        if fallbacks:   # if a safety classifier declines, the API retries on Anthropic's recommended model
            response = client.beta.messages.create(betas=["server-side-fallback-2026-07-01"], fallbacks="default", **request)
        else:           # Claude Haiku 5.5 has no server-side fallback
            response = client.messages.create(**request)
    except TypeError as error:   # raised locally, before any network traffic, when no credentials are set
        if "authentication method" in str(error):
            return None, None, "No Claude credentials found (set ANTHROPIC_API_KEY or run `ant auth login`). Nothing was sent."
        raise
    except anthropic.AuthenticationError:
        return None, None, "Claude rejected the credentials. Check ANTHROPIC_API_KEY."
    except anthropic.RateLimitError as error:
        return None, None, f"Rate limited; retry after {error.response.headers.get('retry-after', '60')} s."
    except anthropic.BadRequestError as error:
        return None, None, f"Claude rejected the request: {error.message}"
    except anthropic.APIStatusError as error:
        return None, None, f"Claude API error {error.status_code}: {error.message}"
    except anthropic.APIConnectionError:
        return None, None, "Could not reach the Claude API. Check the network connection."
    usage = {"request_id": getattr(response, "_request_id", None), "served_by": response.model,
             "input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens}
    if response.stop_reason == "refusal":
        return None, usage, f"The model declined (stop_details: {getattr(response, 'stop_details', None)})."
    if response.stop_reason == "max_tokens":
        return None, usage, "The answer was cut off at max_tokens."
    text = next((b.text for b in response.content if b.type == "text"), "")
    try:
        return json.loads(text), usage, None
    except json.JSONDecodeError:
        return None, usage, "The answer was not valid JSON."


def estimate_cost(user, model_alias, output_tokens=1500):
    """(approximate input tokens, approximate dollars) for one successful call at list price."""
    _, price_in, price_out, _ = MODELS[model_alias]
    tokens = (len(SYSTEM_PROMPT) + len(user)) // 4
    return tokens, tokens * price_in / 1e6 + output_tokens * price_out / 1e6


def render_text(points):
    def block(items, empty):
        return "\n".join(f"- {p['text'].strip()}" for p in items) if items else f"- {empty}"
    return (f"{points['overview']}\n\npositive points:\n\n{block(points['positive_points'], 'No clear positive point was found in this analysis.')}"
            f"\n\nnegative points:\n\n{block(points['negative_points'], 'No clear negative point was found in this analysis.')}"
            f"\n\n{points['note']}\n")


def generate(facts_doc, out_dir, model_alias="opus", use_llm=True, fake_response=None, use_cache=True,
             keep_facts=False, call=call_claude, subject="the instructor"):
    """Writes report_points.json and report_points.txt to out_dir and returns the points.

    use_llm=False gives the rule-based points only. fake_response (a dict) replaces the API call, to test the checks.
    `call` is the function that talks to the API (tests pass a fake). subject="you" writes the rule-based points to
    the instructor (the model always does)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    overview, positive, negative = lag_facts.rule_based_points(facts_doc, subject=subject)
    points = {"source": "rules", "model": None, "overview": overview, "positive_points": positive,
              "negative_points": negative, "note": lag_facts.report_note(facts_doc, subject), "attempts": [],
              "facts_hash": lag_facts.facts_hash(facts_doc), "not_interpreted": facts_doc.get("not_interpreted", [])}

    if use_llm and facts_doc["facts"]:
        model = MODELS[model_alias][0]
        user = build_user_message(facts_doc)
        key = hashlib.sha256((model + PROMPT_VERSION + SYSTEM_PROMPT + user).encode("utf-8")).hexdigest()[:16]
        cache_path = out_dir / "llm_cache" / f"{key}.json"
        output, problems = None, ["not attempted"]
        if fake_response is not None:
            output, problems = fake_response, validate(fake_response, facts_doc)
            points["attempts"].append({"source": "fake", "problems": problems})
        elif use_cache and cache_path.exists():
            with open(cache_path, encoding="utf-8") as f:
                output, problems = json.load(f)["output"], []
            points["attempts"].append({"source": "cache", "problems": []})
        else:
            feedback = ""
            for _ in range(MAX_ATTEMPTS):
                output, usage, error = call(SYSTEM_PROMPT, user + feedback, model_alias)
                problems = [error] if error else validate(output, facts_doc)
                points["attempts"].append({"source": "api", **(usage or {}), "problems": problems})
                if not problems or (error and usage is None):   # passed, or the API is unreachable: stop
                    break
                feedback = "\n\nYour previous answer failed these checks. Fix them and answer again:\n- " + "\n- ".join(problems)
            if not problems and use_cache:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump({"model": model, "prompt_version": PROMPT_VERSION, "output": output}, f, indent=2)
        if not problems:
            points.update(source="llm", model=model, overview=output["overview"],
                          positive_points=[{"text": p["text"], "fact_ids": p["fact_ids"]} for p in output["positive_points"]],
                          negative_points=[{"text": p["text"], "fact_ids": p["fact_ids"]} for p in output["negative_points"]])

    with open(out_dir / POINTS_FILE, "w", encoding="utf-8") as f:
        json.dump(points, f, indent=2)
    with open(out_dir / TEXT_FILE, "w", encoding="utf-8") as f:
        f.write(render_text(points))
    facts_path = out_dir / FACTS_FILE
    if keep_facts:
        with open(facts_path, "w", encoding="utf-8") as f:
            json.dump(facts_doc, f, indent=2)
    elif facts_path.exists():
        os.remove(facts_path)   # an older run's facts would no longer match this report
    return points
