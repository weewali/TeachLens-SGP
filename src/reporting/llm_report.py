"""The LLM report stage: facts.json -> Claude -> checked positive / negative points.

Moved here from the pipeline_creation branch's `main code/llm_report.py` (2026-11-20) and adapted to the
lesson facts.json that src/reporting/facts.py writes (schema_version 1). The model never sees the video, the
tables or any name: it only gets facts.json. Everything it writes is checked in code before it is kept (every
number must come from a cited fact, the right direction, no causal or inner-state wording, hedges for weak
evidence and for detections nobody has checked). One retry with the failures listed.

Claude writes the feedback for every lecture (user decisions 2026-11-20: always on, and no rule-based text). A
failure never crashes the stage: `rewrite()` returns the reason (no API key, no network, rejected text, ...), no
feedback is written, and the web app shows the reason with a "Try again" button. A problem with the API key (missing,
rejected, not allowed) is shown only as "Code=01", a billing problem of the key's account (credit balance too low,
spend limit, on hold) only as "Code=02" (user requests 2026-11-20); the reason stays in the lesson's files.
"""
import hashlib
import json
import re
from pathlib import Path
from typing import Any

# alias -> (model id, input $/M tokens, output $/M tokens, supports server-side fallbacks). Anthropic list prices;
# Haiku 5.5 has no server-side fallback, so none may be sent to it.
MODELS = {
    "opus": ("claude-opus-5-5", 4.00, 20.00, True),
    "sonnet": ("claude-sonnet-5-5", 2.00, 10.00, True),
    "haiku": ("claude-haiku-5-5", 0.10, 0.50, False),
}
DEFAULT_MODEL = "opus"
EFFORT = "medium"            # Opus 5.5 rejects temperature and cannot turn thinking off
PROMPT_VERSION = "4"         # 4: the lesson facts of src/reporting/facts.py ("on task" / "off task"); changes the cache key
MAX_ATTEMPTS = 2             # the second attempt is told which checks the first one failed
MAX_POINTS = 3
MAX_WORDS = 80
MIN_PRECISION = 0.7

SYSTEM_PROMPT = """\
You write the "positive points" and "negative points" of a classroom feedback report for a university instructor.
You do not see the video. A program has already analysed it and produced the JSON facts you are given. Every number
in your answer must come from those facts.

Rules:
1. Use only the facts given. Copy numbers, percentages, counts and times exactly as written (digits, and mm:ss times).
   Never calculate a new number, round differently or convert units. Write numbers as digits.
2. List the fact ids behind each point in fact_ids. A positive point may cite only facts whose direction is
   "favourable"; a negative point only facts whose direction is "unfavourable". Do not write ids, or the word "fact",
   in the text itself.
3. Describe what happened together, never what caused what. Do not use words such as "caused", "led to", "resulted in",
   "due to" or "as a result". Say "while", "compared with", "in the parts of the lecture where".
4. Match your wording to each fact's evidence.strength. "weak" means the gap could easily be chance, so say that
   plainly. "descriptive" means a share, a count or a single stretch, not a test, so do not present it as a pattern.
   Never present a weak finding as a clear one.
5. Where it helps, add one short sentence with a possible explanation taken from alternative_explanations, clearly marked
   as a possibility ("this may be because ..."), and list its id in explanation_ids. Never present it as a finding.
6. Use every measure ("on task", "off task", "facing the front", "phone", "seated", ...) only as the glossary defines it:
   these are observed postures and objects. Do not write about attention, engagement, boredom, understanding,
   motivation, confusion, sleep, tiredness or distraction as if they were measured. Never judge the instructor as a
   person.
7. Write plain English for an instructor who is not a statistician, and speak to them as "you" ("while you faced the
   class"): 2 or 3 short sentences per point. No jargon: not "window", "observable", "p-value", "q-value" or
   "Bonferroni"; say "a few seconds later", "students in view" and "could easily be chance".
8. Write at most 3 positive and 3 negative points. If the facts support fewer, write fewer; never pad. When several
   facts point the same way, combine them into one stronger point instead of repeating. When facts point in different
   directions, say so honestly. Put the most useful points first, preferring ones the instructor can act on.
9. If data_quality or a fact's notes warn that a result may be unreliable, say so in every point that rests on it.
10. indicator_validation says whether a person has checked each detection against the video. Where an indicator has
    not been checked (checked_against_human_labels is false), say the detection has not been verified. Where it has
    been checked and its precision is below 0.7, say it is unreliable. Where it has been checked and is higher, you
    may say it was checked, but still do not call a single detection certain.
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
        "positive_points": {"type": "array", "items": POINT_SCHEMA},
        "negative_points": {"type": "array", "items": POINT_SCHEMA},
    },
    "required": ["positive_points", "negative_points"],
    "additionalProperties": False,
}

TIME_RE = re.compile(r"\b\d{1,2}:\d{2}\b")
NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
CAUSAL_RE = re.compile(r"\b(caus(?:e|es|ed|ing)|led to|leads? to|leading to|result(?:s|ed|ing)? in|due to|as a result|"
                       r"drove|triggered|made (?:the )?students|makes (?:the )?students|proves?|proved)\b", re.I)
INTERNAL_STATE_RE = re.compile(r"\b(attention|attentive|inattentive|engag\w*|bored\w*|understand\w*|understood|motivat\w*|"
                               r"confus\w*|enjoy\w*|asleep|sleep\w*|drows\w*|tired\w*|fatigue\w*|distract\w*)\b", re.I)
JUDGEMENT_RE = re.compile(r"\b(good|bad|poor|great|terrible|ineffective|effective|excellent)\s+"
                          r"(teacher|instructor|teaching|lecturer)\b", re.I)
LEAK_RE = re.compile(r"\bF\d+\b|\bfacts?\b|\bE\d\b")
HEDGES = ("may", "might", "could", "possibly", "possible", "perhaps", "suggest", "tentative", "weak", "unclear", "chance",
          "uncertain", "not certain", "caution", "cannot tell", "can't tell", "not clear")
UNVERIFIED_RE = re.compile(r"unverified|not (?:been )?(?:verified|checked)|not yet (?:verified|checked)|unreliable|"
                           r"no one has checked|has not been checked|have not been checked", re.I)
# The detections each kind of fact stands on (keys of facts.json's indicator_validation); a point resting on one that
# no person has checked must say so.
KIND_INDICATORS = {
    "comparison": ("student_on_task_state", "instructor_role"),
    "activity_comparison": ("student_on_task_state", "instructor_role"),
    "instructor_behaviour_share": ("instructor_role",),
    "highest_on_task_stretch": ("student_on_task_state",),
    "lowest_on_task_stretch": ("student_on_task_state",),
    "lesson_comparison": ("student_on_task_state",),
    "hand_raises": ("hand_raise",),
    "no_participation_stretch": ("hand_raise",),
    "phone_use": ("phone_use",),
}
NUMBER_WORDS = {"two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
                "ten": "10", "eleven": "11", "twelve": "12"}
NO_POSITIVE = "No clear positive point was found in this analysis."
NO_NEGATIVE = "No clear negative point was found in this analysis."


# Problems the web app shows only as a code, never with the reason (user requests 2026-11-20):
KEY_PROBLEM = "01"       # the API key: missing, rejected, not allowed, or the sign-in could not be loaded
BILLING_PROBLEM = "02"   # the account behind the key: credit balance too low, billing, spend limit, on hold
BILLING_ERRORS = {"EnforcedSpendLimitReachedError", "OrganizationOnHoldError"}   # anthropic SDK 2.x classes
BILLING_WORDS = ("credit balance", "billing", "spend limit")


class RewriteUnavailable(Exception):
    """The rewrite could not run (no key, no network, a refused request ...); the message says why. `code` is set
    only for a problem with the API key (KEY_PROBLEM) or the account's billing (BILLING_PROBLEM)."""

    def __init__(self, message: str, code: str | None = None):
        super().__init__(message)
        self.code = code


def status_problem(error: Any) -> RewriteUnavailable:
    """Plain words, and a code for key and billing problems, for an error status returned by the Claude API."""
    status, kind = getattr(error, "status_code", None), getattr(error, "type", None)
    message = str(getattr(error, "message", None) or error)
    if status == 401 or kind == "authentication_error":
        return RewriteUnavailable("Claude rejected the API key.", KEY_PROBLEM)
    if status == 402 or kind == "billing_error" or type(error).__name__ in BILLING_ERRORS or \
            any(word in message.lower() for word in BILLING_WORDS):
        return RewriteUnavailable(f"A billing problem with the API key's account: {message}", BILLING_PROBLEM)
    if status == 403 or kind == "permission_error":
        return RewriteUnavailable(f"The API key may not make this request: {message}", KEY_PROBLEM)
    if status == 429:
        return RewriteUnavailable("Claude is rate-limiting requests; try again in a minute.")
    if status == 400:
        return RewriteUnavailable(f"Claude rejected the request: {message}")
    return RewriteUnavailable(f"Claude API error {status}: {message}")


def unverified(fact: dict[str, Any], facts: dict[str, Any]) -> list[str]:
    """The indicators under a fact that no person has checked, or that were checked and found unreliable."""
    shaky = []
    for name in fact.get("indicators") or KIND_INDICATORS.get(fact.get("kind"), ()):
        entry = facts.get("indicator_validation", {}).get(name, {})
        precision = entry.get("precision", entry.get("accuracy"))
        if not entry.get("checked_against_human_labels") or (precision is not None and precision < MIN_PRECISION):
            shaky.append(name)
    return shaky


def numbers_in(text: str) -> set[str]:
    """The numbers and mm:ss times in a text (a time counts as one token, not two numbers)."""
    times = set(TIME_RE.findall(text))
    return times | set(NUMBER_RE.findall(TIME_RE.sub(" ", text)))


def build_user_message(facts: dict[str, Any]) -> str:
    """The facts file minus the report note (which is printed verbatim, not rewritten)."""
    shown = {k: v for k, v in facts.items() if k != "report_note"}
    return ("Write the points for this lecture from the facts below.\n\n<facts>\n"
            + json.dumps(shown, indent=2, sort_keys=True) + "\n</facts>")


def validate(output: dict[str, Any], facts: dict[str, Any]) -> list[str]:
    """The problems with an answer; an empty list means it passed every check."""
    problems = []
    by_id = {f["id"]: f for f in facts["facts"]}
    explanations = facts.get("alternative_explanations", {})
    lecture_numbers = numbers_in(json.dumps(facts.get("lecture", {})))
    for key, direction in (("positive_points", "favourable"), ("negative_points", "unfavourable")):
        points = output.get(key)
        if not isinstance(points, list):
            problems.append(f"{key} is missing or not a list.")
            continue
        if len(points) > MAX_POINTS:
            problems.append(f"{key} has {len(points)} points; the maximum is {MAX_POINTS}.")
        for n, point in enumerate(points, 1):
            where = f"{key} #{n}"
            text = point.get("text", "")
            ids = point.get("fact_ids", [])
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
            allowed = set(lecture_numbers)
            for fact in cited:
                allowed |= numbers_in(json.dumps(fact))
            for explanation_id in point.get("explanation_ids", []):
                allowed |= numbers_in(explanations.get(explanation_id, ""))
            invented = sorted(numbers_in(text) - allowed)
            if invented:
                problems.append(f"{where}: numbers not found in the cited facts: {invented}.")
            for word, digit in NUMBER_WORDS.items():
                if re.search(rf"\b{word}\b", text, re.I) and digit not in allowed:
                    problems.append(f"{where}: the word '{word}' is a number that is not in the cited facts; "
                                    "write numbers as digits from the facts.")
            for label, pattern in (("causal wording", CAUSAL_RE), ("a word about internal states", INTERNAL_STATE_RE),
                                   ("a judgement of the instructor", JUDGEMENT_RE),
                                   ("a fact id or the word 'fact'", LEAK_RE)):
                found = pattern.search(text)
                if found:
                    problems.append(f"{where}: contains {label} ('{found.group(0)}').")
            for sentence in re.split(r"(?<=[.!?])\s+", text):
                if re.search(r"\bbecause\b", sentence, re.I) and not any(h in sentence.lower() for h in HEDGES):
                    problems.append(f"{where}: says 'because' without marking it as a possibility.")
            if any(f.get("evidence", {}).get("strength") in ("weak", "unknown") for f in cited) \
                    and not any(h in text.lower() for h in HEDGES):
                problems.append(f"{where}: rests on a weak finding but does not say so.")
            shaky = sorted({name for f in cited for name in unverified(f, facts)})
            if shaky and not UNVERIFIED_RE.search(text):
                problems.append(f"{where}: rests on detections that are unverified or unreliable "
                                f"({', '.join(shaky)}) but does not say so.")
    return problems


def render(output: dict[str, Any], facts: dict[str, Any]) -> str:
    """The report text, in the same layout as the rule-based report."""
    def block(points: list[dict[str, Any]], empty: str) -> str:
        return "\n".join(f"- {p['text'].strip()}" for p in points) if points else f"- {empty}"
    return (f"positive points:\n\n{block(output['positive_points'], NO_POSITIVE)}\n\n"
            f"negative points:\n\n{block(output['negative_points'], NO_NEGATIVE)}\n\n"
            f"{facts.get('report_note', '')}\n")


def call_claude(system: str, user: str, model: str = DEFAULT_MODEL) -> tuple[Any, Any, str | None]:
    """One request. Returns (answer or None, response, problem or None); raises RewriteUnavailable when the request
    could not be made or was rejected, with the reason in plain words."""
    try:
        import anthropic
    except ImportError as error:
        raise RewriteUnavailable("The anthropic package is not installed on this computer.") from error
    model_id, _, _, fallbacks = MODELS[model]
    request = dict(model=model_id, max_tokens=16000, system=system, messages=[{"role": "user", "content": user}],
                   output_config={"effort": EFFORT, "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}})
    try:
        client = anthropic.Anthropic()   # inside: a sign-in profile that cannot be read fails here (CredentialsError)
        if fallbacks:   # if a safety classifier declines, retry on Anthropic's recommended model
            response = client.beta.messages.create(betas=["server-side-fallback-2026-07-01"], fallbacks="default",
                                                   **request)
        else:           # Haiku 5.5 has no server-side fallback
            response = client.messages.create(**request)
    except TypeError as error:   # raised locally, before any network traffic, when no credentials are set
        if "authentication method" in str(error):
            raise RewriteUnavailable("No Claude API key is set on this computer (ANTHROPIC_API_KEY). "
                                     "Nothing was sent.", KEY_PROBLEM) from error
        raise
    except getattr(anthropic, "CredentialsError", ()) as error:   # SDK 2.x: a sign-in profile that cannot be read
        raise RewriteUnavailable(f"The Claude sign-in on this computer could not be loaded: {error}. Nothing was "
                                 "sent.", KEY_PROBLEM) from error
    except anthropic.APIStatusError as error:   # 401/403 key, 402/credit/spend limit billing, 429, 400, others
        raise status_problem(error) from error
    except anthropic.APIConnectionError as error:
        raise RewriteUnavailable("Could not reach the Claude API (no internet connection?).") from error
    if response.stop_reason == "refusal":
        return None, response, "The model declined to answer."
    if response.stop_reason == "max_tokens":
        return None, response, "The answer was cut off."
    text = next((b.text for b in response.content if b.type == "text"), "")
    try:
        return json.loads(text), response, None
    except json.JSONDecodeError:
        return None, response, "The answer was not valid JSON."


def cache_key(facts: dict[str, Any], model: str) -> str:
    text = MODELS[model][0] + PROMPT_VERSION + SYSTEM_PROMPT + build_user_message(facts)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def rewrite(facts: dict[str, Any], folder: Path, *, model: str = DEFAULT_MODEL, fake_response: dict | None = None,
            use_cache: bool = True, caller=call_claude) -> dict[str, Any]:
    """Has Claude write the points and checks them. Returns {"source": "llm" | "none", "reason", "text", ...}:
    with "llm" the checked text is in "text"; with "none" no feedback was written and "reason" says why.
    Writes llm_output.json (every attempt, its cost and problems) and caches a passing answer by its input."""
    folder = Path(folder)
    key = cache_key(facts, model)
    cache_path = folder / "llm_cache" / f"{key}.json"
    record: dict[str, Any] = {"model": MODELS[model][0], "prompt_version": PROMPT_VERSION, "facts_hash": key,
                              "attempts": []}
    output, problems, reason, code = None, [], None, None
    user = build_user_message(facts)
    if fake_response is not None:
        output = fake_response
        problems = validate(output, facts)
        record["attempts"].append({"source": "fake", "problems": problems})
    elif use_cache and cache_path.exists():
        record = json.loads(cache_path.read_text(encoding="utf-8"))
        output, problems = record["output"], []
    else:
        _, price_in, price_out, _ = MODELS[model]
        extra = ""
        try:
            for _ in range(MAX_ATTEMPTS):
                output, response, error = caller(SYSTEM_PROMPT, user + extra, model)
                usage = response.usage
                problems = [error] if error else validate(output, facts)
                record["attempts"].append({
                    "request_id": getattr(response, "_request_id", None), "served_by": getattr(response, "model", None),
                    "input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens,
                    "cost_usd": round(usage.input_tokens * price_in / 1e6 + usage.output_tokens * price_out / 1e6, 4),
                    "problems": problems})
                if not problems:
                    break
                extra = "\n\nYour previous answer failed these checks. Fix them and answer again:\n- " + \
                        "\n- ".join(problems)
        except RewriteUnavailable as error:
            reason, code = str(error), error.code
    record["output"] = output
    if reason is None and (problems or output is None):
        reason = (f"Claude's text failed the automatic checks {len(record['attempts'])} time(s), so it was not "
                  "used. First problem: " + (problems[0] if problems else "no answer."))
    record["result"] = {"source": "none" if reason else "llm", "reason": reason, "code": code}
    (folder / "llm_output.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    if reason:
        return {"source": "none", "reason": reason, "code": code, "problems": problems}
    if fake_response is None and not cache_path.exists():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return {"source": "llm", "reason": None, "code": None, "text": render(output, facts)}
