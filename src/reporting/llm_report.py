"""The LLM report stage: facts.json -> Claude -> checked positive / negative points.

Moved here from the pipeline_creation branch's `main code/llm_report.py` (2026-11-20) and adapted to the
lesson facts.json that src/reporting/facts.py writes (schema_version 1). The model never sees the video, the
tables or any name: it only gets facts.json, without the notes meant for the team. Everything it writes is checked
in code before it is kept (every number must come from a cited fact, the right direction, no causal or inner-state
wording, no promises, no talk about how the results were measured). One retry with the failures listed.

Writing style (user decision 2026-11-21, prompt version 5): plain words from a supportive teaching mentor, for
better teaching. A positive point says what went well and what to keep doing, with no warning or doubt; a negative
point says calmly what happened and offers one idea to try next time. No "not verified", "could easily be chance",
"simple shares, not tests" or "this may be because": those caveats stay in the lesson's files (facts.json,
report_note) for the team. A point resting on a detection that a person checked and found unreliable is left out.

Claude writes the feedback for every lecture (user decisions 2026-11-20: always on, and no rule-based text). After
the last attempt, points that still fail a check are left out and the points that passed are shown (user decision
2026-11-21). A failure never crashes the stage: when no point passed, `rewrite()` returns the reason (no API key, no
network, rejected text, ...), no feedback is written, and the web app shows the reason with a "Try again" button. A problem with the API key (missing,
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
PROMPT_VERSION = "5"         # 5: mentor style, no measurement caveats (2026-11-21); changes the cache key
MAX_ATTEMPTS = 2             # the second attempt is told which checks the first one failed
MAX_POINTS = 3
MAX_WORDS = 80
MIN_PRECISION = 0.7

SYSTEM_PROMPT = """\
You write the "positive points" and "negative points" of a feedback report for a university instructor who wants to
teach better. You do not see the video. A program has already analysed it and produced the JSON facts you are given.
Every number in your answer must come from those facts.

How to write:
1. Speak to the instructor as "you", like a supportive teaching mentor, in plain everyday English. Each point says
   what happened in this lecture (when, what you were doing, what the students were doing) and what it means for the
   next lecture. Use 2 or 3 short sentences (about 30 to 50 words) and only the one or two numbers that matter most.
   Example of the tone: "When you used hand gestures, more students were on task: X% compared with Y% at other
   times. Keep using gestures when you explain the main ideas."
2. Positive points are purely positive: say what went well and end with what to keep doing. Add no warning, doubt or
   "but".
3. Negative points are calm and constructive, never blaming: say plainly what happened, then offer one practical idea
   to try next time, as an option ("Next time, you could ..."). Where you can, base the idea on something that went
   better in this same lecture. Never promise that an idea will work.
4. Write only about the teaching and the class, never about how the results were measured or how certain they are: no
   words such as "detected", "verified", "chance", "not a test", "noisy" or "could not tell apart", and no possible
   reasons ("this may be because ..."). Do not use jargon such as "window", "observable", "p-value" or "Bonferroni";
   say "students in view", "a few seconds later", "between 04:00 and 04:30".

Rules your answer is checked against:
5. Use only the facts given. Copy numbers, percentages, counts and times exactly as written (digits, and mm:ss times).
   Never calculate a new number, round differently or convert units. Do not write numbers as words.
6. List the fact ids behind each point in fact_ids. A positive point may cite only facts whose direction is
   "favourable"; a negative point only facts whose direction is "unfavourable". Do not write ids, or the word "fact",
   in the text itself.
7. Say what happened together, never what caused what: do not use "caused", "led to", "resulted in", "due to",
   "because" or "as a result"; say "when", "while", "compared with". Describe this lecture only; do not turn it into a
   general rule about teaching.
8. Use every measure ("on task", "off task", "facing the front", "phone", "seated", ...) only as the glossary defines it.
   These are postures and objects, so never write about attention, engagement, boredom, understanding, motivation,
   confusion, sleep, tiredness or distraction, not even in your ideas for next time. Never judge the instructor as a
   person.
9. Write at most 3 positive and 3 negative points; if the facts support fewer, write fewer, and never pad. Keep one idea
   per point and combine facts that say the same thing. Put first the points the instructor can act on, especially
   what students did during your different activities, and prefer facts with a stronger evidence.strength.
"""

POINT_SCHEMA = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "fact_ids": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["text", "fact_ids"],
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
REASON_RE = re.compile(r"\bbecause\b", re.I)
# Talk about how the results were measured or how certain they are: kept for the team, never in the instructor's
# points (user decision 2026-11-21). "a chance to answer" and "share your ideas" stay allowed.
METHOD_RE = re.compile(r"\b(?:un)?verif\w*|\bdetect\w*|\b(?:be|by|pure|just) chance\b|\bnot (?:a |simple )?tests?\b|"
                       r"\bsimple shares?\b|\bstatistic\w*|\bsignifican\w*|\bnoisy\b|\bunreliable\b|\bunchecked\b|"
                       r"\b(?:could not|couldn't|cannot|can't) tell\b|\bnot (?:yet )?(?:been )?checked\b", re.I)
# An idea for next time is offered, never promised to work.
PROMISE_RE = re.compile(r"\bwill (?:help|increase|improve|raise|reduce|lower|boost|keep|bring|make|get)\b", re.I)
# The detections each kind of fact stands on (keys of facts.json's indicator_validation); a point resting on one that a
# person checked and found unreliable is left out.
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
# a side whose every point failed the checks (never claim that nothing was found when points were left out)
NOT_PASSED_POSITIVE = "No positive point passed the automatic checks this time."
NOT_PASSED_NEGATIVE = "No negative point passed the automatic checks this time."
SIDES = ("positive_points", "negative_points")


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


def unreliable(fact: dict[str, Any], facts: dict[str, Any]) -> list[str]:
    """The detections under a fact that a person checked against the video and found unreliable (precision below
    MIN_PRECISION). Detections nobody has checked yet are used as they are (user decision 2026-11-21)."""
    shaky = []
    for name in fact.get("indicators") or KIND_INDICATORS.get(fact.get("kind"), ()):
        entry = facts.get("indicator_validation", {}).get(name, {})
        precision = entry.get("precision", entry.get("accuracy"))
        if entry.get("checked_against_human_labels") and precision is not None and precision < MIN_PRECISION:
            shaky.append(name)
    return shaky


def numbers_in(text: str) -> set[str]:
    """The numbers and mm:ss times in a text (a time counts as one token, not two numbers)."""
    times = set(TIME_RE.findall(text))
    return times | set(NUMBER_RE.findall(TIME_RE.sub(" ", text)))


# Parts of facts.json meant for the team, not for the instructor's points: the report note (kept verbatim in the
# saved text), the measurement caveats and the possible explanations (user decision 2026-11-21).
TEAM_ONLY = ("report_note", "alternative_explanations", "data_quality", "indicator_validation")


def build_user_message(facts: dict[str, Any]) -> str:
    """The facts file without the team-only parts and without each fact's evidence notes."""
    shown = {k: v for k, v in facts.items() if k not in TEAM_ONLY}
    shown["facts"] = [{**fact, "evidence": {k: v for k, v in fact["evidence"].items() if k != "notes"}}
                      if isinstance(fact.get("evidence"), dict) else fact for fact in facts.get("facts", [])]
    return ("Write the points for this lecture from the facts below.\n\n<facts>\n"
            + json.dumps(shown, indent=2, sort_keys=True) + "\n</facts>")


def validate(output: dict[str, Any], facts: dict[str, Any]) -> list[str]:
    """The problems with an answer; an empty list means it passed every check."""
    problems = []
    by_id = {f["id"]: f for f in facts["facts"]}
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
            if len(text.split()) > MAX_WORDS:
                problems.append(f"{where}: {len(text.split())} words; the maximum is {MAX_WORDS}.")

            cited = [by_id[i] for i in ids if i in by_id]
            allowed = set(lecture_numbers)
            for fact in cited:
                allowed |= numbers_in(json.dumps(fact))
            invented = sorted(numbers_in(text) - allowed)
            if invented:
                problems.append(f"{where}: numbers not found in the cited facts: {invented}.")
            for word, digit in NUMBER_WORDS.items():
                if re.search(rf"\b{word}\b", text, re.I) and digit not in allowed:
                    problems.append(f"{where}: the word '{word}' is a number that is not in the cited facts; "
                                    "write numbers as digits from the facts.")
            for label, pattern in (("causal wording", CAUSAL_RE), ("a word about internal states", INTERNAL_STATE_RE),
                                   ("a judgement of the instructor", JUDGEMENT_RE),
                                   ("a fact id or the word 'fact'", LEAK_RE),
                                   ("a reason ('because'); say what happened, not why", REASON_RE),
                                   ("talk about how the results were measured or how certain they are; write only "
                                    "about the teaching and the class", METHOD_RE),
                                   ("a promise that an idea will work; offer it as an option", PROMISE_RE)):
                found = pattern.search(text)
                if found:
                    problems.append(f"{where}: contains {label} ('{found.group(0)}').")
            shaky = sorted({name for f in cited for name in unreliable(f, facts)})
            if shaky:
                problems.append(f"{where}: rests on detections a person checked and found unreliable "
                                f"({', '.join(shaky)}); leave this point out.")
    return problems


def passing_points(output: Any, facts: dict[str, Any]) -> tuple[dict[str, list], dict[str, int]]:
    """Only the points that pass every check on their own (at most MAX_POINTS a side), and how many were left out
    per side (user decision 2026-11-21: show the points that passed rather than nothing)."""
    kept: dict[str, list] = {side: [] for side in SIDES}
    left_out = {side: 0 for side in SIDES}
    for side in SIDES:
        points = output.get(side) if isinstance(output, dict) else None
        for point in points if isinstance(points, list) else []:
            alone = {side: [point], **{other: [] for other in SIDES if other != side}}
            if isinstance(point, dict) and not validate(alone, facts) and len(kept[side]) < MAX_POINTS:
                kept[side].append(point)
            else:
                left_out[side] += 1
    return kept, left_out


def render(output: dict[str, Any], facts: dict[str, Any], left_out: dict[str, int] | None = None) -> str:
    """The report text, in the same layout as the rule-based report. A side left empty because its points failed
    the checks says so instead of claiming that nothing was found."""
    left_out = left_out or {}

    def block(side: str, empty: str, not_passed: str) -> str:
        points = output[side]
        if points:
            return "\n".join(f"- {p['text'].strip()}" for p in points)
        return f"- {not_passed if left_out.get(side) else empty}"
    return (f"positive points:\n\n{block('positive_points', NO_POSITIVE, NOT_PASSED_POSITIVE)}\n\n"
            f"negative points:\n\n{block('negative_points', NO_NEGATIVE, NOT_PASSED_NEGATIVE)}\n\n"
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
    with "llm" the checked text is in "text" ("left_out" points that failed a check were dropped; every point shown
    passed); with "none" no point passed and "reason" says why. Writes llm_output.json (every attempt, its answer,
    cost and problems) and caches an answer whose every point passed."""
    folder = Path(folder)
    key = cache_key(facts, model)
    cache_path = folder / "llm_cache" / f"{key}.json"
    record: dict[str, Any] = {"model": MODELS[model][0], "prompt_version": PROMPT_VERSION, "facts_hash": key,
                              "attempts": []}
    output, problems, reason, code = None, [], None, None
    answers: list[tuple[int, Any]] = []        # (attempt number, answer) of every attempt that gave an answer
    user = build_user_message(facts)
    if fake_response is not None:
        output = fake_response
        problems = validate(output, facts)
        record["attempts"].append({"source": "fake", "problems": problems})
        answers.append((1, output))
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
                    "problems": problems, "answer": output})
                if output is not None:
                    answers.append((len(record["attempts"]), output))
                if not problems:
                    break
                extra = "\n\nYour previous answer failed these checks. Fix them and answer again:\n- " + \
                        "\n- ".join(problems)
        except RewriteUnavailable as error:
            reason, code = str(error), error.code
    record["output"] = output
    log_path = folder / "llm_output.json"
    if reason is None and output is not None and not problems:          # every point passed every check
        record["result"] = {"source": "llm", "reason": None, "code": None, "left_out": 0}
        log_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        if fake_response is None and not cache_path.exists():
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        return {"source": "llm", "reason": None, "code": None, "left_out": 0, "text": render(output, facts)}

    # Otherwise show the points that passed, from the attempt where most passed (the later one on a tie). Such a
    # partial answer is never cached, so "Try again" asks Claude again.
    best = None
    for number, answer in answers:
        kept, left_out = passing_points(answer, facts)
        count = sum(len(points) for points in kept.values())
        if count and (best is None or count >= best[0]):
            best = (count, number, kept, left_out)
    if best:
        _, number, kept, left_out = best
        dropped = sum(left_out.values())
        record["result"] = {"source": "llm", "reason": None, "code": None, "left_out": dropped,
                            "from_attempt": number, "shown": kept}
        log_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        return {"source": "llm", "reason": None, "code": None, "left_out": dropped,
                "text": render(kept, facts, left_out)}

    if reason is None:
        reason = (f"Claude's text failed the automatic checks {len(record['attempts'])} time(s), so it was not "
                  "used. First problem: " + (problems[0] if problems else "no answer."))
    record["result"] = {"source": "none", "reason": reason, "code": code}
    log_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return {"source": "none", "reason": reason, "code": code, "problems": problems}
