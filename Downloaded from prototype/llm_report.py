"""Turns facts.json into the instructor-facing positive / negative points with Claude, checks the text, and keeps the
rule-based report when the check fails.

The model never sees the video, the raw tables or any name. It only gets facts.json: numbers the pipeline already
computed, each tagged favourable / unfavourable with an evidence strength. Everything it writes is then checked by
code (every number must come from a cited fact, no causal wording, right direction, hedged when the evidence is weak).

Usage (from the repository root):
    python3 "Downloaded from prototype/llm_report.py" --dry-run              # show the prompt, call nothing
    python3 "Downloaded from prototype/llm_report.py" --fake-response r.json # test the checks on a saved answer
    python3 "Downloaded from prototype/llm_report.py"                        # real call (needs the anthropic package + credentials)
"""
import argparse
import glob
import hashlib
import json
import os
import re
import shutil
import sys

# alias -> (model id, input $/M tokens, output $/M tokens, supports server-side fallbacks). Prices are Anthropic's list prices
# (platform.claude.com/docs/en/about-claude/pricing); Haiku 5.5 has no server-side fallback, so none may be sent to it.
MODELS = {
    "opus": ("claude-opus-5-5", 4.00, 20.00, True),
    "sonnet": ("claude-sonnet-5-5", 2.00, 10.00, True),
    "haiku": ("claude-haiku-5-5", 0.10, 0.50, False),
}
EFFORT = "medium"                  # the default for Opus 5.5 and Haiku 5.5; set explicitly. Adaptive thinking cannot be turned off on Opus 5.5.


def use_model(alias):
    """Selects the model used for the call, its prices, and whether the fallback option is sent."""
    global MODEL, PRICE_IN, PRICE_OUT, FALLBACKS
    MODEL, PRICE_IN, PRICE_OUT, FALLBACKS = MODELS[alias]


use_model("opus")
PROMPT_VERSION = "1"
MAX_ATTEMPTS = 2                   # the second attempt is told which checks the first one failed
MAX_POINTS = 3
MAX_WORDS = 80

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
   "due to" or "as a result". Say "while", "compared with", "in the windows where".
4. Match your wording to each fact's evidence.strength. "weak" means the gap could easily be chance, so say that
   plainly. "descriptive" means a single stretch or a count, not a test, so do not present it as a pattern. Never present
   a weak finding as a clear one.
5. Where it helps, add one short sentence with a possible explanation taken from alternative_explanations, clearly marked
   as a possibility ("this may be because ..."), and list its id in explanation_ids. Never present it as a finding.
6. Use "focused" only as the glossary defines it (head up, a posture proxy). Do not write about attention, engagement,
   boredom, understanding, motivation or confusion as if they were measured, and never judge the instructor as a person.
7. Write plain English for an instructor who is not a statistician: 2 or 3 sentences per point, no jargon such as
   "p-value" or "Bonferroni" (say "could easily be chance" instead).
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
                               r"confus\w*|enjoy\w*)\b", re.I)
JUDGEMENT_RE = re.compile(r"\b(good|bad|poor|great|terrible|ineffective|effective|excellent)\s+(teacher|instructor|teaching|lecturer)\b", re.I)
LEAK_RE = re.compile(r"\bF\d+\b|\bfacts?\b|\bE\d\b")
HEDGES = ("may", "might", "could", "possibly", "possible", "perhaps", "suggest", "tentative", "weak", "unclear", "chance",
          "uncertain", "not certain", "caution", "cannot tell", "can't tell", "not clear")
# which detections each kind of fact stands on; used to demand a warning when a detection has not been verified
KIND_INDICATORS = {
    "comparison": ("head_up_posture", "instructor_role"), "highest_focus_stretch": ("head_up_posture",),
    "lowest_focus_stretch": ("head_up_posture",), "mostly_focused_windows": ("head_up_posture",),
    "hand_raises": ("hand_raise", "instructor_role"), "no_participation_stretch": ("hand_raise",), "phone_use": ("phone_use",),
}
MIN_PRECISION = 0.7


def unverified(fact, facts):
    """The indicators under a fact that no person has checked, or that were checked and found unreliable."""
    shaky = []
    for name in KIND_INDICATORS.get(fact.get("kind"), ()):
        entry = facts.get("indicator_validation", {}).get(name, {})
        precision = entry.get("precision", entry.get("accuracy"))
        if not entry.get("checked_against_human_labels") or (precision is not None and precision < MIN_PRECISION):
            shaky.append(name)
    return shaky


NUMBER_WORDS = {"two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
                "ten": "10", "eleven": "11", "twelve": "12"}


def numbers_in(text):
    """The numbers and mm:ss times in a piece of text (times count as one token, not as two numbers)."""
    times = set(TIME_RE.findall(text))
    return times | set(NUMBER_RE.findall(TIME_RE.sub(" ", text)))


def build_user_message(facts):
    """The facts file minus the report note (which is printed verbatim, not rewritten)."""
    shown = {k: v for k, v in facts.items() if k != "report_note"}
    return "Write the points for this lecture from the facts below.\n\n<facts>\n" + json.dumps(shown, indent=2, sort_keys=True) + "\n</facts>"


def validate(output, facts):
    """Returns a list of problems; an empty list means the answer passed every check."""
    problems = []
    by_id = {f["id"]: f for f in facts["facts"]}
    explanations = set(facts.get("alternative_explanations", {}))
    lecture_numbers = numbers_in(json.dumps(facts["lecture"]))
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
                allowed |= numbers_in(facts.get("alternative_explanations", {}).get(explanation_id, ""))
            invented = sorted(numbers_in(text) - allowed)
            if invented:
                problems.append(f"{where}: numbers not found in the cited facts: {invented}.")
            for word, digit in NUMBER_WORDS.items():
                if re.search(rf"\b{word}\b", text, re.I) and digit not in allowed:
                    problems.append(f"{where}: the word '{word}' is a number that is not in the cited facts; write numbers as digits from the facts.")

            for label, pattern in (("causal wording", CAUSAL_RE), ("a word about internal states", INTERNAL_STATE_RE),
                                   ("a judgement of the instructor", JUDGEMENT_RE), ("a fact id or the word 'fact'", LEAK_RE)):
                found = pattern.search(text)
                if found:
                    problems.append(f"{where}: contains {label} ('{found.group(0)}').")
            for sentence in re.split(r"(?<=[.!?])\s+", text):
                if re.search(r"\bbecause\b", sentence, re.I) and not any(h in sentence.lower() for h in HEDGES):
                    problems.append(f"{where}: says 'because' without marking it as a possibility.")
            if any(f.get("evidence", {}).get("strength") in ("weak", "unknown") for f in cited) and not any(h in text.lower() for h in HEDGES):
                problems.append(f"{where}: rests on a weak finding but does not say so.")
            shaky = sorted({name for f in cited for name in unverified(f, facts)})
            if shaky and not re.search(r"unverified|not (?:been )?(?:verified|checked)|not yet (?:verified|checked)|unreliable|"
                                       r"no one has checked|has not been checked|have not been checked", text, re.I):
                problems.append(f"{where}: rests on detections that are unverified or unreliable ({', '.join(shaky)}) but does not say so.")
    return problems


def render(output, facts):
    """The final report text, in the same layout as the rule-based report."""
    def block(points, empty):
        return "\n".join(f"- {p['text'].strip()}" for p in points) if points else f"- {empty}"
    return (f"positive points:\n\n{block(output['positive_points'], 'No clear positive point was found in this analysis.')}\n\n"
            f"negative points:\n\n{block(output['negative_points'], 'No clear negative point was found in this analysis.')}\n\n"
            f"{facts.get('report_note', '')}\n")


def call_claude(system, user):
    """One request to Claude. Imported here so the offline modes work without the anthropic package."""
    import anthropic
    client = anthropic.Anthropic()
    request = dict(model=MODEL, max_tokens=16000, system=system, messages=[{"role": "user", "content": user}],
                   output_config={"effort": EFFORT, "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}})
    try:
        if FALLBACKS:   # if a safety classifier declines, retry on Anthropic's recommended model
            response = client.beta.messages.create(betas=["server-side-fallback-2026-07-01"], fallbacks="default", **request)
        else:           # Haiku 5.5 has no server-side fallback
            response = client.messages.create(**request)
    except TypeError as error:   # the SDK raises this locally, before any network traffic, when no credentials are set
        if "authentication method" in str(error):
            sys.exit("No Claude credentials found. In the terminal run `export ANTHROPIC_API_KEY=...` (or `ant auth login`), "
                     "then run this again. Nothing was sent.")
        raise
    except anthropic.AuthenticationError:
        sys.exit("Claude rejected the credentials. Check ANTHROPIC_API_KEY (or run `ant auth login`).")
    except anthropic.RateLimitError as error:
        sys.exit(f"Rate limited. Retry after {error.response.headers.get('retry-after', '60')} s.")
    except anthropic.BadRequestError as error:
        sys.exit(f"Claude rejected the request: {error.message}")
    except anthropic.APIStatusError as error:
        sys.exit(f"Claude API error {error.status_code}: {error.message}")
    except anthropic.APIConnectionError:
        sys.exit("Could not reach the Claude API. Check the network connection.")
    if response.stop_reason == "refusal":
        return None, response, f"The model declined (stop_details: {getattr(response, 'stop_details', None)})."
    if response.stop_reason == "max_tokens":
        return None, response, "The answer was cut off at max_tokens."
    text = next((b.text for b in response.content if b.type == "text"), "")
    try:
        return json.loads(text), response, None
    except json.JSONDecodeError:
        return None, response, "The answer was not valid JSON."


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--facts", help="path to facts.json (default: the most recently analysed lecture)")
    parser.add_argument("--session", help="lecture ID: uses classroom_project/sessions/<ID>/reports/facts.json")
    parser.add_argument("--dry-run", action="store_true", help="print the prompt size and write a preview; call nothing")
    parser.add_argument("--fake-response", help="use this saved JSON answer instead of calling Claude (tests the checks)")
    parser.add_argument("--no-cache", action="store_true", help="ignore a saved answer for identical facts")
    parser.add_argument("--model", choices=sorted(MODELS), default="opus",
                        help="opus (default, about 4 cents per report), sonnet (about 2 cents) or haiku (about 0.1 cent)")
    args = parser.parse_args()
    use_model(args.model)
    if args.session:
        args.facts = os.path.join("classroom_project", "sessions", args.session, "reports", "facts.json")
    elif args.facts is None:   # no choice given: the most recently analysed lecture
        found = sorted(glob.glob(os.path.join("classroom_project", "sessions", "*", "reports", "facts.json")), key=os.path.getmtime)
        if not found:
            sys.exit("No facts.json found. Run the pipeline first, or pass --facts / --session.")
        args.facts = found[-1]
        print(f"Using the most recent lecture: {args.facts}")

    with open(args.facts, encoding="utf-8") as f:
        facts = json.load(f)
    report_dir = os.path.dirname(os.path.abspath(args.facts))
    # a test with a fake answer must never overwrite the real report
    report_path = os.path.join(report_dir, "final_feedback_report_FAKE_TEST.txt" if args.fake_response else "final_feedback_report.txt")
    user = build_user_message(facts)

    if args.dry_run:
        approx = (len(SYSTEM_PROMPT) + len(user)) // 4
        preview = os.path.join(report_dir, "llm_prompt_preview.txt")
        with open(preview, "w", encoding="utf-8") as f:
            f.write(SYSTEM_PROMPT + "\n\n---\n\n" + user)
        print(f"Dry run: about {approx} input tokens ({len(facts['facts'])} facts). Prompt preview: {preview}")
        print(f"Estimated cost of one successful call at list price: about ${approx * PRICE_IN / 1e6 + 1500 * PRICE_OUT / 1e6:.3f} "
              "(assuming ~1500 output tokens including thinking).")
        return

    key = hashlib.sha256((MODEL + PROMPT_VERSION + SYSTEM_PROMPT + user).encode("utf-8")).hexdigest()[:16]
    cache_path = os.path.join(report_dir, "llm_cache", f"{key}.json")
    record = {"model": MODEL, "prompt_version": PROMPT_VERSION, "facts_hash": key, "attempts": []}
    output = None

    if args.fake_response:
        with open(args.fake_response, encoding="utf-8") as f:
            output = json.load(f)
        problems = validate(output, facts)
        record["attempts"].append({"source": "fake", "problems": problems})
    elif os.path.exists(cache_path) and not args.no_cache:
        with open(cache_path, encoding="utf-8") as f:
            cached = json.load(f)
        output, problems = cached["output"], []
        print(f"Using the saved answer for identical facts ({cache_path}); no API call made.")
        record = cached
    else:
        feedback = ""
        for attempt in range(1, MAX_ATTEMPTS + 1):
            output, response, error = call_claude(SYSTEM_PROMPT, user + feedback)
            usage = response.usage
            cost = usage.input_tokens * PRICE_IN / 1e6 + usage.output_tokens * PRICE_OUT / 1e6
            problems = [error] if error else validate(output, facts)
            record["attempts"].append({"request_id": getattr(response, "_request_id", None), "served_by": response.model,
                                       "input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens,
                                       "cost_usd": round(cost, 4), "problems": problems})
            print(f"Attempt {attempt}: {usage.input_tokens} in / {usage.output_tokens} out tokens, about ${cost:.3f}; "
                  f"{'passed every check' if not problems else str(len(problems)) + ' problem(s)'}.")
            if not problems:
                break
            feedback = ("\n\nYour previous answer failed these checks. Fix them and answer again:\n- " + "\n- ".join(problems))
        record["output"] = output

    record.setdefault("output", output)
    with open(os.path.join(report_dir, "llm_output.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)

    if problems or output is None:
        print("The language-model text did NOT pass the checks, so the rule-based report was left in place:")
        for problem in problems:
            print(f"  - {problem}")
        sys.exit(1)

    if not args.fake_response and not os.path.exists(cache_path):
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
    rules_copy = os.path.join(report_dir, "final_feedback_report_rules.txt")
    if os.path.exists(report_path) and not os.path.exists(rules_copy):
        shutil.copyfile(report_path, rules_copy)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(render(output, facts))
    print(f"Report written to {report_path} (rule-based version kept as final_feedback_report_rules.txt).")


if __name__ == "__main__":
    main()
