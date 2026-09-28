import json
import random
import re
import string
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

from clients import ask_jev, chat, cost_usd, parse_json

PER_BATCH = 20
FRAUD_FIELDS = (
    "amount_band", "merchant_category", "channel", "time_of_day", "account_age", "velocity_1h",
    "device", "geo_match", "prior_chargebacks", "billing_shipping", "customer_note",
)
# Jev reads state as text, so "42" or "$1,200" is as bad as 42; bands like "micro (<$5)" pass.
_NUMERIC_ONLY = re.compile(r"[\s\d.,$%+\-]*")


def _classes(cfg: dict) -> dict[str, str]:
    return cfg["questions"][0]["criteria"]


def _class_lines(classes: dict[str, str]) -> str:
    return "\n".join(f"- {name}: {desc}" for name, desc in classes.items())


def _fraud_prompt(label: str, classes: dict[str, str]) -> str:
    example = {f: "..." for f in FRAUD_FIELDS}
    return f"""You are building a test set for a card-payments fraud classifier.
The classes are:
{_class_lines(classes)}

Write {PER_BATCH} card transactions whose correct class is "{label}".
Each record has "label" and a "state" object with exactly these string fields: {", ".join(FRAUD_FIELDS)}.
Every value must be words or a named band, e.g. "micro (<$5)", "under 30 days", "many (10+)". Never a bare number.
Make them varied and realistic. Include some deliberately ambiguous cases that share signals with other classes, not only textbook examples.
Reply with JSON only: {{"records": [{{"label": "{label}", "state": {json.dumps(example)}}}, ...]}}"""


def _routing_prompt(tier: str, classes: dict[str, str]) -> str:
    return f"""You are building a test set for a router that picks the cheapest model tier able to answer a prompt well.
The tiers are:
{_class_lines(classes)}

Write {PER_BATCH} user prompts that genuinely need the "{tier}" tier: a cheaper tier would do them badly and a dearer one adds nothing.
Vary the topic, length and phrasing. Each record has "tier", "prompt", and "reference": a concise correct answer a grader can compare a candidate answer against.
Reply with JSON only: {{"records": [{{"tier": "{tier}", "prompt": "...", "reference": "..."}}, ...]}}"""


def _fraud_record(r, label: str) -> dict | None:
    if not isinstance(r, dict) or r.get("label") != label:
        return None
    state = r.get("state")
    if not isinstance(state, dict) or set(state) != set(FRAUD_FIELDS):
        return None
    if any(not isinstance(v, str) or _NUMERIC_ONLY.fullmatch(v) for v in state.values()):
        return None
    return {"label": label, "state": {f: state[f] for f in FRAUD_FIELDS}}


def _routing_record(r, tier: str) -> dict | None:
    if not isinstance(r, dict) or r.get("tier") != tier:
        return None
    if not all(isinstance(r.get(k), str) and r[k].strip() for k in ("prompt", "reference")):
        return None
    return {"prompt": r["prompt"], "tier": tier, "reference": r["reference"]}


def _generate(cfg: dict, prompt_fn, record_fn, prefix: str) -> list[dict]:
    classes = _classes(cfg)

    def batch(name: str) -> tuple[str, dict]:
        return name, chat(cfg["generator_model"], prompt_fn(name, classes))

    kept, dropped = [], 0
    with ThreadPoolExecutor(max_workers=cfg.get("workers", 4)) as pool:
        for name, resp in pool.map(batch, classes):
            data = parse_json(resp["text"])
            raw = data.get("records") if isinstance(data, dict) else None
            if not isinstance(raw, list):
                print(f"{name}: batch dropped: {resp['error'] or 'reply is not {records: [...]} JSON'}")
                dropped += PER_BATCH
                continue
            good = [rec for rec in (record_fn(r, name) for r in raw) if rec]
            kept += good
            dropped += len(raw) - len(good)

    records = [{"id": f"{prefix}-{i:03d}", **rec} for i, rec in enumerate(kept, 1)]
    path = Path(cfg["dataset"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    print(f"kept {len(records)}, dropped {dropped}")
    return records


def _refund_prompt(reason: str, classes: dict[str, str]) -> str:
    return f"""You are building a test set for a classifier that reads a customer's refund request and decides why they want a refund.
The reasons are:
{_class_lines(classes)}

Write {PER_BATCH} short customer messages (1-3 sentences each) whose real reason is "{reason}".
Make them realistic and varied in tone and product, and include tricky ones: negations ("it's not broken, I just..."), sarcasm,
mentioning a different issue that is not the real reason, and the reason implied rather than stated. A careful human must still be able to tell the real reason.
Never mention dates, days, weeks, or any timing (no "yesterday", "last month", "arrived a while ago").
Reply with JSON only: {{"records": [{{"reason": "{reason}", "message": "..."}}, ...]}}"""


# Seeded so a regenerated dataset gets the same dates in the same order.
_REFUND_RNG = random.Random(7)
# Half the rows land within two days of a 14- or 30-day window, where date arithmetic matters most.
_NEAR_BOUNDARY_DAYS = [12, 13, 14, 15, 16, 28, 29, 30, 31, 32]


def _refund_record(r, reason: str, windows: dict[str, int]) -> dict | None:
    if not isinstance(r, dict) or r.get("reason") != reason:
        return None
    if not isinstance(r.get("message"), str) or not r["message"].strip():
        return None
    rng = _REFUND_RNG
    days = rng.choice(_NEAR_BOUNDARY_DAYS) if rng.random() < 0.5 else rng.randint(1, 45)
    delivered = date(2026, 8, 1) + timedelta(rng.randint(0, 20))
    requested = delivered + timedelta(days)
    return {"message": r["message"], "reason": reason, "delivered_on": delivered.isoformat(),
            "requested_on": requested.isoformat(), "days": days, "eligible": days <= windows[reason]}


def generate_refund(cfg: dict) -> list[dict]:
    return _generate(cfg, _refund_prompt, lambda r, reason: _refund_record(r, reason, cfg["windows"]), "rf")


def generate_fraud(cfg: dict) -> list[dict]:
    return _generate(cfg, _fraud_prompt, _fraud_record, "f")


def generate_routing(cfg: dict) -> list[dict]:
    return _generate(cfg, _routing_prompt, _routing_record, "r")


ANSWER_FIELD = {"choice": "choice", "score": "score", "noul": "noul"}
ARM_TEXT_LIMIT = 2000


def _run(records: list[dict], cfg: dict, run_dir: Path, record_fn) -> list[dict]:
    # Workers only call models; the main thread is the sole writer of raw.jsonl.
    rows = []
    with open(Path(run_dir) / "raw.jsonl", "w") as f, ThreadPoolExecutor(max_workers=cfg.get("workers", 4)) as pool:
        for fut in as_completed([pool.submit(record_fn, r, cfg) for r in records]):
            row = fut.result()
            f.write(json.dumps(row) + "\n")
            f.flush()
            rows.append(row)
    return rows


def _row_error(row: dict, sources: tuple[str, ...]) -> str | None:
    return next((f"{s}: {row[s]['error']}" for s in sources if row[s]["error"]), None)


def _sum(values) -> float | int | None:
    values = list(values)
    return None if any(v is None for v in values) else sum(values)


def _claude(model: str, resp: dict, cfg: dict) -> dict:
    return {
        "latency_ms": resp["latency_ms"],
        "input_tokens": resp["input_tokens"],
        "output_tokens": resp["output_tokens"],
        "cost_usd": cost_usd(model, resp["input_tokens"], resp["output_tokens"], cfg["prices"]),
        "error": resp["error"],
    }


def _jev_primary(record_state: dict, cfg: dict) -> tuple[dict, dict | None]:
    """Returns (ask_jev response, primary answer or None), with a missing primary as an error."""
    resp = ask_jev(record_state, cfg["questions"], cfg["jev_model"])
    primary = resp["answers"].get(cfg["questions"][0]["id"])
    if not resp["error"] and not (primary and primary.get("choice")):
        resp = {**resp, "error": f"missing answer: {cfg['questions'][0]['id']}"}
    return resp, (None if resp["error"] else primary)


def _fraud_row(record: dict, cfg: dict) -> dict:
    resp, primary = _jev_primary(record["state"], cfg)
    jev = {
        "label": primary and primary["choice"],
        "confidence": primary and primary.get("confidence"),
        "probabilities": primary and primary.get("probabilities"),
    }
    for q in cfg["questions"][1:]:
        answer = resp["answers"].get(q["id"]) or {}
        jev[q["id"]] = None if resp["error"] else answer.get(ANSWER_FIELD[q["type"]])
    jev |= {"latency_ms": resp["latency_ms"], "cost_usd": resp["cost_usd"], "error": resp["error"]}
    return {"id": record["id"], "gold": record["label"], "jev": jev, "error": resp["error"] and f"jev: {resp['error']}"}


def run_fraud(records: list[dict], cfg: dict, run_dir: Path) -> list[dict]:
    return _run(records, cfg, run_dir, _fraud_row)


def _judge_prompt(prompt: str, reference: str, answer: str) -> str:
    return f"""Question: {prompt}
Reference answer: {reference}
Candidate answer: {answer}
Is the candidate correct and complete with respect to the reference? Minor wording differences are fine. Reply with one word: yes or no."""


def _verdict(resp: dict) -> tuple[bool | None, str | None]:
    if resp["error"]:
        return None, resp["error"]
    words = (resp["text"] or "").split()
    word = words[0].strip(string.punctuation).lower() if words else ""
    if word in ("yes", "no"):
        return word == "yes", None
    return None, f"not yes/no: {(resp['text'] or '')[:100]}"


def _arm(model: str | None, resp: dict | None, cfg: dict, error: str | None = None) -> dict:
    if resp is None:
        return {"model": model, "text": None, "correct": None, "latency_ms": None, "input_tokens": None,
                "output_tokens": None, "cost_usd": None, "error": error}
    return {"model": model, "text": resp["text"] and resp["text"][:ARM_TEXT_LIMIT], "correct": None,
            **_claude(model, resp, cfg)}


def _routing_row(record: dict, cfg: dict) -> dict:
    prompt = record["prompt"]
    resp, primary = _jev_primary({"prompt": prompt}, cfg)
    tier = primary and primary["choice"]
    if tier and tier not in cfg["tier_models"]:
        resp, tier = {**resp, "error": f"unknown tier: {tier}"}, None
    jev = {"tier": tier, "confidence": primary and primary.get("confidence"),
           "latency_ms": resp["latency_ms"], "cost_usd": resp["cost_usd"], "error": resp["error"]}

    texts = {}
    if tier:
        t = cfg["tier_models"][tier]
        a = chat(t["model"], prompt, t["extra"])
        arm_a = _arm(t["model"], a, cfg)
        # Arm A pays for routing: its latency and cost include the Jev call.
        arm_a["latency_ms"] = resp["latency_ms"] + a["latency_ms"]
        arm_a["cost_usd"] = _sum([resp["cost_usd"], arm_a["cost_usd"]])
        texts["arm_a"] = a
    else:
        arm_a = _arm(None, None, cfg, "no tier")
    b = chat(cfg["frontier_model"], prompt)
    arm_b = _arm(cfg["frontier_model"], b, cfg)
    texts["arm_b"] = b

    arms = {"arm_a": arm_a, "arm_b": arm_b}
    judge_calls, judge_error = [], None
    for name, r in texts.items():
        if r["error"]:
            continue
        j = chat(cfg["judge_model"], _judge_prompt(prompt, record["reference"], r["text"]))
        judge_calls.append(j)
        arms[name]["correct"], err = _verdict(j)
        judge_error = judge_error or (err and f"{name}: {err}")
    judge = {
        "input_tokens": _sum(j["input_tokens"] for j in judge_calls),
        "output_tokens": _sum(j["output_tokens"] for j in judge_calls),
        "cost_usd": _sum(_claude(cfg["judge_model"], j, cfg)["cost_usd"] for j in judge_calls),
        "error": judge_error,
    }

    row = {"id": record["id"], "gold": record["tier"], "jev": jev, "arm_a": arm_a, "arm_b": arm_b, "judge": judge}
    row["error"] = _row_error(row, ("jev", "arm_a", "arm_b", "judge"))
    return row


def run_routing(records: list[dict], cfg: dict, run_dir: Path) -> list[dict]:
    return _run(records, cfg, run_dir, _routing_row)


def _refund_noul(state: dict, q: dict, model: str) -> dict:
    n = ask_jev(state, [q], model)
    noul = (n["answers"].get(q["id"]) or {}).get("noul")
    if not n["error"] and noul is None:
        n = {**n, "error": f"missing answer: {q['id']}"}
    return {"noul": noul, "eligible": None if n["error"] else noul >= 0.5,
            "latency_ms": n["latency_ms"], "cost_usd": n["cost_usd"], "error": n["error"]}


def _refund_row(record: dict, cfg: dict) -> dict:
    windows, model = cfg["windows"], cfg["jev_model"]
    naive = _refund_noul({k: record[k] for k in ("message", "delivered_on", "requested_on")},
                         cfg["naive_question"], model)
    days = (date.fromisoformat(record["requested_on"]) - date.fromisoformat(record["delivered_on"])).days
    # Code does the date subtraction; Jev still applies the policy windows itself.
    days_approach = _refund_noul({"message": record["message"], "days_since_delivery": days},
                                 cfg["days_question"], model)

    s, primary = _jev_primary({"message": record["message"]}, cfg)
    reason = primary and primary["choice"]
    if reason and reason not in windows:
        s, reason = {**s, "error": f"unknown reason: {reason}"}, None
    # Jev only reads the reason; the date arithmetic stays in code.
    split = {"reason": reason, "confidence": primary and primary.get("confidence"),
             "eligible": days <= windows[reason] if reason else None,
             "latency_ms": s["latency_ms"], "cost_usd": s["cost_usd"], "error": s["error"]}

    row = {"id": record["id"], "gold": record["eligible"], "reason": record["reason"], "days": record["days"],
           "window": windows[record["reason"]], "naive": naive, "days_approach": days_approach, "split": split}
    row["error"] = _row_error(row, ("naive", "days_approach", "split"))
    return row


def run_refund(records: list[dict], cfg: dict, run_dir: Path) -> list[dict]:
    return _run(records, cfg, run_dir, _refund_row)
