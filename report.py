import math

CLASSES = {
    "fraud": ["legitimate", "card_testing", "account_takeover", "friendly_fraud", "merchant_collusion"],
    "routing": ["simple", "medium", "complex", "reasoning"],
    "refund": ["defective", "wrong_item", "changed_mind"],
}
GATE_THRESHOLDS = (0.5, 0.6, 0.7, 0.8)
DASH = "–"


def fmt(x, spec=".2f", prefix=""):
    return DASH if x is None else f"{prefix}{x:{spec}}"


def ratio(num, den):
    return num / den if den else None


def table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def prf(gold: list, pred: list, classes: list[str]) -> dict:
    """{class: (precision, recall, f1, support)}; None where the denominator is zero."""
    out = {}
    for c in classes:
        tp = sum(g == c and p == c for g, p in zip(gold, pred))
        n_pred = sum(p == c for p in pred)
        n_gold = sum(g == c for g in gold)
        p, r = ratio(tp, n_pred), ratio(tp, n_gold)
        f1 = None if p is None or r is None else (2 * p * r / (p + r) if p + r else 0.0)
        out[c] = (p, r, f1, n_gold)
    return out


def accuracy(gold: list, pred: list):
    return ratio(sum(g == p for g, p in zip(gold, pred)), len(gold))


def percentile(values: list, q: float):
    """Nearest-rank percentile; None for an empty list."""
    xs = sorted(v for v in values if v is not None)
    if not xs:
        return None
    return xs[max(0, math.ceil(q / 100 * len(xs)) - 1)]


def threshold_split(conf: list, correct: list[bool], threshold: float) -> dict:
    """{"above": (accuracy, n), "below": (accuracy, n)}; rows with no confidence are skipped."""
    hi = [ok for c, ok in zip(conf, correct) if c is not None and c >= threshold]
    lo = [ok for c, ok in zip(conf, correct) if c is not None and c < threshold]
    return {"above": (ratio(sum(hi), len(hi)), len(hi)), "below": (ratio(sum(lo), len(lo)), len(lo))}


def cost_totals(rows: list[dict], source: str) -> tuple[float, int]:
    """(total USD with None counted as 0, number of missing costs)."""
    costs = [(r.get(source) or {}).get("cost_usd") for r in rows]
    return sum(c for c in costs if c is not None), sum(c is None for c in costs)


def confusion(gold: list, pred: list, classes: list[str]) -> dict:
    return {g: {p: sum(a == g and b == p for a, b in zip(gold, pred)) for p in classes} for g in classes}


def correct_rates(rows: list[dict], arm: str, classes: list[str]) -> dict:
    """{"overall" or tier: (rate, n)} over rows whose `correct` is not None."""
    scored = [(r["gold"], r[arm]["correct"]) for r in rows if r[arm].get("correct") is not None]
    out = {"overall": (ratio(sum(c for _, c in scored), len(scored)), len(scored))}
    for t in classes:
        sub = [c for g, c in scored if g == t]
        out[t] = (ratio(sum(sub), len(sub)), len(sub))
    return out


def _prf_section(title: str, gold: list, pred: list, classes: list[str]) -> str:
    body = [[c, fmt(p), fmt(r), fmt(f), n] for c, (p, r, f, n) in prf(gold, pred, classes).items()]
    return f"### {title}\n\n{table(['class', 'precision', 'recall', 'F1', 'support'], body)}\n\nAccuracy: {fmt(accuracy(gold, pred))} (n={len(gold)})"


def gated(rows: list[dict], threshold: float) -> tuple[int, float | None, float]:
    """(fallbacks, correct rate, total cost) when rows below `threshold` go to arm_b instead of arm_a.

    arm_a's cost already includes Jev; a fallback still paid for the Jev call, so it is added to arm_b's.
    """
    fallbacks, correct, cost = 0, [], 0.0
    for r in rows:
        conf = r["jev"].get("confidence")
        if conf is not None and conf >= threshold:
            arm = r["arm_a"]
        else:
            arm = r["arm_b"]
            fallbacks += 1
            cost += r["jev"].get("cost_usd") or 0
        cost += arm.get("cost_usd") or 0
        if arm.get("correct") is not None:
            correct.append(arm["correct"])
    return fallbacks, ratio(sum(correct), len(correct)), cost


def _errors(rows: list[dict]) -> str:
    errs = [r["error"] for r in rows if r.get("error") is not None]
    return f"## Errors\n\nError rows: {len(errs)}" + "".join(f"\n- {e}" for e in errs[:5])


def _rate(hits: list[bool]) -> str:
    return f"{fmt(ratio(sum(hits), len(hits)))} (n={len(hits)})"


def _refund_summary(rows: list[dict], threshold: float) -> str:
    ok = [r for r in rows if r.get("error") is None]
    gold = [r["gold"] for r in ok]
    near = [abs(r["days"] - r["window"]) <= 2 for r in ok]
    out = ["# refund summary", f"Rows: {len(rows)} (ok: {len(ok)}, errors: {len(rows) - len(ok)})"]

    decision, perf, hits = [], [], {}
    # The "days" approach lives under "days_approach": row["days"] is already the gold day count.
    for label, a in (("naive", "naive"), ("days", "days_approach"), ("split", "split")):
        pred = [r[a]["eligible"] for r in ok]
        hits[a] = [g == p for g, p in zip(gold, pred)]
        p, rec, _, _ = prf(gold, pred, [True])[True]
        decision.append([label, _rate(hits[a]), _rate([h for h, n in zip(hits[a], near) if n]),
                         _rate([h for h, n in zip(hits[a], near) if not n]), fmt(p), fmt(rec)])
        lat = [r[a]["latency_ms"] for r in ok]
        total, missing = cost_totals(rows, a)
        perf.append([label, fmt(percentile(lat, 50), ".0f"), fmt(percentile(lat, 95), ".0f"), fmt(total, ".4f", "$"),
                     fmt(ratio(total * 1000, len(rows)), ".4f", "$"), f"missing: {missing}" if missing else ""])
    out.append("## Eligibility decision (ok rows)\n\nNear boundary: |days - window| <= 2. Precision and recall are for eligible=True.\n\n"
               + table(["approach", "accuracy", "near boundary", "far", "precision", "recall"], decision))
    out.append("## Latency (ms, ok rows) and cost (all rows)\n\n"
               + table(["approach", "p50", "p95", "total cost", "per 1,000 requests", "note"], perf))

    classes = CLASSES["refund"]
    reasons = [r["reason"] for r in ok]
    jev_reasons = [r["split"]["reason"] for r in ok]
    out.append(f"## Split: reason\n\nReason accuracy: {fmt(accuracy(reasons, jev_reasons))} (n={len(ok)})")
    cm = confusion(reasons, jev_reasons, classes)
    out.append("Confusion matrix (rows: gold reason, cols: Jev reason)\n\n"
               + table(["gold \\ jev", *classes], [[g, *cm[g].values()] for g in classes]))
    split = threshold_split([r["split"]["confidence"] for r in ok], hits["split"], threshold)
    out.append(f"## Split decision accuracy by reason confidence (threshold {threshold})\n\n" + table(
        ["confidence", "accuracy", "n"],
        [[f">= {threshold}", fmt(split["above"][0]), split["above"][1]],
         [f"< {threshold}", fmt(split["below"][0]), split["below"][1]]]))

    out.append(_errors(rows))
    return "\n\n".join(out) + "\n"


def summarize(rows: list[dict], usecase: str, threshold: float) -> str:
    if usecase == "refund":
        return _refund_summary(rows, threshold)
    classes = CLASSES[usecase]
    ok = [r for r in rows if r.get("error") is None]
    gold = [r["gold"] for r in ok]
    jev_key = "label" if usecase == "fraud" else "tier"
    jev_pred = [r["jev"].get(jev_key) for r in ok]
    others = [] if usecase == "fraud" else ["arm_a", "arm_b"]
    cost_sources = ["jev", *others] + ([] if usecase == "fraud" else ["judge"])

    out = [f"# {usecase} summary", f"Rows: {len(rows)} (ok: {len(ok)}, errors: {len(rows) - len(ok)})"]

    out.append("## Per-class metrics")
    out.append(_prf_section("Jev", gold, jev_pred, classes))

    split = threshold_split([r["jev"].get("confidence") for r in ok], [g == p for g, p in zip(gold, jev_pred)], threshold)
    out.append(f"## Jev accuracy by confidence (threshold {threshold})\n\n" + table(
        ["confidence", "accuracy", "n"],
        [[f">= {threshold}", fmt(split["above"][0]), split["above"][1]],
         [f"< {threshold}", fmt(split["below"][0]), split["below"][1]]]))

    lat = []
    for s in ["jev", *others]:
        vals = [r[s].get("latency_ms") for r in ok]
        lat.append([s, fmt(percentile(vals, 50), ".0f"), fmt(percentile(vals, 95), ".0f")])
    out.append("## Latency (ms, ok rows)\n\n" + table(["source", "p50", "p95"], lat))

    totals = {s: cost_totals(rows, s) for s in cost_sources}
    cost_rows = [[s, fmt(t, ".4f", "$"), fmt(ratio(t * 1000, len(rows)), ".4f", "$"), f"missing: {m}" if m else ""]
                 for s, (t, m) in totals.items()]
    out.append("## Cost (all rows)\n\n" + table(["source", "total", "per 1,000 requests", "note"], cost_rows))

    if usecase == "routing":
        rates = {a: correct_rates(ok, a, classes) for a in others}
        body = [[k] + [f"{fmt(rates[a][k][0])} (n={rates[a][k][1]})" for a in others] for k in ["overall", *classes]]
        out.append("## Arm correct rate\n\n" + table(["gold tier", "Arm A", "Arm B"], body))
        out.append(f"Arm A cost / Arm B cost: {fmt(ratio(totals['arm_a'][0], totals['arm_b'][0]))}")

        cm = confusion(gold, jev_pred, classes)
        out.append("## Confusion matrix (rows: gold tier, cols: Jev tier)\n\n"
                   + table(["gold \\ jev", *classes], [[g, *cm[g].values()] for g in classes]))

        opus = sum(r["arm_b"].get("cost_usd") or 0 for r in ok)
        body = []
        for t in GATE_THRESHOLDS:
            fallbacks, rate, cost = gated(ok, t)
            body.append([t, fallbacks, fmt(rate), fmt(cost, ".4f", "$"), fmt(ratio(cost, opus))])
        out.append("## Confidence-gated routing (ok rows)\n\nJev confidence >= threshold uses Arm A; below falls back to Arm B (Opus), still paying for the Jev call.\n\n"
                   + table(["threshold", "sent to Opus (fallbacks)", "correct rate", "total cost", "vs always-Opus cost"], body))

    out.append(_errors(rows))
    return "\n\n".join(out) + "\n"
