import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from report import CLASSES, accuracy, confusion, correct_rates, cost_totals, gated, percentile, prf, summarize, threshold_split


def fraud_row(i, gold, jev, conf, error=None):
    return {
        "id": str(i), "gold": gold, "error": error,
        "jev": {"label": jev, "confidence": conf, "probabilities": None, "risk": None,
                "cardholder_present": None, "merchant_suspicious": None,
                "latency_ms": 100 * i, "cost_usd": None if error else 0.001, "error": error},
    }


FRAUD = [
    fraud_row(1, "legitimate", "legitimate", 0.9),
    fraud_row(2, "legitimate", "card_testing", 0.6),
    fraud_row(3, "card_testing", "card_testing", 0.8),
    fraud_row(4, "account_takeover", "account_takeover", 0.95),
    fraud_row(5, "friendly_fraud", "legitimate", 0.5),
    fraud_row(6, "card_testing", None, None, error="TimeoutError: jev timed out"),
]


def arm(correct, cost):
    return {"model": "m", "text": "t", "correct": correct, "latency_ms": 50,
            "input_tokens": 10, "output_tokens": 5, "cost_usd": cost, "error": None}


def routing_row(i, gold, tier, conf, a, b):
    return {
        "id": str(i), "gold": gold, "error": None,
        "jev": {"tier": tier, "confidence": conf, "latency_ms": 10 * i, "cost_usd": 0.001, "error": None},
        "arm_a": arm(a, 0.01), "arm_b": arm(b, 0.02),
        "judge": {"input_tokens": 10, "output_tokens": 5, "cost_usd": 0.005, "error": None},
    }


ROUTING = [
    routing_row(1, "simple", "simple", 0.9, True, True),
    routing_row(2, "medium", "simple", 0.6, False, True),
    routing_row(3, "complex", "complex", 0.8, None, True),
    routing_row(4, "reasoning", "complex", 0.75, True, False),
]


def ok_fraud():
    ok = [r for r in FRAUD if r["error"] is None]
    return [r["gold"] for r in ok], [r["jev"]["label"] for r in ok], ok


def test_fraud_prf_and_accuracy():
    gold, pred, ok = ok_fraud()
    m = prf(gold, pred, CLASSES["fraud"])
    assert m["legitimate"] == (0.5, 0.5, 0.5, 2)
    p, r, f, n = m["card_testing"]
    assert (p, r, n) == (0.5, 1.0, 1) and f == pytest.approx(2 / 3)
    assert m["account_takeover"] == (1.0, 1.0, 1.0, 1)
    assert m["friendly_fraud"] == (None, 0.0, None, 1)
    assert m["merchant_collusion"] == (None, None, None, 0)
    assert accuracy(gold, pred) == 0.6


def test_fraud_threshold_latency_cost():
    gold, pred, ok = ok_fraud()
    split = threshold_split([r["jev"]["confidence"] for r in ok], [g == p for g, p in zip(gold, pred)], 0.7)
    assert split == {"above": (1.0, 3), "below": (0.0, 2)}
    lat = [r["jev"]["latency_ms"] for r in ok]
    assert (percentile(lat, 50), percentile(lat, 95)) == (300, 500)
    total, missing = cost_totals(FRAUD, "jev")
    assert total == pytest.approx(0.005) and missing == 1


def test_fraud_summary_text():
    s = summarize(FRAUD, "fraud", 0.7)
    assert "Error rows: 1" in s and "- TimeoutError: jev timed out" in s
    assert "Accuracy: 0.60 (n=5)" in s and "Frontier" not in s and "frontier" not in s
    assert "| merchant_collusion | – | – | – | 0 |" in s
    assert "| jev | 300 | 500 |" in s
    assert "| jev | $0.0050 | $0.8333 | missing: 1 |" in s


def test_routing_helpers():
    gold = [r["gold"] for r in ROUTING]
    pred = [r["jev"]["tier"] for r in ROUTING]
    cm = confusion(gold, pred, CLASSES["routing"])
    assert cm["simple"]["simple"] == 1 and cm["medium"]["simple"] == 1
    assert cm["complex"]["complex"] == 1 and cm["reasoning"]["complex"] == 1
    assert sum(sum(v.values()) for v in cm.values()) == 4
    a = correct_rates(ROUTING, "arm_a", CLASSES["routing"])
    assert a["overall"] == (pytest.approx(2 / 3), 3) and a["complex"] == (None, 0)
    assert correct_rates(ROUTING, "arm_b", CLASSES["routing"])["overall"] == (0.75, 4)
    split = threshold_split([r["jev"]["confidence"] for r in ROUTING], [g == p for g, p in zip(gold, pred)], 0.7)
    assert split == {"above": (pytest.approx(2 / 3), 3), "below": (0.0, 1)}


def test_routing_summary_text():
    s = summarize(ROUTING, "routing", 0.7)
    assert "Arm A cost / Arm B cost: 0.50" in s
    assert "| medium | 1 | 0 | 0 | 0 |" in s
    assert "| overall | 0.67 (n=3) | 0.75 (n=4) |" in s
    assert "| judge | $0.0200 |" in s
    assert "Error rows: 0" in s


def test_routing_gated():
    # t=0.7: rows 2 (0.6) falls back; a fallback pays arm_b + jev.
    fallbacks, rate, cost = gated(ROUTING, 0.7)
    assert fallbacks == 1 and rate == 1.0
    assert cost == pytest.approx(0.01 * 3 + 0.02 + 0.001)
    s = summarize(ROUTING, "routing", 0.7)
    assert "| 0.7 | 1 | 1.00 | $0.0510 | 0.64 |" in s


def noul_approach(eligible, latency, cost):
    return {"noul": 0.9 if eligible else 0.1, "eligible": eligible, "latency_ms": latency, "cost_usd": cost, "error": None}


def refund_row(i, gold, reason, days, window, naive, days_pred, jev_reason, conf, split):
    return {
        "id": str(i), "gold": gold, "reason": reason, "days": days, "window": window, "error": None,
        "naive": noul_approach(naive, 100, 0.001),
        "days_approach": noul_approach(days_pred, 150, 0.0015),
        "split": {"reason": jev_reason, "confidence": conf, "eligible": split,
                  "latency_ms": 200, "cost_usd": 0.002, "error": None},
    }


REFUND = [
    refund_row(1, True, "defective", 5, 30, True, True, "defective", 0.9, True),
    refund_row(2, False, "changed_mind", 15, 14, True, False, "changed_mind", 0.8, False),  # naive wrong near boundary
    refund_row(3, True, "wrong_item", 29, 30, True, True, "changed_mind", 0.5, False),     # split reason wrong
    refund_row(4, False, "changed_mind", 40, 14, False, True, "changed_mind", 0.95, False),  # days wrong far
]


def test_refund_summary_text():
    s = summarize(REFUND, "refund", 0.7)
    assert "| naive | 0.75 (n=4) | 0.50 (n=2) | 1.00 (n=2) | 0.67 | 1.00 |" in s
    assert "| days | 0.75 (n=4) | 1.00 (n=2) | 0.50 (n=2) | 0.67 | 1.00 |" in s
    assert "| split | 0.75 (n=4) | 0.50 (n=2) | 1.00 (n=2) | 1.00 | 0.50 |" in s
    assert s.index("| naive |") < s.index("| days |") < s.index("| split |")
    assert "| days | 150 | 150 | $0.0060 |" in s
    assert "Reason accuracy: 0.75 (n=4)" in s
    assert "| wrong_item | 0 | 0 | 1 |" in s
    assert "| >= 0.7 | 1.00 | 3 |" in s and "| < 0.7 | 0.00 | 1 |" in s
    assert "| naive | 100 | 100 | $0.0040 |" in s
    assert "Error rows: 0" in s
