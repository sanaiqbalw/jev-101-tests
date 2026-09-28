import json
import os
import time

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from typesafe_sdk import Choice, ChoiceAnswer, Noul, NoulAnswer, Score, ScoreAnswer, TypeSafeClient

# The SDK appends `/v1/systemone` itself, so its base URL stops before `/v1`.
JEV_BASE_URL = "https://openrouter.ai/api"
QUESTION_TYPES = {"choice": Choice, "score": Score, "noul": Noul}

_bedrock = boto3.client(
    "bedrock-runtime",
    config=Config(retries={"max_attempts": 4, "mode": "adaptive"}, read_timeout=300),
)


def _cost(usage) -> float | None:
    # OpenRouter reports `cost` outside the SDK's typed usage fields.
    if isinstance(usage, dict):
        return usage.get("cost")
    return getattr(usage, "cost", None) or (getattr(usage, "model_extra", None) or {}).get("cost")


def _answer(a) -> dict:
    if isinstance(a, ChoiceAnswer):
        return {"choice": a.choice, "confidence": a.confidence, "probabilities": dict(a.probabilities)}
    if isinstance(a, ScoreAnswer):
        return {"score": a.score, "confidence": a.confidence}
    if isinstance(a, NoulAnswer):
        return {"noul": a.noul}
    return {}


def ask_jev(state: dict, questions_cfg: list[dict], model: str) -> dict:
    """Returns {answers: {id: {...}}, latency_ms, cost_usd, error}."""
    start = time.perf_counter()
    try:
        questions = {
            q["id"]: QUESTION_TYPES[q["type"]](instructions=q["instructions"], criteria=q["criteria"])
            for q in questions_cfg
        }
        with TypeSafeClient(api_key=os.environ["OPENROUTER_API_KEY"], base_url=JEV_BASE_URL) as client:
            resp = client.system_one(state, questions, model=model)
        latency = round((time.perf_counter() - start) * 1000)
        # The SDK's Usage model ignores unknown fields, so `cost` only survives in the raw body.
        cost = _cost(resp.raw_http_response.json().get("usage") or resp.usage)
        answers = {name: _answer(a) for name, a in resp.answers.items()}
        return {"answers": answers, "latency_ms": latency, "cost_usd": cost, "error": None}
    except Exception as e:
        latency = round((time.perf_counter() - start) * 1000)
        return {"answers": {}, "latency_ms": latency, "cost_usd": None, "error": f"{type(e).__name__}: {e}"[:300]}


def chat(model_id: str, prompt: str, extra: dict | None = None) -> dict:
    """Returns {text, latency_ms, input_tokens, output_tokens, error}."""
    kwargs = {
        "modelId": model_id,
        "messages": [{"role": "user", "content": [{"text": prompt}]}],
        # 4096 truncated 20-record generation batches mid-JSON. Thinking tokens also
        # count against maxTokens. Only tokens actually produced are billed.
        "inferenceConfig": {"maxTokens": 16000},
    }
    # botocore rejects additionalModelRequestFields=None, so omit the key instead.
    if extra:
        kwargs["additionalModelRequestFields"] = extra
    start = time.perf_counter()
    text = in_tok = out_tok = error = None
    try:
        resp = _bedrock.converse(**kwargs)
        # Extended thinking adds `reasoningContent` blocks; keep only the answer text.
        text = "".join(b["text"] for b in resp["output"]["message"]["content"] if "text" in b)
        in_tok = resp["usage"]["inputTokens"]
        out_tok = resp["usage"]["outputTokens"]
    except ClientError as e:
        err = e.response.get("Error", {})
        error = f"{err.get('Code')}: {err.get('Message')}"[:300]
    except Exception as e:
        error = f"{type(e).__name__}: {e}"[:300]
    latency = round((time.perf_counter() - start) * 1000)
    return {"text": text, "latency_ms": latency, "input_tokens": in_tok, "output_tokens": out_tok, "error": error}


def parse_json(text: str | None) -> dict | list | None:
    """Strips ``` fences and json.loads; None if the text is not JSON."""
    if text is None:
        return None
    s = text.strip()
    if s.startswith("```"):
        s = s[3:].removeprefix("json")
        s = s.removesuffix("```")
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return None


def cost_usd(model_id: str, in_tok: int | None, out_tok: int | None, prices: dict) -> float | None:
    """(in_tok * input + out_tok * output) / 1e6; None if tokens or price missing."""
    p = prices.get(model_id)
    if in_tok is None or out_tok is None or not p or p.get("input") is None or p.get("output") is None:
        return None
    return (in_tok * p["input"] + out_tok * p["output"]) / 1e6
