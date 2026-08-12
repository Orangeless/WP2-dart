"""
llm.py — one logged call site for every LLM request (tokens / $ / latency).

Same design as the D3 starter: because all model traffic funnels through chat(),
the cost record is uniform across every resolution strategy. Uses LiteLLM, so the
same code hits OpenAI / Anthropic / OpenRouter. Set the relevant API key in the env.
"""
import os
import sys
import time
import random
from dataclasses import dataclass
from typing import List

try:
    from litellm import completion, completion_cost
except ImportError:
    completion = None
from types import SimpleNamespace
import httpx
import config


@dataclass
class CallLog:
    resolver: str
    stage: str
    model: str
    inst_id: str
    prompt_tokens: int
    completion_tokens: int
    usd: float
    latency_ms: float
    provider: str = ""


def _api_base() -> str:
    """OpenRouter base URL. `.env` ships OPENROUTER_API_BASE as an empty string, and
    os.getenv(key, default) returns that empty string rather than the default — which
    made every request go to a bare '/chat/completions' and fail with UnsupportedProtocol.
    Treat empty/whitespace as unset."""
    return (os.getenv("OPENROUTER_API_BASE") or "").strip() or "https://openrouter.ai/api/v1"


# prefix -> environment variable holding that provider's key.
_KEY_VAR = {"openrouter/": "OPENROUTER_API_KEY",
            "openai/": "OPENAI_API_KEY",
            "anthropic/": "ANTHROPIC_API_KEY"}


def _require_key(prefix: str, model: str) -> str:
    key = os.getenv(_KEY_VAR[prefix])
    if not key:
        raise RuntimeError(f"Missing {_KEY_VAR[prefix]} for model '{model}'. "
                           f"Set it in .env or switch to a model you have a key for.")
    return key


def _provider_kwargs(model: str, prompt: str, temperature: float, max_tokens: int, seed: int) -> dict:
    if "/" not in model:
        if model.startswith(("gpt-", "o1", "o3")):
            model = f"openai/{model}"
        elif model.startswith(("claude", "anthropic")):
            model = f"anthropic/{model}"
        else:
            raise RuntimeError(
                f"Model '{model}' is missing a provider prefix. Use a value like "
                f"'openrouter/meta-llama/llama-3.2-3b-instruct' or 'openai/gpt-4o-mini'.")

    prefix = next((p for p in _KEY_VAR if model.startswith(p)), None)
    if prefix is None:
        raise RuntimeError(f"Unsupported provider for model '{model}'. "
                           f"Use openrouter/, openai/, or anthropic/.")
    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "seed": seed,
        "api_key": _require_key(prefix, model),
    }
    if prefix == "openrouter/":
        kwargs["api_base"] = _api_base()
    return kwargs


# Errors worth retrying: provider blips, rate limits, and the spurious 400s that
# OpenRouter occasionally returns when it routes to a momentarily-flaky backend
# (the identical request succeeds on retry). Auth errors (401/403) are never retried.
MAX_ATTEMPTS = 4
RETRY_STATUS = {400, 408, 409, 425, 429, 500, 502, 503, 504}

# Providers that advertise a model on OpenRouter but then reject the actual chat request
# (e.g. Novita -> 400 "does not support endpoint: completions" for qwen-2.5-72b-instruct).
# Skipping them means a rate-limited primary provider (e.g. DeepInfra 429) backs off and
# retries cleanly instead of falling through to a broken one. Edit if you hit another.
OPENROUTER_IGNORE_PROVIDERS = ["Novita"]


def _classify(exc: Exception) -> tuple:
    """(category, detail) where category is 'auth' | 'retry' | 'fatal'."""
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        body = (exc.response.text or "")[:200].replace("\n", " ")
        if code in (401, 403):
            return "auth", f"HTTP {code}"
        return ("retry" if code in RETRY_STATUS else "fatal"), f"HTTP {code}: {body}"
    if isinstance(exc, (httpx.TimeoutException, httpx.TransportError)):
        return "retry", type(exc).__name__
    msg = str(exc).lower()
    if any(t in msg for t in ["api key", "authorization", "credential", "unauthorized"]):
        return "auth", str(exc)[:150]
    if any(t in msg for t in ["timed out", "timeout", "connection", "temporarily",
                              "rate limit", "overloaded", "429", "500", "502", "503", "504"]):
        return "retry", str(exc)[:150]
    return "fatal", str(exc)[:200]


def chat(prompt: str, model: str, *, resolver: str, stage: str, inst_id: str,
         temperature: float, max_tokens: int, seed: int, log: List[CallLog]) -> str:
    if completion is None:
        raise RuntimeError("pip install litellm  (see requirements.txt)")
    if not model:
        raise ValueError("No model configured. Update config.MODELS with a valid provider-prefixed model name.")
    for attempt in range(MAX_ATTEMPTS):
        t0 = time.perf_counter()
        served_by = ""
        try:
            # direct OpenRouter HTTP call for openrouter/ models (avoids litellm provider noise)
            if model.startswith("openrouter/"):
                api_base = _api_base()
                key = _require_key("openrouter/", model)
                # strip provider prefix for the remote model id
                model_id = model.split("/", 1)[1]
                payload = {
                    "model": model_id,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "seed": seed,   # was omitted here, so no run was actually seeded
                    # ask OpenRouter to price the call. litellm's completion_cost()
                    # cannot price the SimpleNamespace we build below, so it always
                    # fell through to 0.0 and every usd column was empty.
                    "usage": {"include": True},
                }
                if OPENROUTER_IGNORE_PROVIDERS:
                    payload["provider"] = {"ignore": OPENROUTER_IGNORE_PROVIDERS}
                if config.REASONING_EFFORT:
                    payload["reasoning"] = {"effort": config.REASONING_EFFORT}
                headers = {"Authorization": f"Bearer {key}"}
                r = httpx.post(f"{api_base}/chat/completions", json=payload,
                               headers=headers, timeout=config.REQUEST_TIMEOUT)
                r.raise_for_status()
                j = r.json()
                text = j.get("choices", [{}])[0].get("message", {}).get("content", "")
                usage = j.get("usage", {})
                # which backend actually served this call — quantization and therefore
                # output can differ between providers for the same model slug
                served_by = j.get("provider", "")
                resp = SimpleNamespace(
                    usage=SimpleNamespace(prompt_tokens=usage.get("prompt_tokens", 0),
                                          completion_tokens=usage.get("completion_tokens", 0),
                                          cost=usage.get("cost")),
                    choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
                )
            else:
                kwargs = _provider_kwargs(model, prompt, temperature, max_tokens, seed)
                resp = completion(**kwargs)
        except Exception as exc:
            category, detail = _classify(exc)
            if category == "auth":
                raise RuntimeError(
                    f"LLM request failed for model '{model}': provider credentials missing or invalid ({detail})."
                ) from exc
            if category == "retry" and attempt < MAX_ATTEMPTS - 1:
                time.sleep(1.0 * (2 ** attempt) + random.uniform(0, 0.5))  # backoff + jitter
                continue
            # Out of retries, or a non-transient error: skip this one instance (scored as
            # an abstain) rather than aborting the whole sweep and losing prior results.
            print(f"[warn] {model} {resolver}/{stage} inst={inst_id}: skipped after "
                  f"{attempt + 1} attempt(s) — {detail}", file=sys.stderr)
            return ""
        latency_ms = (time.perf_counter() - t0) * 1000.0
        u = resp.usage
        usd = getattr(u, "cost", None)       # OpenRouter prices the call for us
        if usd is None:
            try:
                usd = completion_cost(completion_response=resp)
            except Exception:
                usd = 0.0
        text = resp.choices[0].message.content or ""
        log.append(CallLog(resolver, stage, model, inst_id, u.prompt_tokens,
                           u.completion_tokens, usd, latency_ms, served_by))
        return text
    return ""  # exhausted retries with no success
