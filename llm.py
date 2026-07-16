"""
llm.py — one logged call site for every LLM request (tokens / $ / latency).

Same design as the D3 starter: because all model traffic funnels through chat(),
the cost record is uniform across every resolution strategy. Uses LiteLLM, so the
same code hits OpenAI / Anthropic / OpenRouter. Set the relevant API key in the env.
"""
import os
import time
from dataclasses import dataclass, field
from typing import List

try:
    from litellm import completion, completion_cost
except ImportError:
    completion = None
from types import SimpleNamespace
import httpx


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
    text: str = field(default="", repr=False)


def _provider_kwargs(model: str, prompt: str, temperature: float, max_tokens: int, seed: int) -> dict:
    if "/" not in model:
        if model.startswith(("gpt-", "o1", "o3")):
            model = f"openai/{model}"
        elif model.startswith(("claude", "anthropic")):
            model = f"anthropic/{model}"
        else:
            raise RuntimeError(
                f"Model '{model}' is missing a provider prefix. Use a value like 'openrouter/meta-llama/llama-3.2-3b-instruct' or 'openai/gpt-4o-mini'."
            )

    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "seed": seed,
    }
    if model.startswith("openrouter/"):
        kwargs["api_base"] = os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
        key = os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError(
                f"Missing OPENROUTER_API_KEY for model '{model}'. Set it in your environment or switch back to a configured model."
            )
        kwargs["api_key"] = key
    elif model.startswith("openai/") or model.startswith("gpt-") or model.startswith("o1") or model.startswith("o3"):
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                f"Missing OPENAI_API_KEY for model '{model}'. Set it in your environment or switch to an OpenRouter model."
            )
        kwargs["api_key"] = key
    elif model.startswith("anthropic/"):
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                f"Missing ANTHROPIC_API_KEY for model '{model}'. Set it in your environment or choose another model."
            )
        kwargs["api_key"] = key
    else:
        raise RuntimeError(
            f"Unsupported provider for model '{model}'. Use openrouter/, openai/, or anthropic/."
        )
    return kwargs


def chat(prompt: str, model: str, *, resolver: str, stage: str, inst_id: str,
         temperature: float, max_tokens: int, seed: int, log: List[CallLog]) -> str:
    if completion is None:
        raise RuntimeError("pip install litellm  (see requirements.txt)")
    if not model:
        raise ValueError("No model configured. Update config.MODELS with a valid provider-prefixed model name.")
    t0 = time.perf_counter()
    try:
        # direct OpenRouter HTTP call for openrouter/ models (avoids litellm provider noise)
        if model.startswith("openrouter/"):
            api_base = os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
            key = os.getenv("OPENROUTER_API_KEY")
            if not key:
                raise RuntimeError(f"Missing OPENROUTER_API_KEY for model '{model}'. Set it in your environment.")
            # strip provider prefix for the remote model id
            model_id = model.split("/", 1)[1]
            payload = {
                "model": model_id,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            headers = {"Authorization": f"Bearer {key}"}
            r = httpx.post(f"{api_base}/chat/completions", json=payload, headers=headers, timeout=60.0)
            r.raise_for_status()
            j = r.json()
            text = j.get("choices", [{}])[0].get("message", {}).get("content", "")
            usage = j.get("usage", {})
            resp = SimpleNamespace(
                usage=SimpleNamespace(prompt_tokens=usage.get("prompt_tokens", 0),
                                      completion_tokens=usage.get("completion_tokens", 0)),
                choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
            )
        else:
            kwargs = _provider_kwargs(model, prompt, temperature, max_tokens, seed)
            resp = completion(**kwargs)
    except Exception as exc:
        msg = str(exc).lower()
        if any(token in msg for token in ["api key", "authorization", "credential", "unauthorized"]):
            raise RuntimeError(
                f"LLM request failed for model '{model}' because the provider credentials are missing or invalid."
            ) from exc
        if any(token in msg for token in ["ssl", "certificate", "connect", "timed out", "network"]):
            raise RuntimeError(
                f"LLM request failed for model '{model}' due to a network/SSL issue. Check your connection or try a different provider."
            ) from exc
        raise RuntimeError(f"LLM request failed for model '{model}': {exc}") from exc
    latency_ms = (time.perf_counter() - t0) * 1000.0
    u = resp.usage
    try:
        usd = completion_cost(completion_response=resp)
    except Exception:
        usd = 0.0
    text = resp.choices[0].message.content or ""
    log.append(CallLog(resolver, stage, model, inst_id,
                       u.prompt_tokens, u.completion_tokens, usd, latency_ms, text))
    return text
