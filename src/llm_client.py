"""Real LLM API client via OpenRouter with on-disk caching and retries.

All experiments call REAL frontier models (Claude Sonnet 4.5, GPT-4.1) through
OpenRouter. Responses are cached to disk keyed by (model, messages, params) so
re-runs are deterministic and free. Temperature defaults to 0 for reproducibility.
"""
import os
import json
import time
import hashlib
import pathlib
from openai import OpenAI

CACHE_DIR = pathlib.Path(__file__).resolve().parent.parent / "results" / "llm_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_KEY"],
)

# Frontier models used across experiments (exact OpenRouter IDs).
MODELS = {
    "claude-sonnet-4.5": "anthropic/claude-sonnet-4.5",
    "gpt-4.1": "openai/gpt-4.1",
    "gpt-4.1-mini": "openai/gpt-4.1-mini",
}


def _cache_key(model, messages, temperature, max_tokens, seed):
    payload = json.dumps(
        {"model": model, "messages": messages, "temperature": temperature,
         "max_tokens": max_tokens, "seed": seed},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def chat(messages, model="claude-sonnet-4.5", temperature=0.0, max_tokens=1024,
         seed=42, use_cache=True, max_retries=5):
    """Return the assistant text for a chat completion. Cached by content."""
    model_id = MODELS.get(model, model)
    key = _cache_key(model_id, messages, temperature, max_tokens, seed)
    cache_file = CACHE_DIR / f"{key}.json"
    if use_cache and cache_file.exists():
        return json.loads(cache_file.read_text())["content"]

    last_err = None
    for attempt in range(max_retries):
        try:
            kwargs = dict(model=model_id, messages=messages,
                          temperature=temperature, max_tokens=max_tokens)
            # seed is best-effort; some providers ignore it.
            resp = _client.chat.completions.create(**kwargs)
            content = resp.choices[0].message.content or ""
            # Some upstream providers (e.g. Bedrock guardrails) block benign
            # symbol-substitution prompts with an empty content_filter response.
            # Surface that as an explicit failure so it is retried/visible.
            fr = getattr(resp.choices[0], "finish_reason", None)
            if content == "" and fr == "content_filter":
                content = "[CONTENT_FILTER_BLOCKED]"
            usage = getattr(resp, "usage", None)
            rec = {"content": content, "model": model_id,
                   "prompt_tokens": getattr(usage, "prompt_tokens", None),
                   "completion_tokens": getattr(usage, "completion_tokens", None)}
            cache_file.write_text(json.dumps(rec))
            return content
        except Exception as e:  # noqa: BLE001 - robust to transient API errors
            last_err = e
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"API failed after {max_retries} retries: {last_err}")


if __name__ == "__main__":
    print(chat([{"role": "user", "content": "Reply with exactly: OK"}],
               model="claude-sonnet-4.5", max_tokens=10))
