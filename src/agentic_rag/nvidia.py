"""NVIDIA NIM client — chat, embeddings, reranking.

Ported from the battle-tested JS client in ~/aiforgift/lib/nvidia.js, which runs
nine production pipelines. The hard-won details preserved here:

- **Streaming for chat.** Non-streamed NVIDIA chat calls can exceed a 30s header
  timeout on long prompts. Stream and accumulate.
- **Reasoning suppression.** Several hosted models emit chain-of-thought that
  pollutes JSON output. Per-family suppression is required and the incantations
  differ (see `_reasoning_kwargs`). Some models emit their answer into
  `reasoning_content` rather than `content`, so we accumulate both.
- **Transient-classifying retry.** Only retry what's actually retryable.
- **Model fallback chains.** The free tier drops models without notice; a chain
  that validates output and moves on is the difference between a degraded run
  and a hard failure.

Three endpoint families, and they are NOT uniform:

    chat       POST https://integrate.api.nvidia.com/v1/chat/completions   (OpenAI-compatible)
    embeddings POST https://integrate.api.nvidia.com/v1/embeddings         (OpenAI-ish + input_type)
    reranking  POST https://ai.api.nvidia.com/v1/retrieval/{model}/reranking  (different HOST and schema)

Note reranking lives on a different host with a bespoke request/response shape,
and rerank models never appear in /v1/models.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

import httpx

INTEGRATE_BASE = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
RETRIEVAL_BASE = os.getenv("NVIDIA_RETRIEVAL_BASE", "https://ai.api.nvidia.com/v1/retrieval")

# Max passages the reranking endpoint accepts in one call.
RERANK_MAX_PASSAGES = 1000

_TRANSIENT = re.compile(
    r"\b(408|429|50\d|52\d)\b|timeout|timed out|connection|terminated|"
    r"temporarily|overload|unavailable",
    re.I,
)


class NvidiaError(RuntimeError):
    def __init__(self, message: str, *, status: int | None = None, transient: bool = False):
        super().__init__(message)
        self.status = status
        self.transient = transient


def _api_key() -> str:
    key = os.getenv("NVIDIA_API_KEY")
    if not key:
        raise NvidiaError(
            "NVIDIA_API_KEY is not set. Get a key at https://build.nvidia.com "
            "and export it, or put it in .env"
        )
    return key


def _headers(stream: bool = False) -> dict[str, str]:
    h = {"Authorization": f"Bearer {_api_key()}", "Content-Type": "application/json"}
    h["Accept"] = "text/event-stream" if stream else "application/json"
    return h


def _classify(exc: Exception) -> bool:
    """Is this worth retrying?"""
    if isinstance(exc, NvidiaError):
        return exc.transient
    if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError)):
        return True
    return bool(_TRANSIENT.search(str(exc)))


def with_retry(fn: Callable[[], Any], *, tries: int = 3, base_delay: float = 0.6) -> Any:
    """Linear backoff over transient failures only. Non-transient raises immediately."""
    last: Exception | None = None
    for i in range(tries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - deliberately broad, then classified
            last = exc
            if not _classify(exc) or i == tries - 1:
                raise
            time.sleep(base_delay * (i + 1))
    raise last  # pragma: no cover - unreachable


# --------------------------------------------------------------------------
# Chat
# --------------------------------------------------------------------------

def _reasoning_kwargs(model: str) -> dict[str, Any]:
    """Per-family reasoning suppression. Each of these was found the hard way.

    - qwen3.x: honors a `/no_think` directive appended to the system prompt
      (handled in `chat`, since it mutates messages rather than the body).
    - nemotron: needs BOTH `chat_template_kwargs.thinking=False` AND the
      "detailed thinking off" system directive. The directive alone silently
      stopped working on long prompts.
    - kimi: `chat_template_kwargs.thinking=False`.
    """
    m = model.lower()
    if "nemotron" in m or "kimi" in m or "qwen3.5" in m:
        return {"chat_template_kwargs": {"thinking": False}}
    return {}


def _reasoning_system_prefix(model: str) -> str | None:
    m = model.lower()
    if "nemotron" in m:
        return "detailed thinking off"
    if re.search(r"qwen3(\.\d)?", m):
        return "/no_think"
    return None


def chat(
    model: str,
    messages: list[dict[str, Any]],
    *,
    temperature: float = 0.2,
    max_tokens: int = 2048,
    timeout: float = 120.0,
    suppress_reasoning: bool = True,
    extra_body: dict[str, Any] | None = None,
) -> str:
    """Streaming chat completion, returned as a single string.

    Streams deliberately: non-streamed NVIDIA calls hit header timeouts on long
    prompts. Accumulates both `content` and `reasoning_content`, preferring
    `content` — some models put their real answer in the latter.
    """
    msgs = [dict(m) for m in messages]
    if suppress_reasoning:
        prefix = _reasoning_system_prefix(model)
        if prefix:
            if msgs and msgs[0].get("role") == "system":
                msgs[0]["content"] = f"{prefix}\n{msgs[0]['content']}"
            else:
                msgs.insert(0, {"role": "system", "content": prefix})

    body: dict[str, Any] = {
        "model": model,
        "messages": msgs,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }
    if suppress_reasoning:
        body.update(_reasoning_kwargs(model))
    if extra_body:
        body.update(extra_body)

    def _call() -> str:
        content: list[str] = []
        reasoning: list[str] = []
        with httpx.Client(timeout=timeout) as client:
            with client.stream(
                "POST", f"{INTEGRATE_BASE}/chat/completions",
                headers=_headers(stream=True), json=body,
            ) as resp:
                if resp.status_code != 200:
                    raw = resp.read().decode("utf-8", "replace")[:400]
                    raise NvidiaError(
                        f"NVIDIA chat {resp.status_code}: {raw}",
                        status=resp.status_code,
                        transient=resp.status_code in (408, 429) or resp.status_code >= 500,
                    )
                for line in resp.iter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload in ("", "[DONE]"):
                        continue
                    try:
                        chunk = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    for choice in chunk.get("choices") or []:
                        delta = choice.get("delta") or {}
                        if delta.get("content"):
                            content.append(delta["content"])
                        if delta.get("reasoning_content"):
                            reasoning.append(delta["reasoning_content"])
        return ("".join(content) or "".join(reasoning)).strip()

    return with_retry(_call)


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def extract_json(text: str) -> Any:
    """Pull the first JSON value out of a model response.

    Hosted NIM endpoints do not document `guided_json` / `nvext` constrained
    decoding (it exists only for self-hosted NIM containers), so we parse
    defensively instead of relying on a schema guarantee.
    """
    text = text.strip()
    for candidate in (text, *(m.group(1) for m in _FENCE.finditer(text))):
        candidate = candidate.strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    # Fall back to the outermost brace/bracket span.
    for opener, closer in (("{", "}"), ("[", "]")):
        i, j = text.find(opener), text.rfind(closer)
        if i != -1 and j > i:
            try:
                return json.loads(text[i : j + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"no JSON found in response: {text[:200]!r}")


def chat_json(
    model: str,
    messages: list[dict[str, Any]],
    *,
    validate: Callable[[Any], bool] | None = None,
    **kwargs: Any,
) -> Any:
    """Chat, then parse JSON, optionally validating the shape."""
    raw = chat(model, messages, **kwargs)
    data = extract_json(raw)
    if validate and not validate(data):
        raise ValueError(f"response failed validation: {json.dumps(data)[:200]}")
    return data


def chat_json_chain(
    models: Sequence[str],
    messages: list[dict[str, Any]],
    *,
    validate: Callable[[Any], bool] | None = None,
    **kwargs: Any,
) -> tuple[Any, str]:
    """Try models in order, first valid result wins. Returns (data, model_used).

    The free tier drops models without notice — this is why aiforgift grew a
    fallback chain after `qwen3-next` vanished mid-run and hard-failed orders.
    """
    errors: list[str] = []
    for m in models:
        try:
            return chat_json(m, messages, validate=validate, **kwargs), m
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{m}: {exc}")
    raise NvidiaError("all models in chain failed:\n  " + "\n  ".join(errors))


# --------------------------------------------------------------------------
# Embeddings
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class EmbedModel:
    """`input_type` is REQUIRED by the nemotron embedders and is what makes
    them asymmetric (query vs passage encoders). The OpenAI SDK cannot pass it,
    which is why this client uses httpx directly."""

    id: str
    dim: int


# Verified live against the catalog on 2026-07-26.
EMBED_NEMOTRON_3 = EmbedModel("nvidia/nemotron-3-embed-1b", 2048)
EMBED_NEMOTRON_V2 = EmbedModel("nvidia/llama-nemotron-embed-1b-v2", 2048)
EMBED_E5 = EmbedModel("nvidia/nv-embedqa-e5-v5", 1024)


def embed(
    texts: Sequence[str],
    *,
    model: str = EMBED_NEMOTRON_3.id,
    input_type: str = "passage",
    batch_size: int = 32,
    timeout: float = 120.0,
) -> list[list[float]]:
    """Embed texts. `input_type` must be "query" or "passage" — asymmetric.

    Using the wrong `input_type` silently degrades retrieval rather than
    erroring, so it is a required-by-convention argument here.
    """
    if input_type not in ("query", "passage"):
        raise ValueError('input_type must be "query" or "passage"')

    out: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = list(texts[start : start + batch_size])

        def _call(batch: list[str] = batch) -> list[list[float]]:
            with httpx.Client(timeout=timeout) as client:
                resp = client.post(
                    f"{INTEGRATE_BASE}/embeddings",
                    headers=_headers(),
                    json={
                        "model": model,
                        "input": batch,
                        "input_type": input_type,
                        "encoding_format": "float",
                        "truncate": "END",
                    },
                )
            if resp.status_code != 200:
                raise NvidiaError(
                    f"NVIDIA embed {resp.status_code}: {resp.text[:300]}",
                    status=resp.status_code,
                    transient=resp.status_code in (408, 429) or resp.status_code >= 500,
                )
            data = resp.json()["data"]
            # The API does not guarantee ordering; sort by index defensively.
            return [d["embedding"] for d in sorted(data, key=lambda d: d["index"])]

        out.extend(with_retry(_call))
    return out


# --------------------------------------------------------------------------
# Reranking
# --------------------------------------------------------------------------

RERANK_NEMOTRON = "nvidia/llama-nemotron-rerank-1b-v2"


def rerank(
    query: str,
    passages: Sequence[str],
    *,
    model: str = RERANK_NEMOTRON,
    top_k: int | None = None,
    timeout: float = 120.0,
) -> list[tuple[int, float]]:
    """Rerank passages against a query. Returns [(original_index, logit), ...]
    sorted best-first.

    Different host and schema from everything else — NOT OpenAI-compatible:
        POST {RETRIEVAL_BASE}/{model}/reranking
        {"model": ..., "query": {"text": ...}, "passages": [{"text": ...}]}

    Scores are raw logits (unbounded, can be negative), not probabilities. Use
    them for ordering, not as calibrated confidences.
    """
    if not passages:
        return []
    if len(passages) > RERANK_MAX_PASSAGES:
        raise ValueError(
            f"reranking accepts at most {RERANK_MAX_PASSAGES} passages, got {len(passages)}"
        )

    def _call() -> list[tuple[int, float]]:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(
                f"{RETRIEVAL_BASE}/{model}/reranking",
                headers=_headers(),
                json={
                    "model": model,
                    "query": {"text": query},
                    "passages": [{"text": p} for p in passages],
                    "truncate": "END",
                },
            )
        if resp.status_code != 200:
            raise NvidiaError(
                f"NVIDIA rerank {resp.status_code}: {resp.text[:300]}",
                status=resp.status_code,
                transient=resp.status_code in (408, 429) or resp.status_code >= 500,
            )
        rankings = resp.json()["rankings"]
        return [(r["index"], r["logit"]) for r in rankings]

    ranked = with_retry(_call)
    return ranked[:top_k] if top_k else ranked
