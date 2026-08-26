"""Optional search modes for the agent's single retrieval capability.

The agent takes retrieval as an injected ``search(query, k)`` callable; three
entry points (scripts/ask.py, eval/run_agent_eval.py,
eval/run_conflict_injection.py) previously each built an identical closure.
This module centralizes that construction and adds one optional mode.

Modes
-----
``dense`` (default, unchanged behavior)
    Dense + rerank, no BM25 -- the configuration sec-rag's ablation measured
    as best on this corpus.

``relgrep`` (optional)
    Relevance-guided corpus grep, after "A New Role for Relevance: Guiding
    Corpus Interaction in Agentic Search" (arXiv 2607.24223): instead of
    embedding the hop query, extract its literal anchors (company, fiscal
    year) deterministically, take the exact-match subset of the corpus --
    a conjunctive filter no bag-of-words retriever provides -- and order
    that subset coarse-to-fine with the hosted reranker. Falls back to the
    dense mode whenever the grep matches fewer than ``k`` chunks, so it can
    only narrow the pool, never come up empty-handed.

Both modes return ``sec_rag.retrieve.hybrid.Hit`` objects, so the agent loop
is agnostic to the mode in use.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Sequence

from sec_rag.retrieve.hybrid import Hit, Retriever

from .nvidia import rerank

# Cap on how many grep matches are sent to the hosted reranker in one call.
RERANK_POOL = 100

_YEAR_RE = re.compile(r"\b(?:FY\s?)?(20\d{2})\b", re.IGNORECASE)
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z&'-]+")

_STOPWORDS = frozenset("""
a an and are as at be between by did does for from has have how in is it its
of on or than that the their to was were what which with fy over under during
""".split())


def company_aliases(chunks: Sequence[dict[str, Any]]) -> dict[str, set[str]]:
    """ticker -> lowercase alias set (ticker itself + company-name words)."""
    aliases: dict[str, set[str]] = {}
    for c in chunks:
        t = c.get("ticker")
        if not t or t in aliases:
            continue
        names = {t.lower()}
        company = c.get("company") or ""
        # First word of the legal name is the colloquial handle
        # ("Apple Inc." -> "apple", "NVIDIA CORP" -> "nvidia").
        first = _WORD_RE.findall(company)
        if first:
            names.add(first[0].lower())
        aliases[t] = names
    return aliases


def extract_anchors(query: str,
                    aliases: dict[str, set[str]]) -> tuple[set[str], set[str], set[str]]:
    """(tickers, years, soft terms) literally present in the query."""
    lowered = {w.lower().rstrip("'").removesuffix("'s")
               for w in _WORD_RE.findall(query)}
    tickers = {t for t, names in aliases.items() if lowered & names}
    years = set(_YEAR_RE.findall(query))
    alias_words = set().union(*aliases.values()) if aliases else set()
    soft = {w for w in lowered
            if w not in _STOPWORDS and w not in alias_words
            and not w.startswith("fy")}
    return tickers, years, soft


def grep_chunks(chunks: Sequence[dict[str, Any]], tickers: set[str],
                years: set[str], soft: set[str]) -> list[dict[str, Any]]:
    """Conjunctive exact-match filter: company AND year must both hold
    (when the query names them); soft terms only order the matches."""
    matched = []
    for c in chunks:
        if tickers and c.get("ticker") not in tickers:
            continue
        if years:
            hay = (c.get("breadcrumb", "") + " " + c.get("text", ""))
            if not any(y in hay for y in years):
                continue
        matched.append(c)
    if len(matched) > RERANK_POOL and soft:
        def soft_hits(c: dict[str, Any]) -> int:
            hay = (c.get("breadcrumb", "") + " " + c.get("text", "")).lower()
            return sum(1 for w in soft if w in hay)
        matched.sort(key=soft_hits, reverse=True)
    return matched[:RERANK_POOL]


def make_search(retriever: Retriever, *, mode: str = "dense",
                candidates: int = 50) -> Callable[[str, int], list[Hit]]:
    """Build the ``search(query, k)`` callable the agent loop consumes."""
    if mode not in ("dense", "relgrep"):
        raise ValueError(f"unknown search mode: {mode!r}")

    def dense_search(query: str, k: int) -> list[Hit]:
        return retriever.search(query, top_k=k, candidates=candidates,
                                use_dense=True, use_sparse=False,
                                use_rerank=True)

    if mode == "dense":
        return dense_search

    chunks = retriever.index.chunks
    aliases = company_aliases(chunks)

    def relgrep_search(query: str, k: int) -> list[Hit]:
        tickers, years, soft = extract_anchors(query, aliases)
        if not tickers and not years:
            return dense_search(query, k)  # nothing literal to grep on
        matched = grep_chunks(chunks, tickers, years, soft)
        if len(matched) < k:
            return dense_search(query, k)
        passages = [c["breadcrumb"] + "\n" + c["text"] for c in matched]
        ranked = rerank(query, passages, model=retriever.rerank_model, top_k=k)
        return [Hit(chunk=matched[idx], score=logit, rerank_logit=logit)
                for idx, logit in ranked]

    return relgrep_search
