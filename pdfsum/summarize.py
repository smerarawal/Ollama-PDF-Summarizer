"""Three ways to summarize a long document.

concat      Your v1 behaviour: summarize every chunk, paste the results together.
            Kept as the baseline so improvements can be measured against it.
map_reduce  Summarize chunks in parallel (map), then repeatedly merge groups of
            summaries (reduce) until they fit one prompt, then write the final
            structured summary.
refine      Walk through the chunks in order, updating one running summary.
            Slower (sequential) but often more coherent.

Every strategy takes an `llm` callable (prompt -> text), so the logic can be
tested without a running model.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from . import config
from .llm import chat

MAP = (
    "Summarize the following excerpt in 3-5 concise bullet points. "
    "Use only the text provided; do not add outside facts.\n\nExcerpt:\n{t}"
)
COMBINE = (
    "Merge the following notes into one set of concise bullet points. "
    "Remove repetition and keep every distinct fact.\n\nNotes:\n{t}"
)
FINAL = (
    "Using only the notes below, write a structured summary with three headings: "
    "Overview (2-3 sentences), Key Points (bullets), Conclusion (1-2 sentences). "
    "Do not add facts that are not in the notes.\n\nNotes:\n{t}"
)
REFINE = (
    "You are building a summary of a long document step by step.\n\n"
    "Current summary:\n{summary}\n\nNew excerpt:\n{t}\n\n"
    "Update the summary to include any important new information from the excerpt. "
    "Keep it under 300 words and use only the given text. "
    "Return only the updated summary."
)

LLM = Callable[[str], str]
Progress = Callable[[int, int], None]


def _parallel(
    fn: Callable[[str], str],
    items: list[str],
    workers: int,
    progress: Progress | None = None,
) -> list[str]:
    """Apply fn to every item with a thread pool, keeping the original order.
    `progress` is called from the calling thread, so it is safe for Streamlit."""
    results: list[str] = [""] * len(items)
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(fn, item): index for index, item in enumerate(items)}
        for future in as_completed(futures):
            results[futures[future]] = future.result()
            done += 1
            if progress:
                progress(done, len(items))
    return results


def pack(items: list[str], limit: int) -> list[str]:
    """Join neighbouring strings into groups of about `limit` characters.

    Every group holds at least two items (except when only one item exists), so
    the number of groups is always smaller than the number of inputs. That
    guarantee is what stops the reduce loop from running forever.
    """
    if len(items) <= 1:
        return list(items)

    groups: list[list[str]] = []
    current: list[str] = []
    size = 0
    for item in items:
        if current and size + len(item) > limit and len(current) >= 2:
            groups.append(current)
            current, size = [], 0
        current.append(item)
        size += len(item)
    if current:
        groups.append(current)
    if len(groups) > 1 and len(groups[-1]) == 1:  # no lonely trailing group
        groups[-2].extend(groups.pop())
    return ["\n".join(group) for group in groups]


def concat(
    chunks: list[str], llm: LLM = chat, progress: Progress | None = None, **_
) -> str:
    """v1 baseline: sequential per-chunk summaries, simply joined."""
    parts = _parallel(lambda c: llm(MAP.format(t=c)), chunks, 1, progress)
    return "\n\n".join(parts)


def map_reduce(
    chunks: list[str],
    llm: LLM = chat,
    progress: Progress | None = None,
    limit: int | None = None,
    workers: int | None = None,
    max_rounds: int = 10,
    **_,
) -> str:
    if not chunks:
        return ""
    limit = limit or config.REDUCE_LIMIT
    workers = workers or config.WORKERS

    level = _parallel(lambda c: llm(MAP.format(t=c)), chunks, workers, progress)
    rounds = 0
    while sum(len(s) for s in level) > limit and len(level) > 1 and rounds < max_rounds:
        groups = pack(level, limit)
        level = _parallel(lambda g: llm(COMBINE.format(t=g)), groups, workers)
        rounds += 1
    return llm(FINAL.format(t="\n".join(level)))


def refine(
    chunks: list[str], llm: LLM = chat, progress: Progress | None = None, **_
) -> str:
    if not chunks:
        return ""
    summary = llm(MAP.format(t=chunks[0]))
    if progress:
        progress(1, len(chunks))
    for number, chunk in enumerate(chunks[1:], start=2):
        summary = llm(REFINE.format(summary=summary, t=chunk))
        if progress:
            progress(number, len(chunks))
    return llm(FINAL.format(t=summary))


STRATEGIES = {"concat": concat, "map_reduce": map_reduce, "refine": refine}


def summarize(
    chunks: list[str],
    strategy: str = "map_reduce",
    llm: LLM = chat,
    progress: Progress | None = None,
) -> str:
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy {strategy!r}; choose from {sorted(STRATEGIES)}")
    return STRATEGIES[strategy](chunks, llm=llm, progress=progress)
