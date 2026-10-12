import threading

import pytest

from pdfsum import summarize
from pdfsum.summarize import COMBINE, FINAL, MAP, REFINE, pack


class FakeLLM:
    """Records every prompt and returns a short deterministic reply."""

    def __init__(self, reply_len: int = 40):
        self.prompts: list[str] = []
        self.reply_len = reply_len
        self._lock = threading.Lock()

    def __call__(self, prompt: str) -> str:
        with self._lock:
            self.prompts.append(prompt)
            n = len(self.prompts)
        return f"s{n}-" + "x" * self.reply_len

    def count(self, template_start: str) -> int:
        return sum(1 for p in self.prompts if p.startswith(template_start))


def test_pack_always_shrinks_and_respects_limit():
    items = ["a" * 100] * 9
    groups = pack(items, limit=250)
    assert len(groups) < len(items)
    assert sum(len(g.split("\n")) for g in groups) == 9  # nothing lost


def test_pack_never_leaves_lonely_trailing_group():
    groups = pack(["x" * 100] * 5, limit=250)
    assert all(len(g.split("\n")) >= 2 for g in groups)


def test_pack_with_huge_items_still_shrinks():
    groups = pack(["y" * 10_000] * 4, limit=100)
    assert len(groups) < 4


def test_pack_single_item_unchanged():
    assert pack(["only"], 10) == ["only"]


def test_map_reduce_runs_map_then_reduce_then_final():
    llm = FakeLLM(reply_len=500)
    chunks = [f"chunk {i}" for i in range(8)]
    out = summarize.map_reduce(chunks, llm=llm, limit=1200, workers=2)
    assert llm.count(MAP.split("{t}")[0]) == 8
    assert llm.count(COMBINE.split("{t}")[0]) >= 1  # at least one reduce round
    assert llm.count(FINAL.split("{t}")[0]) == 1
    assert out.startswith("s")  # the final call's reply is returned


def test_map_reduce_skips_reduce_when_already_small():
    llm = FakeLLM(reply_len=10)
    summarize.map_reduce(["a", "b", "c"], llm=llm, limit=6000, workers=2)
    assert llm.count(COMBINE.split("{t}")[0]) == 0
    assert llm.count(FINAL.split("{t}")[0]) == 1


def test_map_reduce_terminates_even_if_summaries_never_shrink():
    llm = FakeLLM(reply_len=5000)  # every reply is bigger than the limit
    out = summarize.map_reduce([f"c{i}" for i in range(6)], llm=llm, limit=100, workers=2)
    assert out  # finished instead of looping forever


def test_map_reduce_reports_progress():
    seen = []
    summarize.map_reduce(
        ["a", "b", "c"], llm=FakeLLM(), progress=lambda d, t: seen.append((d, t)), workers=2
    )
    assert seen[-1] == (3, 3)


def test_refine_calls_once_per_chunk_plus_final():
    llm = FakeLLM()
    summarize.refine(["a", "b", "c", "d"], llm=llm)
    assert llm.count(MAP.split("{t}")[0]) == 1
    assert llm.count(REFINE.split("{summary}")[0]) == 3
    assert llm.count(FINAL.split("{t}")[0]) == 1


def test_concat_is_one_call_per_chunk_and_joins_results():
    llm = FakeLLM(reply_len=3)
    out = summarize.concat(["a", "b", "c"], llm=llm)
    assert len(llm.prompts) == 3
    assert out.count("\n\n") == 2


def test_empty_input_returns_empty_string():
    llm = FakeLLM()
    assert summarize.map_reduce([], llm=llm) == ""
    assert summarize.refine([], llm=llm) == ""
    assert llm.prompts == []


def test_dispatch_and_unknown_strategy():
    assert summarize.summarize(["a"], "concat", llm=FakeLLM())
    with pytest.raises(ValueError):
        summarize.summarize(["a"], "magic", llm=FakeLLM())
