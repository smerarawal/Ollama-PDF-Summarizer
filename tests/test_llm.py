import pytest

from pdfsum import llm


class FakeModel:
    def __init__(self, name):
        self.model = name


class FakeListed:
    def __init__(self, names):
        self.models = [FakeModel(n) for n in names]


def test_model_names_handles_object_and_dict_styles():
    assert llm._model_names(FakeListed(["mistral:latest"])) == ["mistral:latest"]
    assert llm._model_names({"models": [{"model": "a:1"}, {"name": "b:2"}]}) == ["a:1", "b:2"]


def test_check_ollama_ok_when_model_installed_with_tag(monkeypatch):
    monkeypatch.setattr(llm.ollama, "list", lambda: FakeListed(["mistral:latest", "nomic-embed-text:latest"]))
    llm.check_ollama(["mistral", "nomic-embed-text"])  # must not raise


def test_check_ollama_reports_missing_model(monkeypatch):
    monkeypatch.setattr(llm.ollama, "list", lambda: FakeListed(["mistral:latest"]))
    with pytest.raises(llm.OllamaUnavailable) as err:
        llm.check_ollama(["mistral", "llama3.1"])
    assert "ollama pull llama3.1" in str(err.value)


def test_check_ollama_reports_server_down(monkeypatch):
    def boom():
        raise ConnectionError("refused")

    monkeypatch.setattr(llm.ollama, "list", boom)
    with pytest.raises(llm.OllamaUnavailable) as err:
        llm.check_ollama(["mistral"])
    assert "ollama serve" in str(err.value)


def test_chat_counts_tokens_and_returns_text(monkeypatch):
    llm.STATS.reset()
    monkeypatch.setattr(
        llm.ollama,
        "chat",
        lambda **kw: {"message": {"content": "  hi  "}, "prompt_eval_count": 10, "eval_count": 4},
    )
    assert llm.chat("hello") == "hi"
    assert llm.STATS.snapshot() == {"prompt_tokens": 10, "completion_tokens": 4, "calls": 1}


def test_chat_retries_then_succeeds(monkeypatch):
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def flaky(**kw):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("model loading")
        return {"message": {"content": "ok"}}

    monkeypatch.setattr(llm.ollama, "chat", flaky)
    assert llm.chat("x", retries=2) == "ok"
    assert calls["n"] == 3


def test_chat_gives_up_with_clear_error(monkeypatch):
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)

    def always_fail(**kw):
        raise RuntimeError("nope")

    monkeypatch.setattr(llm.ollama, "chat", always_fail)
    with pytest.raises(llm.OllamaUnavailable):
        llm.chat("x", retries=1)


def test_every_call_sends_num_ctx_and_temperature(monkeypatch):
    seen = {}

    def spy(**kw):
        seen.update(kw)
        return {"message": {"content": "ok"}}

    monkeypatch.setattr(llm.ollama, "chat", spy)
    llm.chat("x", model="m")
    assert "num_ctx" in seen["options"] and "temperature" in seen["options"]
    assert seen["model"] == "m"
