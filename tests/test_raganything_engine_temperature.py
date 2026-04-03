import asyncio

from pgloop.knowledge import raganything_engine as rag_mod


def _run(coro):
    return asyncio.run(coro)


def _make_engine(tmp_path, monkeypatch):
    monkeypatch.setattr(rag_mod, "RAGANYTHING_AVAILABLE", True)
    monkeypatch.setenv("LLM_TEMPERATURE", "0.15")
    monkeypatch.setenv("LLM_CONTEXT_LENGTH", "4096")
    return rag_mod.RAGAnythingEngine(working_dir=tmp_path)


def test_llm_func_injects_default_temperature_and_merges_context(monkeypatch, tmp_path):
    engine = _make_engine(tmp_path, monkeypatch)
    captured = {}

    async def fake_openai_complete(model, prompt, **kwargs):
        captured["model"] = model
        captured["prompt"] = prompt
        captured.update(kwargs)
        return "ok"

    monkeypatch.setattr(rag_mod, "openai_complete_if_cache", fake_openai_complete)
    llm_func = engine._create_llm_func()

    result = _run(llm_func("hello", extra_body={"options": {"seed": 7}}))

    assert result == "ok"
    assert captured["temperature"] == 0.15
    assert captured["extra_body"]["options"]["num_ctx"] == 4096
    assert captured["extra_body"]["options"]["seed"] == 7


def test_llm_func_respects_explicit_temperature(monkeypatch, tmp_path):
    engine = _make_engine(tmp_path, monkeypatch)
    captured = {}

    async def fake_openai_complete(model, prompt, **kwargs):
        captured.update(kwargs)
        return "ok"

    monkeypatch.setattr(rag_mod, "openai_complete_if_cache", fake_openai_complete)
    llm_func = engine._create_llm_func()

    _run(llm_func("hello", temperature=0.8))

    assert captured["temperature"] == 0.8


def test_vision_func_falls_back_when_temperature_is_none(monkeypatch, tmp_path):
    engine = _make_engine(tmp_path, monkeypatch)
    captured = {}

    async def fake_openai_complete(model, prompt, **kwargs):
        captured.update(kwargs)
        return "ok"

    monkeypatch.setattr(rag_mod, "openai_complete_if_cache", fake_openai_complete)
    vision_func = engine._create_vision_func()

    _run(vision_func("hello", messages=[{"role": "user", "content": "x"}], temperature=None))

    assert captured["temperature"] == 0.15
