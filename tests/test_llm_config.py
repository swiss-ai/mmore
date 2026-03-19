from mmore.rag import llm as llm_module
from mmore.rag.llm import LLM, LLMConfig


def test_llm_config_provider_explicit():
    cfg = LLMConfig(llm_name="some-model", provider="openai", max_new_tokens=128)

    assert cfg.provider == "OPENAI"
    assert cfg.generation_kwargs["max_completion_tokens"] == 128


def test_llm_config_legacy_model_name_inference():
    cfg = LLMConfig(llm_name="mistral-7b", max_new_tokens=64)

    assert cfg.provider == "MISTRAL"
    assert cfg.generation_kwargs["max_new_tokens"] == 64


def test_llm_config_legacy_custom_organization_with_base_url(monkeypatch):
    monkeypatch.setenv("SWISSAI_API_KEY", "swissai-token")

    cfg = LLMConfig(
        llm_name="meta-llama/Meta-Llama-3.1-70B-Instruct",
        organization="swissai",
        base_url="https://fmapi.example.org",
    )

    assert cfg.provider == "OPENAI"
    assert cfg.api_key_env_var == "SWISSAI_API_KEY"
    assert cfg.api_key == "swissai-token"


def test_llm_from_config_uses_provider_loader_and_kwargs(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "openai-token")
    captured: dict = {}
    sentinel = object()

    def fake_loader(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setitem(llm_module.loaders, "OPENAI", fake_loader)

    cfg = LLMConfig(
        llm_name="gpt-4.1-mini",
        provider="openai",
        base_url="https://api.example.org/v1",
        max_new_tokens=33,
        temperature=0.15,
        model_kwargs={"top_p": 0.9},
        client_kwargs={"timeout": 30},
    )

    result = LLM.from_config(cfg)
    assert result is sentinel
    assert captured["model"] == "gpt-4.1-mini"
    assert captured["base_url"] == "https://api.example.org/v1"
    assert captured["api_key"] == "openai-token"
    assert captured["temperature"] == 0.15
    assert captured["max_completion_tokens"] == 33
    assert captured["top_p"] == 0.9
    assert captured["timeout"] == 30
