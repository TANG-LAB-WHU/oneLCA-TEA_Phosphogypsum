import os

import pytest
import yaml
from dotenv import load_dotenv


def test_llm_connection():
    """Test Ollama chat completions via /v1/chat/completions."""
    load_dotenv()

    base_url = os.getenv("LLM_BASE_URL")
    if not base_url:
        pytest.skip("LLM_BASE_URL not set")

    api_key = os.getenv("LLM_API_KEY", "ollama")
    model = os.getenv("LLM_MODEL", "qwen3.5:35b")

    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=api_key)
    response = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": "Reply with exactly: OK"}]
    )
    result = response.choices[0].message.content
    print(f"\nLLM response: {result}")
    assert len(result) > 0, "Empty response from LLM"


def test_embedding_connection():
    """Test Ollama embeddings via /v1/embeddings."""
    load_dotenv()

    base_url = os.getenv("LLM_BASE_URL")
    if not base_url:
        pytest.skip("LLM_BASE_URL not set")

    api_key = os.getenv("LLM_API_KEY", "ollama")
    model = os.getenv("EMBEDDING_MODEL", "qwen3-embedding:4b")
    expected_dim = int(os.getenv("EMBEDDING_DIM", "2560"))

    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=api_key)
    response = client.embeddings.create(model=model, input=["hello world"])
    vec = response.data[0].embedding

    print(f"\nEmbedding model: {model}, dim: {len(vec)}")
    assert len(vec) == expected_dim, f"Expected dim={expected_dim}, got {len(vec)}"


def test_config_loading():
    """Test if AI configuration in settings.yaml is correct."""
    config_path = os.path.join("config", "settings.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    assert "ai" in config
    assert config["ai"]["llm_provider"] == "ollama"
    assert "ollama" in config["ai"]
    assert "gemini" not in config["ai"]


def main():
    print("Testing configuration loading...")
    test_config_loading()
    print("Testing Ollama LLM connection...")
    test_llm_connection()
    print("Testing Ollama embedding connection...")
    test_embedding_connection()
    print("\nAll tests passed!")


if __name__ == "__main__":
    main()
