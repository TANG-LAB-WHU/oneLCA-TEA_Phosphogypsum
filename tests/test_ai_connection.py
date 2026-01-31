import os
import yaml
import pytest
from dotenv import load_dotenv
from google import genai
from pgloop.knowledge.llm_extractor import LLMExtractor

def test_gemini_connection():
    """Test if Gemini API connection is working properly"""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    
    assert api_key is not None, "GEMINI_API_KEY not found in .env"
    
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents="Hello, this is a connection test. Please reply with 'OK'."
        )
        print(f"\nGemini Response: {response.text}")
        assert "OK" in response.text or len(response.text) > 0
    except Exception as e:
        pytest.fail(f"Gemini connection failed: {str(e)}")


def test_config_loading():
    """Test if AI configuration in settings.yaml is loaded correctly"""
    config_path = os.path.join("config", "settings.yaml")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    assert "ai" in config
    assert config["ai"]["llm_provider"] == "gemini"
    assert "providers" in config["ai"]
    assert "gemini" in config["ai"]["providers"]


def test_openai_proxy_connection():
    """Test if OpenAI-compatible proxy (for Gemini) is working properly"""
    load_dotenv()
    
    base_url = os.getenv("LLM_BASE_URL")
    api_key = os.getenv("LLM_API_KEY")
    model = os.getenv("LLM_MODEL", "gemini-3-flash")
    
    if not base_url or not api_key:
        pytest.skip("LLM_BASE_URL or LLM_API_KEY not found in .env")
    
    try:
        from openai import OpenAI
        
        client = OpenAI(base_url=base_url, api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with exactly: OK"}]
        )
        result = response.choices[0].message.content
        print(f"\nOpenAI Proxy Response: {result}")
        assert len(result) > 0, "Empty response from OpenAI proxy"
    except Exception as e:
        pytest.fail(f"OpenAI proxy connection failed: {str(e)}")


if __name__ == "__main__":
    # Manual test execution
    print("Testing Gemini connection...")
    test_gemini_connection()
    print("Testing configuration loading...")
    test_config_loading()
    print("Testing OpenAI proxy connection...")
    test_openai_proxy_connection()
    print("\nAll tests passed! ✅")

