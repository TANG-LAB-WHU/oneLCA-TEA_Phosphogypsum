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

if __name__ == "__main__":
    # Manual test execution
    print("Testing Gemini connection...")
    test_gemini_connection()
    print("Testing configuration loading...")
    test_config_loading()
    print("\nAll tests passed! ✅")
