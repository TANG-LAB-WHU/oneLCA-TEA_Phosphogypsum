import json
import pytest
from unittest.mock import MagicMock, patch

from chat_agent.agent import function_to_schema, PhosphogypsumAgent
from chat_agent.tools import get_available_pathways, search_literature
import chat_agent.cli as cli


def dummy_function(query: str, count: int = 5, flag: bool = True) -> str:
    """
    This is a dummy function for testing.
    
    Args:
        query: The search query to run.
        count: The number of results.
        flag: A boolean flag.
        
    Returns:
        A status string.
    """
    return f"Query: {query}, Count: {count}, Flag: {flag}"


def test_function_to_schema():
    schema = function_to_schema(dummy_function)
    
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "dummy_function"
    assert "dummy function for testing" in schema["function"]["description"]
    
    properties = schema["function"]["parameters"]["properties"]
    assert properties["query"]["type"] == "string"
    assert properties["query"]["description"] == "The search query to run."
    
    assert properties["count"]["type"] == "integer"
    assert properties["count"]["description"] == "The number of results."
    
    assert properties["flag"]["type"] == "boolean"
    assert properties["flag"]["description"] == "A boolean flag."
    
    required = schema["function"]["parameters"]["required"]
    assert "query" in required
    assert "count" not in required
    assert "flag" not in required


@patch("chat_agent.agent.OpenAI")
def test_agent_init(mock_openai_class):
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    
    agent = PhosphogypsumAgent(base_url="http://mock-url/v1", api_key="test-key", model="test-model")
    
    assert agent.base_url == "http://mock-url/v1"
    assert agent.api_key == "test-key"
    assert agent.model == "test-model"
    assert len(agent.tool_schemas) > 0
    assert any(s["function"]["name"] == "search_literature" for s in agent.tool_schemas)


@patch("chat_agent.agent.OpenAI")
def test_agent_chat_no_tools(mock_openai_class):
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    
    # Mock completion response (no tool calls)
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_message = MagicMock()
    mock_message.tool_calls = None
    mock_message.content = "This is a direct response."
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_response
    
    agent = PhosphogypsumAgent()
    response = agent.chat("Hello")
    
    assert response == "This is a direct response."
    mock_client.chat.completions.create.assert_called_once()
    assert agent.messages[-1]["role"] == "assistant"
    assert agent.messages[-1]["content"] == "This is a direct response."


@patch("chat_agent.agent.OpenAI")
def test_agent_chat_with_tool_calls(mock_openai_class):
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    
    # Mock tool call response
    mock_tool_call = MagicMock()
    mock_tool_call.id = "call_123"
    mock_tool_call.function.name = "get_available_pathways"
    mock_tool_call.function.arguments = "{}"
    
    mock_message_1 = MagicMock()
    mock_message_1.tool_calls = [mock_tool_call]
    mock_message_1.content = None
    
    mock_choice_1 = MagicMock()
    mock_choice_1.message = mock_message_1
    
    mock_response_1 = MagicMock()
    mock_response_1.choices = [mock_choice_1]
    
    # Mock final response
    mock_message_2 = MagicMock()
    mock_message_2.tool_calls = None
    mock_message_2.content = "Based on available pathways, we have PG-Stack and PG-CementProd."
    
    mock_choice_2 = MagicMock()
    mock_choice_2.message = mock_message_2
    
    mock_response_2 = MagicMock()
    mock_response_2.choices = [mock_choice_2]
    
    # Set side_effect to return mock_response_1 on first call and mock_response_2 on second call
    mock_client.chat.completions.create.side_effect = [mock_response_1, mock_response_2]
    
    agent = PhosphogypsumAgent()
    response = agent.chat("What are the available pathways?")
    
    assert "Based on available pathways" in response
    assert mock_client.chat.completions.create.call_count == 2
    
    # Check that tool execution message was appended
    tool_messages = [m for m in agent.messages if m.get("role") == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0]["name"] == "get_available_pathways"
    assert "PG-Stack" in tool_messages[0]["content"]


@patch("chat_agent.agent.OpenAI")
def test_agent_chat_error_handling(mock_openai_class):
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    
    mock_client.chat.completions.create.side_effect = Exception("Connection refused")
    
    agent = PhosphogypsumAgent()
    response = agent.chat("Hello")
    
    assert "[Agent Error] Connection failed" in response


@patch("chat_agent.agent.OpenAI")
def test_agent_chat_max_iterations(mock_openai_class):
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    
    mock_tool_call = MagicMock()
    mock_tool_call.id = "call_loop"
    mock_tool_call.function.name = "get_available_pathways"
    mock_tool_call.function.arguments = "{}"
    
    mock_message = MagicMock()
    mock_message.tool_calls = [mock_tool_call]
    mock_message.content = None
    
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    
    # Keep returning tool call message to trigger loop
    mock_client.chat.completions.create.return_value = mock_response
    
    agent = PhosphogypsumAgent()
    response = agent.chat("Infinite loop query")
    
    assert "[Agent Error] Max tool iterations reached" in response


@patch("chat_agent.cli.PhosphogypsumAgent")
def test_cli_query(mock_agent_class):
    mock_agent = MagicMock()
    mock_agent.model = "test-model"
    mock_agent.base_url = "http://test-url"
    mock_agent.tools = {"tool_a": lambda: "a"}
    mock_agent.chat.return_value = "CLI query response"
    mock_agent_class.return_value = mock_agent
    
    with patch("sys.argv", ["cli.py", "--query", "What is GWP?", "--model", "custom-model"]):
        cli.main()
        
    mock_agent_class.assert_called_once_with(model="custom-model")
    mock_agent.chat.assert_called_once_with("What is GWP?")
