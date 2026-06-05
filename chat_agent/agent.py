"""
Provides the PhosphogypsumAgent class, which implements an OpenAI tool-calling 
loop (ReAct/Planning) to orchestrate PhosphogypsumBot's backend tools.
"""

import inspect
import json
import os
from typing import List, Dict, Any, Callable

try:
    from openai import OpenAI
except ImportError:
    print("WARNING: 'openai' package is not installed. Please run `pip install openai`.")
    OpenAI = None

from .tools import AVAILABLE_TOOLS


def function_to_schema(func: Callable) -> dict:
    """
    Convert a Python function into an OpenAI function calling schema.
    Relies on docstrings and type hints.
    """
    sig = inspect.signature(func)
    doc = inspect.getdoc(func) or ""
    
    # Simple docstring parsing (description is everything before 'Args:')
    description = doc.split("Args:")[0].strip() if "Args:" in doc else doc.strip()
    
    # Parse parameter descriptions from docstring
    param_descs = {}
    if "Args:" in doc:
        import re
        args_section = doc.split("Args:")[1]
        for next_section in ["Returns:", "Raises:", "Yields:", "Examples:"]:
            if next_section in args_section:
                args_section = args_section.split(next_section)[0]
                
        current_param = None
        for line in args_section.split("\n"):
            stripped = line.strip()
            if not stripped:
                continue
            # Match parameter name: description or name (type): description
            match = re.match(r"^([a-zA-Z_]\w*)\s*(?:\([^)]+\))?\s*:\s*(.*)$", stripped)
            if match:
                current_param = match.group(1)
                param_descs[current_param] = match.group(2).strip()
            else:
                if current_param:
                    param_descs[current_param] += " " + stripped
                    
    properties = {}
    required = []
    
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        
        param_type = "string"  # Default fallback
        if param.annotation == int:
            param_type = "integer"
        elif param.annotation == float:
            param_type = "number"
        elif param.annotation == bool:
            param_type = "boolean"
            
        param_desc = param_descs.get(name, f"Parameter {name}")
            
        properties[name] = {
            "type": param_type,
            "description": param_desc
        }
        
        if param.default == inspect.Parameter.empty:
            required.append(name)
            
    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }
    }


class PhosphogypsumAgent:
    """
    The central orchestration agent for PhosphogypsumBot.
    Connects user queries to backend physical solvers and LCA/TEA tools via LLM function calling.
    """
    def __init__(self, base_url: str = None, api_key: str = None, model: str = None):
        if OpenAI is None:
            raise ImportError("The 'openai' package is required for the Chat Agent. Install with: pip install openai")
            
        # Default to local llama-server if not specified
        self.base_url = base_url or os.getenv("LLM_BASE_URL", "http://127.0.0.1:11434/v1")
        self.api_key = api_key or os.getenv("LLM_API_KEY", "sk-no-key-required")
        self.model = model or os.getenv("LLM_MODEL", "qwen2.5:32b")
        
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key
        )
        
        self.tools = AVAILABLE_TOOLS
        self.tool_schemas = [function_to_schema(func) for func in self.tools.values()]
        
        # System instructions
        self.system_prompt = inspect.cleandoc("""
            You are PhosphogypsumBot, a highly advanced, physics-informed AI agent specializing in Industrial Phosphogypsum Engineering, Life Cycle Assessment (LCA), and Techno-Economic Analysis (TEA).
            You have access to a suite of backend tools (Python functions) that run rigorous thermodynamic physics solvers, MCMC uncertainty calibrations, and GraphRAG knowledge retrievals.

            Follow these strict rules:
            1. Always base your technical claims on the tools provided. If asked about a pathway's GWP, NPV, or viability, immediately CALL the `calculate_lca_tea` or `rank_all_pathways` tool.
            2. If asked about specific literature or reaction kinetics, CALL the `search_literature` tool.
            3. Be professional, scientifically rigorous, and structured in your final answers. Use markdown tables and bullet points.
            4. If a tool returns an error, inform the user clearly and suggest alternative actions.
            5. You can call multiple tools in sequence if the question requires it.
        """)
        
        self.messages = [
            {"role": "system", "content": self.system_prompt}
        ]

    def chat(self, user_input: str) -> str:
        """
        Process user input, run the tool-calling loop, and return the final text response.
        """
        self.messages.append({"role": "user", "content": user_input})
        
        print(f"\n[Agent] Thinking... (Model: {self.model})")
        
        # Max iteration limit to prevent infinite loops
        max_iterations = 5
        
        for _ in range(max_iterations):
            try:
                api_kwargs = {
                    "model": self.model,
                    "messages": self.messages,
                    "temperature": 0.1
                }
                if self.tool_schemas:
                    api_kwargs["tools"] = self.tool_schemas
                    api_kwargs["tool_choice"] = "auto"
                    
                response = self.client.chat.completions.create(**api_kwargs)
            except Exception as e:
                return f"[Agent Error] Connection failed: {e}. Is your LLM server ({self.base_url}) running?"
                
            response_message = response.choices[0].message
            
            # If the model didn't call any tools, it's a final text response
            if not response_message.tool_calls:
                final_answer = response_message.content
                self.messages.append({"role": "assistant", "content": final_answer})
                return final_answer
                
            # If it called tools, we append the assistant message as a dict and execute the tools
            msg_dict = {
                "role": "assistant",
                "content": response_message.content,
            }
            if response_message.tool_calls:
                msg_dict["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    } for tc in response_message.tool_calls
                ]
            self.messages.append(msg_dict)
            
            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                function_args = tool_call.function.arguments
                
                print(f"[Agent] Calling tool: {function_name} with arguments: {function_args}")
                
                # Execute the tool
                try:
                    kwargs = json.loads(function_args)
                    func = self.tools.get(function_name)
                    
                    if func:
                        tool_result = str(func(**kwargs))
                    else:
                        tool_result = f"Error: Tool {function_name} not found."
                except Exception as e:
                    tool_result = f"Error executing {function_name}: {str(e)}"
                    
                # Append tool result to history
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": tool_result
                })
                
        return "[Agent Error] Max tool iterations reached. Could not formulate a final answer."
