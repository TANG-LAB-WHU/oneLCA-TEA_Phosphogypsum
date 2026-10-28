"""
Provides the PhosphogypsumAgent class, which implements an OpenAI tool-calling
loop (ReAct/Plan-and-Solve) to orchestrate PhosphogypsumBot's backend tools.
"""

import inspect
import json
import os
import re
from typing import Callable, Optional

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
        if param.annotation is int:
            param_type = "integer"
        elif param.annotation is float:
            param_type = "number"
        elif param.annotation is bool:
            param_type = "boolean"

        param_desc = param_descs.get(name, f"Parameter {name}")

        properties[name] = {"type": param_type, "description": param_desc}

        if param.default == inspect.Parameter.empty:
            required.append(name)

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


class PhosphogypsumAgent:
    """
    The central orchestration agent for PhosphogypsumBot.
    Connects user queries to backend physical solvers and LCA/TEA tools via LLM function calling
    using a Plan-and-Solve autonomous reasoning cycle.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        if OpenAI is None:
            raise ImportError(
                "The 'openai' package is required for the Chat Agent. Install with: pip install openai"
            )

        # Default to local server if not specified
        self.base_url = base_url or os.getenv("LLM_BASE_URL", "http://127.0.0.1:11434/v1")
        self.api_key = api_key or os.getenv("LLM_API_KEY", "sk-no-key-required")
        self.model = model or os.getenv("LLM_MODEL", "Qwen/Qwen3.8-Flash-Next")
        self.timeout = timeout if timeout is not None else float(os.getenv("LLM_TIMEOUT", "1800.0"))
        self.max_tokens = int(os.getenv("LLM_MAX_TOKENS", "1500"))

        # Set max_retries=0 to avoid wasteful 40-minute CPU retry loops upon client timeout
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
            max_retries=0,
        )

        self.tools = AVAILABLE_TOOLS
        self.tool_schemas = [function_to_schema(func) for func in self.tools.values()]

        # System instructions with Plan-and-Solve Chain-of-Thought (CoT)
        self.system_prompt = inspect.cleandoc("""
            You are PhosphogypsumBot, a world-class, physics-informed AI engineering agent specializing in Industrial Phosphogypsum (PG) Valorization, ISO 14040/14044 Life Cycle Assessment (LCA), Techno-Economic Analysis (TEA), and Circular Economy Policy Design.

            You have access to a suite of deterministic backend Python tools (thermodynamic physics solvers, Bayesian reverse design optimizers, benefit compensation calculators, MCMC uncertainty samplers, and live IoT telemetry streams).

            Follow these strict engineering rules:
            1. Scientific Grounding: NEVER guess numerical values (GWP, NPV, CAPEX, CLCC, reaction temperatures). Always execute the appropriate tool (`calculate_lca_tea`, `optimize_reverse_design`, `optimize_benefit_compensation`, etc.) to calculate them deterministically.
            2. Efficient & Focused Plan-and-Solve:
               - For multi-criteria ranking and screening under policies like "以渣定产":
                 First call `rank_all_pathways` and `search_literature`. Then focus detailed deep dives (such as `calculate_lca_tea` or `optimize_benefit_compensation`) ONLY on the top 2-3 dominant candidate pathways (such as `PG-CementProd`, `PG-Soil`, `PG-REEextract`) rather than redundantly iterating through all 7 pathways.
               - Do NOT run unnecessary Bayesian reverse design or MCMC parameter calibration unless explicitly requested in the prompt.
               - Decisive Reporting: Once key comparative metrics are gathered (within 2-3 tool calling steps), immediately synthesize your findings into a comprehensive 5D TEPES executive decision report. Always deliver a complete, highly structured Markdown report.
            3. Rigorous Formatting: Structure your final reports with clear Markdown headings, comparative tables, LaTeX formulas, and bulleted takeaways.
            4. Error Recovery: If a tool encounters an error, state the cause clearly and call a fallback tool or alternative parameter set.
        """)

        self.messages = [{"role": "system", "content": self.system_prompt}]

    def reset_conversation(self):
        """Reset conversation history back to the initial system prompt."""
        self.messages = [{"role": "system", "content": self.system_prompt}]

    def chat(self, user_input: str, reset_history: bool = False) -> str:
        """
        Process user input, run the tool-calling loop, and return the final text response.
        """
        if reset_history:
            self.reset_conversation()

        self.messages.append({"role": "user", "content": user_input})

        print(f"\n[PhosphogypsumBot] Analyzing intent & planning actions... (Model: {self.model})")

        # Allow up to 4 iterations for focused multi-hop tool execution
        max_iterations = 4

        for iteration in range(max_iterations):
            try:
                api_kwargs = {
                    "model": self.model,
                    "messages": self.messages,
                    "temperature": 0.1,
                    "max_tokens": self.max_tokens,
                }
                # On the final iteration, omit tools to force the model to write the text synthesis
                if self.tool_schemas and iteration < max_iterations - 1:
                    api_kwargs["tools"] = self.tool_schemas
                    api_kwargs["tool_choice"] = "auto"

                response = self.client.chat.completions.create(**api_kwargs)
            except Exception as e:
                return f"[Agent Error] Connection failed: {e}. (Base URL: {self.base_url}, Model: {self.model})"

            response_message = response.choices[0].message

            # If the model didn't call any tools, it's formulating a final text response
            if not response_message.tool_calls:
                final_answer = (response_message.content or "").strip()
                if final_answer:
                    self.messages.append({"role": "assistant", "content": final_answer})
                    return final_answer
                else:
                    # Model stopped calling tools but returned empty content; force final synthesis without tools
                    print("\n[PhosphogypsumBot] Finalizing decision report synthesis...")
                    self.messages.append({
                        "role": "user",
                        "content": "请基于以上所有工具计算与数据结果，直接输出一份结构完整、数据详实的最终 5D-TEPES 决策评估大报告（Markdown 格式）。"
                    })
                    final_resp = self.client.chat.completions.create(
                        model=self.model,
                        messages=self.messages,
                        temperature=0.1,
                        max_tokens=self.max_tokens,
                    )
                    final_answer = (final_resp.choices[0].message.content or "").strip()
                    self.messages.append({"role": "assistant", "content": final_answer})
                    return final_answer

            # If it called tools, append the assistant message and execute each tool
            msg_dict = {
                "role": "assistant",
                "content": response_message.content,
            }
            if response_message.tool_calls:
                msg_dict["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in response_message.tool_calls
                ]
            self.messages.append(msg_dict)

            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                function_args = tool_call.function.arguments

                print(f"[Agent Step {iteration+1}] Calling Tool: {function_name}({function_args})")

                # Execute the tool
                try:
                    kwargs = json.loads(function_args) if function_args else {}
                    func = self.tools.get(function_name)

                    if func:
                        tool_result = str(func(**kwargs))
                    else:
                        tool_result = f"Error: Tool '{function_name}' not found."
                except Exception as e:
                    tool_result = f"Error executing '{function_name}': {str(e)}"

                # Append tool result to history
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": tool_result,
                    }
                )

        # If max tool iterations reached, force final synthesis without tools
        print("\n[PhosphogypsumBot] Tool collection finished; synthesizing final executive report...")
        self.messages.append({
            "role": "user",
            "content": "请基于以上所有已获取的计算与工具数据，直接输出一份结构完整、数据详实的最终 5D-TEPES 决策评估大报告（Markdown 格式）。"
        })
        try:
            final_resp = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                temperature=0.1,
                max_tokens=self.max_tokens,
            )
            final_answer = (final_resp.choices[0].message.content or "").strip()
            self.messages.append({"role": "assistant", "content": final_answer})
            return final_answer
        except Exception as e:
            return f"[Agent Error] Synthesis failed: {e}." 
