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
        self.timeout = timeout if timeout is not None else float(os.getenv("LLM_TIMEOUT", "3600.0"))
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

    def _synthesize_final_report(self) -> str:
        """
        Dedicated report synthesis phase.
        Executes pure text generation without tools schema to eliminate grammar constraints,
        and uses multi-turn continuation if output is truncated by length or syntax.
        """
        print("\n[PhosphogypsumBot] Synthesizing final decision report (pure text generation mode)...")
        synthesis_prompt = (
            "【系统指令：工具数据收集完毕】请基于上述所有工具计算的定量数据与文献事实，不再调用任何工具，直接输出一份结构完整、逻辑严密、数据详实的最终 5D-TEPES 决策评估大报告（Markdown 格式）。\n\n"
            "【输出格式与内容要求】：\n"
            "1. 必须包含完整的核心结论速览表、量化指标详细对比（LCA 环境影响与 TEA 经济指标）、工程可行性/TRL 评估与政策落地建议；\n"
            "2. 严禁中途截断，Markdown 表格与各章节小节必须完整收尾闭合；\n"
            "3. 直接输出最终报告正文，不再输出任何工具调用或占位符。"
        )
        self.messages.append({"role": "user", "content": synthesis_prompt})

        full_response = ""
        max_continuations = 3

        for cont_step in range(max_continuations):
            api_kwargs = {
                "model": self.model,
                "messages": self.messages,
                "temperature": 0.1,
                "max_tokens": self.max_tokens,
            }

            try:
                # Omit tools completely so llama-server operates in unrestricted free-form Markdown mode
                resp = self.client.chat.completions.create(**api_kwargs)
            except Exception as e:
                if full_response:
                    return full_response
                return f"[Agent Error] Synthesis failed: {e}."

            choice = resp.choices[0]
            chunk = (choice.message.content or "").strip()
            finish_reason = getattr(choice, "finish_reason", "stop")

            if chunk:
                if full_response:
                    full_response = full_response + "\n\n" + chunk
                else:
                    full_response = chunk
                self.messages.append({"role": "assistant", "content": chunk})

            print(f"[PhosphogypsumBot] Synthesis step {cont_step + 1}: generated {len(chunk)} chars, finish_reason: '{finish_reason}', total: {len(full_response)} chars")

            # Check if output is complete:
            # - finish_reason is "stop" (not "length")
            # - text length is at least 350 chars
            # - text does not end with incomplete punctuation / table border / trailing hyphen
            is_truncated = (
                finish_reason == "length"
                or full_response.endswith(("–", "-", "、", "，", "|", "：", "(", "低"))
                or full_response.count("```") % 2 != 0
            )

            if not is_truncated and len(full_response) >= 350:
                break

            if cont_step < max_continuations - 1:
                print(f"[PhosphogypsumBot] Output incomplete (finish_reason='{finish_reason}'); triggering seamless continuation...")
                self.messages.append({
                    "role": "user",
                    "content": "请紧接着上文继续输出，严禁重复上文已有内容，直接完整续写未完成的分析、对比表格与结论，直至整篇报告完整收尾。",
                })

        return full_response

    def chat(self, user_input: str, reset_history: bool = False) -> str:
        """
        Process user input, run the tool-calling loop, and return the final text response.
        """
        if reset_history:
            self.reset_conversation()

        self.messages.append({"role": "user", "content": user_input})

        print(f"\n[PhosphogypsumBot] Analyzing intent & planning actions... (Model: {self.model})")

        # Allow up to 3 iterations for multi-hop tool execution
        max_tool_iterations = 3
        has_executed_tools = False

        for iteration in range(max_tool_iterations):
            try:
                api_kwargs = {
                    "model": self.model,
                    "messages": self.messages,
                    "temperature": 0.1,
                    "max_tokens": self.max_tokens,
                    "tools": self.tool_schemas,
                    "tool_choice": "auto",
                }
                response = self.client.chat.completions.create(**api_kwargs)
            except Exception as e:
                return f"[Agent Error] Connection failed: {e}. (Base URL: {self.base_url}, Model: {self.model})"

            choice = response.choices[0]
            response_message = choice.message

            # If the model didn't call any tools
            if not response_message.tool_calls:
                content = (response_message.content or "").strip()
                # If no tools were ever called and content is comprehensive, return directly
                if not has_executed_tools and len(content) > 100:
                    self.messages.append({"role": "assistant", "content": content})
                    return content
                # If tools were already executed, transition to dedicated synthesis
                print(f"\n[PhosphogypsumBot] Tool collection finished at step {iteration + 1}; entering dedicated synthesis...")
                break

            has_executed_tools = True
            msg_dict = {
                "role": "assistant",
                "content": response_message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in response_message.tool_calls
                ],
            }
            self.messages.append(msg_dict)

            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                function_args = tool_call.function.arguments

                print(f"[Agent Step {iteration+1}] Calling Tool: {function_name}({function_args})")

                try:
                    kwargs = json.loads(function_args) if function_args else {}
                    func = self.tools.get(function_name)

                    if func:
                        tool_result = str(func(**kwargs))
                    else:
                        tool_result = f"Error: Tool '{function_name}' not found."
                except Exception as e:
                    tool_result = f"Error executing '{function_name}': {str(e)}"

                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": tool_result,
                    }
                )

        # Transition to dedicated pure-text synthesis for complete long-form output
        return self._synthesize_final_report()