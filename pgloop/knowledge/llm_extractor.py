"""
LLM Extractor Module

Uses Large Language Models to extract structured LCA/TEA data from text.
Calls Ollama's OpenAI-compatible /v1/chat/completions endpoint.
"""

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

_DEFAULT_BASE_URL = "http://127.0.0.1:11434/v1"
_DEFAULT_MODEL = "qwen3.5:35b"


def _read_env_int(*names: str, default: int = 0) -> int:
    """Read first valid integer environment variable."""
    for name in names:
        raw = os.getenv(name)
        if raw is None or str(raw).strip() == "":
            continue
        try:
            return int(raw)
        except ValueError:
            continue
    return default


@dataclass
class ExtractionResult:
    """Result of LLM extraction."""

    success: bool
    data: Dict[str, Any]
    raw_response: str
    confidence: float
    errors: List[str]


# Extraction prompts for different data types
EXTRACTION_PROMPTS = {
    "composition": """
Extract phosphogypsum composition data from the text.
Return JSON with these fields (use null if not found):
- CaSO4: mass fraction (0-1)
- P2O5: mass fraction (0-1)
- F: mass fraction (0-1)
- SiO2: mass fraction (0-1)
- Fe2O3: mass fraction (0-1)
- Al2O3: mass fraction (0-1)
- MgO: mass fraction (0-1)
- K2O: mass fraction (0-1)
- Na2O: mass fraction (0-1)
- heavy_metals: object mapping element (e.g., "As", "Cd", "Cr", "Hg", "Pb") to concentration (mg/kg)
- ra226: activity concentration (Bq/kg)
- pH: value
- moisture: mass fraction (0-1)
- particle_size_d50: microns
- whiteness: index value

Text: {text}
Return ONLY valid JSON.
""",
    "technology": """
Extract treatment technology data aligned with technical risk assessment criteria.
Return JSON with these fields (use null if not found):
- name: technology name
- description: brief description
- type: "purification", "calcination", "crystallization", "granulation", "decomposition", "other"
- trl: estimated Technology Readiness Level (1-9 integer)
- scale_factor: inferred scale relative to lab/pilot
  (e.g., "lab scale", "pilot scale", "industrial", or ratio)
- complexity: "low", "medium", or "high" (based on number of steps/conditions)
- novel_technology: boolean (true if described as novel/innovative/new method)
- input_materials: list of strings
- output_products: list of strings
- process_conditions: object (temp, pressure, residence_time, etc.)

Text: {text}
Return ONLY valid JSON.
""",
    "lci": """
Extract Life Cycle Inventory (LCI) data for environmental impact assessment.
Return JSON with these fields (use null if not found):
- functional_unit: e.g., "1 kg PG", "1 tonne PG"
- energy_consumption: object mapping source
  (electricity, natural_gas, coal, steam) to value_with_unit
- chemical_consumption: list of objects {{ "name": string, "amount": float, "unit": string }}
- water_consumption: value_with_unit
- emissions_air: list of objects {{ "substance": string, "amount": float, "unit": string }}
  (e.g., CO2, SO2, NOx, HF, PM)
- emissions_water: list of objects {{ "substance": string, "amount": float, "unit": string }}
  (e.g., P, F, COD, heavy metals)
- solid_waste: list of objects {{ "type": string, "amount": float, "unit": string }}
- yield: product yield (percentage or mass/mass)

Text: {text}
Return ONLY valid JSON.
""",
    "cost": """
Extract Techno-Economic Analysis (TEA) and cost data.
Return JSON with these fields (use null if not found):
- capex: Capital Expenditure (value + currency + base year)
- opex: Operating Expenditure (value + currency + unit per functional unit)
- revenue: Product selling price/revenue (value + currency + unit)
- payback_period: years
- irr: Internal Rate of Return (%)
- npv: Net Present Value (value + currency)
- chemical_cost: value + currency + unit
- energy_cost: value + currency + unit
- labor_cost: value + currency + unit

Text: {text}
Return ONLY valid JSON.
""",
    "policy": """
Extract policy, regulatory, and market context data for risk assessment.
Return JSON with these fields (use null if not found):
- subsidy_dependency: textual evidence of subsidies, grants, or tax incentives
- carbon_credits: mention of carbon trading, credits, or taxes (ETS)
- regulatory_status: "permitted", "restricted", "banned", or related description
- standards_compliance: list of standards mentioned (e.g., "GB 8978-1996", "ASTM C1398")
- trade_exposure: mention of import/export, international market, or local-only scope
- market_status: "growing", "stable", "declining"

Text: {text}
Return ONLY valid JSON.
""",
}


class LLMExtractor:
    """
    Extracts structured data from text using LLMs.

    Uses an OpenAI-compatible Chat Completions API (Ollama, OpenAI, vLLM, etc.).
    Configure via LLM_BASE_URL, LLM_API_KEY, LLM_MODEL or constructor arguments.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """
        Initialize the LLM extractor.

        Args:
            model: Model name (default: LLM_MODEL env or qwen3.5:35b)
            api_key: API key (default: LLM_API_KEY env or "ollama")
            base_url: API base URL (default: LLM_BASE_URL env or http://127.0.0.1:11434/v1)
        """
        self.model = model or os.getenv("LLM_MODEL", _DEFAULT_MODEL)
        self.base_url = base_url or os.getenv("LLM_BASE_URL", _DEFAULT_BASE_URL)
        self.api_key = api_key or os.getenv("LLM_API_KEY", "ollama")
        self.llm_context_length = _read_env_int(
            "LLM_CONTEXT_LENGTH", "OLLAMA_CONTEXT_LENGTH", default=0
        )
        self.prefer_json_mode = os.getenv("LLM_JSON_MODE", "1").lower() not in {"0", "false", "off"}
        self._client = None

    def _get_client(self):
        """Get or create the OpenAI-compatible client."""
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        return self._client

    def _build_extra_body(self, existing: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Attach optional Ollama context hints without discarding caller data."""
        extra_body: Dict[str, Any] = dict(existing or {})
        if self.llm_context_length > 0:
            options = dict(extra_body.get("options") or {})
            options.setdefault("num_ctx", self.llm_context_length)
            extra_body["options"] = options
        return extra_body or None

    def extract(
        self, text: str, extraction_type: str, custom_prompt: Optional[str] = None
    ) -> ExtractionResult:
        """
        Extract structured data from text.

        Args:
            text: Input text to extract from
            extraction_type: Type of data (composition, technology, lci, cost)
            custom_prompt: Optional custom extraction prompt

        Returns:
            ExtractionResult with extracted data
        """
        if extraction_type not in EXTRACTION_PROMPTS and not custom_prompt:
            raise ValueError(f"Unknown extraction type: {extraction_type}")

        prompt = custom_prompt or EXTRACTION_PROMPTS[extraction_type]
        prompt = prompt.format(text=text)

        try:
            client = self._get_client()

            print(f"DEBUG: LLM extraction (model={self.model}, base_url={self.base_url})")
            request_kwargs: Dict[str, Any] = dict(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a data extraction assistant specialized in Life Cycle "
                            "Assessment and Techno-Economic Analysis of phosphogypsum "
                            "treatment. Extract structured data accurately. "
                            "Return only a valid JSON object. No markdown, no explanations."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=2000,
            )
            extra_body = self._build_extra_body()
            if extra_body:
                request_kwargs["extra_body"] = extra_body

            # Prefer strict JSON mode when supported. If backend does not support
            # response_format, transparently retry without it.
            if self.prefer_json_mode:
                request_kwargs["response_format"] = {"type": "json_object"}

            try:
                response = client.chat.completions.create(**request_kwargs)
            except Exception:
                if "response_format" in request_kwargs:
                    request_kwargs.pop("response_format", None)
                    response = client.chat.completions.create(**request_kwargs)
                else:
                    raise

            raw_response = response.choices[0].message.content

            data, errors = self._parse_json_response(raw_response)

            return ExtractionResult(
                success=len(errors) == 0,
                data=data,
                raw_response=raw_response,
                confidence=0.8 if len(errors) == 0 else 0.5,
                errors=errors,
            )

        except Exception as e:
            return ExtractionResult(
                success=False, data={}, raw_response="", confidence=0.0, errors=[str(e)]
            )

    def _parse_json_response(self, response: str) -> tuple[Dict, List[str]]:
        """Parse JSON from LLM response."""
        errors = []

        try:
            data = json.loads(response)
            return data, errors
        except json.JSONDecodeError:
            pass

        json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                return data, errors
            except json.JSONDecodeError:
                pass

        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                return data, errors
            except json.JSONDecodeError:
                pass

        errors.append("Failed to parse JSON from LLM response")
        return {}, errors

    def extract_from_document(
        self, document_path: Union[str, Path], extraction_types: List[str] = None
    ) -> Dict[str, ExtractionResult]:
        """
        Extract multiple data types from a document.

        Args:
            document_path: Path to document (text or markdown)
            extraction_types: Types to extract (default: all)

        Returns:
            Dict mapping extraction type to result
        """
        document_path = Path(document_path)

        with open(document_path, "r", encoding="utf-8") as f:
            text = f.read()

        types = extraction_types or list(EXTRACTION_PROMPTS.keys())
        results = {}

        for ext_type in types:
            results[ext_type] = self.extract(text, ext_type)

        return results

    def batch_extract(self, texts: List[str], extraction_type: str) -> List[ExtractionResult]:
        """
        Extract from multiple texts.

        Args:
            texts: List of text inputs
            extraction_type: Type of extraction

        Returns:
            List of extraction results
        """
        results = []
        for text in texts:
            result = self.extract(text, extraction_type)
            results.append(result)
        return results
