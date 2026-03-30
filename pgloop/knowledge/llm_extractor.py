"""
LLM Extractor Module

Uses Large Language Models to extract structured LCA/TEA data from text.
Supports multiple LLM backends (Gemini, OpenAI-compatible, local models).
"""

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


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

    Supports:
    - OpenAI API compatible endpoints
    - Google Gemini
    - Local models via Ollama
    """

    def __init__(
        self,
        provider: str = "gemini",
        model: str = "gemini-2.0-flash",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """
        Initialize the LLM extractor.

        Args:
            provider: LLM provider (openai, gemini, ollama)
            model: Model name
            api_key: API key (if required)
            base_url: Custom API base URL
        """
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self._client = None

    def _get_client(self):
        """Get or create the LLM client."""
        if self._client is None:
            if self.provider == "gemini":
                try:
                    from google import genai

                    # Check for proxy settings
                    proxy = (
                        os.environ.get("HTTPS_PROXY")
                        or os.environ.get("http_proxy")
                        or os.environ.get("https_proxy")
                    )

                    client_kwargs = {"api_key": self.api_key}

                    if proxy:
                        print(f"DEBUG: Proxy detected for Gemini: {proxy}")
                        # New SDK uses http_options for transport configuration

                    print(f"DEBUG: Initializing google.genai Client (model={self.model})")
                    self._client = genai.Client(**client_kwargs)
                except ImportError:
                    raise ImportError(
                        "google-genai package not installed. Run: pip install google-genai"
                    )
                except Exception as e:
                    raise RuntimeError(f"Failed to initialize Gemini client: {e}")

            elif self.provider == "openai":
                try:
                    from openai import OpenAI

                    kwargs = {}
                    if self.api_key:
                        kwargs["api_key"] = self.api_key
                    if self.base_url:
                        kwargs["base_url"] = self.base_url

                    self._client = OpenAI(**kwargs)
                except ImportError:
                    raise ImportError("openai package not installed. Run: pip install openai")

            elif self.provider == "ollama":
                try:
                    from openai import OpenAI

                    self._client = OpenAI(
                        base_url="http://localhost:11434/v1",
                        api_key="ollama",  # Required but not used
                    )
                except ImportError:
                    raise ImportError("openai package not installed. Run: pip install openai")

        return self._client

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

            if self.provider == "gemini":
                # New google.genai SDK call
                print(f"DEBUG: Calling google.genai generate_content (provider={self.provider})")
                response = client.models.generate_content(model=self.model, contents=prompt)
                raw_response = response.text
            else:
                # OpenAI-compatible call
                print(
                    f"DEBUG: Calling OpenAI-compatible chat completions (provider={self.provider})"
                )
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a data extraction assistant specialized in Life Cycle "
                                "Assessment and Techno-Economic Analysis of phosphogypsum "
                                "treatment. Extract structured data accurately."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,
                    max_tokens=2000,
                )
                raw_response = response.choices[0].message.content

            # Parse JSON from response
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

        # Try direct JSON parsing
        try:
            data = json.loads(response)
            return data, errors
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from markdown code blocks
        json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                return data, errors
            except json.JSONDecodeError:
                pass

        # Try to find JSON object in response
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
