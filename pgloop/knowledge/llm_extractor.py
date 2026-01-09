"""
LLM Extractor Module

Uses Large Language Models to extract structured LCA/TEA data from text.
Supports multiple LLM backends (Gemini, OpenAI-compatible, local models).
"""

import json
import re
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from pathlib import Path


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
Extract phosphogypsum composition data from the following text.
Return a JSON object with these fields (use null if not found):
- CaSO4: mass fraction (0-1)
- P2O5: mass fraction (0-1)  
- F: mass fraction (0-1)
- SiO2: mass fraction (0-1)
- Fe2O3: mass fraction (0-1)
- Al2O3: mass fraction (0-1)
- moisture: mass fraction (0-1)
- Ra226: radioactivity in Bq/kg
- heavy_metals: dict of metal -> concentration in mg/kg (Cd, Pb, As, Hg, etc.)
- country: country of origin
- source: data source reference

Text:
{text}

Return ONLY valid JSON, no other text.
""",
    
    "technology": """
Extract treatment technology data from the following text.
Return a JSON object with these fields (use null if not found):
- name: technology name
- description: brief description
- trl: technology readiness level (1-9)
- capacity: typical capacity in tonnes/year
- inputs: list of {material, quantity, unit}
- outputs: list of {product, quantity, unit}
- emissions: list of {pollutant, quantity, unit, compartment}
- energy_consumption: in MJ/tonne PG
- costs: {capex_usd, opex_usd_per_tonne}
- country: where implemented
- source: data source reference

Text:
{text}

Return ONLY valid JSON, no other text.
""",

    "lci": """
Extract Life Cycle Inventory data from the following text.
Return a JSON object with:
- functional_unit: description of functional unit
- inputs: list of {name, quantity, unit, source}
- outputs: list of {name, quantity, unit, source}
- emissions_air: list of {pollutant, quantity, unit}
- emissions_water: list of {pollutant, quantity, unit}
- emissions_soil: list of {pollutant, quantity, unit}
- energy: {electricity_kwh, heat_mj, fuel_type, fuel_quantity}
- source: reference

Text:
{text}

Return ONLY valid JSON, no other text.
""",

    "cost": """
Extract cost data from the following text.
Return a JSON object with:
- capex: capital expenditure in USD
- opex: operational cost in USD/year or USD/tonne
- labor_cost: in USD/year or USD/tonne
- material_costs: list of {material, cost, unit}
- energy_costs: in USD/year or USD/tonne
- revenue: from products in USD/year or USD/tonne
- products: list of {product, price, unit}
- year: cost data year
- currency: original currency if not USD
- country: country context
- source: reference

Text:
{text}

Return ONLY valid JSON, no other text.
"""
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
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
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
            if self.provider in ["openai", "gemini"]:
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
                        api_key="ollama"  # Required but not used
                    )
                except ImportError:
                    raise ImportError("openai package not installed. Run: pip install openai")
        
        return self._client
    
    def extract(
        self,
        text: str,
        extraction_type: str,
        custom_prompt: Optional[str] = None
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
            
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a data extraction assistant specialized in Life Cycle Assessment and Techno-Economic Analysis of phosphogypsum treatment. Extract structured data accurately."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
            raw_response = response.choices[0].message.content
            
            # Parse JSON from response
            data, errors = self._parse_json_response(raw_response)
            
            return ExtractionResult(
                success=len(errors) == 0,
                data=data,
                raw_response=raw_response,
                confidence=0.8 if len(errors) == 0 else 0.5,
                errors=errors
            )
            
        except Exception as e:
            return ExtractionResult(
                success=False,
                data={},
                raw_response="",
                confidence=0.0,
                errors=[str(e)]
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
        self,
        document_path: Union[str, Path],
        extraction_types: List[str] = None
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
    
    def batch_extract(
        self,
        texts: List[str],
        extraction_type: str
    ) -> List[ExtractionResult]:
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


if __name__ == "__main__":
    # Example usage (requires API key)
    extractor = LLMExtractor(
        provider="ollama",
        model="llama3.1"
    )
    
    sample_text = """
    The phosphogypsum sample from China contained 92% CaSO4·2H2O,
    with P2O5 content of 0.8-1.2% and fluoride at 0.5%. 
    The Ra-226 activity was measured at 450 Bq/kg.
    """
    
    # result = extractor.extract(sample_text, "composition")
    # print(result.data)
