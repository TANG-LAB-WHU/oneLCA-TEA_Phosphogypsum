"""
Annotations Module

Data source tracking, assumption documentation, and metadata management.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class SourceType(Enum):
    """Type of data source."""

    LITERATURE = "literature"
    DATABASE = "database"
    CALCULATION = "calculation"
    ESTIMATION = "estimation"
    DEFAULT = "default"
    USER_INPUT = "user_input"


class UncertaintyLevel(Enum):
    """Uncertainty classification."""

    LOW = "low"  # CV < 10%
    MEDIUM = "medium"  # 10% < CV < 30%
    HIGH = "high"  # 30% < CV < 50%
    VERY_HIGH = "very_high"  # CV > 50%


@dataclass
class DataSource:
    """
    Data source metadata for traceability.
    """

    source_type: SourceType
    reference: str  # Citation or database name
    year: int = 2024
    url: Optional[str] = None
    doi: Optional[str] = None
    notes: str = ""

    def to_citation(self) -> str:
        """Generate citation string."""
        if self.doi:
            return f"{self.reference} ({self.year}). DOI: {self.doi}"
        elif self.url:
            return f"{self.reference} ({self.year}). URL: {self.url}"
        else:
            return f"{self.reference} ({self.year})"


@dataclass
class Assumption:
    """
    Assumption documentation.
    """

    description: str
    justification: str = ""
    impact: str = ""  # High/Medium/Low impact on results
    alternatives: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "description": self.description,
            "justification": self.justification,
            "impact": self.impact,
            "alternatives": self.alternatives,
        }


@dataclass
class Annotation:
    """
    Complete annotation for a data value.
    """

    value: Any
    unit: str = ""
    source: Optional[DataSource] = None
    assumptions: List[Assumption] = field(default_factory=list)
    uncertainty: Optional[float] = None  # Coefficient of variation
    uncertainty_level: Optional[UncertaintyLevel] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Auto-classify uncertainty level."""
        if self.uncertainty is not None and self.uncertainty_level is None:
            cv = abs(self.uncertainty)
            if cv < 0.10:
                self.uncertainty_level = UncertaintyLevel.LOW
            elif cv < 0.30:
                self.uncertainty_level = UncertaintyLevel.MEDIUM
            elif cv < 0.50:
                self.uncertainty_level = UncertaintyLevel.HIGH
            else:
                self.uncertainty_level = UncertaintyLevel.VERY_HIGH

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "value": self.value,
            "unit": self.unit,
            "source": self.source.to_citation() if self.source else None,
            "uncertainty": self.uncertainty,
            "uncertainty_level": self.uncertainty_level.value if self.uncertainty_level else None,
            "assumptions_count": len(self.assumptions),
        }


def annotate(
    value: Any,
    unit: str = "",
    source: str = None,
    source_type: SourceType = SourceType.DEFAULT,
    year: int = 2024,
    uncertainty: float = None,
    **metadata,
) -> Annotation:
    """
    Quick annotation helper function.

    Args:
        value: The data value
        unit: Unit of measurement
        source: Source reference
        source_type: Type of source
        year: Source year
        uncertainty: Coefficient of variation
        **metadata: Additional metadata

    Returns:
        Annotation object
    """
    data_source = None
    if source:
        data_source = DataSource(
            source_type=source_type,
            reference=source,
            year=year,
        )

    return Annotation(
        value=value,
        unit=unit,
        source=data_source,
        uncertainty=uncertainty,
        metadata=metadata,
    )


# Predefined common data sources
COMMON_SOURCES = {
    "ELCD": DataSource(
        source_type=SourceType.DATABASE,
        reference="European Reference Life Cycle Database (ELCD)",
        url="https://eplca.jrc.ec.europa.eu/ELCD3/",
    ),
    "USLCI": DataSource(
        source_type=SourceType.DATABASE,
        reference="US Life Cycle Inventory Database",
        url="https://www.nrel.gov/lci/",
    ),
    "ecoinvent": DataSource(
        source_type=SourceType.DATABASE,
        reference="ecoinvent Database v3.9",
        url="https://ecoinvent.org/",
    ),
    "EPA": DataSource(
        source_type=SourceType.DATABASE,
        reference="US Environmental Protection Agency",
        url="https://www.epa.gov/",
    ),
    "IPCC": DataSource(
        source_type=SourceType.LITERATURE,
        reference="IPCC Sixth Assessment Report",
        year=2023,
        doi="10.1017/9781009157926",
    ),
}
