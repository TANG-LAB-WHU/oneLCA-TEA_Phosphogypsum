"""
Decision Criteria Module

Defines criteria for multi-criteria decision analysis.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


class Direction(Enum):
    """Optimization direction for a criterion."""
    MINIMIZE = "minimize"  # Lower is better (e.g., cost, emissions)
    MAXIMIZE = "maximize"  # Higher is better (e.g., NPV, revenue)


class Category(Enum):
    """Criterion category."""
    ENVIRONMENTAL = "environmental"
    ECONOMIC = "economic"
    TECHNICAL = "technical"
    SOCIAL = "social"
    RISK = "risk"


@dataclass
class Criterion:
    """
    A decision criterion with weight and direction.
    """
    
    name: str
    weight: float  # 0-1, weights should sum to 1
    direction: Direction
    category: Category
    unit: str = ""
    description: str = ""
    
    # Optional bounds for normalization
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    
    def normalize(self, value: float, all_values: List[float] = None) -> float:
        """
        Normalize value to 0-1 scale.
        
        Uses min-max normalization. Direction is handled in MCDA methods.
        """
        if self.min_value is not None and self.max_value is not None:
            min_val, max_val = self.min_value, self.max_value
        elif all_values:
            min_val, max_val = min(all_values), max(all_values)
        else:
            return value  # Cannot normalize
        
        if max_val == min_val:
            return 0.5
        
        return (value - min_val) / (max_val - min_val)


@dataclass
class CriteriaSet:
    """
    A set of criteria for decision analysis.
    """
    
    criteria: List[Criterion] = field(default_factory=list)
    
    def add(self, criterion: Criterion) -> None:
        self.criteria.append(criterion)
    
    def get_by_name(self, name: str) -> Optional[Criterion]:
        for c in self.criteria:
            if c.name == name:
                return c
        return None
    
    def get_by_category(self, category: Category) -> List[Criterion]:
        return [c for c in self.criteria if c.category == category]
    
    @property
    def total_weight(self) -> float:
        return sum(c.weight for c in self.criteria)
    
    def normalize_weights(self) -> None:
        """Normalize weights to sum to 1."""
        total = self.total_weight
        if total > 0:
            for c in self.criteria:
                c.weight /= total
    
    @property
    def names(self) -> List[str]:
        return [c.name for c in self.criteria]
    
    @property
    def weights(self) -> Dict[str, float]:
        return {c.name: c.weight for c in self.criteria}


def create_default_criteria() -> CriteriaSet:
    """
    Create default criteria set for PG pathway selection.
    
    Balanced across environmental, economic, and risk factors.
    """
    criteria = CriteriaSet()
    
    # Environmental criteria (30%)
    criteria.add(Criterion(
        name="gwp",
        weight=0.15,
        direction=Direction.MINIMIZE,
        category=Category.ENVIRONMENTAL,
        unit="kg CO2-eq/t PG",
        description="Climate change impact",
    ))
    criteria.add(Criterion(
        name="resource_depletion",
        weight=0.08,
        direction=Direction.MINIMIZE,
        category=Category.ENVIRONMENTAL,
        unit="kg Sb-eq/t PG",
        description="Abiotic resource depletion",
    ))
    criteria.add(Criterion(
        name="human_toxicity",
        weight=0.07,
        direction=Direction.MINIMIZE,
        category=Category.ENVIRONMENTAL,
        unit="CTUh/t PG",
        description="Human toxicity potential",
    ))
    
    # Economic criteria (40%)
    criteria.add(Criterion(
        name="npv",
        weight=0.20,
        direction=Direction.MAXIMIZE,
        category=Category.ECONOMIC,
        unit="USD/t PG",
        description="Net present value",
    ))
    criteria.add(Criterion(
        name="irr",
        weight=0.10,
        direction=Direction.MAXIMIZE,
        category=Category.ECONOMIC,
        unit="%",
        description="Internal rate of return",
    ))
    criteria.add(Criterion(
        name="payback",
        weight=0.10,
        direction=Direction.MINIMIZE,
        category=Category.ECONOMIC,
        unit="years",
        description="Payback period",
    ))
    
    # Technical criteria (15%)
    criteria.add(Criterion(
        name="trl",
        weight=0.10,
        direction=Direction.MAXIMIZE,
        category=Category.TECHNICAL,
        unit="1-9",
        description="Technology readiness level",
    ))
    criteria.add(Criterion(
        name="scalability",
        weight=0.05,
        direction=Direction.MAXIMIZE,
        category=Category.TECHNICAL,
        unit="score",
        description="Scale-up feasibility",
    ))
    
    # Risk criteria (15%)
    criteria.add(Criterion(
        name="overall_risk",
        weight=0.15,
        direction=Direction.MINIMIZE,
        category=Category.RISK,
        unit="0-100",
        description="Aggregated risk score",
    ))
    
    return criteria
