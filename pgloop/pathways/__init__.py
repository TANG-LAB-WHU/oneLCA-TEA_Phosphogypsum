"""
Pathways Module Registry

Strictly aligned with the Implementation Plan.
"""

from typing import Dict, List, Type

from pgloop.pathways.base_pathway import BasePathway
from pgloop.pathways.pg_cement import CementPathway
from pgloop.pathways.pg_chemical_recovery import ChemicalRecoveryPathway
from pgloop.pathways.pg_construction import ConstructionMaterialsPathway
from pgloop.pathways.pg_ree_extraction import REEExtractionPathway
from pgloop.pathways.pg_soil_amendment import SoilAmendmentPathway
from pgloop.pathways.pg_stack_disposal import StackDisposalPathway

# Registry of available pathways
PATHWAYS: Dict[str, Type[BasePathway]] = {
    "PG-Stack": StackDisposalPathway,
    "PG-CementProd": CementPathway,
    "PG-ConstructMat": ConstructionMaterialsPathway,
    "PG-Soil": SoilAmendmentPathway,
    "PG-ChemReco": ChemicalRecoveryPathway,
    "PG-REEextract": REEExtractionPathway,
}


def get_pathway(code: str, **kwargs) -> BasePathway:
    """Factory function to get a pathway instance by code."""
    if code not in PATHWAYS:
        raise ValueError(f"Pathway code {code} not found. Available: {list_pathways()}")
    return PATHWAYS[code](**kwargs)


def list_pathways() -> List[str]:
    """List all registered pathway codes."""
    return list(PATHWAYS.keys())


__all__ = [
    "BasePathway",
    "StackDisposalPathway",
    "CementPathway",
    "ConstructionMaterialsPathway",
    "SoilAmendmentPathway",
    "ChemicalRecoveryPathway",
    "REEExtractionPathway",
    "PATHWAYS",
    "get_pathway",
    "list_pathways",
]
