"""
Integrated Assessment Module (Deep Module)

Provides the high-level facade IntegratedAssessmentEngine for end-to-end
multi-dimensional sustainability evaluation, combining LCA, TEA, VPM kinetics,
risk aggregation, and MCDA ranking.
"""

from pgloop.integrated_assessment.engine import IntegratedAssessmentEngine

__all__ = ["IntegratedAssessmentEngine"]
