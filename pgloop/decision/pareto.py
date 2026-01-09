"""
Pareto Analysis Module

Identifies Pareto-optimal (non-dominated) solutions.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set
from pgloop.decision.criteria import Direction


@dataclass
class ParetoSolution:
    """A solution in the Pareto analysis."""
    
    name: str
    objectives: Dict[str, float]  # criterion -> value
    is_dominated: bool = False
    dominators: List[str] = field(default_factory=list)


class ParetoAnalyzer:
    """
    Pareto optimality analyzer.
    
    Identifies non-dominated solutions in multi-objective space.
    """
    
    def __init__(self, directions: Dict[str, Direction] = None):
        """
        Initialize analyzer.
        
        Args:
            directions: {criterion_name: Direction} (default: minimize all)
        """
        self.directions = directions or {}
    
    def _dominates(
        self,
        a: Dict[str, float],
        b: Dict[str, float],
        criteria: List[str]
    ) -> bool:
        """
        Check if solution a dominates solution b.
        
        a dominates b if:
        - a is at least as good as b in all criteria
        - a is strictly better in at least one criterion
        """
        at_least_as_good = True
        strictly_better = False
        
        for crit in criteria:
            a_val = a.get(crit, 0)
            b_val = b.get(crit, 0)
            
            # Handle None
            a_val = 0 if a_val is None else a_val
            b_val = 0 if b_val is None else b_val
            
            direction = self.directions.get(crit, Direction.MINIMIZE)
            
            if direction == Direction.MINIMIZE:
                if a_val > b_val:
                    at_least_as_good = False
                    break
                if a_val < b_val:
                    strictly_better = True
            else:  # MAXIMIZE
                if a_val < b_val:
                    at_least_as_good = False
                    break
                if a_val > b_val:
                    strictly_better = True
        
        return at_least_as_good and strictly_better
    
    def find_pareto_front(
        self,
        solutions: Dict[str, Dict[str, float]]
    ) -> List[ParetoSolution]:
        """
        Find Pareto-optimal solutions.
        
        Args:
            solutions: {name: {criterion: value}}
            
        Returns:
            List of ParetoSolution objects
        """
        names = list(solutions.keys())
        criteria = list(set().union(*[set(v.keys()) for v in solutions.values()]))
        
        results = []
        
        for name in names:
            solution = ParetoSolution(
                name=name,
                objectives=solutions[name].copy(),
            )
            
            for other_name in names:
                if other_name == name:
                    continue
                
                if self._dominates(solutions[other_name], solutions[name], criteria):
                    solution.is_dominated = True
                    solution.dominators.append(other_name)
            
            results.append(solution)
        
        return results
    
    def get_pareto_optimal(
        self,
        solutions: Dict[str, Dict[str, float]]
    ) -> List[str]:
        """Get names of Pareto-optimal solutions."""
        pareto = self.find_pareto_front(solutions)
        return [s.name for s in pareto if not s.is_dominated]
    
    def calculate_crowding_distance(
        self,
        solutions: Dict[str, Dict[str, float]],
        pareto_names: List[str]
    ) -> Dict[str, float]:
        """
        Calculate crowding distance for Pareto solutions.
        
        Used for tie-breaking in selection.
        """
        if len(pareto_names) <= 2:
            return {name: float('inf') for name in pareto_names}
        
        criteria = list(set().union(*[set(solutions[n].keys()) for n in pareto_names]))
        distances = {name: 0.0 for name in pareto_names}
        
        for crit in criteria:
            # Sort by this criterion
            sorted_names = sorted(
                pareto_names,
                key=lambda n: solutions[n].get(crit, 0)
            )
            
            # Boundary points get infinite distance
            distances[sorted_names[0]] = float('inf')
            distances[sorted_names[-1]] = float('inf')
            
            # Interior points
            values = [solutions[n].get(crit, 0) for n in sorted_names]
            val_range = max(values) - min(values) if max(values) != min(values) else 1
            
            for i in range(1, len(sorted_names) - 1):
                distances[sorted_names[i]] += (values[i+1] - values[i-1]) / val_range
        
        return distances
