"""
Dynamic multi-objective optimization helpers.
"""

from typing import Dict, List

from pgloop.decision.pareto import ParetoAnalyzer


class DynamicMultiObjectiveOptimizer:
    """
    Rank pathways under dynamic uncertainty outputs.
    Objectives are minimized: entropy_proxy, lcop, gwp.
    """

    def build_objectives(self, dynamic_result: Dict) -> Dict[str, float]:
        """Build objective vector from serialized DynamicAssessmentResult."""
        points = dynamic_result.get("time_series_metrics", [])
        if not points:
            return {"entropy_proxy": 0.0, "lcop": 0.0, "gwp": 0.0}

        lcop_vals = [p.get("metrics", {}).get("lcop", 0.0) for p in points]
        gwp_vals = [p.get("metrics", {}).get("gwp", 0.0) for p in points]
        clcc_vals = [p.get("metrics", {}).get("clcc", 0.0) for p in points]
        slcc_vals = [p.get("metrics", {}).get("slcc", 0.0) for p in points]
        carbon_vals = [p.get("metrics", {}).get("carbon_cost", 0.0) for p in points]

        entropy_proxy = self._entropy_proxy(
            clcc_vals=clcc_vals,
            slcc_vals=slcc_vals,
            carbon_cost_vals=carbon_vals,
        )
        return {
            "entropy_proxy": entropy_proxy,
            "lcop": sum(lcop_vals) / len(lcop_vals),
            "gwp": sum(gwp_vals) / len(gwp_vals),
        }

    def rank_pathways(self, dynamic_results: List[Dict], weights: Dict[str, float]) -> List[Dict]:
        """
        Return weighted ranking with Pareto tags.
        """
        objectives = {}
        for result in dynamic_results:
            code = result.get("pathway_code", "unknown")
            objectives[code] = self.build_objectives(result)

        if not objectives:
            return []

        normalized = self._normalize_minimize(objectives)
        scores = {}
        for name, values in normalized.items():
            score = 0.0
            for metric, weight in weights.items():
                score += weight * values.get(metric, 0.0)
            scores[name] = score

        pareto = ParetoAnalyzer().get_pareto_optimal(objectives)
        ordered = sorted(scores.items(), key=lambda item: item[1])

        rows = []
        for idx, (name, score) in enumerate(ordered, start=1):
            rows.append(
                {
                    "rank": idx,
                    "pathway_code": name,
                    "weighted_score": score,
                    "is_pareto_optimal": name in pareto,
                    "objectives": objectives[name],
                }
            )
        return rows

    @staticmethod
    def _normalize_minimize(objectives: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        metrics = list(next(iter(objectives.values())).keys())
        mins = {m: min(v.get(m, 0.0) for v in objectives.values()) for m in metrics}
        maxs = {m: max(v.get(m, 0.0) for v in objectives.values()) for m in metrics}

        normalized = {}
        for name, values in objectives.items():
            normalized[name] = {}
            for metric in metrics:
                lo = mins[metric]
                hi = maxs[metric]
                if hi == lo:
                    normalized[name][metric] = 0.0
                else:
                    normalized[name][metric] = (values[metric] - lo) / (hi - lo)
        return normalized

    @staticmethod
    def _entropy_proxy(
        clcc_vals: List[float], slcc_vals: List[float], carbon_cost_vals: List[float]
    ) -> float:
        """
        Interpretable dissipative-cost proxy for irreversible burden.
        """
        n = len(clcc_vals) or 1
        avg_gap = sum(abs(s - c) for s, c in zip(slcc_vals, clcc_vals)) / n
        avg_carbon = sum(max(0.0, x) for x in carbon_cost_vals) / n
        return avg_gap + 0.2 * avg_carbon
