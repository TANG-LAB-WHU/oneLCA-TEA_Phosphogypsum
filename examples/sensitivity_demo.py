"""
Sensitivity and Uncertainty Propagation Demo for PhosphogypsumBot.

This script demonstrates:
1. One-At-a-Time (OAT) local sensitivity analysis.
2. Monte Carlo propagation of parameter uncertainties through both LCA and TEA.
3. Sobol global sensitivity analysis using SALib (if installed).
"""

import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from pgloop import LCAEngine, TEAEngine, get_pathway
from pgloop.uncertainty.propagation import JointUncertaintyPropagator
from pgloop.uncertainty.sensitivity import SensitivityAnalyzer


def main():
    print("=" * 60)
    print("   PhosphogypsumBot: Sensitivity & Uncertainty Propagation   ")
    print("=" * 60)

    # 1. Initialize engines and load the Cement Production Pathway
    lca_engine = LCAEngine()
    tea_engine = TEAEngine(country="China")
    pathway = get_pathway("PG-CementProd")
    print(f"\nPathway under analysis: {pathway.name} [{pathway.code}]")

    # 2. Local Sensitivity Analysis: One-At-a-Time (OAT)
    print("\n--- Running Local One-At-a-Time (OAT) Sensitivity ---")
    analyzer = SensitivityAnalyzer(variation=0.1)  # ±10% variation

    # We define a helper evaluator function that takes a modified parameter dict
    # and returns the calculated conventional life cycle cost (CLCC)
    def clcc_evaluator(param_dict):
        temp_pathway = pathway.copy_with_parameters(param_dict)
        res = tea_engine.calculate(temp_pathway, include_uncertainty=False)
        return {"clcc": float(res.clcc)}

    oat_results = analyzer.oat_analysis(
        parameters=pathway.parameters,
        calculation_func=clcc_evaluator,
        output_name="clcc"
    )

    print("\nOAT Sensitivity Results (Target: CLCC):")
    for r in oat_results[:5]:
        print(f"  Rank {r.importance_rank:2} | Parameter: {r.parameter:28} | Elasticity: {r.elasticity:+.4f}")

    # 3. Monte Carlo Joint Uncertainty Propagation
    print("\n--- Running Joint Monte Carlo Propagation (100 iterations) ---")
    propagator = JointUncertaintyPropagator(
        lca_engine=lca_engine,
        tea_engine=tea_engine,
        n_iterations=100,  # low count for demo speed
        seed=42
    )

    # Propagate default distributions
    propagation_result = propagator.propagate(pathway)
    
    print("\nUncertainty Propagation Summary statistics:")
    for metric_name, stats in propagation_result.summary.items():
        print(f"  {metric_name.upper():12}: Mean = {stats['mean']:10.2f} | Std = {stats['std']:10.2f} | 90% CI = [{stats['p5']:.2f}, {stats['p95']:.2f}]")

    # 4. Global Sensitivity Analysis (Sobol Indices)
    print("\n--- Running Global Sensitivity Analysis (Sobol Method) ---")
    try:
        # Define the problem bounds from pathway parameters
        distributions = pathway.get_parameter_distributions()
        names = []
        bounds = []
        for name, spec in distributions.items():
            if spec.get("type") == "triangular":
                names.append(name)
                bounds.append([spec["min"], spec["max"]])
            elif spec.get("type") == "normal":
                names.append(name)
                bounds.append([spec["mean"] - 3 * spec["std"], spec["mean"] + 3 * spec["std"]])

        problem = {
            "num_vars": len(names),
            "names": names,
            "bounds": bounds
        }

        # Model evaluator function for Sobol sampler
        def model_eval_fn(X_samples):
            Y_outputs = []
            for row in X_samples:
                param_dict = dict(zip(names, row))
                # Evaluate GWP
                temp_pathway = pathway.copy_with_parameters(param_dict)
                res = lca_engine.calculate(temp_pathway, include_uncertainty=False)
                Y_outputs.append(res.impacts.get("climate_change", 0.0))
            return np.array(Y_outputs)

        sobol_results = analyzer.sobol_analysis(
            problem=problem,
            model_eval_fn=model_eval_fn,
            n_samples=32  # low count for demo speed
        )

        print("\nSobol First-Order Sensitivity Indices (Target: GWP):")
        # Sort by index magnitude
        sorted_indices = sorted(sobol_results["S1"].items(), key=lambda item: abs(item[1]), reverse=True)
        for name, s1 in sorted_indices[:5]:
            print(f"  - Parameter: {name:28} | S1 Index: {s1:.4f}")

    except ImportError:
        print("  [SKIP] SALib is not installed in this environment. Skipping Sobol analysis.")

    print("\n" + "=" * 60)
    print("Sensitivity Demo Completed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
