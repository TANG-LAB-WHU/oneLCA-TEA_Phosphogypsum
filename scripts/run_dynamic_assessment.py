"""
Run dynamic uncertainty-aware LCA-TEA assessment and save artifacts.
"""

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import yaml

from pgloop import LCAEngine, TEAEngine, get_pathway
from pgloop.decision.dynamic_optimizer import DynamicMultiObjectiveOptimizer
from pgloop.decision.scenario import DynamicScenarioAnalyzer, Scenario
from pgloop.stochastic_dynamics import stochastic_density_summary_from_timeseries
from pgloop.uncertainty.bayesian_update import BayesianUpdater
from pgloop.utils.schema import DynamicAssessmentResult, PosteriorSummary, TimeSeriesPoint

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Dynamic uncertainty-aware LCA-TEA assessment runner."
    )
    parser.add_argument(
        "--pathways",
        nargs="+",
        default=["PG-CementProd", "PG-REEextract"],
        help="Pathway codes to evaluate.",
    )
    parser.add_argument("--scenario", default="baseline", help="Scenario name from YAML.")
    parser.add_argument("--start-year", type=int, default=2025)
    parser.add_argument("--end-year", type=int, default=2040)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--scenario-config",
        default=str(ROOT / "config" / "dynamic_scenarios.yaml"),
        help="Path to dynamic scenario configuration YAML.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "data" / "processed" / "dynamic_assessment"),
        help="Output directory for JSON/CSV artifacts.",
    )
    parser.add_argument(
        "--prior-config",
        default=str(ROOT / "config" / "uncertainty_priors.yaml"),
        help="Path to uncertainty prior YAML used in Bayesian closed-loop updates.",
    )
    parser.add_argument(
        "--country",
        default="China",
        help="Country context for pathways and TEA engine.",
    )
    parser.add_argument(
        "--enable-bayes-update",
        action="store_true",
        help="Enable Bayesian update loop after dynamic run.",
    )
    return parser.parse_args()


def load_scenario(config_path: Path, scenario_name: str) -> Scenario:
    with open(config_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    scenarios = raw.get("dynamic_scenarios", {})
    if scenario_name not in scenarios:
        raise ValueError(
            f"Scenario '{scenario_name}' not found in {config_path}. "
            f"Available: {list(scenarios.keys())}"
        )
    cfg = scenarios[scenario_name]
    return Scenario(
        name=scenario_name,
        description=cfg.get("description", ""),
        trajectory=cfg.get("trajectory", {}),
    )


def load_priors(config_path: Path) -> Dict[str, Dict]:
    with open(config_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return raw.get("priors", {})


def save_json(path: Path, payload: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)


def save_csv(path: Path, rows: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()

    lca_engine = LCAEngine()
    tea_engine = TEAEngine(country=args.country)
    dynamic_analyzer = DynamicScenarioAnalyzer(lca_engine=lca_engine, tea_engine=tea_engine)
    optimizer = DynamicMultiObjectiveOptimizer()

    scenario = load_scenario(Path(args.scenario_config), args.scenario)
    priors = load_priors(Path(args.prior_config))

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    all_results = []
    flat_rows = []

    for pathway_code in args.pathways:
        pathway = get_pathway(pathway_code, country=args.country)
        yearly = dynamic_analyzer.run(
            pathway=pathway,
            scenario=scenario,
            start_year=args.start_year,
            end_year=args.end_year,
            n_samples=args.samples,
            seed=args.seed,
        )
        robustness = dynamic_analyzer.summarize(yearly, metric="clcc")

        time_series = [TimeSeriesPoint(year=r.year, metrics=r.metrics) for r in yearly]

        posterior = {}
        for metric in ["gwp", "clcc", "slcc", "lcop", "carbon_cost"]:
            values = [r.metrics.get(metric, 0.0) for r in yearly]
            if values:
                sorted_vals = sorted(values)
                p5 = sorted_vals[max(0, int(0.05 * (len(sorted_vals) - 1)))]
                p50 = sorted_vals[int(0.50 * (len(sorted_vals) - 1))]
                p95 = sorted_vals[int(0.95 * (len(sorted_vals) - 1))]
                posterior[metric] = PosteriorSummary(
                    mean=sum(values) / len(values),
                    std=(sum((v - (sum(values) / len(values))) ** 2 for v in values) / len(values))
                    ** 0.5,
                    p5=float(p5),
                    p50=float(p50),
                    p95=float(p95),
                )

        dyn_result = DynamicAssessmentResult(
            pathway_code=pathway_code,
            scenario_name=scenario.name,
            time_series_metrics=time_series,
            posterior_summary=posterior,
            robustness_stats=robustness,
            stochastic_density_summary=stochastic_density_summary_from_timeseries(
                [asdict(ts) for ts in time_series]
            ),
        )
        all_results.append(dyn_result)

        for r in yearly:
            flat_rows.append(
                {
                    "pathway": pathway_code,
                    "scenario": scenario.name,
                    "year": r.year,
                    **r.metrics,
                }
            )

    ranking = optimizer.rank_pathways(
        dynamic_results=[asdict(r) for r in all_results],
        weights={"entropy_proxy": 0.3, "lcop": 0.35, "gwp": 0.35},
    )

    payload = {
        "meta": {
            "scenario": args.scenario,
            "start_year": args.start_year,
            "end_year": args.end_year,
            "samples": args.samples,
            "seed": args.seed,
            "bayesian_update": args.enable_bayes_update,
        },
        "results": [asdict(r) for r in all_results],
        "ranking": ranking,
    }

    if args.enable_bayes_update and all_results:
        updater = BayesianUpdater(observation_noise=0.15)
        pathway = get_pathway(args.pathways[0], country=args.country)
        observations = {
            "gwp": all_results[0].posterior_summary.get("gwp", PosteriorSummary(0, 0, 0, 0, 0)).p50,
            "clcc": all_results[0]
            .posterior_summary.get("clcc", PosteriorSummary(0, 0, 0, 0, 0))
            .p50,
        }
        payload["bayesian_closed_loop"] = updater.run_closed_loop(
            pathway=pathway,
            lca_engine=lca_engine,
            tea_engine=tea_engine,
            priors=priors,
            observations=observations,
            n_iterations=max(150, args.samples // 2),
        )

    save_json(out_root / "dynamic_assessment.json", payload)
    save_csv(out_root / "dynamic_assessment_timeseries.csv", flat_rows)
    save_json(out_root / "dynamic_assessment_ranking.json", {"ranking": ranking})

    print(f"Saved dynamic assessment outputs to: {out_root}")


if __name__ == "__main__":
    main()
