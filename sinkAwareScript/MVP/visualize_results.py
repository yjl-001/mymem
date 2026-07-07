#!/usr/bin/env python3
"""Create plots for a sink-aware MVP output directory."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from plot_utils import save_bar, save_hist, save_scatter
from viz_io import ensure_dir, family_means, load_json, load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, help="Directory produced by run_kodcode_sink_mvp.py.")
    parser.add_argument("--fig-dir", default=None, help="Defaults to <output-dir>/figures.")
    return parser.parse_args()


def finite_rows(rows: list[dict]) -> list[dict]:
    return [
        row for row in rows
        if row.get("utility") is not None
        and row.get("sink_mass_z") is not None
        and row.get("entropy_z") is not None
    ]


def save_summary_report(summary: dict, fig_dir: Path, output_dir: Path) -> None:
    reward = family_means(summary, "strategy_reward_mean")
    insertions = family_means(summary, "strategy_actual_insertions_mean")
    utility = flatten_candidate_metric(summary, "counterfactual_avg_utility_at_budget")
    utility_eff = flatten_candidate_metric(summary, "counterfactual_avg_utility_at_budget_effective")

    lines = [
        "# Sink-aware MVP Visualization Report",
        "",
        f"Source output: `{output_dir}`",
        "",
        "## Key Numbers",
        "",
        f"- Candidate rows: {summary.get('num_candidate_rows')}",
        f"- Strategy rows: {summary.get('num_strategy_rows')}",
        f"- Candidate references: {list(summary.get('candidate_by_reference', {}).keys())}",
        f"- Budget: {summary.get('budget')}",
        "",
        "## Figures",
        "",
        "- `strategy_reward_mean.png`: complete-rollout reward by strategy family.",
        "- `counterfactual_utility_at_budget.png`: Avg U@B by selector.",
        "- `counterfactual_utility_at_budget_effective.png`: Avg U@B on effective samples.",
        "- `actual_insertions_mean.png`: actual insertion count by strategy family.",
        "- `sink_z_vs_utility.png`: candidate-level SinkMassZ vs utility.",
        "- `entropy_z_vs_utility.png`: candidate-level EntropyZ vs utility.",
        "- `utility_hist.png`: distribution of candidate utilities.",
        "",
        "## Aggregates",
        "",
        f"- Reward means: `{reward}`",
        f"- Utility@B: `{utility}`",
        f"- Utility@B effective only: `{utility_eff}`",
        f"- Actual insertions: `{insertions}`",
    ]
    (fig_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def flatten_candidate_metric(summary: dict, metric: str) -> dict[str, float]:
    values = {}
    for ref, payload in summary.get("candidate_by_reference", {}).items():
        for selector, value in payload.get(metric, {}).items():
            if value is not None:
                values[f"{ref}/{selector}"] = float(value)
    return values


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    fig_dir = Path(args.fig_dir) if args.fig_dir else output_dir / "figures"
    ensure_dir(fig_dir)

    summary = load_json(output_dir / "summary.json")
    candidates = finite_rows(load_jsonl(output_dir / "candidate_rows.jsonl"))

    save_bar(
        family_means(summary, "strategy_reward_mean"),
        "Strategy Reward Mean",
        "reward",
        fig_dir / "strategy_reward_mean.png",
    )
    save_bar(
        flatten_candidate_metric(summary, "counterfactual_avg_utility_at_budget"),
        "Counterfactual Avg Utility@Budget",
        "Avg U@B",
        fig_dir / "counterfactual_utility_at_budget.png",
    )
    save_bar(
        flatten_candidate_metric(summary, "counterfactual_avg_utility_at_budget_effective"),
        "Counterfactual Avg Utility@Budget (Effective Samples)",
        "Avg U@B",
        fig_dir / "counterfactual_utility_at_budget_effective.png",
    )
    save_bar(
        family_means(summary, "strategy_actual_insertions_mean"),
        "Actual Insertions Mean",
        "actual insertions",
        fig_dir / "actual_insertions_mean.png",
    )

    if candidates:
        utilities = [float(row["utility"]) for row in candidates]
        pos_colors = [float(row.get("rel_pos", 0.0)) for row in candidates]
        save_scatter(
            [float(row["sink_mass_z"]) for row in candidates],
            utilities,
            pos_colors,
            "SinkMassZ vs Candidate Utility",
            "SinkMassZ",
            "U(j)",
            fig_dir / "sink_z_vs_utility.png",
        )
        save_scatter(
            [float(row["entropy_z"]) for row in candidates],
            utilities,
            pos_colors,
            "EntropyZ vs Candidate Utility",
            "EntropyZ",
            "U(j)",
            fig_dir / "entropy_z_vs_utility.png",
        )
        save_hist(utilities, "Candidate Utility Distribution", "U(j)", fig_dir / "utility_hist.png")

    save_summary_report(summary, fig_dir, output_dir)
    print(f"Wrote figures and report to {fig_dir}")


if __name__ == "__main__":
    main()
