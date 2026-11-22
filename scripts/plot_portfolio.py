"""Render the readme figures from two published experiment reports."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))["periods"]["1"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comparison", type=Path)
    parser.add_argument("study", type=Path)
    parser.add_argument("--output", type=Path, default=Path("docs"))
    args = parser.parse_args()
    comparison, study = read(args.comparison), read(args.study)
    if any(comparison[key] != study[key] for key in ("test_from", "test_to")):
        parser.error("Reports must cover the same evaluation dates.")
    args.output.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    methods = ["V2 ML", "V3 ML", "Equal weight", "Rules", "SPY hold", "Matched SPY"]
    metrics = [study["baseline"], study["relative_ensemble"], study["equal"],
               comparison["rules"], comparison["hold"], study["exposure_matched_passive"]]
    colors = ["#465991", "#7842ae", "#087178", "#cd7719", "#5a6472", "#8693a1"]
    fig, panels = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
    fig.suptitle("Trading Bot V3 | Historical comparison", fontsize=19, fontweight="bold")
    specs = [("annual_return", "Annualized return", True),
             ("average_exposure", "Mean gross exposure", True),
             ("max_drawdown", "Maximum drawdown", True),
             ("turnover", "Gross turnover (× mean equity)", False)]
    for axis, (key, title, percentage) in zip(panels.flat, specs):
        positions = list(range(len(methods)))
        values = [row[key] for row in metrics]
        axis.barh(positions, values, color=colors, height=.62)
        axis.invert_yaxis()
        axis.set_yticks(positions, methods)
        axis.set_title(title, loc="left", fontweight="bold")
        axis.axvline(0, color="#333333", linewidth=.8)
        low, high = min(min(values), 0), max(max(values), 0)
        gap = (high - low) * .03
        axis.set_xlim(low - gap * (7 if low < 0 else 1), high + gap * 6)
        if percentage:
            axis.xaxis.set_major_formatter(PercentFormatter(1))
        for index, value in enumerate(values):
            label = f"{value:.2%}" if percentage else f"{value:.2f}×"
            axis.text(value + (gap if value >= 0 else -gap), index, label,
                      va="center", ha="left" if value >= 0 else "right", fontsize=9)
    fig.supxlabel(f"Alpaca IEX adjusted daily bars | {study['test_from']} to {study['test_to']} | inspected evaluation")
    fig.savefig(args.output / "historical-comparison.png", dpi=160)
    plt.close(fig)

    fig, (axis, exposure) = plt.subplots(2, 1, figsize=(10, 7), layout="constrained")
    fig.suptitle("V3 ML | Uncertainty and strategy allocation", fontsize=18, fontweight="bold")
    baselines = ["Equal weight", "V2 ML", "Matched SPY"]
    for row, key in enumerate(("relative_vs_equal", "relative_vs_baseline", "relative_vs_passive")):
        result = study[key]
        point = 100 * result["annualized_active_return"]
        left, right = [100 * v for v in result["interval_95"]]
        axis.errorbar(point, row, xerr=[[point - left], [right - point]], fmt="o",
                      capsize=4, color="#7842ae")
    axis.axvline(0, color="#384452", linewidth=1)
    axis.axvline(1, color="#cd7719", linestyle="--", label="1 pp materiality threshold")
    axis.set(yticks=range(3), yticklabels=baselines,
             xlabel="V3 minus comparator · annualized arithmetic return (percentage points)")
    axis.set_title("Paired 95% block-bootstrap intervals", loc="left", fontweight="bold")
    axis.set_ylim(2.45, -.45)
    axis.legend(loc="upper right", frameon=True)
    groups = study["relative_ensemble"]["allocation_by_regime"]
    regime_names = list(groups)
    strategies = sorted(set().union(*(group.keys() for group in groups.values())))
    offset = [0.] * len(groups)
    for name, color in zip(strategies, colors):
        values = [groups[key].get(name, 0.) for key in regime_names]
        exposure.barh(regime_names, values, left=offset, color=color, label=name, height=.6)
        offset = [left + right for left, right in zip(offset, values)]
    exposure.set_title("Mean V3 allocation by prior-close regime", loc="left", fontweight="bold")
    exposure.set_xlabel("Share of portfolio equity")
    exposure.xaxis.set_major_formatter(PercentFormatter(1))
    exposure.legend(loc="upper center", bbox_to_anchor=(.5, -.17), frameon=False,
                    ncols=len(strategies))
    fig.savefig(args.output / "uncertainty-and-exposure.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
