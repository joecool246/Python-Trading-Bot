"""Out-of-sample model diagnostics and blocked portfolio experiments."""
import argparse
from contextlib import contextmanager
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .analytics import daily_returns, metrics, paired_edge, paired_sharpe
from .dashboard import compare, config_from
from .data import load
from .research import Research
from .selector import design


def fit_relative(research: Research):
    """Fit differences from the signal group's mean on overlapping candidates."""
    dates = research.training_dates()
    rows = research.samples(dates, fitting=True)
    rows = rows[rows.label_end <= dates[-1]]
    groups = rows.groupby([rows.index, rows.symbol]).target
    relative = rows.assign(target=rows.target - groups.transform("mean"))
    relative = relative.loc[groups.transform("size") > 1]
    return research.selector.fit(relative, legacy=False) if len(relative) >= 30 else None


class RelativeEnsemble:
    """Combine absolute expected utility and a within-stock relative score."""

    def __init__(self, base, relative):
        self.base, self.relative = base, relative

    def predict(self, values):
        absolute = self.base.predict(values)
        relative = np.zeros(len(values))
        # rank() evaluates a same-symbol/date batch. A singleton has no comparator.
        if len(values) > 1:
            relative = self.relative.predict(values)
            relative -= relative.mean()
        return absolute + .5 * relative

    def predict_groups(self, values, groups):
        absolute = self.base.predict(values)
        relative = pd.Series(self.relative.predict(values))
        centered = relative - relative.groupby(groups).transform("mean")
        return absolute + .5 * centered.to_numpy()


@contextmanager
def model_used(research, model):
    original = research.selector.model
    research.selector.model = model
    try:
        yield
    finally:
        research.selector.model = original


def diagnostics(research: Research, *, development: bool = False) -> dict:
    dates = research.training_dates() if development else research.dates[research.cutoff:]
    # Preserve pre-block indicators and restrict both decisions and outcomes to this block.
    rows = research.samples().loc[lambda r: r.index.isin(dates) & (r.label_end <= dates[-1])].reset_index()
    if development:
        rows = rows.loc[rows.symbol.isin(research.fit_symbols)].copy()
    if rows.empty:
        return {"observations": 0}
    rows["prediction"] = research.selector.model.predict(design(rows))
    error = rows.prediction - rows.target
    pairs, correct = 0, 0.
    groups = rows.groupby(["date", "symbol"], sort=False)
    for group in groups:
        block = group[1]
        target, prediction = block.target.to_numpy(), block.prediction.to_numpy()
        upper = np.triu_indices(len(target), 1)
        realized = np.sign((target[:, None] - target)[upper])
        predicted = np.sign((prediction[:, None] - prediction)[upper])
        valid = realized != 0
        correct += np.sum(realized[valid] == predicted[valid]) + .5 * np.sum(predicted[valid] == 0)
        pairs += valid.sum()
    signal_group = groups[["target", "prediction"]].transform("mean")
    residual = rows[["target", "prediction"]] - signal_group
    def correlation(frame):
        value = frame.target.corr(frame.prediction, method="spearman")
        return float(value) if np.isfinite(value) else None
    bins = pd.qcut(rows.prediction, 5, duplicates="drop")
    return {"observations": len(rows), "bias": float(error.mean()), "mae": float(error.abs().mean()),
            "spearman": correlation(rows), "within_group_spearman": correlation(residual),
            "pair_count": int(pairs), "pair_accuracy": float(correct / pairs) if pairs else None,
            "quantiles": rows.groupby(bins, observed=True)[["target", "prediction"]].agg(
                ["mean", "count"]).to_numpy().tolist(),
            "per_symbol": {s: {"mae": float((g.prediction-g.target).abs().mean()),
                               "spearman": correlation(g)} for s, g in rows.groupby("symbol")},
            "importance": {name: float(value) for name, value in research.selector.model.importance.items()}}


def allocation_comparison(research: Research) -> dict:
    research.train()
    initial = research.initial
    base_curve, base_fills = research.run("ml")
    equal_curve, equal_fills = research.run("equal")
    relative = fit_relative(research)
    if relative is None:
        raise ValueError("Insufficient overlapping development signals for relative model")
    with model_used(research, RelativeEnsemble(research.selector.model, relative)):
        alt_curve, alt_fills = research.run("ml")
    benchmark = research.bars[research.config.get("benchmark", "SPY")].close.pct_change()
    weight = (alt_curve.exposure / alt_curve.equity).shift(fill_value=0)
    turnover = weight.diff().abs().fillna(0)
    passive = weight * benchmark.reindex(weight.index) - turnover * (research.fee + research.slippage)
    passive_equity = initial * (1 + passive).cumprod()
    notional = turnover * passive_equity.shift(fill_value=initial)
    synthetic_rebalances = pd.DataFrame(dict(qty=notional, price=1.,
        cost=notional * (research.fee + research.slippage), pnl=np.nan, symbol="SPY"))
    synthetic_rebalances = synthetic_rebalances.loc[notional > 0]
    passive_curve = pd.DataFrame(dict(equity=passive_equity, exposure=passive_equity * weight,
                                      regime=alt_curve.regime))
    alt = daily_returns(alt_curve, initial)
    return {"baseline": metrics(base_curve, base_fills, initial),
            "equal": metrics(equal_curve, equal_fills, initial),
            "relative_ensemble": metrics(alt_curve, alt_fills, initial),
            "exposure_matched_passive": metrics(passive_curve, synthetic_rebalances, initial),
            "relative_vs_baseline": paired_edge(alt, daily_returns(base_curve, initial)),
            "relative_vs_equal": paired_edge(alt, daily_returns(equal_curve, initial)),
            "relative_vs_passive": paired_edge(alt, passive),
            "relative_sharpe_vs_equal": paired_sharpe(alt, daily_returns(equal_curve, initial)),
            "validation_mae": research.selector.validation_mae,
            "last_training_outcome": str(research.selector.last_label.date()),
            "development_diagnostics": diagnostics(research, development=True),
            "holdout_diagnostics": diagnostics(research)}


def coverage(bars: dict, calendar_path: str) -> dict:
    values = pd.read_csv(calendar_path)["date"]
    expected = pd.DatetimeIndex(pd.to_datetime(values, utc=True)).drop_duplicates().sort_values()
    return {s: {"rows": len(frame), "first": str(frame.index[0].date()),
                "last": str(frame.index[-1].date()),
                "missing_within_coverage": [str(d.date()) for d in expected.difference(frame.index)
                                            if frame.index[0] <= d <= frame.index[-1]],
                "off_calendar": [str(d.date()) for d in frame.index.difference(expected)]}
            for s, frame in bars.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("coverage", "study", "compare"))
    parser.add_argument("--data", required=True)
    parser.add_argument("--calendar")
    parser.add_argument("--config", default="config.v3.example.json")
    parser.add_argument("--start")
    parser.add_argument("--end", help="Inclusive data end date")
    parser.add_argument("--cost-multipliers", default="1")
    parser.add_argument("--train-symbols", help="Restrict fitting to comma-separated tickers")
    parser.add_argument("--train-sessions", type=int, help="Rolling training lookback")
    parser.add_argument("--output")
    args = parser.parse_args()
    bars = load(args.data)
    config = config_from(args.config)
    if args.start:
        config["test_start"] = args.start
    if args.train_sessions is not None:
        config["training_sessions"] = args.train_sessions
    if args.train_symbols:
        config["train_symbols"] = [s.strip().upper() for s in args.train_symbols.split(",") if s.strip()]
    if args.end:
        end = pd.to_datetime(args.end, utc=True)
        bars = {s: frame.loc[:end] for s, frame in bars.items()}
    output = {"input_sha256": sha256(Path(args.data).read_bytes()).hexdigest()}
    if args.action == "coverage":
        if not args.calendar:
            parser.error("coverage requires --calendar")
        output["coverage"] = coverage(bars, args.calendar)
    else:
        output["universe"] = config.get("symbols")
        output["train_symbols"] = config.get("train_symbols", config.get("symbols"))
        output["periods"] = {}
        for text in args.cost_multipliers.split(","):
            multiplier = float(text)
            options = dict(config, fee_bps=config.get("fee_bps", 1)*multiplier,
                           slippage_bps=config.get("slippage_bps", 5)*multiplier)
            research = Research(bars, options)
            if args.action == "compare":
                research.train()
                result, _ = compare(research)
            else:
                result = allocation_comparison(research)
            output["periods"][text] = dict(result, test_from=str(research.dates[research.cutoff].date()),
                                            test_to=str(research.dates[-1].date()))
    payload = json.dumps(output, indent=2, allow_nan=False)
    if args.output:
        Path(args.output).write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)

if __name__ == "__main__":
    main()
