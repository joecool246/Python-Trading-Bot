"""Predeclared portfolio diagnostics and allocation comparisons."""
import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .analytics import daily_returns, metrics, paired_edge
from .dashboard import config_from
from .data import FIELDS, load, symbols_from, validate_rows
from .quant import coverage
from .research import Research
from .strategies import REASONS

POLICIES = ("baseline", "equal", "rules", "volatility_target", "risk_parity", "confidence",
            "cost_aware", "weekly", "turnover_cap", "smaller_position_cap", "smaller_exposure_cap", "smaller_trade_risk")


class AllocationResearch(Research):
    policy = "baseline"

    def __init__(self, bars, config):
        super().__init__(bars, config)
        volatility = pd.DataFrame({symbol: self.frames[symbol].vol for symbol in self.universe})
        self.inverse_vol_reference = 1 / (1 / volatility.where(volatility > 0)).mean(axis=1)

    def decision(self, symbol, date, method, previous=(), scores=None):
        if method != "ml" or self.policy == "baseline":
            return super().decision(symbol, date, method, previous, scores)
        if scores is not None and self.policy in {"confidence", "cost_aware"}:
            penalty = .002 if self.policy == "confidence" else self.fee + self.slippage
            scores = {name: score - (penalty if self.policy == "confidence" or name not in previous else 0)
                      for name, score in scores.items()}
        weights, explanation = super().decision(symbol, date, method, previous, scores)
        if weights and self.policy in {"volatility_target", "risk_parity"}:
            sigma = self.frames[symbol].at[date, "vol"]
            target = .18 if self.policy == "volatility_target" else self.inverse_vol_reference.loc[date]
            # Equal nominal sleeves produce less correlated risk only imperfectly.
            # Scale down high-volatility sleeves without borrowing or reallocating cash.
            multiplier = min(1., target / sigma) if np.isfinite(sigma) and sigma > 0 and np.isfinite(target) else 0.
            weights = {key: value * multiplier for key, value in weights.items()}
        return weights, explanation


def summary(curve: pd.DataFrame, fills: pd.DataFrame, initial: float) -> dict:
    returns = daily_returns(curve, initial)
    result = metrics(curve, fills, initial)
    result["cash_allocation"] = float((curve.cash / curve.equity).mean())
    result["all_cash_sessions"] = int(curve.exposure.eq(0).sum())
    result["mean_max_position"] = float(curve.positions.map(lambda x: max(x.values(), default=0.)).mean())
    result["mean_max_strategy_weight"] = float(curve.allocation.map(lambda x: max(x.values(), default=0.)).mean())
    result["strategy_set_changes"] = int(curve.strategy_set_changes.sum())
    result["exposure_by_regime"] = (curve.exposure / curve.equity).groupby(curve.regime).mean().to_dict()
    result["security_pnl"] = (pd.DataFrame.from_records(curve.security_pnl.tolist()).fillna(0).sum()).to_dict()
    result["calendar_return"] = {str(year): float((1 + group).prod() - 1)
                                 for year, group in returns.groupby(returns.index.year)}
    return result


def strategies(bars, config) -> dict:
    net = Research(bars, config)
    gross = Research(bars, dict(config, fee_bps=0, slippage_bps=0))
    results = {}
    for method in config.get("strategies", REASONS):
        curve, fills = net.run(method, trace=True)
        base = summary(curve, fills, net.initial)
        gross_curve, gross_fills = gross.run(method, trace=True)
        base["zero_cost"] = summary(gross_curve, gross_fills, gross.initial)
        results[method] = base
    return results


def allocations(bars, config, *, policies=POLICIES) -> dict:
    research = AllocationResearch(bars, config)
    research.train()
    results, series, curves = {}, {}, {}
    original_limits = research.limits
    reductions = {"smaller_position_cap": {"position": min(original_limits.position, .10)},
                  "smaller_exposure_cap": {"exposure": min(original_limits.exposure, .35)},
                  "smaller_trade_risk": {"trade_risk": original_limits.trade_risk / 2}}
    for policy in policies:
        research.policy = policy
        research.limits = replace(original_limits, **reductions.get(policy, {}))
        method = policy if policy in {"equal", "rules"} else "ml"
        curve, fills = research.run(method, trace=True, rebalance_every=5 if policy == "weekly" else 1,
                                    max_daily_turnover=.10 if policy == "turnover_cap" else None)
        results[policy] = summary(curve, fills, research.initial)
        series[policy] = daily_returns(curve, research.initial)
        curves[policy] = curve
    if "baseline" in curves:
        curve = curves["baseline"]
        weights = (curve.exposure / curve.equity).shift(fill_value=0)
        market = bars[config.get("benchmark", "SPY")].close.pct_change().reindex(weights.index)
        traded = weights.diff().abs().fillna(0)
        passive = weights * market - traded * (research.slippage + research.fee)
        result = (1 + passive).cumprod() * research.initial
        notional = traded * result.shift(fill_value=research.initial)
        simulated = pd.DataFrame({"qty": notional, "price": 1., "pnl": np.nan,
                                  "symbol": config.get("benchmark", "SPY"),
                                  "cost": notional * (research.slippage + research.fee)})
        results["exposure_matched"] = metrics(pd.DataFrame(
            {"equity": result, "exposure": result * weights, "regime": curve.regime}),
            simulated.loc[notional > 0], research.initial)
        series["exposure_matched"] = passive
    result = {"methods": results, "versus_equal": {
        name: paired_edge(values, series["equal"]) for name, values in series.items() if name != "equal"}}
    if "exposure_matched" in series:
        result["baseline_versus_exposure_matched"] = paired_edge(series["baseline"], series["exposure_matched"])
    return result


def merge_universe(base: str, additions: str, config: dict, output: str, *, calendar: str,
                   start: str, end: str) -> dict:
    left, right = load(base), load(additions)
    if set(left) & set(right):
        raise ValueError("Universe inputs contain overlapping symbols")
    merged = left | right
    wanted = set(symbols_from(config)) | {config.get("benchmark", "SPY")}
    if wanted != set(merged):
        raise ValueError(f"Missing {sorted(wanted - set(merged))}; unexpected {sorted(set(merged) - wanted)}")
    start_date, end_date = pd.to_datetime([start, end], utc=True)
    clipped = {symbol: frame.loc[start_date:end_date] for symbol, frame in merged.items()}
    if any(frame.empty for frame in clipped.values()):
        raise ValueError("Coverage window contains an empty configured symbol")
    rows = pd.concat(clipped, names=["symbol", "date"]).reset_index()
    validate_rows(rows[["symbol", "date", *FIELDS]])
    rows.to_csv(output, index=False, date_format="%Y-%m-%d")
    report = coverage(clipped, calendar)
    dates = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(calendar)["date"], utc=True))
    dates = dates[(dates >= start_date) & (dates <= end_date)]
    if dates.empty:
        raise ValueError("Exchange calendar does not cover the selected date range")
    for name, frame in clipped.items():
        report[name]["zero_volume_sessions"] = int(frame.volume.eq(0).sum())
        report[name]["median_daily_notional_proxy"] = float((frame.close * frame.volume).median())
        report[name]["leading_missing_sessions"] = int(sum(dates < frame.index[0]))
        report[name]["trailing_missing_sessions"] = int(sum(dates > frame.index[-1]))
        returns = frame.close.pct_change(fill_method=None)
        report[name]["large_close_moves_review"] = [str(date.date()) for date in returns.index[returns.abs() > .35]]
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["strategies", "allocation", "universe"])
    parser.add_argument("--data", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--additions")
    parser.add_argument("--calendar")
    parser.add_argument("--csv")
    parser.add_argument("--policies", nargs="+", choices=POLICIES)
    parser.add_argument("--train-symbols", nargs="+")
    args = parser.parse_args()
    config = config_from(args.config)
    if args.action == "universe":
        if not all((args.additions, args.calendar, args.csv)):
            parser.error("Universe audit requires --additions, --calendar and --csv")
        result = merge_universe(args.data, args.additions, config, args.csv,
                                calendar=args.calendar, start=args.start, end=args.end)
    else:
        if not args.start <= args.end <= "2025-12-31":
            parser.error("Development experiments use dated blocks ending by 2025-12-31")
        cutoff = pd.to_datetime(args.end, utc=True)
        bars = {key: value.loc[:cutoff] for key, value in load(args.data).items()}
        config["test_start"] = args.start
        if args.train_symbols:
            config["train_symbols"] = args.train_symbols
        if args.action == "allocation" and args.policies and "equal" not in args.policies:
            parser.error("Allocation comparison requires the equal-weight benchmark")
        result = (strategies(bars, config) if args.action == "strategies" else
                  allocations(bars, config, policies=args.policies or POLICIES))
    output = {"period": [args.start, args.end], "results": result}
    Path(args.output).write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
