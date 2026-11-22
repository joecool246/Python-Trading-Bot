"""LLM-feature evaluation. Never used by the broker or deployment selector."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
from requests.exceptions import RequestException

from .analytics import attribution, daily_returns, metrics, paired_edge
from .dashboard import config_from
from .data import load, synthetic
from .events import Event, Interpretation, Source, event_features, load_events, read_jsonl, unique
from .llm import Extractor, Settings
from .quant import fit_relative, model_used, RelativeEnsemble
from .research import Research
from .sources import sec_sources

# This date was the latest observation in the already-inspected V3 holdout.
INSPECTED_THROUGH = pd.Timestamp("2026-11-20", tz="UTC")


def write_jsonl(path: str, rows) -> None:
    with Path(path).open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")


def fixture(bars: dict) -> list[Event]:
    """Deterministic, non-predictive plumbing fixture independent of prices."""
    random = np.random.default_rng(24)
    events = []
    for symbol, frame in bars.items():
        for date in frame.index[30::6]:
            stamp = (date + pd.Timedelta(hours=12)).isoformat()
            evidence = "Synthetic disclosure used to exercise the research pipeline."
            source = Source(symbol, "synthetic", f"https://example.invalid/{symbol}/{date.date()}",
                            stamp, stamp, "Bundled synthetic fixture", evidence)
            item = Interpretation(int(random.integers(-1, 2)), int(random.integers(-1, 2)),
                                  None, None, False, "other", .5, evidence)
            events.append(Event(source, item, stamp, "fixture", "deterministic-v1"))
    return events


class FeatureResearch(Research):
    def __init__(self, bars: dict, config: dict, events: list[Event], max_age_days: int, *,
                 penalty: float = 0., cash_reserve: float = 0.):
        super().__init__(bars, config)
        self.penalty, self.cash_reserve = penalty, cash_reserve
        for symbol in self.universe:
            features = event_features(self.frames[symbol].index,
                                      [item for item in events if item.source.symbol == symbol], max_age_days)
            self.frames[symbol] = self.frames[symbol].join(features)

    def decision(self, symbol, date, method, previous=(), scores=None):
        if method != "ml":
            return super().decision(symbol, date, method, previous, scores)
        if scores is not None and self.penalty:
            scores = {name: score - (self.penalty if name not in previous else 0.)
                      for name, score in scores.items()}
        weights, reasons = super().decision(symbol, date, method, previous, scores)
        return {name: weight * (1 - self.cash_reserve) for name, weight in weights.items()}, reasons


def fit_corrected(research: Research):
    dates = research.training_dates()
    samples = research.samples(dates, fitting=True)
    samples = samples.loc[samples.label_end <= dates[-1]]
    if len(samples) < 30:
        raise ValueError("Insufficient development labels for the controlled model comparison")
    return research.selector.fit(samples, legacy=False)


def study(bars: dict, config: dict, events: list[Event], *, synthetic_run: bool,
          max_age_days: int = 7) -> dict:
    events = unique(events)
    if any((item.provider == "fixture") != synthetic_run for item in events):
        raise ValueError("Synthetic interpretations and real historical experiments must be separate")
    if not synthetic_run:
        if not config.get("test_start"):
            raise ValueError("Lock a future test_start in configuration before a real experiment")
        date = pd.to_datetime(config["test_start"], utc=True).normalize()
        if date <= INSPECTED_THROUGH:
            raise ValueError("The 2026 V3 period is inspected; use a later evaluation start")
        benchmark = config.get("benchmark", "SPY")
        if benchmark not in bars or len(bars[benchmark]) == 0:
            raise ValueError("Missing benchmark market data")
        if date > bars[benchmark].index[-1]:
            return {"status": "insufficient_future_market_data", "source": "historical",
                    "test_from": str(date.date()),
                    "market_data_through": str(bars[benchmark].index[-1].date())}
    baseline = Research(bars, config)
    eligible = [item for item in events if item.source.symbol in baseline.universe]
    candidate = FeatureResearch(bars, config, eligible, max_age_days)
    training_dates = baseline.training_dates()
    trace = {name: 0 for name in ("development", "holdout")}
    for symbol in candidate.universe:
        frame = candidate.frames[symbol]
        trace["development"] += int(frame.event_count.reindex(training_dates).fillna(0).gt(0).sum())
        trace["holdout"] += int(frame.event_count.reindex(baseline.dates[baseline.cutoff:]).fillna(0).gt(0).sum())
    if min(trace.values()) == 0 and not synthetic_run:
        return {"status": "insufficient_point_in_time_coverage", "event_stock_sessions": trace,
                "source": "historical", "test_from": str(baseline.dates[baseline.cutoff].date())}
    baseline.train()
    results, returns, curves = {}, {}, {}
    def save(name, curve, fills):
        results[name] = metrics(curve, fills, baseline.initial)
        if len(fills) and 'allocation' in fills:
            results[name]['strategy_attribution'] = attribution(fills)
        returns[name] = daily_returns(curve, baseline.initial)
        curves[name] = curve

    for method in ("equal", "rules", "hold", "ml"):
        save("v3_quant" if method == "ml" else method, *baseline.run(method))
    relative = fit_relative(baseline)
    if relative is not None:
        with model_used(baseline, RelativeEnsemble(baseline.selector.model, relative)):
            save("v3_relative", *baseline.run("ml"))
    v4_price = fit_corrected(baseline)
    with model_used(baseline, v4_price):
        save("matched_price_control", *baseline.run("ml"))
    candidate.fit_mode = "evaluation"
    event_model = fit_corrected(candidate)
    with model_used(candidate, event_model):
        save("llm_features", *candidate.run("ml"))
        # Fixed, predeclared sensitivity. Separate from the feature-only ablation.
        candidate.penalty, candidate.cash_reserve = candidate.fee + candidate.slippage, .10
        save("llm_cost_cash_sensitivity", *candidate.run("ml"))

    weight = (curves["llm_features"].exposure / curves["llm_features"].equity).shift(fill_value=0.)
    market = bars[config.get("benchmark", "SPY")].close.pct_change().reindex(weight.index)
    turnover = weight.diff().abs().fillna(0.)
    passive = weight * market - turnover * (baseline.fee + baseline.slippage)
    equity = (1 + passive).cumprod() * baseline.initial
    curve = pd.DataFrame({"equity": equity, "exposure": equity * weight,
                          "regime": curves["llm_features"].regime})
    notional = turnover * equity.shift(fill_value=baseline.initial)
    fills = pd.DataFrame({"qty": notional, "price": 1., "pnl": np.nan, "symbol": "SPY",
                          "cost": notional * (baseline.fee + baseline.slippage)})
    save("exposure_matched_passive", curve, fills.loc[notional > 0])
    edge = {name: paired_edge(returns["llm_features"], series, config.get("materiality_hurdle", .01))
            for name, series in returns.items() if name != "llm_features"}
    return {"status": "completed", "source": "SYNTHETIC PIPELINE FIXTURE" if synthetic_run else "historical",
            "event_stock_sessions": trace, "test_from": str(weight.index[0].date()),
            "test_to": str(weight.index[-1].date()), "methods": results, "paired_llm_edge": edge}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    filing = commands.add_parser("sec")
    filing.add_argument("--symbol", required=True)
    filing.add_argument("--cik", required=True)
    filing.add_argument("--contact", required=True)
    filing.add_argument("--limit", type=int, default=3)
    filing.add_argument("--output", required=True)
    extract = commands.add_parser("extract")
    extract.add_argument("--sources", required=True)
    extract.add_argument("--output", required=True)
    extract.add_argument("--ledger", default="llm-research.sqlite3")
    extract.add_argument("--settings")
    extract.add_argument("--allow-provider-processing", action="store_true")
    trial = commands.add_parser("study")
    source = trial.add_mutually_exclusive_group(required=True)
    source.add_argument("--data")
    source.add_argument("--synthetic", action="store_true")
    trial.add_argument("--events")
    trial.add_argument("--config", default="config.example.json")
    trial.add_argument("--start")
    trial.add_argument("--max-age-days", type=int, default=7)
    trial.add_argument("--output")
    args = parser.parse_args()
    try:
        if args.command == "sec":
            from dataclasses import asdict
            sources = sec_sources(args.symbol, args.cik, args.contact, args.limit)
            write_jsonl(args.output, (asdict(source) for source in sources))
            output = {"sources": len(sources)}
        elif args.command == "extract":
            if not args.allow_provider_processing:
                raise ValueError("Confirm permission to send the excerpts with --allow-provider-processing")
            if Path(args.output).exists():
                raise FileExistsError("Choose a new output file; event archives are immutable")
            sources = [Source(**row) for row in read_jsonl(args.sources)]
            settings = Settings(**config_from(args.settings)) if args.settings else Settings()
            extractor = Extractor(args.ledger, settings)
            results = [extractor.extract(source) for source in sources]
            events = unique([item.event for item in results if item.event is not None])
            write_jsonl(args.output, (item.to_dict() for item in events))
            output = {"statuses": dict(Counter(item.status for item in results)), "events": len(events),
                      **extractor.usage()}
        else:
            bars = synthetic() if args.synthetic else load(args.data)
            config = config_from(args.config)
            if args.start:
                config["test_start"] = args.start
            if not args.synthetic and not args.events:
                raise ValueError("Historical research requires a point-in-time event archive")
            events = load_events(args.events) if args.events else fixture(bars)
            output = study(bars, config, events, synthetic_run=args.synthetic, max_age_days=args.max_age_days)
            if args.data:
                output["bars_sha256"] = sha256(Path(args.data).read_bytes()).hexdigest()
                output["events_sha256"] = sha256(Path(args.events).read_bytes()).hexdigest()
        payload = json.dumps(output, allow_nan=False, indent=2)
        if args.command == "study" and args.output:
            Path(args.output).write_text(payload + "\n", encoding="utf-8")
        else:
            print(payload)
    except (ValueError, KeyError, TypeError, OSError, RequestException) as error:
        if isinstance(error, RequestException):
            parser.exit(1, "Source service request failed. No broker calls were made.\n")
        parser.exit(2, f"Research input unavailable or invalid: {error}\n")


if __name__ == "__main__":
    main()
