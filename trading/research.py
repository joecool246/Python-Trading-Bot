"""Daily close decisions and next-open fills with portfolio bookkeeping."""
import numpy as np
import pandas as pd
from .data import features, symbols_from
from .risk import Account, Limits, approve
from .selector import Selector, design, prepare, training_rows
from .strategies import REASONS, candidates, pair_spread, regime

class Research:
    def __init__(self, bars: dict[str, pd.DataFrame], config: dict) -> None:
        self.bars = bars
        self.config = config
        benchmark = config.get("benchmark", "SPY")
        if benchmark not in bars:
            raise ValueError(f"Missing historical bars for benchmark {benchmark}")
        market = bars[benchmark].close
        self.dates = market.index
        if len(self.dates) < 180:
            raise ValueError("At least 180 benchmark sessions are required")
        self.cutoff = int(len(self.dates) * .75)
        self.fit_mode = None
        if test_start := config.get("test_start"):
            self.cutoff = int(self.dates.searchsorted(pd.to_datetime(test_start, utc=True)))
        if not 120 <= self.cutoff < len(self.dates):
            raise ValueError("Test boundary must leave at least 120 development sessions and a holdout")
        self.max_participation = float(config.get("max_prior_volume_participation", .01))
        self.max_mark_gap = int(config.get("max_mark_gap", 5))
        if not 0 < self.max_participation <= 1 or self.max_mark_gap < 0:
            raise ValueError("Invalid liquidity or mark-gap limit")
        self.frames = {s: features(frame, market) for s, frame in bars.items()}
        self.regimes = {s: regime(frame) for s, frame in self.frames.items()}
        self.market = self.regimes[benchmark]
        self.signals = {}
        self.strategy_count = config.get("strategy_count", 2)
        self.min_edge = float(config.get("min_edge", .001))
        if type(self.strategy_count) is not int or self.strategy_count < 1 or not 0 <= self.min_edge < 1:
            raise ValueError("Invalid strategy count or minimum edge")
        self.universe = symbols_from(config, bars)
        self.fit_symbols = config.get("train_symbols") or self.universe
        if not set(self.fit_symbols) <= set(self.universe):
            raise ValueError("Training symbols must be a subset of the study universe")
        if missing := set(self.universe) - set(bars):
            raise ValueError(f"Missing historical bars for {sorted(missing)}")
        self.limits = Limits(**config.get("risk", {}))
        self.initial = float(config.get("initial_cash", 100000))
        if not np.isfinite(self.initial) or self.initial <= 0:
            raise ValueError("Initial cash must be positive and finite")
        self.fee = config.get("fee_bps", 1) / 10000
        self.slippage = config.get("slippage_bps", 5) / 10000
        if not 0 <= self.fee < .1 or not 0 <= self.slippage < .1:
            raise ValueError("Invalid trading friction")
        enabled = config.get("strategies", list(REASONS))
        if not enabled or len(set(enabled)) != len(enabled):
            raise ValueError('Select at least one distinct strategy')
        for symbol in self.universe:
            close = bars[symbol].close
            spread = pd.Series(np.nan, index=close.index)
            if symbol != benchmark and "pairs" in enabled:
                spread = pair_spread(close, market)
            self.signals[symbol] = candidates(self.frames[symbol], spread,
                                              config.get("parameters", {}))[enabled]
        self.selector = Selector()

    def samples(self, dates: pd.DatetimeIndex | None = None, *, fitting: bool = False) -> pd.DataFrame:
        dates = self.dates if dates is None else dates
        return pd.concat([training_rows(self.bars[s].reindex(dates), self.frames[s].reindex(dates),
                                        signals, self.fee + self.slippage).assign(symbol=s)
                          for s, signals in self.signals.items()
                          if not fitting or s in self.fit_symbols]).rename_axis("date")

    def training_dates(self, *, deployment: bool = False) -> pd.DatetimeIndex:
        # Deployment reserves its most recent completed bar for inference.
        cutoff = len(self.dates) - 1 if deployment else self.cutoff
        length = self.config.get("training_sessions", cutoff)
        if type(length) is not int or length < 120:
            raise ValueError("Training lookback must be at least 120 sessions")
        return self.dates[max(0, cutoff - length):cutoff]

    def train(self, *, deployment: bool = False) -> None:
        development = self.training_dates(deployment=deployment)
        rows = self.samples(development, fitting=True)
        benchmark = self.config.get("benchmark", "SPY")
        self.selector.train(rows, self.frames[benchmark], self.market, development, len(development),
                            validate=not deployment)
        self.fit_mode = "deployment" if deployment else "evaluation"

    def decision(self, symbol: str, date: pd.Timestamp, method: str,
                 previous: tuple[str, ...] = (),
                 scores: dict[str, float] | None = None) -> tuple[dict, dict]:
        signals = self.signals[symbol].loc[date]
        if method in REASONS:
            return ({method: float(signals.get(method, 0))} if signals.get(method, 0) else {}), {}
        label = self.regimes[symbol].loc[date]
        broad = self.market.loc[date]
        if label == "uncertain" or broad == "uncertain":
            return {}, {"regime": "insufficient market or security history"}
        if broad == "high volatility":
            label = broad
        return self.selector.rank(self.frames[symbol].loc[[date]], signals, label, method,
                                  previous, count=self.strategy_count, threshold=self.min_edge,
                                  scores=scores)

    def prediction_cache(self) -> dict:
        model = self.selector.model
        dates = self.dates[self.cutoff - 1:]
        cache = {}
        for symbol, signals in self.signals.items():
            frame = prepare(self.frames[symbol])
            columns = signals.loc[signals.index.intersection(dates)]
            rows = pd.concat([frame.loc[scale.index].assign(strategy=name, scale=scale)
                              .loc[scale > 0] for name, scale in columns.items()])
            if rows.empty:
                continue
            values = design(rows)
            if hasattr(model, "predict_groups"):
                predictions = model.predict_groups(values, rows.index)
            else:
                predictions = model.predict(values)
            for date, name, score in zip(rows.index, rows.strategy, predictions):
                cache.setdefault((symbol, date), {})[name] = float(score)
        return cache

    def hold_benchmark(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        symbol = self.config.get("benchmark", "SPY")
        frame = self.bars[symbol].loc[self.dates[self.cutoff:]]
        initial = self.initial
        opening = frame.open.iloc[0]
        price = opening * (1 + self.slippage)
        quantity = int(initial / (price * (1 + self.fee)))
        cash = initial - quantity * price * (1 + self.fee)
        exposure = quantity * frame.close
        curve = pd.DataFrame(dict(equity=cash + exposure, cash=cash, exposure=exposure,
                                  regime=self.market.shift().reindex(frame.index)))
        fills = pd.DataFrame([dict(date=frame.index[0], symbol=symbol, qty=quantity,
                                    price=price, cost=quantity * (price - opening + price * self.fee),
                                    pnl=None, allocation={"hold": 1.})])
        return curve, fills

    def run(self, method: str, *, trace: bool = False, rebalance_every: int = 1,
            max_daily_turnover: float | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
        if type(rebalance_every) is not int or rebalance_every < 1:
            raise ValueError("Rebalance interval must be a positive session count")
        if max_daily_turnover is not None and not 0 < max_daily_turnover <= 1:
            raise ValueError("Turnover cap must be a positive fraction at most one")
        if method == "hold":
            return self.hold_benchmark()
        if method == "ml" and self.fit_mode != "evaluation":
            raise ValueError("Historical evaluation requires a holdout-trained model")
        predictions = self.prediction_cache() if method == "ml" else {}
        cash = peak = self.initial
        held, mark, basis, tags, gaps, entry_session = {}, {}, {}, {}, {}, {}
        prior_regime = self.market.shift()
        prior_allocations = {}
        queued, equity_rows, fills = {}, [], []
        for session_number, date in enumerate(self.dates[self.cutoff - 1:]):
            if date == self.dates[self.cutoff - 1]:
                for symbol in self.universe:
                    if date in self.bars[symbol].index:
                        weights, _ = self.decision(symbol, date, method,
                                                   scores=predictions.get((symbol, date)))
                        weight = sum(weights.values()) * min(self.limits.position,
                                                              self.limits.exposure / len(self.universe))
                        queued[symbol] = weight, self.frames[symbol].at[date, "atr"], weights, self.bars[symbol].at[date, "volume"]
                continue
            day_start = equity_rows[-1]["equity"] if equity_rows else cash
            trade_count = 0
            prior_marks = mark.copy() if trace else {}
            prior_holdings = held.copy() if trace else {}
            flow = {}
            for symbol in self.universe:
                if date in self.bars[symbol].index:
                    mark[symbol] = self.bars[symbol].at[date, "open"]
                    gaps[symbol] = 0
                elif held.get(symbol, 0):
                    gaps[symbol] = gaps.get(symbol, 0) + 1
                    if gaps[symbol] > self.max_mark_gap:
                        raise ValueError(f"Stale held position {symbol} on {date.date()}; supply verified terminal data")
            opening_equity = cash + sum(mark[s] * n for s, n in held.items())
            turnover_budget = (opening_equity * max_daily_turnover
                               if max_daily_turnover is not None else None)
            targets = {s: int(weight * opening_equity / mark[s])
                       for s, (weight, _, _, _) in queued.items() if s in mark}
            for symbol in sorted(targets, key=lambda s: (targets[s] >= held.get(s, 0), s)):
                _, atr, allocations, previous_volume = queued[symbol]
                if date not in self.bars[symbol].index:
                    continue
                qty = held.get(symbol, 0)
                exposure = sum(mark[s] * n for s, n in held.items())
                equity = cash + exposure
                price = mark[symbol]
                capacity = int(previous_volume * self.max_participation)
                if turnover_budget is not None:
                    capacity = min(capacity, max(0, int(turnover_budget / (price * (1 + self.slippage)))))
                requested = int(np.clip(targets[symbol] - qty, -capacity, capacity))
                execution = price * (1 + np.sign(requested) * self.slippage)
                unit_cash = execution * (1 + self.fee)
                sector = self.config.get("sectors", {}).get(symbol)
                sector_value = (sum(mark[s] * n for s, n in held.items()
                                    if self.config["sectors"].get(s) == sector)
                                if sector else None)
                delta, _ = approve(self.limits, Account(equity, cash, exposure, peak, day_start, cash),
                                   qty, requested, unit_cash, atr, orders=trade_count,
                                   sector_value=sector_value)
                if not delta:
                    continue
                fee = abs(delta) * execution * self.fee
                pnl = None if delta > 0 else (-delta) * (execution - basis[symbol]) - fee
                if delta > 0:
                    entry_session[symbol] = (entry_session.get(symbol, 0) * qty
                                             + session_number * delta) / (qty + delta)
                    basis[symbol] = (basis.get(symbol, 0) * qty + delta * execution + fee) / (qty + delta)
                    mix = tags.get(symbol, {})
                    allocation_sum = sum(allocations.values())
                    tags[symbol] = {name: (mix.get(name, 0) * qty
                                          + delta * allocations.get(name, 0) / allocation_sum) / (qty + delta)
                                    for name in mix.keys() | allocations.keys()}
                fills.append(dict(date=date, symbol=symbol, qty=delta, price=execution,
                                  cost=fee + abs(delta) * abs(execution - price), pnl=pnl,
                                  allocation=tags.get(symbol, {}),
                                  holding_sessions=session_number - entry_session[symbol] if delta < 0 else None))
                held[symbol] = qty + delta
                if turnover_budget is not None:
                    turnover_budget -= abs(delta) * execution
                if trace:
                    flow[symbol] = (delta, execution, fee)
                cash -= delta * execution + fee
                trade_count += 1
            queued = {}
            switches = 0
            for symbol in self.universe:
                if date not in self.bars[symbol].index:
                    continue
                mark[symbol] = self.bars[symbol].at[date, "close"]
                if session_number % rebalance_every:
                    continue
                previous = tuple(name for name, weight in tags.get(symbol, {}).items() if weight > 0)
                if not held.get(symbol, 0):
                    previous = ()
                allocations, _ = self.decision(symbol, date, method, previous,
                                                scores=predictions.get((symbol, date)))
                if trace:
                    active = frozenset(allocations)
                    switches += active != prior_allocations.get(symbol, frozenset())
                    prior_allocations[symbol] = active
                atr = self.frames[symbol].at[date, "atr"]
                weight = sum(allocations.values()) * min(self.limits.position, self.limits.exposure / len(self.universe))
                queued[symbol] = weight, atr, allocations, self.bars[symbol].at[date, "volume"]
            exposure = sum(mark[s] * n for s, n in held.items())
            equity = cash + exposure
            peak = max(peak, equity)
            allocation = {}
            for symbol, shares in held.items():
                for name, fraction in tags.get(symbol, {}).items():
                    allocation[name] = allocation.get(name, 0.) + shares * mark[symbol] * fraction / equity
            row = dict(date=date, equity=equity, cash=cash, exposure=exposure,
                       regime=prior_regime.loc[date], allocation=allocation)
            if trace:
                row["positions"] = {s: shares * mark[s] / equity for s, shares in held.items() if shares}
                row["strategy_set_changes"] = switches
                contribution = {s: shares * (mark[s] - prior_marks[s])
                                for s, shares in prior_holdings.items() if shares}
                for symbol, (delta, execution, fee) in flow.items():
                    contribution[symbol] = (contribution.get(symbol, 0.)
                                             + delta * (mark[symbol] - execution) - fee)
                row["security_pnl"] = contribution
            equity_rows.append(row)
        return pd.DataFrame(equity_rows).set_index("date"), pd.DataFrame(fills)
