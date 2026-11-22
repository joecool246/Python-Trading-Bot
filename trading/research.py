"""Daily close decisions and next-open fills with portfolio bookkeeping."""
import numpy as np
import pandas as pd
from .data import features
from .risk import Account, Limits, approve
from .selector import Selector, training_rows
from .strategies import REASONS, candidates, pair_spread, regime

class Research:
    def __init__(self, bars: dict[str, pd.DataFrame], config: dict) -> None:
        self.bars = bars
        self.config = config
        benchmark = config.get("benchmark", "SPY")
        market = bars[benchmark].close
        self.dates = market.index
        if len(self.dates) < 180:
            raise ValueError("At least 180 benchmark sessions are required")
        self.cutoff = int(len(self.dates) * .75)
        self.frames = {s: features(frame, market) for s, frame in bars.items()}
        self.regimes = {s: regime(frame) for s, frame in self.frames.items()}
        self.market = self.regimes[benchmark]
        self.signals = {}
        self.universe = [s for s in config.get("symbols", bars) if s in bars]
        if not self.universe or len(set(self.universe)) != len(self.universe):
            raise ValueError("Select a nonempty universe of distinct symbols with data")
        self.limits = Limits(**config.get("risk", {}))
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

    def train(self) -> None:
        cutoff_date = self.dates[self.cutoff]
        rows = pd.concat([training_rows(self.bars[s].loc[lambda d: d.index < cutoff_date],
                         self.frames[s], signals, self.fee + self.slippage)
                          for s, signals in self.signals.items()])
        benchmark = self.config.get("benchmark", "SPY")
        self.selector.train(rows, self.frames[benchmark], self.market, self.dates, self.cutoff)

    def decision(self, symbol: str, date: pd.Timestamp, method: str,
                 previous: tuple[str, ...] = ()) -> tuple[dict, dict]:
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
                                  previous, count=self.config.get("strategy_count", 2),
                                  threshold=self.config.get("min_edge", .001))

    def hold_benchmark(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        symbol = self.config.get("benchmark", "SPY")
        frame = self.bars[symbol].loc[self.dates[self.cutoff:]]
        initial = self.config.get("initial_cash", 100000)
        opening = frame.open.iloc[0]
        price = opening * (1 + self.slippage)
        quantity = int(initial / (price * (1 + self.fee)))
        cash = initial - quantity * price * (1 + self.fee)
        exposure = quantity * frame.close
        curve = pd.DataFrame(dict(equity=cash + exposure, cash=cash, exposure=exposure,
                                  regime=self.market.reindex(frame.index)))
        fills = pd.DataFrame([dict(date=frame.index[0], symbol=symbol, qty=quantity,
                                    price=price, cost=quantity * (price - opening + price * self.fee),
                                    pnl=None, allocation={"hold": 1.})])
        return curve, fills

    def run(self, method: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        if method == "hold":
            return self.hold_benchmark()
        cash = peak = self.config.get("initial_cash", 100000)
        held, mark, basis, tags, previous = {}, {}, {}, {}, {}
        queued, equity_rows, fills = {}, [], []
        for date in self.dates[self.cutoff - 1:]:
            if date == self.dates[self.cutoff - 1]:
                for symbol in self.universe:
                    if date in self.bars[symbol].index:
                        weights, _ = self.decision(symbol, date, method)
                        weight = sum(weights.values()) * min(self.limits.position,
                                                              self.limits.exposure / len(self.universe))
                        queued[symbol] = weight, self.frames[symbol].at[date, "atr"], weights
                continue
            day_start = equity_rows[-1]["equity"] if equity_rows else cash
            trade_count = 0
            for symbol in self.universe:
                if date in self.bars[symbol].index:
                    mark[symbol] = self.bars[symbol].at[date, "open"]
            for symbol, (weight, atr, allocations) in sorted(queued.items(), key=lambda item: item[1][0]):
                if date not in self.bars[symbol].index:
                    continue
                qty = held.get(symbol, 0)
                exposure = sum(mark[s] * n for s, n in held.items())
                equity = cash + exposure
                price = mark[symbol]
                target = int(weight * equity / price)
                execution = price * (1 + np.sign(target - qty) * self.slippage)
                unit_cash = execution * (1 + self.fee)
                sector = self.config.get("sectors", {}).get(symbol)
                sector_value = (sum(mark[s] * n for s, n in held.items()
                                    if self.config["sectors"].get(s) == sector)
                                if sector else None)
                delta, _ = approve(self.limits, Account(equity, cash, exposure, peak, day_start, cash),
                                   qty, target - qty, unit_cash, atr, orders=trade_count,
                                   sector_value=sector_value)
                if not delta:
                    continue
                fee = abs(delta) * execution * self.fee
                pnl = None if delta > 0 else (-delta) * (execution - basis[symbol]) - fee
                if delta > 0:
                    basis[symbol] = (basis.get(symbol, 0) * qty + delta * execution + fee) / (qty + delta)
                    tags[symbol] = allocations
                fills.append(dict(date=date, symbol=symbol, qty=delta, price=execution,
                                  cost=fee + abs(delta) * abs(execution - price), pnl=pnl,
                                  allocation=tags.get(symbol, {})))
                held[symbol] = qty + delta
                cash -= delta * execution + fee
                trade_count += 1
            queued = {}
            for symbol in self.universe:
                if date not in self.bars[symbol].index:
                    continue
                mark[symbol] = self.bars[symbol].at[date, "close"]
                allocations, _ = self.decision(symbol, date, method, previous.get(symbol, ()))
                previous[symbol] = tuple(allocations)
                atr = self.frames[symbol].at[date, "atr"]
                weight = sum(allocations.values()) * min(self.limits.position, self.limits.exposure / len(self.universe))
                queued[symbol] = weight, atr, allocations
            exposure = sum(mark[s] * n for s, n in held.items())
            equity = cash + exposure
            peak = max(peak, equity)
            equity_rows.append(dict(date=date, equity=equity, cash=cash, exposure=exposure,
                                    regime=self.market.loc[date]))
        return pd.DataFrame(equity_rows).set_index("date"), pd.DataFrame(fills)
