import tempfile
import unittest
from argparse import Namespace
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd

from trading.analytics import paired_edge
from trading.dashboard import ResearchCache, config_from, download
from trading.data import features, load, synthetic, symbols_from
from trading.execution import AlpacaPaper, Execution
from trading.research import Research
from trading.risk import Limits
from trading.selector import training_rows
from test_v2 import FakeBroker

class DownloadTests(unittest.TestCase):
    def test_sdk_paginated_download_to_research_csv(self):
        adapter = AlpacaPaper("fake-key", "fake-secret")
        calls = []
        def response(*, path, data):
            calls.append((path, dict(data)))
            bar = dict(t="2024-01-02T05:00:00Z", o=10, h=12, l=9, c=11, v=123, n=3, vw=10.5)
            if not data["page_token"]:
                return {"bars": {"AAPL": [bar]}, "next_page_token": "second"}
            return {"bars": {"SPY": [bar, dict(bar, t="2024-01-04T05:00:00Z")]}}
        with tempfile.TemporaryDirectory() as directory, patch.object(adapter.data, "get", side_effect=response):
            output = str(Path(directory) / "bars.csv")
            args = Namespace(output=output, start="2024-01-01", end="2024-01-04", feed="iex", adjustment="all")
            download({"symbols": [" aapl "]}, args, adapter)
            result = load(output)
            self.assertEqual(set(result), {"AAPL", "SPY"})
            self.assertEqual(len(result["SPY"]), 1)
            self.assertEqual(result["SPY"].close.iloc[0], 11)
            self.assertEqual(calls[1][1]["page_token"], "second")
            self.assertEqual(calls[0][1]["feed"], "iex")
            self.assertEqual(calls[0][1]["adjustment"], "all")
            with self.assertRaisesRegex(ValueError, "Missing:"):
                download({"symbols": ["MSFT"]}, args, adapter)
            self.assertEqual(set(load(output)), {"AAPL", "SPY"})

    def test_duplicate_and_missing_symbols_fail_explicitly(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            symbols_from({"symbols": [" spy ", "SPY"]})
        with self.assertRaisesRegex(ValueError, "Missing historical bars"):
            Research(synthetic(), {"symbols": ["MSFT"]})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bars.csv"
            path.write_text("symbol,date,open,high,low,close,volume\nSPY,2024-01-02T05:00Z,10,11,9,10,4\nSPY,2024-01-02,10,11,9,10,4\n")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load(path)

    def test_adjustment_rounding_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adjusted.csv"
            path.write_text("symbol,date,open,high,low,close,volume\nXOM,2018-02-27,53.92,54.51,52.89650000000001,52.8965,40\n")
            parsed = load(path)["XOM"].iloc[0]
            self.assertEqual(parsed.low, parsed.close)
            path.write_text("symbol,date,open,high,low,close,volume\nXOM,2018-02-27,53.92,54.51,52.90,52.8965,40\n")
            with self.assertRaisesRegex(ValueError, "Invalid OHLCV"):
                load(path)

class ExecutionAuditTests(unittest.TestCase):
    def test_sell_uses_bid_and_rejected_intent_is_terminal(self):
        broker = FakeBroker()
        account, _, orders, clock = broker.snapshot()
        position = SimpleNamespace(symbol="SPY", qty="5", market_value="500")
        broker.snapshot = Mock(return_value=(account, [position], orders, clock))
        broker.quote = Mock(return_value=(99.994, 100.01, broker.now))
        broker.submit = Mock(return_value=SimpleNamespace(status="rejected", filled_qty=0, filled_avg_price=None))
        broker.lookup = Mock(side_effect=AssertionError("Rejected intent must not be polled"))
        with tempfile.TemporaryDirectory() as directory:
            engine = Execution(str(Path(directory) / "orders.db"), Limits(), broker, paper_enabled=True)
            result = engine.paper("SPY", 0, .01, "2025-01-02", armed=True)
            self.assertIn("rejected", result)
            self.assertEqual(broker.submit.call_args.args[:3], ("SPY", -5, 99.99))
            self.assertIn("already processed", engine.paper("SPY", 0, .01, "2025-01-02", armed=True))
            self.assertEqual(broker.submit.call_count, 1)

    def test_available_shares_and_frequency(self):
        broker = FakeBroker()
        account, _, orders, clock = broker.snapshot()
        position = SimpleNamespace(symbol="SPY", qty="5", qty_available="4", market_value="500")
        broker.snapshot = Mock(return_value=(account, [position], orders, clock))
        broker.submit = Mock(return_value=SimpleNamespace(status="rejected", filled_qty=0, filled_avg_price=None))
        with tempfile.TemporaryDirectory() as directory:
            engine = Execution(str(Path(directory) / "orders.db"), Limits(max_orders=1), broker, paper_enabled=True)
            self.assertIn("available shares", engine.paper("SPY", 0, .01, "2025-01-02", armed=True))
            position.qty_available = "5"
            self.assertIn("rejected", engine.paper("SPY", 0, .01, "2025-01-02", armed=True))
            self.assertIn("frequency", engine.paper("QQQ", .1, .01, "2025-01-02", armed=True))
            self.assertEqual(broker.submit.call_count, 1)

    def test_quote_and_account_fail_closed(self):
        broker = FakeBroker()
        current = broker.snapshot()
        broker.submit = Mock(side_effect=AssertionError("No submission expected"))
        with tempfile.TemporaryDirectory() as directory:
            engine = Execution(str(Path(directory) / "orders.db"), Limits(), broker, paper_enabled=True)
            for bid, ask, timestamp in [(100, 101, broker.now - timedelta(minutes=2)),
                                        (101, 100, broker.now), (float("nan"), 100, broker.now),
                                        (99, 104, broker.now)]:
                broker.quote = Mock(return_value=(bid, ask, timestamp))
                self.assertIn("quote", engine.paper("SPY", .1, .01, "2025-01-02", armed=True))
            self.assertIn("previous completed session", engine.paper("SPY", .1, .01, "2025-01-01", armed=True))
            self.assertIn("Invalid allocation", engine.paper("SPY", float("nan"), .01, "2025-01-02", armed=True))
            account, positions, orders, clock = current
            updated = SimpleNamespace(**vars(account))
            updated.id = "another"
            broker.snapshot = Mock(side_effect=[current, (updated, positions, orders, clock)])
            self.assertIn("changed", engine.paper("SPY", .1, .01, "2025-01-02", armed=True))
            broker.submit.assert_not_called()

class QuantAuditTests(unittest.TestCase):
    def test_strategy_specific_targets_with_identical_signal_timing(self):
        bars = synthetic()["SPY"].iloc[:50]
        frame = features(bars, bars.close)
        date = bars.index[35]
        signals = pd.DataFrame({"trend": 0., "volatility": 0.}, index=bars.index)
        signals.loc[date] = [1., .5]
        rows = training_rows(bars, frame, signals, .001).set_index("strategy")
        self.assertAlmostEqual(rows.at["volatility", "target"], .5 * rows.at["trend", "target"])
        self.assertEqual(rows.at["trend", "label_end"], bars.index[37])
        gapped = bars.copy()
        gapped.loc[bars.index[36], "open"] = float("nan")
        self.assertTrue(training_rows(gapped, frame, signals, .001).empty)

    def test_strict_model_boundaries_and_holdout_invariance(self):
        bars = synthetic()
        config = {"symbols": ["SPY"], "strategies": ["trend", "momentum"]}
        evaluation = Research(bars, config)
        evaluation.train()
        boundary = evaluation.dates[evaluation.cutoff]
        self.assertLess(evaluation.selector.last_label, boundary)
        altered = {s: f.copy() for s, f in bars.items()}
        altered["SPY"].loc[boundary:, ["open", "high", "low", "close"]] *= 2
        challenger = Research(altered, config)
        challenger.train()
        date = evaluation.dates[evaluation.cutoff - 1]
        self.assertEqual(evaluation.decision("SPY", date, "ml"), challenger.decision("SPY", date, "ml"))
        evaluation.train(deployment=True)
        self.assertLess(evaluation.selector.last_label, evaluation.dates[-1])
        self.assertGreaterEqual(evaluation.selector.last_label, boundary)
        with self.assertRaisesRegex(ValueError, "holdout-trained"):
            evaluation.run("ml")

    def test_execution_marks_and_volume_limit(self):
        bars = synthetic()
        config = {"symbols": ["DEMO"], "strategies": ["momentum"], "max_mark_gap": 0,
                  "max_prior_volume_participation": .0001}
        r = Research(bars, config)
        buy_day = r.dates[r.cutoff - 1]
        r.signals["DEMO"].loc[buy_day, "momentum"] = 1
        r.bars["DEMO"].loc[buy_day, "volume"] = 10000
        r.signals["DEMO"].loc[r.dates[r.cutoff], "momentum"] = 0
        curve, fills = r.run("momentum")
        self.assertEqual(fills.iloc[0].qty, 1)
        self.assertEqual(fills.iloc[1].qty, -1)
        self.assertAlmostEqual(curve.cash.iloc[1], 100000 + fills.iloc[1].pnl)
        self.assertEqual(curve.regime.iloc[0], r.market.loc[buy_day])
        r.bars["DEMO"] = r.bars["DEMO"].drop(r.dates[r.cutoff + 1])
        with self.assertRaisesRegex(ValueError, "Stale held position"):
            r.run("momentum")

    def test_paired_uncertainty_and_cache_invalidation(self):
        returns = pd.Series([.001] * 80)
        same = paired_edge(returns, returns)
        self.assertEqual(same["interval_95"], [0., 0.])
        self.assertFalse(same["clears_hurdle"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bars.csv"
            path.write_text("1")
            args = Namespace(data=str(path), synthetic=False)
            values = [object() for _ in range(3)]
            with patch("trading.dashboard.make_research", side_effect=values) as build:
                cache = ResearchCache()
                self.assertIs(cache.get(args, {}, deployment=True), values[0])
                self.assertIs(cache.get(args, {}, deployment=True), values[0])
                self.assertIs(cache.get(args, {}, deployment=False), values[1])
                path.write_text("12")
                self.assertIs(cache.get(args, {}, deployment=True), values[2])
                self.assertEqual(build.call_count, 3)

if __name__ == "__main__":
    unittest.main()
