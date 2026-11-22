import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from trading.analytics import metrics
from trading.dashboard import config_from
from trading.data import features, load, synthetic
from trading.execution import Execution
from trading.research import Research
from trading.risk import Account, Limits, approve

class ResearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bars = synthetic()
        cls.research = Research(cls.bars, config_from("config.example.json"))
        cls.research.train()

    def test_feature_prefix_invariance(self):
        bars = self.bars["DEMO"]
        market = self.bars["SPY"].close
        pd.testing.assert_frame_equal(features(bars, market).iloc[:180],
                                      features(bars.iloc[:180], market.iloc[:180]))

    def test_missing_data(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.csv"
            path.write_text("symbol,date,open,high,low,close,volume\nSPY,2024-01-01,8,9,7,,4\n")
            with self.assertRaisesRegex(ValueError, "Invalid OHLCV"):
                load(path)

    def test_strategy_and_selector(self):
        r = self.research
        self.assertGreater(len(r.selector.validation_mae), 0)
        self.assertIsNotNone(r.selector.classify(r.frames["SPY"].iloc[[-1]]))
        self.assertEqual(len(r.signals["DEMO"].columns), 6)
        self.assertTrue((r.signals["DEMO"][["momentum", "trend", "reversion"]].sum() > 0).all())
        allocation, reasons = r.decision("DEMO", r.dates[-1], "ml")
        self.assertLessEqual(sum(allocation.values()), 1.000001)
        self.assertTrue(reasons)

    def test_deterministic_backtest_and_next_open(self):
        r = self.research
        first, fills = r.run("hold")
        second, _ = r.run("hold")
        pd.testing.assert_frame_equal(first, second)
        first_fill = fills.iloc[0]
        self.assertEqual(first_fill.date, r.dates[r.cutoff])
        expected = r.bars[first_fill.symbol].at[first_fill.date, "open"] * (1 + r.slippage)
        self.assertAlmostEqual(first_fill.price, expected)
        self.assertGreater(metrics(first, fills, 100000)["costs"], 0)
        self.assertGreaterEqual(first.cash.min(), 0)

    def test_config_round_trip(self):
        config = config_from("config.example.json")
        self.assertFalse(config["allow_paper_orders"])
        self.assertEqual(json.loads(json.dumps(config)), config)

class FakeBroker:
    def __init__(self):
        self.calls = 0
        self.order = None
        self.now = datetime.now(timezone.utc)

    def snapshot(self):
        return (SimpleNamespace(id="fake", cash=10000, equity=10000, buying_power=30000,
                                last_equity=10000), [], [],
                SimpleNamespace(timestamp=self.now, is_open=True))

    def quote(self, symbol):
        return 100, 100, self.now

    def latest_complete_session(self, timestamp):
        return "2025-01-02"

    def submit(self, symbol, qty, price, order_id):
        self.calls += 1
        raise TimeoutError("network interrupted")

    def lookup(self, order_id):
        if self.order is None:
            raise TimeoutError("lookup unavailable")
        return self.order

class RiskAndExecutionTests(unittest.TestCase):
    def test_hard_limits_and_sizing(self):
        account = Account(10000, 500, 0, 10000, 10000, 20000)
        quantity, _ = approve(Limits(), account, 0, 50, 100, .1)
        self.assertEqual(quantity, 3)
        self.assertEqual(approve(Limits(kill=True), account, 0, 10, 100, .01)[0], 0)
        self.assertEqual(approve(Limits(), account, 2, -10, 100, .01)[0], -2)
        self.assertEqual(approve(Limits(), account, 0, 5, 100, .01, pending=True)[0], 0)
        self.assertEqual(approve(Limits(), account, 0, 5, 100, .01, age_hours=40)[0], 0)

    def test_dry_run_records_one_intent_without_a_broker(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = Execution(str(Path(directory) / "orders.db"), Limits())
            for _ in range(2):
                self.assertIn("10 shares proposed", engine.dry_run("SPY", .1, 1000, .01, "2025-01-02"))
            journal = engine.journal()
            self.assertEqual(len(journal), 1)
            self.assertEqual(journal[0]["status"], "dry-run")

    def test_timeout_partial_fill_and_deduplication(self):
        fake = FakeBroker()
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "orders.db")
            engine = Execution(path, Limits(), fake, paper_enabled=True)
            with self.assertRaises(RuntimeError):
                engine.paper("SPY", .1, .01, "2025-01-02", armed=False)
            with self.assertRaises(TimeoutError):
                engine.paper("SPY", .1, .01, "2025-01-02", armed=True)
            engine = Execution(path, Limits(), fake, paper_enabled=True)
            with self.assertRaises(TimeoutError):
                engine.paper("SPY", .1, .01, "2025-01-02", armed=True)
            fake.order = SimpleNamespace(status="partially_filled", filled_qty=2, filled_avg_price=100)
            self.assertIn("Pending", engine.paper("SPY", .1, .01, "2025-01-02", armed=True))
            self.assertEqual(engine.reconcile()[0][7], 2)
            fake.order.status = "filled"
            fake.order.filled_qty = 10
            self.assertIn("already processed", engine.paper("SPY", .1, .01, "2025-01-02", armed=True))
            self.assertEqual(fake.calls, 1)

if __name__ == "__main__":
    unittest.main()
