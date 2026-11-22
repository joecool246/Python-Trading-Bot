import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import pandas as pd

from trading.dashboard import assistant_context, paper_adapter
from trading.data import synthetic
from trading.execution import Execution
from trading.quant import diagnostics
from trading.research import Research
from trading.risk import Limits
from trading.selector import FEATURES, training_rows


class PortfolioAuditTests(unittest.TestCase):
    def test_explicit_test_start_supports_default_expanding_training(self):
        bars = synthetic()
        boundary = bars["SPY"].index[500]
        research = Research(bars, {"symbols": ["SPY"], "strategies": ["trend"],
                                   "test_start": boundary.date().isoformat()})
        self.assertEqual(research.training_dates()[-1], bars["SPY"].index[499])
        research.train()
        self.assertLess(research.selector.last_label, boundary)

    def test_targets_charge_both_legs_and_use_next_open(self):
        dates = pd.bdate_range("2024-01-01", periods=8, tz="UTC")
        bars = pd.DataFrame({"open": [77, 100, 110, 260, 260, 260, 260, 260]}, index=dates)
        frame = pd.DataFrame(0., index=dates, columns=FEATURES)
        signals = pd.DataFrame({"trend": [.5, 0, 0, 0, 0, 0, 0, 0]}, index=dates)
        rows = training_rows(bars, frame, signals, .01)
        self.assertEqual(rows.iloc[0].label_end, dates[2])
        self.assertAlmostEqual(rows.iloc[0].target, .5 * (110 * .99 / (100 * 1.01) - 1))

    def test_buy_and_hold_costs_reconcile_to_cash(self):
        bars = synthetic()
        config = {"symbols": ["SPY"], "fee_bps": 100, "slippage_bps": 200}
        research = Research(bars, config)
        curve, trades = research.run("hold")
        fill = trades.iloc[0]
        opening = bars["SPY"].open.loc[fill.date]
        self.assertAlmostEqual(fill.price, opening * 1.02)
        self.assertAlmostEqual(fill.cost, fill.qty * (fill.price - opening + fill.price * .01))
        self.assertAlmostEqual(curve.cash.iloc[0], research.initial - fill.qty * fill.price * 1.01)

    def test_development_error_excludes_symbols_held_out_from_fitting(self):
        dates = pd.bdate_range("2024-01-01", periods=3, tz="UTC")
        rows = pd.DataFrame(0., index=dates[:2].rename("date"), columns=[*FEATURES, "scale"])
        rows = rows.assign(symbol=["A", "B"], strategy="trend", target=[1., 100.], label_end=dates[-1])
        research = Mock()
        research.training_dates.return_value = dates
        research.samples.return_value = rows
        research.fit_symbols = ["A"]
        research.selector.model.predict.return_value = [0.]
        research.selector.model.importance = {}
        result = diagnostics(research, development=True)
        self.assertEqual(result["observations"], 1)
        self.assertEqual(result["mae"], 1.)
        self.assertEqual(result["bias"], -1.)

    def test_missing_or_live_credentials_never_construct_a_broker(self):
        for values, message in [({}, "Set ALPACA"), ({"ALPACA_API_KEY": "fake-live",
                  "ALPACA_API_SECRET": "dummy"}, "paper API key")]:
            with patch.dict(os.environ, values, clear=True), patch("trading.dashboard.AlpacaPaper") as adapter:
                with self.assertRaisesRegex(ValueError, message):
                    paper_adapter()
                adapter.assert_not_called()

    def test_assistant_context_excludes_private_session_fields(self):
        result = assistant_context({"mode": "paper", "portfolio": {"account": "dummy"},
                                    "journal": [{"id": "dummy"}], "source": "private/path",
                                    "stocks": {"SPY": {"execution": "order-id"}}, "results": {"ml": {}}})
        self.assertEqual(result, {"mode": "paper", "results": {"ml": {}}})

    def test_unarmed_and_invalid_dry_run_never_contact_broker(self):
        broker = Mock()
        with tempfile.TemporaryDirectory() as directory:
            engine = Execution(str(Path(directory) / "paper.sqlite3"), Limits(),
                               broker, paper_enabled=True)
            with self.assertRaisesRegex(RuntimeError, "arming"):
                engine.paper("SPY", .1, .01, "2025-01-02", armed=False)
            self.assertIn("invalid", engine.dry_run("SPY", .1, 0, .01, "2025-01-02"))
            self.assertEqual(engine.journal(), [])
            self.assertEqual(broker.mock_calls, [])

    def test_cli_missing_credentials_is_actionable(self):
        environment = {k: v for k, v in os.environ.items() if not k.startswith("ALPACA_API_")}
        response = subprocess.run([sys.executable, "bot.py", "verify-paper"], env=environment,
                                  capture_output=True, text=True, timeout=30)
        self.assertEqual(response.returncode, 2)
        self.assertIn("Set ALPACA_API_KEY", response.stderr)
        self.assertNotIn("Traceback", response.stderr)


if __name__ == "__main__":
    unittest.main()
