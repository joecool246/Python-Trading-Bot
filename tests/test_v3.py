import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd

from trading.analytics import metrics, paired_sharpe
from trading.data import synthetic
from trading.execution import AlpacaPaper
from trading.quant import RelativeEnsemble, coverage, model_used
from trading.research import Research
from trading.selector import design, v2_design
from trading.risk import Account, Limits, approve

class QuantV3Tests(unittest.TestCase):
    def test_calendar_report_contains_only_internal_gaps(self):
        bars = synthetic()
        spy = bars["SPY"].iloc[1:8].drop(bars["SPY"].index[5])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "calendar.csv"
            pd.DataFrame({"date": bars["SPY"].index[:10]}).to_csv(path, index=False)
            result = coverage({"SPY": spy}, str(path))["SPY"]
        self.assertEqual(result["rows"], 6)
        self.assertEqual(result["missing_within_coverage"], [str(bars["SPY"].index[5].date())])
        self.assertEqual(result["off_calendar"], [])

    def test_relative_model_reorders_without_changing_group_average(self):
        base = Mock()
        relative = Mock()
        base.predict.return_value = np.array([.01, .02])
        relative.predict.return_value = np.array([.08, .01])
        ensemble = RelativeEnsemble(base, relative)
        prediction = ensemble.predict(pd.DataFrame({"a": [1, 2]}))
        self.assertAlmostEqual(prediction.mean(), .015)
        self.assertGreater(prediction[0], prediction[1])
        relative.reset_mock()
        base.predict.return_value = np.array([.01])
        self.assertAlmostEqual(ensemble.predict(pd.DataFrame({"a": [1]}))[0], .01)
        relative.predict.assert_not_called()

    def test_batch_scores_match_online_decision(self):
        r = Research(synthetic(), {"symbols": ["SPY", "DEMO"], "train_symbols": ["SPY"],
                                    "strategies": ["trend", "momentum"], "training_sessions": 250})
        r.train()
        self.assertEqual(len(r.training_dates()), 250)
        self.assertEqual(r.training_dates()[-1], r.dates[r.cutoff - 1])
        self.assertEqual(set(r.samples(fitting=True).symbol), {"SPY"})
        cache = r.prediction_cache()
        self.assertTrue(cache)
        for (symbol, date), scores in list(cache.items())[:10]:
            self.assertEqual(r.decision(symbol, date, "ml"), r.decision(symbol, date, "ml", scores=scores))

    def test_candidate_columns_preserve_causal_features(self):
        r = Research(synthetic(), {"symbols": ["SPY"], "strategies": ["trend", "momentum"]})
        rows = r.samples()
        matrix = design(rows)
        legacy = v2_design(matrix)
        for name in ("momentum", "trend", "breakout"):
            with self.subTest(feature=name):
                pd.testing.assert_series_equal(matrix[name], rows[name])
                self.assertTrue((legacy[name] == matrix[f"strategy_{name}"]).all())
        self.assertTrue(matrix.columns.is_unique)

    def test_model_restored_after_failure(self):
        original = object()
        research = SimpleNamespace(selector=SimpleNamespace(model=original))
        with self.assertRaises(RuntimeError), model_used(research, object()):
            raise RuntimeError("failure")
        self.assertIs(research.selector.model, original)

    def test_holdings_and_exposure_summary(self):
        curve = pd.DataFrame({"equity": [100., 110.], "exposure": [20., 55.],
                              "regime": ["bullish", "bullish"],
                              "allocation": [{"trend": .2}, {"trend": .4, "pairs": .1}]})
        fills = pd.DataFrame({"qty": [-1, -3], "pnl": [3, 4], "holding_sessions": [2, 6],
                              "cost": [1, 1], "symbol": ["A", "A"], "price": [120, 135]})
        summary = metrics(curve, fills, 100.)
        self.assertAlmostEqual(summary["average_exposure"], .35)
        self.assertAlmostEqual(summary["average_sale_holding_sessions"], 5.)
        self.assertAlmostEqual(summary["allocation_by_regime"]["bullish"]["trend"], .3)

    def test_cash_buying_power_and_loss_caps(self):
        limits = Limits()
        low_power = Account(10000, 5000, 0, 10000, 10000, 120)
        self.assertEqual(approve(limits, low_power, 0, 20, 100, .01)[0], 1)
        sector_cap = Account(10000, 5000, 3400, 10000, 10000, 5000)
        self.assertEqual(approve(limits, sector_cap, 0, 20, 100, .01, sector_value=3400)[0], 1)
        for peak, start in ((10000, 10300), (12000, 10000)):
            account = Account(10000, 5000, 0, peak, start, 5000)
            self.assertEqual(approve(limits, account, 0, 20, 100, .01)[0], 0)

    def test_paired_sharpe_identical_series(self):
        values = pd.Series(np.sin(np.arange(80)) / 100)
        result = paired_sharpe(values, values, draws=100)
        self.assertEqual(result, {"difference": 0., "interval_95": [0., 0.]})

    def test_metadata_excludes_sensitive_fields_and_order_calls(self):
        client = Mock()
        client.get_account.return_value = SimpleNamespace(status="ACTIVE", trading_blocked=False,
                                                          id="private", cash=123)
        client.get_all_positions.return_value = [object()]
        adapter = AlpacaPaper.__new__(AlpacaPaper)
        adapter.client = client
        self.assertEqual(adapter.read_metadata(), {"status": "ACTIVE", "trading_blocked": False,
                                                   "position_count": 1})
        client.submit_order.assert_not_called()

if __name__ == "__main__":
    unittest.main()
