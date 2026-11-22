import json
import sqlite3
import tempfile
import unittest
from argparse import Namespace
from dataclasses import asdict
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

from trading.dashboard import intelligence_view
from trading.data import synthetic
from trading.events import Event, Interpretation, Source, event_features
from trading.llm import Completion, Extractor, HttpProvider, ProviderFailure, Settings
from trading.llm_research import study, write_jsonl
from trading.selector import FEATURES, design, v2_design
from trading.sources import sec_sources


def source(**kwargs):
    defaults = dict(symbol="SPY", kind="disclosure", url="https://example.org/disclosure",
                    published_at="2026-01-02T15:00:00Z", observed_at="2026-01-02T15:01:00Z",
                    rights="Test excerpt permission", text="Revenue exceeded the stated consensus estimate.")
    return Source(**(defaults | kwargs))


DATA = dict(sentiment=1, direction=1, earnings_surprise=1, guidance_change=None,
            material_risk=False, event_type="earnings", uncertainty=.25,
            evidence="Revenue exceeded the stated consensus estimate.")


def event(published="2026-01-02T15:00:00Z", extracted="2026-01-02T19:00:00Z"):
    item = source(published_at=published, observed_at=published)
    return Event(item, Interpretation.parse(DATA, item), extracted, "openai", "test-model")


class EventTests(unittest.TestCase):
    def test_source_and_extraction_strict_validation(self):
        with self.assertRaises(ValueError):
            source(rights="")
        with self.assertRaises(ValueError):
            source(url="https://example.org/report?secret=oops")
        with self.assertRaises(ValueError):
            source(published_at="2026-01-02")
        with self.assertRaises(ValueError):
            Interpretation.parse(DATA | {"evidence": "A fabricated quote never seen in the source."}, source())
        with self.assertRaises(ValueError):
            Interpretation.parse(DATA | {"direction": True}, source())
        with self.assertRaises(ValueError):
            Interpretation.parse(DATA | {"uncertainty": float("nan")}, source())
        self.assertIsNone(Interpretation.parse(DATA, source()).guidance_change)

    def test_extraction_time_age_dedup_and_future_prefix_invariance(self):
        dates = pd.DatetimeIndex(pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06",
                                                  "2026-01-12"], utc=True))
        item = event()  # Friday at 14:00 New York, after the 13:00 cutoff.
        output = event_features(dates, [item, item])
        self.assertEqual(output.event_count.tolist(), [0, 1, 1, 0])
        self.assertEqual(output.event_sentiment.tolist(), [0, .75, .75, 0])
        future = event("2026-01-05T16:00:00Z", "2026-01-05T18:00:00Z")
        # Equality with the 13:00 ET cutoff is deliberately excluded.
        pd.testing.assert_frame_equal(output.iloc[:2], event_features(dates, [item, future]).iloc[:2])
        self.assertEqual(event_features(dates, [item, future]).event_count.iloc[2], 1.)
        with self.assertRaises(ValueError):
            Event(item.source, item.interpretation, "2025-12-31T19:00:00Z", "openai", "test")

    def test_sec_submission_uses_acceptance_and_actual_observation(self):
        session = Mock()
        listing = Mock(status_code=200, content=b"{}", **{"json.return_value": {"tickers": ["AAPL"],
            "filings": {"recent": {"form": ["8-K"], "accessionNumber": ["0000320193-25-000001"],
                                  "primaryDocument": ["report.htm"],
                                  "acceptanceDateTime": ["2025-01-03T18:00:00Z"]}}}})
        document = Mock(status_code=200, content=b"html", text="<p>Earnings report excerpt.</p><script>omit</script>")
        session.get.side_effect = [listing, document]
        with patch("trading.sources.time.sleep"):
            result = sec_sources("AAPL", "320193", "Research user contact@example.org", 1, session)
        self.assertEqual(result[0].published_at, "2025-01-03T18:00:00Z")
        self.assertGreater(result[0].observed_at, "2025-01-03")
        self.assertEqual(result[0].text, "Earnings report excerpt.")
        self.assertFalse(session.get.call_args.kwargs["allow_redirects"])

    def test_corrected_design_keeps_events_out_of_legacy_model(self):
        frame = pd.DataFrame({**{name: [1.] for name in FEATURES}, "scale": [.5],
                              "strategy": ["trend"], "event_count": [2.]})
        self.assertEqual(design(frame).event_count.iloc[0], 2.)
        self.assertNotIn("event_count", v2_design(design(frame)))

    def test_real_study_reports_unavailable_future_prices(self):
        report = study(synthetic(), {"test_start": "2026-11-23"}, [], synthetic_run=False)
        self.assertEqual(report["status"], "insufficient_future_market_data")
        self.assertNotIn("methods", report)

    def test_real_study_refuses_inspected_holdout(self):
        prices = synthetic()
        dates = pd.bdate_range("2024-01-01", periods=len(prices["SPY"]), tz="UTC")
        prices = {symbol: frame.set_axis(dates) for symbol, frame in prices.items()}
        with self.assertRaisesRegex(ValueError, "inspected"):
            study(prices, {"test_start": "2026-01-02"}, [event()], synthetic_run=False)


class LlmTests(unittest.TestCase):
    def test_success_cache_and_exact_metered_estimate(self):
        with tempfile.TemporaryDirectory() as folder:
            provider = Mock()
            provider.complete.return_value = Completion(json.dumps(DATA), 1000, 250)
            engine = Extractor(str(Path(folder) / "ledger.sqlite3"), Settings(), provider)
            first = engine.extract(source())
            second = engine.extract(source())
            self.assertEqual((first.status, second.status), ("extracted", "cached"))
            self.assertEqual(first.event, second.event)
            self.assertEqual(provider.complete.call_count, 1)
            self.assertAlmostEqual(engine.usage()["estimated_spend_usd"], .000225)

    def test_budget_before_request_and_outage_reservations(self):
        with tempfile.TemporaryDirectory() as folder:
            provider = Mock()
            provider.complete.side_effect = ProviderFailure("provider_unavailable", True)
            blocked = Extractor(str(Path(folder) / "limit.sqlite3"), Settings(budget_usd=0), provider)
            self.assertEqual(blocked.extract(source()).status, "budget_limit")
            provider.complete.assert_not_called()
            engine = Extractor(str(Path(folder) / "failed.sqlite3"), Settings(), provider)
            with patch("trading.llm.time.sleep"):
                self.assertEqual(engine.extract(source()).status, "provider_unavailable")
            self.assertEqual(engine.usage()["attempts"], 2)
            self.assertGreater(engine.usage()["unresolved_reservations_usd"], 0.)

    def test_malformed_replies_are_billed_but_not_cached(self):
        with tempfile.TemporaryDirectory() as folder:
            provider = Mock()
            provider.complete.return_value = Completion("{", 100, 20)
            engine = Extractor(str(Path(folder) / "bad.sqlite3"), Settings(requests_per_minute=1), provider)
            self.assertEqual(engine.extract(source()).status, "invalid_extraction")
            self.assertEqual(engine.extract(source()).status, "rate_limit")
            self.assertGreater(engine.usage()["metered_estimate_usd"], 0)

    def test_no_key_is_not_a_neutral_prediction(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict("os.environ", {}, clear=True):
            engine = Extractor(str(Path(folder) / "empty.sqlite3"), Settings())
            result = engine.extract(source())
            self.assertIsNone(result.event)
            self.assertEqual(result.status, "missing_credentials")
            self.assertEqual(engine.usage()["attempts"], 0)

    def test_both_http_provider_contracts(self):
        examples = {"openai": {"choices": [{"finish_reason": "stop", "message": {"content": "{}"}}],
                               "usage": {"prompt_tokens": 12, "completion_tokens": 4}},
                    "anthropic": {"stop_reason": "end_turn", "content": [{"type": "text", "text": "{}"}],
                                  "usage": {"input_tokens": 12, "output_tokens": 4}}}
        for name, payload in examples.items():
            with self.subTest(provider=name):
                session = Mock()
                session.post.return_value = Mock(status_code=200, **{"json.return_value": payload})
                provider = HttpProvider(name, "test-model", "dummy", session)
                self.assertEqual(provider.complete("instruction", "document", 64), Completion("{}", 12, 4))
                self.assertFalse(session.post.call_args.kwargs["allow_redirects"])

    def test_dashboard_reads_ledger_without_creating_it(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "absent.sqlite3"
            with self.assertRaises(sqlite3.OperationalError):
                intelligence_view(Namespace(llm_ledger=str(path)))
            self.assertFalse(path.exists())
            archive = Path(folder) / "events.jsonl"
            write_jsonl(str(archive), [event().to_dict()])
            with self.assertRaises(FileExistsError):
                write_jsonl(str(archive), [])


if __name__ == "__main__":
    unittest.main()
