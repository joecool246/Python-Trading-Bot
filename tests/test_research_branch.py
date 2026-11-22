from contextlib import closing
from datetime import datetime, time, timezone
from pathlib import Path
import sqlite3
import tempfile
import unittest
from zoneinfo import ZoneInfo

import pandas as pd

from trading.data import load, synthetic
from trading.prospective import (append, connection, decide, digest, freeze, history, observe,
                                 restore, summarize, verify_lock)
from trading.research import Research
from trading.strategy_study import AllocationResearch, allocations, merge_universe, summary

NY = ZoneInfo("America/New_York")


def export_fixture(path, bars):
    pd.concat(bars, names=["symbol", "date"]).reset_index().to_csv(path, index=False)


class ResearchStudyTests(unittest.TestCase):
    def test_marked_contribution_reconciles_to_equity(self):
        r = Research(synthetic(), {"symbols": ["DEMO"], "strategies": ["trend"]})
        curve, fills = r.run("trend", trace=True)
        previous = r.initial
        for row in curve.itertuples():
            self.assertAlmostEqual(sum(row.security_pnl.values()), row.equity - previous)
            previous = row.equity
        measured = summary(curve, fills, r.initial)
        self.assertAlmostEqual(measured["cash_allocation"] + measured["average_exposure"], 1.)
        self.assertGreater(measured["strategy_set_changes"], 0)

    def test_exposure_matched_reports_costs_and_aligned_edge(self):
        config = {"symbols": ["SPY", "DEMO"], "strategies": ["trend", "momentum"]}
        result = allocations(synthetic(), config, policies=("baseline", "equal"))
        self.assertIn("baseline_versus_exposure_matched", result)
        matched = result["methods"]["exposure_matched"]
        self.assertGreater(matched["turnover"], 0)
        self.assertGreater(matched["costs"], 0)

    def test_turnover_cap_checks_daily_filled_notional(self):
        r = Research(synthetic(), {"symbols": ["DEMO"], "strategies": ["trend"]})
        _, fills = r.run("trend", max_daily_turnover=.002)
        self.assertTrue(len(fills))
        turnover = (fills.qty.abs() * fills.price).groupby(fills.date).sum()
        self.assertLess(turnover.max(), 205.)

    def test_allocation_scalers_do_not_expand_position_size(self):
        r = AllocationResearch(synthetic(), {"symbols": ["SPY", "DEMO"], "strategies": ["trend"]})
        date = r.dates[400]
        r.signals["DEMO"].at[date, "trend"] = 1.
        baseline, _ = r.decision("DEMO", date, "ml", scores={"trend": .01})
        self.assertEqual(baseline, {"trend": 1.})
        r.policy = "volatility_target"
        r.frames["DEMO"].at[date, "vol"] = .36
        reduced, _ = r.decision("DEMO", date, "ml", scores={"trend": .01})
        self.assertEqual(reduced, {"trend": .5})
        r.policy = "confidence"
        rejected, _ = r.decision("DEMO", date, "ml", scores={"trend": .002})
        self.assertEqual(rejected, {})

    def test_universe_refuses_unconfigured_or_overlapping_symbols(self):
        bars = synthetic()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            export_fixture(root / "left.csv", {"SPY": bars["SPY"]})
            export_fixture(root / "right.csv", {"DEMO": bars["DEMO"]})
            dates = bars["SPY"].index
            pd.DataFrame({"date": dates}).to_csv(root / "calendar.csv", index=False)
            args = dict(calendar=str(root / "calendar.csv"), start="2020-01-01", end="2021-01-01")
            with self.assertRaisesRegex(ValueError, "unexpected"):
                merge_universe(root / "left.csv", root / "right.csv", {"symbols": ["SPY"]},
                               root / "merged.csv", **args)
            report = merge_universe(root / "left.csv", root / "right.csv", {"symbols": ["DEMO"]},
                                    root / "merged.csv", **args)
            self.assertEqual(set(report), {"SPY", "DEMO"})
            with self.assertRaisesRegex(ValueError, "overlapping"):
                merge_universe(root / "left.csv", root / "left.csv", {"symbols": ["SPY"]},
                               root / "merged.csv", **args)


class ProspectiveTests(unittest.TestCase):
    def test_append_only_and_concurrency_and_duplicate(self):
        with tempfile.TemporaryDirectory() as temp:
            journal = str(Path(temp) / "journal.sqlite3")
            append(journal, "lock", "decision", "2025-01-02", {"a": 1}, "")
            _, head = history(journal, "lock")
            with self.assertRaisesRegex(ValueError, "advanced"):
                append(journal, "lock", "decision", "2025-01-03", {"a": 2}, "")
            with self.assertRaises(sqlite3.IntegrityError):
                append(journal, "lock", "decision", "2025-01-02", {"a": 2}, head)
            with closing(connection(journal)) as db, db:
                for query in ("UPDATE journal SET payload='{}'", "DELETE FROM journal",
                              "INSERT OR REPLACE INTO journal SELECT * FROM journal"):
                    with self.assertRaisesRegex(sqlite3.IntegrityError, "append-only"):
                        db.execute(query)
                db.execute("DROP TRIGGER no_update")
                db.execute("UPDATE journal SET payload='{}'")
            with self.assertRaisesRegex(ValueError, "hash chain"):
                history(journal, "lock")

    def test_lock_rebuild_and_dry_run_sequence(self):
        bars = synthetic()
        calendar = bars["SPY"].index
        cutoff, first, next_day, verify_day = calendar[400:404]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "immutable.csv"
            journal = str(Path(temp) / "journal.sqlite3")
            export_fixture(path, {s: bars[s].loc[:cutoff] for s in ("SPY", "DEMO")})
            frozen_bars = load(path)
            later = {s: pd.concat([frozen_bars[s], bars[s].loc[first:verify_day]])
                     for s in frozen_bars}
            config = {"symbols": ["SPY", "DEMO"], "strategies": ["trend", "momentum"]}
            lock = freeze(str(path), config, str(cutoff.date()))
            with self.assertRaisesRegex(ValueError, "modified"):
                verify_lock(dict(lock, cutoff="2020-01-01"))
            # The synthetic fixture rehearses the state machine without recording real historical decisions.
            lock["frozen_at"] = cutoff.isoformat()
            lock["lock_id"] = digest({k: v for k, v in lock.items() if k != "lock_id"})
            r = restore(lock, str(path), later)
            bad = {s: frame.copy() for s, frame in later.items()}
            bad["SPY"].iloc[0, 0] *= 2
            with self.assertRaisesRegex(ValueError, "prefix changed"):
                restore(lock, str(path), bad)
            late = datetime.combine(next_day.date(), time(10), tzinfo=NY)
            with self.assertRaisesRegex(ValueError, "before 09:30"):
                decide(lock, r, journal, calendar, current=late)
            morning = datetime.combine(next_day.date(), time(8), tzinfo=NY)
            decision = decide(lock, r, journal, calendar, current=morning)
            self.assertEqual(decision["input_session"], str(first.date()))
            self.assertEqual(decision["broker_responses"], [])
            with self.assertRaisesRegex(ValueError, "previous decision"):
                decide(lock, r, journal, calendar, current=morning)
            with self.assertRaisesRegex(ValueError, "Next completed"):
                observe(lock, r, journal, calendar, current=morning)
            next_morning = datetime.combine(verify_day.date(), time(8), tzinfo=NY)
            observation = observe(lock, r, journal, calendar, current=next_morning)
            self.assertEqual(observation["broker_responses"], [])
            records, _ = history(journal, lock["lock_id"])
            self.assertEqual([item["kind"] for item in records], ["decision", "observation"])
            self.assertEqual(summarize(records, 100000)["observations"], 1)


if __name__ == "__main__":
    unittest.main()
