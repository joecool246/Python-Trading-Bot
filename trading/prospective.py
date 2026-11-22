"""Frozen, append-only forward evaluation. This module has no broker adapter."""
import argparse
from contextlib import closing
from datetime import datetime, time, timezone
from zoneinfo import ZoneInfo
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

from .analytics import daily_returns, metrics, paired_edge
from .data import load
from .research import Research
from .risk import Account, approve

METHODS = ("ml", "equal", "rules")
MARKET_ZONE = ZoneInfo("America/New_York")
PACKAGES = ("numpy", "pandas", "scikit-learn", "statsmodels", "scipy")
REVISION = "v4.0.0-rc.1:price-operating-selector"
EVALUATION_SESSIONS = 252
METRICS = ["annualized_return", "max_drawdown", "sharpe", "turnover",
           "average_exposure", "paired_active_return"]
HYPOTHESIS = "ML exceeds equal weighting by at least one paired annualized percentage point."


def digest(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return sha256(payload.encode()).hexdigest()


def tree_id(research: Research) -> str:
    model = research.selector.model.estimator
    trees = []
    for item in model.estimators_.flat:
        tree = item.tree_
        trees.append({name: getattr(tree, name).tolist()
                      for name in ("children_left", "children_right", "feature", "threshold", "value")})
    return digest({"columns": model.feature_names_in_.tolist(), "parameters": model.get_params(),
                   "initial": model.init_.constant_.tolist(), "trees": trees})


def history_through(bars, cutoff):
    return {name: frame.loc[:cutoff] for name, frame in bars.items()}


def freeze(path: str, config: dict, cutoff: str) -> dict:
    if config.get("allow_paper_orders") is True:
        raise ValueError("Prospective configuration cannot arm paper orders")
    config = dict(config)
    config.pop("test_start", None)
    cutoff = pd.to_datetime(cutoff, utc=True).normalize()
    bars = history_through(load(path), cutoff)
    research = Research(bars, config)
    if research.dates[-1] != cutoff:
        raise ValueError("Freeze cutoff must be a benchmark session in the training archive")
    research.train(deployment=True)
    locked = {"version": REVISION, "configuration": config, "configuration_id": digest(config),
              "cutoff": str(cutoff.date()),
              "frozen_at": datetime.now(timezone.utc).isoformat(),
              "training_source_sha256": sha256(Path(path).read_bytes()).hexdigest(),
              "last_label_date": str(research.selector.last_label.date()), "model_id": tree_id(research),
              "package_versions": {name: version(name) for name in PACKAGES},
              "benchmarks": ["equal", "rules", "SPY buy-and-hold", "exposure-matched SPY"],
              "metrics": METRICS, "hypothesis": HYPOTHESIS, "horizon_sessions": EVALUATION_SESSIONS,
              "mode": "dry-run only"}
    return {"lock_id": digest(locked), **locked}


def verify_lock(lock: dict) -> None:
    fields = {key: value for key, value in lock.items() if key != "lock_id"}
    if lock.get("lock_id") != digest(fields) or lock.get("version") != REVISION:
        raise ValueError("Invalid or modified evaluation lock")
    if lock.get("configuration_id") != digest(lock["configuration"]):
        raise ValueError("Evaluation configuration hash differs")
    if lock["configuration"].get("allow_paper_orders") is True:
        raise ValueError("Evaluation is dry-run only")


def restore(lock, training_path, observations):
    verify_lock(lock)
    if sha256(Path(training_path).read_bytes()).hexdigest() != lock["training_source_sha256"]:
        raise ValueError("Frozen training archive hash mismatch")
    if lock["package_versions"] != {name: version(name) for name in PACKAGES}:
        raise ValueError("Frozen modeling environment differs")
    cutoff = pd.to_datetime(lock["cutoff"], utc=True)
    training = history_through(load(training_path), cutoff)
    configured = list(dict.fromkeys([*lock["configuration"].get("symbols", training),
                                     lock["configuration"].get("benchmark", "SPY")]))
    for symbol in configured:
        if symbol not in observations or not training[symbol].equals(observations[symbol].loc[:cutoff]):
            raise ValueError(f"Frozen market-data prefix changed or is missing for {symbol}")
    fitted = Research(training, lock["configuration"])
    fitted.train(deployment=True)
    if tree_id(fitted) != lock["model_id"]:
        raise ValueError("Reconstructed model identifier differs")
    decision = Research(observations, lock["configuration"])
    decision.selector = fitted.selector
    return decision


def connection(path: str):
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    db.executescript("""
        CREATE TABLE IF NOT EXISTS journal (
          seq INTEGER PRIMARY KEY, lock_id TEXT NOT NULL, kind TEXT NOT NULL, session TEXT NOT NULL,
          payload TEXT NOT NULL, previous TEXT NOT NULL, digest TEXT NOT NULL,
          UNIQUE(lock_id,kind,session)
        );
        CREATE TRIGGER IF NOT EXISTS no_update BEFORE UPDATE ON journal
          BEGIN SELECT RAISE(ABORT,'append-only journal'); END;
        CREATE TRIGGER IF NOT EXISTS no_delete BEFORE DELETE ON journal
          BEGIN SELECT RAISE(ABORT,'append-only journal'); END;
        CREATE TRIGGER IF NOT EXISTS no_replace BEFORE INSERT ON journal
          WHEN EXISTS (SELECT 1 FROM journal WHERE seq=NEW.seq OR
                       (lock_id=NEW.lock_id AND kind=NEW.kind AND session=NEW.session))
          BEGIN SELECT RAISE(ABORT,'append-only journal'); END;
    """)
    return db


def history(path: str, lock_id: str) -> tuple[list[dict], str]:
    previous = ""
    records = []
    with closing(connection(path)) as db:
        for row in db.execute("SELECT * FROM journal ORDER BY seq"):
            if row["lock_id"] != lock_id:
                raise ValueError("Journal belongs to another frozen evaluation")
            data = json.loads(row["payload"])
            expected = digest([previous, lock_id, row["kind"], row["session"], data])
            if row["previous"] != previous or row["digest"] != expected:
                raise ValueError("Journal hash chain does not match")
            records.append({"kind": row["kind"], "session": row["session"], "data": data})
            previous = expected
    return records, previous


def append(path, lock_id, kind, session, data, head):
    checksum = digest([head, lock_id, kind, session, data])
    with closing(connection(path)) as db, db:
        db.execute("BEGIN IMMEDIATE")
        last = db.execute("SELECT lock_id,digest FROM journal ORDER BY seq DESC LIMIT 1").fetchone()
        if (last and (last["lock_id"] != lock_id or last["digest"] != head)) or (not last and head):
            raise ValueError("Journal advanced while input was being computed")
        db.execute("INSERT INTO journal(lock_id,kind,session,payload,previous,digest) VALUES (?,?,?,?,?,?)",
                   (lock_id, kind, session, json.dumps(data, allow_nan=False), head, checksum))


def starting_book(cash: float) -> dict:
    return {"cash": cash, "positions": {}, "equity": cash, "peak": cash}


def proposal(research, book, marks, weights, attrs, volumes) -> list[dict]:
    equity = book["cash"] + sum(marks[s] * qty for s, qty in book["positions"].items())
    budget = min(research.limits.position, research.limits.exposure / len(research.universe))
    targets = {s: int(sum(w.values()) * budget * equity / marks[s]) for s, w in weights.items()}
    positions = book["positions"].copy()
    cash = book["cash"]
    result = []
    orders = 0
    sectors = research.config.get("sectors", {})
    for symbol in sorted(targets, key=lambda s: (targets[s] >= positions.get(s, 0), s)):
        old = positions.get(symbol, 0)
        requested = targets[symbol] - old
        max_volume = int(volumes[symbol] * research.max_participation)
        requested = max(-max_volume, min(requested, max_volume))
        if requested == 0:
            continue
        price = marks[symbol] * (1 + np.sign(requested) * research.slippage)
        exposure = sum(marks[s] * qty for s, qty in positions.items())
        sector = sectors.get(symbol)
        sector_value = (sum(marks[s] * qty for s, qty in positions.items() if sectors.get(s) == sector)
                        if sector else None)
        delta, reason = approve(research.limits, Account(cash + exposure, cash, exposure, book["peak"],
            book["equity"], cash), old, requested, price * (1 + research.fee), attrs[symbol],
            orders=orders, sector_value=sector_value)
        result.append({"symbol": symbol, "requested": requested, "approved": delta,
                       "price": float(price), "reason": str(reason)})
        if delta:
            orders += 1
            positions[symbol] = old + delta
            cash -= delta * price + abs(delta) * price * research.fee
    return result


def calendar_from(path: str) -> pd.DatetimeIndex:
    calendar = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(path)["date"], utc=True)).normalize()
    if len(calendar) == 0 or calendar.hasnans or not calendar.is_unique or not calendar.is_monotonic_increasing:
        raise ValueError("Calendar requires distinct, chronological session dates")
    return calendar


def completed_date(research, calendar: pd.DatetimeIndex, as_of: datetime) -> pd.Timestamp:
    local = as_of.astimezone(MARKET_ZONE)
    # Require a prior NY date to avoid treating an incomplete intraday daily bar as final.
    boundary = pd.Timestamp(local.date(), tz="UTC")
    available = calendar[calendar < boundary]
    if not len(available) or calendar[-1] < boundary:
        raise ValueError("Market calendar does not cover the decision clock")
    if available[-1] not in research.dates:
        raise ValueError("The most recently completed exchange session is missing")
    return available[-1]


def decide(lock, research, journal, calendar, *, current: datetime | None = None) -> dict:
    verify_lock(lock)
    current = current or datetime.now(timezone.utc)
    if current < datetime.fromisoformat(lock["frozen_at"]):
        raise ValueError("Cannot record a decision before the freeze")
    date = completed_date(research, calendar, current)
    local = current.astimezone(MARKET_ZONE)
    if local.weekday() < 5 and local.time() >= time(9, 30):
        raise ValueError("Record decisions before 09:30 New York on trading weekdays")
    if date <= pd.to_datetime(lock["cutoff"], utc=True):
        raise ValueError("No completed session after the frozen cutoff")
    if (pd.Timestamp(current) - date).days > 4:
        raise ValueError("Market data is stale for a prospective decision")
    records, head = history(journal, lock["lock_id"])
    if records and records[-1]["kind"] != "observation":
        raise ValueError("Settle the previous decision first")
    if sum(row["kind"] == "observation" for row in records) >= lock["horizon_sessions"]:
        raise ValueError("The preregistered evaluation horizon is complete")
    prior = records[-1]["data"] if records else {}
    if records and str(date.date()) != records[-1]["session"]:
        raise ValueError("A continuing evaluation must use the last observed session's close")
    last_decisions = [row["session"] for row in records if row["kind"] == "decision"]
    if last_decisions and str(date.date()) <= last_decisions[-1]:
        raise ValueError("A decision already exists for this session")
    benchmark = research.config.get("benchmark", "SPY")
    missing = [s for s in research.universe if date not in research.bars[s].index]
    if missing:
        raise ValueError(f"Incomplete decision snapshot: {', '.join(missing)}")
    marks = {s: float(research.bars[s].at[date, "close"]) for s in research.universe}
    atr = {s: float(research.frames[s].at[date, "atr"]) for s in research.universe}
    if not all(np.isfinite(value) and value >= 0 for value in atr.values()):
        raise ValueError("Decision snapshot lacks valid volatility data")
    volume = {s: float(research.bars[s].at[date, "volume"]) for s in research.universe}
    books = prior.get("books") or {method: starting_book(research.initial) for method in METHODS}
    selections, expected = {}, {}
    for method in METHODS:
        weights, symbols = {}, {}
        for symbol in research.universe:
            signals = research.signals[symbol].loc[date]
            active = signals[signals > 0]
            frame = research.frames[symbol].loc[[date]]
            scores = {name: float(research.selector.model.predict(
                design_for(frame, name, scale))[0]) for name, scale in active.items()} if method == "ml" else None
            allocation, explanation = research.decision(symbol, date, method, scores=scores)
            weights[symbol] = allocation
            symbols[symbol] = {"stock_regime": research.regimes[symbol].loc[date],
                               "eligible_signal_strength": active.to_dict(), "scores": scores,
                               "selected_weights": allocation, "explanations": explanation}
        selections[method] = symbols
        expected[method] = proposal(research, books[method], marks, weights, atr, volume)
    payload = {"recorded_at": current.isoformat(), "model_id": lock["model_id"],
               "input_session": str(date.date()), "model_data_cutoff": lock["cutoff"],
               "calendar_id": digest(calendar.strftime("%Y-%m-%d").tolist()),
               "market_regime": research.market.loc[date], "market_marks": marks,
               "atr": atr, "prior_volume": volume, "books": books, "selections": selections,
               "proposed_orders": expected, "benchmark_close": float(research.bars[benchmark].at[date, "close"]),
               "broker_responses": [], "execution": "dry-run only"}
    append(journal, lock["lock_id"], "decision", str(date.date()), payload, head)
    return payload


def design_for(frame, name, scale):
    from .selector import design, prepare
    return design(prepare(frame).assign(strategy=name, scale=scale))


def observe(lock, research, journal, calendar, *, current: datetime | None = None) -> dict:
    verify_lock(lock)
    current = current or datetime.now(timezone.utc)
    records, head = history(journal, lock["lock_id"])
    if not records or records[-1]["kind"] != "decision":
        raise ValueError("No unsettled decision")
    decision = records[-1]["data"]
    if current < datetime.fromisoformat(decision["recorded_at"]):
        raise ValueError("Observation cannot be recorded before its decision")
    decision_date = pd.to_datetime(records[-1]["session"], utc=True)
    eligible = calendar[calendar > decision_date]
    if (not len(eligible) or eligible[0] not in research.dates
            or eligible[0] > completed_date(research, calendar, current)):
        raise ValueError("Next completed market session is not available")
    date = eligible[0]
    market_open = datetime.combine(date.date(), time(9, 30), tzinfo=MARKET_ZONE)
    if datetime.fromisoformat(decision["recorded_at"]) >= market_open:
        raise ValueError("Cannot simulate a fill at an open before the logged decision")
    if any(date not in research.bars[s].index or decision_date not in research.bars[s].index
           for s in research.universe):
        raise ValueError("Incomplete observation snapshot")
    if any(research.bars[s].at[decision_date, "close"] != decision["market_marks"][s]
           or research.bars[s].at[decision_date, "volume"] != decision["prior_volume"][s]
           for s in research.universe):
        raise ValueError("The last archived decision inputs changed")
    benchmark = research.config.get("benchmark", "SPY")
    if research.bars[benchmark].at[decision_date, "close"] != decision["benchmark_close"]:
        raise ValueError("The last archived benchmark close changed")
    opening = {s: float(research.bars[s].at[date, "open"]) for s in research.universe}
    closing_prices = {s: float(research.bars[s].at[date, "close"]) for s in research.universe}
    books = {}
    fills = {}
    for method in METHODS:
        prior = decision["books"][method]
        weights = {s: item["selected_weights"] for s, item in decision["selections"][method].items()}
        fills[method] = proposal(research, prior, opening, weights, decision["atr"], decision["prior_volume"])
        book = {"cash": prior["cash"], "positions": prior["positions"].copy()}
        for fill in fills[method]:
            symbol, qty, price = fill["symbol"], fill["approved"], fill["price"]
            if not qty:
                continue
            book["positions"][symbol] = book["positions"].get(symbol, 0) + qty
            book["cash"] -= qty * price + abs(qty) * price * research.fee
        book["equity"] = book["cash"] + sum(closing_prices[s] * n for s, n in book["positions"].items())
        book["peak"] = max(prior["peak"], book["equity"])
        books[method] = book
    benchmark_open = float(research.bars[benchmark].at[date, "open"])
    benchmark_close = float(research.bars[benchmark].at[date, "close"])
    prior_observations = [row["data"] for row in records if row["kind"] == "observation"]
    prior = prior_observations[-1] if prior_observations else {}
    if prior:
        hold_qty, hold_cash = prior["buy_hold_shares"], prior["buy_hold_cash"]
    else:
        cost = benchmark_open * (1 + research.slippage) * (1 + research.fee)
        hold_qty = int(research.initial / cost)
        hold_cash = research.initial - hold_qty * cost
    ml_prior = decision["books"]["ml"]
    exposure = (ml_prior["equity"] - ml_prior["cash"]) / ml_prior["equity"]
    prior_weight = prior.get("matched_weight", 0.)
    matched_nav = prior.get("matched_equity", research.initial) * (
        1 + exposure * (benchmark_close / decision["benchmark_close"] - 1)
        - abs(exposure - prior_weight) * (research.fee + research.slippage))
    payload = {"recorded_at": current.isoformat(), "model_id": lock["model_id"],
               "decision_session": records[-1]["session"], "books": books, "simulated_fills": fills,
               "daily_returns": {m: books[m]["equity"] / decision["books"][m]["equity"] - 1 for m in METHODS},
               "benchmark_close": benchmark_close, "buy_hold_shares": hold_qty, "buy_hold_cash": hold_cash,
               "buy_hold_equity": hold_cash + hold_qty * benchmark_close,
               "matched_weight": exposure, "matched_equity": matched_nav,
               "fee_rate": research.fee, "slippage_rate": research.slippage, "broker_responses": []}
    append(journal, lock["lock_id"], "observation", str(date.date()), payload, head)
    return payload


def summarize(records: list[dict], initial: float) -> dict:
    observations = [row for row in records if row["kind"] == "observation"]
    if not observations:
        return {"observations": 0, "metrics": {}}
    labels = [row["session"] for row in observations]
    payloads = [row["data"] for row in observations]
    values = {method: [row["books"][method]["equity"] for row in payloads] for method in METHODS}
    values["buy_hold"] = [row["buy_hold_equity"] for row in payloads]
    values["matched"] = [row["matched_equity"] for row in payloads]
    exposure = {method: [item["books"][method]["equity"] - item["books"][method]["cash"]
                         for item in payloads] for method in METHODS}
    exposure["buy_hold"] = [item["buy_hold_shares"] * item["benchmark_close"] for item in payloads]
    exposure["matched"] = [item["matched_weight"] * item["matched_equity"] for item in payloads]
    curves = {name: pd.DataFrame({"equity": equity, "exposure": exposure[name], "regime": "all"},
                                 index=pd.to_datetime(labels)) for name, equity in values.items()}
    realized = {}
    for name, curve in curves.items():
        trades = [fill for row in payloads for fill in row["simulated_fills"].get(name, [])
                  if fill["approved"]]
        # Price is already the slippage-adjusted fill price. Recover the reference
        # opening price to report fees plus the model's estimated slippage.
        fee = float(payloads[-1]["fee_rate"])
        slip = float(payloads[-1]["slippage_rate"])
        fills = pd.DataFrame([{"qty": row["approved"], "price": row["price"],
                               "pnl": None, "symbol": row["symbol"], "cost": abs(row["approved"]) * (row["price"] * fee +
                                       abs(row["price"] - row["price"] /
                                           (1 + np.sign(row["approved"]) * slip)))}
                              for row in trades])
        realized[name] = {key: value for key, value in metrics(curve, fills, initial).items()
                          if key in {"annual_return", "max_drawdown", "sharpe", "turnover", "costs"}
                          and (trades or key not in {"turnover", "costs"})}
        if name in METHODS:
            realized[name]["average_exposure"] = float(np.mean([
                1 - row["books"][name]["cash"] / row["books"][name]["equity"] for row in payloads]))
    return {"observations": len(observations), "metrics": realized,
            "ml_vs_equal": paired_edge(daily_returns(curves["ml"], initial),
                                       daily_returns(curves["equal"], initial))}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "decision", "observe", "verify"))
    parser.add_argument("--training")
    parser.add_argument("--data")
    parser.add_argument("--config", default="config.prospective.example.json")
    parser.add_argument("--cutoff")
    parser.add_argument("--lock", required=True)
    parser.add_argument("--journal")
    parser.add_argument("--calendar")
    parser.add_argument("--kill-switch", action="store_true")
    args = parser.parse_args()
    try:
        if args.action == "freeze":
            if not args.training or not args.cutoff:
                parser.error("Freeze requires --training and --cutoff")
            result = freeze(args.training, json.loads(Path(args.config).read_text(encoding="utf-8-sig")), args.cutoff)
            with Path(args.lock).open("x", encoding="utf-8") as stream:
                json.dump(result, stream, indent=2)
        else:
            if not args.journal:
                parser.error("Specify --journal")
            lock = json.loads(Path(args.lock).read_text(encoding="utf-8-sig"))
            if args.action == "verify":
                verify_lock(lock)
                records, head = history(args.journal, lock["lock_id"])
                result = {"records": len(records), "head": head,
                          **summarize(records, lock["configuration"].get("initial_cash", 100000))}
            else:
                if not args.training or not args.data or not args.calendar:
                    parser.error("Specify --training, --data and --calendar")
                if args.kill_switch:
                    raise ValueError("Operator kill switch is set; no journal decision or observation written")
                research = restore(lock, args.training, load(args.data))
                result = (decide if args.action == "decision" else observe)(
                    lock, research, args.journal, calendar_from(args.calendar))
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, OSError, sqlite3.Error) as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    main()
