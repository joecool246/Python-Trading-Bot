# Prospective evaluation

This is a manual, **dry-run only** journal. The ML variant is the frozen V4 operating price selector, including the exact legacy feature mapping retained in the release. Equal weighting and regime rules run in independent paper-like simulated books. SPY buy-and-hold and lagged exposure-matched SPY are analytical comparators. No broker adapter, order network calls or automated scheduling is used. The broker response field remains empty by design. No future realized observations were generated.

Freeze the universe, configuration, feature and estimator version, training source and cutoff, label end, costs, benchmarks, metrics and primary hypothesis before observing future returns. The lock records the exact training CSV SHA-256, configuration digest, package versions, fitted-tree digest and a separate lock identifier. The deterministic reconstruction checks these before using the estimator. Revised or missing frozen input prefixes are rejected. Use a verified exchange calendar CSV with one ascending date per session and enough lookahead to cover the recording day. A missed session, a revised decision bar, insufficient data or a late decision should pause the run. Keep the lock and SQLite journal backed up. SQLite triggers block ordinary deletion and update, and records carry a verified hash chain. This is local tamper evidence rather than protection from an administrator rewriting the entire journal.

Record a decision only after the relevant daily bars are complete and before 09:30 New York on the next trading morning. The module requires the previous calendar session, a completed date after the freeze cutoff, all selected symbols' bars and valid ATR. It logs model scores, eligible strategies, selections, target allocations, risk decisions, estimated orders, cash, holdings and benchmark marks. After the subsequent exchange session closes, the observation records simulated next-open execution with risk approval recomputed at the opening price, estimated fees/slippage, and closing marks. The conservative completion rule makes the observation eligible the following day. The matched passive benchmark uses prior-close simulated ML exposure. Decisions and observations are separate rows. Use `--kill-switch` to stop a manual decision or observation command before it writes a record. The frozen risk configuration also retains its independent kill switch.

The preregistered horizon is 252 consecutive settled sessions. Report annualized return, maximum drawdown, Sharpe, exposure, turnover, costs and paired block-bootstrap ML-minus-equal return, with a 1-percentage-point materiality hurdle. Interim reports are descriptive. Future broker acceptance, partial fills and real execution quality remain unverified.

```powershell
$py = ".\.venv\Scripts\python.exe"
New-Item -ItemType Directory -Force evaluation-runs | Out-Null
$training = "audit-runs/v3/alpaca.csv"
$lock = "evaluation-runs/lock.json"
$journal = "evaluation-runs/journal.sqlite3"
& $py -m trading.prospective freeze --training $training --cutoff 2026-11-20 --config config.prospective.example.json --lock $lock
& $py -m trading.prospective decision --training $training --data evaluation-runs/current-bars.csv --calendar evaluation-runs/exchange-sessions.csv --lock $lock --journal $journal
& $py -m trading.prospective observe --training $training --data evaluation-runs/current-bars.csv --calendar evaluation-runs/exchange-sessions.csv --lock $lock --journal $journal
& $py -m trading.prospective verify --lock $lock --journal $journal
```

Build current-bars.csv from the unchanged frozen CSV prefix plus new completed adjusted bars for the configured symbols and SPY. The archived data only reaches the freeze cutoff, so it cannot produce a genuine decision by itself. Live or broker-connected evaluation is a separate authorization and implementation step.

## Local verification

The model was frozen through November 20, 2026. Its last training label ended November 19. The archived-input smoke run correctly refused a prospective decision because the most recent completed bar was the freeze cutoff. Journal verification returned zero records. A synthetic fixture separately verified the full decision/observation sequence, including duplicate rejection, the market-open boundary and update, delete and replacement protection. The research branch passed all 54 tests.
