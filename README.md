# Quantitative Equities Research & Paper-Trading Workbench

**Python · Time-series features · Gradient boosting · Walk-forward validation · Backtesting · Alpaca paper API**

A research-first, daily-frequency **long-only equity strategy** workbench. It combines signal construction, machine-learning-assisted strategy selection, portfolio simulation, risk controls, and an optional desktop dashboard.

> **Status: in development.** The default demo uses deterministic synthetic data and offline/dry-run execution. No historical performance, live profitability, or successful real-money deployment is claimed. Broker connectivity is **paper trading only** and requires explicit operator arming.

## Research questions

- Can historical price, volume, trend, volatility, and relative-strength features inform short-horizon strategy selection?
- How do learned strategy scores compare against simpler rule-based selection and an equal-weight baseline?
- What remains after chronological out-of-sample evaluation and modeled transaction friction?

## System components

| Component | Responsibility |
| --- | --- |
| Data | Validate daily OHLCV bars and compute backward-looking indicators |
| Strategies | Momentum, mean reversion, trend following, breakout, benchmark-relative pairs, volatility-adaptive signals |
| ML research | Gradient-boosted strategy return estimates and a supplementary classifier of a rule-defined regime |
| Selector | Rank available candidate strategies, normalize weights, permit abstention |
| Risk | Independent limits and checks before order proposals |
| Backtest & analytics | Next-open simulation, slippage/fees, positions/cash, benchmark and performance statistics |
| Execution | Alpaca **paper** adapter and SQLite intent journal |
| Interface | Tkinter dashboard and CLI; optional assistant summarizes research without execution privileges |

The *pairs* signal is **long-only**: it does not implement a hedged long/short statistical arbitrage portfolio. The regime classifier learns rule labels, so its agreement score is **not** independent evidence of market regime prediction.

## Quick start — offline demo

Python 3.12 was used for verification; Tkinter is supplied with standard Windows Python installations.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item config.example.json config.local.json
python bot.py dashboard --synthetic --mode offline
```

Other safe local commands:

```powershell
python bot.py run --synthetic --mode dry-run
python bot.py backtest --synthetic
python -m unittest discover -s tests -v
```

Synthetic runs test application behavior and **do not establish a trading edge**.

## Research with historical data

Provide a consistent-adjustment CSV with columns:

```text
symbol,date,open,high,low,close,volume
```

Include the configured benchmark and a point-in-time universe when evaluating historical claims.

```powershell
python bot.py train --config config.local.json --data bars.csv
python bot.py backtest --config config.local.json --data bars.csv
```

For broker-assisted historical download, configure your symbol universe and **paper** credentials locally, then use:

```powershell
python bot.py download --config config.local.json --output bars.csv
```

The final quarter of data is held out; earlier data is used for chronological walk-forward development. Training/validation label boundaries are respected by the research pipeline. Results depend on data quality and assumptions.

## Optional paper trading — read before use

The app defaults to non-executing operation. Paper orders require configuration permission **and explicit per-cycle arming**:

```powershell
python bot.py run --config config.local.json --data bars.csv --mode paper --arm-paper
```

Use a dedicated Alpaca **paper** account. Review configuration, recent data, open orders, and risk limits independently before arming. Paper order intents are journaled to reduce duplicate submissions following timeout/restart. Unresolved intents block execution until reconciled. Partial fills and cancellations require operator attention; cancellation is not automated.

Credentials belong in environment variables, never committed to Git:

```text
ALPACA_API_KEY
ALPACA_API_SECRET
OPENAI_API_KEY  # optional
```

## Evaluation caveats

A convincing quant-research result requires **real adjusted historical data**, point-in-time universe construction, delisting treatment, transaction-cost sensitivity, and genuinely untouched out-of-sample evaluation. The historical simulator does not model capacity, queue priority, partial fills, or delistings. End-of-backtest positions are marked, not forced closed.

Read [ARCHITECTURE.md](ARCHITECTURE.md) for strategy definitions, ML target construction, execution mechanics, and additional limitations.

## Portfolio demo checklist

For an external portfolio, add a reproducible **real-data research report** with exact date ranges, universe selection, benchmark, fees/slippage assumptions, train/test split, Sharpe and drawdown, failure cases, and limitations. Until then, describe this as an **engineering and research prototype**, not an alpha discovery.

**Not investment advice. Not for live trading.**
