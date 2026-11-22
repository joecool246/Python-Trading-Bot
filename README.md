# Trading Bot V2

A daily, long-only equities research and Alpaca paper-trading application. The default dashboard uses deterministic synthetic data and dry-run execution. Orders require both configuration and operator arming. The OpenAI assistant summarizes displayed research and has no execution tools.

## Setup

Run these commands in the repository. Python 3.12 was used for verification. Tkinter comes with a standard Windows Python installation.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item config.example.json config.local.json
```

Use PowerShell environment variables for credentials. The example env file lists their names. The application reads the process environment directly. Alpaca credentials are used for paper account reads and historical downloads. OpenAI credentials are optional.

```powershell
$env:ALPACA_API_KEY = "<paper-key>"
$env:ALPACA_API_SECRET = "<paper-secret>"
$env:OPENAI_API_KEY = "<optional-key>"
```

## Run

```powershell
python bot.py
python bot.py dashboard --synthetic --mode offline
python bot.py run --synthetic --mode dry-run
python bot.py backtest --synthetic
python -m unittest discover -s tests -v
```

Supply historical data as CSV with `symbol,date,open,high,low,close,volume` and an entry for the configured benchmark. Use one row per symbol per daily session, UTC-compatible dates and a consistent adjustment methodology. Use a point-in-time universe for meaningful evaluation. A missing current symbol bar skips decisions and fills. Historical positions can retain their last price across gaps, so data containing delistings requires external cleanup and a documented terminal-value convention.

```powershell
python bot.py download --config config.local.json --output bars.csv
python bot.py train --config config.local.json --data bars.csv
python bot.py backtest --config config.local.json --data bars.csv
python bot.py dashboard --config config.local.json --data bars.csv --mode dry-run
```

Set `symbols` in the local JSON configuration before downloading a multi-stock universe. Downloads use adjusted daily IEX bars. Research fitting is in memory and repeats when a CLI run begins. The final quarter is held out, and walk-forward cross-validation uses the first three quarters.

## Paper orders

Use a dedicated Alpaca paper account. Update `allow_paper_orders` to `true` in the local configuration only after independent review. Refresh the historical CSV before starting a session. Launch the dashboard and choose **Run one paper cycle** to arm that cycle, or use an explicitly armed CLI run.

```powershell
python bot.py dashboard --config config.local.json --data bars.csv --mode paper
python bot.py run --config config.local.json --data bars.csv --mode paper --arm-paper
```

Paper execution requires an open market, an exact previous-session signal date, an eligible equity, a fresh quote and an independent risk approval. It persists each order intent before submission, so restart or timeout cannot silently resubmit it. An unresolved intent pauses paper execution until reconciliation succeeds. Keep the same journal for a given account and use one dedicated bot account. A limit price bounds cash use on buys. Orders may stay open or fill partially. Cancellation is available in the broker adapter and is not automated. Review open orders in Alpaca before handling a stopped session. Dry-run proposals are logged separately and use a fixed illustrative cash balance, without maintaining an evolving simulated portfolio.

See [architecture and strategy notes](ARCHITECTURE.md) for model design, simulation assumptions and evaluation limits.
