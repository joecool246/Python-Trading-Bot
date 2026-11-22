# Python Trading Bot (in the works currently updating to fit more strategies and efficient LLM APIs; inspired by QuantGuild)

A Tkinter-based paper-trading dashboard that uses the Alpaca API for market data and the OpenAI API for portfolio-analysis chat.

> **Educational use only.** This project does not provide financial advice. Review all trading behavior carefully and use Alpaca paper trading before considering any real funds.

## Features

- Tkinter desktop interface
- Alpaca paper-trading account integration
- Portfolio and open-order summaries
- Latest-trade price lookup
- OpenAI-powered portfolio chat analysis
- Local equity configuration stored in `equities.json`
- Environment-variable-based credential handling

## Requirements

- Python 3.13 or another supported Python 3 release
- Alpaca paper-trading API credentials
- An OpenAI API key

## Installation

```powershell
python -m pip install alpaca-trade-api openai
```

## Configuration

Set credentials in the PowerShell session before starting the application:

```powershell
$env:ALPACA_API_KEY="your-alpaca-key"
$env:ALPACA_API_SECRET="your-alpaca-secret"
$env:OPENAI_API_KEY="your-openai-key"
```

Environment variables apply only to the current terminal session.

## Run

```powershell
python bot.py
```

The application uses `https://paper-api.alpaca.markets/` and is intended for paper trading.

## Security

Never commit API keys or `.env` files. Revoke any credential that has been exposed or shared in source code, chat, screenshots, or logs.

## Files

- `bot.py` — Tkinter application and Alpaca/OpenAI integrations
- `alpaca.ipynb` — Alpaca API examples
- `openai.ipynb` — OpenAI portfolio-analysis example
- `equities.json` — local saved trading configuration (excluded from Git)
