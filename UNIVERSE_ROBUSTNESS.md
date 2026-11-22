# Universe robustness

The base portfolio uses nine historically available stocks and ETFs from the V3 research configuration. The expanded portfolio adds seven named sector ETFs selected for exposure coverage. XLU adds utilities, XLC communications, XLF financials, XLE energy, XLK technology, XLB materials and XLRE real estate. Existing XLV, XLI, XLY and XLP cover healthcare, industrials and consumer sectors. The resulting portfolio has 16 tradable symbols, with SPY as a non-traded benchmark and pairs reference.

The seven additions were downloaded by authenticated read-only Alpaca IEX with the same `all` adjustment setting as the original archive. The validation requires exact symbol coverage, rejects overlaps and malformed or duplicate OHLCV, and reports date range, calendar gaps, zero-volume sessions and median daily price-times-volume. From July 27, 2020 through December 31, 2025, 15 of 17 series had 1,366 sessions. JPM and XOM each had 1,363. No off-calendar or zero-volume bars appeared. The three known IEX gaps were retained. The data are adjusted provider histories, with no independent corporate-action, delisting-cash-flow or point-in-time membership verification. Their current survivorship remains a limitation. The volume measure is a screening proxy, not executable market depth.

Models fit before January 3, 2023 and were frozen through December 31, 2025. The period predates the inspected 2026 study and is retrospective. In the cross-sectional probe the fit excludes the seven added ETFs, while the evaluation portfolio contains all 16 symbols. This measures adding unseen instruments to a pooled portfolio. The JSON also reports marked contributions by security. It does not isolate a separate unseen-only portfolio.

| Fit and tradable universe | ML annual return | Equal annual return | Rules annual return | ML mean exposure |
| --- | ---: | ---: | ---: | ---: |
| Base nine, base nine | 1.90% | 7.10% | 3.36% | 2.28% |
| Expanded 16, expanded 16 | 1.33% | 6.24% | 2.48% | 1.15% |
| Base nine fit, expanded 16 | 1.64% | 6.24% | 2.48% | 2.19% |

Expanded baseline ML minus equal weighting was -4.85 annualized percentage points with paired interval -9.58 to +0.32 points. The held-out-symbol fit probe was -4.55 points with a -9.17 to +0.46 interval. Expanding affordable sector coverage did not resolve ML's low deployment of capital. The pooled outcomes do not prove which individual securities could generalize.

```powershell
$py = ".\.venv\Scripts\python.exe"
& $py bot.py download --config config.sector-additions.example.json --start 2020-07-27 --end 2026-01-01 --feed iex --adjustment all --output research-runs/sector-additions.csv
& $py -m trading.strategy_study universe --data audit-runs/v3/alpaca.csv --additions research-runs/sector-additions.csv --calendar audit-runs/v3/calendar.csv --config config.universe.example.json --start 2020-07-27 --end 2025-12-31 --csv research-runs/expanded.csv --output research-runs/universe.json
& $py -m trading.strategy_study allocation --data research-runs/expanded.csv --config config.universe.example.json --start 2023-01-03 --end 2025-12-31 --policies baseline equal rules --output research-runs/expanded-final.json
& $py -m trading.strategy_study allocation --data research-runs/expanded.csv --config config.universe.example.json --start 2023-01-03 --end 2025-12-31 --policies baseline equal rules --train-symbols AAPL MSFT JPM XOM XLV XLI XLY XLP QQQ --output research-runs/cross-final.json
```

The merged archived CSV SHA-256 is 8e826516403c1e0eafe946ded6974ebcb810e97a82726a232bc8f4e1791697ef. The validation found no leading or trailing exchange-session gaps in the stated common window and no close changes exceeding 35% in absolute value. This coarse check does not independently reconcile corporate actions.
