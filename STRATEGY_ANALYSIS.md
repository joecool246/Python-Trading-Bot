# Individual strategy analysis

Four-stock Alpaca IEX universe AAPL, MSFT, JPM and XOM, with SPY as benchmark and pairs reference. Adjusted prices, next-open fills, prior-session liquidity limits, 1 bp fee and 5 bp slippage per side. January 2022 through December 2025 is a retrospective block. The zero-cost comparison reruns the same rules at zero fee and slippage, so trade quantities can differ as NAV changes.

| Strategy | Net annual | Zero-cost annual | Vol. | Sharpe | Sortino | Max DD | Win rate | Profit factor | Fills | Sale hold, sessions | Exposure | Cash | Turnover | Costs, USD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Momentum | -2.89% | -1.77% | 3.60% | -0.80 | -1.05 | -12.14% | 58.12% | 0.74 | 867 | 2.92 | 11.05% | 88.95% | 79.13x | 4,400 |
| Mean reversion | 3.31% | 3.83% | 3.71% | 0.90 | 2.00 | -3.17% | 67.95% | 2.28 | 377 | 2.49 | 4.03% | 95.97% | 33.45x | 2,171 |
| Trend | 1.56% | 2.29% | 4.03% | 0.40 | 0.56 | -7.36% | 78.85% | 1.19 | 881 | 7.32 | 18.61% | 81.39% | 47.24x | 2,859 |
| Breakout | -0.95% | -0.39% | 1.46% | -0.65 | -0.86 | -4.68% | 44.08% | 0.70 | 297 | 1.29 | 2.40% | 97.60% | 38.02x | 2,222 |
| Long-only pairs | 0.00% | 0.00% | 0.00% | — | — | 0.00% | — | — | 0 | — | 0.00% | 100.00% | 0.00x | 0 |
| Volatility adaptive | 1.33% | 2.12% | 4.42% | 0.32 | 0.43 | -7.94% | 75.24% | 1.13 | 1,796 | 8.67 | 24.89% | 75.11% | 52.64x | 3,209 |

Annual return, volatility and ratios use the portfolio series. Win rate and profit factor use realized sale P&L. Holding duration is share-weighted. Marked per-security contributions include open and closed positions plus fees and reconcile to the portfolio NAV. The JSON report also contains return by prior-close market regime and calendar year.

Momentum and breakout were negative in three of four net calendar years and lost money even without costs over the pooled period. Transaction costs compounded those losses. Pairs produced no fills, so the flat result says nothing about deployed signal quality. Reversion was positive in all four calendar years, with positive marked contributions from all four symbols in this sample. Its low exposure and retrospective universe limit interpretation. Weak historical results did not cause automatic strategy removal.

```powershell
.\.venv\Scripts\python.exe -m trading.strategy_study strategies --data audit-runs/v3/alpaca.csv --config config.research.example.json --start 2022-01-03 --end 2025-12-31 --output research-runs/strategies.json
```
