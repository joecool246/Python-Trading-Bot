# Quantitative research report

## Abstract

This study asks whether a daily machine-learning selector can allocate among six long-only strategies better than equal-weight selection or rule-based selection after estimated trading costs. V3 reproduced the original Yahoo Finance evaluation, repeated it on authenticated Alpaca IEX history, and tested an alternative model trained to distinguish strategies on dates when their signals overlap. The data and results are archived locally under `audit-runs/v3`.

## Research question and hypotheses

The primary hypothesis requires at least a 1-percentage-point annualized active-return advantage over equal weighting, with the lower bound of a paired 95% interval above that threshold. Secondary comparisons include rule selection, the unchanged V2 model, and a passive exposure-matched SPY benchmark. Diagnostics test whether the model ranks strategy outcomes within the same symbol and decision date. A model that mainly identifies favorable stocks or dates can have positive pooled rank correlation without ranking strategies effectively.

## Data verification and universe

The V2 snapshot uses Yahoo Finance adjusted close divided by raw close to scale OHLC, with source volume. Exact replay matched its reported annualized returns to rounding. An active paper account and position retrieval were verified using the connected Alpaca paper service. Authenticated historical reads used IEX daily bars and the same `all` adjustment setting on every symbol. Account identifiers, balances and credentials are excluded from the archived results. No orders were submitted or canceled.

Alpaca supplied 15,894 bars from July 27, 2020 through November 20, 2026. The exchange-calendar check found no off-calendar bars and three missing sessions for each of JPM and XOM. Direct per-date queries also returned no bars for those two symbols on April 19 and October 25, 2021, and March 8, 2022. Each year-sized request finished without a continuation token. The SDK pagination path has separate mock coverage. Earlier IEX history was insufficient to reproduce the Yahoo development window. The aligned Alpaca run therefore uses a shorter training history and is a cross-provider replication, not a byte-identical dataset. See `coverage.json` for per-symbol evidence and the dataset digest.

The four original stocks were retained for comparability. The expanded study adds liquid sector and broad-market proxies rather than selecting new symbols based on their backtest returns. XLV covers healthcare, XLI industrials, XLY consumer discretionary, XLP consumer staples and QQQ large-cap growth. AAPL and MSFT represent technology, JPM financials and XOM energy. SPY is the benchmark and pairs reference. The universe is retrospective and contains only surviving symbols. The cross-sectional probe fits only the four original stocks and evaluates predictions on the additional ETFs, while the primary temporal study trains on all nine tradable symbols and evaluates later dates.

## Methodology

The six strategy definitions are in the [README](README.md). Orders in historical simulation use the next open, prior-volume capacity, 1 basis point of fees plus 5 basis points of slippage per side, and long-only cash and exposure limits. Short-horizon labels use next-open entry, strategy-signal exit or a five-session cap, estimated friction, and a volatility penalty. Intermediate missing sessions invalidate those labels. Results include an alternate cost scenario at twice the base friction with the models refit using the changed costs.

The original gradient-boosted model and its feature mapping are preserved for replay. The audit found that three strategy one-hot names overwrote the raw momentum, trend and breakout feature columns. V3 gives the relative model separate feature names. It fits a group-centered outcome on overlapping active strategies, centers the predicted incremental score across the active candidates, then adds half of that adjustment to the baseline absolute estimate. Single active candidates retain their baseline estimate. Risk gating and a no-trade threshold still apply.

Three expanding walk-forward fits use calendar years 2023, 2024 and 2025 as separate evaluation blocks. The 2026 fit uses labels ending before January 2, 2026. No thresholds were optimized against that block. Its preliminary diagnostics helped reveal the feature mapping defect, so the corrected 2026 comparison is now an inspected temporal evaluation. A genuinely untouched confirmatory block remains future work. Return and Sharpe intervals use paired 10-session block resampling of whole portfolio days, preserving contemporaneous dependence across securities. Secondary comparisons are exploratory.

## V2 reproduction and strategy diagnostics

| Data source, 2022–2025 | V2 ML | Equal | Rules | SPY buy-and-hold |
| --- | ---: | ---: | ---: | ---: |
| Archived Yahoo Finance replay | 0.91% | 4.81% | 1.60% | 10.93% |
| Authenticated Alpaca IEX | 3.34% | 4.30% | 1.35% | 10.93% |

The Alpaca V2 ML result lagged equal weighting by 1.02 annualized arithmetic return points, with a 95% interval of −4.51 to +2.64 points. This replication does not establish a material advantage. The complete JSON includes regime breakdowns, per-symbol realized P&L, exposure and attribution.

| Alpaca strategy | Return | Volatility | Sharpe | Sortino | Drawdown | Win rate | Profit factor | Fills | Holding days | Exposure | Turnover | Cost USD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Momentum | -2.89% | 3.60% | -0.80 | -1.05 | -12.14% | 58.12% | 0.74 | 867 | 2.9 | 11.05% | 79.13x | 4,400 |
| Reversion | 3.31% | 3.71% | 0.90 | 2.00 | -3.17% | 67.95% | 2.28 | 377 | 2.5 | 4.03% | 33.45x | 2,171 |
| Trend | 1.56% | 4.03% | 0.40 | 0.56 | -7.36% | 78.85% | 1.19 | 881 | 7.3 | 18.61% | 47.24x | 2,859 |
| Breakout | -0.95% | 1.46% | -0.65 | -0.86 | -4.68% | 44.08% | 0.70 | 297 | 1.3 | 2.40% | 38.02x | 2,222 |
| Pairs | 0.00% | 0.00% | — | — | 0.00% | — | — | 0 | — | 0.00% | 0.00x | 0 |
| Volatility adaptive | 1.33% | 4.42% | 0.32 | 0.43 | -7.94% | 75.24% | 1.13 | 1,796 | 8.7 | 24.89% | 52.64x | 3,209 |

Holding duration is share-weighted on sales. Profit factor and win rate use realized sale P&L. Positions remain marked at the evaluation end. Momentum and breakout had negative net returns, and the pairs filter produced no fills in the four-stock Alpaca study. That does not establish that any strategy lacks predictive value in all universes. Reversion had the best risk-adjusted result in this particular four-stock portfolio comparison. Strategy pruning needs its own locked evaluation.

## Prediction diagnostics

The V2 2026 prediction study used 2,563 completed labels. Pooled Spearman was 0.020, within-symbol/date residual Spearman was -0.005, and pair-order accuracy was 50.0% across 1,394 unequal realized pairs, with predicted ties scored as half correct. The prediction MAE was 1.87 percentage points versus 1.54 points in the development fit and 1.46–1.92 points across the pre-holdout validation folds. These errors are measured in short-horizon risk-adjusted target units. The highest prediction bucket still averaged a negative realized utility. The reported quantile arrays contain realized mean, realized count, predicted mean and predicted count in that order.

The most important baseline inputs were price deviation, relative strength and broad-market momentum. Strategy indicators and entry scale received zero fitted importance in this baseline. Coupled with chance same-symbol/date rankings, this is evidence against strategy-specific discrimination in the original model. It does not prove that all of its signal was stock direction. Low selected exposure explains some of the gap to the fully invested benchmark. The exposure-matched comparator isolates the effect only approximately because it uses realized prior-close strategy exposure on SPY. Frictions, switching and risk caps interact. Their individual causal effects were not estimated.

No probability of profit is emitted, so probability calibration was not evaluated. The report archives held-out quantile summaries and per-symbol error and ranking diagnostics. A separate higher-capacity tree library, market-regime feature ablation and covariance-optimized allocation were deferred. Current evidence does not justify increasing model complexity or replacing the simpler benchmarks.

## Model comparison and uncertainty

Annualized geometric returns from the nine-symbol study, except the row that restricts model training to four stocks. Benchmarks and candidates use the same study universe within each row.

| Evaluation | V2 ML | V3 relative ensemble | Equal weight | Exposure-matched passive |
| --- | ---: | ---: | ---: | ---: |
| 2023 walk-forward | 0.92% | 0.95% | 10.27% | 1.64% |
| 2024 walk-forward | 1.27% | 1.28% | 8.25% | 0.03% |
| 2025 walk-forward | 3.55% | 3.36% | 3.13% | 0.43% |
| Jan–Nov 2026 | 1.82% | 1.83% | 1.56% | 1.47% |
| Jan–Nov 2026, twice costs | 0.70% | 0.70% | -0.25% | 0.61% |
| Jan–Nov 2026, four training stocks | 1.85% | 2.44% | 1.56% | 1.22% |

On the main 2026 block, rule selection returned 0.92% and SPY buy-and-hold returned 1.66%. The ensemble had 1.86% mean gross exposure versus 39.24% for equal weight. It had a 1.61 Sharpe, a 2.47 Sortino, a -1.02% maximum drawdown, 5.13x turnover and an estimated $308 of trading costs. The passive comparator matches lagged exposure and pays estimated rebalancing costs.

The ensemble's primary 2026 comparison yielded +0.15 annualized arithmetic return points versus equal weighting, with a 95% interval of -7.99 to +8.02 points. Its Sharpe difference was +1.26, with a 95% interval of -1.51 to +4.34. The ensemble's annualized arithmetic return difference versus V2 was +0.003 points. Against equal weight, the 2023 and 2024 gaps were -8.96 and -6.78 annualized arithmetic points. At twice the friction the 2026 return difference was +0.84 points, with a -7.29 to +8.69 interval. None establishes a material ensemble advantage.

The four-stock training probe is exploratory. On the five held-out ETFs, baseline Spearman ranged from -0.294 on QQQ to +0.285 on XLI. That spread and the wide portfolio interval argue against a generalization claim. More symbols and untouched calendar blocks are needed. Market regimes and per-symbol results for each run are in the local JSON reports.

## Conclusion

Keep the relative ensemble in research and retain equal weighting as the main challenger. The project is ready to present as an engineering and research case study, including for software, ML engineering, quantitative development and research internship interviews. Demonstrate the audit trail and negative finding. Production-readiness or investable-alpha claims would require verified fill behavior, field validation of the paper execution path, point-in-time universes and a future locked evaluation.

## Reproduction

From the repository root in PowerShell. The git-ignored data directory contains exact inputs and JSON reports. The prior audit archive contains the original Yahoo snapshot. Account integration was verified using connected read-only Alpaca calls. The local SDK's metadata and pagination paths used mocks in automated tests.

```powershell
$py = ".\.venv\Scripts\python.exe"
$bars = "audit-runs/v3/alpaca.csv"
& $py -m unittest discover -s tests -v
& $py bot.py
& $py bot.py run --synthetic --mode dry-run
& $py -m trading.quant coverage --data $bars --calendar audit-runs/v3/calendar.csv
& $py -m trading.quant compare --config config.research.example.json --data audit-runs/2026-11-22/audit-history.csv
& $py -m trading.quant compare --config config.research.example.json --data $bars --end 2025-12-31
& $py -m trading.quant compare --data $bars
& $py -m trading.quant study --data $bars
& $py -m trading.quant study --data $bars --start 2023-01-03 --end 2023-12-29
& $py -m trading.quant study --data $bars --start 2024-01-02 --end 2024-12-31
& $py -m trading.quant study --data $bars --start 2025-01-02 --end 2025-12-31
& $py -m trading.quant study --data $bars --cost-multipliers 2
& $py -m trading.quant study --data $bars --train-symbols AAPL,MSFT,JPM,XOM
```

Add `--train-sessions 504` to a study command to evaluate a two-year fixed-length rolling training window. The tables above use expanding development windows. Provider downloads and `verify-paper` require private paper credentials. Archived history, tests, synthetic demo and dry run do not. The original V3 suite contained 27 tests. The final portfolio audit reran the experiments and passed 35 tests, with no automated order submissions. See the [portfolio audit](PORTFOLIO_AUDIT.md) for the replay comparison.
