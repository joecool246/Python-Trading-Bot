# Trading Bot V2 audit

## Prioritized findings

| Priority | Finding | Fix and verification |
| --- | --- | --- |
| High | Paper sells used asks, invalid quote values could reach sizing, and reconciliation was not scoped to the account. | Use the bid for sells and ask for buys. Abort for stale or crossed quotes, wide spreads, invalid allocations, stale clocks, changed accounts and unavailable shares. Scope journal reconciliation by account, preserve monotonic fill quantities and retain unresolved intents. Tested with mock orders and broker snapshots. |
| High | Downloads could write an incomplete data file, and research silently dropped symbols missing from the CSV. | Normalize and verify the configured universe, include the benchmark, validate and atomically replace CSV files, and expose date, feed and adjustment options. Mock HTTP pages go through Alpaca's actual SDK request, pagination and bar parsing before CSV reload. The real public data also exposed expected floating-point differences after corporate-action adjustment. The loader tolerates rounding at 1e-12 relative error and still rejects materially inconsistent prices. |
| High | The simulator could indefinitely carry stale holdings, size orders off changing intra-batch equity, and fill arbitrarily large orders at the open. | Bound stale marks and prior-volume participation, use one opening equity snapshot for targets and process reductions first. Test next-open timing, fill quantities, accounting and stale-price failure. Portfolio marks still use adjusted bars and a fixed friction estimate. |
| High | The reporting and operating paths shared one fit, and sparse symbol calendars could stretch forward labels. | Separate frozen evaluation and deployment fits. Align labels to benchmark sessions, reject gaps and purge labels crossing fit boundaries. Paper additionally requires the previous completed exchange session. Verify that changes in holdout data do not change a pre-holdout decision. |
| Medium | Identical entry dates could obscure differences in strategy exits and exposure. | Model each strategy's exit timing and entry scale, charge both legs, and supply identity and scale as inputs. Identical policies and exposure can correctly have identical realized outcomes. |
| Medium | Same-close regime labels described returns already realized that day, and the evaluation lacked paired uncertainty. | Attribute returns to the prior-close regime. Compare paired session returns with a fixed-seed circular block bootstrap and include sample counts. |
| Medium | The dashboard repeatedly trained on unchanged inputs. | Cache one model per fit mode using the configuration and CSV path, size and modification time. Verify cache reuse and invalidation. |

## Historical evaluation

Yahoo Finance daily observations for SPY, AAPL, MSFT, JPM and XOM from January 2015 through December 2025. Open, high and low were scaled by adjusted close divided by raw close. Volume is the source volume. The last development label ended December 31, 2021. The frozen holdout ran January 3, 2022 through December 31, 2025, covering 1,003 sessions. The CSV SHA-256 was `eb5cdd9a7b4162b44902527e1eadf7445ddabcca49586697db2248bfe73e22b6`. The source can revise past data.

| Method | Annualized return | Sharpe | Sortino | Max drawdown | Gross turnover | Costs, USD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SPY buy-and-hold | 10.93% | 0.67 | 0.97 | -24.44% | 0.92x | 59.82 |
| Momentum | -2.30% | -0.56 | -0.73 | -11.92% | 89.87x | 5,169.14 |
| Mean reversion | 3.25% | 0.88 | 1.99 | -3.20% | 32.68x | 2,113.20 |
| Trend | 1.36% | 0.36 | 0.49 | -7.64% | 47.40x | 2,858.50 |
| Breakout | -0.29% | -0.19 | -0.26 | -3.51% | 35.16x | 2,081.65 |
| Pairs, long leg | 0.40% | 0.89 | 1.57 | -0.41% | 1.21x | 72.80 |
| Volatility adaptive | 1.59% | 0.38 | 0.51 | -7.89% | 51.28x | 3,153.50 |
| Equal-weight | 4.81% | 0.76 | 1.14 | -9.06% | 91.54x | 6,036.82 |
| Rule selector | 1.60% | 0.43 | 0.60 | -8.16% | 59.17x | 3,576.73 |
| ML selector | 0.91% | 0.45 | 0.79 | -2.46% | 20.52x | 1,274.62 |

Turnover is total traded notional divided by mean equity across the period. Sharpe and Sortino use a zero return reference. SPY buy-and-hold is essentially fully invested. The other methods share risk limits and can hold cash, so benchmark comparisons are not exposure matched.

| Prior-close SPY regime | Sessions | ML compounded return | Equal-weight compounded return |
| --- | ---: | ---: | ---: |
| Bearish | 149 | 6.63% | 7.42% |
| Bullish | 410 | -2.47% | 5.61% |
| High volatility | 35 | -0.17% | 1.26% |
| Sideways | 409 | -0.14% | 4.94% |

The primary comparison was ML minus equal-weight. Paired annualized arithmetic active return was **-3.98 percentage points**, with a 95% circular-block bootstrap interval of **-9.38 to +1.50 points**. Against rule selection it was **-0.73 points** with a **-4.48 to +3.31** interval. Neither cleared the 1-point materiality hurdle. The model had lower observed drawdown and turnover, but this sample did not establish a material return advantage. The paired intervals use 2,000 draws of ten-session blocks. Secondary comparisons are exploratory and unadjusted for multiple tests.

The four-stock universe was selected retrospectively and has survivorship and selection bias. Adjusted bars are provider-revised and are not a verified point-in-time corporate-action feed. There is no explicit delisting or market-impact model. Liquid next-open fills remain an assumption. A genuinely broad evaluation needs point-in-time constituent membership, verified delisting cash flows, and sensitivity to costs and fill assumptions. Authenticated Alpaca historical and paper behavior remain unverified. Paper is off by default and all order tests used mocks.

## Reproduce

From the repository root in PowerShell, using the repository virtual environment:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\scripts\Get-AuditHistory.ps1 -Output audit-history.csv
.\.venv\Scripts\python.exe bot.py backtest --config config.research.example.json --data audit-history.csv
.\.venv\Scripts\python.exe bot.py train --config config.research.example.json --data audit-history.csv --deployment
.\.venv\Scripts\python.exe bot.py run --config config.research.example.json --data audit-history.csv --mode dry-run
```

To run the same comparison using Alpaca data, set the read credentials and request the same date window. Data provider and adjustment differences can change results. Neither command calls broker order submission.

```powershell
.\.venv\Scripts\python.exe bot.py download --config config.research.example.json --start 2015-01-01 --end 2026-01-01 --feed iex --adjustment all --output alpaca-history.csv
.\.venv\Scripts\python.exe bot.py backtest --config config.research.example.json --data alpaca-history.csv
```

The 18-test suite passed in the verification environment, including mocked pagination, execution safeguards, label boundaries, liquidity, portfolio accounting and cache invalidation.

The original input and full JSON report are retained in the local, Git-ignored audit-runs/2026-11-22 directory. Use the archived input to reproduce the exact run independently of any later provider revisions.

```powershell
.\.venv\Scripts\python.exe bot.py backtest --config config.research.example.json --data audit-runs/2026-11-22/audit-history.csv
```
