# Allocation research

The 2022 development screen fits the V4 legacy operating estimator on pre-2022 data. All alternatives share the same universe, fee and slippage assumptions, volume limit, and independent order gate. The lower-risk ablations tighten one existing constraint at a time. None of the alternatives was tuned on 2026, promoted, or incorporated into the stable candidate.

The research uses existing scores and covariance-free heuristics. Volatility targeting scales high-volatility sleeves down toward 18% annualized instrument volatility. Risk-parity-inspired sizing scales sleeves down using the contemporaneous harmonic-volatility reference across names and holds unused allocations in cash. It is not a covariance-aware or fully invested risk-parity optimizer. The confidence screen raises the entry hurdle by 20 bps of modeled utility. The cost-aware screen adds one stated fee-plus-slippage allowance to changes in the selected strategy set. Five-session rebalancing and a 10% daily filled-notional cap are distinct turnover controls; the latter can delay reductions. Each is an analytical sensitivity, not a new execution policy.

Gross and net statistics respond to different overnight marks and risk headroom, so they are indicative comparisons. Strategy-set changes count a symbol's requested strategy-name set changing at a rebalance, including transitions from or to cash. They are not fill counts. Concentration is measured by mean maximum position and strategy weight, as fractions of total NAV. Exposure by regime uses the prior close's regime. The exposure-matched comparator scales SPY by the baseline's lagged ML exposure and estimates rebalancing costs. It is not an order simulator.

The 2023–2025 base-nine analysis shows 2.28% ML mean exposure and 597 all-cash sessions, versus 42.11% and zero for equal weighting. Annualized returns were 1.90% and 7.10%. The lagged exposure-matched passive comparator returned 0.75%. These differences support insufficient market participation as a material contributor to the return gap, with trade costs and strategy ranking also relevant. The evidence does not identify a causal decomposition or show that a different exposure target would succeed under the same risk constraints. Regime labels have no independent ground truth here. No policy passed a defensible, untouched promotion test; the existing allocation remains the operating default.

```powershell
$py = ".\.venv\Scripts\python.exe"
& $py -m trading.strategy_study allocation --data audit-runs/v3/alpaca.csv --config config.v3.example.json --start 2022-01-03 --end 2022-12-31 --output research-runs/allocation-final.json
& $py -m trading.strategy_study allocation --data audit-runs/v3/alpaca.csv --config config.v3.example.json --start 2023-01-03 --end 2025-12-31 --policies baseline equal rules --output research-runs/base-final.json
```

## Development screen results

| Policy | Annualized return | Mean exposure | Gross turnover | All-cash sessions |
| --- | ---: | ---: | ---: | ---: |
| Baseline ML | -1.83% | 13.66% | 28.49x | 94 |
| Equal strategy weights | -3.26% | 30.04% | 33.62x | 5 |
| Regime rules | -2.52% | 14.55% | 15.08x | 66 |
| Volatility target | -2.46% | 9.76% | 20.24x | 94 |
| Capped inverse volatility | -2.50% | 11.98% | 24.92x | 94 |
| Higher utility threshold | 1.11% | 11.37% | 25.97x | 119 |
| Cost-aware entry | -0.46% | 13.03% | 27.09x | 102 |
| Five-session rebalancing | 4.83% | 14.46% | 10.55x | 90 |
| 10% daily turnover cap | -2.21% | 9.52% | 15.76x | 80 |
| 10% position cap | -1.83% | 13.66% | 28.49x | 94 |
| 35% portfolio-exposure cap | -0.96% | 7.30% | 15.25x | 94 |
| Half risk per trade | -2.30% | 12.74% | 25.90x | 94 |
| Exposure-matched SPY | -2.52% | 13.61% | 20.78x | — |

The five-session policy's active-return interval versus equal weighting was -6.22 to +22.93 annualized percentage points. Its 10.55x turnover and $651 estimated costs compare with 28.49x and $1,699 for baseline ML. Narrowing the position cap had no effect at realized allocations, while lowering gross portfolio exposure or trade risk reduced deployment. In the later base-nine comparison, ML averaged 0.92% exposure on bullish sessions and 7.23% on bearish sessions. Mean largest position and attributed strategy weight were 1.47% and 2.18% of NAV. These low absolute weights are consistent with concentrated inactivity rather than a heavily invested portfolio. The analyses did not establish a promotion candidate.
