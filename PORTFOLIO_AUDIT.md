# Portfolio-readiness audit

## Publication assessment

The repository is suitable for an internship portfolio as a research and engineering case study. It includes a credential-free demonstration, clearly labeled market-data results, evidence for a negative ML finding, and automated safety tests. The two historical market-data snapshots are local and Git-ignored. A fresh clone can run the synthetic demo and tests. Exact historical replay requires obtaining the specified snapshots separately or using the existing local archive. Keep raw provider data and local broker journals out of publication.

## Findings and fixes

| Priority | Finding | Evidence and change |
| --- | --- | --- |
| High | Explicit holdout dates failed with the default expanding lookback. | The date lookup yielded a NumPy integer and the validator expected a Python integer. Normalize the calculated index and test a dated split with unchanged training settings. |
| High | Externally supplied paper credentials gave raw process errors, and session output could be sent to the optional assistant. | Require paper credentials before constructing the client, provide readable failures, and restrict assistant context to research metrics, regime and risk configuration. The tests use mock clients and verify account, positions, orders, journals and paths are excluded from assistant context. |
| Medium | Development MAE and bias for a restricted training universe included excluded securities. | Filter before scoring development rows. This changes only those diagnostic fields in the cross-sectional study. Models, trades and performance comparisons are unchanged. |
| Medium | The V2 price-feature overwrite was easy to miss in the presentation. | Document the three colliding names. Existing tests cover both the corrected namespace and legacy replay, and compare batched and single-date predictions. |
| Medium | A dry-run call could divide by zero before reaching the risk gate. | Validate target, price and ATR before sizing, and test that invalid input and an unarmed paper request leave the journal empty and never contact the mock broker. |
| Medium | Unignored standard CLI outputs could be staged accidentally. | Ignore default historical exports, the evaluation archive, root SQLite journals and common build and test outputs. Keep the intentionally curated charts and synthetic screenshot. |

## Claims checked

- Read the actual strategy, feature, selector, simulator, reporting and execution paths against the research report. The six strategies and model settings were preserved. The V3 relative component uses prefixed strategy features. The baseline leg and the operating selector deliberately retain the legacy mapping.
- The training labels charge both sides, are limited to completed outcomes, and use the next open. The suite tests those assumptions, missing intermediate data, feature prefix invariance, holdout independence, cash reconciliation, fit boundaries, strategy-column identity, and paper execution failure modes.
- The paper credential pair was previously verified by direct read-only HTTP requests against the official paper account, positions and historical endpoints. This audit used mocks for broker behavior. Actual order acceptance, fills and cancellation remain unverified.
- The tracked and publishable working files and all three commits reachable in the local clone were searched for recognizable OpenAI and Alpaca credentials, literal account IDs, local user paths and private-key markers. No matches were found. The two notebooks had no saved cell outputs. This is a targeted scan rather than a guarantee about all secrets or remote-only Git objects. The paper credentials shared in chat should be rotated separately.
- The charts use archived IEX results and are labeled as inspected historical evaluation. The screenshot is a completed synthetic offline backtest captured from the application.

## Reproduce

Use the exact commands in the [research report](QUANT_RESEARCH_REPORT.md). Figure regeneration is optional and uses Matplotlib 3.10.6.

```powershell
$py = ".\.venv\Scripts\python.exe"
& $py -m unittest discover -s tests -v
& $py bot.py dashboard --synthetic --mode offline
& $py scripts/plot_portfolio.py audit-runs/v3/all-methods.json audit-runs/v3/corrected-final.json
```

Validated on Python 3.12.7 with NumPy 2.1.3, pandas 2.3.3, SciPy 1.17.1, scikit-learn 1.8.0, statsmodels 0.15.0 and alpaca-py 0.45.0. Provider revisions and different dependency versions can affect historical fits. The archived SHA-256 digests identify the inputs used.

## Limits that matter in an interview

The equity universe is retrospective. Corporate-action-adjusted data can be revised. The 2026 evaluation was inspected during development. Estimated fills are not auction fills, and portfolio return comparisons reflect large exposure differences. The relative ensemble remains research-only. The repository supports a portfolio-quality demonstration and does not establish production trading readiness or investable alpha.

## Test and repository gates

The full suite passed 35 tests, including the original 27 and eight new regression tests. The nine-test V3 module also passed after expanding its feature-mapping assertion to cover all three colliding columns. The offline deployment and synthetic dashboard backtest completed. Git whitespace checks passed. The exact supplied paper key and secret were also searched independently, with zero matches in publishable working files and reachable local history.

The code uses small model and broker interfaces and separates simulation, research reporting and execution. Pandas data frames, JSON configuration and estimator wrappers remain dynamic. This audit verified their critical boundaries with tests and did not run a repository-wide static type checker. No settings were tuned against the inspected 2026 period.

## Replay result

Re-ran the exact archived Yahoo replay, four-stock Alpaca comparison, all-method 2026 comparison, V3 study, 2023–2025 walk-forward studies, double-cost study, cross-sectional study and calendar coverage report. All input hashes and historical performance values reproduced within numerical tolerance. Two older reports lack the now-emitted training-symbol metadata. The cross-sectional study changed only the development error scope. MAE went from 0.017586 to 0.018046 and bias moved from 0.004648 to approximately zero. Trades, portfolio metrics, bootstrap estimates and holdout diagnostics reproduced. The reruns are retained locally in `audit-runs/v3/portfolio-replay`.

**Decision:** No critical blocker remains for publishing the code, documentation and curated figures after review. Keep raw downloads and broker state local. Public readers can run the offline demo and tests immediately. Exact historical replay still depends on the archived data. Authenticated order acceptance, partial fills and cancellation remain outside the verified scope.
