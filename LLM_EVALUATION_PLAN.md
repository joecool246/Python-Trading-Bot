# Future LLM evaluation plan

No paid model calls were made in this iteration. Keep provider credentials unset until a separate evaluation is authorized. The existing OpenAI and Anthropic interfaces, cache, retries, rate and spending reservations, and causal event-feature join are exercised by mocks and clearly labeled synthetic fixtures.

1. Prospectively collect 120 issuer events from at least eight names across sectors, stratified by earnings, guidance, filings/risk and corporate actions. Accept only public or rights-cleared excerpts. Predeclare inclusion before returns are available. Record the original publication, observed and extraction timestamps, canonical link, exact excerpt hash, rights statement and model snapshot.
2. Double-label a blind sample of 30 events for event type, direction, explicit guidance/earnings comparator, material risk and evidence span. Report per-field accuracy, missing/abstention rates, inter-reviewer agreement and verbatim citation pass rate. Adjudicate ambiguous cases after independent labels. Do not use generated prose as ground truth.
3. Estimate budget before enabling processing. At the configured rates of $0.10 and $0.50 per million input and output tokens, 120 requests averaging 2,000 input and 250 output tokens cost approximately $0.039 in standard token charges. Set a $0.25 retained-ledger ceiling and use a hard provider account limit where available. Unknown attempts retain cost reservations. Record actual token receipts, retry counts and uncovered sources.
4. Freeze extraction rules and establish a new untouched period after achieving substantive coverage. Compare the namespaced price-only baseline against price-plus-event features, equal weighting, rules and exposure-matched SPY. Add a source-count-only ablation before attributing results to interpretation rather than news coverage. Require extraction availability strictly before the configured decision cutoff, real next-open fills in the simulation and purged outcomes on the training boundary.
5. Report feature coverage and return/Sharpe differences with paired blocks. Compare costs, turnover, regime exposures and potential lift. Keep the observed 2026 V3 block out of selection and future significance claims.

```powershell
$py = ".\.venv\Scripts\python.exe"
& $py -m unittest discover -s tests -p test_llm_research.py -v
& $py -m trading.llm_research study --synthetic --output llm-runs/rehearsal.json
```

The paid extraction command, provider configuration and source formats are documented in the V4 README and [LLM research report](LLM_RESEARCH_REPORT.md). The rehearsal uses no provider credentials. No real LLM return estimate exists yet.
