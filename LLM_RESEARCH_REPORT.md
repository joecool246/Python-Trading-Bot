# V4 LLM research report

## Question and status

Do timestamped, source-grounded LLM features add predictive value beyond a price-only quantitative model? V4 implements the experiment. **Real incremental predictive value is unmeasured.** Neither provider credential was present in the development environment, and the available Alpaca snapshot contains no completed bars after the inspected V3 boundary. The original V3 negative finding remains unchanged.

The exact Alpaca paper credentials supplied for this task authenticated the paper account, positions and historical-data endpoints with HTTP 200. The account was active. An IEX SPY request covering November 21–26, 2026 returned zero daily bars at verification time. The archived CSV ends November 20, 2026. The historical research workflow returned `insufficient_future_market_data` for a November 23 start. These were read-only calls. No paper or live orders were submitted or canceled.

## Sources and extraction

The SEC collector requests recent 8-K, 10-Q, 10-K, 6-K and 20-F documents for a validated ticker/CIK pair. It uses EDGAR acceptance time as a publication proxy and records the actual observation time when downloading. It retains a bounded primary-document excerpt. Some earnings announcements are available only as exhibits and are not automatically collected. For licensed news, issuer earnings releases and other disclosures, use a source manifest with the original publication timestamp, actual first-observed timestamp, canonical HTTPS citation, permitted excerpt and a processing-rights statement. No licensed news subscription is bundled. SEC availability does not establish permission to redistribute every issuer exhibit. [SEC EDGAR data APIs](https://www.sec.gov/search-filings/edgar-application-programming-interfaces)

A provider adapter sends the excerpt as untrusted data without tools or broker access. Both adapters return the same validated schema. Direction and sentiment are ordinal. Earnings surprise and guidance changes require a source comparator and may be null. Evidence must appear verbatim in the source. This confirms provenance of the quote and does not establish factual correctness of its interpretation. Provider output is research metadata, separate from market prices and broker state. Provider errors or invalid output leave an explicit unavailable result.

Only a successful, cached or newly completed extraction produces an event. The archive records source and extraction timestamps, provider, model and cited evidence. The research join uses events strictly before 13:00 New York on the decision date, conservatively covering scheduled early closes. Events age out by publication time. Exact versions are deduplicated and a new version supersedes the earlier version only when it becomes available. Processing historical documents today cannot backdate their feature availability. The input archive still depends on truthful upstream provenance.

## Controlled comparison

The principal comparison uses the same gradient-boosting configuration, feature normalization convention, target, universe, split and simulator for two models. One model uses corrected, namespaced price and strategy features. The second adds event count, sentiment, direction, uncertainty, risk, earnings surprise, guidance change and corporate-event features. A neutral numeric feature vector with zero event count means no eligible successful extraction. It does not certify negative or neutral news.

Report the original V3 operating selector, the V3 relative ensemble when available, equal-weight and rule allocation, SPY buy-and-hold, and SPY scaled to lagged event-model exposure. The simulator retains position and portfolio exposure gates, prior-volume capacity, friction, prior-close regimes and next-open fills. Scores must exceed the existing edge threshold and the strategy count caps breadth. A separate sensitivity subtracts a one-leg fee-plus-slippage penalty from new strategy utility and reserves an additional 10% of its proposed allocation as cash. This sensitivity is not part of the matched feature-only comparison.

The holdout start must be declared after November 20, 2026 with actual market bars and eligible event coverage on both sides. Freeze event processing, model settings, budget, source universe and thresholds before inspecting a fresh cohort. Purge label outcomes at the training cutoff. Use equal weighting and the matched price model as primary comparators, and paired session-block intervals to express uncertainty. The default 1 percentage point hurdle is unchanged. No hyperparameter search used the inspected V3 block. Incremental information may still reflect source coverage or correlated issuer activity, so a prospective study should add a source-count-only ablation and robust coverage requirements.

## Costs and testing

The default OpenAI adapter uses GPT-6 Luna. The Anthropic alternative uses Claude Haiku 5.5. With the configured standard rates of $0.10 per million input tokens and $0.50 per million output tokens, 1,000 input plus 250 output tokens costs approximately **$0.000225 per successful request**, or **$0.225 per 1,000 requests** before retries. Actual charges depend on current provider billing and request content. No paid LLM requests were made for this implementation. [OpenAI model pricing](https://developers.openai.com/api/docs/models/gpt-6-luna) · [Anthropic model pricing](https://platform.claude.com/docs/en/models/haiku-5-5/overview)

The SQLite ledger reserves an estimated per-attempt amount before network access, rate-limits requests across processes sharing it, settles usage on successful responses, and preserves reservations for unconfirmed attempts. The default $0.25 limit applies to the lifetime of one retained ledger. Input byte counts plus framing are a conservative estimate, not a provider-side guarantee. Failed requests can consume a reservation and limits stop further requests rather than silently inventing data. Cache reuse requires compatible source and extraction provenance. No output-quality or semantic calibration claim is made.

The synthetic experiment exercises the full comparison and uncertainty pipeline using deterministic event values independent of prices. Synthetic performance is not evidence of an effect. The automated suite uses mocked provider, SEC and broker requests; see the test and CLI commands in the README.

## Verification record

All 47 tests passed, including the existing 35 V3 tests. The offline broker-independent trading run and the V4 synthetic end-to-end study completed. The latter included 1,389 development and 486 evaluation stock-sessions with eligible synthetic events. The genuine historical run stopped at its future-data check and reported no performance metrics. Local, Git-ignored outputs are in llm-runs/v4-fixture-study.json and llm-runs/v4-real-gate.json.
