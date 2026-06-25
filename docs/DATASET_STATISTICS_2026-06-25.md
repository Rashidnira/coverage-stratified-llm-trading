# Dataset Statistics — Coverage-Stratified Multi-Source Trading Corpus
*Computed from `qwen_news_summaries/<TICKER>/` (16 firms × 10 sources, 2023–2025). Script-reproducible; raw counts in
`dataset_stats_out.json`. Generated 2026-06-25.*

## Overview
This study releases a **coverage-stratified multi-source corpus** for LLM trading-agent evaluation, spanning **16 U.S.
equities** partitioned ex-ante into **6 highly-covered mega-caps** (34–61 analysts) and **10 under-covered firms**
(1–7 analysts), over **Jan 2023 – Dec 2025**. It integrates **ten heterogeneous information sources** — news, social
media, technical, market, industry, fundamentals, analyst-surprise (SUE), and SEC 8-K/10-Q/10-K filings — aligned to a
common trading-day index. The labeled benchmark contains **12,016 annotated instances** (one per firm-trading-day,
3-class BUY/HOLD/SELL on a ±0.5% next-day market-adjusted return band).

## Table 1 — Source statistics (count; total / by coverage tier)
| Source | Granularity | **ALL** | HIGH (6) | UNDER (10) |
|---|---|--:|--:|--:|
| **Annotated instances** (labeled firm-days) | daily | **12,016** | 4,506 | 7,510 |
| News — daily briefs | near-daily | 10,291 | 6,352 | 3,939 |
| News — headlines (summarized) | per-brief | 224,431 | 202,236 | 22,195 |
| Social — WSB posts | near-daily | 21,507 | 21,491 | **16** |
| Social — active post-days | — | 4,355 | 4,340 | 15 |
| SEC filings (8-K/10-Q/10-K) | event | 605 | 212 | 393 |
| SUE earnings-surprise events | quarterly | 172 | 72 | 100 |
| Fundamentals (firm-quarters) | quarterly | 192 | 72 | 120 |
| Technical / Market / Industry | continuous | 12,016 | 4,506 | 7,510 |
| Price / return days | continuous | 12,032 | 4,512 | 7,520 |

*(News raw RavenPack pipeline = 293,174 unique events after dedup, §3.2.1; the count above is summarized-brief headlines fed to the LLM.)*

## Coverage gradient (the central design property)
- **Social media is effectively mega-cap-only:** **21,491** WSB posts on the 6 HIGH firms vs **16** on the 10 under-covered firms (≈1,300×). Confirms the WSB signal concentrates in visible names (Merton incomplete-information).
- **News is steeply coverage-dependent:** 202,236 HIGH headlines vs 22,195 UNDER (≈9×); per firm ≈33,700 vs ≈2,220.
- **Regulatory filings are ~uniform per firm:** 212/6 ≈ **35 HIGH** vs 393/10 ≈ **39 UNDER** per firm — mandatory disclosure does not scale with coverage, making filings the candidate signal where alt-data is absent.
- **Accounting/analyst sources** (SUE, fundamentals) are uniform by construction (≈12 firm-quarters each).

## Label distribution (3-class, ±0.5% band)
| Tier | BUY | HOLD | SELL | total |
|---|--:|--:|--:|--:|
| HIGH | 1,604 | 1,412 | 1,490 | 4,506 |
| UNDER | 2,561 | 2,038 | 2,911 | 7,510 |
| **ALL** | **4,165** | **3,450** | **4,401** | **12,016** |

(Under-covered tier skews more SELL, consistent with its negative 2025 drift.)

## Sub-source breakdowns
- **Filing types:** 8-K 423 · 10-Q 134 · 10-K 48 (total 605).
- **Social sentiment (LLM-labeled):** BULLISH 7,868 · NEUTRAL 7,918 · BEARISH 5,691 (n/a 30).

## Table 2 — Per-firm statistics
| Ticker | Tier | Instances | News briefs | Headlines | WSB posts | Filings |
|---|---|--:|--:|--:|--:|--:|
| AAPL | HIGH | 751 | 1,096 | 46,198 | 1,654 | 33 |
| TSLA | HIGH | 751 | 1,087 | 29,393 | 5,977 | 40 |
| MSFT | HIGH | 751 | 1,096 | 34,358 | 1,421 | 35 |
| AMZN | HIGH | 751 | 912 | 24,719 | 1,338 | 27 |
| GOOGL | HIGH | 751 | 1,092 | 43,578 | 778 | 43 |
| NVDA | HIGH | 751 | 1,069 | 23,990 | 10,323 | 34 |
| NBTB | UNDER | 751 | 409 | 2,272 | 1 | 51 |
| ANDE | UNDER | 751 | 432 | 2,461 | 2 | 36 |
| PRGS | UNDER | 751 | 519 | 3,633 | 3 | 36 |
| MYRG | UNDER | 751 | 506 | 3,029 | 0 | 45 |
| PAHC | UNDER | 751 | 409 | 2,101 | 0 | 44 |
| IIIN | UNDER | 751 | 289 | 1,428 | 2 | 43 |
| WDFC | UNDER | 751 | 487 | 2,842 | 6 | 32 |
| JJSF | UNDER | 751 | 402 | 1,993 | 0 | 26 |
| YORW | UNDER | 751 | 119 | 605 | 2 | 41 |
| MSEX | UNDER | 751 | 367 | 1,831 | 0 | 39 |

## Provenance
Per-ticker sources `qwen_news_summaries/<TK>/<TK>_{news_daily_summary.csv, social_, filings_, sue_, fundamentals_,
technical_, market_, industry_, price_return_}*`; labeled instances `<TK>_training.jsonl` (field `label`). Aggregator:
inline over those files → `dataset_stats_out.json`.
