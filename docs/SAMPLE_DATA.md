# Sample Data — Raw vs Processed (fed to the LLM) — ALL 10 SOURCES
*Before/after examples of every source's raw feed and the LLM-processed text the trading agents actually read.
One HIGH-coverage and one LOW-coverage example per source. Real records pulled from the dataset. Generated 2026-06-25.*

**The unified "processed" form** — all 10 sources are rendered into one daily `state_text` block per firm. Real low-coverage
example (WDFC 2023-10-23) showing every source at once, exactly as the classifier/agent reads it:
```
Stock WDFC, date 2023-10-23.
- news      : **Revenue Beat:** Bullish … **Net Tilt:** Strongly Bullish.
- social    : SILENT (no WSB posts)
- technical : RSI 61.1 (neutral), OBV falling
- market    : VIX 20.4, regime bear, 60d-mom -0.08
- industry  : sector XLP lagging (rel -0.01212)
- 8K  (age 2/5d) : sentiment 0.7  tone optimistic impact positive
- 10K (age 0/21d): sentiment 0.13 tone cautious   impact —
- SUE (age 1/3d) : Earnings BEAT: actual EPS $1.21 vs consensus $1.2055 (+0.4%); SUE z +0.20
- FUND(age 1/21d): valuation OVERVALUED trend IMPROVING
```
Sources below show the raw→processed transformation behind each of these lines.

---

## 1. NEWS

**How we process it.** Raw RavenPack stories (filtered to relevance = 100, deduped on event text → 293,174 events;
each item carries an Event Sentiment Score **ESS** ∈ [−1,1], CSS, and an event category) are grouped by day per ticker
and passed to **Qwen2.5-14B**, which collapses the day's headlines into one signal-preserving, theme-grouped trader
brief — each point tagged bullish/bearish with an explicit **Net Tilt**, preserving concrete numbers (EPS, revenue,
price targets old→new). Same-day headlines only (leakage-safe).

**Data path.** Raw: `news_json/<TICKER>.json` (per-day `news[]` of `{ess, css, category, event_text}`). →
`qwen_news_summaries/extract_news.py` → raw CSV → `qwen_news_summaries/summarize.py` (Qwen2.5-14B) →
**Processed:** `qwen_news_summaries/<TICKER>/<TICKER>_news_daily_summary.csv` (`date, n_headlines, summary`).
Process style: **abstractive LLM summarization** (110–170 headlines/day → ~1 theme-grouped brief).

### Example A — HIGH-coverage: TSLA, 2023-01-03  *(110 headlines; RavenPack net_ess −18.58; 30 pos / 67 neg / 13 neu)*

**RAW** (`news_json/TSLA.json` → day's `news[]`, sample of 7 of 110):
```
[-0.56 | supply-below-expectations]   Tesla Misses Production Targets
[-0.44 | product-price-cut]           Tesla also slashed prices
[-0.74 | earnings-below-expectations] Tesla missed Wall Street estimates
[+0.65 | revenue-up]                  Tesla car sales grow
[+0.65 | revenue-up]                  Tesla posted record sales
[+0.47 | stock-gain]                  Tesla rose by
[-0.57 | sanctions-target]            South Korea fines Tesla $2.2 million for
… (103 more headlines that day)
```

**PROCESSED** (`TSLA_news_daily_summary.csv`, the text fed to the LLM):
```
**Bearish Signals:**
- **Price Target Downgrades:** Cowen lowered PT to $122 from $205; KGI Securities to $230 from $360;
  RBC Capital to $186 from $225; Goldman Sachs to $205 from $235; Deutsche Bank to $250 from $270;
  Piper Sandler to $340 from $360.
- **Revenue/Earnings Shortfalls:** 2022 sales/revenue missed expectations; Q4 delivery estimates missed;
  2022 sales fell short; Tesla missed EPS estimates.
- **Stock Performance:** Shares fell 65% in 2022; tumbled to start 2023; down over 60%.
- **Supply/Demand Issues:** Missed production targets; Shanghai factory suspension; Model X deliveries down 8%.

**Bullish Signals:**
- **Revenue Growth:** 2022 deliveries up 40%; posted record sales; EV sales grew.

**Net Tilt:** Bearish due to multiple downgrades and revenue/earnings misses outweighing growth signals.
```

### Example B — LOW-coverage: JJSF (J&J Snack Foods), 2024-01-16  *(only 7 headlines; RavenPack net_ess +2.21; 6 pos / 1 neg)*

**RAW** (`news_json/JJSF.json` → day's `news[]`, all 7):
```
[-0.34 | ownership-decrease-held]      Inc. Sells 80 Shares of J&J Snack Foods Corp. (NASDAQ:JJSF)
[+0.37 | ownership-increase-held]      Blackrock Inc. lifted its holdings in J&J Snack Foods by 0.6% (Q2)
[+0.37 | ownership-increase-held]      Vanguard Group lifted its holdings by 1.9% (Q3)
[+0.37 | ownership-increase-held]      Allspring Global Investments lifted holdings by 0.7% (Q3)
[+0.37 | ownership-increase-held]      Macquarie Group lifted holdings by 1.1% (Q1)
[+0.37 | ownership-increase-held]      Morgan Stanley lifted holdings by 319.3% (Q4)
[+0.70 | eps-above-expectations]       J&J Snack Foods reported $1.73 EPS, topping the consensus estimate
```

**PROCESSED** (`JJSF_news_daily_summary.csv`, the text fed to the LLM):
```
**Earnings:** Bullish. J&J Snack Foods reported $1.73 EPS, beating the consensus estimate.
**Institutional Ownership:** Bullish. Multiple institutions increased their holdings: Blackrock (+0.6%),
  Vanguard (+1.9%), Allspring (+0.7%), Macquarie (+1.1%), and Morgan Stanley (+319.3%). However, Inc.
  sold 80 shares, slightly bearish.
**Net Tilt:** Bullish, with strong earnings beat and significant institutional buying outweighing minor selling.
```

**What the processing does (and the coverage gradient it exposes):** the *same* Qwen-14B pipeline turns a noisy feed
into one theme-grouped, number-preserving brief with a single decision-ready **Net Tilt** — but the raw input differs
enormously by coverage: the **high-coverage** name (TSLA) gets **110 headlines** of analyst targets, earnings and
production news in a day, while the **low-coverage** name (JJSF) gets only **7** — mostly mechanical 13F ownership
updates plus one earnings line. Same processing, far thinner signal for under-covered firms.

## 2. SOCIAL MEDIA SENTIMENT

**How we process it.** Raw r/wallstreetbets + r/StockMarket posts (Pushshift, queried by ticker/cashtag) are classified
by an LLM into directional **sentiment** (bullish/bearish/neutral) + confidence + post-quality + themes + a one-line
rationale; per day they are aggregated into a count + bull/bear split + theme list. Each post is mapped to the first
trading session on/after its UTC timestamp (leakage-safe).

**Data path.** Raw: `reddit_json/<TICKER>.json` (`posts[]` of `{time, subreddit, title, text, score}`) →
`wsb_signal.py` / `reflect_wsb.py` (LLM classify) → **Processed:** `qwen_news_summaries/<TICKER>/<TICKER>_social_2023_2025.json`
(per post: `sentiment, confidence, post_quality, themes, rationale`) → daily aggregate line.

### HIGH-coverage: TSLA, 2023-01-25
**RAW** (`reddit_json/TSLA.json`):
```
{title:"(1/25) Wednesday's Pre-Market Stock Movers & News", subreddit:"StockMarket", score:3,
 text:"Good morning traders … pre-market stock movers & news on this Wednesday …"}
```
**PROCESSED** (`TSLA_social_2023_2025.json`):
```
{sentiment:"BULLISH", confidence:0.9, post_quality:"OPINION", themes:["strong bullish","symbol"],
 rationale:"The poster expresses a strong bullish opinion on TSLA, referring to it as a 'strong bullish symbol'."}
→ daily line fed to LLM:  "social: 20 WSB posts, 5 bull / 5 bear, themes ['Big Boys','EV tax credits','R.I.P']"
```

### LOW-coverage: WDFC (and every under-covered firm) — **the data IS the finding**
**RAW** (`reddit_json/WDFC.json`): **0 posts in 3 years** (JJSF 0, MSEX 0, PAHC 0; whole under-covered tier = 16 posts total).
**PROCESSED** (`WDFC_social_2023_2025.json`): empty → line fed to LLM:
```
social: SILENT (no WSB posts)
```
*Social media is structurally mega-cap-only (21,491 HIGH posts vs 16 UNDER); its absence for under-covered firms is reported as a result, not imputed.*

---

## 3. TECHNICAL

**How we process it.** Daily adjusted OHLCV → standard indicators (RSI-14, MACD, Bollinger %B, OBV) computed from
trailing data only, then **discretized to states** (oversold/neutral/overbought, rising/falling) so the LLM reads a
condition, not a raw number.

**Data path.** Raw: `qwen_news_summaries/<TK>/<TK>_price_return_2023_2025.json` (OHLCV) → `build_ta_states.py` →
**Processed:** `<TK>_technical_2023_2025.json` (`RSI_14, rsi_state, obv_trend`).

### HIGH: TSLA, 2023-01-03
**RAW:** `open 118.47, high 118.80, low 104.64, close 108.10, volume 231,402,800` → RSI-14 = 24.9, OBV slope +.
**PROCESSED:** `{RSI_14:24.9, rsi_state:"oversold", obv_trend:"rising"}` → `technical: RSI 24.9 (oversold), OBV rising`

### LOW: WDFC, 2023-10-23
**RAW:** daily OHLCV → RSI-14 = 61.1, OBV slope −.
**PROCESSED:** `{RSI_14:61.1, rsi_state:"neutral", obv_trend:"falling"}` → `technical: RSI 61.1 (neutral), OBV falling`
*(Technical is coverage-invariant — full OHLCV exists for every firm; identical pipeline both tiers.)*

---

## 4. MARKET (coverage-invariant — shared across all 16 firms each day)

**How we process it.** Aggregate market block: CBOE VIX, SPY trend (200-day position, 60-day momentum, drawdown, 20-day
vol) and Fama-French factors → a coarse **regime label** (bull/bear/neutral) + momentum, from trailing data only.

**Data path.** Raw: VIX/SPY/Fama-French (Kenneth French library, WRDS) → `build_daily_signal.py` →
**Processed:** `<TK>_market_2023_2025.json` (`vix, market_regime, mom_60d`).

### HIGH date: TSLA, 2023-01-03 | ### LOW date: WDFC, 2023-10-23  *(same block for every firm on a given day)*
**RAW:** `VIX 22.9, SPY 60d-momentum +0.02, regime stats…`  →  **PROCESSED:** `market: VIX 22.9, regime neutral, 60d-mom 0.02`
**RAW:** `VIX 20.4, SPY 60d-momentum −0.08, drawdown…`       →  **PROCESSED:** `market: VIX 20.4, regime bear, 60d-mom -0.08`

---

## 5. INDUSTRY

**How we process it.** Each firm is mapped (GICS) to its SPDR sector ETF; we compute the sector's trailing 20-day return
vs the market's and label the firm's sector **leading/lagging**, so abnormal returns are read relative to the sector.

**Data path.** Raw: sector-ETF & market returns → `build_daily_signal.py` →
**Processed:** `<TK>_industry_2023_2025.json` (`sector_etf, sector_minus_market, relation`).

### HIGH: TSLA, 2023-01-03
**RAW:** `XLY 20d-ret −0.118, market 20d-ret −0.064` → diff −0.054.
**PROCESSED:** `{sector_etf:"XLY", sector_minus_market:-0.054, relation:"lagging"}` → `industry: sector XLY lagging (rel -0.05385)`

### LOW: WDFC, 2023-10-23
**RAW:** `XLP 20d-ret vs market 20d-ret` → diff −0.012.
**PROCESSED:** `{sector_etf:"XLP", relation:"lagging"}` → `industry: sector XLP lagging (rel -0.01212)`

---

## 6. FUNDAMENTALS

**How we process it.** Quarterly Compustat fields are turned into standard ratios deterministically (book-to-market,
ROE, ROA, gross margin, debt-to-equity), then an LLM interprets them into **valuation / profitability / leverage / trend**
views with a rationale. Point-in-time: dated by the Compustat announcement date (`rdq`), available next trading day.

**Data path.** Raw: Compustat `fundq` (WRDS) → ratios → `build_fundq_interpret.py` (LLM) →
cache `compustat_fundamentals/qwen_fundamentals_cache.jsonl` → **Processed:** `<TK>_fundamentals_2023_2025.json` (`meta.valuation, meta.trend`).

### HIGH: TSLA (FY-quarter)
**RAW** (Compustat ratios, point-in-time): high P/E, negative ROE/ROA, rising book-to-market, increasing D/E.
**PROCESSED** (LLM analysis): `{valuation:"OVERVALUED", profitability:"WEAK", leverage:"MODERATE", trend:"DETERIORATING",
key_concerns:["High P/E","Negative ROE/ROA"]}` → `FUND: valuation OVERVALUED trend DETERIORATING`

### LOW: WDFC, 2023-10-23
**RAW** (Compustat ratios): improving book-to-market, decreasing D/E, stable margins.
**PROCESSED:** `{valuation:"OVERVALUED", trend:"IMPROVING"}` → `FUND (age 1/21d): valuation OVERVALUED trend IMPROVING`
*(Compustat coverage is uniform — every firm has quarterly fundamentals; raw numerics come from the WRDS feed, not shipped in the per-ticker folder.)*

---

## 7. ANALYST EXPECTATIONS — SUE (Standardized Unexpected Earnings)

**How we process it.** From I/B/E/S we take reported EPS vs the consensus mean forecast, scale by the estimate
dispersion to a **SUE z-score**, and label **beat/miss** with magnitude — the classic post-earnings-announcement-drift signal.

**Data path.** Raw: I/B/E/S surprise history (`IBES/surprise_history.csv`, WRDS) → `_audit_ibes.py` →
**Processed:** `<TK>_sue_2023_2025.json` (`raw_signal, direction, meta.actual, meta.consensus`).

### HIGH: TSLA, 2023-01-25
**RAW:** `actual EPS 1.19 vs consensus 1.127`.
**PROCESSED:** `{raw_signal:0.561, direction:+1 (BEAT)}` → `SUE: Earnings BEAT: actual EPS $1.19 vs consensus $1.127 (+5.5%)`

### LOW: WDFC, 2023-10-23
**RAW:** `actual EPS 1.21 vs consensus 1.2055`.
**PROCESSED:** `{direction:+1, z:+0.20}` → `SUE (age 1/3d): Earnings BEAT: actual EPS $1.21 vs consensus $1.2055 (+0.4%); SUE z +0.20`
*(SUE coverage is uniform — every firm reports earnings.)*

---

## 8. 8-K FILINGS (material events)

**How we process it.** Each EDGAR 8-K is parsed per item (e.g., 2.02 earnings, 5.02 governance, 1.01 agreements); an LLM
extracts **sentiment, tone, event category, market-impact, key themes, and a summary** per item and overall. Event-driven,
carried forward over a short tier-conditioned window (~5 days).

**Data path.** Raw: SEC EDGAR 8-K text (`edgar/`, `dataset_2022_2025/_quality/filing_text_cache/<accession>.txt`) →
`filing_extract_qwen.py` → **Processed:** `<TK>_filings_2023_2025.json` (`filing_type:"8K", items_found, overall{...}`).

### HIGH: NVDA, 2023-02-22 (Item 2.02 earnings)
**RAW** (EDGAR filing text): full 8-K exhibit — quarterly results of operations, revenue/EPS tables, management commentary.
**PROCESSED:** `{items:["2.02"], sentiment:0.5, tone:"optimistic", event_category:"management", market_impact:"positive",
summary:"NVIDIA highlights strong AI momentum and strategic partnerships"}` → `8K: sentiment 0.5 tone optimistic impact positive`

### LOW: WDFC, 2023-10 (Item 2.02 earnings 8-K)
**RAW** (EDGAR text): `…false 12-31 2022 Q… us-gaap:CommonStockMember … Results of Operations … ` (XBRL-tagged 8-K exhibit).
**PROCESSED:** `{filing_type:"8K", sentiment:0.7, tone:"optimistic", market_impact:"positive"}` → `8K (age 2/5d): sentiment 0.7 tone optimistic impact positive`
*(8-K frequency is ~uniform per firm across tiers — mandatory disclosure does not scale with coverage.)*

---

## 9. 10-Q FILINGS (quarterly report)

**How we process it.** The 10-Q Management's Discussion & Analysis and Risk Factors are LLM-summarized into **disclosure
tone, themes, and market-impact**; carried forward over a ~quarter-relevant window.

**Data path.** Raw: EDGAR 10-Q text (`filing_text_cache/<accession>.txt`) → `filing_extract_qwen.py` →
**Processed:** `<TK>_filings_2023_2025.json` (`filing_type:"10Q", overall{tone, summary}`).

### HIGH example & LOW example
- **HIGH (e.g., AAPL/TSLA 10-Q):** RAW = MD&A + condensed financials text → PROCESSED `{tone, market_impact, summary}`.
- **LOW: JJSF, 2023-05-04** — RAW: 10-Q MD&A/financial-statements text (XBRL-tagged) →
  **PROCESSED:** `{filing_type:"10Q", sentiment:0.0, tone:"neutral", market_impact:"neutral", summary:"The company reports stable operations with no new material risks identified."}` → `10Q: sentiment 0.0 tone neutral`

---

## 10. 10-K FILINGS (annual report)

**How we process it.** The 10-K (business, risk factors, MD&A) is LLM-summarized into **disclosure tone + themes**; the
longest-horizon, lowest-frequency source. Same extraction as 8-K/10-Q, annual cadence.

**Data path.** Raw: EDGAR 10-K text (`filing_text_cache/<accession>.txt`) → `filing_extract_qwen.py` →
**Processed:** `<TK>_filings_2023_2025.json` (`filing_type:"10K", overall{tone, summary}`).

### HIGH example & LOW example
- **HIGH (e.g., NVDA/AMZN 10-K):** RAW = full annual report sections → PROCESSED `{tone, summary}`.
- **LOW: WDFC, 2023-10-23** — RAW: annual 10-K (business + risk factors + MD&A) →
  **PROCESSED:** `{filing_type:"10K", sentiment:0.133, tone:"cautious", summary:"Optimism about product diversification, global
  expansion, digital transformation, but significant macroeconomic & supply-chain risks."}` → `10K (age 0/21d): sentiment 0.13 tone cautious`
*(10-K/10-Q availability is uniform across tiers; only news & social show the steep coverage gradient.)*

---

## Coverage summary across sources
| Source | Coverage gradient (HIGH vs LOW) | Raw → Processed transform |
|---|---|---|
| News | **Steep** (110 vs 7 headlines/day) | abstractive LLM brief + Net Tilt |
| Social | **Absent for LOW** (21,491 vs 16 posts) | per-post sentiment → daily aggregate / SILENT |
| Technical | uniform | OHLCV → indicator states |
| Market | identical (shared) | VIX/SPY/FF → regime label |
| Industry | uniform | sector vs market → leading/lagging |
| Fundamentals | uniform | Compustat ratios → valuation/trend view |
| SUE | uniform | EPS vs consensus → beat/miss + z |
| 8-K / 10-Q / 10-K | **~uniform per firm** | EDGAR text → sentiment/tone/themes/summary |

**Takeaway:** the *same* LLM-processing turns each raw feed into a compact, decision-ready state line; the **information
environment differs by coverage** only for news and social (rich for mega-caps, sparse/absent for under-covered), while
filings, fundamentals, SUE, technical, market, and industry are available roughly uniformly — the design premise of the study.
