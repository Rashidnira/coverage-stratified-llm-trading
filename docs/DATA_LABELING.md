# Data Labeling — Supervised 3-Class Trading Classifier

## Summary
Each observation is one **firm-trading-day**: the input is the 10-source market `state_text`; the target is a
3-class action **{BUY, HOLD, SELL}**. The label is derived **mechanically** from the realized **next-day
market-adjusted return** `r_adj` (next-session close-to-close return minus SPY's), using a symmetric **±0.50%
dead-band**. This turns the task into genuine supervised learning (state → known correct action) and is leakage-safe:
the return used to label never appears in the input text.

## Labeling rule (exact)
For firm *i* on day *t*, let `r_adj(i,t)` = market-adjusted return realized over the **next** trading session:

```
label =  BUY   if  r_adj > +BAND
         SELL  if  r_adj < −BAND
         HOLD  if  |r_adj| ≤ BAND
BAND = 0.005   (a 0.50% dead-band, set by the researcher)
```

- The **dead-band** assigns HOLD to days whose move is within ±0.5%, so the model is not forced to call a direction on
  economically negligible noise; only decisive up/down days become BUY/SELL.
- The day with no available forward return (last session per ticker) is dropped.

## Leakage safety (point-in-time)
- `r_adj` and **all forward-looking fields** are used **only as the label/target** — they are *never* written into the
  input `state_text`. The builder runs an explicit leak self-check that forbids `r_adj, fwd_oc_ret, open_next,
  close_next, next_session, spy_fwd_oc` from appearing in any state string.
- The state is assembled from information available up to the decision day; episodic signals (8-K/10-Q/10-K/SUE/
  fundamentals) carry an age tag and decay, so nothing post-dates the label horizon.

## Train / test split (out-of-sample)
- **Train:** 2023–2024. **Test:** 2025 (post the backbone's knowledge cutoff → contamination control).
- Same rule and same `BAND` for both splits and all 16 tickers.

## Provenance (script → output)
| System | Labeling script | Output |
|---|---|---|
| CLF16 (10-source, Qwen-14B) & Mistral/Qwen-mm | `build_supervised_labeled.py` (`BAND=0.005`, `label_of(r)`) | `dataset_2022_2025/rlvr_data/supervised_labeled_2023_2024.jsonl` (train), `supervised_labeled_2025.jsonl` (test) |
| NST (news+social+technical) | `build_nst_train.py` (`BAND`, same rule) | `dataset_2022_2025/rlvr_data/nst_{train_2023_2024,test_2025}.jsonl` |
| Per-ticker copy | (same) | `qwen_news_summaries/<TICKER>/<TICKER>_training.jsonl` (`state_text, r_adj, label, split`) |

Each record: `{ticker, date/session, split, state_text, r_adj, label}`. Verified: per-ticker `label` reproduces the
±0.5% band rule exactly (e.g., TSLA BUY 311 / SELL 334 / HOLD 106).

## Resulting label distribution (all 16 firms, 2023–2025; 12,016 instances)
| Tier | BUY | HOLD | SELL | total |
|---|--:|--:|--:|--:|
| HIGH (mega-cap) | 1,604 | 1,412 | 1,490 | 4,506 |
| Under-covered | 2,561 | 2,038 | 2,911 | 7,510 |
| **All** | **4,165** | **3,450** | **4,401** | **12,016** |

Roughly balanced 3-class (chance ≈ 33%); the under-covered tier skews slightly more SELL, consistent with its negative
2025 drift.

## Paper-ready prose (drop-in)
> We frame trading as a three-class supervised problem. For every firm-trading-day we construct a point-in-time state
> from ten information sources and assign a label from the realized next-day market-adjusted return `r_adj` (close-to-close,
> SPY-adjusted): BUY if `r_adj` exceeds +0.50%, SELL if below −0.50%, and HOLD within the ±0.50% dead-band. The dead-band
> prevents the model from labeling economically negligible moves as directional trades. Labels and returns are targets only
> and never enter the input text (leakage-checked), models are trained on 2023–2024 and evaluated on the 2025 out-of-sample,
> post-cutoff period.
