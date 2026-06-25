# Verbal-RL (Reflection-Driven) Agent — exact result, script, output
*The "Reflection-Driven (Tian et al., 2025)" column. Self-contained provenance: data → script → output → scored result.
2025 OOS, 249 days × 16 firms, market-adjusted vs SPY. Verified 2026-06-25.*

## System
gpt-4o-mini **is the policy** (in-context / verbal RL, no gradient updates). Each trading day it sees the 10-source signal
string for a firm **plus a rolling log of its own recent `(signals → action → realized net reward)`** and picks a position
to maximize cumulative net-of-cost reward — learning online from its reward history (Reflexion-style). Leakage-safe: the
reward buffer on day *d* holds only outcomes realized strictly before *d*.

## Provenance (data → script → output → result)
| Stage | File |
|---|---|
| **Producer script** | `rl_gpt_agent.py` (policy = gpt-4o-mini; `RL_TIER` ∈ {high,mid,low}; needs `.env` OPENAI key) |
| **Raw output (decisions)** | `dataset_2022_2025/_quality/rlagent_decisions_2025.json` (high), `..._2025_mid.json`, `..._2025_low.json` |
| **Scorer (CR/Sharpe/DA/Acted)** | `_audit_indep_mr.py` → `_audit_indep_mr_out.json`; `score_sota_table.py` → `sota_table_out.json` |
| **Independent auditor** | `eval_rl.py` |

**Reproduce:**
```bash
cd /home/mr1639@students.ad.unt.edu/workspace/corporate_fake_news
RL_TIER=high .venv/bin/python3 rl_gpt_agent.py     # also RL_TIER=mid, RL_TIER=low   (writes rlagent_decisions_2025*.json)
.venv/bin/python3 _audit_indep_mr.py               # scores -> _audit_indep_mr_out.json  (Verbal-RL column)
```

## Exact result — per company (CR / Sharpe / Dir.Acc / Acted), 2025
| Ticker | CR% | Sharpe | Dir.Acc% | Acted% |
|---|--:|--:|--:|--:|
| AAPL | −10.3 | −2.10 | 31.6 | 7.6 |
| TSLA | **+36.1** | +1.37 | 51.9 | 20.9 |
| MSFT | +2.1 | +0.40 | 44.7 | 15.3 |
| AMZN | −7.8 | −0.79 | 46.3 | 21.7 |
| GOOGL | −14.8 | −1.62 | 34.4 | 12.9 |
| NVDA | **+26.1** | +1.34 | 54.7 | 25.7 |
| NBTB | −6.3 | −0.57 | 46.3 | 16.5 |
| ANDE | +5.8 | +0.37 | 52.4 | 33.7 |
| PRGS | −5.8 | −0.10 | 47.8 | 54.6 |
| MYRG | −4.3 | −0.21 | 45.5 | 13.3 |
| PAHC | +15.8 | +0.82 | 54.5 | 26.5 |
| IIIN | −20.4 | −1.70 | 50.0 | 16.1 |
| WDFC | −23.0 | −1.58 | 43.7 | 28.5 |
| JJSF | −1.3 | +0.04 | 51.1 | 36.9 |
| YORW | −10.8 | −1.14 | 43.8 | 12.9 |
| MSEX | −19.8 | −0.56 | 47.5 | 39.8 |
| **EW HIGH (6)** | **+5.2** | **−0.20** | **43.9** | **17.3** |
| **EW UNDER (10)** | **−7.0** | **−0.50** | **48.3** | **27.9** |

*(Matches paper Table §5: AAPL −10.3 / TSLA +36.1 / NVDA +26.1; EW HIGH +5.2 / UNDER −7.0. CR = cumulative market-adjusted return.)*

## Behaviour
Decision mix (all 16): **HOLD 3,031 / SELL 562 / BUY 391** → **≈76% HOLD** (highly selective; mean Acted ≈ 17% HIGH / 28% UNDER).

## Honest read
- The HIGH-tier EW (+5.2) is driven by just **two high-conviction names, TSLA +36.1 and NVDA +26.1**, traded on only ~21–26% of days; the other four high-tier names lose or break even. So it does **not** beat Buy&Hold (+8.9) at the tier level — selectivity captures occasional wins, not systematic alpha.
- Under-covered EW −7.0 (negative); directional accuracy ≈ chance (44–48%).
- Verdict: in-context reward adaptation finds **occasional high-conviction opportunities but no systematic edge**.
