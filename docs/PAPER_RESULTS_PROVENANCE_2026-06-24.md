# Paper Results → Source-File Provenance (verified 2026-06-24)

Maps every number in the **Results** sections of `Rashid_IPM_paper_send_to_Dr.Hong_v2.pdf`
(§5 High/Under-covered tiers, §6 Ablation, §7 Error Analysis) to the file that produces it,
the generating script, and its raw inputs. Scope: 2025 OOS, 16 tickers, market-adjusted vs SPY,
net cost 2.5 bps/side HIGH and 10 bps/side under-covered (B&H pays one entry).

Legend: ✓ = I re-derived the number from the raw file this session. ◎ = verified by a sub-agent. ⚠ = discrepancy, see notes.

---

## §5.1 + §5.2 — Per-ticker & EW tables (all systems EXCEPT FinMem)

**Source of record:** `_audit_indep_mr_out.json`  ·  **Script:** `_audit_indep_mr.py`

Reproduces the paper **exactly** for all 8 non-FinMem systems, per-ticker and EW (✓ I confirmed):
- EW HIGH: B&H +8.9, Judge-4o −4.6, Judge-nano −4.9, Verbal-RL +5.2, GRPO −4.6, SFT→GRPO −9.2, CLF16 +2.3, GRPO-shuf −8.1
- EW UNDER: B&H −9.2, Judge-4o −4.4, Judge-nano +17.9, Verbal-RL −7.0, GRPO +1.3, SFT→GRPO −1.9, CLF16 +6.7, GRPO-shuf +1.1
- Spot per-ticker: AAPL Judge-4o +26.2, TSLA Verbal-RL +36.1, MSEX Judge-4o +67.7, PAHC Judge-nano +112.0, IIIN Judge-nano +81.4 — all match.

**System → raw input file** (read by `_audit_indep_mr.py`, all under `dataset_2022_2025/_quality/` unless noted):
| Paper column | Raw input | Notes |
|---|---|---|
| Buy & Hold | `judge_decisions_gpt-4o-mini_2025{,_mid,_low}.json` (field `r`, pos=+1) | one entry cost |
| Judge-4o | `judge_decisions_gpt-4o-mini_2025{,_mid,_low}.json` (`action`) | FinJudge / GPT-4o-mini |
| Judge-nano | `judge_decisions_gpt-5-nano_2025{,_mid,_low}.json` | FinJudge / GPT-5-nano |
| Verbal-RL | `rlagent_decisions_2025{,_mid,_low}.json` | |
| CLF16 | `clf16_actions_clf16.json` | 3-class classifier |
| SFT | `clf16_actions_sft3_high.json` + `clf16_actions_sft3_midlow.json` | (standalone SFT; not a paper column) |
| GRPO | `clf16_actions_real_h3full.json` + `clf16_actions_real_midlow.json` | |
| SFT→GRPO | `clf16_actions_real_sft3_high.json` + `clf16_actions_real_sft3_midlow.json` | |
| GRPO-shuffled | `rlvr_full14b_actions_base_real_shuf.json` (high) + `clf16_actions_shuffled_midlow.json` | placebo |
| FinMem | `finmem_sota/results/finmem_{C_ctrl_full,mid_2025,low_2025}_daily.csv` | ⚠ see below — scoring convention differs from canonical faithful |

---

## ⚠ FinMem column — the one discrepancy

The paper's FinMem column does **NOT** match `_audit_indep_mr_out.json` (which has AAPL +35.98 / TSLA −66.33 because
`_audit_indep_mr.py` fails to filter `mode=="adaptive"` — it mixes all three risk modes; **known-wrong for FinMem**).

The paper's FinMem numbers match a **recompute of the faithful adaptive CSV**:
- **File:** `_adv_audit_perticker_recompute_out.json` (≡ `_adv_audit_mr_recompute_out.json`) ◎ · **Script:** `_adv_audit_perticker_recompute.py` (filters `mode=="adaptive"`).
- Paper EW HIGH **−6.5**, UNDER **−6.8**; per-ticker AAPL +28.9, **TSLA −13.8**, MSFT −7.7, AMZN −6.9, GOOGL −40.6, NVDA +0.9; NBTB +7.6, ANDE −38.6, PRGS +7.4, MYRG −46.8, PAHC −19.5, IIIN −40.6, WDFC +16.3, JJSF +8.9, YORW −0.9, MSEX +38.0.

But the project's **CANONICAL FAITHFUL** FinMem (the single source of truth) is different:
- **File:** `HANDOFF_2026-06-16/results/CLF16_VS_BASELINES.md` (verified 2026-06-22, two code paths agree).
- **Faithful EW HIGH −4.8, UNDER −5.5**; per-ticker AAPL +28.7, **TSLA −3.6**, MSFT −7.8, AMZN −7.8, GOOGL −39.2, NVDA +1.0; NBTB +9.7, ANDE −36.2, PRGS +6.6, MYRG −43.6, PAHC −19.1, IIIN −37.5, WDFC +16.9, JJSF +10.3, YORW −0.2, MSEX +38.5.

**Both come from the SAME faithful agent and the SAME CSV** (`finmem_C_ctrl_full_daily.csv`, mode=adaptive, discrete `direction`) — **identical real positions.** The gap is purely the **return-scoring convention** (✓ I re-derived both from the CSV, row by row):
| TSLA | Return convention | EW HIGH | Same basis as rest of paper? |
|---|---|---|---|
| **−13.8** | auditor recompute `pos·[(e^lr−1)−(e^spy−1)]` = **simple** market-adj → **the paper** | **−6.5** | **YES** |
| **−3.6** | agent's own `r_adj` column = **log-excess** `pos·(lr−spy)` → CLF16_VS_BASELINES.md | **−4.8** | no (log vs simple) |

**RESOLUTION — the paper's −6.5/−6.8 is the table-correct number.** ✓ I confirmed the **shared `r` grid** that scores every
other system (Judge-4o/nano, CLF16, GRPO, SFT→GRPO, GRPO-shuf, B&H) uses the **simple** convention `(e^lr−1)−(e^spy−1)`
— matches **1454/1494** (ticker,date) cells. The paper's FinMem recompute scores FinMem's real positions on that *same*
simple basis → apples-to-apples. The −4.8/−5.5 "faithful" figure uses the agent's internal **log-excess** `r_adj` (`lr−spy`,
120/137 rows) — faithful to the agent's own bookkeeping but a *different* return metric, and `(1+log-excess)` compounding is
a non-standard hybrid. **Do NOT swap −4.8/−5.5 into the comparison table** — it would mix a log-excess number into a
simple-return table. Both are from the real run; only −6.5/−6.8 is convention-consistent with the table.

**Faithful FinMem implementation (the agent code):** `finmem_sota/code/finmem_faithful.py`
(4 memory stores = news/10-Q/10-K/reflection; sampled importance; discrete action {+1,0,−1}; risk posture = sign of
cumulative return, **no conviction sizing**; mode=adaptive, no best-of-3). Verified line-by-line vs the official repo in
`finmem_sota/spec/FAITHFULNESS_vs_official_repo.md`. The OLD unfaithful code is `finmem.py`/`finmem2.py`
(`finmem_full_all_daily.csv`, ±0.85 continuous conviction) — flagged "do not use" in `finmem_sota/FinMem_SOTA_Findings.md`.

> **NET FOR THE PAPER:** FinMem column (−6.5 / −6.8) is **correct and table-consistent** — keep it. Optionally footnote that
> FinMem's positions are scored on the same simple market-adjusted basis as all other systems. The project's −4.8/−5.5
> "faithful" figure is the agent's own log-excess bookkeeping and is not directly comparable in the table.

---

## §6 — Ablation tables

### Table "Judge market-specialist placebo decomposition (mega-cap)"
- **Docs:** `PLACEBO_AND_SIGNIFICANCE.md` §3, `HANDOFF_2026-06-16/results/COMPLETE_ABLATION_TWO_SEGMENT.md` §1a.
- **Raw inputs:** `judge_decisions_gpt-4o-mini_2025.json` (FULL), `..._high_drop_market.json` (drop), `..._high_shuf_market.json` (shuffle), `..._high_placebo{1,2,3}.json`.
- **Scripts:** `judge_llm_system.py` (drop/shuffle/placebo modes) → `audit_judge_ablation.py` / `build_agent_ablation_tables.py`.
- **Verified ◎:** FULL −2.37, Drop −18.90, Shuffle −11.09 → occlusion **+16.5**, genuine **+8.7 (p=.237)**, footprint **+7.8**; market 6/6 positive → binomial **p=.031**. ⚠ the .031 is computed only — **not written in any committed doc** (add it to `PLACEBO_AND_SIGNIFICANCE.md`).

### Table "Under-covered source-occlusion across CLF16 / GRPO / Verbal-RL"
- **Doc (canonical):** `HANDOFF_2026-06-16/results/COMPLETE_ABLATION_TWO_SEGMENT.md` §2a — all 30 cells match ◎.
- **Raw inputs:** `clf16_actions_real_midlow.json` (+ `..._drop_<source>.json`); Verbal-RL `rlagent_decisions_2025_{mid,low}.json` (+ drops).
- **Scripts:** `rlvr_eval_clf16.py` (CLF_DROP_SRC, for CLF16 & GRPO), `rl_gpt_agent.py` (RL_DROP_SRC); scoring `_audit_2seg_recompute.py` / `run_midlow_accounting.py`.
- ⚠ Sub-agent could not independently re-derive the MID magnitudes (e.g. CLF16 news +6.5, market −15.1) with a quick recompute — convention detail in the locked pipeline; values are auditor-claimed and consistent across committed docs. **Supersedes** the old 3-tier `VERBALRL_GRPO_MIDLOW_OCCLUSION.md` (do not source from it).

### Table "Selected mega-cap sources per system" (Occlusion / Genuine / Footprint / p)
- **Docs:** `HANDOFF_2026-06-16/results/RL_CLF_SOURCE_OCCLUSION_ABLATION.md` §3, `COMPLETE_ABLATION_TWO_SEGMENT.md` §1b — all cells match ◎.
- **Raw inputs:** `clf16_actions_clf16.json` (+ `..._drop_<src>.json`, `..._shuf_{market,industry,social}.json`); GRPO `clf16_actions_real_h3full.json` (+ drops/shuffles, multi-seed `_s1/_s2/_s3`); Verbal-RL `rlagent_decisions_2025.json`.
- **Script:** `rlvr_eval_clf16.py` (CLF_DROP_SRC / CLF_SHUF_SRC / CLF_SHUF_SEED), `rl_gpt_agent.py`.
- **Verified ◎:** CLF16 market occlusion +9.9, industry +12.3, social +11.0 (exact). p-values: GRPO industry +13.0 genuine, **p=.014\*** is the only nominally-significant cell.

---

## §7 — Error Analysis

### Table "Tier-wise error diagnostics with position decomposition"
- **Script:** `detailed_errors_tier.py` (emits 3 tiers; paper pools **mid+low → "Under-covered"**). Render: `ERROR_ANALYSIS_4SYS_TIER.md`. Aggregate auditor: `audit_error_diag.py` → `error_diagnostics.json`; `master_error_tier.py` → `error_diagnostics_tier.json`.
- **Raw inputs:** `judge_decisions_gpt-4o-mini_2025{,_mid,_low}.json`, `judge_decisions_gpt-5-nano_2025{,_mid,_low}.json`, `rlagent_decisions_2025{,_mid,_low}.json`, `clf16_actions_clf16.json`.
- **Verified ◎:** all HOLD%, directional bias, Long/Short n, Long/Short P&L%, hit% cells reproduce (Judge-4o mega HOLD 24/bias +0.35/Long 766/+32.3; CLF16 under Short 1971/+116.5; etc).
- ⚠ Two sub-1pp roundings (Verbal-RL under Long hit 45.6→45; CLF16 under HOLD 7.4→8). ⚠ **Worst-stock magnitudes** in the paper (NVDA −29.6, PRGS −53.7, GOOGL −14.8, WDFC −23, TSLA −18.7, IIIN −47.2) use a **compounded/segment** convention; the script reports additive `Σ pos·r` (NVDA −27, PRGS −65…). **Stock identities all match**; only magnitudes differ — reconcile the convention or footnote it.

### Table XII "Coverage-graded error modes" — verbatim FinJudge rationales
- **Source:** `reason` field of `judge_decisions_gpt-4o-mini_2025{,_mid,_low}.json`. All three confirmed verbatim ◎:
  - **NVDA 2025-01-24** BUY, r −15.6% — "Fresh bullish signals outweigh stale industry concerns…" (`_2025.json`)
  - **IIIN 2025-10-15** BUY, r −18.6% — "Strong revenue growth, bullish market conditions…" (`_2025_mid.json`)
  - **MSEX 2025-10-31** BUY, r −13.3% — "Bull market momentum outweighs bearish technical and industry signals…" (`_2025_low.json`)
- Render: `QUALITATIVE_ERROR_STUDY_TIER.md`, `ERROR_ANALYSIS_DETAILED.md`.

---

## Open items to fix before submission
1. **FinMem convention — RESOLVED:** paper's −6.5/−6.8 is correct (simple market-adj, same basis as all other systems — verified shared grid = simple on 1454/1494 cells). Keep it; optionally footnote the basis. Do not substitute the log-excess −4.8/−5.5.
2. **Binomial p=.031** (market 6/6) — correct but uncommitted; add to a result doc.
3. **§7 worst-stock magnitudes** — additive vs compounded convention mismatch with the §5 per-ticker tables.
4. **Table 2 (under-covered occlusion)** — auditor-claimed; MID magnitudes not independently re-derived this session.
