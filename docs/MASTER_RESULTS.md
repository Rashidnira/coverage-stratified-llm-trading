# MASTER RESULTS — single retrieval point
*Durable index of every result table + where it lives. Start here. Last updated 2026-06-24.*
Repo root: `/home/mr1639@students.ad.unt.edu/workspace/corporate_fake_news/`. Convention (locked, all tables): per-ticker
compound `(1 + pos·r − cost·|pos−prev|)`, `pos`=unit-sign(action), `r`=next-day **simple** market-adjusted return,
**cost 2.5 bps/side HIGH, 10 bps/side under-covered**, B&H one entry. 2025 OOS, 249 days × 16 tickers.

---

## 1. Supervised classifiers — per-company NET %, all 16 (embedded for instant retrieval)
Full doc: [`CLASSIFIER_RESULTS_16CO_2026-06-24.md`](CLASSIFIER_RESULTS_16CO_2026-06-24.md) · scorer `score_classifiers_perticker.py` → `classifier_perticker_out.json`. ✓ CLF16 reproduces paper §5 exactly.

| Ticker | B&H | CLF16 | CLF16-v2 | CLF16-anon | Mistral-mm | Qwen-mm | Mistral-NST |
|---|--:|--:|--:|--:|--:|--:|--:|
| AAPL | −3.6 | +0.0 | −0.5 | −11.6 | +0.0 | −7.5 | −4.4 |
| TSLA | +6.2 | −18.7 | −21.9 | −18.7 | −30.1 | −32.4 | −62.1 |
| MSFT | −1.4 | +0.8 | −0.0 | −1.4 | +0.0 | −2.6 | −0.4 |
| AMZN | −9.1 | −5.9 | +4.3 | −19.1 | +17.1 | +12.0 | −13.2 |
| GOOGL | +41.8 | +19.6 | −0.5 | +18.7 | −13.6 | +5.4 | −2.5 |
| NVDA | +19.4 | +18.0 | −9.1 | −0.6 | +19.4 | +10.4 | +16.7 |
| NBTB | −25.3 | +15.6 | +1.3 | +9.6 | +20.0 | +9.3 | +35.6 |
| ANDE | +12.5 | +31.0 | +0.0 | +44.0 | −17.1 | +147.6 | −22.3 |
| PRGS | −43.8 | +12.1 | +0.0 | +15.1 | +57.7 | +15.7 | +64.3 |
| MYRG | +28.4 | +32.9 | +4.0 | −2.9 | −33.9 | +24.1 | −41.5 |
| PAHC | +55.2 | −13.3 | +0.0 | +1.4 | −49.1 | +20.8 | −12.3 |
| IIIN | +2.8 | −47.2 | +0.0 | −35.9 | −15.4 | −34.0 | −15.7 |
| WDFC | −32.0 | −8.6 | +0.2 | +21.9 | +34.8 | +3.7 | −5.8 |
| JJSF | −51.3 | +30.5 | −2.1 | −2.8 | +57.1 | −0.4 | +70.9 |
| YORW | −19.5 | +12.8 | +2.0 | +37.8 | +10.4 | +8.9 | +5.0 |
| MSEX | −19.5 | +1.5 | −4.7 | +17.1 | +4.2 | −21.4 | +38.5 |
| **EW HIGH** | **+8.9** | **+2.3** | **−4.6** | **−5.5** | **−1.2** | **−2.4** | **−11.0** |
| **EW UNDER** | **−9.2** | **+6.7** | **+0.1** | **+10.5** | **+6.9** | **+17.4** | **+11.7** |

Classifier key: CLF16 = Qwen-14B full-10-source (paper). CLF16-v2 = cost-aware labels + abstention (~95% HOLD). CLF16-anon = ticker-identity removed.
Mistral-mm / Qwen-mm = full-10-source, two backbones. Mistral-NST = 3-source (news+social+technical). ⚠ **Qwen-NST not run** (missing 4th cell).
Honest read: HIGH — none beats B&H +8.9 (CLF16 +2.3 best active). UNDER positives are 77–97% SELL beta (always-SELL ≈ +7.9; n.s. p≥0.38), not skill.

---

## 2. Paper §5 main systems — EW % by tier (FinJudge / FinMem / Verbal-RL / gradient models)
Source: `_audit_indep_mr_out.json` (all but FinMem); FinMem `_adv_audit_perticker_recompute_out.json`. Per-ticker detail: [`PAPER_RESULTS_PROVENANCE_2026-06-24.md`](PAPER_RESULTS_PROVENANCE_2026-06-24.md).

| Tier | B&H | FinMem | Judge-4o | Judge-nano | Verbal-RL | GRPO | SFT→GRPO | CLF16 | GRPO-shuf |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| **HIGH** | +8.9 | −6.5 | −4.6 | −4.9 | +5.2 | −4.6 | −9.2 | +2.3 | −8.1 |
| **UNDER** | −9.2 | −6.8 | −4.4 | +17.9 | −7.0 | +1.3 | −1.9 | +6.7 | +1.1 |

System taxonomy: **SOTA baseline** = FinMem · **Proposed** = FinJudge (Judge-4o/nano) · **In-context** = Verbal-RL ·
**Gradient-trained** = GRPO / SFT→GRPO / CLF16 · **Placebo control** = GRPO-shuffled (RL with permuted reward; ≈ GRPO-real ⇒ no genuine learned signal, RQ3).

---

## 3. Other verified result docs (where to retrieve each)
| Result | Doc |
|---|---|
| **SOTA-taxonomy systems, per-company (16) — CR/Sharpe/DA/Acted** | [`SOTA_RESULTS_16CO_2026-06-24.md`](SOTA_RESULTS_16CO_2026-06-24.md) |
| **Verbal-RL (Reflection-Driven) — exact result + script + output** | [`VERBAL_RL_RESULT_2026-06-25.md`](VERBAL_RL_RESULT_2026-06-25.md) |
| **Classifiers, per-company (16)** | [`CLASSIFIER_RESULTS_16CO_2026-06-24.md`](CLASSIFIER_RESULTS_16CO_2026-06-24.md) |
| **Paper §5/§6/§7 number→source-file map** (+ FinMem convention) | [`PAPER_RESULTS_PROVENANCE_2026-06-24.md`](PAPER_RESULTS_PROVENANCE_2026-06-24.md) |
| **Full reproduction manifest** (§3 data → §4 systems → §5/6/7) | [`IPM_PAPER1_REPRODUCE.md`](IPM_PAPER1_REPRODUCE.md) |
| **Faithful FinMem (two-segment NET)** | `HANDOFF_2026-06-16/results/CLF16_VS_BASELINES.md` |
| **Source-occlusion ablations (RQ2)** | `HANDOFF_2026-06-16/results/COMPLETE_ABLATION_TWO_SEGMENT.md`, `RL_CLF_SOURCE_OCCLUSION_ABLATION.md`, `PLACEBO_AND_SIGNIFICANCE.md` |
| **Error analysis (RQ3)** | `ERROR_ANALYSIS_4SYS_TIER.md`, `QUALITATIVE_ERROR_STUDY_TIER.md` |
| **TSLA news-bias diagnosis + sentiment-gate** | `MD_files_2026-06-24/SESSION_LOG_2026-06-24_TSLA_news_bias.md` |
| **Study-wide system catalog / folder map** | `HANDOFF_2026-06-16/MASTER_REPRODUCIBILITY.md` |
| **Mistral-NST training run** | `MD_files_2026-06-24/MISTRAL7B_NST_LAST_TRAINING_RESULT_2026-06-24_night.md` |
| **Dated folder (all 2026-06-24 md)** | [`MD_files_2026-06-24/INDEX.md`](MD_files_2026-06-24/INDEX.md) |

---
*To refresh §1: `.venv/bin/python3 score_classifiers_perticker.py`. To refresh §2: `_audit_indep_mr.py` + `_adv_audit_perticker_recompute.py`.*
