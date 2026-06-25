# Supervised-Classifier Results — all 16 companies (2026-06-24)

Per-company (16-ticker) NET% for every supervised 3-class (BUY/HOLD/SELL) trading classifier, scored on the
**paper's locked two-segment convention**: per-ticker compound `(1 + pos·r − cost·|pos−prev|)`, `pos`=unit-sign(action),
`r`=next-day simple market-adjusted return, **cost 2.5 bps/side HIGH, 10 bps/side under-covered**, B&H one entry.
Scorer: `score_classifiers_perticker.py` → `classifier_perticker_out.json`. ✓ **CLF16 column reproduces the paper §5
exactly** (HIGH +2.3 / UNDER +6.7, every cell), validating the convention. Test = 2025 OOS, 249 days × 16.

## System catalog (backbone × source-set × variant)
| Label | Backbone | Sources | Train data | Train script | Eval | Output (`Q/`) |
|---|---|---|---|---|---|---|
| **CLF16** | Qwen2.5-14B | full 10-source | `supervised_labeled_2023_2024.jsonl` | `rlvr_clf_full14b.py` | `rlvr_eval_clf16.py clf16` | `clf16_actions_clf16.json` |
| **CLF16-v2** | Qwen2.5-14B | full 10-source | cost-aware + vol-scaled labels, confidence abstention | `clf16_v2_package/` (`build_labels_v2.py`) | `rlvr_eval_clf16_conf.py` | `clf16_actions_clf16_v2.json` |
| **CLF16-anon** | Qwen2.5-14B | full 10-source (**ticker identity removed**) | anonymized state | `clf16_anon` run | `rlvr_eval_clf16.py` | `clf16_actions_clf16_anon.json` |
| **Mistral-mm** | Mistral-7B-Instruct-v0.3 | full 10-source | `mm_train_2023_2024.jsonl` | `sft_mm.py` | `eval_mistral_mm.py mistral_mm` | `mistral_mm_mistral_mm.json` |
| **Qwen-mm** | Qwen2.5-14B | full 10-source | `mm_train_2023_2024.jsonl` | `sft_mm.py` (CLF_OUTDIR=qwen2_5_14b_instruct_mm) | `eval_mistral_mm.py` | `mm_eval_qwen2_5_14b_instruct_mm.json` |
| **Mistral-NST** | Mistral-7B-Instruct-v0.3 | **3-source: news + social + technical** | `nst_train_2023_2024.jsonl` (`build_nst_train.py`) | `sft_mm.py` (CLF_OUTDIR=mistral7b_nst) | `eval_mistral_mm.py mistral7b_nst` | `mistral_mm_mistral7b_nst.json` |

All: LoRA r=32, completion-only CE on the `ACTION:` line, 3 epochs, lr 1e-4, label = ±0.5% band on next-day `r_adj`, greedy decode.

## Per-company NET % (paper two-segment convention)
| Ticker | B&H | CLF16 | CLF16-v2 | CLF16-anon | Mistral-mm | Qwen-mm | Mistral-NST |
|---|---:|---:|---:|---:|---:|---:|---:|
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
| **EW HIGH (6)** | **+8.9** | **+2.3** | **−4.6** | **−5.5** | **−1.2** | **−2.4** | **−11.0** |
| **EW UNDER (10)** | **−9.2** | **+6.7** | **+0.1** | **+10.5** | **+6.9** | **+17.4** | **+11.7** |

*(B&H column = paper canonical; classifier columns = `score_classifiers_perticker.py`. CLF16 = paper §5 exactly.)*

## Classification accuracy (OOS 2025, vs ±0.5% label; aggregation-independent)
| System | HIGH acc% | HIGH wF1 | UNDER acc% | UNDER wF1 | Dir.Acc% (non-HOLD) | Behaviour |
|---|---:|---:|---:|---:|---:|---|
| CLF16 | 40.6 (pooled) | — | — | — | 54.0 | SELL-heavy; HIGH collapses to per-ticker constant labels |
| CLF16-v2 | 28.4 (pooled) | — | — | — | 53.6 | ~95% HOLD (abstention) → near-flat |
| CLF16-anon | 40.8 (pooled) | — | — | — | 54.3 | SELL-heavy; identity removed ⇒ similar to CLF16 |
| Mistral-mm | 42.3 | 41.9 | 39.4 | 23.7 | 51.7 | HIGH beats base wF1 31.6→41.9; under = 97% SELL |
| Qwen-mm | 40.8 | 40.4 | 39.4 | 33.0 | — | under = 77% SELL |
| Mistral-NST | 41.2 | 40.7 | 40.0 | 26.9 | — | under = 93% SELL |

## Honest reading (factual)
- **HIGH tier:** no classifier beats Buy & Hold (+8.9). CLF16 (+2.3) is the best active; the mm/NST classifiers are
  near-zero-to-negative. TSLA is every classifier's worst HIGH name (net-short into a rising stock; −19 to −62).
- **Under-covered tier:** classifiers look strong (CLF16 +6.7, Qwen-mm +17.4, CLF16-anon +10.5, NST +11.7) but the
  mix shows **77–97% SELL** — this is net-short beta on a falling segment (B&H −9.2), not stock selection. The naive
  always-SELL baseline already earns ≈+7.9 there. Mistral-mm's own report: paired bootstrap vs honest baselines
  **n.s. (all p ≥ 0.38)**.
- **CLF16-v2** (cost-aware labels + abstention) trades almost never (~95% HOLD) → collapses to ≈flat (HIGH −4.6, UNDER +0.1).
- **CLF16-anon** (ticker identity stripped) ≈ CLF16 — performance does not depend on the model knowing the ticker name.
- **Per-company scatter is enormous and system-specific** (e.g. ANDE: Qwen-mm +147.6 vs Mistral-mm −17.1; TSLA uniformly bad).

## ⚠ Missing system — Qwen-NST not run
There is a **Mistral-7B NST** (news+social+technical) classifier but **no Qwen-2.5-14B NST** counterpart
(no `*qwen*nst*` output exists). To complete the 2×2 (backbone × {full, NST}):
```bash
# build already exists: nst_{train,test}.jsonl
CLF_OUTDIR=qwen14b_nst MODEL_ID=Qwen/Qwen2.5-14B-Instruct \
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 sft_mm.py              # train
EVAL_TESTFILE=nst_test_2025.jsonl CUDA_VISIBLE_DEVICES=0 \
  .venv/bin/python3 eval_mistral_mm.py qwen14b_nst                # eval
```
Say the word and I'll run it and add the column.

## Provenance / reproduce
- Per-ticker scorer: `score_classifiers_perticker.py` → `classifier_perticker_out.json`.
- Source reports: `mistral_classifier/RESULTS.md` (Mistral-mm), `logs/qwen14b_mm_eval.log` (Qwen-mm),
  `MD_files_2026-06-24/MISTRAL7B_NST_LAST_TRAINING_RESULT_2026-06-24_night.md` (Mistral-NST),
  `logs/clf16{,v2,_anon}_eval.log` (CLF16 family). CLF16 paper numbers also in `_audit_indep_mr_out.json`.
- ⚠ Source-doc EW figures (e.g. Mistral-mm RESULTS.md HIGH +0.9 / UNDER +6.0) use **daily-EW-portfolio** aggregation;
  the table above uses the paper's **per-ticker-compound-then-EW** (so EW differs slightly but is paper-consistent — CLF16 matches §5 exactly).
