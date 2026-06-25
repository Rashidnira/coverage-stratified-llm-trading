# IPM Paper #1 — Reproduction Manifest
**"Beyond Mega-Cap Stocks: Evaluating LLM Trading Agents Across Analyst-Coverage Environments"**
*(`Rashid_IPM_paper_send_to_Dr.Hong_v2.pdf`)*

Section-by-section map of **methods → script → inputs → outputs → result numbers**, every item with a path.
Companion docs (study-wide, broader than this paper): `HANDOFF_2026-06-16/MASTER_REPRODUCIBILITY.md` (full system catalog,
ablation suite, error log, folder map), `REPRODUCE.md` (core retrain/eval), `PAPER_RESULTS_PROVENANCE_2026-06-24.md`
(number→file verification for §5/§6/§7). Verified 2026-06-24.

- **Repo root:** `/home/mr1639@students.ad.unt.edu/workspace/corporate_fake_news/` (all paths relative to it)
- **Interpreter:** `.venv/bin/python3` (torch+peft+trl+accelerate; bare `python` lacks them)
- **Output dir for decisions/metrics:** `dataset_2022_2025/_quality/` (abbrev `Q/`)
- **LoRA adapters:** `rlvr_out_full14b/`   **Secrets:** `.env` (OPENAI_API_KEY)
- **Backbones:** Qwen2.5-14B-Instruct (local, 1×H100) and GPT-4o-mini / GPT-5-nano (API)
- **Test window:** 2025 OOS, 249 trading days, market-adjusted vs SPY, post knowledge-cutoff
- ✓ = re-derived from raw files this session · ◎ = verified by sub-agent · ⚠ = caveat

---

## §3 — Sample Construction & Data Preparation (methods, with paths)

**Universe (§3.1, Table):** 16 tickers, two coverage tiers. HIGH/mega-cap (34–61 analysts): AAPL TSLA MSFT AMZN GOOGL NVDA.
Under-covered (1–7 analysts): WDFC JJSF YORW MSEX NBTB ANDE PRGS MYRG PAHC IIIN. Coverage proxy = I/B/E/S FY2025 analyst count.

**Ten information sources (§3.2, Table "Multi-source framework").** Per-ticker built signals live in
`qwen_news_summaries/<TICKER>/` (built by `qwen_news_summaries/build_sources.py`):

| § | Source | Raw feed (path) | Build script (path) | Per-ticker output (path) |
|---|---|---|---|---|
| 3.2.1 | **News** (RavenPack via WRDS, relevance=100, MiniLM near-dedup → 293,174) | `news_json/`, RavenPack export | `qwen_news_summaries/build_sources.py` (+ LLM summary) | `qwen_news_summaries/<TK>/<TK>_news_daily_summary.csv` |
| 3.2.2 | **Social** (r/wallstreetbets, Pushshift; LLM sentiment) | `reddit_json/`, `Social_WSB/` | `wsb_signal.py`, `build_social_*.py`, `reflect_wsb.py` | `qwen_news_summaries/<TK>/<TK>_social_2023_2025.json` |
| 3.2.3 | **Regulatory 8-K/10-Q/10-K** (EDGAR; Qwen per-section extract) | `edgar/` | `edgar_fetch_accept.py`, `edgar_fetch_meta.py`, `filing_extract_qwen.py`, `rebuild_filings.py` | `qwen_news_summaries/<TK>/<TK>_filings_2023_2025.json` |
| 3.2.4 | **Fundamentals** (Compustat via WRDS, rdq-aligned) | `compustat_fundamentals/` | `build_fundq_interpret.py` | `qwen_news_summaries/<TK>/<TK>_fundamentals_2023_2025.json` |
| 3.2.5 | **Analyst expectations / SUE** (I/B/E/S via WRDS) | `IBES/` | `_audit_ibes.py`, `_audit_ibes2.py` | `qwen_news_summaries/<TK>/<TK>_sue_2023_2025.json` |
| 3.2.6 | **Technical** (Yahoo OHLCV; RSI/MACD/Boll/OBV) | `prices/`, `TA/` | `build_ta_states.py` | `qwen_news_summaries/<TK>/<TK>_technical_2023_2025.json` |
| 3.2.7 | **Market** (VIX, SPY trend, Fama-French via WRDS) | `market/` | `build_daily_signal.py` | `qwen_news_summaries/<TK>/<TK>_market_2023_2025.json` |
| 3.2.8 | **Industry** (GICS + SPDR sector ETF) | `_reference/firm_sector_etf.csv`, `market/sector_etf_returns.csv` | `build_daily_signal.py` | `qwen_news_summaries/<TK>/<TK>_industry_2023_2025.json` |
| 3.2 | **Price / return grid** (OHLCV + market-adj `r_adj`) | `prices/` | `qwen_news_summaries/build_sources.py` | `qwen_news_summaries/<TK>/<TK>_price_return_2023_2025.json` |

**Temporal alignment (§3.3–3.4):** every signal assigned to the first trading session on/after dissemination; SEC filings by
EDGAR acceptance, fundamentals by Compustat `rdq`, social/news by ET timestamp. Train rules 2023–2024, evaluate 2025 only.
Event-study temporal-relevance windows: `multisource_eventstudy.py` (CARs at 1/3/5/10/21/42/63 d; Spearman IC + bootstrap).
RL/SFT/CLF training tables built leak-safe by `build_rlvr_full.py` and `build_supervised_labeled.py`
→ `dataset_2022_2025/rlvr_data/` (`full_{train,test}*.jsonl`, `supervised_labeled_{2023_2024,2025}.jsonl`, `test3_{high,midlow}.jsonl`).

---

## §4 — Systems (methods, scripts, inputs → output decision files)

All decision/action files land in `Q/` and are scored by the convention in the next section.

| Paper system | What it is (§4.x) | Script (path) | Key inputs | Output decision file (`Q/`) |
|---|---|---|---|---|
| **Buy & Hold** | always-long benchmark (§4.2.1) | computed in scorer | shared `r` grid | (metrics only) |
| **FinMem** | SOTA memory agent, adaptive mode (§4.2.2) | `finmem_sota/code/finmem_faithful.py` (+ `precompute_embeddings.py`) | built sources; `.env` | `finmem_sota/results/finmem_{C_ctrl_full,mid_2025,low_2025}_daily.csv` |
| **Judge-4o** | 10-specialist blind votes → GPT-4o-mini meta-judge + monthly trust-rule reflection (§4.2.3, "FinJudge") | `judge_llm_system.py` | specialist signals; `.env` | `judge_decisions_gpt-4o-mini_2025{,_mid,_low}.json` |
| **Judge-nano** | same, GPT-5-nano backbone | `judge_llm_system.py` (nano) | same | `judge_decisions_gpt-5-nano_2025{,_mid,_low}.json` |
| **Verbal-RL** | in-context online RL; GPT-4o-mini policy + rolling reward log, Reflexion (§4.2.4) | `rl_gpt_agent.py` (`RL_TIER`) | judge signal files; `.env` | `rlagent_decisions_2025{,_mid,_low}.json` |
| **CLF16** | Qwen-14B+LoRA supervised 3-class BUY/HOLD/SELL, all 16 (§4.2.4) | train `rlvr_clf_full14b.py`; eval `rlvr_eval_clf16.py clf16` | `supervised_labeled_*.jsonl` | `clf16_actions_clf16.json` |
| **GRPO** | Qwen-14B+LoRA GRPO, reward = pos·r_adj·100 + format (§4.2.4) | train `rlvr_train_full14b.py --train real`; eval `rlvr_eval_clf16.py real` / `real_midlow` | `full_train.jsonl`, `test3_*.jsonl` | `clf16_actions_real_h3full.json` (+ `_real_midlow.json`) |
| **SFT→GRPO** | merge SFT warm-start then GRPO (§4.2.4) | `merge_sft.py` → `rlvr_train_full14b.py --train real_sft`; eval `rlvr_eval_clf16.py` | SFT adapter + `full_train.jsonl` | `clf16_actions_real_sft3_high.json` (+ `_midlow`) |
| **GRPO-shuffled** | placebo: identical GRPO, rewards permuted (§4.2.4) | `rlvr_train_full14b.py --train shuffled` | `full_train_shuffled.jsonl` | `rlvr_full14b_actions_base_real_shuf.json` (+ `clf16_actions_shuffled_midlow.json`) |
| *(SFT standalone)* | 3-class supervised, no RL (audit column, not a paper table column) | `rlvr_sft_full14b.py` | `full_sft3_*.jsonl` | `clf16_actions_sft3_{high,midlow}.json` |

**Hyperparameters (paper Table "Consolidated settings"):** Qwen2.5-14B-Instruct + LoRA (r=32, α=64); GRPO 8 gen, β=0.04,
lr 8e−6, 400 steps; SFT/CLF completion-only CE on ACTION, 3 epochs, lr 1e−4; Verbal-RL temp 0.2, rolling 18-trade log;
Judge-4o temp 0; 1×H100. Train 2023–2024 (RL 3,012 high / 5,020 under; CLF 8,032 all-16; test 3,984).

### Reproduce decision files
```bash
cd /home/mr1639@students.ad.unt.edu/workspace/corporate_fake_news
# Judge (API): per backbone × tier
.venv/bin/python3 judge_llm_system.py run 2025 gpt-4o-mini high   # also mid, low; and gpt-5-nano
# FinMem faithful (API + embeddings): adaptive mode, all tiers
.venv/bin/python3 finmem_sota/code/precompute_embeddings.py
FM_TAG=C_ctrl_full FM_TICKERS=AAPL,TSLA,MSFT,AMZN,GOOGL,NVDA FM_WARM0=2024-01-02 FM_WARM1=2024-12-31 \
  FM_TEST0=2025-01-02 FM_TEST1=2025-12-31 .venv/bin/python3 finmem_sota/code/finmem_faithful.py
# Verbal-RL (API): per tier
RL_TIER=high .venv/bin/python3 rl_gpt_agent.py     # also RL_TIER=mid, RL_TIER=low
# CLF16 / GRPO (GPU, no API): needs adapters under rlvr_out_full14b/
CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 rlvr_eval_clf16.py clf16
CLF_TESTFILE=test3_high.jsonl   CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 rlvr_eval_clf16.py real
CLF_TESTFILE=test3_midlow.jsonl CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 rlvr_eval_clf16.py real_midlow
```

---

## Scoring convention (§4.3–4.5) and the scorer

Single action/day; signal uses info ≤ close(t−1); trade at open(t); measured at close(t). Per ticker, sort by date,
`daily = pos·r − (cost/1e4)·|pos − prev|`, prev starts 0, `CR% = (∏(1+daily)−1)·100`; `pos = unit-sign(action)`
{BUY +1, SELL −1, HOLD 0}; **cost = 2.5 bps/side HIGH, 10 bps/side under-covered** (paper §4.5, two-segment); B&H pays one
entry. `r` = next-day close-to-close **simple market-adjusted** return `(e^lr−1)−(e^spy−1)`, held out of model state.

- **Paper §5 scorer:** `_audit_indep_mr.py` → `_audit_indep_mr_out.json` (per-ticker CR/Sharpe/DirAcc/Acted, all systems).
- **Study-wide scorer:** `master_metrics_perstock.py` / `master_tables.py` (same convention; 3-tier cost variant adds low=28 bps — NOT used by this paper).
- **FinMem-only recompute:** `_adv_audit_perticker_recompute.py` → `_adv_audit_perticker_recompute_out.json` (see FinMem note).

---

## §5 — Results → source files (verified)

**Tables §5.1 (HIGH) & §5.2 (Under-covered):** `_audit_indep_mr_out.json` (script `_audit_indep_mr.py`).
✓ Reproduces the paper **exactly** for all 8 non-FinMem systems, per-ticker and EW:
EW HIGH B&H +8.9 / Judge-4o −4.6 / Judge-nano −4.9 / Verbal-RL +5.2 / GRPO −4.6 / SFT→GRPO −9.2 / CLF16 +2.3 / GRPO-shuf −8.1;
EW UNDER B&H −9.2 / Judge-4o −4.4 / Judge-nano +17.9 / Verbal-RL −7.0 / GRPO +1.3 / SFT→GRPO −1.9 / CLF16 +6.7 / GRPO-shuf +1.1.
Spot per-ticker confirmed: AAPL Judge-4o +26.2, TSLA Verbal-RL +36.1, MSEX Judge-4o +67.7, PAHC Judge-nano +112.0, IIIN Judge-nano +81.4.
**FinMem column is the one exception → see note below.**

---

## §6 — Ablation → source files (RQ2)

| Paper table | Result doc (read here) | Raw inputs (`Q/`) | Generating script |
|---|---|---|---|
| Judge market-specialist placebo decomposition (mega-cap): FULL −2.4 / occlusion +16.5 / genuine +8.7 (p=.237) / footprint +7.8; market 6/6 binomial p=.031 ◎ | `PLACEBO_AND_SIGNIFICANCE.md`, `HANDOFF_2026-06-16/results/COMPLETE_ABLATION_TWO_SEGMENT.md` §1a | `judge_decisions_gpt-4o-mini_2025{,_high_drop_market,_high_shuf_market,_high_placebo{1,2,3}}.json` | `judge_llm_system.py` (JUDGE_DROP/SHUFFLE/PLACEBO) → `audit_judge_ablation.py` |
| Under-covered source-occlusion, CLF16/GRPO/Verbal-RL (30 cells) ◎ | `HANDOFF_2026-06-16/results/COMPLETE_ABLATION_TWO_SEGMENT.md` §2a | `clf16_actions_real_midlow{,_drop_<src>}.json`; `rlagent_decisions_2025_{mid,low}{,_drop_<src>}.json` | `rlvr_eval_clf16.py` (CLF_DROP_SRC), `rl_gpt_agent.py` (RL_DROP_SRC); scoring `_audit_2seg_recompute.py` |
| Selected mega-cap sources per system (Occl/Genuine/Footprint/p) ◎ | `HANDOFF_2026-06-16/results/RL_CLF_SOURCE_OCCLUSION_ABLATION.md` §3, `COMPLETE_ABLATION_TWO_SEGMENT.md` §1b | `clf16_actions_clf16{,_drop_<src>,_shuf_<src>}.json`; `clf16_actions_real_h3full{,_drop/shuf,_s{1,2,3}}.json`; `rlagent_decisions_2025.json` | `rlvr_eval_clf16.py` (CLF_DROP/SHUF_SRC, CLF_SHUF_SEED) |

Orchestrators: `run_rlclf_ablation.sh`, `run_shuffle_ctrl.sh`, `run_multiseed.sh`, `run_grpo_midlow_abl.sh`,
`run_verbalrl_{high,midlow}_abl.sh`. ⚠ §6 under-covered (30-cell) magnitudes are auditor-claimed; not independently
re-derived this session. ⚠ binomial p=.031 is computed only — not in any committed doc; add it if cited.

---

## §7 — Error Analysis → source files (RQ3)

| Paper table | Script | Inputs | Render doc |
|---|---|---|---|
| Tier-wise position decomposition (HOLD%, dir bias, Long/Short n, P&L, hit, worst stock) — all cells match ◎ (mid+low pooled = "Under-covered") | `detailed_errors_tier.py` (aggregate auditor `audit_error_diag.py`, `master_error_tier.py`) | `judge_decisions_gpt-4o-mini_2025{,_mid,_low}.json`, `judge_decisions_gpt-5-nano_2025{,_mid,_low}.json`, `rlagent_decisions_2025{,_mid,_low}.json`, `clf16_actions_clf16.json` | `ERROR_ANALYSIS_4SYS_TIER.md`, `ERROR_ANALYSIS_DETAILED.md` |
| Table XII verbatim FinJudge rationales (NVDA 2025-01-24, IIIN 2025-10-15, MSEX 2025-10-31) — all confirmed verbatim ◎ | (read from `reason` field) | `judge_decisions_gpt-4o-mini_2025{,_mid,_low}.json` | `QUALITATIVE_ERROR_STUDY_TIER.md` |

⚠ §7 **worst-stock magnitudes** use additive `Σ pos·r` (NVDA −27, PRGS −65…) while the paper prints −29.6/−53.7/etc.
(compounded/segment convention); **stock identities all match**, only magnitudes differ — reconcile or footnote.

---

## ⚠ FinMem column — convention note (resolved 2026-06-24)

The paper's FinMem numbers (EW HIGH **−6.5**, UNDER **−6.8**; AAPL +28.9, **TSLA −13.8**) come from
`_adv_audit_perticker_recompute_out.json` (script `_adv_audit_perticker_recompute.py`, filters `mode=="adaptive"`).
This is **correct and table-consistent**: ✓ the shared `r` grid scoring all 8 other systems uses the **simple**
market-adjusted return `(e^lr−1)−(e^spy−1)` (matches 1454/1494 cells), and the FinMem recompute scores FinMem's *real
adaptive positions* on that same basis.

Do **not** substitute the project's "canonical faithful" −4.8/−5.5 (`HANDOFF_2026-06-16/results/CLF16_VS_BASELINES.md`):
those use the agent's own `r_adj` column, which ✓ I showed is a **log-excess** return `lr−spy` (120/137 rows) — a different
metric, not comparable in a simple-return table. Tell-tale ticker: TSLA −13.8 (paper, simple) vs −3.6 (faithful, log).
Faithful agent code = `finmem_sota/code/finmem_faithful.py`; verified vs official repo in
`finmem_sota/spec/FAITHFULNESS_vs_official_repo.md`. **Do NOT use** `_audit_indep_mr.py`'s FinMem column (it omits the
`mode=="adaptive"` filter → mixes 3 risk modes → AAPL +35.98 / TSLA −66.33); it is correct for the 8 other systems only.

---

## End-to-end reproduction order
1. Recreate `.venv` (see `HANDOFF_2026-06-16/ENVIRONMENT.md`), download Qwen2.5-14B-Instruct, set `.env`.
2. **Data (§3):** restore raw feeds → `qwen_news_summaries/build_sources.py` (per-ticker sources) → `build_rlvr_full.py` + `build_supervised_labeled.py` (train/test tables).
3. **Train (§4, GPU, optional):** `rlvr_clf_full14b.py` (CLF16); `rlvr_train_full14b.py --train {real,shuffled,real_sft}` (GRPO family); `rlvr_sft_full14b.py` + `merge_sft.py` (SFT→GRPO).
4. **Decisions (§4):** Judge / FinMem / Verbal-RL / CLF16 / GRPO eval commands above → `Q/*.json` + FinMem CSVs.
5. **§5 tables:** `.venv/bin/python3 _audit_indep_mr.py` → `_audit_indep_mr_out.json`; FinMem via `_adv_audit_perticker_recompute.py`.
6. **§6 ablation:** `run_*_abl.sh` → result docs under `HANDOFF_2026-06-16/results/`.
7. **§7 errors:** `detailed_errors_tier.py` → `ERROR_ANALYSIS_4SYS_TIER.md`; rationales from judge files.
8. **Verify:** re-run two independent code paths on any number before trusting it (auditor rule).

**Full §5/§6/§7 number→file verification:** `PAPER_RESULTS_PROVENANCE_2026-06-24.md`.
