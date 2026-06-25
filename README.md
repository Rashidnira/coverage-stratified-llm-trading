# Coverage-Stratified LLM Trading Agents — Results & Reproducibility

Companion repository for *Beyond Mega-Cap Stocks: Evaluating LLM Trading Agents Across Analyst-Coverage Environments*.
16 U.S. equities (6 mega-cap / 10 under-covered), 10 information sources, 2025 out-of-sample (249 trading days),
market-adjusted vs SPY, two-segment transaction costs (2.5 bps/side HIGH, 10 bps/side under-covered).

## Contents
- `docs/` — results & methodology
  - `MASTER_RESULTS.md` — single entry point (per-company tables + index)
  - `SOTA_RESULTS_16CO_*.md` — all systems × 16 firms (CR/Sharpe/DirAcc/Acted)
  - `CLASSIFIER_RESULTS_16CO_*.md` — supervised classifiers (CLF16, Mistral-mm, Qwen-mm, NST)
  - `VERBAL_RL_RESULT_*.md` — Reflection-Driven agent (exact result + provenance)
  - `PAPER_RESULTS_PROVENANCE_*.md` — every paper number → source file
  - `IPM_PAPER1_REPRODUCE.md` — full reproduction manifest
  - `DATASET_STATISTICS_*.md` — corpus statistics (12,016 labeled instances)
  - `SAMPLE_DATA.md` — raw → LLM-processed examples for all 10 sources
- `scripts/` — agent (`rl_gpt_agent.py`) + scorers (`_audit_indep_mr.py`, `score_sota_table.py`, …)
- `results/` — derived result tables (JSON)

## ⚠ Data licensing
This repo contains **derived signals, results, and code only**. The underlying feeds — RavenPack, Compustat, and
I/B/E/S — are licensed via WRDS and are **not redistributed here**. Reproduce licensed-source fields with your own
WRDS access using the `build/` pipeline described in `docs/IPM_PAPER1_REPRODUCE.md`. Reddit/EDGAR sources are public.

## License
Code: MIT. Documentation/derived data: CC-BY-NC-4.0.
