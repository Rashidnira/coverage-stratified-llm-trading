# GitHub Repo Plan + README/Data-Card — Coverage-Stratified LLM Trading Corpus
*Blueprint to review BEFORE publishing. Covers what to release, an efficient layout, and exact build steps.
Two blockers found 2026-06-25: (a) `git`/`gh` not installed here — repo creation must run elsewhere; (b) licensing — see §1.*

---

## 1. ⚠ Licensing — what can and cannot be public (read first)
The corpus is built from **licensed WRDS feeds**; raw redistribution is prohibited. Plan accordingly:

| Source | Underlying | Public-releasable? | Action |
|---|---|---|---|
| News briefs (`*_news_daily_summary.csv`) | RavenPack (WRDS) → **LLM-summarized** | ⚠ Derivative — likely OK (no raw RavenPack text), but **confirm with WRDS/RavenPack terms** | Release summarized briefs only; never raw RavenPack export |
| Social (`*_social_2023_2025.json`) | r/wallstreetbets via Pushshift | ✅ Public posts + derived sentiment | Release |
| Filings (`*_filings_*.json`) | SEC EDGAR | ✅ Public domain | Release |
| Fundamentals (`*_fundamentals_*.json`) | Compustat (WRDS) | ❌ Licensed | **Release derived ratios only**, not raw Compustat fields; or gate |
| SUE / analyst (`*_sue_*.json`) | I/B/E/S (WRDS) | ❌ Licensed | **Release standardized SUE signal only**, not raw I/B/E/S estimates |
| Technical/Market/Industry | Yahoo/CRSP/Fama-French/SPDR | ✅ derived states | Release |
| Price/return | Yahoo + CRSP-verified | ✅ adjusted OHLCV + r_adj | Release |
| Labels (`*_training.jsonl`) | derived ±0.5% band | ✅ | Release |

**Recommendation:** release the **derived/LLM-processed signals + labels + all code** (fully reproducible), and ship a
`build/` pipeline that regenerates the licensed-source fields for anyone with their own WRDS access. Default repo
**visibility = private** until WRDS/RavenPack redistribution is confirmed in writing.

## 2. Dataset summary (from `DATASET_STATISTICS_2026-06-25.md`)
16 firms (6 mega-cap / 10 under-covered), 2023–2025, 10 sources, **12,016 labeled instances** (BUY 4,165 / HOLD 3,450 /
SELL 4,401). News headlines 224,431; WSB posts 21,507 (21,491 HIGH vs 16 UNDER); filings 605 (8-K 423/10-Q 134/10-K 48).

## 3. Proposed repo layout (efficient)
```
coverage-stratified-llm-trading/
├── README.md                  # data card (this doc's §2,5,6) + quickstart
├── LICENSE                    # code: MIT; data: CC-BY-NC-4.0 + WRDS carve-out note
├── DATA_CARD.md               # provenance, licensing table (§1), known limits
├── .gitignore                 # *.live.txt, __pycache__, *.safetensors, .venv, raw feeds
├── data/
│   ├── parquet/               # one file per source (see §4) — COMPRESSED, the release format
│   │   ├── news_briefs.parquet         # ticker,date,n_headlines,summary
│   │   ├── social_wsb.parquet          # ticker,date,post_id,sentiment,confidence,themes,...
│   │   ├── filings.parquet             # ticker,date,form,items,sentiment,tone,impact
│   │   ├── fundamentals.parquet  *(derived ratios only)*
│   │   ├── sue.parquet           *(standardized signal only)*
│   │   ├── technical.parquet  market.parquet  industry.parquet
│   │   └── price_return.parquet
│   └── labeled/
│       └── instances.parquet           # 12,016 rows: ticker,date,split,state_text,r_adj,label
├── build/                     # regenerate licensed fields (needs your WRDS access)
│   └── build_sources.py  build_rlvr_full.py  build_supervised_labeled.py  ...
├── stats/
│   ├── dataset_stats.py        # reproduces the statistics tables
│   └── dataset_stats_out.json
└── docs/  figures/
```

## 4. Efficiency plan (the "very efficient" part)
Current raw footprint **140 MB**; target **< 25 MB** in-repo. Concrete steps:
1. **Drop redundant `*_news_daily_summary.live.txt` (5.9 MB)** — it duplicates the CSV (streaming log). `.gitignore` it.
2. **JSON → Parquet (columnar + Snappy/ZSTD).** The per-ticker dicts (social 9.3 MB, technical/market/industry, filings) compress ~5–10×. One parquet per source for all 16 tickers (a `ticker` column) instead of 16×N files → far fewer objects, faster clone.
3. **`instances.parquet`** instead of 16 × `training.jsonl` (10 MB): the `state_text` repeats source content; storing once columnar + ZSTD ≈ 1–2 MB.
4. **Don't commit model weights** (`rlvr_out_full14b/*.safetensors`, adapters) — link to a release/HF instead. `.gitignore`.
5. **Git-LFS only if any single file > 50 MB** (after parquet, none should be) — otherwise plain git keeps clones fast.
6. **One canonical schema doc** (`DATA_CARD.md`) so users don't open files to learn columns.

Net: ~140 MB of scattered JSON/CSV/TXT → ~10–25 MB of ~10 parquet files + code, clone in seconds.

## 5. README quickstart (draft)
```python
import pandas as pd
inst = pd.read_parquet("data/labeled/instances.parquet")   # 12,016 firm-days, BUY/HOLD/SELL
news = pd.read_parquet("data/parquet/news_briefs.parquet")
# join any source to a (ticker, date) and reproduce the paper's tables via /stats
```
Cite: *Beyond Mega-Cap Stocks: Evaluating LLM Trading Agents Across Analyst-Coverage Environments* (2026).

## 6. Build steps (run where git is installed; NOT possible in this env)
```bash
# 0. install git + gh first (not present here):  sudo apt-get install git gh   (or conda)
# 1. stage an efficient copy (parquet conversion script provided in build/)
python build/make_release_parquet.py        # 140MB raw -> data/parquet/*.parquet  (TODO: I can write this)
# 2. init + first commit
git init && git add README.md LICENSE DATA_CARD.md data/ build/ stats/ .gitignore
git commit -m "Release: coverage-stratified LLM trading corpus (derived signals + code)"
# 3. create the repo (private until licensing confirmed) and push
gh repo create coverage-stratified-llm-trading --private --source=. --push
```

## 7. Open decisions for you (before I proceed)
1. **Visibility:** private (recommended until WRDS/RavenPack OK) or public?
2. **Scope:** derived signals + code only (recommended), or attempt to include raw licensed fields (not advisable)?
3. **Weights:** exclude model adapters from the repo (link to HF/Release)?
4. Want me to **write `build/make_release_parquet.py`** now and produce the compressed `data/parquet/` locally, so the repo is ready to `git init` the moment git is available?
