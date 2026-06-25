# Dataset — qwen_news_summaries (per-ticker, 10 sources)
16 U.S. equities (6 mega-cap / 10 under-covered), 2023–2025. Each `<TICKER>/` folder has 10 files:
`*_news_daily_summary.csv` (LLM news brief), `*_social_*.json` (WSB sentiment), `*_technical_*.json`,
`*_market_*.json`, `*_industry_*.json`, `*_fundamentals_*.json`, `*_sue_*.json` (earnings surprise),
`*_filings_*.json` (8-K/10-Q/10-K), `*_price_return_*.json` (OHLCV+r_adj), `*_training.jsonl` (labeled state→action).
Derived/LLM-processed signals only; underlying RavenPack/Compustat/I-B-E-S feeds are WRDS-licensed and not included.
