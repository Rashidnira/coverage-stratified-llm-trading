# Coverage-Stratified LLM Trading Dataset

A coverage-stratified, multi-source dataset for evaluating large language model (LLM) trading agents. It comprises sixteen U.S. equities—six highly analyst-covered mega-caps and ten under-covered firms—observed at daily frequency from 2023 to 2025. For each firm-trading-day, ten heterogeneous information sources are encoded as compact, decision-ready signals. Unstructured textual sources (news, social media, and SEC filings) are summarized and sentiment-scored with Qwen2.5-14B in a zero-shot setting; structured sources (technical, market, industry, and analyst-earnings-surprise) are reduced deterministically to interpretable state labels.

## Information sources
1. News summary
2. Social-media sentiment
3. Fundamentals
4. 10-K disclosure signal
5. 10-Q disclosure signal
6. 8-K material-event signal
7. Analyst earnings-surprise (SUE)
8. Market regime
9. Industry (sector) relative strength
10. Technical indicators

## Repository contents
- **`10_Sources_Master_Data/`** — per-firm daily signals for all ten sources; one directory per ticker (2023–2025).
- **`Data_Labeling_for_Classification/`** — supervised BUY/HOLD/SELL labels defined by a ±0.5% band on the next-day market-adjusted return; `train_2023_2024.jsonl` and `test_2025.jsonl`.
