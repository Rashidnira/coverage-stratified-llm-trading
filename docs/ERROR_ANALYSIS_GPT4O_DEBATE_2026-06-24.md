# Error analysis — GPT-4o-mini multi-agent debate (Table XII equivalent)
*Worst directional decisions of the 7-specialist debate + CTO (gpt-4o-mini), with verbatim CTO rationales, 2025 OOS.
Mirrors the FinJudge Table XII. Source: `multiagent/gpt4omini_debate/<company>/<TK>_debate_2025.json`; r from the shared
market-adjusted grid. "Loss" = realized `pos·r` on the day the position was held.*

## Table — Coverage-graded error modes with representative debate rationales
| Tier | Ticker | Date | Decision | r% | Specialist votes (the debate panel) | CTO reason (verbatim, leading clause) | Error mode |
|---|---|---|---|--:|---|---|---|
| High | TSLA | 2025-04-08 | SELL | **+12.2** | NEWS·SOCIAL·TECHNICAL·MARKET = SELL; 8-K = BUY | "The primary specialists are predominantly bearish, with NEWS, SOCIAL, and TECHNICAL all signaling a strong SELL… overall sentiment is clearly negative." | **Consensus-chasing** — 4 bearish specialists treated as conviction; shorted into a +12% rally |
| High | NVDA | 2025-04-08 | SELL | **+8.2** | NEWS = BUY; TECHNICAL·MARKET·INDUSTRY = SELL; SOCIAL = HOLD | "…conflicting signals from the NEWS (BUY) and TECHNICAL (SELL)… given the high confidence of the TECHNICAL specialist, the overall tilt leans bearish." | **Directional ambiguity** — conflicting panel resolved the wrong way |
| Low | PRGS | 2025-06-30 | BUY | **−13.0** | ALL 7 specialists = BUY (unanimous) | "The primary specialists unanimously indicate a strong BUY signal, with high confidence across NEWS, SOCIAL, EARNINGS, 8-K, and TECHNICAL…" | **Consensus-chasing** — unanimous agreement = conviction; bought into a −13% drop |
| Low | MYRG | 2025-01-24 | BUY | **−10.9** | TECHNICAL = BUY; NEWS·SOCIAL·EARNINGS·8-K = SILENT; INDUSTRY = BUY, MARKET = HOLD | "The technical specialist's strong buy is the only primary position, while all other primary specialists are silent. The market and industry context supports a bullish outlook…" | **Macro/sector override** — firm signals silent; bought on market+sector context |
| Low | MSEX | 2025-11-03 | SELL | **+11.1** | NEWS·EARNINGS·TECHNICAL·INDUSTRY = SELL; 8-K = BUY | "The primary specialists are predominantly bearish, with NEWS, EARNINGS, and TECHNICAL all signaling a SELL… outweighs the optimistic tone from the 8-K." | **Consensus-chasing** — bearish news/earnings herd; shorted into a +11% rally |

## Interpretation (parallels §7.3 of the paper)
The GPT-4o-mini debate exhibits the **same coverage-graded failure modes as the FinJudge**:

1. **Consensus-chasing (both tiers).** When many specialists agree, the CTO treats agreement as conviction — but the
   agreement usually reflects *recently observed* price/news momentum, not forward-looking signal. The debate shorts TSLA
   and MSEX on a unanimous bearish panel straight into double-digit rallies, and buys PRGS on a 7-for-7 bullish panel into
   a −13% drop. Agreement amplifies, rather than corrects, the herd.

2. **Macro/sector override (under-covered).** When firm-specific specialists (news, social, earnings, 8-K) are **silent**
   — the normal state for under-covered names — the CTO leans on market/industry context. MYRG is bought purely on
   "market and industry context" with every firm-level specialist silent, into a −10.9% move. This is the same
   firm-specific→macro collapse the paper documents for the judge as coverage falls.

3. **Directional ambiguity.** Genuinely conflicting panels (NVDA: news BUY vs technical SELL) are resolved by whichever
   specialist is loudest, not by forward information — directional accuracy stays at chance (debate EW DA ≈ 51% HIGH / 53% UNDER).

**Net:** the multi-agent debate does not escape the judge's errors; it reproduces them, because the CTO aggregator has the
same two reflexes — follow the consensus, and fall back on macro when firm signals are absent. TSLA is the sharpest case
(every bearish panel shorts a rising stock).

## Provenance / reproduce
Decision files: `multiagent/gpt4omini_debate/<company>/<TK>_debate_2025.json` (fields `decision`, `cto_reasoning`,
`positions`). Worst-decision mining + verbatim pull: ad-hoc over those files vs the shared `r` grid
(`judge_decisions_gpt-4o-mini_2025{,_mid,_low}.json`). Per-company debate CR/Sharpe/DA/Acted: `SOTA_RESULTS_16CO_2026-06-24.md` (Deb-GPT4o).
