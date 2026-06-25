#!/usr/bin/env python3
"""Apply a technical-confirmation gate to the debate decisions and re-score net %
across all 16 tickers, two backbones (Qwen debate, GPT-4o-mini debate).

Gate rule: a BUY/SELL decision is kept ONLY if the `technical` debater's position
agrees with the decision's direction; otherwise the decision becomes HOLD.
Everything else (HOLD, and BUY/SELL confirmed by technical) is unchanged.

Net convention (locked, per SESSION_LOG_2026-06-24 §3):
  per-ticker compound of (1 + pos*r - cost*|pos-prev|), prev starts 0,
  pos = unit sign of action (HOLD=0). cost = 2.5 bps/side HIGH, 10 bps/side under-covered.
  B&H = pos==1 every day (pays one entry cost).
Return grid r: judge_decisions_gpt-4o-mini_2025{,_low,_mid}.json field 'r'.
"""
import json, glob, os
from collections import Counter

ROOT = "/home/mr1639@students.ad.unt.edu/workspace/corporate_fake_news"
Q = os.path.join(ROOT, "dataset_2022_2025/_quality")

HIGH = ["AAPL", "TSLA", "MSFT", "AMZN", "GOOGL", "NVDA"]
LOW  = ["WDFC", "JJSF", "YORW", "MSEX"]
MID  = ["NBTB", "ANDE", "PRGS", "MYRG", "PAHC", "IIIN"]
COST = {**{t: 0.00025 for t in HIGH}, **{t: 0.0010 for t in LOW + MID}}

def load_grid():
    rmap = {}
    for suf in ["", "_low", "_mid"]:
        g = json.load(open(os.path.join(Q, f"judge_decisions_gpt-4o-mini_2025{suf}.json")))
        for row in g["decisions"]:
            if row.get("r") is not None:
                rmap[(row["tk"], row["d"])] = row["r"]
    return rmap

SGN = {"BUY": 1, "SELL": -1, "HOLD": 0}

def score(tk, decmap, rmap, cost):
    """decmap: date -> 'BUY'/'SELL'/'HOLD'. Returns (net%, turnover, n_days)."""
    prev, comp, turn, n = 0, 1.0, 0, 0
    for d in sorted(decmap):
        if (tk, d) not in rmap:
            continue
        pos = SGN.get(decmap[d], 0)
        r = rmap[(tk, d)]
        comp *= (1 + pos * r - cost * abs(pos - prev))
        turn += abs(pos - prev)
        prev = pos
        n += 1
    return round((comp - 1) * 100, 2), turn, n

def bh(tk, dates, rmap, cost):
    return score(tk, {d: "BUY" for d in dates}, rmap, cost)

def gate(decision, positions):
    """technical-confirmation gate."""
    if decision in ("BUY", "SELL"):
        if positions.get("technical", {}).get("position") != decision:
            return "HOLD"
    return decision

def main():
    rmap = load_grid()
    systems = {
        "QwenDeb":  "multiagent/qwen_debate",
        "GPT4oDeb": "multiagent/gpt4omini_debate",
    }
    tiers = {"HIGH": HIGH, "LOW": LOW, "MID": MID}
    out = {}
    for sysname, base in systems.items():
        out[sysname] = {}
        for tk in HIGH + LOW + MID:
            f = glob.glob(os.path.join(ROOT, base, "*", f"{tk}_debate_2025.json"))
            if not f:
                out[sysname][tk] = None
                continue
            d = json.load(open(f[0]))
            base_dec = {k: d[k]["decision"] for k in d}
            gate_dec = {k: gate(d[k]["decision"], d[k].get("positions", {})) for k in d}
            cost = COST[tk]
            dates = list(d.keys())
            bnet, _, _ = bh(tk, dates, rmap, cost)
            b = score(tk, base_dec, rmap, cost)
            gt = score(tk, gate_dec, rmap, cost)
            out[sysname][tk] = {
                "bh": bnet,
                "base_net": b[0], "base_turn": b[1],
                "gate_net": gt[0], "gate_turn": gt[1],
                "base_mix": dict(Counter(base_dec.values())),
                "gate_mix": dict(Counter(gate_dec.values())),
                "n": b[2],
            }
    # ---- print tables ----
    def ew(sysname, tier_tks, key):
        vals = [out[sysname][t][key] for t in tier_tks if out[sysname][t]]
        return round(sum(vals) / len(vals), 2)
    def ew_bh(sysname, tier_tks):
        vals = [out[sysname][t]["bh"] for t in tier_tks if out[sysname][t]]
        return round(sum(vals) / len(vals), 2)

    for sysname in systems:
        print(f"\n{'='*78}\n{sysname}: technical-confirmation gate vs baseline (NET %, 2025)\n{'='*78}")
        print(f"{'Ticker':7} {'tier':5} {'B&H':>7} {'base':>8} {'gated':>8} {'Δ':>7}  {'turn b→g':>10}")
        for tier, tks in tiers.items():
            for tk in tks:
                o = out[sysname][tk]
                if not o:
                    print(f"{tk:7} {tier:5}  (missing)"); continue
                dlt = round(o["gate_net"] - o["base_net"], 2)
                print(f"{tk:7} {tier:5} {o['bh']:>7} {o['base_net']:>8} {o['gate_net']:>8} {dlt:>+7}  {o['base_turn']:>4}→{o['gate_turn']:<4}")
            # EW per tier
            print(f"{'EW '+tier:12}     {ew_bh(sysname,tks):>7} {ew(sysname,tks,'base_net'):>8} {ew(sysname,tks,'gate_net'):>8}")
        # combined under-covered EW (LOW+MID) to match session-log "UNDER"
        und = LOW + MID
        print(f"{'EW UNDER':12}     {ew_bh(sysname,und):>7} {ew(sysname,und,'base_net'):>8} {ew(sysname,und,'gate_net'):>8}")
    json.dump(out, open(os.path.join(ROOT, "gate_techconfirm_out.json"), "w"), indent=1)
    print("\nwrote gate_techconfirm_out.json")

if __name__ == "__main__":
    main()
