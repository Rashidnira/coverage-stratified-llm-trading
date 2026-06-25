#!/usr/bin/env python3
"""Conviction-weighted sizing vs hard technical-confirmation gate vs baseline.

For each BUY/SELL decision, size the position by the fraction of NON-SENTIMENT
debaters (technical, market, earnings, industry, 8k) that corroborate the
decision's direction. Silent debaters are excluded from the tally.

  conviction = (#agree among non-sentiment present) / (#non-sentiment present)
  pos        = sign(decision) * conviction        # fractional, in [-1, 1]
  HOLD       -> pos = 0

This keeps a directional bet alive (unlike the hard gate, which zeroes it) but
shrinks it toward flat when corroboration is weak. Sentiment (news, social) no
longer sets size — it only proposes direction.

Net convention (locked): compound (1 + pos*r - cost*|pos-prev|), prev=0 start,
cost = 2.5 bps/side HIGH, 10 bps/side under-covered. B&H = pos==1 every day.
"""
import json, glob, os, math
from collections import Counter

ROOT = "/home/mr1639@students.ad.unt.edu/workspace/corporate_fake_news"
Q = os.path.join(ROOT, "dataset_2022_2025/_quality")
HIGH = ["AAPL", "TSLA", "MSFT", "AMZN", "GOOGL", "NVDA"]
LOW  = ["WDFC", "JJSF", "YORW", "MSEX"]
MID  = ["NBTB", "ANDE", "PRGS", "MYRG", "PAHC", "IIIN"]
COST = {**{t: 0.00025 for t in HIGH}, **{t: 0.0010 for t in LOW + MID}}
NONSENT = ["technical", "market", "earnings", "industry", "8k"]
SGN = {"BUY": 1, "SELL": -1, "HOLD": 0}

def load_grid():
    rmap = {}
    for suf in ["", "_low", "_mid"]:
        g = json.load(open(os.path.join(Q, f"judge_decisions_gpt-4o-mini_2025{suf}.json")))
        for r in g["decisions"]:
            if r.get("r") is not None:
                rmap[(r["tk"], r["d"])] = r["r"]
    return rmap

def conviction_pos(decision, positions):
    """fractional position from non-sentiment corroboration."""
    if decision not in ("BUY", "SELL"):
        return 0.0
    present = agree = 0
    for s in NONSENT:
        pv = positions.get(s)
        if not pv or pv.get("silent"):
            continue
        p = pv.get("position")
        if p in ("BUY", "SELL"):
            present += 1
            if p == decision:
                agree += 1
    if present == 0:
        return 0.0
    return SGN[decision] * (agree / present)

def hard_gate_pos(decision, positions):
    if decision in ("BUY", "SELL") and positions.get("technical", {}).get("position") != decision:
        return 0
    return SGN.get(decision, 0)

def daily_pnl(tk, posmap, rmap, cost):
    prev = 0.0; out = {}
    for d in sorted(posmap):
        if (tk, d) not in rmap:
            continue
        pos = posmap[d]; r = rmap[(tk, d)]
        out[d] = pos * r - cost * abs(pos - prev); prev = pos
    return out

def net(pnl):
    comp = 1.0
    for d in sorted(pnl):
        comp *= (1 + pnl[d])
    return round((comp - 1) * 100, 2)

def turnover(tk, posmap, rmap):
    prev = 0.0; t = 0.0
    for d in sorted(posmap):
        if (tk, d) not in rmap:
            continue
        t += abs(posmap[d] - prev); prev = posmap[d]
    return round(t, 1)

def paired_vs_bh(series, bh):
    days = sorted(set(series) & set(bh))
    diff = [series[d] - bh[d] for d in days]
    n = len(diff); m = sum(diff) / n
    sd = (sum((x - m) ** 2 for x in diff) / (n - 1)) ** .5
    t = m / (sd / n ** .5)
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / 2 ** .5)))
    return round(m * 250 * 100, 2), round(t, 2), round(p, 3), sum(1 for x in diff if x > 0), n

def main():
    rmap = load_grid()
    systems = {"QwenDeb": "multiagent/qwen_debate", "GPT4oDeb": "multiagent/gpt4omini_debate"}
    out = {}
    for sysname, base in systems.items():
        out[sysname] = {}
        for tk in HIGH + LOW + MID:
            f = glob.glob(os.path.join(ROOT, base, "*", f"{tk}_debate_2025.json"))
            if not f:
                out[sysname][tk] = None; continue
            d = json.load(open(f[0])); cost = COST[tk]
            base_pos = {k: SGN.get(d[k]["decision"], 0) for k in d}
            gate_pos = {k: hard_gate_pos(d[k]["decision"], d[k].get("positions", {})) for k in d}
            conv_pos = {k: conviction_pos(d[k]["decision"], d[k].get("positions", {})) for k in d}
            bh_pos   = {k: 1 for k in d}
            pnl = {n_: daily_pnl(tk, p, rmap, cost) for n_, p in
                   [("base", base_pos), ("gate", gate_pos), ("conv", conv_pos), ("bh", bh_pos)]}
            out[sysname][tk] = {
                "bh": net(pnl["bh"]),
                "base": net(pnl["base"]), "gate": net(pnl["gate"]), "conv": net(pnl["conv"]),
                "conv_turn": turnover(tk, conv_pos, rmap),
                "base_turn": turnover(tk, base_pos, rmap),
                "avg_conv_size": round(sum(abs(v) for v in conv_pos.values() if v) /
                                       max(1, sum(1 for v in conv_pos.values() if v)), 2),
                "_pnl_conv": pnl["conv"], "_pnl_bh": pnl["bh"], "_pnl_gate": pnl["gate"], "_pnl_base": pnl["base"],
            }
    # tables
    def ew(sysname, tks, key):
        v = [out[sysname][t][key] for t in tks if out[sysname][t]]
        return round(sum(v) / len(v), 2)
    tiers = [("HIGH", HIGH), ("UNDER", LOW + MID)]
    for sysname in systems:
        print(f"\n{'='*86}\n{sysname}: baseline vs hard-gate vs conviction-weighted (NET %, 2025)\n{'='*86}")
        print(f"{'Ticker':7} {'tier':6} {'B&H':>7} {'base':>8} {'hard':>8} {'conv':>8} {'avg|sz|':>8} {'turn b→c':>10}")
        for tier, tks in [("HIGH", HIGH), ("LOW", LOW), ("MID", MID)]:
            for tk in tks:
                o = out[sysname][tk]
                print(f"{tk:7} {tier:6} {o['bh']:>7} {o['base']:>8} {o['gate']:>8} {o['conv']:>8} {o['avg_conv_size']:>8} {o['base_turn']:>5}→{o['conv_turn']:<5}")
        for tier, tks in tiers:
            print(f"  EW {tier:5}        {ew(sysname,tks,'bh'):>7} {ew(sysname,tks,'base'):>8} {ew(sysname,tks,'gate'):>8} {ew(sysname,tks,'conv'):>8}")
        # significance: EW-portfolio conviction vs B&H
        for tier, tks in tiers:
            valid = [t for t in tks if out[sysname][t]]
            days = sorted(set.intersection(*[set(out[sysname][t]["_pnl_conv"]) for t in valid]))
            convEW = {dd: sum(out[sysname][t]["_pnl_conv"][dd] for t in valid) / len(valid) for dd in days}
            bhEW   = {dd: sum(out[sysname][t]["_pnl_bh"][dd]   for t in valid) / len(valid) for dd in days}
            ann, tt, p, w, n = paired_vs_bh(convEW, bhEW)
            print(f"  sig {tier:5}: conviction vs B&H  ann={ann:+6}%/yr  t={tt:+5}  p={p:.3f}  windays={w}/{n}")
    # strip pnl before dumping
    for s in out:
        for t in out[s]:
            if out[s][t]:
                for k in list(out[s][t]):
                    if k.startswith("_pnl"):
                        del out[s][t][k]
    json.dump(out, open(os.path.join(ROOT, "gate_conviction_out.json"), "w"), indent=1)
    print("\nwrote gate_conviction_out.json")

if __name__ == "__main__":
    main()
