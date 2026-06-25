#!/usr/bin/env python3
"""Per-ticker (all 16) NET% for every classifier system, on the locked two-segment convention.
NET = compound(1 + pos*r - cost*|pos-prev|), prev=0; pos=unit-sign(action); cost=2.5bps HIGH / 10bps under;
r = embedded r_adj (simple market-adjusted next-day return, == shared grid). Buy&Hold = pos +1 always."""
import json, os
Q = "dataset_2022_2025/_quality/"
HIGH = ["AAPL","TSLA","MSFT","AMZN","GOOGL","NVDA"]
UNDER = ["NBTB","ANDE","PRGS","MYRG","PAHC","IIIN","WDFC","JJSF","YORW","MSEX"]
ALL = HIGH + UNDER
COST = {**{t:0.00025 for t in HIGH}, **{t:0.0010 for t in UNDER}}
SGN = {"BUY":1,"SELL":-1,"HOLD":0}

SYSTEMS = {  # label -> (file, action-key)
  "CLF16 (Qwen,full)":      ("clf16_actions_clf16.json","acts"),
  "CLF16-v2 (Qwen,full)":   ("clf16_actions_clf16_v2.json","acts"),
  "CLF16-anon (Qwen,full)": ("clf16_actions_clf16_anon.json","acts"),
  "Mistral-mm (full)":      ("mistral_mm_mistral_mm.json","actions"),
  "Qwen-mm (full)":         ("mm_eval_qwen2_5_14b_instruct_mm.json","actions"),
  "Mistral-NST (n+s+tech)": ("mistral_mm_mistral7b_nst.json","actions"),
}

def load(fn, akey):
    d = json.load(open(Q+fn))
    rows = {}
    for tk, se, ac, r in zip(d["ticker"], d["session"], d[akey], d["r_adj"]):
        if r is None: continue
        rows.setdefault(tk, []).append((se, SGN.get(str(ac).upper(),0), r))
    return rows

def net(rows, tk):
    rows = sorted(rows, key=lambda x:x[0]); prev=0; comp=1.0; cost=COST[tk]
    for _, pos, r in rows:
        comp *= (1 + pos*r - cost*abs(pos-prev)); prev=pos
    return (comp-1)*100

def bh(rows, tk):
    rows = sorted(rows, key=lambda x:x[0]); comp=1.0; cost=COST[tk]; first=True
    for _, _, r in rows:
        comp *= (1 + 1*r - (cost if first else 0)); first=False
    return (comp-1)*100

# B&H from CLF16 file (has all r); + each system
data = {lab: load(*v) for lab,v in SYSTEMS.items()}
bhrows = load("clf16_actions_clf16.json","acts")
result = {}
for tk in ALL:
    result[tk] = {"B&H": round(bh(bhrows[tk],tk),1)}
    for lab in SYSTEMS:
        result[tk][lab] = round(net(data[lab].get(tk,[]),tk),1) if tk in data[lab] else None

cols = ["B&H"] + list(SYSTEMS.keys())
def ewrow(tks):
    out={}
    for c in cols:
        vals=[result[t][c] for t in tks if result[t][c] is not None]
        out[c]=round(sum(vals)/len(vals),1)
    return out

# print markdown
hdr = "| Ticker | " + " | ".join(cols) + " |"
sep = "|" + "---|"*(len(cols)+1)
print(hdr); print(sep)
for tk in ALL:
    print("| "+tk+" | " + " | ".join(f"{result[tk][c]:+.1f}" if result[tk][c] is not None else "—" for c in cols) + " |")
ewh, ewu = ewrow(HIGH), ewrow(UNDER)
print("| **EW HIGH** | " + " | ".join(f"**{ewh[c]:+.1f}**" for c in cols) + " |")
print("| **EW UNDER** | " + " | ".join(f"**{ewu[c]:+.1f}**" for c in cols) + " |")
json.dump({"result":result,"EW_HIGH":ewh,"EW_UNDER":ewu,"cols":cols},
          open("classifier_perticker_out.json","w"), indent=1)
