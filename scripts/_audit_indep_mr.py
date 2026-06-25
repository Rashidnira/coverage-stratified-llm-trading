import json, csv, math

Q = "dataset_2022_2025/_quality/"
FM = "finmem_sota/results/"

HIGH = ["AAPL","TSLA","MSFT","AMZN","GOOGL","NVDA"]
MID  = ["NBTB","ANDE","PRGS","MYRG","PAHC","IIIN","WDFC","JJSF","YORW","MSEX"]
ALL  = HIGH + MID

def cost_bps(tk):
    return 2.5 if tk in HIGH else 10.0

def unit(a):
    a = a.upper()
    if a == "BUY": return 1
    if a == "SELL": return -1
    return 0

def metrics(rows, tk):
    # rows: list of (date, pos_unit, r) ; sort by date
    rows = sorted(rows, key=lambda x: x[0])
    cost = cost_bps(tk)
    prev = 0
    daily = []
    npos = 0
    ndir = 0
    nhit = 0
    for d, pos, r in rows:
        dd = pos*r - (cost/1e4)*abs(pos-prev)
        daily.append(dd)
        if pos != 0:
            npos += 1
            ndir += 1
            if (1 if r>0 else (-1 if r<0 else 0)) == pos:
                nhit += 1
        prev = pos
    n = len(daily)
    cr = (math.prod(1+x for x in daily)-1)*100 if n else float('nan')
    mean = sum(daily)/n if n else 0.0
    var = sum((x-mean)**2 for x in daily)/n if n else 0.0  # population std
    std = math.sqrt(var)
    sharpe = (mean/std*math.sqrt(252)) if std>0 else float('nan')
    diracc = (100.0*nhit/ndir) if ndir>0 else float('nan')
    acted = 100.0*npos/n if n else float('nan')
    return dict(CR=cr, Sharpe=sharpe, DirAcc=diracc, Acted=acted)

# ---------- loaders return {ticker: [(date,pos_unit,r), ...]} ----------

def load_judge(files):
    out = {}
    for f in files:
        d = json.load(open(Q+f))
        recs = d["decisions"] if isinstance(d,dict) else d
        for rec in recs:
            if rec.get("r") is None: continue
            tk = rec["tk"]
            out.setdefault(tk,[]).append((rec["d"], unit(rec["action"]), rec["r"]))
    return out

def load_buyhold(files):
    # pos=+1 always, r from judge file
    out = {}
    for f in files:
        d = json.load(open(Q+f))
        recs = d["decisions"] if isinstance(d,dict) else d
        for rec in recs:
            if rec.get("r") is None: continue
            tk = rec["tk"]
            out.setdefault(tk,[]).append((rec["d"], 1, rec["r"]))
    return out

def load_rl(files):
    out = {}
    for f in files:
        d = json.load(open(Q+f))
        recs = d["decisions"] if isinstance(d,dict) else d
        for rec in recs:
            if rec.get("r") is None: continue
            tk = rec["tk"]
            out.setdefault(tk,[]).append((rec["d"], unit(rec["action"]), rec["r"]))
    return out

def load_finmem(files):
    out = {}
    for f in files:
        for row in csv.DictReader(open(FM+f)):
            tk = row["ticker"]
            lr = float(row["lr"]); spy = float(row["spy"])
            r = (math.exp(lr)-1)-(math.exp(spy)-1)
            pos = int(float(row["direction"]))
            pos = 1 if pos>0 else (-1 if pos<0 else 0)
            out.setdefault(tk,[]).append((row["date"], pos, r))
    return out

def load_clf(files, actkey="acts"):
    out = {}
    for f in files:
        d = json.load(open(Q+f))
        n = len(d["ticker"])
        for i in range(n):
            tk = d["ticker"][i]
            pos = unit(d[actkey][i])
            r = d["r_adj"][i]
            dt = d["session"][i]
            out.setdefault(tk,[]).append((dt, pos, r))
    return out

def load_shuf_high():
    d = json.load(open(Q+"rlvr_full14b_actions_base_real_shuf.json"))
    acts = d["actions"]["shuffled"]
    out = {}
    n = len(d["ticker"])
    for i in range(n):
        tk = d["ticker"][i]
        out.setdefault(tk,[]).append((d["session"][i], unit(acts[i]), d["r_adj"][i]))
    return out

# ---------- system definitions ----------
systems = {}

systems["BuyHold"]  = load_buyhold(["judge_decisions_gpt-4o-mini_2025.json",
                                    "judge_decisions_gpt-4o-mini_2025_mid.json",
                                    "judge_decisions_gpt-4o-mini_2025_low.json"])

systems["FinMem"]   = load_finmem(["finmem_C_ctrl_full_daily.csv",
                                   "finmem_mid_2025_daily.csv",
                                   "finmem_low_2025_daily.csv"])

systems["Judge-4o"] = load_judge(["judge_decisions_gpt-4o-mini_2025.json",
                                  "judge_decisions_gpt-4o-mini_2025_mid.json",
                                  "judge_decisions_gpt-4o-mini_2025_low.json"])

systems["Judge-nano"]= load_judge(["judge_decisions_gpt-5-nano_2025.json",
                                   "judge_decisions_gpt-5-nano_2025_mid.json",
                                   "judge_decisions_gpt-5-nano_2025_low.json"])

systems["Verbal-RL"]= load_rl(["rlagent_decisions_2025.json",
                               "rlagent_decisions_2025_mid.json",
                               "rlagent_decisions_2025_low.json"])

systems["CLF16"]    = load_clf(["clf16_actions_clf16.json"])

# SFT: HIGH sft3_high + MID sft3_midlow
sft = load_clf(["clf16_actions_sft3_high.json"])
sft.update(load_clf(["clf16_actions_sft3_midlow.json"]))
systems["SFT"] = sft

# GRPO: HIGH real_h3full + MID real_midlow
grpo = load_clf(["clf16_actions_real_h3full.json"])
grpo.update(load_clf(["clf16_actions_real_midlow.json"]))
systems["GRPO"] = grpo

# SFT->GRPO: HIGH real_sft3_high + MID real_sft3_midlow
sg = load_clf(["clf16_actions_real_sft3_high.json"])
sg.update(load_clf(["clf16_actions_real_sft3_midlow.json"]))
systems["SFT->GRPO"] = sg

# GRPO-shuffled: HIGH from shuf file 'shuffled' + MID clf16_actions_shuffled_midlow
gs = load_shuf_high()
gs.update(load_clf(["clf16_actions_shuffled_midlow.json"]))
systems["GRPO-shuffled"] = gs

ORDER = ["BuyHold","FinMem","Judge-4o","Judge-nano","Verbal-RL","CLF16","SFT","GRPO","SFT->GRPO","GRPO-shuffled"]

result = {}
for sysname in ORDER:
    data = systems[sysname]
    for tk in ALL:
        if tk not in data:
            result.setdefault(tk,{})[sysname] = None
            continue
        m = metrics(data[tk], tk)
        result.setdefault(tk,{})[sysname] = {k: round(v,4) if v==v else None for k,v in m.items()}

json.dump({"order":ORDER,"HIGH":HIGH,"MID":MID,"result":result}, open("_audit_indep_mr_out.json","w"), indent=1)

# print summary table
for tk in ALL:
    tier = "HIGH" if tk in HIGH else "MID"
    print("\n#### %s (%s, cost=%s bps/side)"%(tk,tier,cost_bps(tk)))
    print("%-14s %8s %8s %8s %8s"%("system","CR%","Sharpe","DirAcc%","Acted%"))
    for s in ORDER:
        m = result[tk][s]
        if m is None:
            print("%-14s   MISSING"%s); continue
        def fmt(x): return ("%8.2f"%x) if x is not None else "     n/a"
        print("%-14s %s %s %s %s"%(s,fmt(m["CR"]),fmt(m["Sharpe"]),fmt(m["DirAcc"]),fmt(m["Acted"])))
print("\nDONE -> _audit_indep_mr_out.json")
