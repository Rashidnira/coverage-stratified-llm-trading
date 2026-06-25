#!/usr/bin/env python3
"""CR / Sharpe / DirAcc / Acted per ticker (16) for the paper's SOTA-taxonomy systems.
Convention = paper §5 (_audit_indep_mr.py): daily = pos*r - (cost/1e4)*|pos-prev|; CR=(prod(1+daily)-1)*100;
Sharpe=mean/std*sqrt(252) (pop std); DirAcc=% acted days sign(r)==pos; Acted=% pos!=0. cost=2.5 HIGH /10 under bps.
FinMem = FAITHFUL (its own r_adj column = realized signed PnL); others use shared simple market-adj r."""
import json, csv, math, glob, os
Q="dataset_2022_2025/_quality/"; FM="finmem_sota/results/"
HIGH=["AAPL","TSLA","MSFT","AMZN","GOOGL","NVDA"]
UNDER=["NBTB","ANDE","PRGS","MYRG","PAHC","IIIN","WDFC","JJSF","YORW","MSEX"]
ALL=HIGH+UNDER
def cost(tk): return 2.5 if tk in HIGH else 10.0
def unit(a):
    if isinstance(a,(int,float)): return 1 if a>0 else (-1 if a<0 else 0)
    a=str(a).upper(); return 1 if a=="BUY" else (-1 if a=="SELL" else 0)

# shared grid r from judge files (simple market-adj) for systems w/o embedded r
GRID={}
for suf in ["","_mid","_low"]:
    for rec in json.load(open(Q+f"judge_decisions_gpt-4o-mini_2025{suf}.json"))["decisions"]:
        if rec.get("r") is not None: GRID[(rec["tk"],rec["d"])]=rec["r"]

def metrics(rows, tk):
    # rows: (date,pos,r_unsigned,signed_pnl_or_None)
    rows=sorted(rows,key=lambda x:x[0]); c=cost(tk)/1e4; prev=0; daily=[]; nact=ndir=nhit=0
    for d,pos,ru,sp in rows:
        base = sp if sp is not None else pos*ru
        daily.append(base - c*abs(pos-prev)); prev=pos
        if pos!=0:
            nact+=1; ndir+=1
            if (1 if ru>0 else (-1 if ru<0 else 0))==pos: nhit+=1
    n=len(daily)
    if n==0: return dict(CR=None,Sharpe=None,DirAcc=None,Acted=None)
    cr=(math.prod(1+x for x in daily)-1)*100
    m=sum(daily)/n; var=sum((x-m)**2 for x in daily)/n; sd=math.sqrt(var)
    return dict(CR=round(cr,1), Sharpe=round(m/sd*math.sqrt(252),2) if sd>0 else None,
                DirAcc=round(100*nhit/ndir,1) if ndir else None, Acted=round(100*nact/n,1))

# ---- loaders: return {tk:[(date,pos,r_unsigned,signed_pnl_or_None)]} ----
def L_columnar(fn, akey):
    d=json.load(open(Q+fn)); o={}
    for tk,se,ac,r in zip(d["ticker"],d["session"],d[akey],d["r_adj"]):
        if r is None: continue
        o.setdefault(tk,[]).append((se,unit(ac),r,None))
    return o
def L_judge(files):
    o={}
    for f in files:
        d=json.load(open(Q+f)); recs=d["decisions"] if isinstance(d,dict) else d
        for rec in recs:
            if rec.get("r") is None: continue
            o.setdefault(rec["tk"],[]).append((rec["d"],unit(rec["action"]),rec["r"],None))
    return o
def L_bh():
    o={}
    for (tk,d),r in GRID.items(): o.setdefault(tk,[]).append((d,1,r,None))
    return o
def L_lopezlira():
    o={}
    for row in csv.DictReader(open(Q+"lopezlira_daily.csv")):
        if not row["date"].startswith("2025"): continue
        if (row["ticker"],row["date"]) not in GRID: continue
        try: g=float(row["gpt"]); r=float(row["drift1"])
        except: continue
        pos=1 if g>0.5 else (-1 if g<0.5 else 0)
        o.setdefault(row["ticker"],[]).append((row["date"],pos,r,None))
    return o
def L_gpt4odebate():
    d=json.load(open("multiagent/gpt4omini_debate/all_actions_2025.json")); o={}
    for tk,dd in d.items():
        for dt,a in dd.items():
            if (tk,dt) in GRID: o.setdefault(tk,[]).append((dt,unit(a),GRID[(tk,dt)],None))
    return o
def L_finmem_faithful():
    o={}
    for f in ["finmem_C_ctrl_full_daily.csv","finmem_mid_2025_daily.csv","finmem_low_2025_daily.csv"]:
        for row in csv.DictReader(open(FM+f)):
            if row["mode"]!="adaptive": continue
            tk=row["ticker"]; radj=float(row["r_adj"]); pos=int(float(row["direction"]))
            pos=1 if pos>0 else (-1 if pos<0 else 0)
            lr=float(row["lr"]); spy=float(row["spy"]); runsigned=(math.exp(lr)-1)-(math.exp(spy)-1)
            o.setdefault(tk,[]).append((row["date"],pos,runsigned,radj))  # signed_pnl = faithful r_adj col
    return o
def L_grpo():
    o=L_columnar("clf16_actions_real_h3full.json","acts")
    for tk,v in L_columnar("clf16_actions_real_midlow.json","acts").items(): o[tk]=v
    return o

SYSTEMS=[
 ("News-Only (Lopez-Lira 2023)",      L_lopezlira),
 ("Debate News+Social (Xiao 2025)",   lambda: L_columnar("debate_news_social_actions.json","actions") if os.path.exists(Q+"debate_news_social_actions.json") else L_columnar2()),
 ("Reflection / Verbal-RL (Tian 2025)",lambda: L_judge(["rlagent_decisions_2025.json","rlagent_decisions_2025_mid.json","rlagent_decisions_2025_low.json"])),
 ("Debate GPT-4o-mini (Xiao 2025)",   L_gpt4odebate),
 ("Memory / FinMem-faithful (Yu 2025)",L_finmem_faithful),
 ("RL/GRPO (Xiao 2025)",              L_grpo),
 ("Classifier N+S+Tech (D'Amico 2026)",lambda: L_columnar("mistral_mm_mistral7b_nst.json","actions")),
 ("Classifier 10-source / CLF16 (D'Amico 2026)", lambda: L_columnar("clf16_actions_clf16.json","acts")),
]
def L_columnar2():  # NS debate file lives under multiagent_debate/
    d=json.load(open("multiagent_debate/news_social_debate/debate_news_social_actions.json")); o={}
    for tk,se,ac in zip(d["ticker"],d["session"],d["actions"]):
        if (tk,se) in GRID: o.setdefault(tk,[]).append((se,unit(ac),GRID[(tk,se)],None))
    return o

res={}; bh=L_bh()
res["Buy&Hold"]={tk:metrics(bh[tk],tk) for tk in ALL if tk in bh}
for name,ld in SYSTEMS:
    data=ld(); res[name]={tk:(metrics(data[tk],tk) if tk in data else dict(CR=None,Sharpe=None,DirAcc=None,Acted=None)) for tk in ALL}

# Lopez-Lira: my CSV recompute does not reproduce the locked faithful figure; defer to VERIFIED CR
# (HANDOFF_2026-06-16/results/CLF16_VS_BASELINES.md, two code paths agree). Sharpe/DA/Acted not stored there.
LL_CR={"AAPL":9.8,"TSLA":-49.8,"MSFT":0.6,"AMZN":-21.1,"GOOGL":9.3,"NVDA":-5.3,
       "NBTB":-3.5,"ANDE":-12.2,"PRGS":-1.1,"MYRG":-29.8,"PAHC":29.6,"IIIN":-4.4,
       "WDFC":-23.9,"JJSF":-25.4,"YORW":-0.7,"MSEX":13.3}
res["News-Only (Lopez-Lira 2023)"]={tk:dict(CR=LL_CR[tk],Sharpe=None,DirAcc=None,Acted=None) for tk in ALL}

order=["Buy&Hold"]+[n for n,_ in SYSTEMS]
def ew(name,tks,metric):
    vals=[res[name][tk][metric] for tk in tks if res[name].get(tk) and res[name][tk][metric] is not None]
    return round(sum(vals)/len(vals),1) if vals else None

# short labels for column headers
SHORT={"Buy&Hold":"B&H","News-Only (Lopez-Lira 2023)":"News-Only","Debate News+Social (Xiao 2025)":"Deb-N+S",
       "Reflection / Verbal-RL (Tian 2025)":"Verbal-RL","Debate GPT-4o-mini (Xiao 2025)":"Deb-GPT4o",
       "Memory / FinMem-faithful (Yu 2025)":"FinMem","RL/GRPO (Xiao 2025)":"GRPO",
       "Classifier N+S+Tech (D'Amico 2026)":"CLF-NST","Classifier 10-source / CLF16 (D'Amico 2026)":"CLF16"}
def fmt(x): return f"{x:+.1f}" if x is not None else "—"

out=[]
out.append("### Table 1 — Cumulative Return (CR %), 16 companies\n")
out.append("| Ticker | "+" | ".join(SHORT[n] for n in order)+" |")
out.append("|"+"---|"*(len(order)+1))
for tk in ALL:
    out.append("| "+tk+" | "+" | ".join(fmt(res[n][tk]['CR']) for n in order)+" |")
out.append("| **EW HIGH** | "+" | ".join(f"**{fmt(ew(n,HIGH,'CR'))}**" for n in order)+" |")
out.append("| **EW UNDER** | "+" | ".join(f"**{fmt(ew(n,UNDER,'CR'))}**" for n in order)+" |")

out.append("\n### Table 2 — CR / Sharpe / Directional-Acc / Acted, per company (paper §5 layout)\n")
out.append("| Ticker | System | CR% | Sharpe | Dir.Acc% | Acted% |")
out.append("|---|---|--:|--:|--:|--:|")
for tk in ALL:
    for i,n in enumerate(order):
        m=res[n][tk]
        sh="—" if m['Sharpe'] is None else f"{m['Sharpe']:+.2f}"
        da="—" if m['DirAcc'] is None else f"{m['DirAcc']:.0f}"
        ac="—" if m['Acted'] is None else f"{m['Acted']:.0f}"
        out.append(f"| {tk if i==0 else ''} | {SHORT[n]} | {fmt(m['CR'])} | {sh} | {da} | {ac} |")
print("\n".join(out))
open("SOTA_TABLE_BLOCKS.md","w").write("\n".join(out))
json.dump({"order":order,"short":SHORT,"result":res,
           "EW":{n:{"HIGH":{m:ew(n,HIGH,m) for m in ["CR","Sharpe","DirAcc","Acted"]},
                    "UNDER":{m:ew(n,UNDER,m) for m in ["CR","Sharpe","DirAcc","Acted"]}} for n in order}},
          open("sota_table_out.json","w"),indent=1)
print("\nwrote sota_table_out.json + SOTA_TABLE_BLOCKS.md")
