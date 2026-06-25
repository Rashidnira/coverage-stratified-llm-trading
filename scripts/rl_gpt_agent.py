#!/usr/bin/env python3
"""REAL-TIME gpt-4o-mini RL agent (in-context / verbal RL). gpt-4o-mini IS the policy: each trading day it sees the
10-source signals for a stock AND a rolling log of its OWN recent (signals->action->realized net reward), and must
choose a position to MAXIMIZE cumulative net-of-cost reward -- learning online from its reward history. 6 mega-caps,
2025, processed chronologically (real-time). Saves decisions for an independent auditor to evaluate vs B&H/judge/
majority/placebo. NO look-ahead: the reward buffer at day d contains only outcomes realized strictly before d."""
import json,os,hashlib,time,numpy as np
from collections import defaultdict
from dotenv import load_dotenv; from openai import OpenAI
import judge_llm_system as J
load_dotenv('.env'); client=OpenAI(timeout=45,max_retries=3); MODEL='gpt-4o-mini'
CACHE='dataset_2022_2025/_quality/rlagent_cache'; os.makedirs(CACHE,exist_ok=True)
import sys
TIER=os.environ.get('RL_TIER','high')
TIERSETS={'high':['AAPL','TSLA','MSFT','AMZN','GOOGL','NVDA'],'mid':['NBTB','ANDE','PRGS','MYRG','PAHC','IIIN'],'low':['WDFC','JJSF','YORW','MSEX']}
HI=TIERSETS[TIER]; _sfx='' if TIER=='high' else f'_{TIER}'
AGENTS=['news','social','technical','8k','sue','10q','10k','fundamentals','market','industry']
DROP=set(s for s in os.environ.get('RL_DROP_SRC','').split(',') if s)   # source-occlusion: exclude these from the signal string
if DROP: print(f"[RL OCCLUDE] dropping sources {sorted(DROP)} from every signal string",flush=True)
SIGN={'BUY':1,'SELL':-1,'HOLD':0}; COST=lambda pos:2*2.5/1e4*abs(pos)
def sigstr(tk,d):
    v=J.specialists_for(tk,d); parts=[]
    for a in AGENTS:
        if a in DROP: continue
        vi=v.get(a,{})
        if vi.get('action') and vi.get('action')!='HOLD': parts.append(f"{a}:{vi['action'][0]}{vi.get('conviction',0):.1f}")
    return ", ".join(parts) or "no active signals"
SYS=("You are a reinforcement-learning equity trader. GOAL: maximize CUMULATIVE net-of-cost reward over many trades. "
     "Each step you get today's multi-source signals for ONE stock and a LOG of your own RECENT trades with their "
     "REALIZED rewards. LEARN online from that reward log: lean into signal patterns that have PAID, avoid patterns "
     "that LOST. A wrong-sized or wrong-direction bet costs you. Output strict JSON only.")
def decide(tk,d,sig,buf):
    usr=(f"Your recent trades (signal-pattern -> your action -> realized net reward %):\n{buf or '(none yet)'}\n\n"
         f"TODAY  {tk}  {d}  signals: {sig}\n"
         "Choose a position to maximize cumulative reward, using what your reward log taught you. "
         'JSON {"action":"BUY|SELL|HOLD","conviction":0.0-1.0,"why":"<=15 words"}')
    k=hashlib.md5((MODEL+SYS+usr).encode()).hexdigest(); fp=f"{CACHE}/{k}.json"
    if os.path.exists(fp): return json.load(open(fp))
    for a in range(4):
        try:
            r=client.chat.completions.create(model=MODEL,temperature=0.2,max_tokens=70,response_format={"type":"json_object"},
                messages=[{"role":"system","content":SYS},{"role":"user","content":usr}])
            o=json.loads(r.choices[0].message.content); json.dump(o,open(fp,"w")); return o
        except Exception:
            if a==3: return {"action":"HOLD","conviction":0}
            time.sleep(1.5*(a+1))

dec=[x for x in json.load(open(f'dataset_2022_2025/_quality/judge_decisions_gpt-4o-mini_2025{_sfx}.json'))['decisions'] if x['r'] is not None and x['tk'] in HI]
days=sorted(set(x['d'] for x in dec)); byday=defaultdict(list)
for x in dec: byday[x['d']].append(x)
print(f'real-time RL agent over {len(days)} days x {len(HI)} names...',flush=True)
buf=[]; out=[]   # buf = global rolling reward log (realized BEFORE current day)
from concurrent.futures import ThreadPoolExecutor
for di,d in enumerate(days):
    bufstr="\n".join(buf[-18:])
    items=byday[d]
    def go(x):
        sig=sigstr(x['tk'],d); o=decide(x['tk'],d,sig,bufstr)
        pos=SIGN.get(o.get('action','HOLD'),0)*float(o.get('conviction',0) or 0)
        net=pos*x['r']-COST(pos)
        return dict(tk=x['tk'],d=d,action=o.get('action'),conviction=o.get('conviction',0),pos=pos,r=x['r'],net=net,sig=sig)
    with ThreadPoolExecutor(max_workers=6) as ex: res=list(ex.map(go,items))
    out+=res
    for rr in res: buf.append(f"[{rr['tk']} {d}] {rr['sig'][:60]} -> {rr['action']} {rr['conviction']:.1f} -> reward {rr['net']*100:+.2f}%")
    if di%40==0: print(f'  {di}/{len(days)}  cum-buf {len(buf)}',flush=True)
OUTTAG=os.environ.get('RL_OUTTAG','')   # output filename suffix (e.g. _mid_drop_news); default keeps the FULL name
_fn=f'rlagent_decisions_2025{OUTTAG or _sfx}.json'
json.dump(out,open(f'dataset_2022_2025/_quality/{_fn}','w'))
print(f'saved {_fn} ({len(out)} decisions). Hand to auditor for evaluation.')
