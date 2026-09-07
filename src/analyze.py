import json, math, os, statistics, time
from datetime import datetime, timezone, timedelta
from pathlib import Path
import requests

BASE="https://api.upbit.com"
OUT=Path("docs/data/latest.json")
TOP_N=int(os.getenv("TOP_N","10"))
DAYS_BACK=int(os.getenv("DAYS_BACK","60"))
EXCLUDE_PUMP_PCT=float(os.getenv("EXCLUDE_PUMP_PCT","20"))
LOOKAHEAD=int(os.getenv("LOOKAHEAD_HOURS","6"))
K=int(os.getenv("K_NEIGHBORS","50"))
DELAY=float(os.getenv("REQUEST_DELAY","0.11"))
STABLE={"USDT","USDC","DAI","TUSD","FDUSD","USDE","PYUSD","USDS","USD1","RLUSD","BUSD","USTC"}
S=requests.Session()
S.headers.update({"Accept":"application/json","User-Agent":"upbit-v4-event-backtest-scanner/4.0"})

def api(path,params=None,retries=6):
    last=None
    for n in range(retries):
        try:
            r=S.get(BASE+path,params=params,timeout=20)
            if r.status_code==429:
                time.sleep(1.0+n); continue
            r.raise_for_status(); time.sleep(DELAY); return r.json()
        except Exception as e:
            last=e; time.sleep(min(2**n,8))
    raise RuntimeError(f"{path}: {last}")

def market_list():
    out=[]
    for x in api("/v1/market/all",{"is_details":"true"}):
        mk=x["market"]
        if not mk.startswith("KRW-"): continue
        sym=mk.split("-",1)[1].upper()
        if sym in STABLE: continue
        ev=x.get("market_event") or {}
        if ev.get("warning") is True: continue
        out.append({"market":mk,"name":x.get("korean_name",sym),"symbol":sym})
    return out

def minute(mk,u,count=200,to=None):
    p={"market":mk,"count":min(count,200)}
    if to: p["to"]=to
    return api(f"/v1/candles/minutes/{u}",p)

def day(mk,count=35):
    return list(reversed(api("/v1/candles/days",{"market":mk,"count":count})))

def history_1h(mk,days_back=DAYS_BACK):
    need=days_back*24
    allrows=[]; to=None
    while len(allrows)<need:
        rows=minute(mk,60,min(200,need-len(allrows)),to)
        if not rows: break
        allrows.extend(rows)
        oldest=rows[-1]["candle_date_time_utc"]
        dt=datetime.fromisoformat(oldest).replace(tzinfo=timezone.utc)-timedelta(seconds=1)
        to=dt.strftime("%Y-%m-%dT%H:%M:%S")
        if len(rows)<min(200,need-len(allrows)+len(rows)): break
    # API pages newest->oldest; de-duplicate and sort ascending
    uniq={r["candle_date_time_utc"]:r for r in allrows}
    return [uniq[k] for k in sorted(uniq)]

def div(a,b,d=0): return a/b if b not in (0,None) else d
def mean(x): return statistics.fmean(x) if x else 0
def sd(x): return statistics.pstdev(x) if len(x)>1 else 0
def clamp(x,a,b): return max(a,min(b,x))
def ema(v,n):
    if not v:return 0
    a=2/(n+1); z=v[0]
    for x in v[1:]: z=a*x+(1-a)*z
    return z
def rsi(v,n=14):
    if len(v)<n+1:return 50
    d=[v[i]-v[i-1] for i in range(1,len(v))]
    g=mean([max(x,0) for x in d[-n:]]); l=mean([max(-x,0) for x in d[-n:]])
    if l==0:return 100 if g else 50
    return 100-100/(1+g/l)

F=["ret1","ret3","ret6","ret24","rsi","ema_gap","vol_ratio","volatility","bb_width","range_pos","breakout_dist","body"]

def features(rows,i=None):
    if i is None:i=len(rows)-1
    if i<30:return None
    w=rows[max(0,i-40):i+1]
    c=[float(x["trade_price"]) for x in w]; h=[float(x["high_price"]) for x in w]
    l=[float(x["low_price"]) for x in w]; v=[float(x["candle_acc_trade_price"]) for x in w]
    now=c[-1]
    def ret(n):return (div(now,c[-1-n],1)-1)*100 if len(c)>n else 0
    rr=[(div(c[j],c[j-1],1)-1)*100 for j in range(1,len(c))]
    ma=mean(c[-20:]); s=sd(c[-20:]); hi=max(h[-20:]); lo=min(l[-20:]); ph=max(h[-21:-1])
    op=float(w[-1]["opening_price"]); hh=float(w[-1]["high_price"]); ll=float(w[-1]["low_price"])
    pv=v[-24:-3] if len(v)>=24 else v[:-3]
    return {"ret1":ret(1),"ret3":ret(3),"ret6":ret(6),"ret24":ret(24),"rsi":rsi(c),
      "ema_gap":(div(ema(c[-21:],9),ema(c[-21:],21),1)-1)*100,
      "vol_ratio":div(mean(v[-3:]),mean(pv) or mean(v),1),"volatility":sd(rr[-24:]),
      "bb_width":div(4*s,ma,0)*100,"range_pos":div(now-lo,hi-lo,.5)*100,
      "breakout_dist":(div(now,ph,1)-1)*100,"body":div(now-op,hh-ll,0)*100}

def future_gain(rows,i,h=LOOKAHEAD):
    if i+h>=len(rows):return None
    b=float(rows[i]["trade_price"]); mx=max(float(x["high_price"]) for x in rows[i+1:i+h+1])
    return (div(mx,b,1)-1)*100

def recent_pump(d):
    mx=-999
    for i in range(1,len(d)):
        base=min(float(d[i]["opening_price"]),float(d[i-1]["trade_price"]))
        mx=max(mx,(div(float(d[i]["high_price"]),base,1)-1)*100)
    return mx>=EXCLUDE_PUMP_PCT,mx

def current_micro(mk,h1):
    r5=list(reversed(minute(mk,5,120))); r15=list(reversed(minute(mk,15,120))); r4=list(reversed(minute(mk,240,120)))
    f5,f15,f1,f4=features(r5),features(r15),features(h1),features(r4)
    return {"price":float(r5[-1]["trade_price"]),
      "vol5":round(f5["vol_ratio"],2),"vol15":round(f15["vol_ratio"],2),"vol1h":round(f1["vol_ratio"],2),"vol4h":round(f4["vol_ratio"],2),
      "rsi15":round(f15["rsi"],1),"rsi1h":round(f1["rsi"],1),"rsi4h":round(f4["rsi"],1),
      "ret1h":round(f1["ret1"],2),"ret6h":round(f1["ret6"],2),"ret24h":round(f1["ret24"],2),
      "breakout":round(f1["breakout_dist"],2),"ema_gap":round(f1["ema_gap"],2),"range_pos":round(f1["range_pos"],1)}

def btc():
    try:
        h=list(reversed(minute("KRW-BTC",60,100))); f=features(h); c=[float(x["trade_price"]) for x in h]
        sc=sum([f["ret6"]>-1.5,f["ret24"]>-3,ema(c[-30:],20)>=ema(c[-60:],50),f["rsi"]>=42])
        return {"score":sc,"ret6":round(f["ret6"],2),"ret24":round(f["ret24"],2),"rsi1h":round(f["rsi"],1)}
    except:return {"score":2,"ret6":0,"ret24":0,"rsi1h":50}

def stats(vecs):
    o={}
    for k in F:
        a=[x[k] for x in vecs]; med=statistics.median(a); mad=statistics.median([abs(x-med) for x in a]) or sd(a) or 1
        o[k]=(med,mad*1.4826)
    return o
def zv(f,s):return [(f[k]-s[k][0])/(s[k][1] or 1) for k in F]
W=[1,1,1,.8,1.2,1,1.4,.8,.9,1,1.3,.8]
def dist(a,b):return math.sqrt(sum(w*(x-y)**2 for x,y,w in zip(a,b,W)))

def readiness(m,p3,p5,p10,sim,btcscore):
    vol=clamp((m["vol5"]-.8)/1.8,0,1)*12+clamp((m["vol15"]-.8)/1.8,0,1)*10+clamp((m["vol1h"]-.7)/1.6,0,1)*7+clamp((m["vol4h"]-.7)/1.6,0,1)*4
    early=(8 if m["vol5"]>=1.4 and m["vol15"]>=1.15 and m["ret1h"]<2.5 else 0)+(5 if m["vol15"]>m["vol1h"]*1.25 and m["ret1h"]<2 else 0)
    tech=(8 if 45<=m["rsi1h"]<=66 else 4 if 40<=m["rsi1h"]<=70 else 0)+(7 if -3<=m["breakout"]<=1.2 else 0)+(5 if -1<=m["ema_gap"]<=2.5 else 0)+(3 if 45<=m["range_pos"]<=92 else 0)
    hist=clamp(p3*.18+p5*.22+p10*.20,0,28)
    z=vol+early+tech+hist+clamp(sim/100,0,1)*8
    if p5==0 and p10==0:z=min(z,70)
    if m["rsi1h"]>72 or m["rsi4h"]>76:z-=8
    if m["ret6h"]>8:z-=8
    if btcscore<=1:z*=.78
    elif btcscore==2:z*=.90
    return round(clamp(z,0,100),1)

def main():
    regime=btc(); ms=market_list(); candidates=[]; examples=[]; errors=[]
    # First use cheap daily filter, then download 60-day hourly history only for eligible coins.
    eligible=[]
    for m in ms:
        try:
            ex,mx=recent_pump(day(m["market"],35))
            if not ex: eligible.append((m,mx))
        except Exception as e: errors.append(f'{m["market"]} daily: {e}')
    print(f"eligible {len(eligible)}/{len(ms)}")

    for n,(m,mx) in enumerate(eligible,1):
        try:
            h=history_1h(m["market"])
            if len(h)<200: continue
            # Non-overlapping-ish 6-hour snapshots over ~60 days.
            local=[]
            for i in range(30,len(h)-LOOKAHEAD,6):
                f=features(h,i); g=future_gain(h,i)
                if f and g is not None: local.append({"market":m["market"],"f":f,"gain":g})
            examples.extend(local)
            micro=current_micro(m["market"],h)
            liq=mean([float(x["candle_acc_trade_price"]) for x in h[-24:]])
            candidates.append({"m":m,"h":h,"f":features(h),"micro":micro,"max30":mx,"liq":liq})
        except Exception as e: errors.append(f'{m["market"]}: {e}')
        if n%10==0: print(f"{n}/{len(eligible)}; examples={len(examples)}")
    if len(examples)<500 or not candidates: raise RuntimeError("insufficient data")

    s=stats([e["f"] for e in examples]); ez=[zv(e["f"],s) for e in examples]
    out=[]
    for x in candidates:
        z=zv(x["f"],s)
        ds=sorted(((dist(z,q),i) for i,q in enumerate(ez)),key=lambda a:a[0])
        neigh=[]
        # Cross-coin examples first, and use only one snapshot per source coin until needed.
        used=set()
        for d,i in ds:
            e=examples[i]
            if e["market"]==x["m"]["market"] or e["market"] in used: continue
            neigh.append((d,e)); used.add(e["market"])
            if len(neigh)>=K: break
        if len(neigh)<K:
            for d,i in ds:
                e=examples[i]
                if e["market"]==x["m"]["market"]: continue
                if (d,e) not in neigh: neigh.append((d,e))
                if len(neigh)>=K: break
        gains=[e["gain"] for _,e in neigh]
        p3=100*sum(g>=3 for g in gains)/len(gains); p5=100*sum(g>=5 for g in gains)/len(gains); p10=100*sum(g>=10 for g in gains)/len(gains)
        sim=100/(1+mean([d for d,_ in neigh])); m=x["micro"]
        score=readiness(m,p3,p5,p10,sim,regime["score"])
        if x["liq"]<30_000_000:score=round(score*.75,1)
        elif x["liq"]<80_000_000:score=round(score*.9,1)
        early=m["vol5"]>=1.4 and m["vol15"]>=1.15 and m["ret1h"]<2.5
        why=[]
        if early:why.append("가격보다 단기 거래대금이 먼저 증가")
        if p3>=30:why.append(f"60일 유사패턴 +3% {p3:.0f}%")
        if p5>=15:why.append(f"60일 유사패턴 +5% {p5:.0f}%")
        if m["vol5"]>=1.5 and m["vol15"]>=1.2:why.append("5·15분 거래대금 동시 증가")
        if 45<=m["rsi1h"]<=66:why.append("1시간 RSI 과열 전")
        if -3<=m["breakout"]<=1.2:why.append("최근 고점 근처 압축")
        out.append({"market":x["m"]["market"],"name":x["m"]["name"],"score":score,"similarity":round(sim,1),
          "hist_p3":round(p3,1),"hist_p5":round(p5,1),"hist_p10":round(p10,1),"hist_avg_gain6h":round(mean(gains),2),
          "max_30d_pump_pct":round(x["max30"],1),"early_volume_signal":early,"why":why[:4] or ["복합 조건 상위"],**m})
    out.sort(key=lambda q:(q["score"],q["hist_p5"],q["hist_p3"]),reverse=True)

    # Honest walk-forward-like diagnostic: older 80% examples predict newer 20% examples.
    # Limit test size for runtime; this is a diagnostic, not a guarantee.
    split=int(len(examples)*.8); train=examples[:split]; test=examples[split:][-500:]
    ts=stats([e["f"] for e in train]); tz=[zv(e["f"],ts) for e in train]
    preds=[]; actual=[]
    for e in test:
        q=zv(e["f"],ts); near=sorted(((dist(q,z),i) for i,z in enumerate(tz)),key=lambda a:a[0])[:30]
        pp=mean([1 if train[i]["gain"]>=5 else 0 for _,i in near])
        preds.append(pp); actual.append(1 if e["gain"]>=5 else 0)
    # Top-quartile signal precision vs overall +5 base rate.
    if preds:
        cut=sorted(preds)[max(0,int(len(preds)*.75)-1)]
        sel=[a for p,a in zip(preds,actual) if p>=cut]
        bt={"test_samples":len(test),"base_p5":round(mean(actual)*100,1),"signal_p5":round(mean(sel)*100,1) if sel else 0,"signal_samples":len(sel)}
    else: bt={}

    payload={"version":"V4","generated_at_utc":datetime.now(timezone.utc).isoformat(),
      "config":{"days_back":DAYS_BACK,"lookahead_hours":LOOKAHEAD,"neighbors":K,"historical_examples":len(examples),"current_candidates":len(candidates)},
      "btc_regime":regime,"backtest":bt,"results":out[:TOP_N],"errors":errors[:30],
      "disclaimer":"급등준비점수는 확률이 아닙니다. +3/+5/+10은 약 60일 1시간봉의 교차코인 유사패턴이 향후 6시간 내 도달한 관측 비율입니다. 백테스트는 제한된 진단이며 미래 수익을 보장하지 않습니다."}
    OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(payload,ensure_ascii=False,indent=2))

if __name__=="__main__": main()
