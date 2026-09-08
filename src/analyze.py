import json, math, os, statistics, time
from datetime import datetime, timezone, timedelta
from pathlib import Path
import requests

BASE="https://api.upbit.com"
OUT=Path("docs/data/latest.json")
DAYS_BACK=int(os.getenv("DAYS_BACK","60"))
EXCLUDE_PUMP_PCT=float(os.getenv("EXCLUDE_PUMP_PCT","20"))
LOOKAHEAD=int(os.getenv("LOOKAHEAD_HOURS","6"))
K=int(os.getenv("K_NEIGHBORS","50"))
TOP_A=int(os.getenv("TOP_A","5"))
TOP_B=int(os.getenv("TOP_B","5"))
STABLE={"USDT","USDC","DAI","TUSD","FDUSD","USDE","PYUSD","USDS","USD1","RLUSD","BUSD","USTC"}
S=requests.Session()
S.headers.update({"Accept":"application/json","User-Agent":"upbit-v6-dual-scanner/6.0"})

def api(path,params=None,retries=6):
    last=None
    for n in range(retries):
        try:
            r=S.get(BASE+path,params=params,timeout=20)
            if r.status_code==429:
                time.sleep(1+n); continue
            r.raise_for_status(); time.sleep(.11); return r.json()
        except Exception as e:
            last=e; time.sleep(min(2**n,8))
    raise RuntimeError(f"{path}: {last}")

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

def markets():
    out=[]
    for x in api("/v1/market/all",{"is_details":"true"}):
        mk=x["market"]
        if not mk.startswith("KRW-"): continue
        sym=mk.split("-",1)[1].upper()
        if sym in STABLE: continue
        ev=x.get("market_event") or {}
        if ev.get("warning") is True: continue
        out.append({"market":mk,"name":x.get("korean_name",sym)})
    return out

def minute(mk,u,count=200,to=None):
    p={"market":mk,"count":min(count,200)}
    if to:p["to"]=to
    return api(f"/v1/candles/minutes/{u}",p)

def day(mk,count=35):
    return list(reversed(api("/v1/candles/days",{"market":mk,"count":count})))

def history_1h(mk):
    need=DAYS_BACK*24; allrows=[]; to=None
    while len(allrows)<need:
        cnt=min(200,need-len(allrows))
        rows=minute(mk,60,cnt,to)
        if not rows: break
        allrows.extend(rows)
        oldest=rows[-1]["candle_date_time_utc"]
        dt=datetime.fromisoformat(oldest).replace(tzinfo=timezone.utc)-timedelta(seconds=1)
        to=dt.strftime("%Y-%m-%dT%H:%M:%S")
        if len(rows)<cnt: break
    uniq={r["candle_date_time_utc"]:r for r in allrows}
    return [uniq[k] for k in sorted(uniq)]

F=["ret1","ret3","ret6","ret24","rsi","ema_gap","vol_ratio","volatility","range_pos","breakout_dist"]
def features(rows,i=None):
    if i is None:i=len(rows)-1
    if i<30:return None
    w=rows[max(0,i-40):i+1]
    c=[float(x["trade_price"]) for x in w]; h=[float(x["high_price"]) for x in w]
    l=[float(x["low_price"]) for x in w]; v=[float(x["candle_acc_trade_price"]) for x in w]
    now=c[-1]
    def ret(n): return (div(now,c[-1-n],1)-1)*100 if len(c)>n else 0
    rr=[(div(c[j],c[j-1],1)-1)*100 for j in range(1,len(c))]
    hi=max(h[-20:]); lo=min(l[-20:]); ph=max(h[-21:-1]); pv=v[-24:-3] if len(v)>=24 else v[:-3]
    return {
      "ret1":ret(1),"ret3":ret(3),"ret6":ret(6),"ret24":ret(24),"rsi":rsi(c),
      "ema_gap":(div(ema(c[-21:],9),ema(c[-21:],21),1)-1)*100,
      "vol_ratio":div(mean(v[-3:]),mean(pv) or mean(v),1),
      "volatility":sd(rr[-24:]),"range_pos":div(now-lo,hi-lo,.5)*100,
      "breakout_dist":(div(now,ph,1)-1)*100
    }

def future_gain(rows,i):
    if i+LOOKAHEAD>=len(rows):return None
    b=float(rows[i]["trade_price"]); hi=max(float(x["high_price"]) for x in rows[i+1:i+1+LOOKAHEAD])
    return (div(hi,b,1)-1)*100

def pump_stats(d):
    mx=-999; recent=False
    for i in range(1,len(d)):
        base=min(float(d[i]["opening_price"]),float(d[i-1]["trade_price"]))
        g=(div(float(d[i]["high_price"]),base,1)-1)*100
        mx=max(mx,g)
        if i>=len(d)-7 and g>=EXCLUDE_PUMP_PCT: recent=True
    return mx,recent

def micro(mk,h1):
    r5=list(reversed(minute(mk,5,120))); r15=list(reversed(minute(mk,15,120))); r4=list(reversed(minute(mk,240,120)))
    f5,f15,f1,f4=features(r5),features(r15),features(h1),features(r4)
    hi24=max(float(x["high_price"]) for x in h1[-24:]); cur=float(r5[-1]["trade_price"])
    c=[float(x["trade_price"]) for x in h1]
    return {"price":cur,"vol5":round(f5["vol_ratio"],2),"vol15":round(f15["vol_ratio"],2),"vol1h":round(f1["vol_ratio"],2),"vol4h":round(f4["vol_ratio"],2),
      "rsi15":round(f15["rsi"],1),"rsi1h":round(f1["rsi"],1),"rsi4h":round(f4["rsi"],1),
      "ret1h":round(f1["ret1"],2),"ret3h":round(f1["ret3"],2),"ret6h":round(f1["ret6"],2),"ret24h":round(f1["ret24"],2),
      "breakout":round(f1["breakout_dist"],2),"ema_gap":round(f1["ema_gap"],2),"range_pos":round(f1["range_pos"],1),
      "dist_24h_high":round((div(cur,hi24,1)-1)*100,2),"ema20_over_50":ema(c[-30:],20)>=ema(c[-60:],50)}

def btc_regime():
    try:
        h=list(reversed(minute("KRW-BTC",60,100))); f=features(h); c=[float(x["trade_price"]) for x in h]
        s=sum([f["ret6"]>-1.5,f["ret24"]>-3,ema(c[-30:],20)>=ema(c[-60:],50),f["rsi"]>=42])
        return {"score":s,"ret6":round(f["ret6"],2),"ret24":round(f["ret24"],2),"rsi1h":round(f["rsi"],1)}
    except:return {"score":2,"ret6":0,"ret24":0,"rsi1h":50}

def robust_stats(vecs):
    o={}
    for k in F:
        a=[x[k] for x in vecs]; med=statistics.median(a); mad=statistics.median([abs(x-med) for x in a]) or sd(a) or 1
        o[k]=(med,mad*1.4826)
    return o
def zv(f,s): return [(f[k]-s[k][0])/(s[k][1] or 1) for k in F]
W=[1,1,1,.8,1.2,1,1.4,.8,1,1.3]
def dist(a,b): return math.sqrt(sum(w*(x-y)**2 for x,y,w in zip(a,b,W)))

def hist_metrics(x,examples,ez,s):
    z=zv(x["f"],s)
    ds=sorted(((dist(z,q),i) for i,q in enumerate(ez)),key=lambda a:a[0])
    neigh=[]; used=set()
    for d,i in ds:
        e=examples[i]
        if e["market"]==x["m"]["market"] or e["market"] in used: continue
        neigh.append((d,e)); used.add(e["market"])
        if len(neigh)>=K: break
    if len(neigh)<K:
        for d,i in ds:
            e=examples[i]
            if e["market"]==x["m"]["market"]: continue
            neigh.append((d,e))
            if len(neigh)>=K: break
    gains=[e["gain"] for _,e in neigh]
    p3=100*sum(g>=3 for g in gains)/len(gains); p5=100*sum(g>=5 for g in gains)/len(gains); p10=100*sum(g>=10 for g in gains)/len(gains)
    sim=100/(1+mean([d for d,_ in neigh]))
    return p3,p5,p10,sim,mean(gains)

def score_a(m,p3,p5,p10,sim,btc):
    s=0
    s+=clamp((m["vol5"]-.8)/1.8,0,1)*12+clamp((m["vol15"]-.8)/1.8,0,1)*10+clamp((m["vol1h"]-.7)/1.6,0,1)*7
    if m["vol5"]>=1.4 and m["vol15"]>=1.15 and m["ret1h"]<2.5:s+=10
    if 45<=m["rsi1h"]<=66:s+=8
    if -3<=m["breakout"]<=1.2:s+=7
    if -1<=m["ema_gap"]<=2.5:s+=5
    s+=clamp(p3*.18+p5*.22+p10*.20,0,28)+clamp(sim/100,0,1)*8
    if m["rsi1h"]>72:s-=8
    if m["ret6h"]>8:s-=8
    if btc<=1:s*=.78
    elif btc==2:s*=.9
    return round(clamp(s,0,100),1)

def score_b(m,p3,p5,p10,sim,btc,recent):
    s=0
    s+=clamp((m["vol5"]-1)/3,0,1)*14+clamp((m["vol15"]-1)/2.5,0,1)*12+clamp((m["vol1h"]-.8)/2.2,0,1)*8
    if m["ret1h"]>0:s+=8
    if 1<=m["ret3h"]<=12:s+=8
    elif m["ret3h"]>0:s+=3
    if 3<=m["ret6h"]<=20:s+=7
    elif m["ret6h"]>0:s+=3
    if -4<=m["dist_24h_high"]<=1.5:s+=8
    if m["ema20_over_50"]:s+=6
    if 50<=m["rsi1h"]<=74:s+=6
    s+=clamp(p3*.1+p5*.16+p10*.14,0,20)+clamp(sim/100,0,1)*5
    if recent:s+=6
    if m["rsi1h"]>82 or m["rsi15"]>88:s-=10
    if m["ret1h"]>15:s-=10
    if m["ret6h"]>35:s-=12
    if m["dist_24h_high"]<-10:s-=8
    if btc<=1:s*=.82
    elif btc==2:s*=.92
    return round(clamp(s,0,100),1)

def main():
    btc=btc_regime(); examples=[]; current=[]; errors=[]
    ms=markets()
    for n,m in enumerate(ms,1):
        try:
            d=day(m["market"],35); mx,recent=pump_stats(d)
            h=history_1h(m["market"])
            if len(h)<200: continue
            for i in range(30,len(h)-LOOKAHEAD,6):
                f=features(h,i); g=future_gain(h,i)
                if f and g is not None: examples.append({"market":m["market"],"f":f,"gain":g})
            mm=micro(m["market"],h)
            liq=mean([float(x["candle_acc_trade_price"]) for x in h[-24:]])
            current.append({"m":m,"f":features(h),"micro":mm,"max30":mx,"recent":recent,"liq":liq})
        except Exception as e:
            errors.append(f'{m["market"]}: {e}')
        if n%10==0: print(f"{n}/{len(ms)} examples={len(examples)}")
    s=robust_stats([e["f"] for e in examples]); ez=[zv(e["f"],s) for e in examples]
    A=[]; B=[]
    for x in current:
        p3,p5,p10,sim,avg=hist_metrics(x,examples,ez,s); m=x["micro"]
        common={"market":x["m"]["market"],"name":x["m"]["name"],"price":m["price"],"similarity":round(sim,1),
                "hist_p3":round(p3,1),"hist_p5":round(p5,1),"hist_p10":round(p10,1),"hist_avg_gain6h":round(avg,2),
                "max_30d_pump_pct":round(x["max30"],1),"recent_big_pump":x["recent"],**m}
        if x["max30"]<EXCLUDE_PUMP_PCT:
            sc=score_a(m,p3,p5,p10,sim,btc["score"])
            if x["liq"]<30_000_000:sc=round(sc*.75,1)
            why=[]
            if m["vol5"]>=1.4 and m["vol15"]>=1.15 and m["ret1h"]<2.5:why.append("거래대금 선행")
            if p5>=15:why.append(f"유사 +5% {p5:.0f}%")
            if -3<=m["breakout"]<=1.2:why.append("고점 근처 압축")
            A.append({**common,"score":sc,"why":why[:4] or ["복합 조건 상위"],"mode":"latent"})
        sc=score_b(m,p3,p5,p10,sim,btc["score"],x["recent"])
        if x["liq"]<30_000_000:sc=round(sc*.78,1)
        why=[]
        if x["recent"]:why.append("최근 급등 이력")
        if m["vol5"]>=1.8 and m["vol15"]>=1.4:why.append("단기 거래대금 재확대")
        if -4<=m["dist_24h_high"]<=1.5:why.append("24h 고점 재접근")
        if 1<=m["ret3h"]<=12:why.append("3시간 모멘텀 유지")
        if m["ema20_over_50"]:why.append("EMA20>EMA50")
        B.append({**common,"score":sc,"why":why[:4] or ["모멘텀 복합 조건"],"mode":"momentum"})
    A.sort(key=lambda q:(q["score"],q["hist_p5"]),reverse=True)
    B.sort(key=lambda q:(q["score"],q["ret3h"],q["vol15"]),reverse=True)
    payload={"version":"V6","generated_at_utc":datetime.now(timezone.utc).isoformat(),
             "config":{"days_back":DAYS_BACK,"historical_examples":len(examples),"markets":len(current)},
             "btc_regime":btc,"latent_results":A[:TOP_A],"momentum_results":B[:TOP_B],
             "errors":errors[:30],"disclaimer":"A형은 급등 전 잠복형, B형은 최근 급등 이력을 허용하는 재급등/모멘텀형입니다. 점수는 확률이 아닙니다."}
    OUT.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")

if __name__=="__main__": main()
