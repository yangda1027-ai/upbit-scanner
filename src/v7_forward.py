import json, math, os, time
from datetime import datetime, timezone
from pathlib import Path
import requests

API="https://api.upbit.com"
OUT=Path("docs/data/v7_signals.json")
LATEST=Path("docs/data/v7_latest.json")
STABLE={"USDT","USDC","DAI"}
TOP_N=10
MAX_ROWS=12000

S=requests.Session()
S.headers.update({"User-Agent":"upbit-v7-forward/1.0"})

def get(path, params=None, tries=4):
    last=None
    for i in range(tries):
        try:
            r=S.get(API+path, params=params, timeout=15)
            if r.status_code==429:
                time.sleep(1.2*(i+1)); continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last=e; time.sleep(.7*(i+1))
    raise last

def f(x, d=0.0):
    try: return float(x)
    except: return d

def pct(a,b):
    return (a/b-1)*100 if b else 0.0

def markets():
    xs=get("/v1/market/all", {"is_details":"false"})
    return [x["market"] for x in xs if x["market"].startswith("KRW-") and x["market"].split("-",1)[1] not in STABLE]

def candles(m, unit, count):
    return list(reversed(get(f"/v1/candles/minutes/{unit}", {"market":m,"count":count})))

def score_market(m):
    c5=candles(m,5,24)
    if len(c5)<20: return None
    closes=[f(x["trade_price"]) for x in c5]
    vals=[f(x["candle_acc_trade_price"]) for x in c5]
    cur=closes[-1]
    r5=pct(closes[-1],closes[-2])
    r15=pct(closes[-1],closes[-4])
    r60=pct(closes[-1],closes[-13])
    v5=vals[-1]/(sum(vals[-7:-1])/6 or 1)
    v15=sum(vals[-3:])/(sum(vals[-9:-3])/2 or 1)
    accel=(sum(vals[-3:])/3)/(sum(vals[-9:-3])/6 or 1)
    # Momentum score deliberately simple and frozen for forward collection.
    score=50
    score += max(-15,min(15,r15*3.0))
    score += max(-10,min(10,r60*1.2))
    score += max(-10,min(10,(v5-1)*8))
    score += max(-10,min(10,(accel-1)*10))
    score=max(0,min(100,score))
    return {
        "market":m,"price":cur,"score":round(score,2),
        "ret_5m":round(r5,3),"ret_15m":round(r15,3),"ret_60m":round(r60,3),
        "value_ratio_5m":round(v5,3),"value_accel_15m":round(accel,3),
        "value_5m_krw":round(vals[-1],0)
    }

def btc_state():
    c=candles("KRW-BTC",5,24)
    cl=[f(x["trade_price"]) for x in c]
    return {
        "ret_15m":round(pct(cl[-1],cl[-4]),3),
        "ret_60m":round(pct(cl[-1],cl[-13]),3)
    }

def main():
    now=datetime.now(timezone.utc).replace(second=0,microsecond=0)
    ms=markets()
    rows=[]
    # ticker prefilter keeps API load reasonable and uses only contemporaneous public data
    tick=[]
    for i in range(0,len(ms),100):
        tick += get("/v1/ticker", {"markets":",".join(ms[i:i+100])})
        time.sleep(.12)
    liquid=sorted(tick,key=lambda x:f(x.get("acc_trade_price_24h")),reverse=True)[:120]
    for i,x in enumerate(liquid):
        try:
            z=score_market(x["market"])
            if z: rows.append(z)
        except Exception as e:
            print("skip",x["market"],e)
        time.sleep(.08)
    rows.sort(key=lambda x:(x["score"],x["value_accel_15m"],x["value_ratio_5m"]),reverse=True)
    top=rows[:TOP_N]
    state=btc_state()
    recs=[]
    for rank,z in enumerate(top,1):
        q=dict(z)
        q.update({"ts":now.isoformat(),"rank":rank,"btc":state,
                  "h1":None,"h3":None,"h6":None,"h24":None})
        recs.append(q)
    hist=[]
    if OUT.exists():
        try: hist=json.loads(OUT.read_text(encoding="utf-8"))
        except: hist=[]
    # immutable-style append: never rewrite signal features/prices; evaluator only fills future outcomes
    hist.extend(recs)
    hist=hist[-MAX_ROWS:]
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(hist,ensure_ascii=False,indent=2),encoding="utf-8")
    LATEST.write_text(json.dumps({"updated_at":now.isoformat(),"btc":state,"top":top},ensure_ascii=False,indent=2),encoding="utf-8")
    print("saved",len(recs),"signals; total",len(hist))

if __name__=="__main__":
    main()
