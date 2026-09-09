import json, time, statistics
from datetime import datetime, timezone
from pathlib import Path
import requests

API="https://api.upbit.com"
OUT=Path("docs/data/v7_1_market_latest.json")
HIST=Path("docs/data/v7_1_market_history.json")
STABLE={"USDT","USDC","DAI"}
S=requests.Session()
S.headers.update({"User-Agent":"upbit-v7-1-market-risk/1.0"})

def get(path,params=None):
    for i in range(4):
        try:
            r=S.get(API+path,params=params,timeout=15)
            if r.status_code==429:
                time.sleep(1+i); continue
            r.raise_for_status(); return r.json()
        except Exception:
            if i==3: raise
            time.sleep(.6*(i+1))

def f(x,d=0.0):
    try:return float(x)
    except:return d

def pct(a,b): return (a/b-1)*100 if b else 0.0
def clamp(x,a,b): return max(a,min(b,x))

def markets():
    xs=get("/v1/market/all",{"is_details":"false"})
    return [x["market"] for x in xs if x["market"].startswith("KRW-") and x["market"].split("-",1)[1] not in STABLE]

def feat(m):
    c=list(reversed(get("/v1/candles/minutes/5",{"market":m,"count":30})))
    if len(c)<25:return None
    cl=[f(x["trade_price"]) for x in c]
    tv=[f(x["candle_acc_trade_price"]) for x in c]
    absr=[abs(pct(cl[i],cl[i-1])) for i in range(1,len(cl))]
    prev=statistics.mean(absr[-24:-6]) if absr[-24:-6] else .0001
    recent=statistics.mean(absr[-6:]) if absr[-6:] else 0
    return {"market":m,"ret5":pct(cl[-1],cl[-2]),"ret15":pct(cl[-1],cl[-4]),
            "ret60":pct(cl[-1],cl[-13]),"vol_ratio":recent/(prev or .0001),
            "value_accel":(sum(tv[-3:])/3)/(sum(tv[-9:-3])/6 or 1)}

def calc(btc,alts):
    risk=0.0; reasons=[]
    if btc["ret5"]<-0.25:
        risk+=clamp((-btc["ret5"]-.25)*12,0,22); reasons.append(f"BTC 5m shock {btc['ret5']:.2f}%")
    if btc["ret15"]<-0.75:
        risk+=clamp((-btc["ret15"]-.75)*7,0,16); reasons.append(f"BTC 15m weakness {btc['ret15']:.2f}%")
    if btc["vol_ratio"]>1.6:
        risk+=clamp((btc["vol_ratio"]-1.6)*8,0,14); reasons.append(f"BTC volatility x{btc['vol_ratio']:.2f}")

    n=max(1,len(alts))
    b15=sum(x["ret15"]>0 for x in alts)/n*100
    b60=sum(x["ret60"]>0 for x in alts)/n*100
    med15=statistics.median([x["ret15"] for x in alts]) if alts else 0
    med60=statistics.median([x["ret60"] for x in alts]) if alts else 0
    crash=sum(x["ret15"]<=-2 for x in alts)/n*100
    flow=sum(x["value_accel"]>=1.2 for x in alts)/n*100

    if b15<45: risk+=clamp((45-b15)*.65,0,20); reasons.append(f"15m breadth {b15:.0f}%")
    if b60<40: risk+=clamp((40-b60)*.40,0,12); reasons.append(f"60m breadth {b60:.0f}%")
    if med15<-.35: risk+=clamp((-med15-.35)*8,0,12); reasons.append(f"alt median15 {med15:.2f}%")
    if crash>12: risk+=clamp((crash-12)*.75,0,14); reasons.append(f"15m crash share {crash:.0f}%")
    if flow<25: risk+=clamp((25-flow)*.40,0,10); reasons.append(f"value diffusion {flow:.0f}%")

    emergency=(btc["ret5"]<=-1.5) or (btc["ret15"]<=-2.2) or (b15<=18 and crash>=25)
    if emergency:
        risk=max(risk,85); reasons.insert(0,"EMERGENCY")
    risk=round(clamp(risk,0,100),1)
    state="RED" if risk>=70 else ("YELLOW" if risk>=40 else "GREEN")
    return {"risk":risk,"state":state,"emergency":emergency,"breadth15":round(b15,1),
            "breadth60":round(b60,1),"median15":round(med15,3),"median60":round(med60,3),
            "crash15_share":round(crash,1),"value_diffusion":round(flow,1),"reasons":reasons[:6]}

def main():
    now=datetime.now(timezone.utc).replace(second=0,microsecond=0)
    ms=markets(); tick=[]
    for i in range(0,len(ms),100):
        tick+=get("/v1/ticker",{"markets":",".join(ms[i:i+100])}); time.sleep(.12)
    liquid=sorted(tick,key=lambda x:f(x.get("acc_trade_price_24h")),reverse=True)[:120]
    alts=[]; btc=None
    for x in liquid:
        try:
            z=feat(x["market"])
            if not z: continue
            if x["market"]=="KRW-BTC": btc=z
            else: alts.append(z)
        except Exception as e: print("skip",x["market"],e)
        time.sleep(.07)
    if btc is None: btc=feat("KRW-BTC")
    obj={"ts":now.isoformat(),"btc":btc,"sample_size":len(alts),**calc(btc,alts)}
    OUT.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding="utf-8")
    hist=[]
    if HIST.exists():
        try: hist=json.loads(HIST.read_text(encoding="utf-8"))
        except: pass
    hist=(hist+[obj])[-5000:]
    HIST.write_text(json.dumps(hist,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(obj,ensure_ascii=False))

if __name__=="__main__": main()
