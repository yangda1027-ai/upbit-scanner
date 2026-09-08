import json, time, statistics
from datetime import datetime, timezone, timedelta
from pathlib import Path
import requests

LATEST=Path("docs/data/latest.json")
SIGNALS=Path("docs/data/signals.json")
S=requests.Session()
S.headers.update({"Accept":"application/json","User-Agent":"upbit-v6-live-validator/6.0"})

def mean(x): return statistics.fmean(x) if x else 0
def iso(s): return datetime.fromisoformat(s.replace("Z","+00:00"))
def load(p,d):
    try:return json.loads(p.read_text(encoding="utf-8")) if p.exists() else d
    except:return d
def api(path,params):
    for n in range(6):
        r=S.get("https://api.upbit.com"+path,params=params,timeout=20)
        if r.status_code==429: time.sleep(1+n); continue
        r.raise_for_status(); time.sleep(.12); return r.json()
    return []
def fwd(market,t,entry,h):
    st=iso(t); end=st+timedelta(hours=h); to=(end+timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%S")
    rows=api("/v1/candles/minutes/60",{"market":market,"count":min(h+8,200),"to":to})
    use=[]
    for r in rows:
        dt=datetime.fromisoformat(r["candle_date_time_utc"]).replace(tzinfo=timezone.utc)
        if st<dt<=end:use.append((dt,r))
    if not use:return None
    use.sort(key=lambda x:x[0]); hi=max(float(r["high_price"]) for _,r in use); lo=min(float(r["low_price"]) for _,r in use); last=float(use[-1][1]["trade_price"])
    return {"max_gain_pct":round((hi/entry-1)*100,2),"max_drawdown_pct":round((lo/entry-1)*100,2),"end_return_pct":round((last/entry-1)*100,2)}
def summary(hist,h,mode):
    k=str(h); rows=[r for r in hist if r.get("mode")==mode and k in r.get("outcomes",{})]
    if not rows:return {"samples":0}
    def hit(v):return round(100*sum(r["outcomes"][k]["max_gain_pct"]>=v for r in rows)/len(rows),1)
    return {"samples":len(rows),"p3":hit(3),"p5":hit(5),"p10":hit(10),"avg_max_gain":round(mean([r["outcomes"][k]["max_gain_pct"] for r in rows]),2),"avg_max_drawdown":round(mean([r["outcomes"][k]["max_drawdown_pct"] for r in rows]),2)}
def main():
    d=load(LATEST,{})
    hist=load(SIGNALS,[])
    now=datetime.now(timezone.utc)
    for r in hist:
        age=(now-iso(r["signal_time"])).total_seconds()/3600
        r.setdefault("outcomes",{})
        for h in (1,3,6,12,24):
            if age>=h and str(h) not in r["outcomes"]:
                q=fwd(r["market"],r["signal_time"],float(r["entry_price"]),h)
                if q:r["outcomes"][str(h)]=q
    run=d["generated_at_utc"]; seen={(r.get("signal_time"),r.get("market"),r.get("mode")) for r in hist}
    for mode,key in [("latent","latent_results"),("momentum","momentum_results")]:
        for rank,x in enumerate(d.get(key,[])[:5],1):
            z=(run,x["market"],mode)
            if z not in seen:
                hist.append({"signal_time":run,"market":x["market"],"name":x["name"],"mode":mode,"rank":rank,"entry_price":x["price"],"score":x["score"],"hist_p5":x.get("hist_p5",0),"outcomes":{}})
    cutoff=now-timedelta(days=45)
    hist=[r for r in hist if iso(r["signal_time"])>=cutoff]
    d["live_validation"]={"latent":{str(h):summary(hist,h,"latent") for h in (1,3,6,12,24)},"momentum":{str(h):summary(hist,h,"momentum") for h in (1,3,6,12,24)}}
    d["signal_history_count"]=len(hist)
    SIGNALS.write_text(json.dumps(hist,ensure_ascii=False,indent=2),encoding="utf-8")
    LATEST.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding="utf-8")
if __name__=="__main__":main()
