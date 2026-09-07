import json, os, statistics, time
from datetime import datetime, timezone, timedelta
from pathlib import Path
import requests

BASE="https://api.upbit.com"
LATEST=Path("docs/data/latest.json")
SIGNALS=Path("docs/data/signals.json")
DELAY=float(os.getenv("REQUEST_DELAY","0.12"))

S=requests.Session()
S.headers.update({"Accept":"application/json","User-Agent":"upbit-v5-live-validator/5.0"})

def mean(xs):
    return statistics.fmean(xs) if xs else 0.0

def api(path,params=None,retries=6):
    last=None
    for n in range(retries):
        try:
            r=S.get(BASE+path,params=params,timeout=20)
            if r.status_code==429:
                time.sleep(1+n)
                continue
            r.raise_for_status()
            time.sleep(DELAY)
            return r.json()
        except Exception as e:
            last=e
            time.sleep(min(2**n,8))
    raise RuntimeError(f"{path}: {last}")

def candles_1h(market,count=40,to=None):
    p={"market":market,"count":min(count,200)}
    if to:
        p["to"]=to
    return api("/v1/candles/minutes/60",p)

def load_json(path,default):
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return default

def iso(s):
    return datetime.fromisoformat(s.replace("Z","+00:00"))

def forward_stats(market,signal_time,entry_price,hours):
    start=iso(signal_time)
    end=start+timedelta(hours=hours)
    to=(end+timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%S")
    rows=candles_1h(market,min(hours+8,200),to)
    usable=[]
    for r in rows:
        dt=datetime.fromisoformat(r["candle_date_time_utc"]).replace(tzinfo=timezone.utc)
        if start < dt <= end:
            usable.append((dt,r))
    if not usable:
        return None
    usable.sort(key=lambda x:x[0])
    hi=max(float(r["high_price"]) for _,r in usable)
    lo=min(float(r["low_price"]) for _,r in usable)
    last=float(usable[-1][1]["trade_price"])
    return {
        "max_gain_pct":round((hi/entry_price-1)*100,2),
        "max_drawdown_pct":round((lo/entry_price-1)*100,2),
        "end_return_pct":round((last/entry_price-1)*100,2)
    }

def summarize(history,hours):
    key=str(hours)
    rows=[r for r in history if key in r.get("outcomes",{})]
    if not rows:
        return {"samples":0}
    def hit(level,subset):
        return round(100*sum(r["outcomes"][key]["max_gain_pct"]>=level for r in subset)/len(subset),1) if subset else 0
    strong=[
        r for r in rows
        if float(r.get("score",0))>=65
        and bool(r.get("early_volume_signal"))
        and float(r.get("hist_p5",0))>=15
    ]
    return {
        "samples":len(rows),
        "p3":hit(3,rows),
        "p5":hit(5,rows),
        "p10":hit(10,rows),
        "avg_max_gain":round(mean([r["outcomes"][key]["max_gain_pct"] for r in rows]),2),
        "avg_max_drawdown":round(mean([r["outcomes"][key]["max_drawdown_pct"] for r in rows]),2),
        "strong_samples":len(strong),
        "strong_p5":hit(5,strong) if strong else None,
        "strong_avg_max_gain":round(mean([r["outcomes"][key]["max_gain_pct"] for r in strong]),2) if strong else None,
    }

def main():
    latest=load_json(LATEST,{})
    if not latest.get("results"):
        raise RuntimeError("latest.json has no V4 results")

    history=load_json(SIGNALS,[])
    if not isinstance(history,list):
        history=[]

    now=datetime.now(timezone.utc)

    # Evaluate past recommendations once the requested horizon has elapsed.
    for rec in history:
        try:
            age=(now-iso(rec["signal_time"])).total_seconds()/3600
            rec.setdefault("outcomes",{})
            for h in (1,3,6,12,24):
                k=str(h)
                if age>=h and k not in rec["outcomes"]:
                    st=forward_stats(rec["market"],rec["signal_time"],float(rec["entry_price"]),h)
                    if st:
                        rec["outcomes"][k]=st
        except Exception as e:
            rec["tracking_error"]=str(e)[:180]

    # Add this run's TOP10 only once.
    run_id=latest.get("generated_at_utc")
    existing={(r.get("signal_time"),r.get("market")) for r in history}
    for rank,x in enumerate(latest["results"][:10],1):
        key=(run_id,x["market"])
        if key in existing:
            continue
        history.append({
            "signal_time":run_id,
            "market":x["market"],
            "name":x.get("name",x["market"]),
            "rank":rank,
            "entry_price":x["price"],
            "score":x["score"],
            "similarity":x.get("similarity"),
            "hist_p3":x.get("hist_p3",0),
            "hist_p5":x.get("hist_p5",0),
            "hist_p10":x.get("hist_p10",0),
            "vol5":x.get("vol5"),
            "vol15":x.get("vol15"),
            "vol1h":x.get("vol1h"),
            "early_volume_signal":x.get("early_volume_signal",False),
            "outcomes":{}
        })

    # Keep about 45 days of hourly signals.
    cutoff=now-timedelta(days=45)
    history=[r for r in history if iso(r["signal_time"])>=cutoff]

    live={str(h):summarize(history,h) for h in (1,3,6,12,24)}
    latest["version"]="V5"
    latest["live_validation"]=live
    latest["signal_history_count"]=len(history)
    latest["strong_signal_rule"]="score>=65 AND early_volume_signal AND hist_p5>=15"
    latest["disclaimer"]=(
        "급등준비점수는 확률이 아닙니다. 과거 유사패턴 수치도 미래 확률이 아닙니다. "
        "V5 실전검증은 추천이 나온 뒤 실제 1시간봉의 최고가/최저가를 추적하는 out-of-sample 기록입니다. "
        "표본이 적을 때는 수치를 과신하면 안 됩니다."
    )

    SIGNALS.parent.mkdir(parents=True,exist_ok=True)
    SIGNALS.write_text(json.dumps(history,ensure_ascii=False,indent=2),encoding="utf-8")
    LATEST.write_text(json.dumps(latest,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"version":"V5","signals":len(history),"live_validation":live},ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
