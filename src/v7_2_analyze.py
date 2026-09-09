import json, math
from datetime import datetime
from pathlib import Path

SIGNALS = Path("docs/data/v7_signals.json")
RISKS = Path("docs/data/v7_1_market_history.json")
OUT = Path("docs/data/v7_2_risk_analysis.json")

SCORE_THRESHOLDS = [70, 75, 80]
RISK_BUCKETS = [(0,20),(20,40),(40,60),(60,80),(80,101)]
HORIZONS = ["h1","h3","h6","h24"]
MAX_MATCH_MINUTES = 10

def dt(s):
    return datetime.fromisoformat(s.replace("Z","+00:00"))

def pf(returns):
    gp = sum(x for x in returns if x > 0)
    gl = -sum(x for x in returns if x < 0)
    if gl == 0:
        return None if gp == 0 else 999.0
    return gp / gl

def stats(rows, horizon):
    vals=[]
    hit5=0
    stop3=0
    for r in rows:
        o=r.get(horizon)
        if not o:
            continue
        vals.append(float(o.get("close_pct",0)))
        if o.get("hit_5"): hit5 += 1
        if float(o.get("min_pct",0)) <= -3: stop3 += 1
    n=len(vals)
    if not n:
        return {"n":0,"avg_close":None,"median_close":None,"hit5":None,"stop3":None,"pf":None}
    s=sorted(vals)
    med=s[n//2] if n%2 else (s[n//2-1]+s[n//2])/2
    return {
        "n":n,
        "avg_close":round(sum(vals)/n,4),
        "median_close":round(med,4),
        "hit5":round(hit5/n*100,2),
        "stop3":round(stop3/n*100,2),
        "pf":None if pf(vals) is None else round(pf(vals),3)
    }

def nearest_risk(ts, risk_rows):
    t=dt(ts).timestamp()
    best=None; bd=10**9
    for rr in risk_rows:
        if not rr.get("ts") or rr.get("risk") is None:
            continue
        d=abs(dt(rr["ts"]).timestamp()-t)
        if d<bd:
            bd=d; best=rr
    if best is None or bd > MAX_MATCH_MINUTES*60:
        return None
    return best

def main():
    sig=json.loads(SIGNALS.read_text(encoding="utf-8")) if SIGNALS.exists() else []
    risks=json.loads(RISKS.read_text(encoding="utf-8")) if RISKS.exists() else []
    joined=[]
    for r in sig:
        rr=nearest_risk(r["ts"], risks)
        if rr is None: 
            continue
        q=dict(r)
        q["market_risk"]=float(rr["risk"])
        q["market_state"]=rr.get("state")
        q["market_ts"]=rr.get("ts")
        joined.append(q)

    result={
        "version":"V7.2 Forward Risk Analysis",
        "joined_signals":len(joined),
        "source_signals":len(sig),
        "source_risk_snapshots":len(risks),
        "rules":{
            "score_thresholds":SCORE_THRESHOLDS,
            "risk_buckets":[f"{a}-{b-1}" for a,b in RISK_BUCKETS],
            "max_risk_match_minutes":MAX_MATCH_MINUTES,
            "note":"V7/V7.1의 실제 미래 기록만 분석. 과거 데이터 재구성 없음."
        },
        "rows":[]
    }

    for th in SCORE_THRESHOLDS:
        eligible=[r for r in joined if float(r.get("score",0))>=th]
        # ALL
        row={"score_threshold":th,"risk_bucket":"ALL"}
        for h in HORIZONS:
            row[h]=stats(eligible,h)
        result["rows"].append(row)
        for a,b in RISK_BUCKETS:
            sub=[r for r in eligible if a <= r["market_risk"] < b]
            row={"score_threshold":th,"risk_bucket":f"{a}-{b-1}"}
            for h in HORIZONS:
                row[h]=stats(sub,h)
            result["rows"].append(row)

    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print("joined",len(joined),"of",len(sig),"signals")

if __name__=="__main__":
    main()
