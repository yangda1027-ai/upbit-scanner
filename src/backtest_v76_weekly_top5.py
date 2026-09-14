#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json, os, time
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
import requests

API = "https://api.upbit.com"
KST = timezone(timedelta(hours=9))
UTC = timezone.utc

START = datetime.strptime(os.getenv("BT_START_KST", "2026-03-02 09:00"), "%Y-%m-%d %H:%M").replace(tzinfo=KST)
END = datetime.strptime(os.getenv("BT_END_KST", "2026-09-14 09:00"), "%Y-%m-%d %H:%M").replace(tzinfo=KST)
WEEKDAY = int(os.getenv("BT_WEEKDAY", "0"))  # Monday=0
HOUR_KST = int(os.getenv("BT_HOUR_KST", "9"))
TOP_K = int(os.getenv("BT_TOP_K", "5"))
DELAY = float(os.getenv("BT_REQUEST_DELAY", "0.12"))
INITIAL_CAPITAL = float(os.getenv("BT_INITIAL_CAPITAL", "15000000"))
TARGET_PCT = float(os.getenv("BT_TARGET_PCT", "20"))
STOP_PCTS = tuple(float(x) for x in os.getenv("BT_STOP_PCTS", "3,5,7,10").split(","))
OUT = Path("docs/data/v76_weekly_top5_backtest_latest.json")
CACHE = Path(".cache/v76_weekly_top5")
OUT.parent.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)

STABLE={"USDT","USDC","DAI","USD1","USDE","FDUSD","TUSD","RLUSD","EURC"}
S=requests.Session()
S.headers.update({"User-Agent":"upbit-v76-weekly-top5-backtest/1.0"})


def get(path, params=None, tries=7):
    last=None
    for i in range(tries):
        try:
            r=S.get(API+path,params=params,timeout=25)
            if r.status_code==429:
                time.sleep(0.8*(i+1)); continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last=e
            time.sleep(0.8*(i+1))
    raise last


def iso(dt): return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
def pct(a,b): return (a/b-1)*100 if b else 0.0
def clamp(x,a,b): return max(a,min(b,x))


def markets():
    xs=get("/v1/market/all", {"is_details":"false"})
    return sorted(x["market"] for x in xs if x["market"].startswith("KRW-") and x["market"].split("-",1)[1] not in STABLE)


def dt_utc(s):
    return datetime.fromisoformat(s).replace(tzinfo=UTC)


def fetch_5m_before(m, at_utc, count=30):
    key=f"pre_{m.replace('-','_')}_{at_utc.strftime('%Y%m%dT%H%M')}.json"
    p=CACHE/key
    if p.exists():
        try:return json.loads(p.read_text(encoding="utf-8"))
        except:pass
    # Upbit 'to' is exclusive-ish around boundary; +1 sec makes the decision candle retrievable.
    rows=get("/v1/candles/minutes/5", {"market":m,"count":count,"to":iso(at_utc+timedelta(seconds=1))})
    rows=sorted(rows,key=lambda x:x["candle_date_time_utc"])
    p.write_text(json.dumps(rows,ensure_ascii=False),encoding="utf-8")
    time.sleep(DELAY)
    return rows


def fetch_5m_range(m, start_utc, end_utc):
    key=f"path_{m.replace('-','_')}_{start_utc.strftime('%Y%m%dT%H%M')}_{end_utc.strftime('%Y%m%dT%H%M')}.json"
    p=CACHE/key
    if p.exists():
        try:return json.loads(p.read_text(encoding="utf-8"))
        except:pass
    out=[]; seen=set(); to=end_utc+timedelta(seconds=1)
    while True:
        rows=get("/v1/candles/minutes/5", {"market":m,"count":200,"to":iso(to)})
        if not rows: break
        oldest=None
        for x in rows:
            t=dt_utc(x["candle_date_time_utc"])
            oldest=t if oldest is None or t<oldest else oldest
            if start_utc < t <= end_utc and x["candle_date_time_utc"] not in seen:
                seen.add(x["candle_date_time_utc"]); out.append(x)
        if oldest is None or oldest<=start_utc: break
        to=oldest-timedelta(seconds=1)
        time.sleep(DELAY)
    out.sort(key=lambda x:x["candle_date_time_utc"])
    p.write_text(json.dumps(out,ensure_ascii=False),encoding="utf-8")
    return out


def v76(rows):
    if len(rows)<25:return None
    rows=sorted(rows,key=lambda x:x["candle_date_time_utc"])
    cl=np.array([float(x["trade_price"]) for x in rows],dtype=float)
    tv=np.array([float(x["candle_acc_trade_price"]) for x in rows],dtype=float)
    r5=pct(cl[-1],cl[-2]); r15=pct(cl[-1],cl[-4]); r30=pct(cl[-1],cl[-7]); r60=pct(cl[-1],cl[-13])
    vr=tv[-1]/((tv[-7:-1].sum()/6) or 1)
    a15=(tv[-3:].sum()/3)/((tv[-9:-3].sum()/6) or 1)
    a30=(tv[-6:].sum()/6)/((tv[-18:-6].sum()/12) or 1)
    score=35+clamp((vr-1)*12,-8,18)+clamp((a15-1)*18,-10,25)+clamp((a30-1)*10,-6,14)+clamp(r5*3,-8,8)+clamp(r15*1.8,-8,10)
    over=0
    if r5>2.5:over+=(r5-2.5)*5
    if r15>4:over+=(r15-4)*4
    if r60>7:over+=(r60-7)*2
    score-=clamp(over,0,35)
    chase=(r5>=3) or (r15>=5) or (r60>=8)
    early=(not chase) and a15>=1.25 and vr>=1.15
    label="CHASE" if chase else ("EARLY" if early else "WATCH")
    z={"price":float(cl[-1]),"score":round(clamp(score,0,100),2),"label":label,
       "ret_5m":round(r5,3),"ret_15m":round(r15,3),"ret_30m":round(r30,3),"ret_60m":round(r60,3),
       "value_ratio_5m":round(float(vr),3),"value_accel_15m":round(float(a15),3),"value_accel_30m":round(float(a30),3)}
    z["A"]=bool(label=="EARLY" and z["score"]>=85 and a15>=3 and a30>=2)
    z["B"]=bool(label=="EARLY" and z["score"]>=80 and a15>=1.5 and a30>=3)
    z["C"]=bool(label=="EARLY" and z["score"]>=70 and a15>=2 and a30>=2 and -1.5<=r5<=0.5 and -2<=r15<=1 and r30<=0.5)
    z["ABC_ALL"]=z["A"] and z["B"] and z["C"]
    return z


def replay(rows, entry, target_pct=20.0, stop_pct=None):
    if not rows:return None
    tp=entry*(1+target_pct/100)
    sl=entry*(1-stop_pct/100) if stop_pct is not None else None
    mfe=-1e9; mae=1e9
    for x in rows:
        hi=float(x["high_price"]); lo=float(x["low_price"])
        mfe=max(mfe,pct(hi,entry)); mae=min(mae,pct(lo,entry))
        hit_tp=hi>=tp
        hit_sl=(sl is not None and lo<=sl)
        # Conservative OHLC convention if both occur in same 5m candle.
        if hit_tp and hit_sl:
            return {"outcome":"stop_first_conservative","return_pct":-float(stop_pct),"exit_ts":dt_utc(x["candle_date_time_utc"]).isoformat(),"ambiguous_same_bar":True,"mfe_pct":round(mfe,3),"mae_pct":round(mae,3)}
        if hit_sl:
            return {"outcome":"stop_first","return_pct":-float(stop_pct),"exit_ts":dt_utc(x["candle_date_time_utc"]).isoformat(),"ambiguous_same_bar":False,"mfe_pct":round(mfe,3),"mae_pct":round(mae,3)}
        if hit_tp:
            return {"outcome":"target_first","return_pct":float(target_pct),"exit_ts":dt_utc(x["candle_date_time_utc"]).isoformat(),"ambiguous_same_bar":False,"mfe_pct":round(mfe,3),"mae_pct":round(mae,3)}
    last=float(rows[-1]["trade_price"])
    return {"outcome":"timeout","return_pct":round(pct(last,entry),3),"exit_ts":dt_utc(rows[-1]["candle_date_time_utc"]).isoformat(),"ambiguous_same_bar":False,"mfe_pct":round(mfe,3),"mae_pct":round(mae,3)}


def weekly_dates():
    cur=START
    # first requested weekday/hour on or after START
    cur=cur.replace(hour=HOUR_KST,minute=0,second=0,microsecond=0)
    while cur.weekday()!=WEEKDAY or cur<START:
        cur+=timedelta(days=1)
        cur=cur.replace(hour=HOUR_KST,minute=0,second=0,microsecond=0)
    out=[]
    while cur<=END:
        out.append(cur)
        cur+=timedelta(days=7)
    return out


def max_drawdown(equity):
    peak=-1; mdd=0.0
    for v in equity:
        peak=max(peak,v)
        if peak>0:mdd=min(mdd,(v/peak-1)*100)
    return round(mdd,3)


def main():
    ms=markets()
    weeks=weekly_dates()
    print("markets",len(ms),"weeks",len(weeks))
    week_rows=[]

    for wi,wk in enumerate(weeks,1):
        at=wk.astimezone(UTC)
        print(f"WEEK {wi}/{len(weeks)} {wk.isoformat()}")
        candidates=[]
        for i,m in enumerate(ms,1):
            try:
                z=v76(fetch_5m_before(m,at,30))
                if z:
                    z["market"]=m
                    candidates.append(z)
            except Exception as e:
                print("scan_error",m,str(e)[:120])
        pri={"EARLY":0,"WATCH":1,"CHASE":2}
        candidates.sort(key=lambda x:(pri[x["label"]],-x["score"],-x["value_accel_15m"]))
        picks=candidates[:TOP_K]
        week={"decision_ts_kst":wk.isoformat(),"eligible_count":len(candidates),"picks":[]}
        for rank,z in enumerate(picks,1):
            m=z["market"]; entry=z["price"]
            path=fetch_5m_range(m,at,at+timedelta(days=7))
            rec={"rank":rank,**z}
            rec["no_stop"]=replay(path,entry,TARGET_PCT,None)
            for s in STOP_PCTS:
                rec[f"sl_{int(s) if s.is_integer() else s}"]=replay(path,entry,TARGET_PCT,s)
            week["picks"].append(rec)
        week_rows.append(week)

    strategies=["no_stop"]+[f"sl_{int(s) if s.is_integer() else s}" for s in STOP_PCTS]
    summaries={}
    for strat in strategies:
        cap=INITIAL_CAPITAL
        equity=[cap]
        weekly=[]
        all_trade_returns=[]
        tgt=stop=timeout=amb=0
        doubled_at=None
        for w in week_rows:
            vals=[p[strat] for p in w["picks"] if p.get(strat)]
            rets=[v["return_pct"] for v in vals]
            if not rets:continue
            wr=float(np.mean(rets))
            cap*=1+wr/100
            equity.append(cap)
            all_trade_returns.extend(rets)
            tgt+=sum(v["outcome"]=="target_first" for v in vals)
            stop+=sum(v["outcome"] in ("stop_first","stop_first_conservative") for v in vals)
            timeout+=sum(v["outcome"]=="timeout" for v in vals)
            amb+=sum(v.get("ambiguous_same_bar",False) for v in vals)
            weekly.append({"decision_ts_kst":w["decision_ts_kst"],"portfolio_return_pct":round(wr,3),"capital":round(cap,0)})
            if doubled_at is None and cap>=INITIAL_CAPITAL*2:
                doubled_at=w["decision_ts_kst"]
        n=len(all_trade_returns)
        summaries[strat]={
            "weeks":len(weekly),"trades":n,
            "target_first_rate":round(100*tgt/n,2) if n else None,
            "stop_first_rate":round(100*stop/n,2) if n else None,
            "timeout_rate":round(100*timeout/n,2) if n else None,
            "ambiguous_same_bar_rate":round(100*amb/n,2) if n else None,
            "avg_trade_return_pct":round(float(np.mean(all_trade_returns)),3) if n else None,
            "median_trade_return_pct":round(float(np.median(all_trade_returns)),3) if n else None,
            "avg_weekly_portfolio_return_pct":round(float(np.mean([x["portfolio_return_pct"] for x in weekly])),3) if weekly else None,
            "final_capital":round(cap,0),
            "total_return_pct":round((cap/INITIAL_CAPITAL-1)*100,2),
            "max_drawdown_pct":max_drawdown(equity),
            "doubled_at_kst":doubled_at,
            "weekly_equity":weekly
        }

    result={
        "version":"V76_WEEKLY_TOP5_PATH_V1",
        "config":{
            "start_kst":START.isoformat(),"end_kst":END.isoformat(),"weekday":WEEKDAY,"hour_kst":HOUR_KST,
            "top_k":TOP_K,"initial_capital_krw":INITIAL_CAPITAL,"target_pct":TARGET_PCT,"stop_pcts":list(STOP_PCTS),
            "selection_rule":"Every Monday 09:00 KST, scan all current Upbit KRW markets with the unchanged V7.6 ranking logic and buy TOP5 equally (20% each).",
            "holding_rule":"Each position exits at +20%, tested stop, or after 7 days. Proceeds stay idle until next weekly rebalance.",
            "same_bar_rule":"If TP and SL touch in the same 5m candle, stop is counted first (conservative).",
            "fees_slippage":"excluded",
            "warning":"Historical replay, not clean OOS. Current-market survivorship bias exists because today's KRW market list is used."
        },
        "summary":summaries,
        "weeks":week_rows
    }
    OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(summaries,ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
