#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json, os, time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import requests

API = "https://api.upbit.com"
KST = timezone(timedelta(hours=9))
UTC = timezone.utc
START = datetime.strptime(os.getenv("BT_START_KST","2026-08-15 00:00"),"%Y-%m-%d %H:%M").replace(tzinfo=KST)
END = datetime.strptime(os.getenv("BT_END_KST","2026-09-14 23:00"),"%Y-%m-%d %H:%M").replace(tzinfo=KST)
DELAY = float(os.getenv("BT_REQUEST_DELAY","0.12"))
MAX_MARKETS = int(os.getenv("BT_MAX_MARKETS","0"))
OUT = Path("docs/data/hybrid_month_backtest_latest.json")
CACHE = Path(".cache/hybrid_bt")
OUT.parent.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)

STABLE={"USDT","USDC","DAI","USD1","USDE","FDUSD","TUSD","RLUSD","EURC"}
S=requests.Session()
S.headers.update({"User-Agent":"upbit-hybrid-backtest/1.0"})

def get(path, params=None, tries=6):
    last=None
    for i in range(tries):
        try:
            r=S.get(API+path,params=params,timeout=20)
            if r.status_code==429:
                time.sleep(0.7*(i+1)); continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last=e; time.sleep(0.7*(i+1))
    raise last

def iso(dt): return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
def f(x,d=0.0):
    try:return float(x)
    except:return d
def clamp(x,a,b): return max(a,min(b,x))
def pct(a,b): return (a/b-1)*100 if b else 0.0

def markets():
    xs=get("/v1/market/all",{"is_details":"false"})
    ms=sorted(x["market"] for x in xs if x["market"].startswith("KRW-") and x["market"].split("-",1)[1] not in STABLE)
    return ms if MAX_MARKETS<=0 else ms[:MAX_MARKETS]

def fetch_range(m, endpoint, start_utc, end_utc, kind):
    p=CACHE/f"{m.replace('-','_')}_{kind}.json"
    if p.exists():
        try:
            x=json.loads(p.read_text(encoding="utf-8"))
            if x:return x
        except:pass
    out=[]; seen=set(); to=end_utc+timedelta(seconds=1)
    while True:
        rows=get(endpoint,{"market":m,"count":200,"to":iso(to)})
        if not rows:break
        oldest=None
        for x in rows:
            t=datetime.fromisoformat(x["candle_date_time_utc"]).replace(tzinfo=UTC)
            oldest=t if oldest is None or t<oldest else oldest
            if start_utc<=t<=end_utc and x["candle_date_time_utc"] not in seen:
                seen.add(x["candle_date_time_utc"]); out.append(x)
        if oldest is None or oldest<=start_utc:break
        to=oldest-timedelta(seconds=1)
        time.sleep(DELAY)
    out.sort(key=lambda x:x["candle_date_time_utc"])
    p.write_text(json.dumps(out,ensure_ascii=False),encoding="utf-8")
    return out

def frame(rows):
    if not rows:return pd.DataFrame()
    z=pd.DataFrame(rows)
    z["ts"]=pd.to_datetime(z["candle_date_time_utc"],utc=True)
    z=z.set_index("ts").sort_index()
    for c in ["opening_price","high_price","low_price","trade_price","candle_acc_trade_price"]:
        if c in z:z[c]=pd.to_numeric(z[c],errors="coerce")
    return z

def ema(s,n): return s.ewm(span=n,adjust=False).mean()
def rsi(s,n=14):
    d=s.diff(); up=d.clip(lower=0); dn=-d.clip(upper=0)
    au=up.ewm(alpha=1/n,adjust=False,min_periods=n).mean()
    ad=dn.ewm(alpha=1/n,adjust=False,min_periods=n).mean()
    rs=au/ad.replace(0,np.nan)
    return 100-(100/(1+rs))
def macd_hist(s):
    m=ema(s,12)-ema(s,26)
    return m-ema(m,9)

def v76(w):
    if len(w)<25:return None
    cl=w["trade_price"].astype(float).to_numpy()
    tv=w["candle_acc_trade_price"].astype(float).to_numpy()
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
       "ret_5m":r5,"ret_15m":r15,"ret_30m":r30,"ret_60m":r60,
       "value_ratio_5m":vr,"value_accel_15m":a15,"value_accel_30m":a30}
    z["A"]=bool(label=="EARLY" and z["score"]>=85 and a15>=3 and a30>=2)
    z["B"]=bool(label=="EARLY" and z["score"]>=80 and a15>=1.5 and a30>=3)
    z["C"]=bool(label=="EARLY" and z["score"]>=70 and a15>=2 and a30>=2 and -1.5<=r5<=0.5 and -2<=r15<=1 and r30<=0.5)
    z["ABC_ALL"]=z["A"] and z["B"] and z["C"]
    return z

def chart_score(h1,day,at):
    h=h1[h1.index<=at].tail(220); d=day[day.index<=at].tail(120)
    if len(h)<60 or len(d)<35:return None
    hc=h["trade_price"].astype(float); dc=d["trade_price"].astype(float)
    cur=float(hc.iloc[-1]); e5,e10,e20,e60=[float(ema(hc,n).iloc[-1]) for n in (5,10,20,60)]
    hr=float(rsi(hc).iloc[-1]); dr=float(rsi(dc).iloc[-1]); mh=float(macd_hist(hc).iloc[-1])
    d20=float(ema(dc,20).iloc[-1]); d60=float(ema(dc,60).iloc[-1])
    trend=0
    if cur>e20:trend+=8
    if e5>e10>e20:trend+=10
    elif e5>e10:trend+=5
    if e20>e60:trend+=7
    if 45<=hr<=65:trend+=8
    elif 40<=hr<45:trend+=4
    elif hr>72:trend-=8
    if mh>0:trend+=6
    daily=0
    if cur>d20:daily+=6
    if d20>d60:daily+=6
    if 40<=dr<=62:daily+=8
    elif 35<=dr<40:daily+=4
    elif dr>70:daily-=8
    def rd(n):
        return pct(float(dc.iloc[-1]),float(dc.iloc[-1-n])) if len(dc)>n else 0
    r3,r7,r14=rd(3),rd(7),rd(14)
    low14=float(d["low_price"].tail(14).min()); dist14=pct(cur,low14)
    ext=0
    if r3>12:ext+=min(12,(r3-12)*0.8)
    if r7>25:ext+=min(14,(r7-25)*0.5)
    if r14>45:ext+=min(16,(r14-45)*0.35)
    if dist14>60:ext+=min(12,(dist14-60)*0.25)
    if dr>75:ext+=10
    return {"trend_score":clamp(trend,0,40),"daily_score":clamp(daily,0,20),"extension_penalty":clamp(ext,0,45),"h_rsi":round(hr,2),"d_rsi":round(dr,2),"ret3d":round(r3,3),"ret7d":round(r7,3),"ret14d":round(r14,3)}

def hybrid(latest,recent,ch):
    base=0
    base+=clamp((latest["score"]-65)/35,0,1)*25
    base+=clamp((latest["value_accel_30m"]-1)/5,0,1)*8
    base+=clamp((latest["value_accel_15m"]-1)/7,0,1)*6
    positives=sum(x>0 for x in [latest["ret_5m"],latest["ret_15m"],latest["ret_30m"],latest["ret_60m"]])
    base+=positives*4
    base+=clamp(latest["ret_30m"]/1.5,0,1)*8
    base+=clamp(latest["ret_60m"]/3,0,1)*10
    base+=ch["trend_score"]+ch["daily_score"]-ch["extension_penalty"]
    base+=clamp(sum(r["score"]>=85 for r in recent)/5,0,1)*6
    if latest["label"]=="CHASE":base-=20
    sc=clamp(base,0,100)
    if sc>=72 and positives>=3 and latest["ret_30m"]>0 and latest["ret_60m"]>0:stage="LAUNCH"
    elif sc>=62 and (latest["ret_30m"]>0 or latest["ret_60m"]>0):stage="TRANSITION"
    elif sc>=55:stage="LATENT"
    else:stage="WATCH"
    return round(sc,2),stage

def future(h1,at,entry):
    fut=h1[(h1.index>at)&(h1.index<=at+timedelta(days=7))]
    if fut.empty:return None
    o={}
    for d in (1,3,5,7):
        w=fut[fut.index<=at+timedelta(days=d)]
        o[f"d{d}_max_pct"]=round(pct(float(w["high_price"].max()),entry),3) if not w.empty else None
    for t in (5,10,20):
        hit=fut[fut["high_price"]>=entry*(1+t/100)]
        o[f"hit_{t}"]=bool(not hit.empty)
    return o

def summary(xs):
    if not xs:return {"n":0}
    return {"n":len(xs),"hit_5_rate":round(100*sum(x["hit_5"] for x in xs)/len(xs),2),"hit_10_rate":round(100*sum(x["hit_10"] for x in xs)/len(xs),2),"hit_20_rate":round(100*sum(x["hit_20"] for x in xs)/len(xs),2)}

def main():
    ms=markets(); print("markets",len(ms))
    d5={}; h1={}; day={}
    for i,m in enumerate(ms,1):
        print(i,len(ms),m)
        d5[m]=frame(fetch_range(m,"/v1/candles/minutes/5",(START-timedelta(days=1)).astimezone(UTC),END.astimezone(UTC),"5m"))
        h1[m]=frame(fetch_range(m,"/v1/candles/minutes/60",(START-timedelta(days=12)).astimezone(UTC),(END+timedelta(days=7,hours=2)).astimezone(UTC),"60m"))
        day[m]=frame(fetch_range(m,"/v1/candles/days",(START-timedelta(days=75)).astimezone(UTC),(END+timedelta(days=8)).astimezone(UTC),"day"))

    hist=defaultdict(list); v76_sel=[]; hybrid_sel=[]
    decisions=set()
    t=START.replace(minute=0,second=0,microsecond=0)
    while t<=END:
        decisions.add(t.astimezone(UTC)); t+=timedelta(hours=1)

    t5=START.replace(minute=(START.minute//5)*5,second=0,microsecond=0)
    while t5<=END:
        at=t5.astimezone(UTC); rows=[]
        for m in ms:
            z=v76(d5[m][d5[m].index<=at].tail(30)) if not d5[m].empty else None
            if z:z["market"]=m;z["_dt"]=at;rows.append(z)
        pri={"EARLY":0,"WATCH":1,"CHASE":2}
        rows.sort(key=lambda x:(pri[x["label"]],-x["score"],-x["value_accel_15m"]))
        for rank,z in enumerate(rows,1):
            if rank<=30 or (z["label"]=="EARLY" and (z["A"] or z["B"] or z["C"])) or (z["label"]=="EARLY" and z["score"]>=65):
                h=hist[z["market"]]
                if not h or at-h[-1]["_dt"]>timedelta(minutes=60) or (h[-1]["A"],h[-1]["B"],h[-1]["C"])!=(z["A"],z["B"],z["C"]):
                    h.append(dict(z,rank=rank))
        if at in decisions:
            stamp=at.astimezone(KST).isoformat()
            for z in rows[:3]:
                fm=future(h1[z["market"]],at,z["price"])
                if fm:v76_sel.append({"decision_ts":stamp,"market":z["market"],"entry_price":z["price"],"score":z["score"],**fm})
            c=[]
            for m,hs in hist.items():
                recent=[r for r in hs if at-timedelta(hours=72)<=r["_dt"]<=at]
                if not recent:continue
                latest=recent[-1]
                if at-latest["_dt"]>timedelta(hours=6):continue
                ch=chart_score(h1[m],day[m],at)
                if not ch:continue
                sc,stage=hybrid(latest,recent,ch)
                if latest["label"]=="CHASE" or ch["extension_penalty"]>25 or sc<58 or stage=="WATCH":continue
                c.append((sc,m,latest,stage,ch))
            c.sort(key=lambda x:x[0],reverse=True)
            for sc,m,latest,stage,ch in c[:3]:
                cur=float(h1[m][h1[m].index<=at]["trade_price"].iloc[-1])
                fm=future(h1[m],at,cur)
                if fm:hybrid_sel.append({"decision_ts":stamp,"market":m,"entry_price":cur,"hybrid_score":sc,"stage":stage,**ch,**fm})
        t5+=timedelta(minutes=5)

    result={"version":"HYBRID_REPLAY_V1","config":{"start_kst":START.isoformat(),"end_kst":END.isoformat(),"market_count":len(ms),"primary_target":"+20% within 7d","warning":"historical replay; not clean OOS; current-market survivorship bias possible"},"summary":{"v76_top3":summary(v76_sel),"hybrid_top0_3":summary(hybrid_sel)},"v76_selections":v76_sel,"hybrid_selections":hybrid_sel}
    OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(result["summary"],ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
