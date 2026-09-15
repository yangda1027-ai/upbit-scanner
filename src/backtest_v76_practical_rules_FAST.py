#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V7.6 PRACTICAL RULES BACKTEST — preregistered / no live-code changes

Purpose
-------
Test the *operating rules* around the unchanged V7.6 selector, not modify V7.6.

Fixed decision sequence:
  1) V7.6 TOP3 (original V7.6 score/label logic reproduced verbatim)
  2) EARLY only (CHASE excluded)
  3) 3-day momentum: +5% <= ret_3d <= +10%
  4) repeated TOP3 appearance is recorded and used only as a tie-breaker
     (NOT a hard filter, matching the agreed operating principle)
  5) choose ONE candidate
  6) 72h same-market cooldown to reduce duplicate-signal inflation

Paired benchmark
----------------
At the SAME decision timestamps, Random chooses one coin from the universe that
passes the same EARLY + ret_3d filter. This asks whether V7.6 ranking adds value
after the operating filter.

Important:
- This is historical replay, not clean prospective OOS.
- Current Upbit KRW market list => survivorship bias.
- Fees/slippage excluded.
- Exact +5~10% ret_3d band is preregistered here. DO NOT tune it after seeing results.
- Same-bar TP/SL ambiguity is resolved conservatively: stop first.
"""

import os, json, time, random, math
from pathlib import Path
from datetime import datetime, timedelta, timezone
import requests
import numpy as np

API = "https://api.upbit.com"
KST = timezone(timedelta(hours=9))
UTC = timezone.utc

START = datetime.strptime(os.getenv("BT_START_KST", "2026-03-02 09:00"), "%Y-%m-%d %H:%M").replace(tzinfo=KST)
END   = datetime.strptime(os.getenv("BT_END_KST",   "2026-09-14 23:00"), "%Y-%m-%d %H:%M").replace(tzinfo=KST)

# To keep runtime practical, decisions are sampled every 6h by default.
# This does NOT alter V7.6 logic; it only defines historical evaluation checkpoints.
STEP_HOURS = int(os.getenv("BT_STEP_HOURS", "6"))
TOP_N = 3
RET3_MIN = float(os.getenv("BT_RET3_MIN", "5"))
RET3_MAX = float(os.getenv("BT_RET3_MAX", "10"))
COOLDOWN_HOURS = int(os.getenv("BT_COOLDOWN_HOURS", "72"))
HOLD_DAYS = int(os.getenv("BT_HOLD_DAYS", "7"))
TARGET_PCT = float(os.getenv("BT_TARGET_PCT", "20"))
STOP_PCTS = tuple(float(x) for x in os.getenv("BT_STOP_PCTS", "3,5,7,10").split(","))
RANDOM_RUNS = int(os.getenv("BT_RANDOM_RUNS", "1000"))
SEED = int(os.getenv("BT_RANDOM_SEED", "7610"))
DELAY = float(os.getenv("BT_REQUEST_DELAY", "0.11"))

OUT = Path("docs/data/v76_practical_rules_backtest_latest.json")
CACHE = Path(".cache/v76_practical_rules")
OUT.parent.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)

# Runtime optimization: one 200-candle block can serve multiple 6h checkpoints.
# This changes retrieval only, not V7.6 scoring/selection rules.
BLOCK_CACHE = {}  # market -> {oldest, newest, rows}

STABLE = {"USDT","USDC","DAI","USD1","USDE","FDUSD","TUSD","RLUSD","EURC"}

S = requests.Session()
S.headers.update({"User-Agent":"upbit-v76-practical-rules-backtest/1.0"})

def get(path, params=None, tries=8):
    last=None
    for i in range(tries):
        try:
            r=S.get(API+path, params=params, timeout=30)
            if r.status_code==429:
                time.sleep(0.8*(i+1)); continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last=e; time.sleep(0.8*(i+1))
    raise last

def iso(dt):
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

def dt_utc(s):
    return datetime.fromisoformat(str(s).replace("Z","+00:00"))

def pct(a,b):
    return (a/b-1)*100 if b else 0.0

def clamp(x,a,b):
    return max(a,min(b,x))

def markets():
    xs=get("/v1/market/all",{"is_details":"false"})
    return sorted(x["market"] for x in xs
                  if x["market"].startswith("KRW-")
                  and x["market"].split("-",1)[1] not in STABLE)

def checkpoints():
    cur=START
    out=[]
    while cur<=END:
        out.append(cur)
        cur += timedelta(hours=STEP_HOURS)
    return out

def cache_read(p):
    try:
        if p.exists(): return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        pass
    return None

def cache_write(p,x):
    p.write_text(json.dumps(x,ensure_ascii=False),encoding="utf-8")

def fetch_5m_before(m, at, count=30):
    # Disk cache keeps exact checkpoint results across successful reruns.
    p=CACHE/f"pre_{m.replace('-','_')}_{at.strftime('%Y%m%dT%H%M')}.json"
    x=cache_read(p)
    if x is not None:
        return x

    # A 200-candle request spans ~16h40m, so with 6h checkpoints the same
    # response can usually serve 2-3 adjacent checkpoints.  We intentionally
    # fetch a larger block but still return the exact last `count` candles
    # at/before `at`, preserving V7.6 inputs.
    b=BLOCK_CACHE.get(m)
    if b is not None and b["oldest"] <= at <= b["newest"] + timedelta(minutes=5):
        eligible=[z for z in b["rows"] if dt_utc(z["candle_date_time_utc"]) <= at + timedelta(seconds=1)]
        if len(eligible) >= count:
            rows=eligible[-count:]
            cache_write(p,rows)
            return rows

    block_to = min(at + timedelta(hours=12), END.astimezone(UTC) + timedelta(seconds=1))
    raw=get("/v1/candles/minutes/5",{"market":m,"count":200,"to":iso(block_to)})
    block=sorted(raw,key=lambda z:z["candle_date_time_utc"])
    if block:
        BLOCK_CACHE[m]={
            "oldest":dt_utc(block[0]["candle_date_time_utc"]),
            "newest":dt_utc(block[-1]["candle_date_time_utc"]),
            "rows":block,
        }
    rows=block[-count:]
    cache_write(p,rows); time.sleep(DELAY)
    return rows

def fetch_days_before(m, at, count=5):
    p=CACHE/f"day_{m.replace('-','_')}_{at.strftime('%Y%m%dT%H%M')}.json"
    x=cache_read(p)
    if x is not None:return x
    rows=get("/v1/candles/days",{"market":m,"count":count,"to":iso(at+timedelta(seconds=1))})
    rows=sorted(rows,key=lambda z:z["candle_date_time_utc"])
    cache_write(p,rows); time.sleep(DELAY)
    return rows

def ret3d(m, at, current_price):
    ds=fetch_days_before(m,at,5)
    if len(ds)<4:return None
    # Match the live monitor's daily_extension convention:
    # current / close 3 daily candles back.
    old=float(ds[-4]["trade_price"])
    return pct(current_price,old) if old else None

def v76(rows):
    # Unchanged V7.6 feature/score/label logic.
    if len(rows)<25:return None
    rows=sorted(rows,key=lambda x:x["candle_date_time_utc"])
    cl=np.array([float(x["trade_price"]) for x in rows],dtype=float)
    tv=np.array([float(x["candle_acc_trade_price"]) for x in rows],dtype=float)

    r5=pct(cl[-1],cl[-2])
    r15=pct(cl[-1],cl[-4])
    r30=pct(cl[-1],cl[-7])
    r60=pct(cl[-1],cl[-13])

    v5=tv[-1]/((tv[-7:-1].sum()/6) or 1)
    a15=(tv[-3:].sum()/3)/((tv[-9:-3].sum()/6) or 1)
    a30=(tv[-6:].sum()/6)/((tv[-18:-6].sum()/12) or 1)

    score=35.0
    score += clamp((v5-1)*12,-8,18)+clamp((a15-1)*18,-10,25)+clamp((a30-1)*10,-6,14)
    score += clamp(r5*3,-8,8)+clamp(r15*1.8,-8,10)

    over=0.0
    if r5>2.5: over+=(r5-2.5)*5
    if r15>4.0: over+=(r15-4.0)*4
    if r60>7.0: over+=(r60-7.0)*2
    score -= clamp(over,0,35)

    chase=(r5>=3.0) or (r15>=5.0) or (r60>=8.0)
    early=(not chase) and (a15>=1.25) and (v5>=1.15)
    label="CHASE" if chase else ("EARLY" if early else "WATCH")

    return {
        "price":float(cl[-1]),"score":round(clamp(score,0,100),2),"label":label,
        "ret_5m":round(r5,3),"ret_15m":round(r15,3),"ret_30m":round(r30,3),"ret_60m":round(r60,3),
        "value_ratio_5m":round(float(v5),3),"value_accel_15m":round(float(a15),3),
        "value_accel_30m":round(float(a30),3)
    }

def fetch_5m_range(m,start,end):
    p=CACHE/f"path_{m.replace('-','_')}_{start.strftime('%Y%m%dT%H%M')}_{end.strftime('%Y%m%dT%H%M')}.json"
    x=cache_read(p)
    if x is not None:return x
    out=[]; seen=set(); to=end+timedelta(seconds=1)
    while True:
        rows=get("/v1/candles/minutes/5",{"market":m,"count":200,"to":iso(to)})
        if not rows:break
        oldest=None
        for z in rows:
            t=dt_utc(z["candle_date_time_utc"])
            oldest=t if oldest is None or t<oldest else oldest
            if start < t <= end and z["candle_date_time_utc"] not in seen:
                seen.add(z["candle_date_time_utc"]); out.append(z)
        if oldest is None or oldest<=start:break
        to=oldest-timedelta(seconds=1); time.sleep(DELAY)
    out.sort(key=lambda z:z["candle_date_time_utc"])
    cache_write(p,out)
    return out

def replay(rows,entry,target=20.0,stop=None):
    if not rows:return None
    tp=entry*(1+target/100)
    sl=entry*(1-stop/100) if stop is not None else None
    mfe=-1e9; mae=1e9
    hit5=hit10=hit20=False
    for z in rows:
        hi=float(z["high_price"]); lo=float(z["low_price"])
        mfe=max(mfe,pct(hi,entry)); mae=min(mae,pct(lo,entry))
        hit5 |= hi>=entry*1.05
        hit10 |= hi>=entry*1.10
        hit20 |= hi>=entry*1.20
        htp=hi>=tp; hsl=(sl is not None and lo<=sl)
        if htp and hsl:
            return {"outcome":"stop_first_conservative","return_pct":-float(stop),
                    "mfe_pct":round(mfe,3),"mae_pct":round(mae,3),
                    "hit_5":hit5,"hit_10":hit10,"hit_20":hit20,"ambiguous_same_bar":True}
        if hsl:
            return {"outcome":"stop_first","return_pct":-float(stop),
                    "mfe_pct":round(mfe,3),"mae_pct":round(mae,3),
                    "hit_5":hit5,"hit_10":hit10,"hit_20":hit20,"ambiguous_same_bar":False}
        if htp:
            return {"outcome":"target_first","return_pct":float(target),
                    "mfe_pct":round(mfe,3),"mae_pct":round(mae,3),
                    "hit_5":hit5,"hit_10":hit10,"hit_20":hit20,"ambiguous_same_bar":False}
    last=float(rows[-1]["trade_price"])
    return {"outcome":"timeout","return_pct":round(pct(last,entry),3),
            "mfe_pct":round(mfe,3),"mae_pct":round(mae,3),
            "hit_5":hit5,"hit_10":hit10,"hit_20":hit20,"ambiguous_same_bar":False}

def max_drawdown(eq):
    peak=-1; mdd=0
    for x in eq:
        peak=max(peak,x)
        if peak>0:mdd=min(mdd,(x/peak-1)*100)
    return round(mdd,3)

def profit_factor(rets):
    gp=sum(x for x in rets if x>0)
    gl=-sum(x for x in rets if x<0)
    return round(gp/gl,3) if gl>0 else None

def stats(trades, stop_key):
    xs=[t["results"][stop_key] for t in trades if t.get("results",{}).get(stop_key)]
    if not xs:return {"n":0}
    rets=[x["return_pct"] for x in xs]
    eq=[1.0]
    for r in rets:eq.append(eq[-1]*(1+r/100))
    n=len(xs)
    return {
        "n":n,
        "hit_5_rate":round(100*sum(x["hit_5"] for x in xs)/n,2),
        "hit_10_rate":round(100*sum(x["hit_10"] for x in xs)/n,2),
        "hit_20_rate":round(100*sum(x["hit_20"] for x in xs)/n,2),
        "target_first_rate":round(100*sum(x["outcome"]=="target_first" for x in xs)/n,2),
        "stop_first_rate":round(100*sum(x["outcome"].startswith("stop_first") for x in xs)/n,2),
        "avg_return_pct":round(float(np.mean(rets)),3),
        "median_return_pct":round(float(np.median(rets)),3),
        "profit_factor":profit_factor(rets),
        "compounded_return_pct":round((eq[-1]-1)*100,2),
        "max_drawdown_pct":max_drawdown(eq)
    }

def split_stats(trades,stop_key):
    if not trades:return {}
    ts=sorted(dt_utc(t["decision_ts_utc"]) for t in trades)
    cut=ts[int(len(ts)*2/3)] if len(ts)>=3 else ts[-1]
    a=[t for t in trades if dt_utc(t["decision_ts_utc"])<cut]
    b=[t for t in trades if dt_utc(t["decision_ts_utc"])>=cut]
    return {"first_2_3":stats(a,stop_key),"last_1_3":stats(b,stop_key),"cutoff_utc":iso(cut)}

def main():
    rng=random.Random(SEED)
    ms=markets(); cps=checkpoints()
    print("markets",len(ms),"checkpoints",len(cps))

    top3_history={}  # market -> list of timestamps
    last_selected={}
    events=[]
    eligible_by_event=[]

    for ci,ck in enumerate(cps,1):
        at=ck.astimezone(UTC)
        print(f"CHECK {ci}/{len(cps)} {ck.isoformat()}",flush=True)
        scan=[]
        for m in ms:
            try:
                z=v76(fetch_5m_before(m,at,30))
                if z:
                    z["market"]=m; scan.append(z)
            except Exception as e:
                print("scan_error",m,str(e)[:80])

        ranked=sorted(scan,key=lambda x:(-x["score"],-x["value_accel_15m"]))
        top3=ranked[:TOP_N]

        # repetition is TOP3 appearances during previous 72h, excluding current checkpoint
        for z in top3:
            hist=top3_history.setdefault(z["market"],[])
            hist[:]=[t for t in hist if at-t<=timedelta(hours=72)]
            z["repeat_top3_72h_before"]=len(hist)

        # Compute ret3d only for TOP3 (keeps API load practical)
        candidates=[]
        for z in top3:
            try:
                r3=ret3d(z["market"],at,z["price"])
            except Exception:
                r3=None
            z["ret_3d"]=round(r3,3) if r3 is not None else None
            if z["label"]=="EARLY" and r3 is not None and RET3_MIN<=r3<=RET3_MAX:
                candidates.append(z)

        # record current TOP3 after evaluating repetition
        for z in top3:
            top3_history.setdefault(z["market"],[]).append(at)

        # score first; repetition is only tie-breaker, never a hard gate
        candidates.sort(key=lambda x:(-x["score"],-x["repeat_top3_72h_before"],-x["value_accel_15m"]))
        chosen=None
        for z in candidates:
            prev=last_selected.get(z["market"])
            if prev is None or at-prev>=timedelta(hours=COOLDOWN_HOURS):
                chosen=z; break

        if chosen is None:
            continue

        last_selected[chosen["market"]]=at

        # Fair random universe: coins from the same TOP3 that pass the same operating filters.
        # This is deliberately paired and avoids fetching paths for the whole KRW universe.
        # It tests whether V7.6's ordering among its filtered TOP3 adds value.
        rand_pool=[z["market"] for z in candidates]
        event={
            "decision_ts_utc":iso(at),
            "decision_ts_kst":ck.isoformat(),
            "top3":[{k:z.get(k) for k in ("market","score","label","price","ret_3d","repeat_top3_72h_before",
                                          "ret_5m","ret_15m","ret_60m")} for z in top3],
            "selected_market":chosen["market"],
            "selected_score":chosen["score"],
            "selected_ret_3d":chosen["ret_3d"],
            "selected_repeat_top3_72h_before":chosen["repeat_top3_72h_before"],
            "random_pool":rand_pool
        }
        events.append(event)

    # Unique market/timestamp paths only for markets actually appearing in eligible pools.
    packed={}
    def pack(at,m,entry):
        key=(iso(at),m)
        if key in packed:return packed[key]
        path=fetch_5m_range(m,at,at+timedelta(days=HOLD_DAYS))
        rr={"no_stop":replay(path,entry,TARGET_PCT,None)}
        for s in STOP_PCTS:
            rr[f"sl_{int(s)}"]=replay(path,entry,TARGET_PCT,s)
        packed[key]=rr
        return rr

    # Need entry prices for every random-pool member. Recover from event TOP3.
    vtrades=[]
    event_random_options=[]
    for i,e in enumerate(events,1):
        at=dt_utc(e["decision_ts_utc"])
        meta={z["market"]:z for z in e["top3"]}
        m=e["selected_market"]
        vr={
            "decision_ts_utc":e["decision_ts_utc"],"market":m,
            "score":meta[m]["score"],"ret_3d":meta[m]["ret_3d"],
            "repeat_top3_72h_before":meta[m]["repeat_top3_72h_before"],
            "results":pack(at,m,float(meta[m]["price"]))
        }
        vtrades.append(vr)
        opts=[]
        for rm in e["random_pool"]:
            z=meta[rm]
            opts.append({"decision_ts_utc":e["decision_ts_utc"],"market":rm,
                         "results":pack(at,rm,float(z["price"]))})
        event_random_options.append(opts)
        print("PATH",i,"/",len(events),flush=True)

    stop_keys=["no_stop"]+[f"sl_{int(x)}" for x in STOP_PCTS]
    summary={}
    for sk in stop_keys:
        v=stats(vtrades,sk)
        random_stats=[]
        for run in range(RANDOM_RUNS):
            rtr=[]
            rrng=random.Random(SEED+run)
            for opts in event_random_options:
                if opts:rtr.append(rrng.choice(opts))
            random_stats.append(stats(rtr,sk))
        finals=np.array([x.get("compounded_return_pct",0) for x in random_stats],dtype=float)
        vret=v.get("compounded_return_pct")
        percentile=round(100*float(np.mean(finals < vret)),2) if vret is not None and len(finals) else None
        summary[sk]={
            "v76":v,
            "v76_time_split":split_stats(vtrades,sk),
            "random_runs":RANDOM_RUNS,
            "random_compounded_return_median_pct":round(float(np.median(finals)),2),
            "random_compounded_return_p10_pct":round(float(np.quantile(finals,.10)),2),
            "random_compounded_return_p90_pct":round(float(np.quantile(finals,.90)),2),
            "v76_percentile_vs_random":percentile
        }

    result={
        "version":"V76_PRACTICAL_RULES_PREREG_V1",
        "created_at_utc":iso(datetime.now(UTC)),
        "config":{
            "start_kst":START.isoformat(),"end_kst":END.isoformat(),"step_hours":STEP_HOURS,
            "top_n":TOP_N,"ret3_band_pct":[RET3_MIN,RET3_MAX],
            "early_only":True,"chase_excluded":True,
            "repeat_top3_72h":"tie_break_only_not_filter",
            "same_market_cooldown_hours":COOLDOWN_HOURS,
            "hold_days":HOLD_DAYS,"target_pct":TARGET_PCT,
            "stop_pcts":STOP_PCTS,"random_runs":RANDOM_RUNS,
            "fees_slippage_included":False
        },
        "warnings":[
            "Historical replay; not clean prospective OOS.",
            "Current KRW market list creates survivorship bias.",
            "Do not retune the +5~10% ret_3d band after seeing this result.",
            "Random benchmark is paired within the same filtered V7.6 TOP3 pool; it tests ordering/choice, not V7.6 universe discovery versus all KRW coins.",
            "Decision checkpoints default to every 6 hours for runtime; this is evaluation sampling, not a change to live V7.6."
        ],
        "events":events,
        "v76_trades":vtrades,
        "summary":summary
    }
    OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print("DONE",OUT)

if __name__=="__main__":
    main()
