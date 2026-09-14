#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json, os, time, random, math
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
import requests

API = "https://api.upbit.com"
KST = timezone(timedelta(hours=9))
UTC = timezone.utc

START = datetime.strptime(os.getenv("BT_START_KST", "2026-03-02 09:00"), "%Y-%m-%d %H:%M").replace(tzinfo=KST)
END = datetime.strptime(os.getenv("BT_END_KST", "2026-09-14 09:00"), "%Y-%m-%d %H:%M").replace(tzinfo=KST)
WEEKDAY = int(os.getenv("BT_WEEKDAY", "0"))       # Monday=0
HOUR_KST = int(os.getenv("BT_HOUR_KST", "9"))
TOP_K = int(os.getenv("BT_TOP_K", "5"))
DELAY = float(os.getenv("BT_REQUEST_DELAY", "0.12"))
INITIAL_CAPITAL = float(os.getenv("BT_INITIAL_CAPITAL", "15000000"))
TARGET_PCT = float(os.getenv("BT_TARGET_PCT", "20"))
STOP_PCTS = tuple(float(x) for x in os.getenv("BT_STOP_PCTS", "3,5,7,10").split(","))
RANDOM_RUNS = int(os.getenv("BT_RANDOM_RUNS", "1000"))
SEED = int(os.getenv("BT_RANDOM_SEED", "7605"))

OUT = Path("docs/data/v76_weekly_benchmark_backtest_latest.json")
CACHE = Path(".cache/v76_weekly_benchmark")
OUT.parent.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)

STABLE={"USDT","USDC","DAI","USD1","USDE","FDUSD","TUSD","RLUSD","EURC"}

S=requests.Session()
S.headers.update({"User-Agent":"upbit-v76-weekly-benchmark-backtest/1.0"})


def get(path, params=None, tries=7):
    last=None
    for i in range(tries):
        try:
            r=S.get(API+path,params=params,timeout=25)
            if r.status_code==429:
                time.sleep(0.8*(i+1))
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last=e
            time.sleep(0.8*(i+1))
    raise last


def iso(dt):
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

def dt_utc(s):
    return datetime.fromisoformat(s).replace(tzinfo=UTC)

def pct(a,b):
    return (a/b-1)*100 if b else 0.0

def clamp(x,a,b):
    return max(a,min(b,x))


def markets():
    xs=get("/v1/market/all", {"is_details":"false"})
    return sorted(
        x["market"] for x in xs
        if x["market"].startswith("KRW-")
        and x["market"].split("-",1)[1] not in STABLE
    )


def weekly_dates():
    cur=START.replace(hour=HOUR_KST,minute=0,second=0,microsecond=0)
    while cur.weekday()!=WEEKDAY or cur<START:
        cur += timedelta(days=1)
        cur=cur.replace(hour=HOUR_KST,minute=0,second=0,microsecond=0)
    out=[]
    while cur<=END:
        out.append(cur)
        cur += timedelta(days=7)
    return out


def fetch_5m_before(m, at_utc, count=30):
    p=CACHE/f"pre_{m.replace('-','_')}_{at_utc.strftime('%Y%m%dT%H%M')}.json"
    if p.exists():
        try:return json.loads(p.read_text(encoding="utf-8"))
        except:pass
    rows=get("/v1/candles/minutes/5", {
        "market":m, "count":count, "to":iso(at_utc+timedelta(seconds=1))
    })
    rows=sorted(rows,key=lambda x:x["candle_date_time_utc"])
    p.write_text(json.dumps(rows,ensure_ascii=False),encoding="utf-8")
    time.sleep(DELAY)
    return rows


def fetch_5m_range(m, start_utc, end_utc):
    p=CACHE/f"path_{m.replace('-','_')}_{start_utc.strftime('%Y%m%dT%H%M')}_{end_utc.strftime('%Y%m%dT%H%M')}.json"
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
                seen.add(x["candle_date_time_utc"])
                out.append(x)
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

    r5=pct(cl[-1],cl[-2])
    r15=pct(cl[-1],cl[-4])
    r30=pct(cl[-1],cl[-7])
    r60=pct(cl[-1],cl[-13])

    vr=tv[-1]/((tv[-7:-1].sum()/6) or 1)
    a15=(tv[-3:].sum()/3)/((tv[-9:-3].sum()/6) or 1)
    a30=(tv[-6:].sum()/6)/((tv[-18:-6].sum()/12) or 1)

    score=35 \
        + clamp((vr-1)*12,-8,18) \
        + clamp((a15-1)*18,-10,25) \
        + clamp((a30-1)*10,-6,14) \
        + clamp(r5*3,-8,8) \
        + clamp(r15*1.8,-8,10)

    over=0
    if r5>2.5: over+=(r5-2.5)*5
    if r15>4: over+=(r15-4)*4
    if r60>7: over+=(r60-7)*2
    score-=clamp(over,0,35)

    chase=(r5>=3) or (r15>=5) or (r60>=8)
    early=(not chase) and a15>=1.25 and vr>=1.15
    label="CHASE" if chase else ("EARLY" if early else "WATCH")

    return {
        "price":float(cl[-1]),
        "score":round(clamp(score,0,100),2),
        "label":label,
        "ret_5m":round(r5,3),
        "ret_15m":round(r15,3),
        "ret_30m":round(r30,3),
        "ret_60m":round(r60,3),
        "value_ratio_5m":round(float(vr),3),
        "value_accel_15m":round(float(a15),3),
        "value_accel_30m":round(float(a30),3),
        "trade_value_5m":round(float(tv[-1]),3),
    }


def replay(rows, entry, target_pct=20.0, stop_pct=None):
    if not rows:
        return None
    tp=entry*(1+target_pct/100)
    sl=entry*(1-stop_pct/100) if stop_pct is not None else None

    mfe=-1e9
    mae=1e9
    for x in rows:
        hi=float(x["high_price"])
        lo=float(x["low_price"])
        mfe=max(mfe,pct(hi,entry))
        mae=min(mae,pct(lo,entry))
        hit_tp=hi>=tp
        hit_sl=(sl is not None and lo<=sl)

        # Conservative rule for OHLC ambiguity.
        if hit_tp and hit_sl:
            return {
                "outcome":"stop_first_conservative",
                "return_pct":-float(stop_pct),
                "ambiguous_same_bar":True,
                "mfe_pct":round(mfe,3),
                "mae_pct":round(mae,3)
            }
        if hit_sl:
            return {
                "outcome":"stop_first",
                "return_pct":-float(stop_pct),
                "ambiguous_same_bar":False,
                "mfe_pct":round(mfe,3),
                "mae_pct":round(mae,3)
            }
        if hit_tp:
            return {
                "outcome":"target_first",
                "return_pct":float(target_pct),
                "ambiguous_same_bar":False,
                "mfe_pct":round(mfe,3),
                "mae_pct":round(mae,3)
            }

    last=float(rows[-1]["trade_price"])
    return {
        "outcome":"timeout",
        "return_pct":round(pct(last,entry),3),
        "ambiguous_same_bar":False,
        "mfe_pct":round(mfe,3),
        "mae_pct":round(mae,3)
    }


def max_drawdown(equity):
    peak=-1
    mdd=0.0
    for v in equity:
        peak=max(peak,v)
        if peak>0:
            mdd=min(mdd,(v/peak-1)*100)
    return round(mdd,3)


def simulate_portfolio(weeks, selector_key, stop_key):
    cap=INITIAL_CAPITAL
    equity=[cap]
    weekly=[]
    all_rets=[]
    tgt=stp=timeout=amb=0

    for w in weeks:
        picks=w[selector_key]
        vals=[p["results"][stop_key] for p in picks if p["results"].get(stop_key)]
        if not vals:
            continue
        rets=[x["return_pct"] for x in vals]
        wr=float(np.mean(rets))  # equal weight
        cap*=1+wr/100
        equity.append(cap)
        all_rets.extend(rets)

        tgt+=sum(x["outcome"]=="target_first" for x in vals)
        stp+=sum(x["outcome"] in ("stop_first","stop_first_conservative") for x in vals)
        timeout+=sum(x["outcome"]=="timeout" for x in vals)
        amb+=sum(x.get("ambiguous_same_bar",False) for x in vals)
        weekly.append({
            "decision_ts_kst":w["decision_ts_kst"],
            "portfolio_return_pct":round(wr,3),
            "capital":round(cap,0)
        })

    n=len(all_rets)
    return {
        "weeks":len(weekly),
        "trades":n,
        "target_first_rate":round(100*tgt/n,2) if n else None,
        "stop_first_rate":round(100*stp/n,2) if n else None,
        "timeout_rate":round(100*timeout/n,2) if n else None,
        "avg_trade_return_pct":round(float(np.mean(all_rets)),3) if n else None,
        "median_trade_return_pct":round(float(np.median(all_rets)),3) if n else None,
        "avg_weekly_return_pct":round(float(np.mean([x["portfolio_return_pct"] for x in weekly])),3) if weekly else None,
        "final_capital":round(cap,0),
        "total_return_pct":round((cap/INITIAL_CAPITAL-1)*100,2),
        "max_drawdown_pct":max_drawdown(equity),
        "weekly_equity":weekly,
    }


def btc_weekly(weeks, stop_key):
    cap=INITIAL_CAPITAL
    equity=[cap]
    all_rets=[]
    weekly=[]
    for w in weeks:
        x=w["btc"]["results"][stop_key]
        r=x["return_pct"]
        cap*=1+r/100
        equity.append(cap)
        all_rets.append(r)
        weekly.append({
            "decision_ts_kst":w["decision_ts_kst"],
            "return_pct":round(r,3),
            "capital":round(cap,0)
        })
    return {
        "weeks":len(weekly),
        "trades":len(all_rets),
        "avg_trade_return_pct":round(float(np.mean(all_rets)),3) if all_rets else None,
        "median_trade_return_pct":round(float(np.median(all_rets)),3) if all_rets else None,
        "final_capital":round(cap,0),
        "total_return_pct":round((cap/INITIAL_CAPITAL-1)*100,2),
        "max_drawdown_pct":max_drawdown(equity),
        "weekly_equity":weekly
    }


def summarize_random(random_results):
    finals=np.array([x["final_capital"] for x in random_results],dtype=float)
    rets=np.array([x["total_return_pct"] for x in random_results],dtype=float)
    mdds=np.array([x["max_drawdown_pct"] for x in random_results],dtype=float)
    return {
        "runs":len(random_results),
        "final_capital_mean":round(float(finals.mean()),0),
        "final_capital_median":round(float(np.median(finals)),0),
        "final_capital_p10":round(float(np.quantile(finals,0.10)),0),
        "final_capital_p90":round(float(np.quantile(finals,0.90)),0),
        "total_return_mean_pct":round(float(rets.mean()),2),
        "total_return_median_pct":round(float(np.median(rets)),2),
        "max_drawdown_median_pct":round(float(np.median(mdds)),2),
    }


def main():
    rng=random.Random(SEED)
    ms=markets()
    weeks=weekly_dates()
    print("markets",len(ms),"weeks",len(weeks))

    raw_weeks=[]

    for wi,wk in enumerate(weeks,1):
        at=wk.astimezone(UTC)
        print(f"WEEK {wi}/{len(weeks)} {wk.isoformat()}")

        scan=[]
        for m in ms:
            try:
                rows=fetch_5m_before(m,at,30)
                z=v76(rows)
                if z:
                    z["market"]=m
                    scan.append(z)
            except Exception as e:
                print("scan_error",m,str(e)[:100])

        pri={"EARLY":0,"WATCH":1,"CHASE":2}
        v76_ranked=sorted(scan,key=lambda x:(pri[x["label"]],-x["score"],-x["value_accel_15m"]))
        tv_ranked=sorted(scan,key=lambda x:-x["trade_value_5m"])

        # Current universe for random benchmark: all coins with enough data at that moment.
        universe=[x["market"] for x in scan]

        # Cache per-market 7d path/replay once for all benchmark selectors.
        replay_cache={}
        def pack(m, meta=None):
            if m not in replay_cache:
                entry_meta=next(x for x in scan if x["market"]==m)
                entry=entry_meta["price"]
                path=fetch_5m_range(m,at,at+timedelta(days=7))
                results={}
                results["no_stop"]=replay(path,entry,TARGET_PCT,None)
                for s in STOP_PCTS:
                    key=f"sl_{int(s) if s.is_integer() else s}"
                    results[key]=replay(path,entry,TARGET_PCT,s)
                replay_cache[m]={
                    "market":m,
                    "entry_price":entry,
                    "score":entry_meta["score"],
                    "label":entry_meta["label"],
                    "trade_value_5m":entry_meta["trade_value_5m"],
                    "results":results
                }
            return replay_cache[m]

        v76_picks=[pack(x["market"]) for x in v76_ranked[:TOP_K]]
        tv_picks=[pack(x["market"]) for x in tv_ranked[:TOP_K]]

        # Random benchmark stores picks for every run for this week.
        random_picks=[]
        for _ in range(RANDOM_RUNS):
            chosen=rng.sample(universe,TOP_K)
            random_picks.append([pack(m) for m in chosen])

        # BTC benchmark
        btc_m="KRW-BTC"
        btc=pack(btc_m)

        raw_weeks.append({
            "decision_ts_kst":wk.isoformat(),
            "universe_size":len(universe),
            "v76_top5":v76_picks,
            "trade_value_top5":tv_picks,
            "random_runs":random_picks,
            "btc":btc,
        })

    stop_keys=["no_stop"]+[f"sl_{int(s) if s.is_integer() else s}" for s in STOP_PCTS]
    summary={}

    for sk in stop_keys:
        v76_res=simulate_portfolio(raw_weeks,"v76_top5",sk)
        tv_res=simulate_portfolio(raw_weeks,"trade_value_top5",sk)
        btc_res=btc_weekly(raw_weeks,sk)

        random_results=[]
        for run_idx in range(RANDOM_RUNS):
            cap=INITIAL_CAPITAL
            eq=[cap]
            allrets=[]
            for w in raw_weeks:
                picks=w["random_runs"][run_idx]
                rets=[p["results"][sk]["return_pct"] for p in picks]
                wr=float(np.mean(rets))
                cap*=1+wr/100
                eq.append(cap)
                allrets.extend(rets)
            random_results.append({
                "final_capital":round(cap,0),
                "total_return_pct":round((cap/INITIAL_CAPITAL-1)*100,2),
                "max_drawdown_pct":max_drawdown(eq),
            })

        random_summary=summarize_random(random_results)
        v76_beat_random_pct=round(
            100*sum(v76_res["final_capital"] > r["final_capital"] for r in random_results)/RANDOM_RUNS,2
        )

        summary[sk]={
            "v76_top5":v76_res,
            "trade_value_top5":tv_res,
            "btc":btc_res,
            "random_top5":random_summary,
            "v76_percentile_vs_random":v76_beat_random_pct,
            "v76_excess_return_vs_random_median_pct":round(
                v76_res["total_return_pct"]-random_summary["total_return_median_pct"],2
            ),
            "v76_excess_return_vs_trade_value_pct":round(
                v76_res["total_return_pct"]-tv_res["total_return_pct"],2
            ),
            "v76_excess_return_vs_btc_pct":round(
                v76_res["total_return_pct"]-btc_res["total_return_pct"],2
            ),
        }

    # Strip huge random weekly detail from output; keep actual selected weeks for V7.6/TV/BTC.
    compact_weeks=[]
    for w in raw_weeks:
        compact_weeks.append({
            "decision_ts_kst":w["decision_ts_kst"],
            "universe_size":w["universe_size"],
            "v76_top5":w["v76_top5"],
            "trade_value_top5":w["trade_value_top5"],
            "btc":w["btc"],
        })

    result={
        "version":"V76_WEEKLY_BENCHMARK_V1",
        "config":{
            "start_kst":START.isoformat(),
            "end_kst":END.isoformat(),
            "weekday":WEEKDAY,
            "hour_kst":HOUR_KST,
            "top_k":TOP_K,
            "initial_capital_krw":INITIAL_CAPITAL,
            "target_pct":TARGET_PCT,
            "stop_pcts":list(STOP_PCTS),
            "random_runs":RANDOM_RUNS,
            "random_seed":SEED,
            "benchmarks":[
                "V7.6 TOP5 equal weight",
                "Random 5 equal weight (Monte Carlo)",
                "5-minute trade-value TOP5 equal weight",
                "KRW-BTC"
            ],
            "selection_time":"Every Monday 09:00 KST",
            "holding_rule":"Exit at +20%, tested stop, or after 7 days. Rebalance next Monday.",
            "same_bar_rule":"If TP and SL both touch inside one 5m candle, count stop first.",
            "fees_slippage":"excluded",
            "warning":"Historical replay, not clean OOS. Uses today's available KRW market list, so survivorship bias remains."
        },
        "summary":summary,
        "weeks":compact_weeks
    }

    OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
