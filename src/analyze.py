import json, math, os, statistics, time
from datetime import datetime, timezone
from pathlib import Path
import requests

BASE = "https://api.upbit.com"
OUT = Path(os.getenv("OUTPUT_JSON", "docs/data/latest.json"))
TOP_N = int(os.getenv("TOP_N", "10"))
EXCLUDE_PUMP_PCT = float(os.getenv("EXCLUDE_PUMP_PCT", "20"))
REQUEST_DELAY = float(os.getenv("REQUEST_DELAY", "0.11"))
LOOKAHEAD_HOURS = int(os.getenv("LOOKAHEAD_HOURS", "6"))
K_NEIGHBORS = int(os.getenv("K_NEIGHBORS", "30"))

STABLE_SYMBOLS = {
    "USDT","USDC","DAI","TUSD","FDUSD","USDE","PYUSD","USDS","USD1","RLUSD","BUSD","USTC"
}

S = requests.Session()
S.headers.update({"Accept":"application/json","User-Agent":"upbit-v2-pattern-scanner/2.0"})

def get(path, params=None, retries=6):
    url = BASE + path
    err = None
    for n in range(retries):
        try:
            r = S.get(url, params=params, timeout=20)
            if r.status_code == 429:
                time.sleep(1.2 + n)
                continue
            r.raise_for_status()
            time.sleep(REQUEST_DELAY)
            return r.json()
        except Exception as e:
            err = e
            time.sleep(min(2**n, 8))
    raise RuntimeError(f"API failed {url} {params}: {err}")

def markets():
    rows = get("/v1/market/all", {"is_details":"true"})
    out=[]
    for x in rows:
        mk=x["market"]
        if not mk.startswith("KRW-"): continue
        sym=mk.split("-",1)[1].upper()
        if sym in STABLE_SYMBOLS: continue
        ev=x.get("market_event") or {}
        if ev.get("warning") is True: continue
        out.append({"market":mk,"symbol":sym,"korean_name":x.get("korean_name",sym),"english_name":x.get("english_name","")})
    return out

def candles(market, unit, count=200):
    rows = get(f"/v1/candles/minutes/{unit}", {"market":market,"count":min(count,200)})
    return list(reversed(rows))

def days(market,count=35):
    rows=get("/v1/candles/days",{"market":market,"count":min(count,200)})
    return list(reversed(rows))

def div(a,b,d=0.0): return a/b if b not in (0,None) else d
def mean(xs): return statistics.fmean(xs) if xs else 0.0
def stdev(xs): return statistics.pstdev(xs) if len(xs)>1 else 0.0

def ema(vals,span):
    if not vals: return 0.0
    a=2/(span+1); v=vals[0]
    for x in vals[1:]: v=a*x+(1-a)*v
    return v

def rsi(vals,p=14):
    if len(vals)<p+1: return 50.0
    ds=[vals[i]-vals[i-1] for i in range(1,len(vals))]
    g=mean([max(x,0) for x in ds[-p:]])
    l=mean([max(-x,0) for x in ds[-p:]])
    if l==0: return 100.0 if g>0 else 50.0
    rs=g/l
    return 100-100/(1+rs)

FEATURES = [
 "ret1","ret3","ret6","ret24","rsi14","ema_gap","vol_ratio",
 "volatility","bb_width","range_pos","breakout_dist","body_strength"
]

def feat(rows,idx=None):
    if idx is None: idx=len(rows)-1
    if idx<30: return None
    w=rows[max(0,idx-40):idx+1]
    c=[float(x["trade_price"]) for x in w]
    h=[float(x["high_price"]) for x in w]
    l=[float(x["low_price"]) for x in w]
    v=[float(x["candle_acc_trade_price"]) for x in w]
    now=c[-1]
    def ret(n): return (div(now,c[-1-n],1)-1)*100 if len(c)>n else 0
    rr=[(div(c[i],c[i-1],1)-1)*100 for i in range(1,len(c))]
    ma20=mean(c[-20:]); sd20=stdev(c[-20:])
    hi20=max(h[-20:]); lo20=min(l[-20:]); prev_hi=max(h[-21:-1])
    op=float(w[-1]["opening_price"]); hi=float(w[-1]["high_price"]); lo=float(w[-1]["low_price"])
    prevvol=v[-24:-3] if len(v)>=24 else v[:-3]
    return {
      "ret1":ret(1),"ret3":ret(3),"ret6":ret(6),"ret24":ret(24),
      "rsi14":rsi(c,14),
      "ema_gap":(div(ema(c[-21:],9),ema(c[-21:],21),1)-1)*100,
      "vol_ratio":div(mean(v[-3:]),mean(prevvol) or mean(v),1),
      "volatility":stdev(rr[-24:]),
      "bb_width":div(4*sd20,ma20,0)*100,
      "range_pos":div(now-lo20,hi20-lo20,0.5)*100,
      "breakout_dist":(div(now,prev_hi,1)-1)*100,
      "body_strength":div(now-op,hi-lo,0)*100
    }

def recent_pump(day_rows):
    mx=-999
    for i in range(1,len(day_rows)):
        d,p=day_rows[i],day_rows[i-1]
        base=min(float(d["opening_price"]),float(p["trade_price"]))
        gain=(div(float(d["high_price"]),base,1)-1)*100
        mx=max(mx,gain)
    return mx>=EXCLUDE_PUMP_PCT,mx

def future_gain(rows,i,hours=LOOKAHEAD_HOURS):
    if i+hours>=len(rows): return None
    base=float(rows[i]["trade_price"])
    hi=max(float(x["high_price"]) for x in rows[i+1:i+1+hours])
    return (div(hi,base,1)-1)*100

def build_examples(rows,market):
    ex=[]
    for i in range(30,len(rows)-LOOKAHEAD_HOURS):
        f=feat(rows,i)
        if not f: continue
        # sample every 2 hours to reduce duplicate neighbors
        if i%2: continue
        g=future_gain(rows,i)
        ex.append({"market":market,"f":f,"gain":g})
    return ex

def robust_stats(vectors):
    out={}
    for k in FEATURES:
        vals=[float(v[k]) for v in vectors]
        med=statistics.median(vals)
        mad=statistics.median([abs(x-med) for x in vals]) or stdev(vals) or 1
        out[k]=(med,mad*1.4826)
    return out

def zvec(f,stats):
    return [(float(f[k])-stats[k][0])/(stats[k][1] or 1) for k in FEATURES]

def distance(a,b):
    # weighted euclidean; volume/RSI/breakout get slightly more weight
    weights=[1,1,1,0.8,1.2,1,1.35,0.8,0.9,1,1.25,0.8]
    return math.sqrt(sum(w*(x-y)**2 for x,y,w in zip(a,b,weights)))

def percentile_rank(values, x):
    if not values: return 0.5
    return sum(v<=x for v in values)/len(values)

def btc_regime():
    try:
        h1=candles("KRW-BTC",60,80)
        f=feat(h1)
        closes=[float(x["trade_price"]) for x in h1]
        e20=ema(closes[-30:],20); e50=ema(closes[-60:],50)
        score=0
        if f["ret6"] > -1.5: score+=1
        if f["ret24"] > -3.0: score+=1
        if e20>=e50: score+=1
        if f["rsi14"]>=42: score+=1
        return {"score":score,"ret6":round(f["ret6"],2),"ret24":round(f["ret24"],2),"rsi1h":round(f["rsi14"],1)}
    except Exception:
        return {"score":2,"ret6":0,"ret24":0,"rsi1h":50}

def vol_metrics(m5,m15,h1,h4):
    return {
      "vol_ratio5m":round(feat(m5)["vol_ratio"],2),
      "vol_ratio15m":round(feat(m15)["vol_ratio"],2),
      "vol_ratio1h":round(feat(h1)["vol_ratio"],2),
      "vol_ratio4h":round(feat(h4)["vol_ratio"],2)
    }

def explain(x):
    why=[]
    if x["hist_p5"]>=35: why.append(f"유사 과거 패턴의 +5% 도달률 {x['hist_p5']:.0f}%")
    if x["hist_p10"]>=15: why.append(f"유사 과거 패턴의 +10% 도달률 {x['hist_p10']:.0f}%")
    if x["vol_ratio5m"]>=1.3 or x["vol_ratio15m"]>=1.3: why.append("단기 거래대금 증가")
    if x["vol_ratio1h"]>=1.1: why.append("1시간 거래대금 유입")
    if 45<=x["rsi1h"]<=68: why.append("1시간 RSI 과열 전 구간")
    if -3<=x["breakout_dist1h"]<=1.5: why.append("최근 고점 부근 압축")
    return why[:4] or ["복합 패턴 점수 상위"]

def main():
    btc=btc_regime()
    ms=markets()
    all_examples=[]
    current=[]
    errors=[]
    for n,m in enumerate(ms,1):
        mk=m["market"]
        try:
            h1=candles(mk,60,200)
            d=days(mk,35)
            excluded,mx=recent_pump(d)
            # Historical examples come from all non-stable markets, including ones currently excluded.
            all_examples.extend(build_examples(h1,mk))
            if excluded:
                continue
            m5=candles(mk,5,120)
            m15=candles(mk,15,120)
            h4=candles(mk,240,120)
            f=feat(h1)
            if not f: continue
            current.append({"meta":m,"f":f,"h1":h1,"m5":m5,"m15":m15,"h4":h4,"max30":mx})
        except Exception as e:
            errors.append(f"{mk}: {e}")
        if n%25==0:
            print(f"{n}/{len(ms)} markets; examples={len(all_examples)} candidates={len(current)}")

    if not current or len(all_examples)<50:
        raise RuntimeError("Insufficient market data")

    stats=robust_stats([e["f"] for e in all_examples])
    exz=[zvec(e["f"],stats) for e in all_examples]

    raw=[]
    for x in current:
        z=zvec(x["f"],stats)
        ds=sorted(((distance(z,ez),i) for i,ez in enumerate(exz)), key=lambda q:q[0])
        neighbors=[all_examples[i] for _,i in ds[:min(K_NEIGHBORS,len(ds))]]
        gains=[e["gain"] for e in neighbors if e["gain"] is not None]
        p5=100*sum(g>=5 for g in gains)/len(gains) if gains else 0
        p10=100*sum(g>=10 for g in gains)/len(gains) if gains else 0
        avg_gain=mean(gains) if gains else 0
        avg_dist=mean([d for d,_ in ds[:min(K_NEIGHBORS,len(ds))]])
        similarity=100/(1+avg_dist)

        vm=vol_metrics(x["m5"],x["m15"],x["h1"],x["h4"])
        f=x["f"]
        # Raw quality score. This will be percentile-normalized across current candidates.
        volume_quality = (
            min(vm["vol_ratio5m"],3)*0.22 +
            min(vm["vol_ratio15m"],3)*0.20 +
            min(vm["vol_ratio1h"],3)*0.18 +
            min(vm["vol_ratio4h"],3)*0.10
        )
        setup=0
        setup += 0.8 if 45<=f["rsi14"]<=68 else (0.25 if 35<=f["rsi14"]<45 else 0)
        setup += 0.6 if -3<=f["breakout_dist"]<=1.5 else 0
        setup += 0.45 if -1.2<=f["ema_gap"]<=2.5 else 0
        setup += 0.35 if f["ret6"]<7 else -0.3
        setup += 0.25 if f["ret24"]<12 else -0.25
        hist = p5*0.025 + p10*0.04 + max(avg_gain,0)*0.05
        raw_score = hist + similarity*0.03 + volume_quality + setup

        # BTC market penalty
        if btc["score"]<=1: raw_score *= 0.78
        elif btc["score"]==2: raw_score *= 0.90

        # Liquidity penalty
        liq=mean([float(r["candle_acc_trade_price"]) for r in x["h1"][-24:]])
        if liq<30_000_000: raw_score*=0.75
        elif liq<80_000_000: raw_score*=0.9

        raw.append({
          "market":x["meta"]["market"],"name":x["meta"]["korean_name"],"english_name":x["meta"]["english_name"],
          "price":float(x["m5"][-1]["trade_price"]),
          "raw_score":raw_score,
          "similarity":round(similarity,1),
          "hist_p5":round(p5,1),"hist_p10":round(p10,1),"hist_avg_gain6h":round(avg_gain,2),
          "max_30d_pump_pct":round(x["max30"],1),
          "rsi15m":round(feat(x["m15"])["rsi14"],1),
          "rsi1h":round(f["rsi14"],1),
          "rsi4h":round(feat(x["h4"])["rsi14"],1),
          "ret1h":round(f["ret1"],2),"ret6h":round(f["ret6"],2),"ret24h":round(f["ret24"],2),
          "breakout_dist1h":round(f["breakout_dist"],2),
          **vm
        })

    vals=[r["raw_score"] for r in raw]
    for r in raw:
        pr=percentile_rank(vals,r["raw_score"])
        # 50~99.5 range prevents fake 100% certainty and preserves rank spread.
        r["score"]=round(50+49.5*pr,1)
        r["why"]=explain(r)
        del r["raw_score"]

    raw.sort(key=lambda q:(q["score"],q["hist_p10"],q["hist_p5"],q["similarity"]), reverse=True)

    payload={
      "generated_at_utc":datetime.now(timezone.utc).isoformat(),
      "config":{
        "exclude_pump_pct":EXCLUDE_PUMP_PCT,
        "lookahead_hours":LOOKAHEAD_HOURS,
        "neighbors":K_NEIGHBORS,
        "top_n":TOP_N,
        "historical_examples":len(all_examples),
        "markets_considered":len(ms),
        "current_candidates":len(current)
      },
      "btc_regime":btc,
      "results":raw[:TOP_N],
      "errors":errors[:30],
      "disclaimer":"score는 후보 간 상대 순위 점수입니다. +5%/+10% 수치는 최근 1시간봉 데이터의 유사 과거 패턴에서 향후 6시간 내 해당 상승폭을 기록한 관측 비율이며 미래 확률을 보장하지 않습니다."
    }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(payload,ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
