import json, math, os, statistics, time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import requests

BASE = "https://api.upbit.com"
OUT = Path(os.getenv("OUTPUT_JSON", "docs/data/latest.json"))
TOP_N = int(os.getenv("TOP_N", "10"))
EXCLUDE_PUMP_PCT = float(os.getenv("EXCLUDE_PUMP_PCT", "20"))
TRAIN_PUMP_PCT = float(os.getenv("TRAIN_PUMP_PCT", "10"))
REQUEST_DELAY = float(os.getenv("REQUEST_DELAY", "0.12"))

S = requests.Session()
S.headers.update({"Accept": "application/json", "User-Agent": "upbit-hourly-pattern-scanner/1.0"})

def get(path, params=None, retries=5):
    url = BASE + path
    last = None
    for n in range(retries):
        try:
            r = S.get(url, params=params, timeout=15)
            if r.status_code == 429:
                time.sleep(1.5 + n)
                continue
            r.raise_for_status()
            time.sleep(REQUEST_DELAY)
            return r.json()
        except Exception as e:
            last = e
            time.sleep(min(2 ** n, 8))
    raise RuntimeError(f"API failed: {url} {params}: {last}")

def markets():
    rows = get("/v1/market/all", {"is_details": "true"})
    out = []
    for x in rows:
        if not x["market"].startswith("KRW-"):
            continue
        ev = x.get("market_event") or {}
        if ev.get("warning") is True:
            continue
        out.append({
            "market": x["market"],
            "korean_name": x.get("korean_name", x["market"]),
            "english_name": x.get("english_name", ""),
        })
    return out

def candles(market, unit, count):
    rows = get(f"/v1/candles/minutes/{unit}", {"market": market, "count": min(count, 200)})
    return list(reversed(rows))

def day_candles(market, count=35):
    rows = get("/v1/candles/days", {"market": market, "count": min(count, 200)})
    return list(reversed(rows))

def safe_div(a,b,default=0.0):
    return a/b if b not in (0, None) else default

def mean(xs):
    return statistics.fmean(xs) if xs else 0.0

def stdev(xs):
    return statistics.pstdev(xs) if len(xs) > 1 else 0.0

def ema(values, span):
    if not values: return 0.0
    alpha = 2/(span+1)
    v = values[0]
    for x in values[1:]:
        v = alpha*x + (1-alpha)*v
    return v

def rsi(values, period=14):
    if len(values) < period+1: return 50.0
    ds = [values[i]-values[i-1] for i in range(1,len(values))]
    gains = [max(d,0) for d in ds[-period:]]
    losses = [max(-d,0) for d in ds[-period:]]
    ag, al = mean(gains), mean(losses)
    if al == 0: return 100.0 if ag > 0 else 50.0
    rs = ag/al
    return 100 - 100/(1+rs)

FEATURES = [
    "ret1","ret3","ret6","ret24","rsi14","ema_gap","vol_ratio",
    "volatility","bb_width","range_pos","breakout_dist","body_strength"
]

def feature_at(rows, idx=None):
    if idx is None: idx = len(rows)-1
    if idx < 25: return None
    w = rows[max(0,idx-30):idx+1]
    closes = [float(x["trade_price"]) for x in w]
    highs = [float(x["high_price"]) for x in w]
    lows = [float(x["low_price"]) for x in w]
    vols = [float(x["candle_acc_trade_price"]) for x in w]
    c = closes[-1]
    def ret(n):
        return (safe_div(c, closes[-1-n],1)-1)*100 if len(closes)>n else 0
    rets = [(safe_div(closes[i],closes[i-1],1)-1)*100 for i in range(1,len(closes))]
    ma20 = mean(closes[-20:])
    sd20 = stdev(closes[-20:])
    hi20, lo20 = max(highs[-20:]), min(lows[-20:])
    prev_hi20 = max(highs[-21:-1])
    op = float(w[-1]["opening_price"])
    hi = float(w[-1]["high_price"])
    lo = float(w[-1]["low_price"])
    return {
        "ret1": ret(1), "ret3": ret(3), "ret6": ret(6), "ret24": ret(24),
        "rsi14": rsi(closes,14),
        "ema_gap": (safe_div(ema(closes[-21:],9), ema(closes[-21:],21),1)-1)*100,
        "vol_ratio": safe_div(mean(vols[-3:]), mean(vols[-24:-3]) or mean(vols[-24:]), 1),
        "volatility": stdev(rets[-24:]),
        "bb_width": safe_div(4*sd20, ma20,0)*100,
        "range_pos": safe_div(c-lo20, hi20-lo20,0.5)*100,
        "breakout_dist": (safe_div(c,prev_hi20,1)-1)*100,
        "body_strength": safe_div(c-op, hi-lo,0)*100,
    }

def detect_recent_pump(day_rows):
    """True if any of last ~30 daily candles showed >= threshold intraday/close expansion."""
    if len(day_rows) < 2: return False, 0.0
    mx = -999
    for i in range(1,len(day_rows)):
        d, p = day_rows[i], day_rows[i-1]
        base = min(float(d["opening_price"]), float(p["trade_price"]))
        gain = (safe_div(float(d["high_price"]), base,1)-1)*100
        mx = max(mx, gain)
    return mx >= EXCLUDE_PUMP_PCT, mx

def pump_examples(hour_rows, market):
    """Feature snapshots immediately before a >= TRAIN_PUMP_PCT forward 6h move."""
    out = []
    last_idx = -99
    for i in range(30, len(hour_rows)-6):
        if i-last_idx < 12:
            continue
        f = feature_at(hour_rows,i)
        if not f: continue
        base = float(hour_rows[i]["trade_price"])
        fwd_hi = max(float(x["high_price"]) for x in hour_rows[i+1:i+7])
        gain = (safe_div(fwd_hi,base,1)-1)*100
        # Avoid examples already in obvious vertical pumps.
        if gain >= TRAIN_PUMP_PCT and f["ret6"] < 7 and f["ret24"] < 12:
            out.append({"market":market,"gain":gain,"f":f})
            last_idx = i
    return out

def robust_stats(vectors):
    stats = {}
    for k in FEATURES:
        vals = sorted(float(v[k]) for v in vectors)
        med = statistics.median(vals)
        absdev = [abs(x-med) for x in vals]
        mad = statistics.median(absdev) or stdev(vals) or 1.0
        stats[k]=(med, mad*1.4826)
    return stats

def zvec(f, stats):
    return [(float(f[k])-stats[k][0])/(stats[k][1] or 1) for k in FEATURES]

def cosine(a,b):
    dot=sum(x*y for x,y in zip(a,b))
    na=math.sqrt(sum(x*x for x in a)); nb=math.sqrt(sum(y*y for y in b))
    return dot/(na*nb) if na and nb else 0.0

def current_context(m15,m60,m240):
    f15, f60, f240 = feature_at(m15), feature_at(m60), feature_at(m240)
    if not all([f15,f60,f240]): return {}
    return {
        "price": float(m15[-1]["trade_price"]),
        "rsi15m": round(f15["rsi14"],1),
        "rsi1h": round(f60["rsi14"],1),
        "rsi4h": round(f240["rsi14"],1),
        "vol_ratio15m": round(f15["vol_ratio"],2),
        "ret1h": round(f60["ret1"],2),
        "ret6h": round(f60["ret6"],2),
        "ret24h": round(f60["ret24"],2),
        "breakout_dist1h": round(f60["breakout_dist"],2),
    }

def explain(f, sim):
    why=[]
    if 0.8 <= f["vol_ratio"] <= 2.8: why.append("거래대금이 과열 전 단계")
    if 45 <= f["rsi14"] <= 68: why.append("1시간 RSI가 중립~강세")
    if -2.5 <= f["breakout_dist"] <= 1.5: why.append("최근 20시간 고점 부근 압축")
    if f["ema_gap"] > -1.0: why.append("단기/중기 이평 괴리 양호")
    if f["ret6"] < 7: why.append("직전 6시간 과열 제한")
    if sim >= 0.65: why.append("최근 급등 직전 패턴과 높은 유사도")
    return why[:4] or ["복합 패턴 점수 상위"]

def main():
    ms=markets()
    collected={}
    examples=[]
    errors=[]
    for n,m in enumerate(ms,1):
        mk=m["market"]
        try:
            h1=candles(mk,60,168)      # recent 7 days
            d=day_candles(mk,35)       # ~1 month exclusion
            m15=candles(mk,15,120)
            h4=candles(mk,240,120)
            excluded,max30=detect_recent_pump(d)
            f=feature_at(h1)
            if f:
                examples.extend(pump_examples(h1,mk))
                collected[mk]={"meta":m,"h1":h1,"m15":m15,"h4":h4,"f":f,
                               "excluded":excluded,"max30":max30}
        except Exception as e:
            errors.append(f"{mk}: {e}")
        if n % 25 == 0:
            print(f"collected {n}/{len(ms)} markets, examples={len(examples)}")

    candidates=[v for v in collected.values() if not v["excluded"]]
    if not candidates:
        raise RuntimeError("No candidates collected")
    train_vectors=[x["f"] for x in examples]
    # If the market was quiet, use candidate distribution only for normalization;
    # similarity itself will be neutral rather than fabricated.
    stats=robust_stats(train_vectors + [x["f"] for x in candidates])

    exz=[zvec(x["f"],stats) for x in examples]
    scored=[]
    for x in candidates:
        z=zvec(x["f"],stats)
        sims=sorted((cosine(z,e) for e in exz), reverse=True)
        top_sim=mean(sims[:min(5,len(sims))]) if sims else 0.0
        f=x["f"]
        # Pattern score, not a statistically validated probability.
        sim_component=max(0,min(1,(top_sim+1)/2))
        setup=0.0
        setup += 0.18 if 42<=f["rsi14"]<=68 else 0
        setup += 0.16 if 0.8<=f["vol_ratio"]<=3.0 else 0
        setup += 0.15 if -3<=f["breakout_dist"]<=1.5 else 0
        setup += 0.12 if -1.2<=f["ema_gap"]<=2.5 else 0
        setup += 0.10 if f["ret6"]<7 else 0
        setup += 0.09 if f["ret24"]<12 else 0
        score=100*(0.55*sim_component + setup)
        score=max(0,min(100,score))
        # Liquidity sanity check from recent 1h traded value.
        liq=mean([float(r["candle_acc_trade_price"]) for r in x["h1"][-24:]])
        if liq < 30_000_000:
            score*=0.8
        ctx=current_context(x["m15"],x["h1"],x["h4"])
        scored.append({
            "market":x["meta"]["market"],
            "name":x["meta"]["korean_name"],
            "english_name":x["meta"]["english_name"],
            "score":round(score,1),
            "similarity":round(top_sim*100,1),
            "max_30d_pump_pct":round(x["max30"],1),
            **ctx,
            "why":explain(f,top_sim),
        })
    scored.sort(key=lambda x:x["score"], reverse=True)
    payload={
        "generated_at_utc":datetime.now(timezone.utc).isoformat(),
        "config":{
            "exclude_pump_pct":EXCLUDE_PUMP_PCT,
            "training_forward_pump_pct":TRAIN_PUMP_PCT,
            "top_n":TOP_N,
            "training_examples":len(examples),
            "markets_analyzed":len(collected),
        },
        "results":scored[:TOP_N],
        "errors":errors[:30],
        "disclaimer":"패턴 점수는 통계적으로 검증된 상승확률이 아니며 투자수익을 보장하지 않습니다."
    }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(payload,ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
