import json, math, os, statistics, time
from datetime import datetime, timezone
from pathlib import Path
import requests

BASE = "https://api.upbit.com"
OUT = Path(os.getenv("OUTPUT_JSON", "docs/data/latest.json"))
TOP_N = int(os.getenv("TOP_N", "10"))
EXCLUDE_PUMP_PCT = float(os.getenv("EXCLUDE_PUMP_PCT", "20"))
LOOKAHEAD_HOURS = int(os.getenv("LOOKAHEAD_HOURS", "6"))
K_NEIGHBORS = int(os.getenv("K_NEIGHBORS", "40"))
REQUEST_DELAY = float(os.getenv("REQUEST_DELAY", "0.11"))

STABLE_SYMBOLS = {
    "USDT","USDC","DAI","TUSD","FDUSD","USDE","PYUSD","USDS","USD1","RLUSD","BUSD","USTC"
}

S = requests.Session()
S.headers.update({"Accept":"application/json","User-Agent":"upbit-v3-breakout-scanner/3.0"})

def get(path, params=None, retries=6):
    err=None
    for n in range(retries):
        try:
            r=S.get(BASE+path, params=params, timeout=20)
            if r.status_code==429:
                time.sleep(1.2+n)
                continue
            r.raise_for_status()
            time.sleep(REQUEST_DELAY)
            return r.json()
        except Exception as e:
            err=e
            time.sleep(min(2**n,8))
    raise RuntimeError(f"API failed {path} {params}: {err}")

def markets():
    rows=get("/v1/market/all",{"is_details":"true"})
    out=[]
    for x in rows:
        mk=x["market"]
        if not mk.startswith("KRW-"): continue
        sym=mk.split("-",1)[1].upper()
        if sym in STABLE_SYMBOLS: continue
        ev=x.get("market_event") or {}
        if ev.get("warning") is True: continue
        out.append({
            "market":mk,"symbol":sym,
            "korean_name":x.get("korean_name",sym),
            "english_name":x.get("english_name","")
        })
    return out

def candles(market, unit, count=200):
    rows=get(f"/v1/candles/minutes/{unit}",{"market":market,"count":min(count,200)})
    return list(reversed(rows))

def days(market,count=35):
    rows=get("/v1/candles/days",{"market":market,"count":min(count,200)})
    return list(reversed(rows))

def div(a,b,d=0.0): return a/b if b not in (0,None) else d
def mean(xs): return statistics.fmean(xs) if xs else 0.0
def stdev(xs): return statistics.pstdev(xs) if len(xs)>1 else 0.0
def clamp(x,a,b): return max(a,min(b,x))

def ema(vals,span):
    if not vals: return 0.0
    alpha=2/(span+1); v=vals[0]
    for x in vals[1:]:
        v=alpha*x+(1-alpha)*v
    return v

def rsi(vals,p=14):
    if len(vals)<p+1: return 50.0
    ds=[vals[i]-vals[i-1] for i in range(1,len(vals))]
    g=mean([max(x,0) for x in ds[-p:]])
    l=mean([max(-x,0) for x in ds[-p:]])
    if l==0: return 100.0 if g>0 else 50.0
    rs=g/l
    return 100-100/(1+rs)

FEATURES=[
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
    def ret(n): return (div(now,c[-1-n],1)-1)*100 if len(c)>n else 0.0
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
    out=[]
    # Sample every 3 hours to reduce highly overlapping observations.
    for i in range(30,len(rows)-LOOKAHEAD_HOURS):
        if i%3: continue
        f=feat(rows,i)
        if not f: continue
        g=future_gain(rows,i)
        if g is None: continue
        out.append({"market":market,"f":f,"gain":g})
    return out

def robust_stats(vectors):
    out={}
    for k in FEATURES:
        vals=[float(v[k]) for v in vectors]
        med=statistics.median(vals)
        mad=statistics.median([abs(x-med) for x in vals]) or stdev(vals) or 1.0
        out[k]=(med,mad*1.4826)
    return out

def zvec(f,stats):
    return [(float(f[k])-stats[k][0])/(stats[k][1] or 1) for k in FEATURES]

WEIGHTS=[1.0,1.0,1.0,0.8,1.2,1.0,1.4,0.8,0.9,1.0,1.3,0.8]
def distance(a,b):
    return math.sqrt(sum(w*(x-y)**2 for x,y,w in zip(a,b,WEIGHTS)))

def current_micro(m5,m15,h1,h4):
    f5,f15,f1,f4=feat(m5),feat(m15),feat(h1),feat(h4)
    return {
      "price":float(m5[-1]["trade_price"]),
      "vol_ratio5m":round(f5["vol_ratio"],2),
      "vol_ratio15m":round(f15["vol_ratio"],2),
      "vol_ratio1h":round(f1["vol_ratio"],2),
      "vol_ratio4h":round(f4["vol_ratio"],2),
      "rsi15m":round(f15["rsi14"],1),
      "rsi1h":round(f1["rsi14"],1),
      "rsi4h":round(f4["rsi14"],1),
      "ret1h":round(f1["ret1"],2),
      "ret6h":round(f1["ret6"],2),
      "ret24h":round(f1["ret24"],2),
      "breakout_dist1h":round(f1["breakout_dist"],2),
      "ema_gap1h":round(f1["ema_gap"],2),
      "range_pos1h":round(f1["range_pos"],1),
      "bb_width1h":round(f1["bb_width"],2),
    }

def btc_regime():
    try:
        h1=candles("KRW-BTC",60,80)
        f=feat(h1)
        closes=[float(x["trade_price"]) for x in h1]
        e20=ema(closes[-30:],20); e50=ema(closes[-60:],50)
        score=0
        if f["ret6"]>-1.5: score+=1
        if f["ret24"]>-3.0: score+=1
        if e20>=e50: score+=1
        if f["rsi14"]>=42: score+=1
        return {"score":score,"ret6":round(f["ret6"],2),"ret24":round(f["ret24"],2),"rsi1h":round(f["rsi14"],1)}
    except Exception:
        return {"score":2,"ret6":0,"ret24":0,"rsi1h":50}

def readiness_score(m, hist_p3, hist_p5, hist_p10, similarity, btc_score):
    # 0-100 setup score. Not a probability.
    # A coin with zero historical +5/+10 success cannot reach the top purely on short volume.
    vol_accel = (
        clamp((m["vol_ratio5m"]-0.8)/1.8,0,1)*12 +
        clamp((m["vol_ratio15m"]-0.8)/1.8,0,1)*10 +
        clamp((m["vol_ratio1h"]-0.7)/1.6,0,1)*7 +
        clamp((m["vol_ratio4h"]-0.7)/1.6,0,1)*4
    )
    # Reward volume arriving earlier than price acceleration.
    early_flow=0
    if m["vol_ratio5m"]>=1.4 and m["vol_ratio15m"]>=1.2 and m["ret1h"]<2.5:
        early_flow+=8
    if m["vol_ratio15m"]>m["vol_ratio1h"]*1.25 and m["ret1h"]<2.0:
        early_flow+=5

    technical=0
    if 45<=m["rsi1h"]<=66: technical+=8
    elif 40<=m["rsi1h"]<45 or 66<m["rsi1h"]<=70: technical+=4
    if -3.0<=m["breakout_dist1h"]<=1.2: technical+=7
    if -1.0<=m["ema_gap1h"]<=2.5: technical+=5
    if 45<=m["range_pos1h"]<=92: technical+=3
    if m["ret6h"]<6: technical+=3
    if m["ret24h"]<10: technical+=2

    history = hist_p3*0.18 + hist_p5*0.22 + hist_p10*0.20
    history = clamp(history,0,25)
    similarity_part=clamp(similarity/100,0,1)*8

    score=vol_accel+early_flow+technical+history+similarity_part
    if hist_p5==0 and hist_p10==0:
        score=min(score,74)
    if m["rsi1h"]>72 or m["rsi4h"]>76:
        score-=8
    if m["ret6h"]>8:
        score-=8
    if btc_score<=1: score*=0.78
    elif btc_score==2: score*=0.90
    return round(clamp(score,0,100),1)

def explain(x):
    why=[]
    if x["early_volume_signal"]: why.append("가격보다 단기 거래대금이 먼저 증가")
    if x["hist_p3"]>=30: why.append(f"유사패턴 +3% 관측률 {x['hist_p3']:.0f}%")
    if x["hist_p5"]>=15: why.append(f"유사패턴 +5% 관측률 {x['hist_p5']:.0f}%")
    if x["vol_ratio5m"]>=1.5 and x["vol_ratio15m"]>=1.2: why.append("5·15분 거래대금 동시 증가")
    if 45<=x["rsi1h"]<=66: why.append("1시간 RSI 과열 전 구간")
    if -3<=x["breakout_dist1h"]<=1.2: why.append("최근 고점 근처 압축")
    return why[:4] or ["복합 조건 상위"]

def main():
    btc=btc_regime()
    ms=markets()
    examples=[]
    candidates=[]
    errors=[]

    for n,m in enumerate(ms,1):
        mk=m["market"]
        try:
            h1=candles(mk,60,200)
            examples.extend(build_examples(h1,mk))
            d=days(mk,35)
            excluded,mx=recent_pump(d)
            if excluded: continue

            m5=candles(mk,5,120)
            m15=candles(mk,15,120)
            h4=candles(mk,240,120)
            f1=feat(h1)
            if not f1: continue
            micro=current_micro(m5,m15,h1,h4)
            candidates.append({"meta":m,"f":f1,"micro":micro,"h1":h1,"max30":mx})
        except Exception as e:
            errors.append(f"{mk}: {e}")
        if n%25==0:
            print(f"{n}/{len(ms)} markets; examples={len(examples)} candidates={len(candidates)}")

    if len(examples)<100 or not candidates:
        raise RuntimeError("Insufficient market data")

    stats=robust_stats([e["f"] for e in examples])
    exz=[zvec(e["f"],stats) for e in examples]

    results=[]
    for x in candidates:
        z=zvec(x["f"],stats)
        ds=sorted(((distance(z,ez),i) for i,ez in enumerate(exz)), key=lambda q:q[0])
        # Prefer neighbors from other markets to reduce same-coin autocorrelation.
        selected=[]
        same=[]
        for d,i in ds:
            e=examples[i]
            if e["market"]!=x["meta"]["market"] and len(selected)<K_NEIGHBORS:
                selected.append((d,e))
            elif len(same)<10:
                same.append((d,e))
            if len(selected)>=K_NEIGHBORS: break
        neigh=selected if len(selected)>=max(15,K_NEIGHBORS//2) else selected+same[:K_NEIGHBORS-len(selected)]
        gains=[e["gain"] for _,e in neigh]
        p3=100*sum(g>=3 for g in gains)/len(gains) if gains else 0
        p5=100*sum(g>=5 for g in gains)/len(gains) if gains else 0
        p10=100*sum(g>=10 for g in gains)/len(gains) if gains else 0
        avg_gain=mean(gains) if gains else 0
        avg_dist=mean([d for d,_ in neigh]) if neigh else 99
        similarity=100/(1+avg_dist)

        m=x["micro"]
        early = (
            m["vol_ratio5m"]>=1.4 and
            m["vol_ratio15m"]>=1.15 and
            m["ret1h"]<2.5
        )
        score=readiness_score(m,p3,p5,p10,similarity,btc["score"])

        liq=mean([float(r["candle_acc_trade_price"]) for r in x["h1"][-24:]])
        if liq<30_000_000: score=round(score*0.75,1)
        elif liq<80_000_000: score=round(score*0.9,1)

        row={
          "market":x["meta"]["market"],"name":x["meta"]["korean_name"],"english_name":x["meta"]["english_name"],
          **m,
          "score":score,
          "similarity":round(similarity,1),
          "hist_p3":round(p3,1),"hist_p5":round(p5,1),"hist_p10":round(p10,1),
          "hist_avg_gain6h":round(avg_gain,2),
          "max_30d_pump_pct":round(x["max30"],1),
          "early_volume_signal":early
        }
        row["why"]=explain(row)
        results.append(row)

    results.sort(key=lambda q:(q["score"],q["hist_p5"],q["hist_p3"],q["similarity"]), reverse=True)

    payload={
      "generated_at_utc":datetime.now(timezone.utc).isoformat(),
      "version":"V3",
      "config":{
        "exclude_pump_pct":EXCLUDE_PUMP_PCT,
        "lookahead_hours":LOOKAHEAD_HOURS,
        "neighbors":K_NEIGHBORS,
        "top_n":TOP_N,
        "historical_examples":len(examples),
        "markets_considered":len(ms),
        "current_candidates":len(candidates)
      },
      "btc_regime":btc,
      "results":results[:TOP_N],
      "errors":errors[:30],
      "disclaimer":"급등준비점수는 확률이 아닌 0~100 종합 셋업 점수입니다. +3/+5/+10 수치는 최근 1시간봉 데이터에서 유사했던 과거 패턴이 향후 6시간 내 해당 상승폭에 도달한 관측 비율이며 미래 수익을 보장하지 않습니다."
    }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(payload,ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
