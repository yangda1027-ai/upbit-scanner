import os, json, math, time, gzip, statistics
from pathlib import Path
from datetime import datetime, timezone, timedelta
import requests
import numpy as np
from scipy.spatial import cKDTree

BASE="https://api.upbit.com"
OUT=Path("docs/data/backtest_latest.json")

TRAIN_DAYS=int(os.getenv("TRAIN_DAYS","120"))
TEST_DAYS=int(os.getenv("TEST_DAYS","60"))
LOOKAHEAD=int(os.getenv("LOOKAHEAD_HOURS","6"))
K=int(os.getenv("K_NEIGHBORS","50"))
EXCLUDE_PUMP_PCT=float(os.getenv("EXCLUDE_PUMP_PCT","20"))
MAX_MARKETS=int(os.getenv("MAX_MARKETS","0"))  # 0 = all KRW
REQUEST_DELAY=float(os.getenv("REQUEST_DELAY","0.12"))
TOP_N=5
STABLE={"USDT","USDC","DAI","TUSD","FDUSD","USDE","PYUSD","USDS","USD1","RLUSD","BUSD","USTC"}

S=requests.Session()
S.headers.update({"Accept":"application/json","User-Agent":"upbit-v6-walkforward-backtest/1.0"})

def api(path,params=None,retries=8):
    last=None
    for n in range(retries):
        try:
            r=S.get(BASE+path,params=params,timeout=30)
            if r.status_code==429:
                time.sleep(1+n*.5); continue
            r.raise_for_status()
            time.sleep(REQUEST_DELAY)
            return r.json()
        except Exception as e:
            last=e
            time.sleep(min(2**n,10))
    raise RuntimeError(f"{path}: {last}")

def mean(x): return statistics.fmean(x) if x else 0.0
def sd(x): return statistics.pstdev(x) if len(x)>1 else 0.0
def div(a,b,d=0.0): return a/b if b not in (0,None) else d
def clamp(x,a,b): return max(a,min(b,x))
def ema(v,n):
    if not v:return 0.0
    a=2/(n+1); z=v[0]
    for x in v[1:]: z=a*x+(1-a)*z
    return z
def rsi(v,n=14):
    if len(v)<n+1:return 50.0
    d=[v[i]-v[i-1] for i in range(1,len(v))]
    g=mean([max(x,0) for x in d[-n:]])
    l=mean([max(-x,0) for x in d[-n:]])
    if l==0:return 100.0 if g else 50.0
    return 100-100/(1+g/l)
def parse_utc(s):
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)
def floor_hour(dt):
    return dt.replace(minute=0,second=0,microsecond=0)

def markets():
    out=[]
    for x in api("/v1/market/all",{"is_details":"true"}):
        mk=x["market"]
        if not mk.startswith("KRW-"): continue
        sym=mk.split("-",1)[1].upper()
        if sym in STABLE: continue
        ev=x.get("market_event") or {}
        if ev.get("warning") is True: continue
        out.append({"market":mk,"name":x.get("korean_name",sym)})
    return out[:MAX_MARKETS] if MAX_MARKETS>0 else out

def fetch_minutes(mk,unit,start,end):
    rows=[]; to=(end+timedelta(minutes=unit)).strftime("%Y-%m-%dT%H:%M:%S")
    while True:
        batch=api(f"/v1/candles/minutes/{unit}",{"market":mk,"count":200,"to":to})
        if not batch: break
        stop=False
        for r in batch:
            dt=parse_utc(r["candle_date_time_utc"])
            if start <= dt <= end:
                rows.append(r)
            if dt < start:
                stop=True
        oldest=parse_utc(batch[-1]["candle_date_time_utc"])
        if stop or oldest <= start or len(batch)<200:
            break
        to=(oldest-timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%S")
    uniq={r["candle_date_time_utc"]:r for r in rows}
    return [uniq[k] for k in sorted(uniq)]

F=["ret1","ret3","ret6","ret24","rsi","ema_gap","vol_ratio","volatility","range_pos","breakout_dist"]
W=np.array([1,1,1,.8,1.2,1,1.4,.8,1,1.3],dtype=float)

def features(rows,i):
    if i<30:return None
    w=rows[max(0,i-40):i+1]
    c=[float(x["trade_price"]) for x in w]
    h=[float(x["high_price"]) for x in w]
    l=[float(x["low_price"]) for x in w]
    v=[float(x["candle_acc_trade_price"]) for x in w]
    now=c[-1]
    def ret(n): return (div(now,c[-1-n],1)-1)*100 if len(c)>n else 0
    rr=[(div(c[j],c[j-1],1)-1)*100 for j in range(1,len(c))]
    hi=max(h[-20:]); lo=min(l[-20:]); ph=max(h[-21:-1])
    pv=v[-24:-3] if len(v)>=24 else v[:-3]
    return {"ret1":ret(1),"ret3":ret(3),"ret6":ret(6),"ret24":ret(24),"rsi":rsi(c),
            "ema_gap":(div(ema(c[-21:],9),ema(c[-21:],21),1)-1)*100,
            "vol_ratio":div(mean(v[-3:]),mean(pv) or mean(v),1),
            "volatility":sd(rr[-24:]),"range_pos":div(now-lo,hi-lo,.5)*100,
            "breakout_dist":(div(now,ph,1)-1)*100}

def aggregate(rows,minutes):
    buckets={}
    for r in rows:
        dt=parse_utc(r["candle_date_time_utc"])
        epoch=int(dt.timestamp())
        span=minutes*60
        b=datetime.fromtimestamp((epoch//span)*span,tz=timezone.utc)
        k=b.isoformat()
        p=float(r["trade_price"]); op=float(r["opening_price"]); hi=float(r["high_price"]); lo=float(r["low_price"]); tv=float(r["candle_acc_trade_price"])
        if k not in buckets:
            buckets[k]={"candle_date_time_utc":b.replace(tzinfo=None).isoformat(timespec="seconds"),
                        "opening_price":op,"high_price":hi,"low_price":lo,"trade_price":p,
                        "candle_acc_trade_price":tv,"_first":dt,"_last":dt}
        else:
            q=buckets[k]
            q["high_price"]=max(q["high_price"],hi); q["low_price"]=min(q["low_price"],lo)
            q["candle_acc_trade_price"]+=tv
            if dt<q["_first"]: q["opening_price"]=op; q["_first"]=dt
            if dt>q["_last"]: q["trade_price"]=p; q["_last"]=dt
    out=[]
    for k in sorted(buckets):
        q=buckets[k]; q.pop("_first",None); q.pop("_last",None); out.append(q)
    return out

def rolling_pump(hourly,i):
    if i<24:return 0.0,False
    start=max(0,i-30*24)
    recent_start=max(start,i-7*24)
    mx=-999; recent=False
    for j in range(start+24,i+1,24):
        seg=hourly[max(start,j-24):j+1]
        if not seg: continue
        op=float(seg[0]["opening_price"]); hi=max(float(x["high_price"]) for x in seg)
        prev=float(hourly[max(start,j-24)-1]["trade_price"]) if max(start,j-24)>0 else op
        g=(div(hi,min(op,prev),1)-1)*100
        mx=max(mx,g)
        if j>=recent_start and g>=EXCLUDE_PUMP_PCT: recent=True
    return max(mx,0),recent

def future_outcomes(rows5, idx5):
    entry=float(rows5[idx5]["trade_price"])
    out={}
    for h in (1,3,6,12,24):
        end=min(len(rows5),idx5+1+h*12)
        if end<=idx5+1: continue
        seg=rows5[idx5+1:end]
        hi=max(float(x["high_price"]) for x in seg)
        lo=min(float(x["low_price"]) for x in seg)
        last=float(seg[-1]["trade_price"])
        out[h]={"max_gain":(hi/entry-1)*100,"max_dd":(lo/entry-1)*100,"end_ret":(last/entry-1)*100}
    return out

def score_a(m,p3,p5,p10,sim,btc):
    s=0
    s+=clamp((m["vol5"]-.8)/1.8,0,1)*12+clamp((m["vol15"]-.8)/1.8,0,1)*10+clamp((m["vol1h"]-.7)/1.6,0,1)*7
    if m["vol5"]>=1.4 and m["vol15"]>=1.15 and m["ret1h"]<2.5:s+=10
    if 45<=m["rsi1h"]<=66:s+=8
    if -3<=m["breakout"]<=1.2:s+=7
    if -1<=m["ema_gap"]<=2.5:s+=5
    s+=clamp(p3*.18+p5*.22+p10*.20,0,28)+clamp(sim/100,0,1)*8
    if m["rsi1h"]>72:s-=8
    if m["ret6h"]>8:s-=8
    if btc<=1:s*=.78
    elif btc==2:s*=.9
    return clamp(s,0,100)

def score_b(m,p3,p5,p10,sim,btc,recent):
    s=0
    s+=clamp((m["vol5"]-1)/3,0,1)*14+clamp((m["vol15"]-1)/2.5,0,1)*12+clamp((m["vol1h"]-.8)/2.2,0,1)*8
    if m["ret1h"]>0:s+=8
    if 1<=m["ret3h"]<=12:s+=8
    elif m["ret3h"]>0:s+=3
    if 3<=m["ret6h"]<=20:s+=7
    elif m["ret6h"]>0:s+=3
    if -4<=m["dist_24h_high"]<=1.5:s+=8
    if m["ema20_over_50"]:s+=6
    if 50<=m["rsi1h"]<=74:s+=6
    s+=clamp(p3*.1+p5*.16+p10*.14,0,20)+clamp(sim/100,0,1)*5
    if recent:s+=6
    if m["rsi1h"]>82 or m["rsi15"]>88:s-=10
    if m["ret1h"]>15:s-=10
    if m["ret6h"]>35:s-=12
    if m["dist_24h_high"]<-10:s-=8
    if btc<=1:s*=.82
    elif btc==2:s*=.92
    return clamp(s,0,100)

def micro_at(rows5, rows15, rows1, rows4, t):
    def last_index(rows):
        lo,hi=0,len(rows)-1; ans=-1
        while lo<=hi:
            mid=(lo+hi)//2
            dt=parse_utc(rows[mid]["candle_date_time_utc"])
            if dt<=t: ans=mid; lo=mid+1
            else: hi=mid-1
        return ans
    i5,i15,i1,i4=last_index(rows5),last_index(rows15),last_index(rows1),last_index(rows4)
    if min(i5,i15,i1,i4)<30:return None,None
    f5,f15,f1,f4=features(rows5,i5),features(rows15,i15),features(rows1,i1),features(rows4,i4)
    cur=float(rows5[i5]["trade_price"])
    hi24=max(float(x["high_price"]) for x in rows1[max(0,i1-23):i1+1])
    c1=[float(x["trade_price"]) for x in rows1[max(0,i1-60):i1+1]]
    m={"price":cur,"vol5":f5["vol_ratio"],"vol15":f15["vol_ratio"],"vol1h":f1["vol_ratio"],"vol4h":f4["vol_ratio"],
       "rsi15":f15["rsi"],"rsi1h":f1["rsi"],"rsi4h":f4["rsi"],
       "ret1h":f1["ret1"],"ret3h":f1["ret3"],"ret6h":f1["ret6"],"ret24h":f1["ret24"],
       "breakout":f1["breakout_dist"],"ema_gap":f1["ema_gap"],"range_pos":f1["range_pos"],
       "dist_24h_high":(div(cur,hi24,1)-1)*100,
       "ema20_over_50":ema(c1[-30:],20)>=ema(c1[-60:],50)}
    return m,(i5,i1)

def btc_state(btc1,t):
    idx=-1
    for i,r in enumerate(btc1):
        if parse_utc(r["candle_date_time_utc"])<=t: idx=i
        else: break
    if idx<30:return 2
    f=features(btc1,idx); c=[float(x["trade_price"]) for x in btc1[max(0,idx-60):idx+1]]
    return sum([f["ret6"]>-1.5,f["ret24"]>-3,ema(c[-30:],20)>=ema(c[-60:],50),f["rsi"]>=42])

def robust_scaler(X):
    med=np.median(X,axis=0)
    mad=np.median(np.abs(X-med),axis=0)*1.4826
    std=np.std(X,axis=0)
    scale=np.where(mad>1e-9,mad,np.where(std>1e-9,std,1.0))
    return med,scale

def metrics_template():
    return {"n":0,"h1_3":0,"h3_5":0,"h6_5":0,"h6_10":0,"h24_5":0,"gain6":[],"dd6":[]}

def add_metric(m,out):
    m["n"]+=1
    if 1 in out and out[1]["max_gain"]>=3:m["h1_3"]+=1
    if 3 in out and out[3]["max_gain"]>=5:m["h3_5"]+=1
    if 6 in out:
        if out[6]["max_gain"]>=5:m["h6_5"]+=1
        if out[6]["max_gain"]>=10:m["h6_10"]+=1
        m["gain6"].append(out[6]["max_gain"]); m["dd6"].append(out[6]["max_dd"])
    if 24 in out and out[24]["max_gain"]>=5:m["h24_5"]+=1

def finalize(m):
    n=m["n"]
    if not n:return {"n":0}
    pct=lambda x:round(100*x/n,2)
    return {"n":n,"1h_3":pct(m["h1_3"]),"3h_5":pct(m["h3_5"]),"6h_5":pct(m["h6_5"]),
            "6h_10":pct(m["h6_10"]),"24h_5":pct(m["h24_5"]),
            "avg_max_gain_6h":round(mean(m["gain6"]),2),"avg_max_dd_6h":round(mean(m["dd6"]),2)}

def main():
    end=floor_hour(datetime.now(timezone.utc))-timedelta(hours=1)
    test_start=end-timedelta(days=TEST_DAYS)
    train_start=test_start-timedelta(days=TRAIN_DAYS)
    print("Window:",train_start,test_start,end)

    ms=markets()
    print("Markets:",len(ms))

    # BTC hourly for regime during test
    btc1=fetch_minutes("KRW-BTC",60,train_start,end)

    data={}
    train_vec=[]; train_gain=[]; train_market=[]
    for n,m in enumerate(ms,1):
        mk=m["market"]
        try:
            print(f"[{n}/{len(ms)}] {mk}: hourly")
            h1=fetch_minutes(mk,60,train_start,end)
            if len(h1)<200: continue

            # Build training examples ONLY before test_start: strict OOS separation
            for i in range(30,len(h1)-LOOKAHEAD,6):
                t=parse_utc(h1[i]["candle_date_time_utc"])
                if t>=test_start: break
                f=features(h1,i)
                if not f: continue
                base=float(h1[i]["trade_price"])
                hi=max(float(x["high_price"]) for x in h1[i+1:i+1+LOOKAHEAD])
                g=(hi/base-1)*100
                train_vec.append([f[k] for k in F]); train_gain.append(g); train_market.append(mk)

            print(f"[{n}/{len(ms)}] {mk}: 5m test data")
            r5=fetch_minutes(mk,5,test_start-timedelta(days=3),end+timedelta(days=1))
            if len(r5)<500: continue
            r15=aggregate(r5,15); r4=aggregate(r5,240)
            data[mk]={"name":m["name"],"h1":h1,"r5":r5,"r15":r15,"r4":r4}
        except Exception as e:
            print("ERROR",mk,e)

    X=np.asarray(train_vec,dtype=float)
    gains=np.asarray(train_gain,dtype=float)
    med,scale=robust_scaler(X)
    Xz=((X-med)/scale)*np.sqrt(W)
    tree=cKDTree(Xz)
    print("Training examples:",len(Xz),"test markets:",len(data))

    thresholds=[50,60,65,70,75,80]
    agg={mode:{str(t):metrics_template() for t in thresholds} for mode in ("A","B")}
    rankagg={mode:{str(r):metrics_template() for r in (1,3,5)} for mode in ("A","B")}
    market_rows=[]
    top_signals=[]

    t=test_start
    hours=0
    while t<=end-timedelta(hours=24):
        hours+=1
        btc=btc_state(btc1,t)
        A=[]; B=[]
        for mk,d in data.items():
            m,idxs=micro_at(d["r5"],d["r15"],d["h1"],d["r4"],t)
            if not m: continue
            i5,i1=idxs
            f=features(d["h1"],i1)
            if not f: continue

            q=((np.asarray([f[k] for k in F])-med)/scale)*np.sqrt(W)
            kk=min(400,len(Xz))
            dd,ii=tree.query(q,k=kk)
            if kk==1: dd=[dd]; ii=[ii]
            neigh=[]; used=set()
            for distv,j in zip(dd,ii):
                tm=train_market[int(j)]
                if tm==mk or tm in used: continue
                neigh.append((float(distv),int(j))); used.add(tm)
                if len(neigh)>=K: break
            if len(neigh)<K:
                for distv,j in zip(dd,ii):
                    if train_market[int(j)]==mk: continue
                    neigh.append((float(distv),int(j)))
                    if len(neigh)>=K: break
            if not neigh: continue
            ng=[gains[j] for _,j in neigh]
            p3=100*sum(x>=3 for x in ng)/len(ng)
            p5=100*sum(x>=5 for x in ng)/len(ng)
            p10=100*sum(x>=10 for x in ng)/len(ng)
            sim=100/(1+mean([x for x,_ in neigh]))

            max30,recent=rolling_pump(d["h1"],i1)
            liq=mean([float(x["candle_acc_trade_price"]) for x in d["h1"][max(0,i1-23):i1+1]])
            out=future_outcomes(d["r5"],i5)

            sa=score_a(m,p3,p5,p10,sim,btc)
            if liq<30_000_000:sa*=.75
            sb=score_b(m,p3,p5,p10,sim,btc,recent)
            if liq<30_000_000:sb*=.78
            elif liq<80_000_000: sb*=.9

            if max30<EXCLUDE_PUMP_PCT:
                A.append((sa,mk,d["name"],out))
                for th in thresholds:
                    if sa>=th:add_metric(agg["A"][str(th)],out)
            B.append((sb,mk,d["name"],out))
            for th in thresholds:
                if sb>=th:add_metric(agg["B"][str(th)],out)

        A.sort(reverse=True,key=lambda x:x[0]); B.sort(reverse=True,key=lambda x:x[0])
        for mode,arr in (("A",A),("B",B)):
            for cutoff in (1,3,5):
                for rec in arr[:cutoff]:
                    add_metric(rankagg[mode][str(cutoff)],rec[3])
            for rank,rec in enumerate(arr[:5],1):
                top_signals.append({"time":t.isoformat(),"mode":mode,"rank":rank,"market":rec[1],"name":rec[2],
                                    "score":round(rec[0],1),"future_6h_max":round(rec[3].get(6,{}).get("max_gain",0),2),
                                    "future_24h_max":round(rec[3].get(24,{}).get("max_gain",0),2)})
        if hours%24==0: print("Backtest days processed:",hours//24)
        t+=timedelta(hours=1)

    result={
      "version":"V6 Backtest 1.0",
      "generated_at_utc":datetime.now(timezone.utc).isoformat(),
      "method":{
        "train_days":TRAIN_DAYS,"test_days":TEST_DAYS,"walkforward_note":"120일 학습구간의 데이터만 KNN 참조에 사용하고 이후 60일은 완전 분리된 OOS 검증구간으로 평가",
        "markets":len(data),"training_examples":len(Xz),"test_hours":hours,
        "hit_definition":"각 신호 가격 이후 해당 시간 내 장중 고가가 목표 상승률에 도달하면 hit",
        "fees_slippage":"미반영"
      },
      "score_thresholds":{mode:{th:finalize(v) for th,v in vals.items()} for mode,vals in agg.items()},
      "top_rank":{mode:{r:finalize(v) for r,v in vals.items()} for mode,vals in rankagg.items()},
      "recent_top_signals":top_signals[-100:],
      "warning":"백테스트는 과거 시뮬레이션이며 미래 수익을 보장하지 않습니다. 고가 도달 기준이라 실제 체결·수수료·슬리피지와 다를 수 있습니다."
    }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print("Saved",OUT)

if __name__=="__main__":
    main()
