import os,json,time,statistics,math
from pathlib import Path
from datetime import datetime,timezone,timedelta
import requests,numpy as np
from scipy.spatial import cKDTree

BASE="https://api.upbit.com"; OUT=Path("docs/data/backtest_trade_latest.json")
TRAIN_DAYS=int(os.getenv("TRAIN_DAYS","120")); TEST_DAYS=int(os.getenv("TEST_DAYS","60"))
K=int(os.getenv("K_NEIGHBORS","50")); TP=float(os.getenv("TAKE_PROFIT_PCT","5")); SL=float(os.getenv("STOP_LOSS_PCT","3"))
HOLD=int(os.getenv("MAX_HOLD_HOURS","6")); DELAY=float(os.getenv("REQUEST_DELAY","0.12"))
THS=[60,65,70,75,80]; STABLE={"USDT","USDC","DAI","TUSD","FDUSD","USDE","PYUSD","USDS","USD1","RLUSD","BUSD","USTC"}
F=["ret1","ret3","ret6","ret24","rsi","ema_gap","vol_ratio","volatility","range_pos","breakout_dist"]
W=np.sqrt(np.array([1,1,1,.8,1.2,1,1.4,.8,1,1.3],float))
S=requests.Session(); S.headers.update({"Accept":"application/json","User-Agent":"upbit-v6-2-trade-bt"})

def api(path,p=None):
    for n in range(8):
        r=S.get(BASE+path,params=p,timeout=30)
        if r.status_code==429: time.sleep(1+n*.5); continue
        r.raise_for_status(); time.sleep(DELAY); return r.json()
    return []
def mean(x): return statistics.fmean(x) if x else 0
def sd(x): return statistics.pstdev(x) if len(x)>1 else 0
def div(a,b,d=0): return a/b if b not in (0,None) else d
def clamp(x,a,b): return max(a,min(b,x))
def ema(v,n):
    if not v:return 0
    a=2/(n+1); z=v[0]
    for x in v[1:]: z=a*x+(1-a)*z
    return z
def rsi(v,n=14):
    if len(v)<n+1:return 50
    d=[v[i]-v[i-1] for i in range(1,len(v))]
    g=mean([max(x,0) for x in d[-n:]]); l=mean([max(-x,0) for x in d[-n:]])
    return 100 if l==0 and g else (50 if l==0 else 100-100/(1+g/l))
def dt(s): return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)

def markets():
    out=[]
    for x in api("/v1/market/all",{"is_details":"true"}):
        if not x["market"].startswith("KRW-"): continue
        sym=x["market"].split("-")[1].upper()
        if sym in STABLE: continue
        if (x.get("market_event") or {}).get("warning") is True: continue
        out.append({"market":x["market"],"name":x.get("korean_name",sym)})
    return out

def fetch(mk,u,start,end):
    rows=[]; to=(end+timedelta(minutes=u)).strftime("%Y-%m-%dT%H:%M:%S")
    while True:
        b=api(f"/v1/candles/minutes/{u}",{"market":mk,"count":200,"to":to})
        if not b: break
        stop=False
        for r in b:
            t=dt(r["candle_date_time_utc"])
            if start<=t<=end: rows.append(r)
            if t<start: stop=True
        oldest=dt(b[-1]["candle_date_time_utc"])
        if stop or oldest<=start or len(b)<200: break
        to=(oldest-timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%S")
    q={r["candle_date_time_utc"]:r for r in rows}
    return [q[k] for k in sorted(q)]

def feat(rows,i):
    if i<30:return None
    w=rows[max(0,i-40):i+1]; c=[float(x["trade_price"]) for x in w]; h=[float(x["high_price"]) for x in w]
    l=[float(x["low_price"]) for x in w]; v=[float(x["candle_acc_trade_price"]) for x in w]; now=c[-1]
    ret=lambda n:(div(now,c[-1-n],1)-1)*100
    rr=[(div(c[j],c[j-1],1)-1)*100 for j in range(1,len(c))]
    ph=max(h[-21:-1]); hi=max(h[-20:]); lo=min(l[-20:]); pv=v[-24:-3]
    return {"ret1":ret(1),"ret3":ret(3),"ret6":ret(6),"ret24":ret(24),"rsi":rsi(c),
            "ema_gap":(div(ema(c[-21:],9),ema(c[-21:],21),1)-1)*100,
            "vol_ratio":div(mean(v[-3:]),mean(pv) or mean(v),1),"volatility":sd(rr[-24:]),
            "range_pos":div(now-lo,hi-lo,.5)*100,"breakout_dist":(div(now,ph,1)-1)*100}

def agg(rows,mins):
    b={}
    for r in rows:
        t=dt(r["candle_date_time_utc"]); span=mins*60; bt=datetime.fromtimestamp((int(t.timestamp())//span)*span,tz=timezone.utc)
        k=bt.isoformat(); p=float(r["trade_price"]); o=float(r["opening_price"]); h=float(r["high_price"]); l=float(r["low_price"]); tv=float(r["candle_acc_trade_price"])
        if k not in b:b[k]={"candle_date_time_utc":bt.replace(tzinfo=None).isoformat(timespec="seconds"),"opening_price":o,"high_price":h,"low_price":l,"trade_price":p,"candle_acc_trade_price":tv,"first":t,"last":t}
        else:
            q=b[k]; q["high_price"]=max(q["high_price"],h); q["low_price"]=min(q["low_price"],l); q["candle_acc_trade_price"]+=tv
            if t<q["first"]:q["opening_price"]=o;q["first"]=t
            if t>q["last"]:q["trade_price"]=p;q["last"]=t
    out=[]
    for k in sorted(b):
        q=b[k]; q.pop("first"); q.pop("last"); out.append(q)
    return out

def li(rows,t):
    lo,hi=0,len(rows)-1; ans=-1
    while lo<=hi:
        m=(lo+hi)//2
        if dt(rows[m]["candle_date_time_utc"])<=t:ans=m;lo=m+1
        else:hi=m-1
    return ans

def score(m,p3,p5,p10,sim,btc,recent):
    s=clamp((m["vol5"]-1)/3,0,1)*14+clamp((m["vol15"]-1)/2.5,0,1)*12+clamp((m["vol1h"]-.8)/2.2,0,1)*8
    s+=8 if m["ret1h"]>0 else 0; s+=8 if 1<=m["ret3h"]<=12 else (3 if m["ret3h"]>0 else 0)
    s+=7 if 3<=m["ret6h"]<=20 else (3 if m["ret6h"]>0 else 0); s+=8 if -4<=m["dist24"]<=1.5 else 0
    s+=6 if m["ema2050"] else 0; s+=6 if 50<=m["rsi1h"]<=74 else 0; s+=clamp(p3*.1+p5*.16+p10*.14,0,20)+clamp(sim/100,0,1)*5
    if recent:s+=6
    if m["rsi1h"]>82 or m["rsi15"]>88:s-=10
    if m["ret1h"]>15:s-=10
    if m["ret6h"]>35:s-=12
    if m["dist24"]<-10:s-=8
    if btc<=1:s*=.82
    elif btc==2:s*=.92
    return clamp(s,0,100)


def fixed_trade(r5,i,entry,hold_hours=6):
    tp=entry*1.05; sl=entry*.97; end=min(len(r5),i+1+hold_hours*12)
    for j in range(i+1,end):
        r=r5[j]; hi=float(r["high_price"]); lo=float(r["low_price"])
        # ë³´ìì : ê°ì 5ë¶ë´ìì ìµì /ìì  ëì í°ì¹ë©´ ìì  ì°ì 
        if lo<=sl and hi>=tp:return {"ret":-3.0,"exit_i":j,"result":"SL"}
        if lo<=sl:return {"ret":-3.0,"exit_i":j,"result":"SL"}
        if hi>=tp:return {"ret":5.0,"exit_i":j,"result":"TP"}
    j=end-1
    last=float(r5[j]["trade_price"])
    return {"ret":(last/entry-1)*100,"exit_i":j,"result":"TIME"}

def split_deep_trade(r5,i,entry,hold_hours=6):
    """
    ì¬ì©ìê° ìíë ê¹ì 3ë¨ ë¶í  ìì:
      30% ì¦ì / 30% -2% / 40% -4%
      ì ë¶ í©ì¹ íê· ë¨ê° ê¸°ì¤ +5% ìµì 
      ì´ê¸° ì í¸ê° ê¸°ì¤ -6% íëì¤í
      ìµë 6ìê°
    ì£¼ì: baselineì -3% ìì ë³´ë¤ ë ëì ìíì íì©íë¯ë¡
    íê· ììµë¿ ìëë¼ íê· ìì¤/ìµììì¤ë ê°ì´ ë´ì¼ íë¤.
    """
    levels=[entry,entry*.98,entry*.96]; weights=[.30,.30,.40]
    fills=[(entry,.30)]
    hard_sl=entry*.94
    end=min(len(r5),i+1+hold_hours*12)
    for j in range(i+1,end):
        r=r5[j]; hi=float(r["high_price"]); lo=float(r["low_price"])
        if lo<=hard_sl:
            invested=sum(w for _,w in fills)
            avg=sum(p*w for p,w in fills)/invested
            # ë¯¸ì²´ê²° íê¸ì 0% ììµì¼ë¡ ëê³ , ì²´ê²°ë ë¹ì¤ë§ ììµ ë°ì
            ret=((hard_sl/avg)-1)*100*invested
            return {"ret":ret,"exit_i":j,"result":"SL","filled":invested}
        # ê°ì ë´ìì ìë ë¶í ê°ê° ë¿ì¼ë©´ ëª¨ë ì²´ê²°ë ê²ì¼ë¡ ê°ì 
        if len(fills)<2 and lo<=levels[1]: fills.append((levels[1],weights[1]))
        if len(fills)<3 and lo<=levels[2]: fills.append((levels[2],weights[2]))
        invested=sum(w for _,w in fills)
        avg=sum(p*w for p,w in fills)/invested
        tp=avg*1.05
        if hi>=tp:
            return {"ret":5.0*invested,"exit_i":j,"result":"TP","filled":invested}
    j=end-1
    last=float(r5[j]["trade_price"]); invested=sum(w for _,w in fills)
    avg=sum(p*w for p,w in fills)/invested
    return {"ret":((last/avg)-1)*100*invested,"exit_i":j,"result":"TIME","filled":invested}

def runner_trade(r5,i,entry,scorev,runner_fraction=.5,trail_pct=3.0,max_hours=24,only_80=False):
    """
    +5% ëë¬ ì : -3% ìì 
    +5% ëë¬ ì:
      - only_80=False: í­ì ì¼ë¶ ìµì  í runner
      - only_80=True : ì§ìì ì 80ì  ì´ìììë§ runner, ê·¸ ì¸ +5% ì ëìµì 
    runnerë ìµê³ ê° ëë¹ -3% í¸ë ì¼ë§, ìµì ì¤íì ì§ìê°(ë³¸ì ),
    ìµë 24ìê° ë³´ì .
    """
    tp=entry*1.05; sl=entry*.97; end=min(len(r5),i+1+max_hours*12)
    use_runner=(not only_80) or scorev>=80
    for j in range(i+1,end):
        r=r5[j]; hi=float(r["high_price"]); lo=float(r["low_price"])
        if lo<=sl and hi>=tp:return {"ret":-3.0,"exit_i":j,"result":"SL"}
        if lo<=sl:return {"ret":-3.0,"exit_i":j,"result":"SL"}
        if hi>=tp:
            if not use_runner:
                return {"ret":5.0,"exit_i":j,"result":"TP"}
            realized=(1-runner_fraction)*5.0
            peak=hi
            # ê°ì ë´ ìì ììë¥¼ ëª¨ë¥´ë¯ë¡ runner ì²­ì° íë¨ì ë¤ì ë´ë¶í° ìì
            for k in range(j+1,end):
                q=r5[k]; qhi=float(q["high_price"]); qlo=float(q["low_price"])
                peak=max(peak,qhi)
                trail=max(entry,peak*(1-trail_pct/100))
                if qlo<=trail:
                    rr=(trail/entry-1)*100
                    return {"ret":realized+runner_fraction*rr,"exit_i":k,"result":"RUNNER"}
            k=end-1
            last=float(r5[k]["trade_price"])
            rr=(last/entry-1)*100
            return {"ret":realized+runner_fraction*rr,"exit_i":k,"result":"TIME_RUNNER"}
    j=end-1
    last=float(r5[j]["trade_price"])
    return {"ret":(last/entry-1)*100,"exit_i":j,"result":"TIME"}

def summarize(rows):
    if not rows:return {"n":0}
    rets=[x["ret"] for x in rows]; n=len(rows)
    wins=[x for x in rets if x>0]; losses=[x for x in rets if x<0]
    gw=sum(wins); gl=-sum(losses)
    # 5% ì´ì ì¤íë ê±°ë ë¹ì¨ì runner ì ëµìì ë¨ì TPì¨ê³¼ ë¤ë¥´ë¯ë¡ positive rateë í¨ê» ì ê³µ
    return {
        "n":n,
        "win_rate":round(100*len(wins)/n,2),
        "loss_rate":round(100*len(losses)/n,2),
        "avg_return":round(mean(rets),3),
        "median_return":round(float(np.median(rets)),3),
        "profit_factor":round(div(gw,gl,0),2),
        "avg_win":round(mean(wins),3) if wins else 0,
        "avg_loss":round(mean(losses),3) if losses else 0,
        "best_trade":round(max(rets),3),
        "worst_trade":round(min(rets),3),
    }

def main():
    end=datetime.now(timezone.utc).replace(minute=0,second=0,microsecond=0)-timedelta(hours=1)
    test=end-timedelta(days=TEST_DAYS); train=test-timedelta(days=TRAIN_DAYS)
    data={}; X=[]; G=[]; MK=[]
    for n,m in enumerate(markets(),1):
        try:
            h1=fetch(m["market"],60,train,end)
            if len(h1)<200:continue
            for i in range(30,len(h1)-6,6):
                t=dt(h1[i]["candle_date_time_utc"])
                if t>=test:break
                f=feat(h1,i)
                if not f:continue
                b=float(h1[i]["trade_price"]); hi=max(float(x["high_price"]) for x in h1[i+1:i+7])
                X.append([f[k] for k in F]); G.append((hi/b-1)*100); MK.append(m["market"])
            r5=fetch(m["market"],5,test-timedelta(days=3),end+timedelta(days=1))
            if len(r5)<500:continue
            data[m["market"]]={"name":m["name"],"h1":h1,"r5":r5,"r15":agg(r5,15),"r4":agg(r5,240)}
            print(n,m["market"],len(r5))
        except Exception as e: print("ERR",m["market"],e)

    X=np.asarray(X,float); G=np.asarray(G,float)
    med=np.median(X,0); mad=np.median(np.abs(X-med),0)*1.4826; std=np.std(X,0)
    sca=np.where(mad>1e-9,mad,np.where(std>1e-9,std,1))
    tree=cKDTree(((X-med)/sca)*W)

    # ì í¸ë¥¼ ë¨¼ì  ì ì¥. V6.2ì ëì¼íê² ìê°ë¹ íê°.
    signals=[]
    t=test
    while t<=end-timedelta(hours=24):
        for mk,d in data.items():
            i1=li(d["h1"],t); i5=li(d["r5"],t); i15=li(d["r15"],t); i4=li(d["r4"],t)
            if min(i1,i5,i15,i4)<30:continue
            f1=feat(d["h1"],i1); f5=feat(d["r5"],i5); f15=feat(d["r15"],i15)
            c1=[float(x["trade_price"]) for x in d["h1"][max(0,i1-60):i1+1]]
            cur=float(d["r5"][i5]["trade_price"])
            hi24=max(float(x["high_price"]) for x in d["h1"][max(0,i1-23):i1+1])
            m={"vol5":f5["vol_ratio"],"vol15":f15["vol_ratio"],"vol1h":f1["vol_ratio"],
               "rsi15":f15["rsi"],"rsi1h":f1["rsi"],"ret1h":f1["ret1"],
               "ret3h":f1["ret3"],"ret6h":f1["ret6"],"dist24":(cur/hi24-1)*100,
               "ema2050":ema(c1[-30:],20)>=ema(c1[-60:],50)}
            q=((np.asarray([f1[k] for k in F])-med)/sca)*W
            dd,ii=tree.query(q,k=min(400,len(X)))
            if np.isscalar(ii): ii=[ii]; dd=[dd]
            neigh=[]; used=set()
            for dis,jj in zip(dd,ii):
                tm=MK[int(jj)]
                if tm==mk or tm in used:continue
                neigh.append((float(dis),int(jj))); used.add(tm)
                if len(neigh)>=K:break
            if not neigh:continue
            ng=[G[jj] for _,jj in neigh]
            p3=100*sum(x>=3 for x in ng)/len(ng)
            p5=100*sum(x>=5 for x in ng)/len(ng)
            p10=100*sum(x>=10 for x in ng)/len(ng)
            sim=100/(1+mean([x for x,_ in neigh]))
            base=min(float(x["trade_price"]) for x in d["h1"][max(0,i1-7*24):i1+1])
            high=max(float(x["high_price"]) for x in d["h1"][max(0,i1-7*24):i1+1])
            recent=(high/base-1)*100>=20
            btc=2  # V6.2ì ê°ì ê·¼ì¬
            sv=score(m,p3,p5,p10,sim,btc,recent)
            if sv>=70:
                signals.append({"time":t,"market":mk,"score":float(sv),"i5":i5,"entry":cur})
        t+=timedelta(hours=1)

    signals.sort(key=lambda x:x["time"])
    thresholds=[70,75,80]
    strategies={
        "fixed_5_3_6h":"ì¦ì 100% Â· +5% ìµì  / -3% ìì  / 6h",
        "split_30_30_40":"3ë¨ ë¶í  30/30/40 Â· 0/-2/-4% Â· íê· ë¨ê° +5% / ì´ê¸° -6% íëì¤í / 6h",
        "runner50_all":"ì¦ì 100% Â· +5%ìì 50% ìµì  Â· ëë¨¸ì§ 3% í¸ë ì¼ë§ / ìµë 24h",
        "runner70_80only":"ì¦ì 100% Â· 80ì +ë§ +5%ìì 30% ìµì  Â· 70% 3% í¸ë ì¼ë§ / ìµë 24h"
    }
    results={str(th):{} for th in thresholds}

    for th in thresholds:
        eligible=[s for s in signals if s["score"]>=th]
        for skey in strategies:
            rows=[]; busy_until={}
            for s in eligible:
                mk=s["market"]
                # ê°ì ì½ì¸ì í¬ì§ìì´ ì´ë ¤ ìë ëì ì¤ë³µ ì§ì ê¸ì§
                if mk in busy_until and s["time"]<=busy_until[mk]:
                    continue
                d=data[mk]; r5=d["r5"]
                if skey=="fixed_5_3_6h":
                    tr=fixed_trade(r5,s["i5"],s["entry"],6)
                elif skey=="split_30_30_40":
                    tr=split_deep_trade(r5,s["i5"],s["entry"],6)
                elif skey=="runner50_all":
                    tr=runner_trade(r5,s["i5"],s["entry"],s["score"],.50,3.0,24,False)
                else:
                    tr=runner_trade(r5,s["i5"],s["entry"],s["score"],.70,3.0,24,True)
                exit_t=dt(r5[tr["exit_i"]]["candle_date_time_utc"])
                busy_until[mk]=exit_t
                rows.append({**tr,"market":mk,"time":s["time"].isoformat(),"score":round(s["score"],2)})
            results[str(th)][skey]=summarize(rows)

    out={
        "version":"V6.3 Strategy Backtest",
        "generated_at_utc":datetime.now(timezone.utc).isoformat(),
        "config":{"train_days":TRAIN_DAYS,"test_days":TEST_DAYS,"markets":len(data),"k_neighbors":K},
        "strategy_labels":strategies,
        "results":results,
        "notes":[
            "V6.2ì ê°ì ì í¸ ìì§ì ì¬ì©í ì ëµ ë¹êµì© ê·¼ì¬ ë°±íì¤í¸",
            "ëì¼ ì½ì¸ í¬ì§ì ë³´ì  ì¤ìë ì ì í¸ ì¤ë³µì§ì ê¸ì§",
            "ëì¼ 5ë¶ë´ìì +5%ì -3%ê° ëìì ë¿ì¼ë©´ ìì  ì°ì ",
            "ë¶í ë§¤ì ì ëµì -4% ì¶ê°ë§¤ìë¥¼ íì©íê¸° ìí´ ì´ê¸° ì í¸ê° ê¸°ì¤ -6% íëì¤í ì¬ì©",
            "runnerë +5% ëë¬ ë¤ ë¤ì 5ë¶ë´ë¶í° ìµê³ ê° ëë¹ -3% í¸ë ì¼ë§, ìµì ì¤íì ì§ìê°",
            "ììë£Â·ì¬ë¦¬í¼ì§Â·í¸ê°ì²´ê²° ìí¥ ë¯¸ë°ì",
            "ê³¼ê±° BTC regimeì V6.2ì ëì¼íê² ìì ê·¼ì¬(btc=2)ë¥¼ ì¬ì©"
        ]
    }
    OUT2=Path("docs/data/backtest_strategy_latest.json")
    OUT2.parent.mkdir(parents=True,exist_ok=True)
    OUT2.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")

if __name__=="__main__":
    main()
