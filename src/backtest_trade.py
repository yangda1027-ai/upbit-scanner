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

def trade(r5,i,entry):
    tp=entry*(1+TP/100); sl=entry*(1-SL/100); end=min(len(r5),i+1+HOLD*12)
    for r in r5[i+1:end]:
        hi=float(r["high_price"]); lo=float(r["low_price"])
        if lo<=sl and hi>=tp:return "SL",-SL,True
        if lo<=sl:return "SL",-SL,False
        if hi>=tp:return "TP",TP,False
    last=float(r5[end-1]["trade_price"]); return "TIME",(last/entry-1)*100,False

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
    X=np.asarray(X,float); G=np.asarray(G,float); med=np.median(X,0); mad=np.median(np.abs(X-med),0)*1.4826; std=np.std(X,0); sca=np.where(mad>1e-9,mad,np.where(std>1e-9,std,1))
    tree=cKDTree(((X-med)/sca)*W); trades={str(x):[] for x in THS}; sigs={mk:[] for mk in data}

    t=test
    while t<=end-timedelta(hours=24):
        for mk,d in data.items():
            i1=li(d["h1"],t); i5=li(d["r5"],t); i15=li(d["r15"],t); i4=li(d["r4"],t)
            if min(i1,i5,i15,i4)<30:continue
            f1=feat(d["h1"],i1); f5=feat(d["r5"],i5); f15=feat(d["r15"],i15)
            c1=[float(x["trade_price"]) for x in d["h1"][max(0,i1-60):i1+1]]; cur=float(d["r5"][i5]["trade_price"]); hi24=max(float(x["high_price"]) for x in d["h1"][max(0,i1-23):i1+1])
            m={"vol5":f5["vol_ratio"],"vol15":f15["vol_ratio"],"vol1h":f1["vol_ratio"],"rsi15":f15["rsi"],"rsi1h":f1["rsi"],"ret1h":f1["ret1"],"ret3h":f1["ret3"],"ret6h":f1["ret6"],"dist24":(cur/hi24-1)*100,"ema2050":ema(c1[-30:],20)>=ema(c1[-60:],50)}
            q=((np.asarray([f1[k] for k in F])-med)/sca)*W; dd,ii=tree.query(q,k=min(400,len(X)))
            if np.isscalar(ii):ii=[ii];dd=[dd]
            neigh=[];used=set()
            for dis,j in zip(dd,ii):
                tm=MK[int(j)]
                if tm==mk or tm in used:continue
                neigh.append((float(dis),int(j)));used.add(tm)
                if len(neigh)>=K:break
            if not neigh:continue
            ng=[G[j] for _,j in neigh]; p3=100*sum(x>=3 for x in ng)/len(ng);p5=100*sum(x>=5 for x in ng)/len(ng);p10=100*sum(x>=10 for x in ng)/len(ng);sim=100/(1+mean([x for x,_ in neigh]))
            # Approximate recent-pump flag from previous 7d hourly highs
            base=min(float(x["trade_price"]) for x in d["h1"][max(0,i1-7*24):i1+1]); high=max(float(x["high_price"]) for x in d["h1"][max(0,i1-7*24):i1+1]); recent=(high/base-1)*100>=20
            btc=2; scorev=score(m,p3,p5,p10,sim,btc,recent)
            result,ret,both=trade(d["r5"],i5,cur); sigs[mk].append((t,scorev))
            for th in THS:
                if scorev>=th: trades[str(th)].append({"time":t.isoformat(),"market":mk,"score":scorev,"result":result,"ret":ret,"both":both})
        t+=timedelta(hours=1)

    def stats(rows):
        if not rows:return {"n":0}
        n=len(rows);tp=sum(r["result"]=="TP" for r in rows);sl=sum(r["result"]=="SL" for r in rows);tm=n-tp-sl;rets=[r["ret"] for r in rows]
        gw=sum(max(x,0) for x in rets);gl=-sum(min(x,0) for x in rets)
        return {"n":n,"tp_rate":round(tp/n*100,2),"sl_rate":round(sl/n*100,2),"time_rate":round(tm/n*100,2),"avg_return":round(mean(rets),3),"profit_factor":round(div(gw,gl,0),2),"same_candle_conflicts":sum(r["both"] for r in rows)}
    recall={}
    for th in (10,20,30):
        ev=hit1=hit3=hit6=0
        for mk,d in data.items():
            r5=d["r5"]; last_event=None
            for i,r in enumerate(r5):
                tt=dt(r["candle_date_time_utc"])
                if tt<test or tt>end-timedelta(hours=6) or tt.minute!=0:continue
                j=min(len(r5),i+73); entry=float(r["trade_price"]); hi=max(float(x["high_price"]) for x in r5[i+1:j])
                if (hi/entry-1)*100<th:continue
                if last_event and tt-last_event<timedelta(hours=6):continue
                last_event=tt;ev+=1
                for h,key in [(1,"h1"),(3,"h3"),(6,"h6")]:
                    ok=any(tt-timedelta(hours=h)<=st<tt and sc>=75 for st,sc in sigs[mk])
                    if ok:
                        if h==1:hit1+=1
                        elif h==3:hit3+=1
                        else:hit6+=1
        recall[str(th)]={"events":ev,"detected_1h_pct":round(div(hit1,ev,0)*100,2),"detected_3h_pct":round(div(hit3,ev,0)*100,2),"detected_6h_pct":round(div(hit6,ev,0)*100,2)}
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps({"version":"V6.2 Trade Backtest","generated_at_utc":datetime.now(timezone.utc).isoformat(),"config":{"train_days":TRAIN_DAYS,"test_days":TEST_DAYS,"tp":TP,"sl":SL,"hold_hours":HOLD,"markets":len(data),"surge_signal_threshold":75},"threshold_trade_stats":{k:stats(v) for k,v in trades.items()},"surge_recall":recall,"notes":["동일 5분봉에서 TP/SL 동시 도달 시 SL 우선","수수료·슬리피지 미반영"]},ensure_ascii=False,indent=2),encoding="utf-8")
if __name__=="__main__":main()
