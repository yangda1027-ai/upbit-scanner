import json, time
from datetime import datetime, timezone
from pathlib import Path
import requests

API='https://api.upbit.com'
OUT=Path('docs/data/v7_3_signals.json')
LATEST=Path('docs/data/v7_3_latest.json')
STABLE={'USDT','USDC','DAI'}
TOP_N=10
MAX_ROWS=15000
S=requests.Session(); S.headers.update({'User-Agent':'upbit-v7-3-early/1.0'})

def get(path, params=None, tries=4):
    last=None
    for i in range(tries):
        try:
            r=S.get(API+path,params=params,timeout=15)
            if r.status_code==429:
                time.sleep(1.0*(i+1)); continue
            r.raise_for_status(); return r.json()
        except Exception as e:
            last=e; time.sleep(.6*(i+1))
    raise last

def f(x,d=0.0):
    try:return float(x)
    except:return d

def pct(a,b): return (a/b-1)*100 if b else 0.0
def clamp(x,a,b): return max(a,min(b,x))

def markets():
    xs=get('/v1/market/all',{'is_details':'false'})
    return [x['market'] for x in xs if x['market'].startswith('KRW-') and x['market'].split('-',1)[1] not in STABLE]

def candles(m,unit,count):
    return list(reversed(get(f'/v1/candles/minutes/{unit}',{'market':m,'count':count})))

def features(m):
    c=candles(m,5,30)
    if len(c)<25:return None
    cl=[f(x['trade_price']) for x in c]; tv=[f(x['candle_acc_trade_price']) for x in c]
    r5=pct(cl[-1],cl[-2]); r15=pct(cl[-1],cl[-4]); r30=pct(cl[-1],cl[-7]); r60=pct(cl[-1],cl[-13])
    v5=tv[-1]/(sum(tv[-7:-1])/6 or 1)
    a15=(sum(tv[-3:])/3)/(sum(tv[-9:-3])/6 or 1)
    a30=(sum(tv[-6:])/6)/(sum(tv[-18:-6])/12 or 1)
    score=35.0
    score += clamp((v5-1)*12,-8,18) + clamp((a15-1)*18,-10,25) + clamp((a30-1)*10,-6,14)
    score += clamp(r5*3,-8,8) + clamp(r15*1.8,-8,10)
    over=0.0
    if r5>2.5: over+=(r5-2.5)*5
    if r15>4.0: over+=(r15-4.0)*4
    if r60>7.0: over+=(r60-7.0)*2
    score -= clamp(over,0,35)
    chase=(r5>=3.0) or (r15>=5.0) or (r60>=8.0)
    early=(not chase) and (a15>=1.25) and (v5>=1.15)
    label='CHASE' if chase else ('EARLY' if early else 'WATCH')
    return {'market':m,'price':cl[-1],'score':round(clamp(score,0,100),2),'label':label,
            'ret_5m':round(r5,3),'ret_15m':round(r15,3),'ret_30m':round(r30,3),'ret_60m':round(r60,3),
            'value_ratio_5m':round(v5,3),'value_accel_15m':round(a15,3),'value_accel_30m':round(a30,3)}

def btc_state():
    c=candles('KRW-BTC',5,24); cl=[f(x['trade_price']) for x in c]
    return {'ret_15m':round(pct(cl[-1],cl[-4]),3),'ret_60m':round(pct(cl[-1],cl[-13]),3)}

def main():
    now=datetime.now(timezone.utc).replace(second=0,microsecond=0)
    ms=markets(); tick=[]
    for i in range(0,len(ms),100):
        tick += get('/v1/ticker',{'markets':','.join(ms[i:i+100])}); time.sleep(.12)
    liquid=sorted(tick,key=lambda x:f(x.get('acc_trade_price_24h')),reverse=True)[:120]
    rows=[]
    for x in liquid:
        try:
            z=features(x['market'])
            if z: rows.append(z)
        except Exception as e: print('skip',x['market'],e)
        time.sleep(.07)
    priority={'EARLY':0,'WATCH':1,'CHASE':2}
    rows.sort(key=lambda x:(priority[x['label']],-x['score'],-x['value_accel_15m']))
    top=rows[:TOP_N]; btc=btc_state()
    recs=[]
    for rank,z in enumerate(top,1):
        q=dict(z); q.update({'ts':now.isoformat(),'rank':rank,'btc':btc,'h1':None,'h3':None,'h6':None,'h24':None}); recs.append(q)
    hist=[]
    if OUT.exists():
        try: hist=json.loads(OUT.read_text(encoding='utf-8'))
        except: hist=[]
    hist.extend(recs); hist=hist[-MAX_ROWS:]
    OUT.write_text(json.dumps(hist,ensure_ascii=False,indent=2),encoding='utf-8')
    LATEST.write_text(json.dumps({'updated_at':now.isoformat(),'btc':btc,'top':top},ensure_ascii=False,indent=2),encoding='utf-8')
    print('saved',len(recs),'signals; total',len(hist))

if __name__=='__main__': main()
