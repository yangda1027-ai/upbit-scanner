import json, time
from datetime import datetime, timezone
from pathlib import Path
import requests
API='https://api.upbit.com'; P=Path('docs/data/v7_3_signals.json')
S=requests.Session(); S.headers.update({'User-Agent':'upbit-v7-3-eval/1.0'})
def get(path,params):
    for i in range(4):
        r=S.get(API+path,params=params,timeout=15)
        if r.status_code==429: time.sleep(1+i); continue
        r.raise_for_status(); return r.json()
    return []
def parse(s): return datetime.fromisoformat(s.replace('Z','+00:00'))
def pct(a,b): return (a/b-1)*100 if b else 0
def outcome(m,entry,ts,hours):
    cnt=min(200,int(hours*12)+8); xs=get('/v1/candles/minutes/5',{'market':m,'count':cnt})
    start=parse(ts); end=start.timestamp()+hours*3600; ys=[]
    for x in xs:
        t=parse(x['candle_date_time_utc']+'+00:00').timestamp()
        if start.timestamp() < t <= end: ys.append(x)
    if not ys:return None
    highs=[float(x['high_price']) for x in ys]; lows=[float(x['low_price']) for x in ys]
    ys=sorted(ys,key=lambda x:x['candle_date_time_utc']); last=float(ys[-1]['trade_price'])
    return {'close_pct':round(pct(last,entry),3),'max_pct':round(pct(max(highs),entry),3),'min_pct':round(pct(min(lows),entry),3),
            'hit_3':max(highs)>=entry*1.03,'hit_5':max(highs)>=entry*1.05,'hit_10':max(highs)>=entry*1.10}
def main():
    if not P.exists(): return
    rows=json.loads(P.read_text(encoding='utf-8')); now=datetime.now(timezone.utc); changed=0
    for r in rows:
        age=(now-parse(r['ts'])).total_seconds()/3600
        for h,key in [(1,'h1'),(3,'h3'),(6,'h6'),(24,'h24')]:
            if r.get(key) is None and age>=h+0.1:
                try: r[key]=outcome(r['market'],float(r['price']),r['ts'],h); changed+=1
                except Exception as e: print('eval skip',r['market'],key,e)
                time.sleep(.07)
    if changed: P.write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    print('updated',changed)
if __name__=='__main__': main()
