import json, time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from collections import defaultdict
import requests

API = "https://api.upbit.com"
SIGNALS = Path("docs/data/v7_6_signals.json")
MAX_AGE_HOURS = 30
HORIZONS = (1, 3, 6, 24)

S = requests.Session()
S.headers.update({"User-Agent": "upbit-v7-6-evaluator/1.0"})


def get(path, params=None, tries=5):
    last = None
    for i in range(tries):
        try:
            r = S.get(API + path, params=params, timeout=15)
            if r.status_code == 429:
                time.sleep(1.0 * (i + 1))
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last = e
            time.sleep(0.7 * (i + 1))
    raise last


def dt(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)


def pct(a, b):
    return (a / b - 1.0) * 100.0 if b else 0.0


def fetch_5m_history(market, earliest, now):
    """
    Fetch enough completed 5m candles to cover earliest -> now.
    Upbit returns newest first. Paginate backwards using `to`.
    """
    rows = []
    cursor = now + timedelta(minutes=5)

    while cursor >= earliest:
        batch = get(
            "/v1/candles/minutes/5",
            {
                "market": market,
                "count": 200,
                "to": cursor.strftime("%Y-%m-%dT%H:%M:%S"),
            },
        )
        if not batch:
            break

        rows.extend(batch)

        oldest = min(
            datetime.fromisoformat(x["candle_date_time_utc"]).replace(tzinfo=timezone.utc)
            for x in batch
        )
        if oldest <= earliest:
            break

        cursor = oldest
        time.sleep(0.12)

    # Deduplicate and sort oldest -> newest
    uniq = {}
    for x in rows:
        t = datetime.fromisoformat(x["candle_date_time_utc"]).replace(tzinfo=timezone.utc)
        uniq[t] = x
    return sorted(uniq.items(), key=lambda z: z[0])


def outcome(candles, start, end, entry):
    """
    CUMULATIVE outcome from signal time through horizon.
    max = highest 5m candle high in [start, end]
    close = last available 5m candle trade price in [start, end]
    """
    xs = [(t, x) for t, x in candles if start <= t <= end]
    if not xs:
        return None

    highs = [float(x["high_price"]) for _, x in xs]
    last_price = float(xs[-1][1]["trade_price"])
    max_ret = pct(max(highs), entry)
    close_ret = pct(last_price, entry)

    return {
        "close": round(close_ret, 3),
        "max": round(max_ret, 3),
        "hit3": bool(max_ret >= 3.0),
        "hit5": bool(max_ret >= 5.0),
        "hit10": bool(max_ret >= 10.0),
    }


def main():
    if not SIGNALS.exists():
        print("No signals file.")
        return

    data = json.loads(SIGNALS.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        print("No signals.")
        return

    now = datetime.now(timezone.utc)

    # Only records that still need at least one matured horizon.
    pending = []
    for r in data:
        try:
            start = dt(r["ts"])
        except Exception:
            continue

        age_h = (now - start).total_seconds() / 3600
        if age_h < 1 or age_h > MAX_AGE_HOURS:
            continue

        need = False
        for h in HORIZONS:
            if age_h >= h and r.get(f"h{h}") is None:
                need = True
        if need:
            pending.append(r)

    if not pending:
        print("Nothing to evaluate.")
        return

    by_market = defaultdict(list)
    for r in pending:
        by_market[r["market"]].append(r)

    print("pending records:", len(pending))
    print("markets:", len(by_market))

    cache = {}
    for i, (market, recs) in enumerate(by_market.items(), 1):
        earliest = min(dt(r["ts"]) for r in recs) - timedelta(minutes=5)
        try:
            cache[market] = fetch_5m_history(market, earliest, now)
            print(f"[{i}/{len(by_market)}] {market}: {len(cache[market])} candles")
        except Exception as e:
            print("history failed", market, e)
            cache[market] = []
        time.sleep(0.12)

    changed = 0

    for r in pending:
        candles = cache.get(r["market"], [])
        if not candles:
            continue

        start = dt(r["ts"])
        entry = float(r["price"])
        age_h = (now - start).total_seconds() / 3600

        for h in HORIZONS:
            key = f"h{h}"
            if age_h < h or r.get(key) is not None:
                continue

            end = start + timedelta(hours=h)
            o = outcome(candles, start, end, entry)
            if o is not None:
                r[key] = o
                changed += 1

    SIGNALS.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("updated outcomes:", changed)


if __name__ == "__main__":
    main()
