import json
import math
import statistics
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests

BASE = "https://api.upbit.com"
OUT = Path("docs/data/market_scan_latest.json")
STABLE = {"USDT","USDC","DAI","TUSD","FDUSD","USDE","PYUSD","USDS","USD1","RLUSD","BUSD","USTC","EURC","USDG"}
S = requests.Session()
S.headers.update({"Accept":"application/json","User-Agent":"upbit-independent-market-scan/1.1"})


def api(path, params=None, retries=6):
    last = None
    for n in range(retries):
        try:
            r = S.get(BASE + path, params=params, timeout=20)
            if r.status_code == 429:
                time.sleep(1 + n)
                continue
            r.raise_for_status()
            time.sleep(0.12)
            return r.json()
        except Exception as e:
            last = e
            time.sleep(min(2 ** n, 8))
    raise RuntimeError(f"{path}: {last}")


def mean(xs):
    return statistics.fmean(xs) if xs else 0.0


def rsi(closes, n=14):
    if len(closes) < n + 1:
        return 50.0
    diffs = [closes[i] - closes[i-1] for i in range(1, len(closes))]
    gains = mean([max(x, 0) for x in diffs[-n:]])
    losses = mean([max(-x, 0) for x in diffs[-n:]])
    if losses == 0:
        return 100.0 if gains else 50.0
    rs = gains / losses
    return 100 - 100 / (1 + rs)


def markets():
    out = []
    for x in api("/v1/market/all", {"is_details":"true"}):
        mk = x["market"]
        if not mk.startswith("KRW-"):
            continue
        sym = mk.split("-", 1)[1].upper()
        if sym in STABLE:
            continue
        ev = x.get("market_event") or {}
        if ev.get("warning") is True:
            continue
        out.append((mk, sym, x.get("korean_name", sym)))
    return out


def ticker_map(markets_list):
    out = {}
    names = [m[0] for m in markets_list]
    for i in range(0, len(names), 100):
        rows = api("/v1/ticker", {"markets": ",".join(names[i:i+100])})
        for x in rows:
            out[x["market"]] = x
    return out


def floor_utc(dt, minutes):
    minute = (dt.minute // minutes) * minutes
    return dt.replace(minute=minute, second=0, microsecond=0)


def candles(market, unit, count=121):
    rows = api(f"/v1/candles/minutes/{unit}", {"market":market, "count":min(count, 200)})
    rows = list(reversed(rows))
    now = datetime.now(timezone.utc)
    if rows:
        last_dt = datetime.fromisoformat(rows[-1]["candle_date_time_utc"]).replace(tzinfo=timezone.utc)
        if last_dt >= floor_utc(now, unit):
            rows = rows[:-1]
    return rows


def tf_metrics(rows):
    if len(rows) < 61:
        return None
    c = [float(x["trade_price"]) for x in rows]
    h = [float(x["high_price"]) for x in rows]
    l = [float(x["low_price"]) for x in rows]
    v = [float(x["candle_acc_trade_price"]) for x in rows]

    price = c[-1]
    ma5 = mean(c[-5:]); ma10 = mean(c[-10:]); ma20 = mean(c[-20:]); ma60 = mean(c[-60:])
    last3 = mean(v[-3:])
    baseline = mean(v[-15:-3]) or mean(v[-30:-3]) or 1.0
    value_accel = last3 / baseline

    prev_high = max(h[-13:-1])
    support = min(l[-12:])
    comp6 = (max(h[-6:]) - min(l[-6:])) / price * 100 if price else 0.0
    resistance_distance = (prev_high / price - 1) * 100 if price else 0.0

    def ret(n):
        return (price / c[-1-n] - 1) * 100 if len(c) > n and c[-1-n] else 0.0

    ma_reclaim = price >= ma5 and price >= ma10
    ma_aligned = ma5 >= ma10 >= ma20

    return {
        "price": round(price, 8),
        "rsi": round(rsi(c), 2),
        "ma5": round(ma5, 8),
        "ma10": round(ma10, 8),
        "ma20": round(ma20, 8),
        "ma60": round(ma60, 8),
        "ma_reclaim": ma_reclaim,
        "ma_aligned": ma_aligned,
        "value_accel": round(value_accel, 3),
        "prev_high": round(prev_high, 8),
        "support": round(support, 8),
        "resistance_distance_pct": round(resistance_distance, 3),
        "compression_pct": round(comp6, 3),
        "ret_3bars_pct": round(ret(3), 3),
        "ret_12bars_pct": round(ret(12), 3),
    }


def qualifies(m5, m15, m1):
    if not all((m5, m15, m1)):
        return False
    # Fresh money first: short timeframes must be accelerating, and 1h cannot be dead.
    inflow = (m5["value_accel"] >= 1.5 and m15["value_accel"] >= 1.3 and m1["value_accel"] >= 0.75)
    inflow = inflow or (m15["value_accel"] >= 1.8 and m1["value_accel"] >= 1.0 and m5["value_accel"] >= 1.15)
    if not inflow:
        return False

    # Exclude names that already ran.
    if m5["ret_12bars_pct"] > 2.8 or m15["ret_12bars_pct"] > 4.0 or m1["ret_12bars_pct"] > 5.0:
        return False
    if m5["rsi"] > 76 or m15["rsi"] > 74 or m1["rsi"] > 74:
        return False

    # Compression / support / MA recovery.
    if m5["compression_pct"] > 3.0 or m15["compression_pct"] > 5.0:
        return False
    ma_ok = sum([
        m5["ma_reclaim"] or m5["ma_aligned"],
        m15["ma_reclaim"] or m15["ma_aligned"],
        m1["ma_reclaim"] or m1["ma_aligned"],
    ])
    if ma_ok < 2:
        return False

    # Prefer pre-breakout rather than names far above prior highs or too far below them.
    if not (-0.8 <= m5["resistance_distance_pct"] <= 3.0):
        return False
    if m1["price"] < m1["ma20"] * 0.97:
        return False
    return True


def priority(m5, m15, m1, trade_value_24h):
    # Ranking is mechanical and independent from all V7.6 / Live Ranker / similarity scores.
    inflow = math.log1p(max(m5["value_accel"], 0)) + math.log1p(max(m15["value_accel"], 0)) + 0.8 * math.log1p(max(m1["value_accel"], 0))
    not_run = max(0, 4.5 - max(0, m1["ret_12bars_pct"])) + max(0, 3.0 - max(0, m15["ret_12bars_pct"]))
    compression = max(0, 2.5 - m5["compression_pct"]) + 0.5 * max(0, 4.0 - m15["compression_pct"])
    structure = sum([m5["ma_aligned"], m15["ma_aligned"], m1["ma_aligned"], m5["ma_reclaim"], m15["ma_reclaim"], m1["ma_reclaim"]]) * 0.35
    liquidity = min(2.5, math.log10(max(trade_value_24h, 1)) - 7.5)
    return inflow * 3 + not_run + compression + structure + liquidity


def main():
    ms = markets()
    tmap = ticker_map(ms)
    qualified = []
    errors = []

    for i, (mk, sym, name) in enumerate(ms, 1):
        try:
            m5 = tf_metrics(candles(mk, 5))
            m15 = tf_metrics(candles(mk, 15))
            m1 = tf_metrics(candles(mk, 60))
            if not all((m5, m15, m1)):
                continue

            tick = tmap.get(mk, {})
            current_price = float(tick.get("trade_price", m5["price"]))
            tv24 = float(tick.get("acc_trade_price_24h", 0.0))

            if not qualifies(m5, m15, m1):
                continue

            # Entry zone uses nearby 5m/15m support and current market price.
            support_near = max(float(m5["support"]), float(m15["support"]))
            observe_low = max(support_near, current_price * 0.975)
            observe_high = current_price * 1.005
            trigger = max(float(m5["prev_high"]), float(m15["prev_high"]), current_price)
            invalidation = min(float(m15["support"]), float(m1["support"])) * 0.985

            reasons = [
                f'5m 거래대금 {m5["value_accel"]:.2f}x',
                f'15m 거래대금 {m15["value_accel"]:.2f}x',
                f'1h 거래대금 {m1["value_accel"]:.2f}x',
            ]
            if m5["compression_pct"] <= 1.5:
                reasons.append("5m 가격 압축")
            if m5["ma_aligned"]:
                reasons.append("5m MA 정배열")
            if m15["ma_aligned"]:
                reasons.append("15m MA 정배열")
            if m1["ma_aligned"]:
                reasons.append("1h MA 정배열")
            elif m1["ma_reclaim"]:
                reasons.append("1h MA5/10 회복")

            qualified.append({
                "market": mk,
                "symbol": sym,
                "name": name,
                "price": current_price,
                "trade_value_24h": round(tv24, 2),
                "entry_observe_low": round(observe_low, 8),
                "entry_observe_high": round(observe_high, 8),
                "trigger_price": round(trigger, 8),
                "invalidation_price": round(invalidation, 8),
                "reasons": reasons,
                "5m": m5,
                "15m": m15,
                "1h": m1,
                "_priority": priority(m5, m15, m1, tv24),
            })
        except Exception as e:
            errors.append(f"{mk}: {e}")
        if i % 25 == 0:
            print(f"{i}/{len(ms)} qualified={len(qualified)}")

    qualified.sort(key=lambda x: x["_priority"], reverse=True)
    for x in qualified:
        x.pop("_priority", None)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "version": "INDEPENDENT_MARKET_SCAN_V1_1",
        "note": "Independent from V7.6 score/ABC logic, Live Ranker, and historical similarity. Closed candles only. Focus: fresh trade-value inflow before large price expansion.",
        "universe_count": len(ms),
        "qualified_count": len(qualified),
        "top5": qualified[:5],
        "qualified": qualified,
        "error_count": len(errors),
        "errors_sample": errors[:10],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {OUT} universe={len(ms)} qualified={len(qualified)} errors={len(errors)}")


if __name__ == "__main__":
    main()
