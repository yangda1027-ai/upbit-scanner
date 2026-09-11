import json, time
from datetime import datetime, timezone
from pathlib import Path
import requests

API = "https://api.upbit.com"
OUT = Path("docs/data/v7_6_signals.json")
LATEST = Path("docs/data/v7_6_latest.json")

STABLE = {"USDT", "USDC", "DAI"}
TOP_N = 10
MAX_ROWS = 20000

S = requests.Session()
S.headers.update({"User-Agent": "upbit-v7-6-all-krw/1.0"})


def get(path, params=None, tries=4):
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
            time.sleep(0.6 * (i + 1))
    raise last


def f(x, d=0.0):
    try:
        return float(x)
    except Exception:
        return d


def pct(a, b):
    return (a / b - 1) * 100 if b else 0.0


def clamp(x, a, b):
    return max(a, min(b, x))


def markets():
    xs = get("/v1/market/all", {"is_details": "false"})
    return [
        x["market"]
        for x in xs
        if x["market"].startswith("KRW-")
        and x["market"].split("-", 1)[1] not in STABLE
    ]


def candles(m, unit, count):
    return list(
        reversed(
            get(
                f"/v1/candles/minutes/{unit}",
                {"market": m, "count": count},
            )
        )
    )


def features(m):
    # V7.3ì ëì¼í 5ë¶ë´ feature / score / EARLY ê·ì¹
    c = candles(m, 5, 30)
    if len(c) < 25:
        return None

    cl = [f(x["trade_price"]) for x in c]
    tv = [f(x["candle_acc_trade_price"]) for x in c]

    r5 = pct(cl[-1], cl[-2])
    r15 = pct(cl[-1], cl[-4])
    r30 = pct(cl[-1], cl[-7])
    r60 = pct(cl[-1], cl[-13])

    v5 = tv[-1] / (sum(tv[-7:-1]) / 6 or 1)
    a15 = (sum(tv[-3:]) / 3) / (sum(tv[-9:-3]) / 6 or 1)
    a30 = (sum(tv[-6:]) / 6) / (sum(tv[-18:-6]) / 12 or 1)

    score = 35.0
    score += (
        clamp((v5 - 1) * 12, -8, 18)
        + clamp((a15 - 1) * 18, -10, 25)
        + clamp((a30 - 1) * 10, -6, 14)
    )
    score += clamp(r5 * 3, -8, 8) + clamp(r15 * 1.8, -8, 10)

    over = 0.0
    if r5 > 2.5:
        over += (r5 - 2.5) * 5
    if r15 > 4.0:
        over += (r15 - 4.0) * 4
    if r60 > 7.0:
        over += (r60 - 7.0) * 2
    score -= clamp(over, 0, 35)

    chase = (r5 >= 3.0) or (r15 >= 5.0) or (r60 >= 8.0)
    early = (not chase) and (a15 >= 1.25) and (v5 >= 1.15)
    label = "CHASE" if chase else ("EARLY" if early else "WATCH")

    return {
        "market": m,
        "price": cl[-1],
        "score": round(clamp(score, 0, 100), 2),
        "label": label,
        "ret_5m": round(r5, 3),
        "ret_15m": round(r15, 3),
        "ret_30m": round(r30, 3),
        "ret_60m": round(r60, 3),
        "value_ratio_5m": round(v5, 3),
        "value_accel_15m": round(a15, 3),
        "value_accel_30m": round(a30, 3),
    }


def btc_state():
    c = candles("KRW-BTC", 5, 24)
    cl = [f(x["trade_price"]) for x in c]
    return {
        "ret_15m": round(pct(cl[-1], cl[-4]), 3),
        "ret_60m": round(pct(cl[-1], cl[-13]), 3),
    }


def load_json(path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)

    ms = markets()

    # 24h ê±°ëëê¸ ììë "íí°"ê° ìëë¼ ê¸°ë¡/ë¹êµì©ì¼ë¡ë§ ì¬ì©
    tick = []
    for i in range(0, len(ms), 100):
        tick += get("/v1/ticker", {"markets": ",".join(ms[i:i + 100])})
        time.sleep(0.12)

    ranked = sorted(
        tick,
        key=lambda x: f(x.get("acc_trade_price_24h")),
        reverse=True,
    )
    rank24 = {x["market"]: i + 1 for i, x in enumerate(ranked)}
    value24 = {x["market"]: f(x.get("acc_trade_price_24h")) for x in ranked}

    # íµì¬ ë³ê²½ì :
    # V7.3 = ê±°ëëê¸ ìì 120ê°ë§ feature ê³ì°
    # V7.6 = KRW ì ì²´ ì¢ëª©ì feature ê³ì°
    rows = []
    total = len(ms)

    for idx, m in enumerate(ms, 1):
        try:
            z = features(m)
            if z:
                r = rank24.get(m)
                z["trade_value_rank_24h"] = r
                z["trade_value_24h"] = round(value24.get(m, 0.0), 2)
                z["outside_top120"] = bool(r and r > 120)

                # A/B/Cë V7.5ì ëì¼í ê´ì°° ì¡°ê±´
                is_early = z["label"] == "EARLY"
                not_chase = z["label"] != "CHASE"

                z["A"] = bool(
                    is_early and not_chase
                    and z["score"] >= 85
                    and z["value_accel_15m"] >= 3
                    and z["value_accel_30m"] >= 2
                )

                z["B"] = bool(
                    is_early and not_chase
                    and z["score"] >= 80
                    and z["value_accel_15m"] >= 1.5
                    and z["value_accel_30m"] >= 3
                )

                z["C"] = bool(
                    is_early and not_chase
                    and z["score"] >= 70
                    and z["value_accel_15m"] >= 2
                    and z["value_accel_30m"] >= 2
                    and -1.5 <= z["ret_5m"] <= 0.5
                    and -2.0 <= z["ret_15m"] <= 1.0
                    and z["ret_30m"] <= 0.5
                )

                z["ABC_ALL"] = bool(z["A"] and z["B"] and z["C"])
                rows.append(z)

        except Exception as e:
            print("skip", m, e)

        # Upbit public API ë¶ë´ ìí
        time.sleep(0.07)

        if idx % 25 == 0 or idx == total:
            print(f"scanned {idx}/{total}")

    priority = {"EARLY": 0, "WATCH": 1, "CHASE": 2}
    rows.sort(
        key=lambda x: (
            priority[x["label"]],
            -x["score"],
            -x["value_accel_15m"],
        )
    )

    top = rows[:TOP_N]
    btc = btc_state()

    # ì ì²´ ìì¥ìì TOP120 ë°ì¸ë° ì¡°ê±´ì íµê³¼í íë³´ë¥¼ ë³ë ì ì¥
    outside_candidates = [
        x for x in rows
        if x.get("outside_top120")
        and x["label"] == "EARLY"
        and (x["A"] or x["B"] or x["C"])
    ]

    recs = []
    for rank, z in enumerate(top, 1):
        q = dict(z)
        q.update({
            "ts": now.isoformat(),
            "rank": rank,
            "btc": btc,
            "h1": None,
            "h3": None,
            "h6": None,
            "h24": None,
        })
        recs.append(q)

    hist = load_json(OUT, [])
    if not isinstance(hist, list):
        hist = []

    hist = recs + hist
    hist = hist[:MAX_ROWS]
    OUT.write_text(
        json.dumps(hist, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    latest = {
        "updated_at": now.isoformat(),
        "version": "V7.6 ALL_KRW",
        "scan_universe": "ALL_KRW",
        "scanned_markets": len(rows),
        "market_count": len(ms),
        "top_n": TOP_N,
        "btc": btc,
        "top": top,
        "outside_top120_candidates": outside_candidates[:30],
        "outside_top120_candidate_count": len(outside_candidates),
        "note": (
            "V7.3 score/EARLY ê·ì¹ì ê·¸ëë¡ ì ì§. "
            "24h ê±°ëëê¸ TOP120 ì íë§ ì ê±°. "
            "trade_value_rank_24hë íí°ê° ìëë¼ ë¹êµì© ê¸°ë¡."
        ),
    }

    LATEST.write_text(
        json.dumps(latest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("done")
    print("markets:", len(ms))
    print("feature rows:", len(rows))
    print("outside top120 A/B/C candidates:", len(outside_candidates))
    for x in top:
        print(
            x["market"],
            x["label"],
            x["score"],
            "rank24=", x.get("trade_value_rank_24h"),
            "OUT120" if x.get("outside_top120") else "",
        )


if __name__ == "__main__":
    main()
