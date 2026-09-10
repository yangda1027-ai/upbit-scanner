#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
V7.5 Project Phoenix - 3 Strategy Forward Tournament
=====================================================

목적
- V7.3 점수/분류/수집 로직은 수정하지 않는다.
- V7.3이 쌓은 docs/data/v7_3_signals.json만 읽는다.
- 서로 다른 3개 Phoenix 후보 규칙을 동시에 고정해서 forward-test 한다.
- 첫 실행 시각을 forward_start_utc로 저장하고, 그 이후 신호만 "공식 Forward" 통계에 포함한다.
- 첫 실행 이전 데이터는 참고용 Historical(In-sample) 통계로 분리한다.

중요
- 이 파일은 매수/매도 프로그램이 아니다.
- h1/h3/h6/h24 결과는 V7.3 evaluator가 기록한 값을 그대로 사용한다.
- 현재 h24 의미도 V7.3 evaluator의 기존 정의를 그대로 상속한다.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "data" / "v7_3_signals.json"
OUT = ROOT / "docs" / "data" / "v7_5_tournament.json"

EPISODE_GAP_MINUTES = 60

def f(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default

def parse_ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))

def is_chase(r: Dict[str, Any]) -> bool:
    # V7.3 CHASE 기준 그대로
    return (
        f(r.get("ret_5m")) >= 3.0
        or f(r.get("ret_15m")) >= 5.0
        or f(r.get("ret_60m")) >= 8.0
    )

# --------------------------
# 3개 전략: 지금부터 고정
# --------------------------

def strategy_a(r: Dict[str, Any]) -> bool:
    """
    A = Legacy Phoenix
    이전에 잡았던 85 / 3 / 2 규칙.
    """
    return (
        r.get("label") == "EARLY"
        and not is_chase(r)
        and f(r.get("score")) >= 85.0
        and f(r.get("value_accel_15m")) >= 3.0
        and f(r.get("value_accel_30m")) >= 2.0
    )

def strategy_b(r: Dict[str, Any]) -> bool:
    """
    B = 30m Flow
    최신 표본에서 30분 거래대금 가속의 분리력이 더 좋아 보여
    30m 흐름에 무게를 둔 탐색 규칙.
    """
    return (
        r.get("label") == "EARLY"
        and not is_chase(r)
        and f(r.get("score")) >= 80.0
        and f(r.get("value_accel_15m")) >= 1.5
        and f(r.get("value_accel_30m")) >= 3.0
    )

def strategy_c(r: Dict[str, Any]) -> bool:
    """
    C = Compression + Flow
    가격은 아직 눌리거나 정체인데 거래대금 가속이 살아있는 잠복형.
    이 규칙은 '가격 선행 없이 거래대금이 먼저 붙는가'를 따로 검증하기 위한 실험군.
    """
    r5 = f(r.get("ret_5m"))
    r15 = f(r.get("ret_15m"))
    r30 = f(r.get("ret_30m"))

    return (
        r.get("label") == "EARLY"
        and not is_chase(r)
        and f(r.get("score")) >= 70.0
        and f(r.get("value_accel_15m")) >= 2.0
        and f(r.get("value_accel_30m")) >= 2.0
        and -1.5 <= r5 <= 0.5
        and -2.0 <= r15 <= 1.0
        and r30 <= 0.5
    )

STRATEGIES: Dict[str, Dict[str, Any]] = {
    "A": {
        "name": "Legacy 85/3/2",
        "description": "SCORE≥85 · 15m가속≥3 · 30m가속≥2",
        "fn": strategy_a,
    },
    "B": {
        "name": "30m Flow",
        "description": "SCORE≥80 · 15m가속≥1.5 · 30m가속≥3",
        "fn": strategy_b,
    },
    "C": {
        "name": "Compression + Flow",
        "description": "SCORE≥70 · 가격눌림/정체 · 15m가속≥2 · 30m가속≥2",
        "fn": strategy_c,
    },
}

def outcome(r: Dict[str, Any], key: str) -> Optional[Dict[str, Any]]:
    v = r.get(key)
    if not isinstance(v, dict):
        return None
    return {
        "close_pct": v.get("close_pct"),
        "max_pct": v.get("max_pct"),
        "min_pct": v.get("min_pct"),
        "hit_3": v.get("hit_3"),
        "hit_5": v.get("hit_5"),
        "hit_10": v.get("hit_10"),
    }

def row_to_signal(r: Dict[str, Any], strategy_id: str) -> Dict[str, Any]:
    x = {
        "strategy": strategy_id,
        "market": r.get("market"),
        "price": r.get("price"),
        "score": r.get("score"),
        "source_label": r.get("label"),
        "ret_5m": r.get("ret_5m"),
        "ret_15m": r.get("ret_15m"),
        "ret_30m": r.get("ret_30m"),
        "ret_60m": r.get("ret_60m"),
        "value_ratio_5m": r.get("value_ratio_5m"),
        "value_accel_15m": r.get("value_accel_15m"),
        "value_accel_30m": r.get("value_accel_30m"),
        "ts": r.get("ts"),
        "rank": r.get("rank"),
        "btc": r.get("btc"),
        "h1": outcome(r, "h1"),
        "h3": outcome(r, "h3"),
        "h6": outcome(r, "h6"),
        "h24": outcome(r, "h24"),
    }
    x["status"] = classify(x)
    return x

def classify(x: Dict[str, Any]) -> str:
    h6 = x.get("h6")
    h24 = x.get("h24")

    if not isinstance(h6, dict) or h6.get("max_pct") is None:
        return "TRACKING"

    h6max = f(h6.get("max_pct"))
    if h6max >= 3.0:
        return "FAST_MOVE"

    if not isinstance(h24, dict) or h24.get("max_pct") is None:
        return "SLEEPING"

    h24max = f(h24.get("max_pct"))
    if h24max >= 10.0:
        return "PHOENIX_10"
    if h24max >= 5.0:
        return "PHOENIX_5"
    if h24max >= 3.0:
        return "PHOENIX_3"
    return "FAILED_LATE"

def build_episodes(rows: List[Dict[str, Any]], sid: str, fn: Callable[[Dict[str, Any]], bool]) -> List[Dict[str, Any]]:
    matched = [r for r in rows if isinstance(r, dict) and fn(r)]
    matched.sort(key=lambda r: parse_ts(r["ts"]))

    episodes: List[Dict[str, Any]] = []
    last_by_market: Dict[str, datetime] = {}

    for r in matched:
        market = r.get("market")
        if not market or not r.get("ts"):
            continue
        ts = parse_ts(r["ts"])
        prev = last_by_market.get(market)

        # 같은 코인이 60분 이내 반복되면 한 에피소드로 묶고 최초 신호만 사용
        if prev is not None and (ts - prev) <= timedelta(minutes=EPISODE_GAP_MINUTES):
            last_by_market[market] = ts
            continue

        last_by_market[market] = ts
        episodes.append(row_to_signal(r, sid))

    return episodes

def pct(n: int, d: int) -> Optional[float]:
    return round(100.0 * n / d, 2) if d else None

def stats_for(signals: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(signals)
    h6_done = [x for x in signals if isinstance(x.get("h6"), dict) and x["h6"].get("max_pct") is not None]
    fast = [x for x in h6_done if f(x["h6"].get("max_pct")) >= 3.0]

    sleeper_eval = [
        x for x in signals
        if isinstance(x.get("h6"), dict)
        and x["h6"].get("max_pct") is not None
        and f(x["h6"].get("max_pct")) < 3.0
        and isinstance(x.get("h24"), dict)
        and x["h24"].get("max_pct") is not None
    ]

    hit3 = [x for x in sleeper_eval if f(x["h24"].get("max_pct")) >= 3.0]
    hit5 = [x for x in sleeper_eval if f(x["h24"].get("max_pct")) >= 5.0]
    hit10 = [x for x in sleeper_eval if f(x["h24"].get("max_pct")) >= 10.0]

    return {
        "episodes": total,
        "h6_matured": len(h6_done),
        "fast_move_n": len(fast),
        "fast_move_rate_pct": pct(len(fast), len(h6_done)),
        "sleeper_evaluated_n": len(sleeper_eval),
        "sleeper_hit3_n": len(hit3),
        "sleeper_hit3_pct": pct(len(hit3), len(sleeper_eval)),
        "sleeper_hit5_n": len(hit5),
        "sleeper_hit5_pct": pct(len(hit5), len(sleeper_eval)),
        "sleeper_hit10_n": len(hit10),
        "sleeper_hit10_pct": pct(len(hit10), len(sleeper_eval)),
    }

def load_or_create_forward_start() -> str:
    # 기존 결과 파일이 있으면 첫 시작 시각을 절대 바꾸지 않는다.
    if OUT.exists():
        try:
            old = json.loads(OUT.read_text(encoding="utf-8"))
            fs = old.get("meta", {}).get("forward_start_utc")
            if fs:
                return fs
        except Exception:
            pass
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def main() -> None:
    if not SRC.exists():
        raise FileNotFoundError(f"V7.3 데이터가 없습니다: {SRC}")

    rows = json.loads(SRC.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("v7_3_signals.json 최상위 구조가 list가 아닙니다.")

    forward_start_utc = load_or_create_forward_start()
    forward_start = parse_ts(forward_start_utc)

    all_signals: List[Dict[str, Any]] = []
    strategy_output: Dict[str, Any] = {}

    for sid, info in STRATEGIES.items():
        episodes = build_episodes(rows, sid, info["fn"])
        historical = [x for x in episodes if parse_ts(x["ts"]) < forward_start]
        forward = [x for x in episodes if parse_ts(x["ts"]) >= forward_start]

        strategy_output[sid] = {
            "name": info["name"],
            "description": info["description"],
            "historical_in_sample": stats_for(historical),
            "official_forward": stats_for(forward),
        }
        all_signals.extend(episodes)

    all_signals.sort(key=lambda x: parse_ts(x["ts"]), reverse=True)

    # Forward 승자 판정은 24h sleeper 평가 표본이 충분할 때만
    eligible = []
    for sid, s in strategy_output.items():
        st = s["official_forward"]
        if st["sleeper_evaluated_n"] >= 30 and st["sleeper_hit5_pct"] is not None:
            eligible.append((st["sleeper_hit5_pct"], sid))

    leader = None
    if eligible:
        eligible.sort(reverse=True)
        leader = eligible[0][1]

    payload = {
        "meta": {
            "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "forward_start_utc": forward_start_utc,
            "episode_gap_minutes": EPISODE_GAP_MINUTES,
            "source": "docs/data/v7_3_signals.json",
            "note": "Official Forward 통계는 forward_start_utc 이후 신호만 포함. 그 이전은 참고용 in-sample.",
            "h24_note": "h24는 V7.3 evaluator가 기록한 기존 의미를 그대로 사용함.",
            "leader_rule": "공식 sleeper_evaluated_n >= 30인 전략만 +5% 적중률로 임시 리더 선정",
            "leader": leader,
        },
        "strategies": strategy_output,
        "signals": all_signals[:1000],
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== V7.5 Phoenix Tournament ===")
    print("Forward start:", forward_start_utc)
    for sid, s in strategy_output.items():
        print(f"[{sid}] {s['name']}")
        print("  historical:", s["historical_in_sample"])
        print("  forward   :", s["official_forward"])
    print("Leader:", leader)
    print("Output:", OUT)

if __name__ == "__main__":
    main()
