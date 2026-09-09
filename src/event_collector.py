#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V7.4 Upbit Event Collector
- V7.3 점수와 완전 분리
- 업비트 공식 Announcement WebSocket 수신
- API 키는 코드에 절대 직접 넣지 말 것

필수 환경변수:
  UPBIT_ACCESS_KEY
  UPBIT_SECRET_KEY

필수 패키지:
  pip install websocket-client PyJWT

출력:
  docs/data/v7_4_events.json
"""
import json, os, re, signal, sys, time, uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
import jwt
import websocket

WS_URL = "wss://api.upbit.com/websocket/v1/private"
OUT_PATH = Path("docs/data/v7_4_events.json")
MAX_EVENTS = 1000
CATEGORIES = ["trade", "digital_asset", "wallet", "event"]
KST = timezone(timedelta(hours=9))
STOP = False

POSITIVE_WORDS = [
    "신규 거래지원", "마켓 추가", "거래 지원", "에어드랍",
    "스냅샷", "토큰 소각", "바이백", "메인넷", "업그레이드", "상장"
]
NEGATIVE_WORDS = [
    "거래지원 종료", "유의 종목", "유의종목",
    "입출금 중단", "거래 중단", "유통량 변경"
]

def now_kst_iso():
    return datetime.now(KST).isoformat(timespec="seconds")

def load_events():
    if not OUT_PATH.exists():
        return []
    try:
        data = json.loads(OUT_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []

def save_events(events):
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(events[-MAX_EVENTS:], ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

def classify_event(title, body=""):
    text = f"{title} {body}".lower()
    pos = sum(x.lower() in text for x in POSITIVE_WORDS)
    neg = sum(x.lower() in text for x in NEGATIVE_WORDS)
    if neg > pos:
        return "NEGATIVE"
    if pos > neg:
        return "POSITIVE"
    return "NEUTRAL"

def extract_tickers(title, body=""):
    text = f"{title} {body}"
    found = []
    for m in re.finditer(r"\(([A-Z0-9]{2,12})\)", text):
        t = m.group(1)
        if t not in found:
            found.append(t)
    return found[:20]

def make_jwt():
    access = os.getenv("UPBIT_ACCESS_KEY", "").strip()
    secret = os.getenv("UPBIT_SECRET_KEY", "").strip()
    if not access or not secret:
        raise RuntimeError(
            "UPBIT_ACCESS_KEY / UPBIT_SECRET_KEY 환경변수가 없습니다. "
            "키를 코드에 직접 적지 마세요."
        )
    token = jwt.encode(
        {"access_key": access, "nonce": str(uuid.uuid4())},
        secret,
        algorithm="HS512"
    )
    return token.decode() if isinstance(token, bytes) else token

def normalize_message(msg):
    title = msg.get("title") or ""
    body = msg.get("body") or ""
    return {
        "uuid": str(msg.get("uuid") or ""),
        "event_type": msg.get("event_type"),
        "category": msg.get("category"),
        "title": title,
        "url": msg.get("url"),
        "first_listed_at": msg.get("first_listed_at"),
        "listed_at": msg.get("listed_at"),
        "received_at": now_kst_iso(),
        "tickers": extract_tickers(title, body),
        "sentiment": classify_event(title, body),
        "source": "UPBIT_OFFICIAL"
    }

def upsert_event(events, item):
    uid = item.get("uuid")
    if uid:
        for i, old in enumerate(events):
            if old.get("uuid") == uid:
                events[i] = item
                return
    events.append(item)

def stop_handler(signum, frame):
    global STOP
    STOP = True

def main():
    global STOP
    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)

    token = make_jwt()
    headers = [f"Authorization: Bearer {token}"]
    subscription = [
        {"ticket": f"v7-4-{uuid.uuid4()}"},
        {
            "type": "announcement",
            "categories": CATEGORIES,
            "include_body": True
        },
        {"format": "DEFAULT"}
    ]

    events = load_events()
    print("[V7.4] Upbit Announcement WebSocket 연결 시작")

    while not STOP:
        ws = None
        try:
            ws = websocket.create_connection(
                WS_URL, header=headers, timeout=30
            )
            ws.send(json.dumps(subscription, ensure_ascii=False))

            while not STOP:
                raw = ws.recv()
                if not raw:
                    continue
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8", errors="replace")
                msg = json.loads(raw)
                if msg.get("type") != "announcement":
                    continue

                item = normalize_message(msg)
                upsert_event(events, item)
                save_events(events)

                print(
                    f"[{item['received_at']}] "
                    f"{item['category']} / {item['sentiment']} / "
                    f"{','.join(item['tickers']) or '-'} / {item['title']}"
                )

        except KeyboardInterrupt:
            STOP = True
        except Exception as e:
            if STOP:
                break
            print(f"[V7.4] 연결 오류: {e}", file=sys.stderr)
            time.sleep(5)
        finally:
            try:
                if ws is not None:
                    ws.close()
            except Exception:
                pass

    save_events(events)
    print("[V7.4] 안전 종료")

if __name__ == "__main__":
    main()
