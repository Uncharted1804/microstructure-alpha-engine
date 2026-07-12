"""
Binance WebSocket LOB + Trade data collector.

Usage:
    python -m src.data_collection.websocket_collector --duration 3600
"""

import asyncio
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import websockets
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SYMBOL = os.getenv("SYMBOL", "BTCUSDT").lower()
DEPTH_LEVELS = int(os.getenv("DEPTH_LEVELS", "5"))
DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
BASE_WS = os.getenv("BINANCE_BASE_WS", "wss://stream.binance.com:9443")

STREAM_URL = (
    f"{BASE_WS}/stream?streams={SYMBOL}@depth{DEPTH_LEVELS}@100ms/{SYMBOL}@trade"
)


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------
def parse_depth_snapshot(data: dict, event_time_ms: int) -> Optional[dict]:
    """
    Parse a depth5@100ms snapshot.

    This stream sends:
        lastUpdateId  : int
        bids          : [ [price_str, qty_str], ... ]  best first
        asks          : [ [price_str, qty_str], ... ]  best first

    NOTE: this format has NO "e" or "s" field — the stream name
    tells us what it is. We pass event_time_ms from the wrapper.
    """
    try:
        bids = sorted(data.get("b", data.get("bids", [])), key=lambda x: -float(x[0]))
        asks = sorted(data.get("a", data.get("asks", [])), key=lambda x: float(x[0]))

        if not bids or not asks:
            logger.warning("Depth snapshot arrived with empty bids or asks")
            return None

        record = {
            "timestamp_ms": event_time_ms,
            "symbol": SYMBOL.upper(),
        }

        for i, (price, qty) in enumerate(bids[:DEPTH_LEVELS], 1):
            record[f"bid_price_{i}"] = float(price)
            record[f"bid_qty_{i}"] = float(qty)

        for i, (price, qty) in enumerate(asks[:DEPTH_LEVELS], 1):
            record[f"ask_price_{i}"] = float(price)
            record[f"ask_qty_{i}"] = float(qty)

        # Pad any missing levels with zeros
        for i in range(len(bids) + 1, DEPTH_LEVELS + 1):
            record[f"bid_price_{i}"] = 0.0
            record[f"bid_qty_{i}"] = 0.0
        for i in range(len(asks) + 1, DEPTH_LEVELS + 1):
            record[f"ask_price_{i}"] = 0.0
            record[f"ask_qty_{i}"] = 0.0

        return record

    except (KeyError, ValueError, IndexError, TypeError) as e:
        logger.warning(f"parse_depth_snapshot failed: {e}")
        return None


def parse_trade_event(data: dict) -> Optional[dict]:
    """
    Parse a trade stream event.

    Fields:
        T  = trade time ms
        E  = event time ms
        s  = symbol
        p  = price
        q  = quantity
        m  = is_buyer_maker
             True  = aggressive SELL (hit the bid)
             False = aggressive BUY  (lifted the ask)
    """
    try:
        is_buyer_maker = bool(data["m"])
        quantity = float(data["q"])
        return {
            "timestamp_ms": int(data["T"]),
            "event_ms": int(data["E"]),
            "symbol": data["s"],
            "price": float(data["p"]),
            "quantity": quantity,
            "is_buyer_maker": is_buyer_maker,
            "signed_qty": quantity * (1 if not is_buyer_maker else -1),
        }
    except (KeyError, ValueError, TypeError) as e:
        logger.warning(f"parse_trade_event failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------
class LOBCollector:
    FLUSH_INTERVAL_SECONDS = 60  # write to disk every 60 seconds

    def __init__(self, run_id: str):
        self.run_id = run_id
        self.depth_buffer = []
        self.trade_buffer = []
        self.last_flush = time.time()
        self.depth_count = 0
        self.trade_count = 0
        self.msg_count = 0

        self.depth_dir = DATA_DIR / "raw" / "depth" / run_id
        self.trade_dir = DATA_DIR / "raw" / "trades" / run_id
        self.depth_dir.mkdir(parents=True, exist_ok=True)
        self.trade_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"LOBCollector started | run_id: {run_id}")
        logger.info(f"Stream: {STREAM_URL}")

    def handle_message(self, raw: str) -> None:
        self.msg_count += 1

        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON")
            return

        stream = msg.get("stream", "")
        data = msg.get("data", msg)
        event_type = data.get("e", "")

        # ------------------------------------------------------------------
        # DEPTH snapshot  (stream name contains "depth", no "e" field)
        # ------------------------------------------------------------------
        if "depth" in stream:
            # Use current system time as the snapshot timestamp
            # because depth5@100ms has no "E" timestamp in its payload
            event_time_ms = int(time.time() * 1000)
            record = parse_depth_snapshot(data, event_time_ms)
            if record:
                self.depth_buffer.append(record)
                self.depth_count += 1
                if self.depth_count % 100 == 0:
                    logger.info(f"Depth snapshots: {self.depth_count:,}")

        # ------------------------------------------------------------------
        # TRADE event  (stream name contains "trade", event type = "trade")
        # ------------------------------------------------------------------
        elif "trade" in stream or event_type == "trade":
            record = parse_trade_event(data)
            if record:
                self.trade_buffer.append(record)
                self.trade_count += 1
                if self.trade_count % 500 == 0:
                    logger.info(f"Trades: {self.trade_count:,}")

        else:
            if self.msg_count <= 10:
                logger.warning(
                    f"Unknown event | stream={stream!r} | "
                    f"event={event_type!r} | keys={list(data.keys())}"
                )

        # Periodic flush to disk
        if time.time() - self.last_flush >= self.FLUSH_INTERVAL_SECONDS:
            self._flush()

    def _flush(self) -> None:
        ts = int(time.time())

        if self.depth_buffer:
            df = pd.DataFrame(self.depth_buffer)
            path = self.depth_dir / f"depth_{ts}.parquet"
            df.to_parquet(path, index=False)
            logger.info(f"Flushed {len(self.depth_buffer):,} depth rows → {path}")
            self.depth_buffer = []

        if self.trade_buffer:
            df = pd.DataFrame(self.trade_buffer)
            path = self.trade_dir / f"trades_{ts}.parquet"
            df.to_parquet(path, index=False)
            logger.info(f"Flushed {len(self.trade_buffer):,} trade rows → {path}")
            self.trade_buffer = []

        self.last_flush = time.time()

    def finalize(self) -> None:
        self._flush()
        logger.info(
            f"Done | messages: {self.msg_count:,} | "
            f"depth: {self.depth_count:,} | trades: {self.trade_count:,}"
        )


# ---------------------------------------------------------------------------
# Async loop
# ---------------------------------------------------------------------------
async def run_collector(duration_seconds: int = 3600) -> None:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    collector = LOBCollector(run_id)
    start_time = time.time()

    logger.info(f"Collecting for {duration_seconds}s ...")

    while time.time() - start_time < duration_seconds:
        try:
            async with websockets.connect(
                STREAM_URL,
                ping_interval=20,
                ping_timeout=20,
                close_timeout=10,
            ) as ws:
                logger.info("WebSocket connected.")
                async for message in ws:
                    collector.handle_message(message)
                    if time.time() - start_time >= duration_seconds:
                        break

        except (websockets.ConnectionClosed, ConnectionResetError) as e:
            logger.warning(f"Disconnected: {e} — reconnecting in 2s...")
            await asyncio.sleep(2)
        except Exception as e:
            logger.error(f"Error: {e} — reconnecting in 5s...")
            await asyncio.sleep(5)

    collector.finalize()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=3600)
    args = parser.parse_args()

    logs_dir = DATA_DIR / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger.add(logs_dir / "collector_{time}.log", rotation="100 MB", level="DEBUG")

    asyncio.run(run_collector(args.duration))
