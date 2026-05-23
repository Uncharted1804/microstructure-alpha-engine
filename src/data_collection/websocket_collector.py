"""
Binance WebSocket LOB + Trade data collector.

Connects to Binance combined stream, records depth snapshots and
trade events to disk in Parquet format for later analysis.

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


# Configuration — reads from your .env file
# ---------------------------------------------------------------------------
SYMBOL = os.getenv("SYMBOL", "BTCUSDT").lower()
DEPTH_LEVELS = int(os.getenv("DEPTH_LEVELS", "5"))
DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
BASE_WS = os.getenv("BINANCE_BASE_WS", "wss://stream.binance.com:9443")

STREAM_URL = (
    f"{BASE_WS}/stream?streams={SYMBOL}@depth{DEPTH_LEVELS}@100ms/{SYMBOL}@trade"
)


# Parsers — turn raw JSON into flat Python dicts


def parse_depth_event(data: dict) -> Optional[dict]:
    """Parse a depthUpdate event into a flat dict."""
    try:
        record = {
            "timestamp_ms": data["E"],
            "symbol": data["s"],
        }
        bids = sorted(data["b"], key=lambda x: -float(x[0]))
        asks = sorted(data["a"], key=lambda x: float(x[0]))

        for i, (price, qty) in enumerate(bids[:DEPTH_LEVELS], 1):
            record[f"bid_price_{i}"] = float(price)
            record[f"bid_qty_{i}"] = float(qty)

        for i, (price, qty) in enumerate(asks[:DEPTH_LEVELS], 1):
            record[f"ask_price_{i}"] = float(price)
            record[f"ask_qty_{i}"] = float(qty)

        return record
    except (KeyError, ValueError, IndexError) as e:
        logger.warning(f"Failed to parse depth event: {e}")
        return None


def parse_trade_event(data: dict) -> Optional[dict]:
    """Parse a trade event into a flat dict."""
    try:
        return {
            "timestamp_ms": data["T"],
            "event_ms": data["E"],
            "symbol": data["s"],
            "price": float(data["p"]),
            "quantity": float(data["q"]),
            # m=True means buyer is market maker = aggressive SELL
            # m=False means seller is market maker = aggressive BUY
            "is_buyer_maker": data["m"],
            "signed_qty": float(data["q"]) * (1 if not data["m"] else -1),
        }
    except (KeyError, ValueError) as e:
        logger.warning(f"Failed to parse trade event: {e}")
        return None


# Collector class


class LOBCollector:
    FLUSH_INTERVAL_SECONDS = 300  # write to disk every 5 minutes

    def __init__(self, run_id: str):
        self.run_id = run_id
        self.depth_buffer: list = []
        self.trade_buffer: list = []
        self.last_flush = time.time()
        self.depth_count = 0
        self.trade_count = 0

        self.depth_dir = DATA_DIR / "raw" / "depth" / run_id
        self.trade_dir = DATA_DIR / "raw" / "trades" / run_id
        self.depth_dir.mkdir(parents=True, exist_ok=True)
        self.trade_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"LOBCollector started. Run ID: {run_id}")
        logger.info(f"Streaming: {STREAM_URL}")

    def handle_message(self, raw: str) -> None:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return

        data = msg.get("data", {})
        event_type = data.get("e", "")

        if event_type == "depthUpdate":
            record = parse_depth_event(data)
            if record:
                self.depth_buffer.append(record)
                self.depth_count += 1

        elif event_type == "trade":
            record = parse_trade_event(data)
            if record:
                self.trade_buffer.append(record)
                self.trade_count += 1

        if time.time() - self.last_flush > self.FLUSH_INTERVAL_SECONDS:
            self._flush()

    def _flush(self) -> None:
        ts = int(time.time())

        if self.depth_buffer:
            df = pd.DataFrame(self.depth_buffer)
            path = self.depth_dir / f"depth_{ts}.parquet"
            df.to_parquet(path, index=False)
            logger.info(f"Flushed {len(self.depth_buffer)} depth rows → {path}")
            self.depth_buffer = []

        if self.trade_buffer:
            df = pd.DataFrame(self.trade_buffer)
            path = self.trade_dir / f"trades_{ts}.parquet"
            df.to_parquet(path, index=False)
            logger.info(f"Flushed {len(self.trade_buffer)} trades → {path}")
            self.trade_buffer = []

        self.last_flush = time.time()

    def finalize(self) -> None:
        self._flush()
        logger.info(
            f"Collection complete. "
            f"Depth snapshots: {self.depth_count:,} | "
            f"Trades: {self.trade_count:,}"
        )


# ---------------------------------------------------------------------------
# Main async loop
# ---------------------------------------------------------------------------
async def run_collector(duration_seconds: int = 3600) -> None:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    collector = LOBCollector(run_id)
    start_time = time.time()

    logger.info(f"Collecting for {duration_seconds}s...")

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
            logger.warning(f"Disconnected: {e}. Reconnecting in 2s...")
            await asyncio.sleep(2)
        except Exception as e:
            logger.error(f"Error: {e}. Reconnecting in 5s...")
            await asyncio.sleep(5)

    collector.finalize()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=3600)
    args = parser.parse_args()

    logs_dir = DATA_DIR / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger.add(logs_dir / "collector_{time}.log", rotation="100 MB", level="INFO")

    asyncio.run(run_collector(args.duration))
