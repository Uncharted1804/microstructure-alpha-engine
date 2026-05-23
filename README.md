# Microstructure Alpha Engine

**LOB order flow imbalance → short-term price prediction on crypto tick data**

## What this project does

This engine collects live tick-level limit order book (LOB) data from the Binance 
WebSocket API and builds a short-term price prediction signal based on Order Flow 
Imbalance (OFI). It estimates Kyle's lambda (price impact coefficient), models trade 
arrival clustering via Hawkes processes, and backtests the resulting signal with 
realistic transaction costs.

## Why this is unusual

Most quant projects backtest on daily OHLCV data. This operates at the 
**millisecond tick level** — the same data resolution that prop desks at Citadel 
Securities and Jane Street use. I made this project to show the depth of understanding I 
have of real market mocrostructure. I modelled the LOB and estimated Kyle's lambda from crypto order flow.

## Pipeline

Raw LOB data → OFI features → Price impact model → Signal backtest → TC-adjusted PnL

## Core concepts

| Concept | What it is |
|---------|-----------|
| **OFI** | Net bid/ask pressure from order events → linear predictor of mid-price moves |
| **Kyle's λ** | Price impact coefficient: ΔP = λ * signed_flow |
| **Hawkes** | Self-exciting point processes → models trade clustering |
| **Adverse selection** | Why market makers widen spreads against informed flow |

## Stack

Python 3.10+ · pandas/polars · Binance WS API · statsmodels · scipy (Hawkes MLE) · 
sklearn · matplotlib/plotly

## Structure

├── src/
│   ├── data_collection/    # WebSocket LOB collection
│   ├── features/           # OFI, spread, queue imbalance
│   ├── models/             # Kyle's lambda, price impact regression
│   ├── hawkes/             # Hawkes process calibration
│   └── backtest/           # Signal backtesting with TC
├── notebooks/              # Research notebooks
├── data/                   # Raw + processed data (gitignored)
└── tests/                  # Unit tests

## Setup

```powershell
git clone https://github.com/YOUR_USERNAME/microstructure-alpha-engine
cd microstructure-alpha-engine
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env  # Add your config
```
