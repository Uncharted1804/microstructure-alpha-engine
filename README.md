# Microstructure Alpha Engine

**Tick-level LOB order flow imbalance signal with Kyle's lambda, Hawkes process calibration, and TC-adjusted backtesting on live Binance crypto data**

---

## What this project does

This engine streams live limit order book (LOB) data from the Binance WebSocket API at 100ms resolution and builds a short-term price prediction signal based on **Order Flow Imbalance (OFI)**. It estimates Kyle's lambda to decompose price impact into permanent and temporary components, calibrates a Hawkes self-exciting point process on trade arrival times, and backtests the resulting signal with full transaction cost accounting.

The project operates at **tick level** — 100ms LOB snapshots and individual trade events — rather than daily OHLCV data. This is the same data resolution that prop desks at Citadel Securities and Jane Street operate at. The implementation validates the OFI signal theoretically and empirically, and diagnoses precisely why it cannot be captured at retail infrastructure latency — a finding that demonstrates genuine understanding of market microstructure.

---

## Pipeline

```
Raw LOB data → OFI features → Price impact model → Signal backtest → TC-adjusted PnL
```

---

## Core concepts

| Concept | What it is |
|---------|-----------|
| **Order Flow Imbalance (OFI)** | Net bid vs ask pressure from order book events — linear predictor of short-term mid-price moves. Based on Cont, Kukanov & Stoikov (2014) |
| **Kyle's λ** | Price impact coefficient: ΔP = λ × signed_flow. Decomposed into permanent (informed trading) and temporary (liquidity) components. Based on Kyle (1985) |
| **Hawkes process** | Self-exciting point process modelling trade clustering — trades beget more trades. Calibrated via maximum likelihood estimation implemented from scratch in scipy |
| **Adverse selection** | Why market makers widen spreads against informed flow — the core microstructure problem linking OFI predictability to the Glosten-Milgrom model |

---

## Results

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **OFI β coefficient** | 6.03e-06 | Positive — buying pressure predicts upward price moves |
| **OFI R²** | 3.71% | Strong for a 1-second tick-level signal (literature reports 2–8%) |
| **OFI t-statistic** | 12.55 | Overwhelmingly significant |
| **OFI p-value** | 4.03e-36 | Essentially zero — signal is real, not noise |
| **Ridge OOS R²** | 4.66% | Higher than in-sample OLS — no overfitting, generalises well |
| **Kyle's λ (full sample)** | 1e-06 | Very liquid market — deep book absorbs flow |
| **Permanent impact** | -11.6% | Price overshoots and reverses — noise trading dominant in sample |
| **Temporary impact** | 111.6% | Full mean reversion — consistent with microstructure literature |
| **Hawkes μ (baseline rate)** | 2.52 trades/sec | Background trade intensity |
| **Hawkes branching ratio n** | 0.247 | 24.7% of trades are endogenous (self-exciting) |
| **Hawkes decay half-life** | ~0.17 seconds | Excitement decays in under 200ms |
| **Signal gross per trade** | 7.90 bps | Consistent across all parameter combinations |
| **Break-even TC** | 8.00 bps | 4 bps taker fee × 2 sides + spread |
| **TC gap** | −0.10 bps | Signal misses break-even by 0.10 bps |

### Key finding

The OFI signal is statistically real (R² = 3.71%, p ≈ 0) but generates 7.90 bps gross per trade against an 8.00 bps break-even cost at 100ms resolution. This 0.10 bps gap is a direct consequence of **signal decay** — OFI is most predictive at sub-10ms horizons. The profitable edge exists but requires co-located infrastructure operating at microsecond latency. This finding is consistent with Cont et al. (2014) and explains precisely why HFT firms invest in low-latency execution infrastructure.

---

## Data

- **Source**: Binance WebSocket API (`btcusdt@depth5@100ms` + `btcusdt@trade`)
- **Asset**: BTC/USDT perpetual
- **Collection period**: 1 hour of live tick data
- **Depth snapshots**: 35,994 (100ms resolution)
- **Individual trades**: 23,000
- **Mid-price range**: ~$76,600–$76,800
- **Average spread**: $0.01 (0.013 bps — extremely liquid)
- **Trade coverage**: 82% of snapshots have at least one trade in the preceding second

---

## Stack

```
Python 3.13  ·  pandas  ·  numpy  ·  scipy  ·  statsmodels  ·  scikit-learn
websockets   ·  pyarrow  ·  matplotlib  ·  seaborn  ·  loguru  ·  python-dotenv
Binance WebSocket API  ·  Parquet storage
```

No external Hawkes library. The full log-likelihood is implemented from scratch using `scipy.optimize` with L-BFGS-B and explicit parameter bounds — equivalent in accuracy to `tick` or `hawkeslib` but fully transparent.

---

## Project structure

```
microstructure-alpha/
├── src/
│   ├── data_collection/
│   │   ├── websocket_collector.py     ← Binance WS stream → Parquet
│   │   └── lob_reconstructor.py       ← mid-price, spread, micro-price, trade merge
│   ├── features/
│   │   └── ofi_features.py            ← multi-level OFI, z-score, forward returns
│   ├── models/
│   │   ├── kyles_lambda.py            ← OLS estimation, rolling lambda, decomposition
│   │   └── price_impact.py            ← OLS + Ridge with TimeSeriesSplit CV
│   ├── hawkes/
│   │   └── hawkes_calibration.py      ← MLE from scratch, rolling calibration
│   └── backtest/
│       └── signal_backtest.py         ← TC-adjusted backtest in basis points
├── notebooks/
│   ├── 01_data_exploration.ipynb      ← price, spread, queue imbalance
│   ├── 02_ofi_analysis.ipynb          ← OFI vs forward returns, decile plot
│   ├── 03_kyles_lambda.ipynb          ← rolling lambda, impact decomposition
│   ├── 04_hawkes.ipynb                ← intensity plot, rolling calibration
│   └── 05_backtest.ipynb              ← equity curve, parameter sweep heatmap
├── data/                              ← gitignored
│   ├── raw/depth/                     ← LOB snapshots (Parquet)
│   └── raw/trades/                    ← trade events (Parquet)
├── tests/
│   ├── test_features.py
│   └── test_models.py
├── .env.example
├── requirements.txt
└── README.md
```

---

## Setup

```powershell
git clone https://github.com/Uncharted1804/microstructure-alpha-engine
cd microstructure-alpha-engine
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

---

## Usage

### Collect live LOB data

```powershell
# Collect 1 hour of BTC/USDT tick data
python -m src.data_collection.websocket_collector --duration 3600
```

Data saves to `data/raw/depth/{run_id}/` and `data/raw/trades/{run_id}/` as Parquet files, flushed every 60 seconds.

### Run the full analysis

Open notebooks in order in VSCode (select `.venv` as kernel):

```
notebooks/01_data_exploration.ipynb
notebooks/02_ofi_analysis.ipynb
notebooks/03_kyles_lambda.ipynb
notebooks/04_hawkes.ipynb
notebooks/05_backtest.ipynb
```

Replace `RUN_ID` in the first cell of each notebook with your collected run_id.

### Run tests

```powershell
pytest tests/ -v
```

---

## References

- Cont, R., Kukanov, A., & Stoikov, S. (2014). *The Price Impact of Order Book Events*. Journal of Financial Econometrics, 12(1), 47–88.
- Kyle, A. S. (1985). *Continuous Auctions and Insider Trading*. Econometrica, 53(6), 1315–1335.
- Glosten, L. R., & Milgrom, P. R. (1985). *Bid, Ask and Transaction Prices in a Specialist Market with Heterogeneously Informed Traders*. Journal of Financial Economics, 14(1), 71–100.
- Hawkes, A. G. (1971). *Spectra of some self-exciting and mutually exciting point processes*. Biometrika, 58(1), 83–90.
- Bacry, E., Mastromatteo, I., & Muzy, J. F. (2015). *Hawkes Processes in Finance*. Market Microstructure and Liquidity, 1(1), 1550005.

---

## Author

**Shaurya Mittal Nair** — built as a portfolio project demonstrating tick-level quantitative research methodology. All data collected live from public Binance API endpoints. No proprietary data used.