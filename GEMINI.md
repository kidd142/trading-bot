# kidd142 Trading Bot

## Project Overview
This project is a highly advanced, standalone Python-based cryptocurrency trading simulator. It integrates **Fuzzy Logic**, **Smart Money Concepts (SMC)**, and **Exponential Moving Average (EMA) Retest** strategies to simulate trading decisions in a volatility-rich environment.

The system is designed to run in a lightweight environment without heavy dependencies like `pandas` or `numpy`. It uses pure Python to implement complex mathematical indicators and logic.

### Key Features
*   **Real-time Data Integration:** Fetches live OHLCV data directly from Binance Futures API (BTCUSDT, ETHUSDT, etc.).
*   **Advanced Indicators (Manual Implementation):**
    *   **RSI (Relative Strength Index):** Overbought/Oversold detection.
    *   **MACD (Moving Average Convergence Divergence):** Momentum and trend confirmation.
    *   **ATR (Average True Range):** Volatility-based Stop Loss and Take Profit calculations.
    *   **Bollinger Bands:** Volatility and deviation measurement (`bandwidth` for squeeze detection).
    *   **ADX (Average Directional Index):** Trend strength filter.
    *   **EMA (Exponential Moving Average):** Trend baseline (period 21).
    *   **SMC (Smart Money Concepts):**
        *   `find_swing_points_advanced`: Identifikasi Swing High/Low yang lebih robust.
        *   `detect_bos`: Deteksi Break of Structure (BoS) untuk konfirmasi perubahan struktur pasar.
        *   `detect_order_block`: Deteksi Order Block (OB) sederhana sebagai area minat untuk retest.
    *   **Delta Metrics:** `calculate_delta_metrics` untuk Volume, Open Interest, dan Funding Rate (digunakan sebagai konfirmasi tambahan).
*   **Artificial Intelligence:**
    *   **Fuzzy Logic Engine:** Calculates a "Confidence Score" (0-100%) for every trade based on multiple inputs (EMA distance, RSI, MACD histogram, **SMC confluence, Volume/OI delta**).
*   **Market Simulation:**
    *   Generates synthetic OHLCV data with realistic properties if Binance API fetching fails.
*   **Multi-Mode Strategy Engine:**
    *   `SCALP`: 15-minute timeframe, aggressive entries, tight risk management.
    *   `INTRADAY`: 1-hour timeframe, balanced approach.
    *   `SWING`: 4-hour timeframe, trend following.
    *   `MTF` (Multi-Timeframe): Filters trend on 4H, executes entries on 1H.
*   **Intelligent Entry Filters:**
    *   **ADX Filter:** Only enters trades if ADX > 20 (confirms trending market). Confidence boosted if ADX > 35.
    *   **Bollinger Squeeze Filter:** Avoids entries if Bollinger Bandwidth is too narrow (< 0.5% volatility), preventing trades in dead/sideways markets.
    *   **SMC Confluence:** Prioritizes trades that align with detected Break of Structure and retest of Order Blocks.
    *   **Volume/OI Delta Confirmation:** Uses changes in Volume and Open Interest to confirm market conviction behind a move.
*   **Adaptive Exit Strategies:**
    *   **Hard SL/TP:** Fixed Stop Loss and Take Profit based on ATR.
    *   **Adaptive RSI Exit (`RSI_TP`):** Closes positions early if RSI hits overbought/oversold levels (e.g., LONG exits if RSI > 70, SHORT exits if RSI < 30) to lock in profits or avoid reversals.
    *   **Time-Based Exit (`TIME_EXIT`):** Closes stale positions after a set number of candles (e.g., 12 candles / 3 hours for 15m TF) to free up capital.
*   **Enhanced CLI Output:** Colorful and informative terminal logs using ANSI escape codes for better readability.

## Project Structure
*   **`trading_sim.py`**: The core application file. Contains:
    *   `MarketData`: Fetches real-time data from Binance Futures or generates synthetic data.
    *   `Indicators`: Manual implementation of technical analysis formulas, including advanced SMC concepts.
    *   `AdvancedFuzzy`: The AI logic for trade scoring with enhanced inputs.
    *   `TradingEngine`: Manages the simulation loop, orders, and PnL tracking, incorporating strategy modes, filters, and adaptive exits.
*   **`simulation_report.html`**: A comprehensive visual report generated after a simulation run. Includes:
    *   Price Action & EMA chart.
    *   RSI chart with overbought/oversold levels.
    *   MACD chart with Signal and Histogram.
    *   Detailed trade history table.
    *   Summary statistics (Final Balance, Total Return, Total Trades, Win Rate).
*   **`test_fuzzy.py`**: (Deprecated) Unit tests for an older version of the fuzzy logic class.

## Usage

The project is an interactive CLI application. You run the script and it will guide you through choosing a strategy mode and a cryptocurrency symbol.

### Prerequisites
*   Python 3.x
*   Internet connection (for Binance API data)

### Running the Simulation
Execute the script without any arguments to start the interactive menu:

```bash
python3 trading_sim.py
```

The script will then prompt you to select a mode and enter a symbol.

**Available Modes:**
*   `SCALP` (Default)
*   `INTRADAY`
*   `SWING`
*   `MTF`

**Available Symbols (Binance Futures):**
Any valid Futures pair on Binance, e.g., `BTCUSDT` (Default), `ETHUSDT`, `BNBUSDT`, `SOLUSDT`, `XRPUSDT`.

### Output
1.  **Console Logs:** Highly informative and colorful trade logs are printed to the terminal, detailing entries, exits, PnL, and current balance.
2.  **Visual Report:** A `simulation_report.html` file is generated/overwritten in the current directory. Open this file in a web browser to view the comprehensive simulation dashboard.

## Development Conventions
*   **Pure Python:** The project avoids external heavy data libraries to ensure compatibility in restricted environments. All math and data structures are native Python lists and dictionaries.
*   **Manual Math:** Indicators are calculated from scratch using sliding window logic.
*   **Fuzzy Logic:** Implemented using manual trapezoidal/triangular membership functions, now enriched with SMC and delta metrics inputs.