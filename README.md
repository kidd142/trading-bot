# 🚀 kidd142 Trading Bot

![Python Version](https://img.shields.io/badge/Python-3.x-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg) <!-- Asumsi MIT License, bisa diganti -->

## Project Overview

Selamat datang di **kidd142 Trading Bot**! Ini adalah simulator trading futures cryptocurrency canggih yang dibangun sepenuhnya menggunakan Python murni. Bot ini mengintegrasikan **Fuzzy Logic (AI)**, **Smart Money Concepts (SMC)**, dan strategi **EMA (Exponential Moving Average) Retest** untuk mensimulasikan keputusan trading yang cerdas dalam lingkungan pasar yang dinamis.

Dirancang untuk berjalan di lingkungan yang ringan tanpa ketergantungan eksternal yang kompleks (seperti `pandas` atau `numpy`), bot ini mengimplementasikan semua indikator dan logika matematika secara manual.

## Key Features

*   **Real-time Data Integration:** Mengambil data OHLCV (Open, High, Low, Close, Volume) secara langsung dari Binance Futures API (mendukung berbagai pasangan seperti BTCUSDT, ETHUSDT, dll.).
*   **Advanced Indicators (Pure Python Implementation):**
    *   **RSI (Relative Strength Index):** Deteksi Overbought/Oversold.
    *   **MACD (Moving Average Convergence Divergence):** Konfirmasi momentum dan tren.
    *   **ATR (Average True Range):** Digunakan untuk perhitungan Stop Loss dan Take Profit yang dinamis berdasarkan volatilitas.
    *   **Bollinger Bands:** Mengukur volatilitas dan deviasi harga, dengan deteksi `bandwidth` untuk Bollinger Squeeze.
    *   **ADX (Average Directional Index):** Filter kekuatan tren.
    *   **EMA (Exponential Moving Average):** Garis dasar tren (periode 21).
    *   **SMC (Smart Money Concepts):**
        *   **Swing Points:** Identifikasi Swing High/Low yang signifikan.
        *   **Break of Structure (BoS):** Konfirmasi perubahan struktur pasar.
        *   **Order Block (OB):** Deteksi zona Order Block sederhana sebagai area minat untuk retest.
    *   **Delta Metrics:** Perhitungan perubahan Volume, Open Interest (OI), dan Funding Rate untuk konfirmasi sentimen pasar.
*   **Artificial Intelligence (Fuzzy Logic):**
    *   **Fuzzy Logic Engine:** Menghitung "Confidence Score" (0-100%) untuk setiap sinyal trade. Skor ini mempertimbangkan berbagai input seperti jarak ke EMA, RSI, histogram MACD, konfirmasi SMC, serta perubahan Volume/OI.
*   **Market Simulation:**
    *   Generates synthetic OHLCV data with realistic properties if Binance API fetching fails.
*   **Multi-Mode Strategy Engine:**
    *   `SCALP`: Timeframe 15 menit, entri agresif, manajemen risiko ketat.
    *   `INTRADAY`: Timeframe 1 jam, pendekatan seimbang untuk trading harian.
    *   `SWING`: Timeframe 4 jam, berfokus pada mengikuti tren jangka menengah.
    *   `MTF` (Multi-Timeframe): Menggunakan filter tren di 4H dan mencari entri presisi di 1H untuk konfirmasi ganda.
*   **Intelligent Entry Filters:**
    *   **ADX Filter:** Hanya masuk trade jika ADX > 20 (mengkonfirmasi pasar sedang trending). Confidence boosted jika ADX > 35.
    *   **Bollinger Squeeze Filter:** Menghindari entri jika Bollinger Bandwidth terlalu sempit (< 0.5% volatilitas), mencegah trade di pasar sideways/mati.
    *   **SMC Confluence:** Memprioritaskan trade yang sejalan dengan Break of Structure yang terdeteksi dan retest pada Order Blocks.
    *   **Volume/OI Delta Confirmation:** Menggunakan perubahan Volume dan Open Interest untuk mengkonfirmasi keyakinan pasar di balik pergerakan harga.
*   **Adaptive Exit Strategies:**
    *   **Hard SL/TP:** Stop Loss dan Take Profit tetap berdasarkan ATR.
    *   **Adaptive RSI Exit (`RSI_TP`):** Menutup posisi lebih awal jika RSI mencapai level overbought/oversold untuk mengunci keuntungan atau menghindari pembalikan harga.
    *   **Time-Based Exit (`TIME_EXIT`):** Menutup posisi yang "stale" (tidak bergerak) setelah sejumlah candle tertentu (misal: 12 candle) untuk membebaskan modal.
*   **Enhanced CLI Output:** Log terminal yang informatif dan berwarna menggunakan kode ANSI escape untuk keterbacaan yang optimal.
*   **Comprehensive HTML Dashboard:** Laporan visual yang dihasilkan setelah setiap simulasi, menampilkan chart harga, RSI, MACD, tabel riwayat trade terperinci, dan statistik ringkasan.

## Project Structure

```
.
├── trading_sim.py          # Logika inti bot: MarketData, Indicators, AdvancedFuzzy, TradingEngine
├── simulation_report.html  # Output dashboard visual setelah simulasi
├── GEMINI.md               # Dokumentasi internal project untuk AI agent
└── README.md               # Dokumentasi proyek untuk GitHub
```

## Getting Started

### Prerequisites

*   **Python 3.x** terinstal di sistem Anda.
*   Koneksi internet yang aktif untuk mengakses Binance Futures API.

### Installation

Tidak ada instalasi khusus yang diperlukan selain Python standar. Semua dependensi (seperti `urllib`, `json`, `math`, `datetime`) adalah modul bawaan Python.

### Running the Bot

Bot ini dijalankan melalui antarmuka Command Line (CLI) interaktif.

1.  **Navigasi** ke direktori proyek di terminal Anda:
    ```bash
    cd /path/to/your/project
    ```
2.  **Jalankan** script Python:
    ```bash
    python3 trading_sim.py
    ```
3.  Ikuti **prompt interaktif** di terminal untuk memilih mode strategi dan pasangan mata uang kripto (symbol) yang ingin Anda simulasikan.

    ```
    === kidd142 Trading Bot Menu ===
    Pilih Mode Strategi:
      1. SCALP (EMA21 15m - Agresif)
      2. INTRADAY (EMA21 1H - Harian)
      3. SWING (EMA21 4H - Mingguan)
      4. MTF (Multi-Timeframe 1H + 4H - Konfirmasi Ganda)
      0. Exit
    Pilih nomor mode (0-4): [masukkan pilihan Anda]
    Masukkan Symbol (contoh: BTCUSDT, ETHUSDT) [BTCUSDT]: [masukkan symbol atau tekan Enter untuk default]
    ```

### Output

Setelah simulasi selesai, Anda akan mendapatkan:

*   **Log Terminal:** Ringkasan hasil trade dan statistik akan ditampilkan langsung di terminal dengan pewarnaan yang informatif.
*   **HTML Dashboard:** File `simulation_report.html` akan dibuat/diperbarui di direktori proyek Anda. Buka file ini di browser web pilihan Anda untuk melihat dashboard visual interaktif yang mencakup grafik harga, indikator, dan riwayat trade.

## Contoh Penggunaan

Berikut adalah beberapa contoh bagaimana Anda dapat berinteraksi dengan bot melalui menu interaktif:

```bash
# Untuk memulai bot dan memilih mode/symbol secara interaktif
python3 trading_sim.py
```

## Contributing

Fitur-fitur masa depan dapat mencakup:
*   Integrasi dengan API trading (paper trading atau live trading).
*   Optimasi parameter strategi (backtesting mendalam).
*   Penambahan indikator dan strategi SMC yang lebih kompleks.
*   Antarmuka pengguna grafis (GUI).

Jika Anda memiliki ide atau ingin berkontribusi, silakan fork repositori ini dan kirimkan Pull Request Anda!

## License

Proyek ini dilisensikan di bawah [MIT License](LICENSE). <!-- Buat file LICENSE.md jika ini menjadi proyek open source -->
