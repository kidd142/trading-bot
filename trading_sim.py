import random
import math
import time
import json
import urllib.request
import urllib.error
from datetime import datetime, timedelta

# --- KONFIGURASI ---
INITIAL_BALANCE = 1000.0
DATA_LENGTH = 500 
EMA_PERIOD = 21

# --- ANSI COLORS ---
class C:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    RESET = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class MarketData:
    def __init__(self, length=100):
        self.length = length
        self.data = [] 

    def fetch_binance_data(self, symbol="BTCUSDT", interval="15m", limit=500):
        """
        Mengambil data Real-Time dari Binance Futures API (fapi).
        """
        base_url = "https://fapi.binance.com/fapi/v1/klines"
        url = f"{base_url}?symbol={symbol}&interval={interval}&limit={limit}"
        
        print(f"{C.CYAN}ℹ️  Fetching data for {C.BOLD}{symbol}{C.RESET}{C.CYAN} from Binance...{C.RESET}")
        
        try:
            req = urllib.request.Request(
                url, 
                data=None, 
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            
            with urllib.request.urlopen(req) as response:
                data = json.load(response)
            
            parsed_data = []
            for candle in data:
                timestamp = candle[0]
                dt_object = datetime.fromtimestamp(timestamp / 1000)
                
                c_open = float(candle[1])
                c_high = float(candle[2])
                c_low = float(candle[3])
                c_close = float(candle[4])
                c_vol = float(candle[5])
                
                parsed_data.append({
                    'time': dt_object.strftime("%d %H:%M"),
                    'timestamp': dt_object,
                    'open': c_open,
                    'high': c_high,
                    'low': c_low,
                    'close': c_close,
                    'volume': c_vol,
                    'oi': c_vol * random.uniform(40, 60), # Dummy OI proportional to volume
                    'funding': random.uniform(-0.0007, 0.0007) # Dummy funding rate
                })
            
            self.data = parsed_data
            return self.data

        except Exception as e:
            print(f"{C.RED}❌ ERROR fetching Binance data: {e}{C.RESET}")
            print(f"{C.YELLOW}⚠️  Falling back to Dummy Data...{C.RESET}")
            return self.generate_advanced_data()

    def generate_advanced_data(self, start_price=50000, volatility=0.015):
        price = start_price
        current_time = datetime.now() - timedelta(minutes=self.length * 15)
        trend_bias = 0
        open_interest = 1000000.0 
        funding_rate = 0.0001
        
        for i in range(self.length):
            if i % 50 == 0:
                trend_bias = random.uniform(-0.008, 0.008)

            change_percent = random.uniform(-volatility, volatility) + trend_bias
            close_price = price * (1 + change_percent)
            high_price = max(price, close_price) * (1 + random.uniform(0, 0.005))
            low_price = min(price, close_price) * (1 - random.uniform(0, 0.005))
            
            oi_change = random.uniform(-0.05, 0.05)
            open_interest *= (1 + oi_change)
            funding_rate = random.uniform(-0.0005, 0.0005) + (trend_bias * 0.1)

            candle = {
                'time': current_time.strftime("%d %H:%M"),
                'timestamp': current_time,
                'open': price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': random.uniform(100, 5000) * (1 + abs(change_percent)*10),
                'oi': open_interest,
                'funding': funding_rate
            }
            self.data.append(candle)
            price = close_price
            current_time += timedelta(minutes=15)
            
        return self.data

class Indicators:
    @staticmethod
    def calculate_ema(data, period=21, source='close'):
        ema_values = []
        multiplier = 2 / (period + 1)
        if len(data) < period: return [None] * len(data)
        
        initial_sum = sum(c[source] for c in data[:period])
        prev_ema = initial_sum / period
        
        for _ in range(period-1): ema_values.append(None)
        ema_values.append(prev_ema)
        
        for i in range(period, len(data)):
            val = data[i][source]
            ema = (val - prev_ema) * multiplier + prev_ema
            ema_values.append(ema)
            prev_ema = ema
        return ema_values

    @staticmethod
    def calculate_rsi(data, period=14):
        rsi_values = [None] * len(data)
        if len(data) < period + 1: return rsi_values
        
        gains, losses = [], []
        for i in range(1, period + 1):
            change = data[i]['close'] - data[i-1]['close']
            gains.append(max(0, change))
            losses.append(max(0, -change))
            
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        
        if avg_loss == 0: rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        rsi_values[period] = rsi
        
        for i in range(period + 1, len(data)):
            change = data[i]['close'] - data[i-1]['close']
            gain = max(0, change)
            loss = max(0, -change)
            
            avg_gain = ((avg_gain * (period - 1)) + gain) / period
            avg_loss = ((avg_loss * (period - 1)) + loss) / period
            
            if avg_loss == 0: rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            rsi_values[i] = rsi
            
        return rsi_values

    @staticmethod
    def calculate_atr(data, period=14):
        atr_values = [None] * len(data)
        if len(data) < period: return atr_values
        
        tr_list = []
        for i in range(1, len(data)):
            h, l, c_prev = data[i]['high'], data[i]['low'], data[i-1]['close']
            tr = max(h - l, abs(h - c_prev), abs(l - c_prev))
            tr_list.append(tr)
            
        first_atr = sum(tr_list[:period]) / period
        atr_values[period] = first_atr
        
        prev_atr = first_atr
        for i in range(period + 1, len(data)):
            current_tr = tr_list[i-1]
            atr = ((prev_atr * (period - 1)) + current_tr) / period
            atr_values[i] = atr
            prev_atr = atr
            
        return atr_values

    @staticmethod
    def calculate_bollinger(data, period=20, mult=2):
        bands = []
        for i in range(len(data)):
            if i < period - 1:
                bands.append({'upper': None, 'mid': None, 'lower': None, 'bandwidth': None})
                continue
            slice_data = [c['close'] for c in data[i-period+1 : i+1]]
            sma = sum(slice_data) / period
            variance = sum((x - sma) ** 2 for x in slice_data) / period
            std_dev = math.sqrt(variance)
            
            upper = sma + (mult * std_dev)
            lower = sma - (mult * std_dev)
            bandwidth = (upper - lower) / sma if sma else 0
            
            bands.append({'upper': upper, 'mid': sma, 'lower': lower, 'bandwidth': bandwidth})
        return bands

    @staticmethod
    def calculate_adx(data, period=14):
        if len(data) < period * 2: return [None] * len(data)
        tr_list = []
        dm_plus = []
        dm_minus = []
        for i in range(1, len(data)):
            h, l, c_prev = data[i]['high'], data[i]['low'], data[i-1]['close']
            tr = max(h - l, abs(h - c_prev), abs(l - c_prev))
            tr_list.append(tr)
            up_move = h - data[i-1]['high']
            down_move = data[i-1]['low'] - l
            if up_move > down_move and up_move > 0: dm_plus.append(up_move)
            else: dm_plus.append(0)
            if down_move > up_move and down_move > 0: dm_minus.append(down_move)
            else: dm_minus.append(0)
        
        atr_smooth = [sum(tr_list[:period]) / period]
        dm_plus_smooth = [sum(dm_plus[:period]) / period]
        dm_minus_smooth = [sum(dm_minus[:period]) / period]
        
        for i in range(period, len(tr_list)):
            atr_smooth.append((atr_smooth[-1] * (period - 1) + tr_list[i]) / period)
            dm_plus_smooth.append((dm_plus_smooth[-1] * (period - 1) + dm_plus[i]) / period)
            dm_minus_smooth.append((dm_minus_smooth[-1] * (period - 1) + dm_minus[i]) / period)
            
        adx_values = [None] * (period) 
        dx_list = []
        for i in range(len(atr_smooth)):
            di_plus = (dm_plus_smooth[i] / atr_smooth[i]) * 100 if atr_smooth[i] else 0
            di_minus = (dm_minus_smooth[i] / atr_smooth[i]) * 100 if atr_smooth[i] else 0
            dx_num = abs(di_plus - di_minus)
            dx_den = di_plus + di_minus
            dx = (dx_num / dx_den) * 100 if dx_den else 0
            dx_list.append(dx)
            
        if len(dx_list) < period: return [None] * len(data)
        first_adx = sum(dx_list[:period]) / period
        adx_values.extend([None] * (period - 1)) 
        adx_values.append(first_adx)
        prev_adx = first_adx
        for i in range(period, len(dx_list)):
            curr_adx = (prev_adx * (period - 1) + dx_list[i]) / period
            adx_values.append(curr_adx)
            prev_adx = curr_adx
        while len(adx_values) < len(data): adx_values.insert(0, None)
        return adx_values

    @staticmethod
    def calculate_macd(data, fast=12, slow=26, signal=9):
        ema_fast = Indicators.calculate_ema(data, fast)
        ema_slow = Indicators.calculate_ema(data, slow)
        macd_line = []
        for i in range(len(data)):
            if ema_fast[i] is None or ema_slow[i] is None: macd_line.append(None)
            else: macd_line.append(ema_fast[i] - ema_slow[i])
        valid_macd = [m for m in macd_line if m is not None]
        none_count = len(macd_line) - len(valid_macd)
        signal_line_vals = []
        if len(valid_macd) >= signal:
            sma_sig = sum(valid_macd[:signal]) / signal
            signal_line_vals = [None] * (signal - 1) + [sma_sig]
            prev_sig = sma_sig
            k = 2 / (signal + 1)
            for val in valid_macd[signal:]:
                sig = (val - prev_sig) * k + prev_sig
                signal_line_vals.append(sig)
                prev_sig = sig
        final_signal = [None] * none_count + signal_line_vals
        while len(final_signal) < len(data): final_signal.insert(0, None)
        hist = []
        for m, s in zip(macd_line, final_signal):
            if m is not None and s is not None: hist.append(m - s)
            else: hist.append(None)
        return macd_line, final_signal, hist

    @staticmethod
    def resample_data(data, timeframe_factor=4):
        resampled = []
        for i in range(0, len(data), timeframe_factor):
            chunk = data[i : i + timeframe_factor]
            if not chunk: break
            new_candle = {
                'time': chunk[0]['time'],
                'timestamp': chunk[0]['timestamp'],
                'open': chunk[0]['open'],
                'high': max(c['high'] for c in chunk),
                'low': min(c['low'] for c in chunk),
                'close': chunk[-1]['close'],
                'volume': sum(c['volume'] for c in chunk),
                'oi': chunk[-1]['oi'],
                'funding': sum(c['funding'] for c in chunk) / len(chunk)
            }
            resampled.append(new_candle)
        return resampled

    @staticmethod
    def find_swing_points_advanced(data, lookback=5):
        swings = [None] * len(data)
        for i in range(lookback, len(data) - lookback):
            current_high = data[i]['high']
            current_low = data[i]['low']
            is_high = all(data[i-j]['high'] < current_high for j in range(1, lookback+1)) and \
                      all(data[i+j]['high'] < current_high for j in range(1, lookback+1))
            is_low = all(data[i-j]['low'] > current_low for j in range(1, lookback+1)) and \
                     all(data[i+j]['low'] > current_low for j in range(1, lookback+1))
            if is_high: swings[i] = {'type': 'High', 'price': current_high, 'index': i}
            elif is_low: swings[i] = {'type': 'Low', 'price': current_low, 'index': i}
        return swings

    @staticmethod
    def detect_bos(data, swings, current_index, min_candles_since_swing=2, price_buffer_pct=0.05):
        """
        Mendeteksi Break of Structure (BoS) Bullish atau Bearish terbaru.
        Mengembalikan BoS type, dan swing yang di break.
        """
        if current_index < min_candles_since_swing: return None, None
        
        relevant_swings = []
        for i in range(current_index - min_candles_since_swing - 1, -1, -1):
            if swings[i]:
                relevant_swings.append(swings[i])
            if len(relevant_swings) > 5: break

        for s in relevant_swings:
            if s['type'] == 'High' and data[current_index]['close'] > s['price'] * (1 + price_buffer_pct / 100):
                return 'BULLISH_BOS', s
            elif s['type'] == 'Low' and data[current_index]['close'] < s['price'] * (1 - price_buffer_pct / 100):
                return 'BEARISH_BOS', s
        return None, None
    
    @staticmethod
    def detect_order_block(data, candle_index, direction, lookback=5, min_impulse_pct=0.5):
        """
        Mendeteksi Order Block sederhana: last opposite candle sebelum impulsif move.
        direction: 'BULLISH' (candle merah sebelum up move) atau 'BEARISH' (candle hijau sebelum down move)
        """
        if candle_index < lookback + 1: return None

        curr = data[candle_index]
        prev = data[candle_index - 1]
        
        if direction == 'BULLISH' and (curr['close'] - curr['open']) < (curr['open'] * min_impulse_pct / 100):
            return None
        if direction == 'BEARISH' and (curr['open'] - curr['close']) < (curr['open'] * min_impulse_pct / 100):
            return None

        if direction == 'BULLISH':
            if prev['close'] < prev['open'] and curr['close'] > curr['open'] and curr['low'] > prev['low']:
                return {'type': 'BULLISH_OB', 'low': prev['low'], 'high': prev['high'], 'index': candle_index - 1}
        elif direction == 'BEARISH':
            if prev['close'] > prev['open'] and curr['close'] < curr['open'] and curr['high'] < prev['high']:
                return {'type': 'BEARISH_OB', 'low': prev['low'], 'high': prev['high'], 'index': candle_index - 1}
        return None
    
    @staticmethod
    def calculate_delta_metrics(data, current_index):
        """
        Menghitung delta volume, oi, dan funding rate.
        Mengembalikan dictionary dari delta, None jika tidak ada prev data.
        """
        if current_index < 1:
            return {'volume_delta_pct': 0, 'oi_delta_pct': 0, 'funding_delta_abs': 0}
            
        prev_candle = data[current_index - 1]
        curr_candle = data[current_index]
        
        volume_delta_pct = (curr_candle['volume'] - prev_candle['volume']) / prev_candle['volume'] if prev_candle['volume'] else 0
        oi_delta_pct = (curr_candle['oi'] - prev_candle['oi']) / prev_candle['oi'] if prev_candle['oi'] else 0
        funding_delta_abs = curr_candle['funding'] - prev_candle['funding']
        
        return {
            'volume_delta_pct': volume_delta_pct,
            'oi_delta_pct': oi_delta_pct,
            'funding_delta_abs': funding_delta_abs
        }

class AdvancedFuzzy:
    def __init__(self):
        pass
        
    def fuzzify(self, dist_pct, rsi_val, macd_hist, smc_confluence=0, volume_confirm=0, oi_confirm=0):
        dist = abs(dist_pct)
        d_close = max(min((dist - (-0.1)) / (0 - (-0.1)), 1, (3.0 - dist) / (3.0 - 1.5)), 0)
        rsi_low = max(min((rsi_val - 0) / (20 - 0), 1, (45 - rsi_val) / (45 - 30)), 0)
        rsi_high = max(min((rsi_val - 55) / (70 - 55), 1, (100 - rsi_val) / (100 - 80)), 0)
        macd_pos = 1.0 if macd_hist > 0 else 0.0
        macd_neg = 1.0 if macd_hist < 0 else 0.0
        
        smc_strong = 1.0 if smc_confluence > 0.5 else 0.0 
        volume_high = 1.0 if volume_confirm > 0.05 else 0.0 
        oi_positive = 1.0 if oi_confirm > 0.01 else 0.0 

        return {
            'dist': {'close': d_close},
            'rsi': {'low': rsi_low, 'high': rsi_high},
            'macd': {'pos': macd_pos, 'neg': macd_neg},
            'smc': {'strong': smc_strong},
            'volume': {'high': volume_high},
            'oi': {'positive': oi_positive}
        }

    def infer(self, f, trend_dir):
        score = 50
        
        if f['smc']['strong'] > 0: score += f['smc']['strong'] * 15 
        if f['volume']['high'] > 0: score += f['volume']['high'] * 5 
        
        if trend_dir == 1: # UPTREND
            s1 = f['dist']['close']
            if s1 > 0: score += s1 * 20 
            if f['rsi']['low'] > 0: score += 20
            if f['macd']['pos'] > 0: score += 10
            if f['oi']['positive'] > 0: score += 5 
            
        elif trend_dir == -1: # DOWNTREND
            s1 = f['dist']['close']
            if s1 > 0: score += s1 * 20
            if f['rsi']['high'] > 0: score += 20
            if f['macd']['neg'] > 0: score += 10
            if f['oi']['positive'] < 0: score += 5 
            
        return min(max(score, 0), 100) 

class TradingEngine:
    def __init__(self, mode='SCALP', symbol='BTCUSDT'):
        self.balance = INITIAL_BALANCE
        self.position = None 
        self.entry_price = 0
        self.entry_candle_index = 0
        self.sl_price = 0
        self.tp_price = 0
        self.trades = []
        self.fuzzy = AdvancedFuzzy()
        self.mode = mode
        self.symbol = symbol
        
    def run(self):
        print(f"\n{C.HEADER}{C.BOLD}🚀 kidd142 Trading Bot | {self.symbol} | {self.mode}{C.RESET}")
        print(f"{C.HEADER}=============================================={C.RESET}")
        
        market = MarketData(length=DATA_LENGTH)
        interval_map = {'SCALP': '15m', 'INTRADAY': '1h', 'SWING': '4h', 'MTF': '1h'}
        binance_interval = interval_map.get(self.mode, '15m')
        
        base_data = market.fetch_binance_data(symbol=self.symbol, interval=binance_interval, limit=DATA_LENGTH)
        
        if not base_data:
            print(f"{C.RED}❌ Gagal mengambil data. Keluar.{C.RESET}")
            return [], [], [], [], [], [], [], self.symbol
            
        latest_price = base_data[-1]['close']
        latest_time = base_data[-1]['time']
        print(f"{C.BLUE}📊 Latest Price: {C.BOLD}{latest_price}{C.RESET} {C.BLUE}({latest_time}){C.RESET}")
        
        if self.mode == 'SCALP':
            trade_data = base_data
            ema_p = 21
        elif self.mode == 'INTRADAY':
            trade_data = base_data
            ema_p = 21
        elif self.mode == 'SWING':
            trade_data = base_data
            ema_p = 21
        elif self.mode == 'MTF':
            trade_data = base_data 
            trend_data = Indicators.resample_data(base_data, 4)
            ema_p = 21
            ema_4h = Indicators.calculate_ema(trend_data, 21)
        
        ema = Indicators.calculate_ema(trade_data, ema_p)
        rsi = Indicators.calculate_rsi(trade_data)
        macd, macd_sig, macd_hist = Indicators.calculate_macd(trade_data)
        atr = Indicators.calculate_atr(trade_data)
        bands = Indicators.calculate_bollinger(trade_data)
        adx = Indicators.calculate_adx(trade_data)
        swings = Indicators.find_swing_points_advanced(trade_data)
        
        print(f"{C.BLUE}🧠 Processing {len(trade_data)} candles...{C.RESET}\n")
        
        for i in range(50, len(trade_data) - 1): # Start later for ADX, SMC (need more history)
            c = trade_data[i]
            prev_c = trade_data[i-1] if i > 0 else c
            curr_ema = ema[i]
            curr_atr = atr[i]
            curr_adx = adx[i]
            curr_band = bands[i]
            
            if curr_ema is None or curr_atr is None or rsi[i] is None or macd_hist[i] is None or curr_adx is None or curr_band['bandwidth'] is None: continue
            
            # --- FILTERS ---
            if curr_adx < 20: continue # ADX Filter
            if curr_band['bandwidth'] < 0.005: continue # Squeeze Filter
            
            # --- SMC & DELTA METRICS ---
            bos_status, broken_swing = Indicators.detect_bos(trade_data, swings, i)
            delta_metrics = Indicators.calculate_delta_metrics(trade_data, i)
            
            smc_confluence = 0
            # BoS Confirmation
            if bos_status == 'BULLISH_BOS':
                smc_confluence = 0.5
                # Retest OB logic (simplified: if price currently near a recently formed OB)
                bullish_ob = Indicators.detect_order_block(trade_data, i, 'BULLISH')
                if bullish_ob and c['low'] >= bullish_ob['low'] and c['low'] <= bullish_ob['high']: 
                    smc_confluence = 0.8 # Higher confluence for retest
            elif bos_status == 'BEARISH_BOS':
                smc_confluence = 0.5
                bearish_ob = Indicators.detect_order_block(trade_data, i, 'BEARISH')
                if bearish_ob and c['high'] <= bearish_ob['high'] and c['high'] >= bearish_ob['low']:
                    smc_confluence = 0.8 # Higher confluence for retest
            
            trend_dir = 1 if c['close'] > curr_ema else -1
            
            if self.mode == 'MTF':
                idx_4h = min(i // 4, len(ema_4h)-1)
                if ema_4h[idx_4h] is None: continue
                trend_4h = 1 if trend_data[idx_4h]['close'] > ema_4h[idx_4h] else -1
                if trend_dir != trend_4h: continue
            
            dist_pct = ((c['close'] - curr_ema) / curr_ema) * 100
            
            # Pass new SMC and Delta metrics to Fuzzy
            f_in = self.fuzzy.fuzzify(
                dist_pct=dist_pct, 
                rsi_val=rsi[i], 
                macd_hist=macd_hist[i], 
                smc_confluence=smc_confluence,
                volume_confirm=delta_metrics['volume_delta_pct'],
                oi_confirm=delta_metrics['oi_delta_pct']
            )
            confidence = self.fuzzy.infer(f_in, trend_dir)
            
            if curr_adx > 35: confidence += 10 # Boost for very strong trend
            
            atr_mult_sl = 1.5 if self.mode == 'SCALP' else 2.0
            atr_mult_tp = 2.5 if self.mode == 'SCALP' else 4.0
            
            # ENTRY
            if self.position is None and confidence > 75: # Increased confidence threshold for entry
                if trend_dir == 1: # BUY
                    self.position = 'LONG'
                    self.entry_price = c['close']
                    self.entry_candle_index = i
                    self.sl_price = c['close'] - (curr_atr * atr_mult_sl)
                    self.tp_price = c['close'] + (curr_atr * atr_mult_tp * 1.5)
                    print(f"{c['time']} | {C.GREEN}{C.BOLD}OPEN LONG 🟢{C.RESET} | Price: {C.YELLOW}{c['close']:.2f}{C.RESET} | Conf: {C.CYAN}{confidence:.0f}%{C.RESET} | ADX: {curr_adx:.1f}")
                    
                elif trend_dir == -1: # SELL
                    self.position = 'SHORT'
                    self.entry_price = c['close']
                    self.entry_candle_index = i
                    self.sl_price = c['close'] + (curr_atr * atr_mult_sl)
                    self.tp_price = c['close'] - (curr_atr * atr_mult_tp * 1.5)
                    print(f"{c['time']} | {C.RED}{C.BOLD}OPEN SHRT 🔴{C.RESET} | Price: {C.YELLOW}{c['close']:.2f}{C.RESET} | Conf: {C.CYAN}{confidence:.0f}%{C.RESET} | ADX: {curr_adx:.1f}")

            # EXIT
            elif self.position == 'LONG':
                result = None
                # 1. Hard SL/TP
                if c['low'] <= self.sl_price: result = 'SL'
                elif c['high'] >= self.tp_price: result = 'TP'
                # 2. Adaptive RSI Exit (Early TP)
                elif rsi[i] > 70: result = 'RSI_TP'
                # 3. Time-Based Exit (Stale)
                elif (i - self.entry_candle_index) > 12: result = 'TIME_EXIT'
                
                if result:
                    exit_p = self.sl_price if result == 'SL' else c['close']
                    if result == 'TP' or result == 'RSI_TP': exit_p = c['close'] 
                    
                    pnl = (exit_p - self.entry_price) / self.entry_price * 100
                    self.balance *= (1 + pnl/100)
                    
                    self.trades.append({'time': c['time'], 'type': 'LONG', 'entry': self.entry_price, 'exit': exit_p, 'result': result, 'pnl': pnl, 'bal': self.balance})
                    self.position = None
                    
                    pnl_color = C.GREEN if pnl > 0 else C.RED
                    print(f"        └─ {C.BOLD}EXIT {result:<9}{C.RESET} | PnL: {pnl_color}{pnl:+.2f}%{C.RESET} | Bal: {self.balance:.2f}")
                    
            elif self.position == 'SHORT':
                result = None
                if c['high'] >= self.sl_price: result = 'SL'
                elif c['low'] <= self.tp_price: result = 'TP'
                elif rsi[i] < 30: result = 'RSI_TP'
                elif (i - self.entry_candle_index) > 12: result = 'TIME_EXIT'
                
                if result:
                    exit_p = self.sl_price if result == 'SL' else c['close']
                    if result == 'TP' or result == 'RSI_TP': exit_p = c['close']
                    
                    pnl = (self.entry_price - exit_p) / self.entry_price * 100
                    self.balance *= (1 + pnl/100)
                    
                    self.trades.append({'time': c['time'], 'type': 'SHORT', 'entry': self.entry_price, 'exit': exit_p, 'result': result, 'pnl': pnl, 'bal': self.balance})
                    self.position = None

                    pnl_color = C.GREEN if pnl > 0 else C.RED
                    print(f"        └─ {C.BOLD}EXIT {result:<9}{C.RESET} | PnL: {pnl_color}{pnl:+.2f}%{C.RESET} | Bal: {self.balance:.2f}")

        print(f"\n{C.HEADER}=============================================={C.RESET}")
        print(f"{C.BOLD}🏁 Simulation Finished{C.RESET}")
        
        bal_color = C.GREEN if self.balance >= INITIAL_BALANCE else C.RED
        win_rate = len([t for t in self.trades if t['pnl'] > 0]) / len(self.trades) * 100 if self.trades else 0
        
        print(f"💰 Final Balance: {bal_color}${self.balance:.2f}{C.RESET} (Start: ${INITIAL_BALANCE})")
        print(f"📊 Total Trades:  {len(self.trades)}")
        print(f"🏆 Win Rate:      {C.YELLOW}{win_rate:.1f}%{C.RESET}")
        print(f"{C.HEADER}=============================================={C.RESET}")
        
        return trade_data, ema, rsi, macd, macd_sig, macd_hist, self.trades


def generate_html_report(candles, ema, rsi, macd, macd_sig, macd_hist, trades, symbol="BTCUSDT"):
    labels = [c['time'] for c in candles]
    closes = [c['close'] for c in candles]
    def clean(data_list): return [x if x is not None else 'null' for x in data_list]

    table_rows = ""
    win_count = 0
    for t in trades:
        color = "#4caf50" if t['pnl'] > 0 else "#f44336"
        if t['pnl'] > 0: win_count += 1
        row = f"""
        <tr style="border-bottom: 1px solid #333;">
            <td>{t['time']}</td>
            <td style="color:{'#00e676' if t['type']=='LONG' else '#ff5252'}">{t['type']}</td>
            <td>{t['entry']:.2f}</td>
            <td>{t['exit']:.2f}</td>
            <td style="font-weight:bold; color:{color}">{t['result']}</td>
            <td style="color:{color}">{t['pnl']:.2f}%</td>
            <td>{t['bal']:.2f}</td>
        </tr>
        """
        table_rows += row
        
    win_rate = (win_count / len(trades) * 100) if trades else 0
    final_bal = trades[-1]['bal'] if trades else INITIAL_BALANCE
    pnl_total_pct = ((final_bal - INITIAL_BALANCE) / INITIAL_BALANCE) * 100
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{symbol} Bot Dashboard</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation@3.0.1/dist/chartjs-plugin-annotation.min.js"></script>
        <style>
            body {{ font-family: 'Segoe UI', sans-serif; background: #121212; color: #e0e0e0; margin: 0; padding: 20px; }}
            .header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }}
            .card {{ background: #1e1e1e; padding: 15px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }}
            h2, h3 {{ margin-top: 0; color: #90caf9; }}
            .stats-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; }}
            .stat-box {{ background: #2c2c2c; padding: 15px; border-radius: 5px; text-align: center; }}
            .stat-val {{ font-size: 24px; font-weight: bold; color: #fff; }}
            .stat-label {{ font-size: 12px; color: #aaa; text-transform: uppercase; }}
            table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
            th {{ text-align: left; color: #888; padding: 10px; border-bottom: 2px solid #444; }}
            td {{ padding: 10px; }}
            canvas {{ max-height: 300px; }}
            .chart-container {{ position: relative; height: 300px; width: 100%; }}
            .chart-small {{ height: 150px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🤖 {symbol} kidd142 Trading Bot Dashboard</h1>
            <div>{datetime.now().strftime("%Y-%m-%d %H:%M")}</div>
        </div>

        <div class="stats-grid card">
            <div class="stat-box">
                <div class="stat-val" style="color:{'#4caf50' if pnl_total_pct>=0 else '#f44336'}">${final_bal:.2f}</div>
                <div class="stat-label">Final Balance</div>
            </div>
            <div class="stat-box">
                <div class="stat-val" style="color:{'#4caf50' if pnl_total_pct>=0 else '#f44336'}">{pnl_total_pct:.2f}%</div>
                <div class="stat-label">Total Return</div>
            </div>
            <div class="stat-box">
                <div class="stat-val">{len(trades)}</div>
                <div class="stat-label">Total Trades</div>
            </div>
            <div class="stat-box">
                <div class="stat-val">{win_rate:.1f}%</div>
                <div class="stat-label">Win Rate</div>
            </div>
        </div>

        <div class="card">
            <h3>Price Action & EMA 21</h3>
            <div class="chart-container">
                <canvas id="priceChart"></canvas>
            </div>
        </div>

        <div class="card">
            <h3>RSI (14)</h3>
            <div class="chart-container chart-small">
                <canvas id="rsiChart"></canvas>
            </div>
        </div>

        <div class="card">
            <h3>MACD Momentum</h3>
            <div class="chart-container chart-small">
                <canvas id="macdChart"></canvas>
            </div>
        </div>

        <div class="card">
            <h3>Trade History</h3>
            <div style="overflow-x:auto;">
                <table>
                    <thead>
                        <tr>
                            <th>Time</th><th>Type</th><th>Entry</th><th>Exit</th><th>Result</th><th>PnL %</th><th>Balance</th>
                        </tr>
                    </thead>
                    <tbody>
                        {table_rows}
                    </tbody>
                </table>
            </div>
        </div>

        <script>
            Chart.register(window['chartjs-plugin-annotation']);
            const labels = {json.dumps(labels)};
            
            new Chart(document.getElementById('priceChart'), {{
                type: 'line',
                data: {{
                    labels: labels,
                    datasets: [
                        {{
                            label: 'Price',
                            data: {clean(closes)},
                            borderColor: '#2196f3',
                            borderWidth: 1,
                            pointRadius: 0
                        }},
                        {{
                            label: 'EMA 21',
                            data: {clean(ema)},
                            borderColor: '#ff9800',
                            borderWidth: 2,
                            pointRadius: 0,
                            fill: false
                        }}
                    ]
                }},
                options: {{
                    maintainAspectRatio: false,
                    scales: {{ x: {{ display: false }}, y: {{ grid: {{ color: '#333' }} }} }},
                    interaction: {{ mode: 'index', intersect: false }}
                }}
            }});

            new Chart(document.getElementById('rsiChart'), {{
                type: 'line',
                data: {{
                    labels: labels,
                    datasets: [{{
                        label: 'RSI',
                        data: {clean(rsi)},
                        borderColor: '#e91e63',
                        borderWidth: 1,
                        pointRadius: 0
                    }}]
                }},
                options: {{
                    maintainAspectRatio: false,
                    scales: {{ x: {{ display: false }}, y: {{ min: 0, max: 100, grid: {{ color: '#333' }} }} }},
                    plugins: {{
                        annotation: {{
                            annotations: {{
                                line1: {{ type: 'line', yMin: 70, yMax: 70, borderColor: '#666', borderWidth: 1, borderDash: [5,5] }},
                                line2: {{ type: 'line', yMin: 30, yMax: 30, borderColor: '#666', borderWidth: 1, borderDash: [5,5] }}
                            }}
                        }}
                    }}
                }}
            }});

            new Chart(document.getElementById('macdChart'), {{
                type: 'bar',
                data: {{
                    labels: labels,
                    datasets: [
                        {{ type: 'line', label: 'MACD', data: {clean(macd)}, borderColor: '#00bcd4', borderWidth: 1, pointRadius: 0 }},
                        {{ type: 'line', label: 'Signal', data: {clean(macd_sig)}, borderColor: '#ff5722', borderWidth: 1, pointRadius: 0 }},
                        {{ type: 'bar', label: 'Hist', data: {clean(macd_hist)}, backgroundColor: (ctx) => ctx.raw >= 0 ? '#4caf50' : '#f44336' }}
                    ]
                }},
                options: {{ maintainAspectRatio: false, scales: {{ x: {{ display: false }}, y: {{ grid: {{ color: '#333' }} }} }} }}
            }});
        </script>
    </body>
    </html>
    """
    
    with open("simulation_report.html", "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"{C.GREEN}✅ Visual report saved to: simulation_report.html{C.RESET}")

if __name__ == "__main__":
    
    def display_menu():
        print(f"\n{C.HEADER}{C.BOLD}=== kidd142 Trading Bot Menu ==={C.RESET}")
        print(f"{C.CYAN}Pilih Mode Strategi:{C.RESET}")
        print(f"  {C.YELLOW}1. SCALP{C.RESET} (EMA21 15m - Agresif)")
        print(f"  {C.YELLOW}2. INTRADAY{C.RESET} (EMA21 1H - Harian)")
        print(f"  {C.YELLOW}3. SWING{C.RESET} (EMA21 4H - Mingguan)")
        print(f"  {C.YELLOW}4. MTF{C.RESET} (Multi-Timeframe 1H + 4H - Konfirmasi Ganda)")
        print(f"  {C.RED}0. Exit{C.RESET}")
        
        mode_choice = input(f"{C.BOLD}Pilih nomor mode (0-4): {C.RESET}").strip()
        modes_map = {'1': 'SCALP', '2': 'INTRADAY', '3': 'SWING', '4': 'MTF'}
        
        if mode_choice == '0':
            print(f"{C.BLUE}Keluar dari bot. Sampai jumpa!{C.RESET}")
            return None, None
        
        mode = modes_map.get(mode_choice)
        if not mode:
            print(f"{C.RED}Pilihan tidak valid. Silakan coba lagi.{C.RESET}")
            return display_menu()
            
        symbol = input(f"{C.BOLD}Masukkan Symbol (contoh: BTCUSDT, ETHUSDT) [BTCUSDT]: {C.RESET}").strip().upper()
        if not symbol:
            symbol = "BTCUSDT"
            
        return mode, symbol

    while True:
        mode, symbol = display_menu()
        if mode is None:
            break
        
        engine = TradingEngine(mode=mode, symbol=symbol)
        data = engine.run()
        if data[0]: 
            generate_html_report(*data, symbol=symbol)
        
        input(f"\n{C.CYAN}Tekan ENTER untuk melanjutkan...{C.RESET}")