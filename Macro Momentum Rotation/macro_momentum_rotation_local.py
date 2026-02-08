import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ============================
# Config & Parameters
# ============================
ASSETS = ["VT", "VGLT", "VGIT", "GLD", "DBC", "BIL", "HYG", "VCIT", "VCLT", "VCSH", "VGSH"]
CRYPTO = ["IBIT"]
ALL_SYMBOLS = ASSETS + CRYPTO

MOMENTUM_LOOKBACKS = [21, 63, 126, 189, 252]
SMA_PERIOD = 168
VOL_LOOKBACK = 252
TARGET_VOL = 0.12
MAX_WEIGHT = 1.0
CASH_PROXY = "BIL"

def get_data(symbols):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=500)  # Enough for lookbacks
    data = yf.download(symbols, auto_adjust=True, start=start_date, end=end_date)['Close']
    return data

def calculate_momentum(series):
    returns = []
    for lb in MOMENTUM_LOOKBACKS:
        if len(series) > lb:
            ret = (series.iloc[-1] / series.iloc[-(lb + 1)]) - 1
            returns.append(ret)
    return np.mean(returns) if returns else -np.inf

def get_realized_vol(series):
    if len(series) < VOL_LOOKBACK + 1:
        return np.nan
    returns = np.log(series / series.shift(1)).dropna()
    vol = returns.tail(VOL_LOOKBACK).std() * np.sqrt(252)
    return vol

def run_strategy():
    print(f"Fetching data for {len(ALL_SYMBOLS)} assets...")
    data = get_data(ALL_SYMBOLS)
    
    # Eligibility checks
    eligible = []
    bil_series = data[CASH_PROXY].dropna()
    # Absolute 6M return for BIL
    bil_6m_ret = (bil_series.iloc[-1] / bil_series.iloc[-127]) - 1 if len(bil_series) > 127 else -np.inf

    print("\n--- Strategy Screening ---")
    for symbol in ALL_SYMBOLS:
        px = data[symbol].dropna()
        if len(px) < max(MOMENTUM_LOOKBACKS + [SMA_PERIOD, VOL_LOOKBACK]):
            continue
        
        # 1. SMA Gate
        sma = px.rolling(SMA_PERIOD).mean().iloc[-1]
        passes_sma = (px.iloc[-1] > sma) or (symbol == CASH_PROXY)
        
        # 2. Absolute Return Filter (Beat BIL 6M return)
        abs_6m_ret = (px.iloc[-1] / px.iloc[-127]) - 1 if len(px) > 127 else -np.inf
        beats_cash = (abs_6m_ret > bil_6m_ret) or (symbol == CASH_PROXY)
        
        if passes_sma and beats_cash:
            eligible.append(symbol)
            print(f"[ELIGIBLE] {symbol} | 6M Ret: {abs_6m_ret:.2%}")
        else:
            reason = "SMA" if not passes_sma else "Abs Return"
            print(f"[EXCLUDED] {symbol} | Failed {reason}")

    if not eligible:
        print("\nNo assets eligible. Moving to 100% Cash (BIL).")
        return {CASH_PROXY: 1.0}

    # 3. Momentum Ranking
    scores = {s: calculate_momentum(data[s].dropna()) for s in eligible}
    winner = max(scores, key=scores.get)
    
    print(f"\nWinner Selected: {winner} (Score: {scores[winner]:.4f})")

    # 4. Vol Targeting
    if winner == CASH_PROXY:
        weight = 1.0
    else:
        vol = get_realized_vol(data[winner].dropna())
        weight = min(MAX_WEIGHT, TARGET_VOL / vol) if vol > 0 else 0
        print(f"Realized Vol: {vol:.2%} | Target Weight: {weight:.2%}")

    cash_weight = 1.0 - weight
    
    # Output Final Allocation
    print("\n--- Final Target Allocation ---")
    print(f"{winner}: {weight:.2%}")
    if cash_weight > 0:
        print(f"{CASH_PROXY} (Cash): {cash_weight:.2%}")

if __name__ == "__main__":
    run_strategy()
