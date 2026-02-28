import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ============================
# Config aligned with QC Algo
# ============================
ETFS = ["VTI", "VGLT", "VGIT", "GLD", "DBC", "SHV", "VEA", "VWO", "BND", "BNDX", "EMB"]
SECTORS = ["IXP", "RXI", "KXI", "IXC", "IXG", "IXJ", "EXI", "IXN", "MXI", "REET", "JXI"]
CRYPTO = ["IBIT"] 
ALL_SYMBOLS = ETFS + SECTORS + CRYPTO

TOP_N = 3  # <--- Updated: Pick top X assets
MOMENTUM_LOOKBACKS = [21, 63, 126, 189, 252]
SMA_PERIOD = 168
BOND_SMA_PERIOD = 126
CVAR_LOOKBACK = 756
TARGET_CVAR = 0.03
MAX_WEIGHT = 1.0
CASH_PROXY = "SHV"
BOND_ASSETS = ["VGIT", "BND", "BNDX"]

def get_data(symbols):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=1200) 
    data = yf.download(symbols, auto_adjust=True, start=start_date, end=end_date)['Close']
    return data

def calculate_momentum(series):
    returns = []
    for lb in MOMENTUM_LOOKBACKS:
        if len(series) > lb:
            ret = (series.iloc[-1] / series.iloc[-(lb + 1)]) - 1
            returns.append(ret)
    return np.mean(returns) if returns else -np.inf

def get_cvar(series, alpha=0.95):
    if len(series) < 252:
        return np.nan
    rets = series.pct_change().dropna().tail(CVAR_LOOKBACK)
    var_threshold = np.percentile(rets, (1 - alpha) * 100)
    tail_losses = rets[rets <= var_threshold]
    return abs(tail_losses.mean()) if not tail_losses.empty else np.nan

def run_strategy():
    print(f"Fetching data for {len(ALL_SYMBOLS)} assets...")
    data = get_data(ALL_SYMBOLS)
    
    # 1. Eligibility (Trend Filter)
    eligible = []
    print("\n--- Trend Filter Screening ---")
    for symbol in ALL_SYMBOLS:
        if symbol not in data.columns: continue
        px = data[symbol].dropna()
        period = BOND_SMA_PERIOD if symbol in BOND_ASSETS else SMA_PERIOD
        
        if len(px) < period: continue
            
        sma = px.rolling(period).mean().iloc[-1]
        if px.iloc[-1] > sma or symbol == CASH_PROXY:
            eligible.append(symbol)
            print(f"[PASSED] {symbol}")
        else:
            print(f"[FAILED] {symbol}")

    # 2. Breadth Filter (Threshold updated to 10)
    num_eligible = len(eligible)
    if num_eligible < 10:
        print(f"\nLOW BREADTH: Only {num_eligible} assets passed. Moving to {CASH_PROXY}.")
        final_allocations = {CASH_PROXY: 1.0}
    else:
        # 3. Momentum Ranking for Top N
        scores = {s: calculate_momentum(data[s].dropna()) for s in eligible}
        sorted_assets = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        winners = [x[0] for x in sorted_assets[:TOP_N]]

        print(f"\nTop {TOP_N} Selected: {winners} (Breadth: {num_eligible})")

        # 4. CVaR Targeting & Final Weighting
        final_allocations = {}
        total_allocated_weight = 0

        for winner in winners:
            if winner == CASH_PROXY:
                indiv_weight = 1.0 / TOP_N
            else:
                asset_cvar = get_cvar(data[winner].dropna())
                
                if np.isnan(asset_cvar) or asset_cvar == 0:
                    indiv_weight = 0.0
                else:
                    # Risk budget per slot (Target CVaR / N)
                    slot_target = TARGET_CVAR / TOP_N
                    raw_weight = min(MAX_WEIGHT / TOP_N, slot_target / asset_cvar)
                    
                    # Crypto Cap (15%)
                    if winner in CRYPTO:
                        indiv_weight = min(raw_weight, 0.15)
                    else:
                        indiv_weight = raw_weight
            
            final_allocations[winner] = indiv_weight
            total_allocated_weight += indiv_weight

        # 5. Fill remainder with Cash Proxy
        cash_fill = 1.0 - total_allocated_weight
        if cash_fill > 0.001:
            final_allocations[CASH_PROXY] = final_allocations.get(CASH_PROXY, 0) + cash_fill

    print("\n--- Final Target Allocation ---")
    for asset, weight in sorted(final_allocations.items(), key=lambda x: x[1], reverse=True):
        if weight > 0.001:
            print(f"{asset:6}: {weight:.2%}")

if __name__ == "__main__":
    run_strategy()
