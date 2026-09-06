"""
Simple CMF / Volume Debug Script
---------------------------------
Fetches recent OHLCV data for the algo's tickers and prints raw volume stats
and the Chaikin Money Flow (CMF) calculation to the terminal, day by day and
as a final summary - useful for sanity-checking the CMF logic against what
you'd expect, without needing to run the full dashboard.

Run with:
    python check_cmf.py
    python check_cmf.py VTI GLD BTC-USD     (check specific tickers only)

IMPORTANT: this was written without live network access (the sandbox it was
authored in can't reach Yahoo Finance) - the CMF math itself was verified
against synthetic data, but the actual yfinance download is untested until
you run it locally.
"""

import sys

import numpy as np
import pandas as pd
import yfinance as yf

DIRECTIONAL_VOLUME_LOOKBACK = 63  # matches the algo's current setting

DEFAULT_TICKERS = [
    "VTI", "VEA", "VWO", "IXN", "GLD", "PDBC",   # equities
    "BND", "BNDX", "EMB", "SHV",                          # bonds
    "BTC-USD", "ETH-USD",                          # crypto (yfinance naming)
]


def compute_cmf(close, high, low, volume, lookback):
    """
    Returns (daily_detail_df, cmf) where daily_detail_df has one row per day
    in the lookback window showing high/low/close/volume/money-flow-multiplier/
    money-flow-volume, and cmf is the final Chaikin Money Flow value.
    """
    recent_close = close.iloc[-lookback:]
    recent_high = high.iloc[-lookback:]
    recent_low = low.iloc[-lookback:]
    recent_volume = volume.iloc[-lookback:]

    high_low_range = recent_high - recent_low
    money_flow_multiplier = np.where(
        high_low_range.values > 0,
        ((recent_close.values - recent_low.values) - (recent_high.values - recent_close.values))
        / np.where(high_low_range.values > 0, high_low_range.values, np.nan),
        0.0,
    )
    money_flow_multiplier = pd.Series(money_flow_multiplier, index=recent_close.index).fillna(0.0)
    money_flow_volume = money_flow_multiplier * recent_volume.values

    detail = pd.DataFrame({
        "High": recent_high,
        "Low": recent_low,
        "Close": recent_close,
        "Volume": recent_volume,
        "MoneyFlowMultiplier": money_flow_multiplier,
        "MoneyFlowVolume": money_flow_volume,
    })

    total_volume = recent_volume.sum()
    cmf = money_flow_volume.sum() / total_volume if total_volume > 0 else float("nan")

    return detail, cmf


def main():
    tickers = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_TICKERS

    print(f"Fetching {len(tickers)} ticker(s): {', '.join(tickers)}")
    print(f"CMF lookback: {DIRECTIONAL_VOLUME_LOOKBACK} trading days\n")

    period_days = DIRECTIONAL_VOLUME_LOOKBACK + 30
    data = yf.download(tickers, period=f"{period_days}d", progress=False, auto_adjust=True)

    if data.empty:
        print("No data returned - check tickers and network access.")
        return

    for ticker in tickers:
        print("=" * 70)
        print(f"{ticker}")
        print("=" * 70)

        try:
            if len(tickers) == 1:
                close, high, low, volume = data["Close"], data["High"], data["Low"], data["Volume"]
            else:
                close, high, low, volume = data["Close"][ticker], data["High"][ticker], data["Low"][ticker], data["Volume"][ticker]
        except KeyError:
            print(f"  No data for {ticker} - skipping.\n")
            continue

        close, high, low, volume = close.dropna(), high.dropna(), low.dropna(), volume.dropna()

        if len(close) < DIRECTIONAL_VOLUME_LOOKBACK:
            print(f"  Only {len(close)} bars available, need {DIRECTIONAL_VOLUME_LOOKBACK} - skipping.\n")
            continue

        detail, cmf = compute_cmf(close, high, low, volume, DIRECTIONAL_VOLUME_LOOKBACK)

        # Print the last 10 days of detail so you can see the raw numbers behind the CMF
        print(f"  Last 10 days of {DIRECTIONAL_VOLUME_LOOKBACK}-day CMF window:")
        print(detail.head(10).round(4).to_string())
        print(detail.tail(10).round(4).to_string())

        up_days = (detail["Close"].diff() > 0).sum()
        down_days = (detail["Close"].diff() < 0).sum()
        total_volume = detail["Volume"].sum()
        avg_daily_volume = detail["Volume"].mean()

        print()
        print(f"  Current price:              {close.iloc[-1]:,.2f}")
        print(f"  Up days / Down days:        {up_days} / {down_days}")
        print(f"  Total volume ({DIRECTIONAL_VOLUME_LOOKBACK}d):        {total_volume:,.0f}")
        print(f"  Avg daily volume:           {avg_daily_volume:,.0f}")
        print(f"  Chaikin Money Flow (CMF):   {cmf:.4f}")
        print(f"  Positive CMF (qualifies):   {cmf > 0}")
        print()


if __name__ == "__main__":
    main()