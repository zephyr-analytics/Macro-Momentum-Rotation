"""signals.py — Absolute/Relative Momentum signal runner.

Run this script directly to print today's signal:

    python signals.py

Requires:
    pip install yfinance pandas numpy
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import date, timedelta

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
ETF_TICKERS:         list[str] = ["VTI", "VEU", "BND", "BNDX", "VGIT", "GLD", "DBC", "SGOV"]
BTC_TICKER:          str       = "BTC-USD"
CASH_TICKER:         str       = "SGOV"

MOMENTUM_LOOKBACKS:  list[int] = [21, 63, 126, 189, 252]
MAX_LOOKBACK:        int       = max(MOMENTUM_LOOKBACKS)

SMA_PERIOD:          int       = 168
SMA_OVERRIDES:       dict[str, int] = {
    "BND":  126,
    "BNDX": 126,
    "VGIT": 126,
}
VOL_LOOKBACK:        int       = 63
TARGET_VOL:          float     = 0.20
MAX_WEIGHT:          float     = 1.0
TOP_N:               int       = 2

# Extra buffer so rolling windows are fully populated on the first valid row
_N_BARS: int = MAX_LOOKBACK + SMA_PERIOD + 10


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

def fetch_etf_closes() -> pd.DataFrame:
    """Download adjusted close prices for all ETFs.

    Returns
    -------
    pd.DataFrame
        Wide-format adjusted closes; columns are ticker strings.
    """
    raw = yf.download(
        ETF_TICKERS,
        period=f"{_N_BARS}d",
        auto_adjust=True,
        progress=False,
    )
    closes = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    closes.columns = [c if isinstance(c, str) else c[0] for c in closes.columns]
    return closes.dropna(how="all")


def fetch_btc_closes() -> pd.DataFrame | None:
    """Download raw close prices for BTC-USD.

    Returns
    -------
    pd.DataFrame or None
        Single-column wide-format closes, or ``None`` on failure.
    """
    try:
        raw = yf.download(
            BTC_TICKER,
            period=f"{_N_BARS}d",
            auto_adjust=True,
            progress=False,
        )
        if raw.empty:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [col[0] for col in raw.columns]
        if "Close" not in raw.columns:
            return None
        closes = raw[["Close"]].rename(columns={"Close": BTC_TICKER})
        return closes.dropna(how="all")
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Signal helpers
# --------------------------------------------------------------------------- #

def momentum(ticker: str, closes: pd.DataFrame) -> float:
    """Compute composite momentum as the mean return over multiple lookbacks."""
    if ticker not in closes.columns:
        return -np.inf

    px = closes[ticker].dropna()
    if len(px) < MAX_LOOKBACK + 1:
        return -np.inf

    return float(np.mean([
        px.iloc[-1] / px.iloc[-(lb + 1)] - 1
        for lb in MOMENTUM_LOOKBACKS
    ]))


def passes_sma_gate(ticker: str, closes: pd.DataFrame) -> bool:
    """Check whether the asset's last close is above its long-run SMA.

    BIL is unconditionally exempt from this filter. BND, BNDX, and VGIT
    use a 126-day SMA; all other assets use the default ``SMA_PERIOD``.
    """
    if ticker == CASH_TICKER:
        return True

    if ticker not in closes.columns:
        return False

    px = closes[ticker].dropna()
    sma_period = SMA_OVERRIDES.get(ticker, SMA_PERIOD)

    if len(px) < sma_period:
        return False

    sma = float(px.rolling(sma_period).mean().iloc[-1])
    if np.isnan(sma):
        return False

    return bool(px.iloc[-1] > sma)


def realized_vol(ticker: str, closes: pd.DataFrame) -> float:
    """Estimate annualised realised volatility from log returns."""
    if ticker not in closes.columns:
        return np.nan

    px = closes[ticker].dropna()
    if len(px) < VOL_LOOKBACK + 1:
        return np.nan

    log_rets = np.log(px / px.shift(1)).dropna()
    if len(log_rets) < VOL_LOOKBACK:
        return np.nan

    vol = log_rets.tail(VOL_LOOKBACK).std()
    if vol is None or np.isnan(vol):
        return np.nan

    return float(vol * np.sqrt(252))


def absolute_return_6m(ticker: str, closes: pd.DataFrame) -> float:
    """Compute the trailing six-month (126-day) total return."""
    if ticker not in closes.columns:
        return -np.inf

    px = closes[ticker].dropna()
    if len(px) < 127:
        return -np.inf

    return float(px.iloc[-1] / px.iloc[-127] - 1)


# --------------------------------------------------------------------------- #
# Signal
# --------------------------------------------------------------------------- #

def compute_signal() -> dict:
    """Fetch latest data and compute the current momentum signal."""
    etf_closes = fetch_etf_closes()
    btc_closes = fetch_btc_closes()

    bil_ret_6m = absolute_return_6m(CASH_TICKER, etf_closes)

    # Build full asset list with their respective closes DataFrames
    all_assets: list[tuple[str, pd.DataFrame]] = [
        (t, etf_closes) for t in ETF_TICKERS
    ]
    if btc_closes is not None:
        all_assets.append((BTC_TICKER, btc_closes))

    # Compute diagnostics for every asset (eligible or not)
    diagnostics: dict[str, dict] = {}
    for ticker, closes in all_assets:
        sma_period = SMA_OVERRIDES.get(ticker, SMA_PERIOD)
        trend_pass = passes_sma_gate(ticker, closes)
        mom        = momentum(ticker, closes)
        ret_6m     = absolute_return_6m(ticker, closes)
        abs_pass   = (ticker == CASH_TICKER) or (ret_6m > bil_ret_6m)
        eligible   = trend_pass and abs_pass
        diagnostics[ticker] = {
            "sma_period": sma_period,
            "trend_pass": trend_pass,
            "momentum":   mom,
            "ret_6m":     ret_6m,
            "abs_pass":   abs_pass,
            "eligible":   eligible,
        }

    eligible_list: list[tuple[str, pd.DataFrame]] = [
        (t, c) for t, c in all_assets if diagnostics[t]["eligible"]
    ]

    if not eligible_list:
        return {
            "winners":      [(CASH_TICKER, 1.0)],
            "cash_weight":  0.0,
            "scores":       {CASH_TICKER: 0.0},
            "diagnostics":  diagnostics,
            "as_of":        str(etf_closes.index[-1].date()),
        }

    scores: dict[str, float] = {t: momentum(t, c) for t, c in eligible_list}
    sorted_eligible = sorted(eligible_list, key=lambda pair: scores[pair[0]], reverse=True)
    top_n = sorted_eligible[:TOP_N]

    per_slot_cap = MAX_WEIGHT / len(top_n)

    winners: list[tuple[str, float]] = []
    total_allocated = 0.0

    for winner, winner_closes in top_n:
        if winner == CASH_TICKER:
            weight = per_slot_cap
        else:
            vol = realized_vol(winner, winner_closes)
            weight = (
                min(per_slot_cap, TARGET_VOL / vol)
                if vol and not np.isnan(vol)
                else 0.0
            )
        winners.append((winner, weight))
        total_allocated += weight

    cash_weight = max(0.0, 1.0 - total_allocated)

    return {
        "winners":      winners,
        "cash_weight":  cash_weight,
        "scores":       scores,
        "diagnostics":  diagnostics,
        "as_of":        str(etf_closes.index[-1].date()),
    }


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    signal = compute_signal()
    diag   = signal["diagnostics"]

    print(f"\n{'─' * 66}")
    print(f"  Signal as of : {signal['as_of']}")
    print(f"{'─' * 66}")
    for i, (ticker, weight) in enumerate(signal["winners"], 1):
        print(f"  Winner #{i:<6} : {ticker}  ({weight:.1%})")
    print(f"  Cash (SGOV)  : {signal['cash_weight']:.1%}")

    # ------------------------------------------------------------------ #
    # All-asset diagnostics table
    # ------------------------------------------------------------------ #
    print(f"{'─' * 66}")
    print(f"  {'Ticker':<10} {'SMA':>5}  {'Trend':>5}  {'AbsMom':>6}  {'Momentum':>9}  {'Status'}")
    print(f"  {'──────':<10} {'───':>5}  {'─────':>5}  {'──────':>6}  {'────────':>9}  {'──────'}")
    for ticker, d in diag.items():
        trend_str = "✓" if d["trend_pass"] else "✗"
        abs_str   = "✓" if d["abs_pass"]   else "✗"
        mom_str   = f"{d['momentum']:>+.2%}" if d["momentum"] not in (-np.inf, np.inf) else "    n/a"
        ret_str   = f"{d['ret_6m']:>+.2%}"   if d["ret_6m"]   not in (-np.inf, np.inf) else "   n/a"
        status    = "ELIGIBLE" if d["eligible"] else "filtered"
        sma_lbl   = f"{d['sma_period']}d"
        print(f"  {ticker:<10} {sma_lbl:>5}  {trend_str:>5}  {abs_str:>6}  {mom_str:>9}  {status}")

    # ------------------------------------------------------------------ #
    # Eligible asset momentum ranking
    # ------------------------------------------------------------------ #
    winner_tickers = {t for t, _ in signal["winners"]}
    print(f"{'─' * 66}")
    print("  Momentum ranking (eligible assets):")
    for ticker, score in sorted(signal["scores"].items(), key=lambda x: -x[1]):
        marker = " ← WINNER" if ticker in winner_tickers else ""
        print(f"    {ticker:<10} {score:>+.2%}{marker}")

    print(f"{'─' * 66}\n")
