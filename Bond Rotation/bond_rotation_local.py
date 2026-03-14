"""bond_signals.py — Cash & Bond Momentum signal runner.

Run this script directly to print today's signal:

    python bond_signals.py

Requires:
    pip install yfinance pandas numpy
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import date

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
ETF_TICKERS:        list[str] = ["SHV", "SGOV", "BND", "BNDX", "VGIT", "ICSH"]
CASH_TICKERS:       frozenset = frozenset({"SHV", "SGOV"})
CASH_BENCHMARK:     str       = "SHV"

MOMENTUM_LOOKBACKS: list[int] = [21, 63, 126, 189, 252]
MAX_LOOKBACK:       int       = max(MOMENTUM_LOOKBACKS)

SMA_PERIOD:         int       = 126

# Extra buffer so rolling windows are fully populated
_N_BARS: int = MAX_LOOKBACK + SMA_PERIOD + 10


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

def fetch_closes() -> pd.DataFrame:
    """Download adjusted close prices for all tickers.

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


# --------------------------------------------------------------------------- #
# Signal helpers
# --------------------------------------------------------------------------- #

def composite_momentum(ticker: str, closes: pd.DataFrame) -> float:
    """Mean total return across all lookback periods."""
    if ticker not in closes.columns:
        return -np.inf
    px = closes[ticker].dropna()
    if len(px) < MAX_LOOKBACK + 1:
        return -np.inf
    return float(np.mean([
        px.iloc[-1] / px.iloc[-(lb + 1)] - 1
        for lb in MOMENTUM_LOOKBACKS
    ]))


def composite_win_rate(ticker: str, closes: pd.DataFrame) -> float:
    """Mean win rate (% of daily returns > 0) across all lookback periods."""
    if ticker not in closes.columns:
        return -np.inf
    px = closes[ticker].dropna()
    if len(px) < MAX_LOOKBACK + 1:
        return -np.inf

    daily_rets = px.pct_change().dropna()
    rates = []
    for lb in MOMENTUM_LOOKBACKS:
        window = daily_rets.tail(lb)
        if len(window) < lb:
            continue
        rates.append(float((window > 0).mean()))

    return float(np.mean(rates)) if rates else -np.inf


def passes_sma_gate(ticker: str, closes: pd.DataFrame) -> bool:
    """Check whether the asset's last close is above its 126-day SMA.

    Cash proxies (SHV, SGOV, ICSH) are unconditionally exempt.
    """
    if ticker in CASH_TICKERS:
        return True
    if ticker not in closes.columns:
        return False
    px = closes[ticker].dropna()
    if len(px) < SMA_PERIOD:
        return False
    sma = float(px.rolling(SMA_PERIOD).mean().iloc[-1])
    if np.isnan(sma):
        return False
    return bool(px.iloc[-1] > sma)


def absolute_return_6m(ticker: str, closes: pd.DataFrame) -> float:
    """Trailing 126-day total return."""
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
    """Fetch latest data and compute the current cash/bond momentum signal.

    Returns
    -------
    dict
        Keys: ``weights``, ``diagnostics``, ``as_of``.
    """
    closes = fetch_closes()

    bench_ret_6m = absolute_return_6m(CASH_BENCHMARK, closes)

    # Build diagnostics for every asset
    diagnostics: dict[str, dict] = {}
    scores: dict[str, float] = {}

    for ticker in ETF_TICKERS:
        is_cash   = ticker in CASH_TICKERS
        trend     = passes_sma_gate(ticker, closes)
        ret_6m    = absolute_return_6m(ticker, closes)
        abs_pass  = is_cash or (ret_6m > bench_ret_6m)
        mom       = composite_momentum(ticker, closes)
        win       = composite_win_rate(ticker, closes)

        if is_cash:
            filter_reason = None
        elif not trend:
            filter_reason = "below SMA"
        elif not abs_pass:
            filter_reason = f"6m ret {ret_6m:+.2%} ≤ benchmark {bench_ret_6m:+.2%}"
        elif mom == -np.inf or win == -np.inf:
            filter_reason = "insufficient data"
        else:
            filter_reason = None

        raw_score = 0.0
        if filter_reason is None:
            if mom != -np.inf and win != -np.inf:
                raw_score = max(mom * win, 0.0)

        diagnostics[ticker] = {
            "is_cash":       is_cash,
            "trend_pass":    trend,
            "ret_6m":        ret_6m,
            "abs_pass":      abs_pass,
            "momentum":      mom,
            "win_rate":      win,
            "score":         raw_score,
            "filter_reason": filter_reason,
            "eligible":      filter_reason is None,
        }
        scores[ticker] = raw_score

    total = sum(scores.values())

    if total <= 0:
        weights = {CASH_BENCHMARK: 1.0}
    else:
        weights = {t: v / total for t, v in scores.items() if v > 0}

    return {
        "weights":     weights,
        "diagnostics": diagnostics,
        "as_of":       str(closes.index[-1].date()),
    }


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    signal = compute_signal()
    diag   = signal["diagnostics"]

    print(f"\n{'─' * 74}")
    print(f"  Cash/Bond Signal as of : {signal['as_of']}")
    print(f"{'─' * 74}")

    # ------------------------------------------------------------------ #
    # All-asset diagnostics table
    # ------------------------------------------------------------------ #
    print(f"  {'Ticker':<6}  {'Cash':>4}  {'Trend':>5}  {'AbsMom':>6}  {'6m Ret':>7}  {'Momentum':>9}  {'Win Rate':>8}  {'Score':>8}  Status")
    print(f"  {'──────':<6}  {'────':>4}  {'─────':>5}  {'──────':>6}  {'──────':>7}  {'────────':>9}  {'────────':>8}  {'─────':>8}  ──────")
    for ticker, d in diag.items():
        cash_str  = "✓" if d["is_cash"]    else " "
        trend_str = "✓" if d["trend_pass"] else "✗"
        abs_str   = "✓" if d["abs_pass"]   else "✗"
        mom_str   = f"{d['momentum']:>+.2%}"  if d["momentum"]  not in (-np.inf, np.inf) else "    n/a"
        win_str   = f"{d['win_rate']:>.1%}"   if d["win_rate"]  not in (-np.inf, np.inf) else "   n/a"
        ret_str   = f"{d['ret_6m']:>+.2%}"   if d["ret_6m"]    not in (-np.inf, np.inf) else "   n/a"
        score_str = f"{d['score']:.6f}"       if d["score"] > 0 else "      —"
        status    = "ELIGIBLE" if d["eligible"] else f"filtered ({d['filter_reason']})"
        print(f"  {ticker:<6}  {cash_str:>4}  {trend_str:>5}  {abs_str:>6}  {ret_str:>7}  {mom_str:>9}  {win_str:>8}  {score_str:>8}  {status}")

    # ------------------------------------------------------------------ #
    # Final weights
    # ------------------------------------------------------------------ #
    print(f"{'─' * 74}")
    print("  Final weights:")
    for ticker, w in sorted(signal["weights"].items(), key=lambda x: -x[1]):
        bar = "█" * int(w * 40)
        print(f"    {ticker:<6}  {w:>6.1%}  {bar}")
    print(f"{'─' * 74}\n")
