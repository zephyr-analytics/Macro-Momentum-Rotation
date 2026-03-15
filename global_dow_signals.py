"""
Dow Global Titans 50 — Live Signal Scanner
===========================================
Pulls the latest data from yfinance and outputs today's signals:

  Gate 1 (always on) : Price > 200-day EMA
  Gate 2 (optional)  : Screener filters
                         ① 6m return > SHV  (cash proxy)
                         ② 6m return > VGIT (intermediate Treasury)
                         ③ 3m, 6m AND 1y returns each > ACWI

Prints a ranked table and saves dow_titans_signals.csv

Requirements:
    pip install yfinance pandas --break-system-packages
"""

import yfinance as yf
import pandas as pd
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")


# ── CONFIG ────────────────────────────────────────────────────────────────────

EMA_PERIOD           = 200
USE_SCREENER_FILTERS = True   # Set True to also apply SHV / VGIT / ACWI gates
CASH_ETF             = "SHV"
TREASURY_ETF         = "VGIT"
ACWI_ETF             = "ACWI"
OUTPUT_CSV           = "dow_titans_signals.csv"

DOW_TITANS = [
    # Information Technology
    "AAPL",  "ADBE",  "AMD",   "ASML",  "AVGO",  "CRM",   "CSCO",  "HPQ",   "IBM",
    "INFY",  "INTC",  "INTU",  "MSFT",  "NOW",   "NVDA",  "ORCL",  "PLTR",  "PYPL",
    "QCOM",  "SAP",   "TSM",
    # Communication Services
    "AMX",   "BABA",  "DIS",   "ERIC",  "GOOG",  "GOOGL", "META",  "NFLX",  "SONY",
    "T",     "TEF",   "VZ",
    # Consumer Discretionary
    "AMZN",  "BKNG",  "CCL",   "EBAY",  "HD",    "HMC",   "MCD",   "NKE",   "SBUX",
    "TM",    "TSLA",
    # Consumer Staples
    "BUD",   "CL",    "COST",  "KO",    "MDLZ",  "PEP",   "PG",    "PM",    "WMT",
    # Health Care
    "ABBV",  "ABT",   "AMGN",  "AZN",   "BAX",   "CVS",   "GILD",  "GSK",   "JNJ",
    "LLY",   "MDT",   "MRK",   "NVS",   "PFE",   "TAK",   "UNH",
    # Financials
    "AIG",   "AXP",   "BAC",   "BBVA",  "BK",    "BRK-B", "DB",    "GS",    "HSBC",
    "JPM",   "MFG",   "MUFG",  "RY",    "SAN",   "SMP",   "TRV",   "UBS",   "V",
    "WFC",
    # Energy
    "BP",    "COP",   "CVX",   "PBR-A", "SHEL",  "SLB",   "TTE",   "XOM",
    # Industrials
    "ABB",   "BA",    "CAT",   "DD",    "DE",    "FDX",   "GE",    "HON",   "HWM",
    "MMM",   "RTX",   "UNP",   "UPS",
    # Materials
    "BHP",   "LIN",   "MT",    "NTR",   "RIO",
    # Real Estate
    "PLD",   "SPG",
    # Utilities
    "DUK",   "NEE",   "NGG",
]

BENCHMARKS = [CASH_ETF, TREASURY_ETF, ACWI_ETF]


# ── FETCH ─────────────────────────────────────────────────────────────────────

def fetch_prices(tickers: list, days: int) -> pd.DataFrame:
    start = (datetime.today() - timedelta(days=days)).strftime("%Y-%m-%d")
    raw = yf.download(tickers, start=start, auto_adjust=True, progress=False)["Close"]
    if isinstance(raw, pd.Series):
        raw = raw.to_frame(tickers[0])
    raw.index = pd.to_datetime(raw.index).tz_localize(None)
    return raw


# ── HELPERS ───────────────────────────────────────────────────────────────────

def ema(series: pd.Series, period: int) -> float:
    """Latest EMA value for a price series."""
    return series.dropna().ewm(span=period, adjust=False).mean().iloc[-1]


def period_return(series: pd.Series, calendar_days: int):
    """Trailing return (%) over the last `calendar_days` calendar days."""
    s = series.dropna()
    cutoff = s.index[-1] - timedelta(days=calendar_days)
    window = s[s.index >= cutoff]
    if len(window) < 10:
        return None
    return (window.iloc[-1] / window.iloc[0] - 1) * 100


# ── SIGNAL GENERATION ────────────────────────────────────────────────────────

def run_signals(stock_prices: pd.DataFrame, bench_prices: pd.DataFrame) -> pd.DataFrame:

    # Pre-compute benchmark returns (used by all stocks if screener is on)
    bench_cache = {}
    if USE_SCREENER_FILTERS:
        for label, ticker, days in [
            ("shv_6m",  CASH_ETF,     126),
            ("vgit_6m", TREASURY_ETF, 126),
            ("acwi_3m", ACWI_ETF,      63),
            ("acwi_6m", ACWI_ETF,     126),
            ("acwi_1y", ACWI_ETF,     252),
        ]:
            if ticker in bench_prices.columns:
                bench_cache[label] = period_return(bench_prices[ticker], days)

    rows = []
    for ticker in DOW_TITANS:
        if ticker not in stock_prices.columns:
            rows.append({"Ticker": ticker, "Signal": "NO DATA"})
            continue

        series = stock_prices[ticker].dropna()
        if series.empty:
            rows.append({"Ticker": ticker, "Signal": "NO DATA"})
            continue

        price     = series.iloc[-1]
        ema_val   = ema(series, EMA_PERIOD)
        above     = price > ema_val
        pct_vs_ema = (price / ema_val - 1) * 100

        r3m = period_return(series, 63)
        r6m = period_return(series, 126)
        r1y = period_return(series, 252)

        # Gate 1: EMA
        gate1 = above

        # Gate 2: Screener filters (optional)
        f1 = f2 = f3 = None
        if USE_SCREENER_FILTERS:
            shv_6m  = bench_cache.get("shv_6m")
            vgit_6m = bench_cache.get("vgit_6m")
            acwi_3m = bench_cache.get("acwi_3m")
            acwi_6m = bench_cache.get("acwi_6m")
            acwi_1y = bench_cache.get("acwi_1y")

            if acwi_3m is not None and acwi_3m < 0: acwi_3m = 0
            if acwi_6m is not None and acwi_6m < 0: acwi_6m = 0
            if acwi_1y is not None and acwi_1y < 0: acwi_1y = 0

            f1 = r6m is not None and shv_6m  is not None and r6m > shv_6m
            f2 = r6m is not None and vgit_6m is not None and r6m > vgit_6m
            f3 = (r3m is not None and acwi_3m is not None and r3m > acwi_3m and
                  r6m is not None and acwi_6m is not None and r6m > acwi_6m and
                  r1y is not None and acwi_1y is not None and r1y > acwi_1y)
            gate2 = f1 and f2 and f3
        else:
            gate2 = True  # not applicable

        eligible = gate1 and gate2

        row = {
            "Ticker":      ticker,
            "Price":       round(price, 2),
            "EMA_200":     round(ema_val, 2),
            "% vs EMA":    round(pct_vs_ema, 2),
            "Above EMA":   "✓" if above else "✗",
            "Ret 3m (%)":  round(r3m, 2) if r3m is not None else None,
            "Ret 6m (%)":  round(r6m, 2) if r6m is not None else None,
            "Ret 1y (%)":  round(r1y, 2) if r1y is not None else None,
        }

        if USE_SCREENER_FILTERS:
            row["F1 >SHV"]  = "✓" if f1 else "✗"
            row["F2 >VGIT"] = "✓" if f2 else "✗"
            row["F3 >ACWI"] = "✓" if f3 else "✗"

        row["Signal"] = "✅ BUY/HOLD" if eligible else "❌ OUT"
        rows.append(row)

    df = pd.DataFrame(rows)

    # Sort: eligible first (by % vs EMA desc), then ineligible
    df["_eligible"] = df["Signal"].str.startswith("✅")
    df = df.sort_values(["_eligible", "% vs EMA"], ascending=[False, False])
    df = df.drop(columns="_eligible").reset_index(drop=True)
    return df


# ── PRINT TABLE ───────────────────────────────────────────────────────────────

def print_table(df: pd.DataFrame, bench_prices: pd.DataFrame):
    today = datetime.today().strftime("%B %d, %Y")
    n_eligible = df["Signal"].str.startswith("✅").sum()

    print("\n" + "=" * 72)
    print(f"  DOW GLOBAL TITANS 50 — LIVE SIGNALS   {today}")
    print(f"  EMA filter: {EMA_PERIOD}-day   |   "
          f"Screener: {'ON' if USE_SCREENER_FILTERS else 'OFF'}   |   "
          f"Eligible: {n_eligible} / {len(df)}")
    print("=" * 72)

    if USE_SCREENER_FILTERS:
        # Print benchmark returns for reference
        print("\n  Benchmarks:")
        for label, ticker, days in [
            ("SHV  6m", CASH_ETF,     126),
            ("VGIT 6m", TREASURY_ETF, 126),
            ("ACWI 3m", ACWI_ETF,      63),
            ("ACWI 6m", ACWI_ETF,     126),
            ("ACWI 1y", ACWI_ETF,     252),
        ]:
            if ticker in bench_prices.columns:
                ret = period_return(bench_prices[ticker], days)
                val = f"{ret:+.2f}%" if ret is not None else "N/A"
                print(f"    {label} : {val}")
        print()

    # Column widths
    col_fmt = "{:<6} {:>9} {:>9} {:>8} {:>6} {:>8} {:>8} {:>8}  {}"
    if USE_SCREENER_FILTERS:
        col_fmt = "{:<6} {:>9} {:>9} {:>8} {:>6} {:>8} {:>8} {:>8} {:>7} {:>7} {:>7}  {}"

    def header():
        if USE_SCREENER_FILTERS:
            return col_fmt.format(
                "Ticker", "Price", "EMA 200", "% vs EMA", "EMA",
                "Ret 3m", "Ret 6m", "Ret 1y",
                "①>SHV", "②>VGIT", "③>ACWI", "Signal")
        return col_fmt.format(
            "Ticker", "Price", "EMA 200", "% vs EMA", "EMA",
            "Ret 3m", "Ret 6m", "Ret 1y", "Signal")

    def divider():
        return "-" * (100 if USE_SCREENER_FILTERS else 80)

    print(header())
    print(divider())

    prev_eligible = None
    for _, row in df.iterrows():
        is_eligible = row["Signal"].startswith("✅")
        if prev_eligible is not None and prev_eligible != is_eligible:
            print(divider())   # separator between eligible / ineligible blocks
        prev_eligible = is_eligible

        def fmt_ret(v):
            return f"{v:+.1f}%" if v is not None and not (isinstance(v, float) and pd.isna(v)) else "  N/A"

        args = [
            row["Ticker"],
            f"${row['Price']:,.2f}"  if "Price"   in row and row["Price"]   else "N/A",
            f"${row['EMA_200']:,.2f}"if "EMA_200" in row and row["EMA_200"] else "N/A",
            f"{row['% vs EMA']:+.1f}%" if "% vs EMA" in row and row["% vs EMA"] is not None else "N/A",
            row.get("Above EMA", ""),
            fmt_ret(row.get("Ret 3m (%)")),
            fmt_ret(row.get("Ret 6m (%)")),
            fmt_ret(row.get("Ret 1y (%)")),
        ]
        if USE_SCREENER_FILTERS:
            args += [row.get("F1 >SHV",""), row.get("F2 >VGIT",""), row.get("F3 >ACWI","")]
        args.append(row["Signal"])

        print(col_fmt.format(*args))

    print(divider())
    print(f"\n  ✅ {n_eligible} stocks eligible for equal-weight portfolio")
    if n_eligible > 0:
        weight = 100 / n_eligible
        print(f"  Target weight per stock: {weight:.1f}%")
    print()


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    # Need 400 calendar days to reliably seed 200-day EMA + 1y return window
    fetch_days = 400

    print("Dow Global Titans 50 — Live Signal Scanner")
    print("Fetching prices...")

    stock_prices = fetch_prices(DOW_TITANS, fetch_days)
    bench_prices = fetch_prices(BENCHMARKS, fetch_days) if USE_SCREENER_FILTERS else pd.DataFrame()

    df = run_signals(stock_prices, bench_prices)
    print_table(df, bench_prices)

    # Save CSV (clean numeric version without emoji for easy sorting)
    csv_df = df.copy()
    csv_df["Signal"] = csv_df["Signal"].str.replace("✅ ", "").str.replace("❌ ", "")
    csv_df.to_csv(OUTPUT_CSV, index=False)
    print(f"  Saved to: {OUTPUT_CSV}\n")


if __name__ == "__main__":
    main()
