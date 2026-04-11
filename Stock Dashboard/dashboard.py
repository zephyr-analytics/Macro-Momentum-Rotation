"""
Momentum & Historical Band Ceiling Dashboard
Plotly Dash + yfinance

Data flow
---------
1. Sector Excel files in stock_files/ are read at startup to build the universe.
2. "Fetch Closes"  pulls 5y daily Close for every ticker → saved to cache/closes.csv
3. "Fetch OHLC"    pulls 60d daily OHLC  for every ticker → saved to cache/ohlc.csv
4. "Run Analysis"  reads both cache files and runs the full algorithm.
"""

import glob, os, re, time, traceback, warnings
warnings.filterwarnings("ignore")

import dash
from dash import dcc, html, dash_table, Input, Output, State
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import yfinance as yf
from typing import Optional   # ← Python 3.9-safe instead of X | Y

# ── Constants ─────────────────────────────────────────────────────────────────
LOOKBACKS     = [21, 63, 126, 189, 252]
BAND_LEN      = 189
HIST_LEN      = 126
BOTTOM_LEVELS = {0, 1, 2, 3, 4}
ADX_PERIOD    = 14
ADX_LIMIT     = 35
TOP_PER_FILE  = 100
CHUNK_SIZE    = 50
SLEEP_S       = 2.0

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
STOCK_FILES_DIR = os.path.join(BASE_DIR, "stock_files")
CACHE_DIR       = os.path.join(BASE_DIR, "cache")
CLOSES_CSV      = os.path.join(CACHE_DIR, "closes.csv")
OHLC_CSV        = os.path.join(CACHE_DIR, "ohlc.csv")
ERROR_LOG       = os.path.join(CACHE_DIR, "errors.log")

os.makedirs(CACHE_DIR, exist_ok=True)

def log_error(context: str, exc: Exception) -> None:
    """Write exception + traceback to errors.log and also print to console."""
    msg = f"\n[ERROR] {context}\n{traceback.format_exc()}\n"
    print(msg)
    try:
        with open(ERROR_LOG, "a") as f:
            f.write(msg)
    except Exception:
        pass

# ── Colour palette ─────────────────────────────────────────────────────────────
C = dict(
    bg="#0f1117", surface="#1a1d27", card="#21253a",
    border="#2e3250", text="#e8eaf6", muted="#7b82b0",
    green="#4caf82", red="#e05c5c", amber="#f0a830",
    blue="#4d9de0", purple="#8b7fdb", teal="#3ec9c9",
    grid="rgba(255,255,255,0.04)",
)

BASE_LAYOUT = dict(
    paper_bgcolor=C["surface"], plot_bgcolor=C["surface"],
    font=dict(color=C["text"], family="Inter, system-ui, sans-serif", size=12),
    margin=dict(l=12, r=12, t=40, b=12),
    colorway=[C["blue"], C["green"], C["purple"], C["amber"], C["teal"], C["red"]],
    xaxis=dict(gridcolor=C["grid"], linecolor=C["border"],
               tickcolor=C["border"], zerolinecolor=C["border"]),
    yaxis=dict(gridcolor=C["grid"], linecolor=C["border"],
               tickcolor=C["border"], zerolinecolor=C["border"]),
)

def layout(**overrides):
    """Return BASE_LAYOUT merged with overrides — avoids duplicate keyword errors."""
    return {**BASE_LAYOUT, **overrides}

# ── Market cap parser ──────────────────────────────────────────────────────────
def parse_mktcap(val) -> float:
    if val is None:
        return 0.0
    s = str(val).strip().replace(",", "").replace("$", "").upper()
    try:
        if s.endswith("T"):  return float(s[:-1]) * 1e12
        if s.endswith("B"):  return float(s[:-1]) * 1e9
        if s.endswith("M"):  return float(s[:-1]) * 1e6
        return float(s)
    except ValueError:
        return 0.0

# ── Sector file reader ─────────────────────────────────────────────────────────
def read_sector_file(path: str) -> pd.DataFrame:
    filename = os.path.basename(path)
    try:
        df = pd.read_csv(path) if path.lower().endswith(".csv") else pd.read_excel(path)
    except Exception as e:
        print(f"  x  {filename}: {e}")
        return pd.DataFrame()

    df.columns = [str(c).strip() for c in df.columns]
    col_map = {}
    for c in df.columns:
        cl = c.lower().replace(" ", "_")
        if cl == "symbol":                                  col_map[c] = "ticker"
        elif "company" in cl:                               col_map[c] = "company"
        elif "market_cap" in cl or "capitalization" in cl: col_map[c] = "mktcap_raw"
        elif "sector" in cl:                                col_map[c] = "sector"
        elif "sub" in cl and "industry" in cl:              col_map[c] = "sub_industry"
        elif "industry" in cl:                              col_map[c] = "industry"
        elif "exchange" in cl:                              col_map[c] = "exchange"
    df = df.rename(columns=col_map)

    if "ticker" not in df.columns:
        print(f"  x  {filename}: no Symbol column")
        return pd.DataFrame()

    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df = df[df["ticker"].str.match(r"^[A-Z]{1,6}$")].copy()
    df["mktcap"] = df["mktcap_raw"].apply(parse_mktcap) if "mktcap_raw" in df.columns else 0.0

    if "sector" not in df.columns or df["sector"].isna().all():
        df["sector"] = re.sub(r"[_\-]", " ", filename.rsplit(".", 1)[0]).title()

    for col in ["company", "industry", "sub_industry", "exchange"]:
        if col not in df.columns:
            df[col] = ""

    return df[["ticker", "company", "sector", "industry",
               "sub_industry", "exchange", "mktcap"]].copy()


def load_universe(top_n: int = TOP_PER_FILE):
    patterns = [os.path.join(STOCK_FILES_DIR, ext)
                for ext in ("*.xlsx", "*.xls", "*.csv")]
    paths = sorted(p for pat in patterns for p in glob.glob(pat))

    log = []
    if not paths:
        log.append(f"No files found in stock_files/")
        return pd.DataFrame(), log

    sector_dfs = []
    for path in paths:
        df = read_sector_file(path)
        if not df.empty:
            sector_dfs.append(df)
            log.append(f"  {os.path.basename(path)}  ({len(df)} tickers)")
        else:
            log.append(f"  x  {os.path.basename(path)}  (parse error)")

    if not sector_dfs:
        return pd.DataFrame(), log

    combined = pd.concat(sector_dfs, ignore_index=True)
    combined = combined[combined["mktcap"] > 0]
    top = (combined
           .sort_values("mktcap", ascending=False)
           .groupby("sector", group_keys=False)
           .head(top_n)
           .sort_values("mktcap", ascending=False)
           .drop_duplicates(subset="ticker")
           .reset_index(drop=True))

    log.append(f"  -> {len(top)} tickers in universe (top {top_n}/sector, deduped)")
    return top, log


# ── yfinance MultiIndex helper ─────────────────────────────────────────────────
def _extract_field(raw: pd.DataFrame, ticker: str, field: str) -> Optional[pd.Series]:
    """
    Safely extract a single price field for one ticker from a yfinance
    multi-ticker download result.

    yfinance ≥ 0.2 returns a MultiIndex with TWO possible orderings:
      - (Price, Ticker)  e.g. columns = [("Close","AAPL"), ("Close","MSFT"), …]
      - (Ticker, Price)  e.g. columns = [("AAPL","Close"), ("AAPL","High"), …]

    We try both, then fall back to a flat single-ticker DataFrame.
    """
    if raw is None or raw.empty:
        return None

    cols = raw.columns

    # MultiIndex case
    if isinstance(cols, pd.MultiIndex):
        # Detect ordering by checking which level contains known price names
        price_fields = {"Open", "High", "Low", "Close", "Volume",
                        "open", "high", "low", "close", "volume"}
        lvl0_is_price = any(str(v) in price_fields for v in cols.get_level_values(0))

        if lvl0_is_price:
            # (Price, Ticker) ordering — standard for recent yfinance
            try:
                return raw[field][ticker].dropna()
            except KeyError:
                pass
            # Try case-insensitive match on ticker
            for t in cols.get_level_values(1).unique():
                if str(t).upper() == ticker.upper():
                    try:
                        return raw[field][t].dropna()
                    except KeyError:
                        pass
        else:
            # (Ticker, Price) ordering
            for t in cols.get_level_values(0).unique():
                if str(t).upper() == ticker.upper():
                    try:
                        return raw[t][field].dropna()
                    except KeyError:
                        pass

    # Flat DataFrame (single ticker downloaded alone)
    if field in raw.columns:
        return raw[field].dropna()

    return None


def _extract_ohlc(raw: pd.DataFrame, ticker: str) -> Optional[pd.DataFrame]:
    """Return a DataFrame with columns High, Low, Close for one ticker."""
    result = {}
    for f in ("High", "Low", "Close"):
        s = _extract_field(raw, ticker, f)
        if s is None:
            return None
        result[f] = s
    try:
        df = pd.DataFrame(result).dropna()
        return df if not df.empty else None
    except Exception:
        return None


# ── yfinance chunked fetch helpers ────────────────────────────────────────────
def fetch_closes(tickers: list) -> pd.DataFrame:
    """Pull 5-year daily Close for each ticker in chunks. Saves to CLOSES_CSV."""
    print(f"\nFetching closes for {len(tickers)} tickers "
          f"in chunks of {CHUNK_SIZE} …")
    chunks = [tickers[i:i+CHUNK_SIZE] for i in range(0, len(tickers), CHUNK_SIZE)]
    frames = []

    for i, chunk in enumerate(chunks):
        print(f"  close chunk {i+1}/{len(chunks)}: "
              f"{chunk[:3]}{'…' if len(chunk)>3 else ''}")
        try:
            raw = yf.download(chunk, period="5y", interval="1d",
                              auto_adjust=True, progress=False,
                              group_by="ticker")   # explicit group_by
            if raw is None or raw.empty:
                print("    no data returned")
                continue

            print(f"    raw shape: {raw.shape}  "
                  f"col levels: {raw.columns.nlevels}  "
                  f"sample cols: {list(raw.columns[:6])}")

            for t in chunk:
                try:
                    s = _extract_field(raw, t, "Close")
                    if s is not None and len(s) > 50:
                        frames.append(s.rename(t))
                    elif s is None:
                        print(f"    {t}: Close not found in result")
                except Exception as e:
                    print(f"    {t}: {e}")

        except Exception as e:
            log_error(f"fetch_closes chunk {i+1}", e)

        if i < len(chunks) - 1:
            time.sleep(SLEEP_S)

    if not frames:
        print("  No close data collected — check errors.log")
        return pd.DataFrame()

    df = pd.concat(frames, axis=1)
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    df.to_csv(CLOSES_CSV)
    print(f"  Saved {df.shape[1]} tickers × {df.shape[0]} rows → {CLOSES_CSV}")
    return df


def fetch_ohlc(tickers: list) -> pd.DataFrame:
    """Pull 60-day daily OHLC for each ticker in chunks. Saves to OHLC_CSV."""
    print(f"\nFetching OHLC (60d) for {len(tickers)} tickers "
          f"in chunks of {CHUNK_SIZE} …")
    chunks = [tickers[i:i+CHUNK_SIZE] for i in range(0, len(tickers), CHUNK_SIZE)]
    frames = []

    for i, chunk in enumerate(chunks):
        print(f"  ohlc chunk {i+1}/{len(chunks)}: "
              f"{chunk[:3]}{'…' if len(chunk)>3 else ''}")
        try:
            raw = yf.download(chunk, period="60d", interval="1d",
                              auto_adjust=True, progress=False,
                              group_by="ticker")
            if raw is None or raw.empty:
                print("    no data returned")
                continue

            for t in chunk:
                try:
                    sub = _extract_ohlc(raw, t)
                    if sub is None or len(sub) < ADX_PERIOD * 2:
                        continue
                    sub = sub.copy()
                    sub["Ticker"] = t
                    frames.append(sub)
                except Exception as e:
                    print(f"    {t}: {e}")

        except Exception as e:
            log_error(f"fetch_ohlc chunk {i+1}", e)

        if i < len(chunks) - 1:
            time.sleep(SLEEP_S)

    if not frames:
        print("  No OHLC data collected — check errors.log")
        return pd.DataFrame()

    df = pd.concat(frames)
    df.index.name = "Date"
    df = df.reset_index()
    df.to_csv(OHLC_CSV, index=False)
    print(f"  Saved {df['Ticker'].nunique()} tickers → {OHLC_CSV}")
    return df


# ── Cache readers ─────────────────────────────────────────────────────────────
def load_closes_cache() -> pd.DataFrame:
    if not os.path.exists(CLOSES_CSV):
        return pd.DataFrame()
    try:
        df = pd.read_csv(CLOSES_CSV, index_col="Date", parse_dates=True)
        print(f"  Closes cache: {df.shape[1]} tickers × {df.shape[0]} rows")
        return df
    except Exception as e:
        log_error("load_closes_cache", e)
        return pd.DataFrame()

def load_ohlc_cache() -> pd.DataFrame:
    if not os.path.exists(OHLC_CSV):
        return pd.DataFrame()
    try:
        df = pd.read_csv(OHLC_CSV, parse_dates=["Date"])
        print(f"  OHLC cache: {df['Ticker'].nunique()} tickers, "
              f"{len(df)} rows")
        return df
    except Exception as e:
        log_error("load_ohlc_cache", e)
        return pd.DataFrame()

def cache_status() -> dict:
    def info(path):
        if not os.path.exists(path):
            return None, None
        mtime = os.path.getmtime(path)
        ts    = pd.Timestamp(mtime, unit="s").strftime("%Y-%m-%d %H:%M")
        with open(path) as f:
            rows = sum(1 for _ in f) - 1
        return ts, rows
    c_ts, c_rows = info(CLOSES_CSV)
    o_ts, o_rows = info(OHLC_CSV)
    return dict(closes_ts=c_ts, closes_rows=c_rows,
                ohlc_ts=o_ts,   ohlc_rows=o_rows)


# ── Quant helpers ──────────────────────────────────────────────────────────────
def ema_series(arr: np.ndarray, period: int) -> np.ndarray:
    k   = 2.0 / (period + 1)
    out = np.empty(len(arr), dtype=float)
    out[0] = float(arr[0])
    for i in range(1, len(arr)):
        out[i] = float(arr[i]) * k + out[i - 1] * (1 - k)
    return out

def band_index(price: float, bands: list) -> int:
    for i in range(len(bands) - 1):
        if bands[i] <= price < bands[i + 1]:
            return i
    return len(bands) - 2

def compute_adx(hi: np.ndarray, lo: np.ndarray, cl: np.ndarray,
                period: int = 14) -> float:
    if len(cl) < period * 2:
        return 0.0
    tr, pdm, ndm = [], [], []
    for i in range(1, len(cl)):
        tr.append(max(hi[i]-lo[i], abs(hi[i]-cl[i-1]), abs(lo[i]-cl[i-1])))
        up   = hi[i]   - hi[i-1]
        down = lo[i-1] - lo[i]
        pdm.append(up   if up > down and up > 0   else 0.0)
        ndm.append(down if down > up and down > 0 else 0.0)
    atr = ema_series(np.array(tr,  float), period)
    pdi = 100.0 * ema_series(np.array(pdm, float), period) / (atr + 1e-9)
    ndi = 100.0 * ema_series(np.array(ndm, float), period) / (atr + 1e-9)
    dx  = 100.0 * np.abs(pdi - ndi) / (pdi + ndi + 1e-9)
    return float(ema_series(dx, period)[-1])

# Fixed QC-matching breadth band multipliers (from OnData)
BREADTH_BANDS_MULT = [1.618, 1.382, 1.0, 0.809, 0.5, 0.382]  # symmetric around mid

def _breadth_band_index(price: float, mid: float, dev: float) -> int:
    """
    Compute band index using fixed golden-ratio multipliers.
    Matches QC OnData exactly — does NOT use dynamic lm.
    """
    bands = [
        mid - dev * 1.618,
        mid - dev * 1.382,
        mid - dev * 1.0,
        mid - dev * 0.809,
        mid - dev * 0.5,
        mid - dev * 0.382,
        mid,
        mid + dev * 0.382,
        mid + dev * 0.5,
        mid + dev * 0.809,
        mid + dev * 1.0,
        mid + dev * 1.382,
        mid + dev * 1.618,
    ]
    return band_index(price, bands)


def compute_breadth_history(closes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Replicate QC OnData breadth tracking bar-by-bar over the full close history.

    For every trading day, computes the fixed-band index for every ticker,
    then calculates bottom_frac (fraction of universe in bands 0-4).
    Also computes the regime state (risk_off, was_risk_off, max_stress_level)
    matching QC's recovery logic exactly.

    Returns a DataFrame indexed by Date with columns:
        bottom_frac, regime  ('risk_on' | 'risk_off' | 'recovery')
        max_stress_level, improvement (for recovery display)
    """
    closes = closes_df.dropna(axis=1, how="all")
    tickers = closes.columns.tolist()
    dates   = closes.index

    min_bars = BAND_LEN + 10

    # Pre-compute EMA arrays for each ticker (vectorised via pandas ewm)
    k = 2.0 / (BAND_LEN + 1)
    ema_df = closes.ewm(span=BAND_LEN, adjust=False).mean()

    # Rolling std — use min_periods so early rows are NaN
    std_df = closes.rolling(BAND_LEN, min_periods=BAND_LEN).std()

    records = []

    # Regime state (mirrors QC instance variables)
    was_risk_off     = False
    max_stress_level = 0.0
    risk_off         = False

    for i, date in enumerate(dates):
        if i < min_bars:
            continue

        row_close = closes.iloc[i]
        row_ema   = ema_df.iloc[i]
        row_std   = std_df.iloc[i]

        band_idxs = []
        for t in tickers:
            c   = row_close[t]
            mid = row_ema[t]
            dev = row_std[t]
            if pd.isna(c) or pd.isna(mid) or pd.isna(dev) or dev <= 0:
                continue
            band_idxs.append(_breadth_band_index(c, mid, dev))

        if len(band_idxs) < 50:
            continue

        bottom_frac = sum(1 for idx in band_idxs if idx in BOTTOM_LEVELS) / len(band_idxs)
        max_stress_level = max(max_stress_level, bottom_frac)

        # ── Regime logic matching QC exactly ──
        if bottom_frac >= 0.45:
            risk_off     = True
            was_risk_off = True
            regime       = "risk_off"
            improvement  = 0.0

        elif was_risk_off:
            denominator = max(max_stress_level, 0.10)
            improvement = (max_stress_level - bottom_frac) / denominator

            if improvement >= 0.60 or bottom_frac < 0.15:
                # Recovery — ceilings would reset here in QC
                was_risk_off     = False
                risk_off         = False
                max_stress_level = 0.0
                regime           = "recovery"
            else:
                regime   = "risk_off"   # still waiting for full recovery
                risk_off = True
        else:
            risk_off    = False
            improvement = 0.0
            regime      = "risk_on"

        records.append(dict(
            date=date,
            bottom_frac=bottom_frac,
            regime=regime,
            max_stress_level=max_stress_level,
            improvement=improvement,
        ))

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records).set_index("date")
    return df


def analyse_ticker(ticker: str,
                   close_series: pd.Series,
                   ohlc_df: Optional[pd.DataFrame],
                   ceiling_reset_dates: Optional[list] = None) -> Optional[dict]:
    """
    Matches QC Rebalance logic exactly:
    - stretch_ema  : clean manual EMA of raw stretch values (not price-fed)
    - peak_stretch : max of raw instantaneous stretch (not EMA peak)
    - band_hist    : updated on every bar for every ticker (not just top picks)
    - sizing bands : dynamic lm-based multipliers
    - breadth bands: NOT used here (handled separately in compute_breadth_history)

    ceiling_reset_dates: list of date indices where band_hist was reset
                         (from recovery events in breadth history)
    """
    try:
        closes = close_series.dropna()
        dates  = closes.index
        vals   = closes.values.astype(float)
        n      = len(vals)

        if n < max(LOOKBACKS) + BAND_LEN + 10:
            return None

        reset_set = set(ceiling_reset_dates or [])

        ema_arr     = ema_series(vals, BAND_LEN)
        stretch_arr = []          # raw stretch per bar
        stretch_ema_arr = []      # manual EMA of stretch (matches QC stretch_ema)
        band_idx_hist   = []      # updated every bar (matches QC OnData + band_hist)
        peak_stretch    = 0.0     # max of raw stretch (matches QC stretch_max)

        ema_k = 2.0 / (BAND_LEN + 1)
        stretch_ema_val = None    # running EMA state

        # rolling band_hist window of length HIST_LEN (mirrors QC RollingWindow)
        bh_window: list = []

        for i in range(BAND_LEN, n):
            # ── Ceiling reset event (matches QC recovery branch) ──
            if dates[i] in reset_set:
                bh_window = []

            window = vals[i - BAND_LEN: i]
            dev    = float(np.std(window))
            if dev <= 0:
                continue

            mid    = float(ema_arr[i])
            close  = float(vals[i])

            # 1. Raw stretch (QC: stretch = abs(close - mid) / dev)
            stretch = abs(close - mid) / dev
            stretch_arr.append(stretch)

            # 2. Peak of raw stretch (QC: stretch_max)
            if stretch > peak_stretch:
                peak_stretch = stretch

            # 3. Manual EMA of stretch (QC: stretch_ema.Update(time, stretch))
            if stretch_ema_val is None:
                stretch_ema_val = stretch
            else:
                stretch_ema_val = stretch * ema_k + stretch_ema_val * (1 - ema_k)
            stretch_ema_arr.append(stretch_ema_val)

            # 4. Dynamic sizing bands using current stretch_ema (QC Rebalance bands)
            lm  = stretch_ema_val
            lm2 = lm / 2.0
            lm3 = lm2 * 0.38196601
            lm4 = lm * 1.38196601
            lm5 = lm * 1.61803399
            lm6 = (lm + lm2) / 2.0
            sizing_bands = [
                mid-dev*lm5, mid-dev*lm4, mid-dev*lm,  mid-dev*lm6,
                mid-dev*lm2, mid-dev*lm3, mid,
                mid+dev*lm3, mid+dev*lm2, mid+dev*lm6,
                mid+dev*lm,  mid+dev*lm4, mid+dev*lm5,
            ]

            # 5. band_hist updated every bar for every ticker (QC: OnData + band_hist)
            idx = band_index(close, sizing_bands)
            bh_window.append(idx)
            if len(bh_window) > HIST_LEN:
                bh_window.pop(0)

            band_idx_hist.append(idx)

        if not stretch_ema_arr:
            return None

        final_stretch_ema = stretch_ema_val
        last_close = float(vals[-1])
        ema_val    = float(ema_arr[-1])
        dev_val    = float(np.std(vals[-BAND_LEN:]))
        if dev_val <= 0:
            return None

        # Final sizing bands at last bar
        lm  = final_stretch_ema
        lm2 = lm / 2.0
        lm3 = lm2 * 0.38196601
        lm4 = lm * 1.38196601
        lm5 = lm * 1.61803399
        lm6 = (lm + lm2) / 2.0
        final_bands = [
            ema_val-dev_val*lm5, ema_val-dev_val*lm4, ema_val-dev_val*lm,
            ema_val-dev_val*lm6, ema_val-dev_val*lm2, ema_val-dev_val*lm3, ema_val,
            ema_val+dev_val*lm3, ema_val+dev_val*lm2, ema_val+dev_val*lm6,
            ema_val+dev_val*lm,  ema_val+dev_val*lm4, ema_val+dev_val*lm5,
        ]

        idx       = band_index(last_close, final_bands)
        hist_high = max(bh_window) if bh_window else idx

        if hist_high <= 0:      scale = 1.0
        elif idx >= hist_high:  scale = 0.0
        else:                   scale = max(0.2, 1.0 - idx / hist_high)

        # Exhaustion: peak is raw stretch_max, current is stretch_ema (matches QC)
        exhausted = (idx >= 10 and peak_stretch > 0
                     and final_stretch_ema < peak_stretch * 0.80)
        if exhausted:
            scale = 0.2

        mom = float(np.mean([
            vals[-1] / vals[-lb - 1] - 1
            for lb in LOOKBACKS if n > lb + 1
        ]))

        adx_val = 0.0
        if ohlc_df is not None and len(ohlc_df) >= ADX_PERIOD * 2:
            adx_val = compute_adx(
                ohlc_df["High"].values.astype(float),
                ohlc_df["Low"].values.astype(float),
                ohlc_df["Close"].values.astype(float),
                ADX_PERIOD,
            )

        return dict(
            ticker=ticker,
            last_close=last_close, ema_val=ema_val, dev_val=dev_val,
            idx=idx, hist_high=hist_high, scale=scale,
            mom=mom, above_ema=bool(last_close > ema_val),
            exhausted=exhausted,
            stretch_ema_val=final_stretch_ema, peak_stretch=peak_stretch,
            adx=adx_val,
            band_idx_hist=band_idx_hist[-60:],
            company="", sector="", industry="", mktcap=0,
        )
    except Exception as e:
        log_error(f"analyse_ticker {ticker}", e)
        return None


def run_portfolio(results: list, breadth_df: pd.DataFrame,
                  top_n: int, max_weight: float) -> dict:
    """
    Portfolio construction using the pre-computed breadth history.
    Current regime is taken from the last row of breadth_df.
    """
    if breadth_df.empty:
        # Fallback: compute breadth from current band indices only
        all_idx     = [r["idx"] for r in results]
        bottom_frac = sum(1 for i in all_idx if i in BOTTOM_LEVELS) / max(len(all_idx), 1)
        regime      = "risk_off" if bottom_frac >= 0.45 else "risk_on"
        improvement = 0.0
        max_stress  = bottom_frac
    else:
        last        = breadth_df.iloc[-1]
        bottom_frac = float(last["bottom_frac"])
        regime      = str(last["regime"])
        improvement = float(last["improvement"])
        max_stress  = float(last["max_stress_level"])

    risk_off = regime in ("risk_off",)   # "recovery" allows trading

    eligible = [r for r in results
                if r["above_ema"] and r["mom"] > 0 and r["adx"] <= ADX_LIMIT]
    eligible.sort(key=lambda r: r["mom"] * r["scale"], reverse=True)
    top = eligible[:top_n]

    if risk_off or not top:
        return dict(risk_off=risk_off, regime=regime,
                    bottom_frac=bottom_frac, improvement=improvement,
                    max_stress=max_stress,
                    positions=[], final_weights={}, all_results=results)

    raw     = {r["ticker"]: r["mom"] * r["scale"] for r in top}
    total   = sum(raw.values())
    capped  = {t: min(max_weight, v / total) for t, v in raw.items()}
    cs      = sum(capped.values())
    final_w = {t: v / cs for t, v in capped.items()} if cs > 0 else {}

    return dict(risk_off=False, regime=regime,
                bottom_frac=bottom_frac, improvement=improvement,
                max_stress=max_stress,
                positions=top, final_weights=final_w, all_results=results)


# ── UI helpers ─────────────────────────────────────────────────────────────────
def card(children, extra=None):
    style = dict(background=C["card"], borderRadius="10px",
                 border=f"1px solid {C['border']}", padding="18px 22px",
                 marginBottom="18px")
    if extra:
        style.update(extra)
    return html.Div(children, style=style)

def metric_box(label, value, sub=None, color=None):
    return html.Div([
        html.Div(label, style=dict(fontSize="11px", color=C["muted"],
                                   textTransform="uppercase", letterSpacing=".07em",
                                   marginBottom="6px")),
        html.Div(value, style=dict(fontSize="26px", fontWeight="600",
                                   color=color or C["text"])),
        html.Div(sub, style=dict(fontSize="11px", color=C["muted"],
                                  marginTop="3px")) if sub else None,
    ], style=dict(background=C["surface"], borderRadius="8px",
                  padding="14px 18px", flex="1", minWidth="130px"))

def chart_card(fig, height=None):
    if height:
        fig.update_layout(height=height)
    return html.Div(
        dcc.Graph(figure=fig, config=dict(displayModeBar=False),
                  style=dict(width="100%")),
        style=dict(flex="1", background=C["card"], borderRadius="10px",
                   border=f"1px solid {C['border']}", padding="8px",
                   minWidth="320px"),
    )

def flex_row(*children):
    return html.Div(list(children),
                    style=dict(display="flex", gap="16px",
                               flexWrap="wrap", marginBottom="18px"))

def btn(label, id_, color=None):
    return html.Button(label, id=id_, n_clicks=0, style=dict(
        background=color or C["blue"], color="#fff",
        border="none", borderRadius="8px",
        padding="10px 22px", fontSize="13px",
        fontWeight="600", cursor="pointer", whiteSpace="nowrap",
    ))

def status_line(text, color=None):
    return html.Div(text, style=dict(
        fontSize="11px", color=color or C["muted"],
        marginTop="6px", lineHeight="1.6",
    ))

def error_banner(context: str, exc: Exception) -> html.Div:
    tb = traceback.format_exc()
    return html.Div([
        html.Div(f"⚠ Error in {context}: {exc}",
                 style=dict(color=C["red"], fontWeight="600",
                            marginBottom="8px")),
        html.Pre(tb, style=dict(fontSize="10px", color=C["muted"],
                                whiteSpace="pre-wrap", wordBreak="break-all",
                                background=C["bg"], padding="10px",
                                borderRadius="6px", maxHeight="300px",
                                overflow="auto")),
        html.Div(f"Full traceback also saved to: {ERROR_LOG}",
                 style=dict(fontSize="10px", color=C["muted"], marginTop="6px")),
    ], style=dict(background=C["card"], border=f"1px solid {C['red']}44",
                  borderRadius="10px", padding="18px 22px", marginBottom="18px"))


# ── Startup ────────────────────────────────────────────────────────────────────
print(f"\nyfinance version: {yf.__version__}")
print(f"Loading universe from: {STOCK_FILES_DIR}")
UNIVERSE_DF, STARTUP_LOG = load_universe(TOP_PER_FILE)
for l in STARTUP_LOG:
    print(l)
print()

# ── App ────────────────────────────────────────────────────────────────────────
app    = dash.Dash(__name__, title="Momentum Dashboard",
                   suppress_callback_exceptions=True)
server = app.server

if UNIVERSE_DF.empty:
    univ_block = html.Div(
        f"No files found in stock_files/  (expected: {STOCK_FILES_DIR})",
        style=dict(color=C["red"], fontSize="13px"),
    )
else:
    n_sectors = UNIVERSE_DF["sector"].nunique()
    n_tickers = len(UNIVERSE_DF)
    univ_block = html.Div([
        html.Div(f"{n_sectors} sector files  |  {n_tickers} tickers in universe",
                 style=dict(color=C["green"], fontSize="13px", marginBottom="4px")),
        html.Div("  ·  ".join(sorted(UNIVERSE_DF["sector"].unique())),
                 style=dict(fontSize="11px", color=C["muted"])),
    ])

def make_cache_block():
    cs = cache_status()
    def file_line(label, ts, rows, path):
        exists = ts is not None
        color  = C["green"] if exists else C["muted"]
        text   = (f"{label}:  {os.path.basename(path)}  "
                  f"({rows} rows, updated {ts})" if exists
                  else f"{label}:  no cache yet")
        return html.Div(text, style=dict(fontSize="11px", color=color,
                                          marginBottom="2px"))
    return html.Div([
        file_line("Closes", cs["closes_ts"], cs["closes_rows"], CLOSES_CSV),
        file_line("OHLC",   cs["ohlc_ts"],   cs["ohlc_rows"],   OHLC_CSV),
    ])

app.layout = html.Div(style=dict(
    background=C["bg"], minHeight="100vh", color=C["text"],
    fontFamily="Inter, system-ui, sans-serif", padding="24px 32px",
), children=[

    html.Div([
        html.H1("Momentum · Band Ceiling Dashboard",
                style=dict(fontSize="22px", fontWeight="600", margin="0")),
        html.Div("Sector-neutral large-cap · historical band ceiling sizing · yfinance",
                 style=dict(fontSize="13px", color=C["muted"], marginTop="4px")),
    ], style=dict(marginBottom="28px")),

    card([
        html.Div("Data & Configuration", style=dict(
            fontSize="11px", color=C["muted"], textTransform="uppercase",
            letterSpacing=".07em", marginBottom="18px")),

        html.Div([

            html.Div([
                html.Div("Universe  (stock_files/)",
                         style=dict(fontSize="12px", color=C["muted"],
                                    marginBottom="8px", fontWeight="500")),
                univ_block,
            ], style=dict(minWidth="300px", maxWidth="420px")),

            html.Div(style=dict(width="1px", background=C["border"],
                                margin="0 20px", alignSelf="stretch")),

            html.Div([
                html.Div("Cached data",
                         style=dict(fontSize="12px", color=C["muted"],
                                    marginBottom="8px", fontWeight="500")),
                html.Div(id="cache-status-block", children=make_cache_block()),
                html.Div([
                    btn("Fetch Closes (5y)", "fetch-closes-btn", C["purple"]),
                    btn("Fetch OHLC (60d)",  "fetch-ohlc-btn",   C["teal"]),
                ], style=dict(display="flex", gap="10px",
                              marginTop="12px", flexWrap="wrap")),
                html.Div(id="fetch-status",
                         style=dict(fontSize="11px", color=C["muted"],
                                    marginTop="8px", minHeight="18px")),
            ], style=dict(minWidth="340px")),

            html.Div(style=dict(width="1px", background=C["border"],
                                margin="0 20px", alignSelf="stretch")),

            html.Div([
                html.Div("Parameters",
                         style=dict(fontSize="12px", color=C["muted"],
                                    marginBottom="14px", fontWeight="500")),
                html.Div([
                    html.Div([
                        html.Div("Top N per sector",
                                 style=dict(fontSize="11px", color=C["muted"],
                                            marginBottom="4px")),
                        dcc.Slider(id="top-per-file", min=10, max=100, step=10,
                                   value=100,
                                   marks={10:"10", 25:"25", 50:"50",
                                          75:"75", 100:"100"},
                                   tooltip=dict(placement="bottom",
                                                always_visible=True)),
                    ], style=dict(minWidth="240px")),
                    html.Div([
                        html.Div("Portfolio holdings",
                                 style=dict(fontSize="11px", color=C["muted"],
                                            marginBottom="4px")),
                        dcc.Slider(id="top-n", min=3, max=20, step=1, value=10,
                                   marks={3:"3", 5:"5", 10:"10",
                                          15:"15", 20:"20"},
                                   tooltip=dict(placement="bottom",
                                                always_visible=True)),
                    ], style=dict(minWidth="240px")),
                    html.Div([
                        html.Div("Max weight per position (%)",
                                 style=dict(fontSize="11px", color=C["muted"],
                                            marginBottom="4px")),
                        dcc.Slider(id="max-weight", min=5, max=40, step=5,
                                   value=20,
                                   marks={5:"5", 10:"10", 20:"20",
                                          30:"30", 40:"40"},
                                   tooltip=dict(placement="bottom",
                                                always_visible=True)),
                    ], style=dict(minWidth="240px")),
                ], style=dict(display="flex", flexDirection="column", gap="18px")),
            ]),

            html.Div(style=dict(width="1px", background=C["border"],
                                margin="0 20px", alignSelf="stretch")),

            html.Div([
                btn("Run Analysis", "run-btn"),
                html.Div(id="run-status",
                         style=dict(fontSize="11px", color=C["muted"],
                                    marginTop="8px", textAlign="center")),
            ], style=dict(display="flex", flexDirection="column",
                          alignItems="center", justifyContent="center",
                          minWidth="140px")),

        ], style=dict(display="flex", alignItems="flex-start",
                      flexWrap="wrap", gap="8px")),
    ]),

    dcc.Loading(id="loading-fetch",  type="dot",   color=C["purple"],
                children=html.Div(id="fetch-spinner")),
    dcc.Loading(id="loading-output", type="circle", color=C["blue"],
                children=html.Div(id="dashboard-output")),
])


# ── Callback: Fetch Closes ─────────────────────────────────────────────────────
@app.callback(
    Output("fetch-status",       "children"),
    Output("cache-status-block", "children"),
    Input("fetch-closes-btn",    "n_clicks"),
    prevent_initial_call=True,
)
def do_fetch_closes(n_clicks):
    try:
        if UNIVERSE_DF.empty:
            return status_line("No universe loaded.", C["red"]), make_cache_block()
        tickers = UNIVERSE_DF["ticker"].tolist()
        df      = fetch_closes(tickers)
        if df.empty:
            return (status_line(
                        f"Closes fetch returned no data. "
                        f"Check console and {ERROR_LOG}", C["red"]),
                    make_cache_block())
        msg = (f"Closes saved: {df.shape[1]} tickers × {df.shape[0]} rows "
               f"→ cache/closes.csv")
        return status_line(msg, C["green"]), make_cache_block()
    except Exception as e:
        log_error("do_fetch_closes", e)
        return status_line(f"Error: {e}", C["red"]), make_cache_block()


# ── Callback: Fetch OHLC ───────────────────────────────────────────────────────
@app.callback(
    Output("fetch-status",       "children", allow_duplicate=True),
    Output("cache-status-block", "children", allow_duplicate=True),
    Input("fetch-ohlc-btn",      "n_clicks"),
    prevent_initial_call=True,
)
def do_fetch_ohlc(n_clicks):
    try:
        if UNIVERSE_DF.empty:
            return status_line("No universe loaded.", C["red"]), make_cache_block()
        tickers = UNIVERSE_DF["ticker"].tolist()
        df      = fetch_ohlc(tickers)
        if df.empty:
            return (status_line(
                        f"OHLC fetch returned no data. "
                        f"Check console and {ERROR_LOG}", C["red"]),
                    make_cache_block())
        msg = (f"OHLC saved: {df['Ticker'].nunique()} tickers "
               f"→ cache/ohlc.csv")
        return status_line(msg, C["green"]), make_cache_block()
    except Exception as e:
        log_error("do_fetch_ohlc", e)
        return status_line(f"Error: {e}", C["red"]), make_cache_block()


# ── Callback: Run Analysis ─────────────────────────────────────────────────────
@app.callback(
    Output("dashboard-output", "children"),
    Output("run-status",       "children"),
    Input("run-btn",           "n_clicks"),
    State("top-per-file",      "value"),
    State("top-n",             "value"),
    State("max-weight",        "value"),
    prevent_initial_call=True,
)
def run_analysis(n_clicks, top_per_file, top_n, max_weight_pct):
    try:
        return _run_analysis_inner(top_per_file, top_n, max_weight_pct)
    except Exception as e:
        log_error("run_analysis", e)
        return error_banner("run_analysis", e), f"Error: {e}"


def _run_analysis_inner(top_per_file, top_n, max_weight_pct):
    if UNIVERSE_DF.empty:
        return html.Div("No universe loaded.",
                        style=dict(color=C["red"], padding="2rem")), ""

    closes_df = load_closes_cache()
    ohlc_df   = load_ohlc_cache()

    if closes_df.empty:
        return html.Div(
            [html.Div("No close data cached.", style=dict(color=C["amber"])),
             html.Div("Click  Fetch Closes (5y)  first.",
                      style=dict(color=C["muted"], fontSize="12px",
                                 marginTop="4px"))],
            style=dict(padding="2rem")), ""

    universe = (UNIVERSE_DF
                .sort_values("mktcap", ascending=False)
                .groupby("sector", group_keys=False)
                .head(top_per_file)
                .drop_duplicates(subset="ticker")
                .reset_index(drop=True))

    tickers    = universe["ticker"].tolist()
    max_weight = max_weight_pct / 100.0
    meta       = universe.set_index("ticker").to_dict("index")

    # ── Step 1: Breadth history (bar-by-bar, fixed bands, full recovery logic) ──
    print("  Computing breadth history…")
    universe_closes = closes_df[[t for t in tickers if t in closes_df.columns]]
    breadth_df = compute_breadth_history(universe_closes)
    print(f"  Breadth history: {len(breadth_df)} trading days")

    # Extract dates where ceilings were reset (recovery events)
    if not breadth_df.empty:
        reset_dates = set(breadth_df.index[breadth_df["regime"] == "recovery"])
        print(f"  Ceiling reset events: {len(reset_dates)}")
    else:
        reset_dates = set()

    # ── Step 2: OHLC map for ADX ──
    ohlc_map: dict = {}
    if not ohlc_df.empty and "Ticker" in ohlc_df.columns:
        for t, grp in ohlc_df.groupby("Ticker"):
            g = grp.set_index("Date")[["High", "Low", "Close"]].sort_index()
            if len(g) >= ADX_PERIOD * 2:
                ohlc_map[t] = g

    # ── Step 3: Per-ticker analysis with reset dates passed in ──
    results, failed = [], []
    for t in tickers:
        if t not in closes_df.columns:
            failed.append(t)
            continue
        r = analyse_ticker(t, closes_df[t], ohlc_map.get(t),
                           ceiling_reset_dates=list(reset_dates))
        if r:
            m = meta.get(t, {})
            r["company"]  = m.get("company", "")
            r["sector"]   = m.get("sector", "")
            r["industry"] = m.get("industry", "")
            r["mktcap"]   = m.get("mktcap", 0)
            results.append(r)
        else:
            failed.append(t)

    print(f"  analysed={len(results)}  failed={len(failed)}")

    if not results:
        return html.Div(
            [html.Div("No tickers had sufficient price history.",
                      style=dict(color=C["muted"])),
             html.Div(f"Failed tickers ({len(failed)}): "
                      f"{', '.join(failed[:20])}{'…' if len(failed)>20 else ''}",
                      style=dict(fontSize="11px", color=C["muted"],
                                 marginTop="6px"))],
            style=dict(padding="2rem")), ""

    # ── Step 4: Portfolio construction using breadth regime ──
    portfolio = run_portfolio(results, breadth_df, top_n, max_weight)
    bf        = portfolio["bottom_frac"]
    roff      = portfolio["risk_off"]
    regime    = portfolio["regime"]
    improve   = portfolio["improvement"]
    max_stress= portfolio["max_stress"]
    positions = portfolio["positions"]
    final_w   = portfolio["final_weights"]
    all_idx   = [r["idx"] for r in results]

    regime_color = (C["red"]    if roff             else
                    C["amber"]  if regime=="recovery" else
                    C["amber"]  if bf > 0.30          else C["green"])
    regime_label = ("Risk-Off — liquidated"         if roff              else
                    f"Recovery — improvement {improve*100:.0f}%"
                                                     if regime=="recovery" else
                    "Caution — elevated stress"      if bf > 0.30         else
                    "Risk-On — fully invested")

    avg_mom     = float(np.mean([r["mom"] for r in positions])) if positions else 0.0
    exhausted_n = sum(1 for r in positions if r["exhausted"])

    # ── Breadth history chart ──
    fig_breadth = go.Figure()
    if not breadth_df.empty:
        bd = breadth_df.reset_index()

        # Shade risk-off periods
        in_roff   = False
        roff_start = None
        shapes = []
        for _, row in bd.iterrows():
            if row["regime"] == "risk_off" and not in_roff:
                in_roff    = True
                roff_start = row["date"]
            elif row["regime"] != "risk_off" and in_roff:
                shapes.append(dict(
                    type="rect", xref="x", yref="paper",
                    x0=roff_start, x1=row["date"], y0=0, y1=1,
                    fillcolor=C["red"], opacity=0.12, line_width=0,
                ))
                in_roff = False
        if in_roff:
            shapes.append(dict(
                type="rect", xref="x", yref="paper",
                x0=roff_start, x1=bd["date"].iloc[-1], y0=0, y1=1,
                fillcolor=C["red"], opacity=0.12, line_width=0,
            ))

        # Recovery markers
        rec_dates = bd[bd["regime"] == "recovery"]["date"]
        if not rec_dates.empty:
            fig_breadth.add_trace(go.Scatter(
                x=rec_dates,
                y=breadth_df.loc[rec_dates, "bottom_frac"] if len(rec_dates) else [],
                mode="markers",
                marker=dict(symbol="triangle-up", size=10,
                            color=C["green"], line=dict(width=1, color=C["text"])),
                name="Ceiling reset",
                hovertemplate="Recovery / ceiling reset<br>%{x}<extra></extra>",
            ))

        # bottom_frac line
        fig_breadth.add_trace(go.Scatter(
            x=bd["date"], y=bd["bottom_frac"] * 100,
            mode="lines", name="Bottom-band %",
            line=dict(color=C["blue"], width=1.5),
            hovertemplate="%{x}<br>Stress: %{y:.1f}%<extra></extra>",
        ))

        # Threshold lines
        fig_breadth.add_hline(y=45, line=dict(color=C["red"],   dash="dot", width=1),
                              annotation_text="Risk-off (45%)",
                              annotation_font=dict(color=C["red"], size=10))
        fig_breadth.add_hline(y=15, line=dict(color=C["green"], dash="dot", width=1),
                              annotation_text="Recovery floor (15%)",
                              annotation_font=dict(color=C["green"], size=10))
        fig_breadth.add_hline(y=30, line=dict(color=C["amber"], dash="dot", width=1),
                              annotation_text="Caution (30%)",
                              annotation_font=dict(color=C["amber"], size=10))

        fig_breadth.update_layout(**layout(
            height=280,
            title=dict(text="Universe breadth stress — bottom-band fraction (fixed bands)",
                       font=dict(size=13), x=0),
            yaxis_title="% in bands 0–4",
            yaxis=dict(gridcolor=C["grid"], linecolor=C["border"],
                       tickcolor=C["border"], zerolinecolor=C["border"],
                       ticksuffix="%", range=[0, max(55, bd["bottom_frac"].max()*110)]),
            shapes=shapes,
            legend=dict(orientation="h", y=1.08, x=0,
                        font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
        ))
    else:
        fig_breadth.update_layout(**layout(height=280,
            title="Breadth history unavailable",
            annotations=[dict(text="Need ≥50 tickers with ≥200 bars",
                              x=.5, y=.5, showarrow=False,
                              font=dict(color=C["muted"], size=13))]))

    # ── Weight bar ──
    if positions and not roff:
        t_s   = sorted(final_w, key=final_w.get)
        w_s   = [round(final_w[t] * 100, 1) for t in t_s]
        ex    = {r["ticker"]: r["exhausted"] for r in results}
        col_w = [C["red"] if ex.get(t) else C["blue"] for t in t_s]
        fig_w = go.Figure(go.Bar(
            x=w_s, y=t_s, orientation="h", marker_color=col_w,
            text=[f"{v:.1f}%" for v in w_s], textposition="outside",
            textfont=dict(size=11, color=C["text"]),
        ))
        fig_w.update_layout(**layout(
            title=dict(text="Final portfolio weights", font=dict(size=13), x=0),
            xaxis_title="Weight %",
            height=max(200, len(t_s) * 36 + 80),
            margin=dict(l=12, r=60, t=40, b=12)))
        fig_w.update_xaxes(range=[0, max(w_s) * 1.3])
    else:
        fig_w = go.Figure()
        fig_w.update_layout(**layout(height=150,
            title="Portfolio liquidated (risk-off)",
            annotations=[dict(text="No positions held", x=.5, y=.5,
                              showarrow=False,
                              font=dict(color=C["muted"], size=14))]))

    # ── Band distribution ──
    counts = [sum(1 for i in all_idx if i == b) for b in range(12)]
    b_col  = [C["red"] if b in BOTTOM_LEVELS else
              (C["amber"] if b >= 10 else C["teal"]) for b in range(12)]
    fig_b  = go.Figure(go.Bar(
        x=list(range(12)), y=counts, marker_color=b_col,
        text=counts, textposition="outside",
        textfont=dict(size=10, color=C["text"]),
    ))
    fig_b.update_layout(**layout(height=250,
        title=dict(text="Universe band distribution (sizing bands)", font=dict(size=13), x=0),
        xaxis_title="Band index  (red=stress 0-4, amber=extended 10-11)",
        yaxis_title="# stocks"))

    # ── Momentum vs scale scatter ──
    in_top = {r["ticker"] for r in positions}
    fig_s  = go.Figure(go.Scatter(
        x=[r["mom"] * 100 for r in results],
        y=[r["scale"] * 100 for r in results],
        mode="markers+text",
        text=[r["ticker"] for r in results],
        textposition="top center",
        textfont=dict(size=8, color=C["muted"]),
        marker=dict(
            color=[C["green"] if r["ticker"] in in_top else C["muted"]
                   for r in results],
            size=[10 if r["ticker"] in in_top else 5 for r in results],
            opacity=0.85, line=dict(width=.5, color=C["border"])),
        customdata=[(r["ticker"], r.get("sector",""),
                     round(r["scale"]*100, 1)) for r in results],
        hovertemplate=(
            "<b>%{customdata[0]}</b>  %{customdata[1]}<br>"
            "Momentum: %{x:.2f}%<br>Scale: %{customdata[2]:.1f}%<extra></extra>"),
    ))
    fig_s.update_layout(**layout(height=320,
        title=dict(text="Momentum vs scale  (green = selected)",
                   font=dict(size=13), x=0),
        xaxis_title="Momentum (%)", yaxis_title="Scale (%)",
        shapes=[dict(type="line", x0=0, x1=0, y0=0, y1=100,
                     line=dict(color=C["border"], dash="dot"))]))

    # ── Sector donut ──
    sec_w: dict = {}
    for r in positions:
        s = r.get("sector") or "Unknown"
        sec_w[s] = sec_w.get(s, 0) + final_w.get(r["ticker"], 0) * 100
    if sec_w:
        fig_sec = go.Figure(go.Pie(
            labels=list(sec_w.keys()),
            values=[round(v, 1) for v in sec_w.values()],
            hole=0.5,
            marker_colors=[C["blue"], C["teal"], C["purple"], C["amber"],
                           C["green"], C["red"], C["muted"]],
            textinfo="label+percent", textfont=dict(size=11),
        ))
        fig_sec.update_layout(**layout(height=300,
            title=dict(text="Sector allocation", font=dict(size=13), x=0),
            showlegend=False))
    else:
        fig_sec = go.Figure()
        fig_sec.update_layout(**layout(height=300,
            title="Sector allocation",
            annotations=[dict(text="No positions", x=.5, y=.5,
                              showarrow=False,
                              font=dict(color=C["muted"]))]))

    # ── Band heatmap ──
    fig_heat = None
    if positions:
        z_data, y_lab = [], []
        for r in positions[:12]:
            h = r.get("band_idx_hist", [])
            if h:
                z_data.append(h[-40:])
                y_lab.append(r["ticker"])
        if z_data:
            ml    = max(len(row) for row in z_data)
            z_pad = [row + [None] * (ml - len(row)) for row in z_data]
            fig_heat = go.Figure(go.Heatmap(
                z=z_pad, y=y_lab,
                colorscale=[[0, C["green"]], [0.4, C["teal"]],
                             [0.75, C["amber"]], [1, C["red"]]],
                zmin=0, zmax=11,
                colorbar=dict(title="Band", tickfont=dict(size=10), len=0.8),
            ))
            fig_heat.update_layout(**layout(height=300,
                title=dict(text="Band index history — top holdings (last 40 periods)",
                           font=dict(size=13), x=0),
                xaxis_title="<- older  |  recent ->",
                margin=dict(l=12, r=70, t=40, b=12)))

    # ── Universe table ──
    rows = []
    for r in sorted(results, key=lambda x: final_w.get(x["ticker"], 0),
                    reverse=True):
        w  = final_w.get(r["ticker"], 0)
        mc = r.get("mktcap", 0) or 0
        rows.append({
            "Ticker":    r["ticker"],
            "Company":   r.get("company", ""),
            "Sector":    r.get("sector", ""),
            "Industry":  r.get("industry", ""),
            "Price":     f"${r['last_close']:.2f}",
            "MktCap":    (f"${mc/1e12:.2f}T" if mc >= 1e12 else
                          f"${mc/1e9:.1f}B"  if mc >= 1e9  else
                          f"${mc/1e6:.0f}M"),
            "Momentum":  f"{r['mom']*100:+.2f}%",
            "EMA":       f"${r['ema_val']:.2f}",
            "Band":      f"{r['idx']} / {r['hist_high']}",
            "Scale":     f"{r['scale']*100:.0f}%",
            "ADX":       f"{r['adx']:.1f}",
            "Weight":    f"{w*100:.1f}%" if w > 0 else "—",
            "Status":    ("exhausted" if r["exhausted"] else
                          "ceiling"   if r["scale"] < 0.5 else "clear"),
            "Above EMA": "yes" if r["above_ema"] else "no",
        })

    cond = [
        {"if": {"filter_query": '{Momentum} contains "+"',
                "column_id": "Momentum"}, "color": C["green"]},
        {"if": {"filter_query": '{Momentum} contains "-"',
                "column_id": "Momentum"}, "color": C["red"]},
        {"if": {"filter_query": '{Status} = "exhausted"',
                "column_id": "Status"},   "color": C["red"]},
        {"if": {"filter_query": '{Status} = "ceiling"',
                "column_id": "Status"},   "color": C["amber"]},
        {"if": {"filter_query": '{Status} = "clear"',
                "column_id": "Status"},   "color": C["green"]},
        {"if": {"filter_query": '{Above EMA} = "yes"',
                "column_id": "Above EMA"}, "color": C["green"]},
        {"if": {"filter_query": '{Above EMA} = "no"',
                "column_id": "Above EMA"}, "color": C["muted"]},
        {"if": {"filter_query": '{Weight} != "—"',
                "column_id": "Weight"}, "color": C["blue"],
         "fontWeight": "600"},
    ]

    tbl = dash_table.DataTable(
        data=rows,
        columns=[{"name": c, "id": c} for c in rows[0].keys()],
        style_table={"overflowX": "auto",
                     "border": f"1px solid {C['border']}",
                     "borderRadius": "8px"},
        style_cell=dict(background=C["card"], color=C["text"],
                        border=f"1px solid {C['border']}",
                        padding="8px 14px", fontSize="12px",
                        fontFamily="inherit", textAlign="left",
                        whiteSpace="nowrap", overflow="hidden",
                        textOverflow="ellipsis", maxWidth="200px"),
        style_header=dict(background=C["surface"], color=C["muted"],
                          fontWeight="500", fontSize="11px",
                          textTransform="uppercase", letterSpacing=".05em",
                          border=f"1px solid {C['border']}"),
        style_data_conditional=cond,
        sort_action="native", filter_action="native", page_size=30,
        tooltip_data=[
            {c: {"value": str(r.get(c, "")), "type": "markdown"}
             for c in ["Company", "Sector", "Industry"]}
            for r in rows
        ],
        tooltip_duration=None,
        style_filter=dict(background=C["surface"], color=C["text"],
                          border=f"1px solid {C['border']}"),
    )

    n_resets = len(reset_dates)
    status_msg = (f"{len(results)} analysed  ·  "
                  f"{len(failed)} skipped  ·  "
                  f"{len(positions)} positions  ·  "
                  f"{n_resets} ceiling reset(s)")

    return html.Div([

        # ── Regime banner ──
        html.Div([
            html.Div(style=dict(width="10px", height="10px", borderRadius="50%",
                                background=regime_color, flexShrink="0")),
            html.Span(regime_label,
                      style=dict(fontWeight="600", fontSize="14px")),
            html.Div([
                html.Span(f"Breadth stress: {bf*100:.1f}%",
                          style=dict(fontSize="12px", color=C["muted"])),
                html.Span(" · ", style=dict(color=C["border"], padding="0 6px")),
                html.Span(f"Peak stress: {max_stress*100:.1f}%",
                          style=dict(fontSize="12px", color=C["muted"])),
                html.Span(" · ", style=dict(color=C["border"], padding="0 6px")),
                html.Span(f"Ceiling resets: {n_resets}",
                          style=dict(fontSize="12px", color=C["muted"])),
                html.Span(" · ", style=dict(color=C["border"], padding="0 6px")),
                html.Span("Risk-off ≥45%  ·  Recovery: improvement ≥60% or stress <15%",
                          style=dict(fontSize="12px", color=C["muted"])),
            ], style=dict(marginLeft="auto", display="flex",
                          alignItems="center", flexWrap="wrap", gap="2px")),
        ], style=dict(display="flex", alignItems="center", gap="12px",
                      background=regime_color+"18",
                      border=f"1px solid {regime_color}44",
                      borderRadius="10px", padding="14px 20px",
                      marginBottom="18px")),

        # ── Summary metrics ──
        card([
            html.Div("Summary", style=dict(
                fontSize="11px", color=C["muted"], textTransform="uppercase",
                letterSpacing=".07em", marginBottom="14px")),
            html.Div([
                metric_box("Universe", str(len(results)), "tickers with history"),
                metric_box("Sectors",  str(universe["sector"].nunique()), "loaded"),
                metric_box("Positions",
                           "0" if roff else str(len(positions)),
                           f"of {top_n} target",
                           C["red"] if roff else None),
                metric_box("Avg momentum",
                           f"{avg_mom*100:.1f}%", "5-period composite",
                           C["green"] if avg_mom > 0 else C["red"]),
                metric_box("Exhaustion flags", str(exhausted_n),
                           f"of {len(positions)} picks",
                           C["red"] if exhausted_n > 0 else None),
                metric_box("Breadth stress", f"{bf*100:.1f}%",
                           "bottom-band fraction",
                           C["red"] if roff else
                           (C["amber"] if bf > 0.3 else C["green"])),
                metric_box("Ceiling resets", str(n_resets),
                           "recovery events in history",
                           C["amber"] if n_resets > 0 else None),
            ], style=dict(display="flex", gap="10px", flexWrap="wrap")),
        ]),

        # ── Breadth history chart (full width) ──
        card([
            html.Div("Breadth History", style=dict(
                fontSize="11px", color=C["muted"], textTransform="uppercase",
                letterSpacing=".07em", marginBottom="8px")),
            dcc.Graph(figure=fig_breadth,
                      config=dict(displayModeBar=False),
                      style=dict(width="100%")),
        ]),

        flex_row(
            chart_card(fig_w,
                       height=max(200, len(positions)*36+80) if positions else 150),
            chart_card(fig_b, height=250),
        ),
        flex_row(
            chart_card(fig_s,   height=320),
            chart_card(fig_sec, height=300),
        ),
        (flex_row(chart_card(fig_heat, height=300)) if fig_heat else html.Div()),

        card([
            html.Div([
                html.Span("Full universe scan", style=dict(
                    fontSize="11px", color=C["muted"],
                    textTransform="uppercase", letterSpacing=".07em")),
                html.Span(f"  ·  {len(results)} stocks",
                          style=dict(fontSize="11px", color=C["muted"])),
            ], style=dict(marginBottom="14px")),
            tbl,
        ]),

    ]), status_msg


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
