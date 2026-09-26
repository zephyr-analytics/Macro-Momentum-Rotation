"""
Momentum & Historical Band Ceiling Dashboard  —  QC-faithful replay
Plotly Dash + yfinance

This dashboard no longer approximates StockOnlyMomentum with daily,
per-ticker calculations. It REPLAYS the QC algorithm bar by bar from the
algorithm's warm-up start, with the same state machine, the same update
order, and the same arithmetic.

Data flow
---------
1. Sector Excel/CSV files in stock_files/ build the universe (SectorTopUniverse
   filters applied: mcap >= $5B, NYSE/NASDAQ/AMEX, blacklist, top-N per sector).
2. "Fetch Prices" pulls daily High/Low/Close/Adj Close/Dividends (unadjusted
   for dividends, split-adjusted) from (algo start - 3y) to today, plus SPY
   for the trading calendar  ->  cache/prices.pkl
3. "Run Analysis" replays the algorithm and shows:
     - CURRENT holdings  = targets from the last month-end Rebalance QC ran
     - NEXT-REBALANCE PREVIEW = Rebalance() evaluated on a copy of the state
       using data through the latest bar
   and writes cache/rebalance_log.csv for line-by-line validation vs QC Debug.

Logic replicated exactly
------------------------
 * Indicator lifetimes: ma (EMA-189), ADX-14, stretch_ema, close_win and
   stretch_max all start at the symbol's subscription start = warm-up start
   (algo start - 300 bars) or its first bar if later.
 * Lean EMA semantics (verified against Lean source): Current = 0 until n
   samples, first value = SMA of the first n samples, then k = 2/(n+1).
 * Lean ADX (verified against Lean source), including its seeding, +DM tie
   rule, and "ADX = 50 when +DI + -DI == 0" behaviour.
 * dev = np.std (population, ddof=0) of the last 189 closes INCLUDING the
   current bar (close_win.Add runs before np.std in OnData).
 * stretch_ema fed once per bar (stretch only) — confirmed to match QC output.
 * Breadth: current_band_idx carried forward per symbol, fixed golden-ratio
   bands, and _band_index's fall-through (price outside the ladder -> 11).
 * Rebalance timing: month_end("SPY"), 120 min before close. With daily
   bars the month-end bar has not arrived yet, so ALL Rebalance inputs
   (breadth, History(), Securities[].Price, indicators) are as of the
   PREVIOUS trading day's close.
 * Regime state machine (allow_universe / was_risk_off / max_stress_level)
   evaluated only on rebalance days, exactly as in Rebalance().
 * `len(idxs) < 50` early return leaves holdings untouched.
 * Momentum from a 253-bar History() window. In TotalReturn mode a History()
   request accumulates dividends from the start of its own window.
 * band_hist: RollingWindow(126) of MONTHLY entries, appended only for `top`
   symbols with ready ma/stretch_ema and dev > 0, BEFORE historical_high is
   taken (so a new high always scales to 0).
 * ADX gate, EMA gate, raw-momentum ranking, scale, exhaustion override,
   proportional weights, single-pass cap + renormalisation, w > 0 holdings.
 * Holdings are frozen between rebalances (QC never trades intra-month).

Cannot be replicated from yfinance + static files (data, not logic)
-------------------------------------------------------------------
 * Point-in-time universe: QC re-selects daily from Morningstar fundamentals
   (sector codes, market cap, price > $5 on that day). The files here are a
   static, present-day snapshot (survivorship bias; sector taxonomy may be
   GICS not Morningstar). Symbols leaving/re-entering QC's universe lose
   their indicator state; here they never leave.
 * Price vendor: QC (AlgoSeek/Morningstar factor files) vs Yahoo.
 * TotalReturn dividends: reconstructed as split-adjusted close + cumulative
   split-adjusted dividends from subscription start.

Optional: stretch_ema double-feed
---------------------------------
Reading Lean's source, self.EMA(symbol, ...) registers the indicator for
automatic close updates, so the manual .Update(self.Time, stretch) in OnData
would give it two inputs per bar. Comparing against QC backtest output showed
QC does NOT behave that way, so this is OFF by default. The checkbox is kept
only for experiments.
"""

import copy, glob, json, os, re, time, traceback, warnings
warnings.filterwarnings("ignore")
from collections import deque
from typing import Optional

import dash
from dash import dcc, html, dash_table, Input, Output, State
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import yfinance as yf
from numpy.lib.stride_tricks import sliding_window_view
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

# ── QC parameters (mirror StockOnlyMomentum.Initialize) ───────────────────────
LOOKBACKS          = [21, 63, 126, 189, 252]
STOCK_COUNT        = 10
MAX_WEIGHT         = 0.20
BAND_LEN           = 189
HIST_LEN           = 126
ADX_PERIOD         = 14
ADX_LIMIT          = 35
WARMUP_BARS        = 300
BOTTOM_LEVELS      = {0, 1, 2, 3, 4}
BREADTH_FRACTIONS  = [1.618, 1.382, 1.0, 0.809, 0.5, 0.382]
CEILING_FRACTIONS  = [1.618, 1.382, 1.0, 0.75, 0.5, 0.190983]
MIN_BREADTH_SAMPLE = 50
RISK_OFF_FRAC      = 0.45
RECOVERY_IMPROVE   = 0.60
RECOVERY_FLOOR     = 0.15
STRESS_DENOM_FLOOR = 0.10
EXHAUST_IDX        = 10
EXHAUST_DECAY      = 0.80

# ── Universe parameters (mirror SectorTopUniverse) ────────────────────────────
TOP_PER_SECTOR = 100
MIN_MKTCAP     = 5_000_000_000
MIN_PRICE      = 5.0
BLACKLIST      = {"GME", "AMC"}

# ── Replay / data parameters ──────────────────────────────────────────────────
DEFAULT_ALGO_START = "2004-01-01"   # the commented-out SetStartDate in the algo
CALENDAR_TICKER    = "SPY"          # DateRules.month_end("SPY")
HISTORY_PAD_YEARS  = 3              # 300-bar warm-up + 253-bar History() window
CHUNK_SIZE         = 50
SLEEP_S            = 2.0
PRICE_FIELDS       = ("High", "Low", "Close", "Adj Close", "Dividends")

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
STOCK_FILES_DIR = os.path.join(BASE_DIR, "stock_files")
CACHE_DIR       = os.path.join(BASE_DIR, "cache")
PRICES_PKL      = os.path.join(CACHE_DIR, "prices.pkl")
PRICES_META     = os.path.join(CACHE_DIR, "prices_meta.json")
REBAL_LOG_CSV   = os.path.join(CACHE_DIR, "rebalance_log.csv")
ERROR_LOG       = os.path.join(CACHE_DIR, "errors.log")

os.makedirs(CACHE_DIR, exist_ok=True)


def log_error(context: str, exc: Exception) -> None:
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
    return {**BASE_LAYOUT, **overrides}


# ══════════════════════════════════════════════════════════════════════════════
# Universe (SectorTopUniverse equivalent, from static files)
# ══════════════════════════════════════════════════════════════════════════════
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


def _exchange_ok(val) -> bool:
    """QC: primary_exchange_id in ('NYS', 'NAS', 'ASE'). Unknown -> keep."""
    s = str(val).strip().upper()
    if not s or s == "NAN":
        return True
    if "ARCA" in s or "BATS" in s or "CBOE" in s or "OTC" in s:
        return False
    keys = ("NYSE", "NASDAQ", "AMEX", "NYS", "NAS", "ASE", "NMS",
            "NGM", "NCM", "NGS", "NYQ", "AMERICAN", "MKT")
    return any(k in s for k in keys)


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

    # Share classes (BRK.B, BF/B) are in QC's universe; yfinance wants BRK-B.
    df["ticker"] = (df["ticker"].astype(str).str.strip().str.upper()
                    .str.replace(r"[./]", "-", regex=True))
    df = df[df["ticker"].str.match(r"^[A-Z]{1,6}(-[A-Z]{1,2})?$")].copy()
    df["mktcap"] = df["mktcap_raw"].apply(parse_mktcap) if "mktcap_raw" in df.columns else 0.0

    if "sector" not in df.columns or df["sector"].isna().all():
        df["sector"] = re.sub(r"[_\-]", " ", filename.rsplit(".", 1)[0]).title()

    for col in ["company", "industry", "sub_industry", "exchange"]:
        if col not in df.columns:
            df[col] = ""

    return df[["ticker", "company", "sector", "industry",
               "sub_industry", "exchange", "mktcap"]].copy()


def load_universe(top_n: int = TOP_PER_SECTOR):
    patterns = [os.path.join(STOCK_FILES_DIR, ext) for ext in ("*.xlsx", "*.xls", "*.csv")]
    paths = sorted(p for pat in patterns for p in glob.glob(pat))

    log = []
    if not paths:
        log.append("No files found in stock_files/")
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
    n0 = len(combined)
    # SectorTopUniverse filters (price > $5 is applied at run time on price data)
    combined = combined[combined["mktcap"] >= MIN_MKTCAP]
    combined = combined[~combined["ticker"].isin(BLACKLIST)]
    combined = combined[combined["exchange"].apply(_exchange_ok)]
    top = (combined
           .sort_values("mktcap", ascending=False)
           .groupby("sector", group_keys=False)
           .head(top_n)
           .sort_values("mktcap", ascending=False)
           .drop_duplicates(subset="ticker")
           .reset_index(drop=True))

    log.append(f"  -> {len(top)} tickers in universe (of {n0}; mcap >= $5B, "
               f"NYSE/NASDAQ/AMEX, blacklist, top {top_n}/sector, deduped)")
    return top, log


# ══════════════════════════════════════════════════════════════════════════════
# yfinance fetch
# ══════════════════════════════════════════════════════════════════════════════
def _extract_field(raw: pd.DataFrame, ticker: str, field: str) -> Optional[pd.Series]:
    if raw is None or raw.empty:
        return None
    cols = raw.columns
    if isinstance(cols, pd.MultiIndex):
        price_fields = {"Open", "High", "Low", "Close", "Adj Close", "Volume",
                        "Dividends", "Stock Splits", "Capital Gains"}
        lvl0_is_price = any(str(v) in price_fields for v in cols.get_level_values(0))
        if lvl0_is_price:
            try:
                return raw[field][ticker].dropna()
            except KeyError:
                pass
            for t in cols.get_level_values(1).unique():
                if str(t).upper() == ticker.upper():
                    try:
                        return raw[field][t].dropna()
                    except KeyError:
                        pass
        else:
            for t in cols.get_level_values(0).unique():
                if str(t).upper() == ticker.upper():
                    try:
                        return raw[t][field].dropna()
                    except KeyError:
                        pass
    if field in raw.columns:
        return raw[field].dropna()
    return None


def _clean_index(s: pd.Series) -> pd.Series:
    idx = pd.to_datetime(s.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    s = s.copy()
    s.index = idx.normalize()
    return s[~s.index.duplicated(keep="last")]


def fetch_prices(tickers: list, algo_start: str) -> dict:
    start = (pd.Timestamp(algo_start)
             - pd.DateOffset(years=HISTORY_PAD_YEARS)).strftime("%Y-%m-%d")
    universe = [CALENDAR_TICKER] + [t for t in tickers if t != CALENDAR_TICKER]
    print(f"\nFetching daily prices from {start} for {len(universe)} tickers "
          f"in chunks of {CHUNK_SIZE} …")
    chunks = [universe[i:i + CHUNK_SIZE] for i in range(0, len(universe), CHUNK_SIZE)]
    cols = {f: {} for f in PRICE_FIELDS}

    for i, chunk in enumerate(chunks):
        print(f"  chunk {i+1}/{len(chunks)}: {chunk[:3]}{'…' if len(chunk) > 3 else ''}")
        try:
            raw = yf.download(chunk, start=start, interval="1d",
                              auto_adjust=False, actions=True,
                              progress=False, group_by="ticker")
            if raw is None or raw.empty:
                print("    no data returned")
                continue
            for t in chunk:
                try:
                    close = _extract_field(raw, t, "Close")
                    if close is None or len(close) < 50:
                        continue
                    close = _clean_index(close)
                    cols["Close"][t] = close
                    for f in ("High", "Low", "Adj Close", "Dividends"):
                        s = _extract_field(raw, t, f)
                        if s is None:
                            s = (pd.Series(0.0, index=close.index) if f == "Dividends"
                                 else close.copy())
                        cols[f][t] = _clean_index(s)
                except Exception as e:
                    print(f"    {t}: {e}")
        except Exception as e:
            log_error(f"fetch_prices chunk {i+1}", e)
        if i < len(chunks) - 1:
            time.sleep(SLEEP_S)

    if not cols["Close"]:
        print("  No price data collected — check errors.log")
        return {}

    fields = {f: pd.DataFrame(v).sort_index() for f, v in cols.items()}
    payload = dict(fields=fields, start=start, algo_start=algo_start,
                   fetched=str(pd.Timestamp.now()))
    pd.to_pickle(payload, PRICES_PKL)
    close = fields["Close"]
    meta = dict(start=start, algo_start=algo_start,
                first=str(close.index.min().date()), last=str(close.index.max().date()),
                tickers=int(close.shape[1]), rows=int(close.shape[0]),
                fetched=payload["fetched"][:16])
    with open(PRICES_META, "w") as f:
        json.dump(meta, f)
    print(f"  Saved {close.shape[1]} tickers × {close.shape[0]} rows → {PRICES_PKL}")
    return payload


def load_prices_cache() -> Optional[dict]:
    if not os.path.exists(PRICES_PKL):
        return None
    try:
        return pd.read_pickle(PRICES_PKL)
    except Exception as e:
        log_error("load_prices_cache", e)
        return None


def cache_meta() -> Optional[dict]:
    if not os.path.exists(PRICES_META):
        return None
    try:
        with open(PRICES_META) as f:
            return json.load(f)
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# QC-equivalent primitives
# ══════════════════════════════════════════════════════════════════════════════
_BF = np.array(BREADTH_FRACTIONS, dtype=float)


def _build_bands(mid, dev, unit=1.0, fractions=BREADTH_FRACTIONS):
    """Identical arithmetic to StockOnlyMomentum._build_bands."""
    lower = [mid - dev * unit * f for f in fractions]
    upper = [mid + dev * unit * f for f in reversed(fractions)]
    return lower + [mid] + upper


def _band_index(price, bands):
    """Identical to StockOnlyMomentum._band_index (incl. fall-through -> 11)."""
    for i in range(len(bands) - 1):
        if bands[i] <= price < bands[i + 1]:
            return i
    return len(bands) - 2


class _LeanEMA:
    """Port of Lean ExponentialMovingAverage: Current = 0 until `period`
    samples, first ready value = SMA of the first `period` samples, then
    x*k + prev*(1-k) with k = 2/(period+1)."""
    __slots__ = ("p", "k", "n", "s", "v")

    def __init__(self, period):
        self.p = period; self.k = 2.0 / (period + 1)
        self.n = 0; self.s = 0.0; self.v = 0.0

    def update(self, x):
        self.n += 1
        if self.n <= self.p:
            self.s += x
        if self.n < self.p:
            self.v = 0.0
        elif self.n == self.p:
            self.v = self.s / self.p
        else:
            self.v = x * self.k + self.v * (1.0 - self.k)
        return self.v


def _ticker_indicators(c: np.ndarray, quirk: bool) -> dict:
    """
    Replays OnData + the auto-updated indicators for ONE symbol from its
    subscription start. Consolidator (auto) updates fire before OnData, so on
    each bar: ma <- close; [quirk] stretch_ema <- close; then OnData:
    close_win.Add, dev = np.std(close_win), stretch, stretch_ema <- stretch,
    stretch_max, current_band_idx.
    """
    n = len(c)
    dev = np.full(n, np.nan)
    if n >= BAND_LEN:
        # list(RollingWindow) is newest-first; keep that order for np.std
        dev[BAND_LEN - 1:] = sliding_window_view(c, BAND_LEN)[:, ::-1].std(axis=1)

    ma      = np.empty(n)
    sema    = np.full(n, np.nan)
    sema_n  = np.zeros(n, dtype=np.int64)
    stretch = np.full(n, np.nan)

    cl, dl = c.tolist(), dev.tolist()
    e_ma, e_st = _LeanEMA(BAND_LEN), _LeanEMA(BAND_LEN)
    for i in range(n):
        x = cl[i]
        m = e_ma.update(x)
        ma[i] = m
        if quirk:                                        # auto-feed of close
            e_st.update(x)
        if i >= BAND_LEN - 1:                            # close_win & ma ready
            d = dl[i]
            if d > 0:
                st = abs(x - m) / d
                stretch[i] = st
                e_st.update(st)                          # manual stretch update
        if e_st.n:
            sema[i] = e_st.v
        sema_n[i] = e_st.n

    smax = np.maximum.accumulate(np.nan_to_num(stretch, nan=0.0))

    # Breadth band index (fixed fractions, unit=1.0), carried forward
    valid = ~np.isnan(stretch)
    bidx = np.full(n, -1, dtype=np.int16)
    if valid.any():
        mv = ma[valid][:, None]
        dv = dev[valid][:, None]
        lower = mv - (dv * 1.0) * _BF[None, :]
        upper = mv + (dv * 1.0) * _BF[::-1][None, :]
        bands = np.hstack([lower, mv, upper])
        pos = (bands <= c[valid][:, None]).sum(axis=1) - 1
        pos = np.where((pos < 0) | (pos > 11), 11, pos)
        tmp = np.full(n, np.nan)
        tmp[valid] = pos
        bidx = pd.Series(tmp).ffill().fillna(-1).to_numpy().astype(np.int16)

    return dict(ma=ma, dev=dev, sema=sema, sema_n=sema_n, smax=smax, bidx=bidx)


def _adx_series(h: np.ndarray, l: np.ndarray, c: np.ndarray, period: int = ADX_PERIOD):
    """
    Line-for-line port of Lean AverageDirectionalIndex (Indicators/
    AverageDirectionalIndex.cs + WilderMovingAverage.cs), from subscription start.
      * bar 1: TR = +DM = -DM = 0 (no previous bar)
      * +DM = H-pH if H > pH and H-pH >= pL-L ; -DM = pL-L if pL > L and pL-L > H-pH
      * smoothed TR/DM: S = S + x - (S/period if samples > period+1 else 0)
        -> plain sum over bars 1..period+1, Wilder afterwards; ready when samples > period
      * +DI/-DI = 100*S_dm/S_tr once smoothed DM is ready and S_tr != 0, else 0
      * if +DI + -DI == 0: ADX.Current = 50 and the Wilder average is NOT updated
      * DX -> WilderMovingAverage(period): SMA while samples < period, then
        dx/period + prev*(1-1/period) (the period-th sample already uses Wilder)
      * IsReady = Wilder average has `period` samples
    """
    n = len(c)
    out = np.full(n, np.nan)
    ready = np.zeros(n, dtype=bool)
    hl, ll, cl = h.tolist(), l.tolist(), c.tolist()
    s_tr = s_p = s_m = 0.0
    w_n = 0; w_sum = 0.0; w_v = 0.0; kw = 1.0 / period
    cur = 0.0
    for i in range(n):
        samples = i + 1
        if i == 0:
            tr = pdm = mdm = 0.0
        else:
            H, L = hl[i], ll[i]
            pH, pL, pC = hl[i - 1], ll[i - 1], cl[i - 1]
            tr = max(H - L, abs(H - pC), abs(L - pC))
            pdm = (H - pH) if (H > pH and H - pH >= pL - L) else 0.0
            mdm = (pL - L) if (pL > L and pL - L > H - pH) else 0.0
        dec = samples > period + 1
        s_tr = s_tr + tr  - (s_tr / period if dec else 0.0)
        s_p  = s_p  + pdm - (s_p  / period if dec else 0.0)
        s_m  = s_m  + mdm - (s_m  / period if dec else 0.0)
        sm_ready = samples > period
        pdi = 100.0 * s_p / s_tr if (s_tr != 0 and sm_ready) else 0.0
        ndi = 100.0 * s_m / s_tr if (s_tr != 0 and sm_ready) else 0.0
        ssum = pdi + ndi
        if ssum == 0:
            cur = 50.0
        else:
            dx = 100.0 * abs(pdi - ndi) / ssum
            w_n += 1
            if w_n < period:
                w_sum += dx
                w_v = w_sum / w_n
            else:
                w_v = dx * kw + w_v * (1.0 - kw)
            cur = w_v
        out[i] = cur
        ready[i] = w_n >= period
    return out, ready


# ══════════════════════════════════════════════════════════════════════════════
# Panel build: per-symbol indicator replay + snapshots at rebalance inputs
# ══════════════════════════════════════════════════════════════════════════════
def build_panel(prices: dict, tickers: list, algo_start: str,
                total_return: bool, quirk: bool) -> dict:
    F = prices["fields"]
    close_df = F["Close"]
    notes = []

    if CALENDAR_TICKER in close_df.columns and close_df[CALENDAR_TICKER].notna().sum() > 0:
        cal = close_df[CALENDAR_TICKER].dropna().index
    else:
        cal = close_df.dropna(how="all").index
        notes.append("SPY missing from cache — calendar built from union of tickers.")
    cal = pd.DatetimeIndex(sorted(set(cal)))
    T = len(cal)

    tickers = [t for t in dict.fromkeys(tickers)
               if t in close_df.columns and t != CALENDAR_TICKER]
    last_raw = close_df[tickers].ffill().iloc[-1]
    dropped_px = [t for t in tickers if not (last_raw.get(t, np.nan) > MIN_PRICE)]
    if dropped_px:
        notes.append(f"{len(dropped_px)} ticker(s) dropped for price <= ${MIN_PRICE:.0f}.")
    tickers = [t for t in tickers if t not in set(dropped_px)]
    N = len(tickers)

    def mat(name):
        df = F.get(name)
        if df is None:
            return None
        return df.reindex(index=cal, columns=tickers).to_numpy(dtype=float)

    Cx = mat("Close"); Hx = mat("High"); Lx = mat("Low")
    Ax = mat("Adj Close"); Dx = mat("Dividends")
    if Ax is None: Ax = Cx.copy()
    if Dx is None: Dx = np.zeros_like(Cx)
    Dx = np.nan_to_num(Dx, nan=0.0)

    missing = np.isnan(Cx)
    first = np.full(N, -1, dtype=np.int64)
    last  = np.full(N, -1, dtype=np.int64)
    for j in range(N):
        v = np.flatnonzero(~missing[:, j])
        if len(v):
            first[j], last[j] = v[0], v[-1]

    rows = np.arange(T)[:, None]
    live = (first[None, :] >= 0) & (rows >= first[None, :]) & (rows <= last[None, :])
    # Lean fill-forward bars inside the live range: O = H = L = C = prior close
    Cff = pd.DataFrame(Cx).ffill().to_numpy()
    Aff = pd.DataFrame(Ax).ffill().to_numpy()
    ffbar = live & missing
    Cx = np.where(live, Cff, np.nan)
    Ax = np.where(live, Aff, np.nan)
    Hx = np.where(ffbar | np.isnan(Hx), Cx, Hx); Hx = np.where(live, Hx, np.nan)
    Lx = np.where(ffbar | np.isnan(Lx), Cx, Lx); Lx = np.where(live, Lx, np.nan)
    Dx = np.where(live & ~missing, Dx, 0.0)

    start_idx = int(cal.searchsorted(pd.Timestamp(algo_start)))
    if start_idx >= T:
        raise ValueError(f"Algo start {algo_start} is after the last cached bar.")
    warm_needed = start_idx - WARMUP_BARS
    warm_idx = max(0, warm_needed)
    if warm_needed < 0:
        notes.append(f"Price cache starts {-warm_needed} bars too late for the full "
                     f"300-bar warm-up — re-fetch with this algo start date.")
    fstart = np.maximum(first, warm_idx)

    # Month-end rebalance days (DateRules.month_end("SPY"))
    ym = cal.year.values * 12 + cal.month.values
    is_me = np.zeros(T, dtype=bool)
    is_me[:-1] = ym[1:] != ym[:-1]
    nxt = cal[-1] + CustomBusinessDay(calendar=USFederalHolidayCalendar())
    is_me[-1] = nxt.month != cal[-1].month
    reb_pos = [int(p) for p in np.flatnonzero(is_me) if p >= start_idx and p >= 1]

    snap_t = sorted(set([p - 1 for p in reb_pos] + [T - 1]))
    row_of = {t: r for r, t in enumerate(snap_t)}
    R = len(snap_t)
    snap_arr = np.array(snap_t)

    # Momentum snapshots (History(symbols, 253, Daily) ending at t)
    if total_return:
        CS = np.cumsum(Dx, axis=0)
        Cm = Cx
    else:
        CS = np.zeros_like(Cx)
        Cm = Ax
    max_lb = max(LOOKBACKS)
    MOM = np.full((R, N), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        for r, t in enumerate(snap_t):
            t0 = t - max_lb
            if t0 < 0:
                continue
            base = CS[t0]
            pt = Cm[t] + (CS[t] - base)
            rets = [pt / (Cm[t - lb] + (CS[t - lb] - base)) - 1 for lb in LOOKBACKS]
            MOM[r] = np.mean(rets, axis=0)

    snap = {k: np.full((R, N), np.nan) for k in ("price", "ma", "dev", "sema", "smax", "adx")}
    snapb = {k: np.zeros((R, N), dtype=bool)
             for k in ("fed", "ma_ready", "sema_ready", "adx_ready")}
    BIDX = np.full((T, N), -1, dtype=np.int16)

    t_start = time.time()
    for j in range(N):
        f, e = int(fstart[j]), int(last[j])
        if first[j] < 0 or f > e:
            continue
        seg = slice(f, e + 1)
        if total_return:
            off = CS[seg, j] - CS[f, j]
            c = Cx[seg, j] + off; h = Hx[seg, j] + off; l = Lx[seg, j] + off
        else:
            ratio = Ax[seg, j] / Cx[seg, j]
            c = Ax[seg, j]; h = Hx[seg, j] * ratio; l = Lx[seg, j] * ratio

        ind = _ticker_indicators(c, quirk)
        adx, adx_rdy = _adx_series(h, l, c)
        BIDX[seg, j] = ind["bidx"]

        m = (snap_arr >= f) & (snap_arr <= e)
        if m.any():
            rr = np.flatnonzero(m)
            li = snap_arr[m] - f
            snap["price"][rr, j] = c[li]
            snap["ma"][rr, j]    = ind["ma"][li]
            snap["dev"][rr, j]   = ind["dev"][li]
            snap["sema"][rr, j]  = ind["sema"][li]
            snap["smax"][rr, j]  = ind["smax"][li]
            snap["adx"][rr, j]   = adx[li]
            snapb["fed"][rr, j]        = True
            snapb["ma_ready"][rr, j]   = (li + 1) >= BAND_LEN
            snapb["sema_ready"][rr, j] = ind["sema_n"][li] >= BAND_LEN
            snapb["adx_ready"][rr, j]  = adx_rdy[li]
        if (j + 1) % 100 == 0:
            print(f"    indicators {j+1}/{N}  ({time.time() - t_start:.0f}s)")

    return dict(cal=cal, tickers=tickers, T=T, N=N, start_idx=start_idx,
                warm_idx=warm_idx, reb_pos=reb_pos, row_of=row_of, snap_t=snap_t,
                MOM=MOM, snap=snap, snapb=snapb, BIDX=BIDX, notes=notes)


# ══════════════════════════════════════════════════════════════════════════════
# Rebalance() — line-for-line port
# ══════════════════════════════════════════════════════════════════════════════
def qc_rebalance(state: dict, panel: dict, t: int,
                 stock_count: int, max_weight: float) -> dict:
    tickers = panel["tickers"]
    r = panel["row_of"][t]
    snap, snapb = panel["snap"], panel["snapb"]
    rec = dict(status="", regime="", bottom_frac=np.nan, n_breadth=0,
               max_stress=np.nan, improvement=np.nan, reset=False,
               candidates=[], final_weights={}, holdings={})

    # -------- UNIVERSE-WIDE BREADTH --------
    b = panel["BIDX"][t]
    vals = b[b >= 0]
    n = len(vals)
    rec["n_breadth"] = n
    if n < MIN_BREADTH_SAMPLE:
        rec["status"] = "breadth<50 (return)"
        rec["regime"] = "unchanged"
        return rec

    bottom_frac = int(((vals >= 0) & (vals <= 4)).sum()) / n
    state["max_stress"] = max(state["max_stress"], bottom_frac)
    rec["bottom_frac"] = bottom_frac

    # -------- BREADTH REGIME --------
    if bottom_frac >= RISK_OFF_FRAC:
        state["allow"] = False
        state["was_risk_off"] = True
    elif state["was_risk_off"]:
        denominator = max(state["max_stress"], STRESS_DENOM_FLOOR)
        improvement = (state["max_stress"] - bottom_frac) / denominator
        rec["improvement"] = improvement
        if improvement >= RECOVERY_IMPROVE or bottom_frac < RECOVERY_FLOOR:
            state["band_hist"] = {s: deque(maxlen=HIST_LEN) for s in state["band_hist"]}
            state["allow"] = True
            state["was_risk_off"] = False
            state["max_stress"] = 0.0
            rec["reset"] = True
    else:
        state["allow"] = True
    rec["max_stress"] = state["max_stress"]

    if not state["allow"]:
        rec["status"] = "risk_off (liquidate)"
        rec["regime"] = "risk_off"
        return rec
    rec["regime"] = "recovery" if rec["reset"] else "risk_on"

    # -------- MOMENTUM --------
    mom = panel["MOM"][r]
    adx = snap["adx"][r]; price = snap["price"][r]; ma = snap["ma"][r]
    with np.errstate(invalid="ignore"):
        elig = (snapb["fed"][r] & snapb["adx_ready"][r] & ~(adx > ADX_LIMIT)
                & snapb["ma_ready"][r] & ~(price <= ma) & (mom > 0))
    cand = np.flatnonzero(elig).tolist()
    rec["n_eligible"] = len(cand)
    if not cand:
        rec["status"] = "no momentum (liquidate)"
        return rec

    top = sorted(cand, key=lambda j: mom[j], reverse=True)[:stock_count]

    # -------- CEILING SCALING --------
    scaled = {}
    for j in top:
        s = tickers[j]
        c = dict(ticker=s, mom=float(mom[j]), adx=float(adx[j]), idx=None,
                 hist_high=None, scale=None, exhausted=False,
                 lm=float(snap["sema"][r, j]), peak=float(snap["smax"][r, j]),
                 note="")
        if not snapb["ma_ready"][r, j] or not snapb["sema_ready"][r, j]:
            c["note"] = "stretch_ema not ready (skipped)"
            rec["candidates"].append(c)
            continue
        dev = float(snap["dev"][r, j])
        if not dev > 0:
            c["note"] = "dev <= 0 (skipped)"
            rec["candidates"].append(c)
            continue
        mid = float(ma[j]); lm = float(snap["sema"][r, j]); px = float(price[j])
        bands = _build_bands(mid, dev, unit=lm, fractions=CEILING_FRACTIONS)
        idx = _band_index(px, bands)

        bh = state["band_hist"].setdefault(s, deque(maxlen=HIST_LEN))
        bh.append(idx)
        historical_high = max(bh)

        if historical_high <= 0:
            scale = 1.0
        elif idx >= historical_high:
            scale = 0.0
        else:
            scale = max(0.2, 1.0 - idx / historical_high)

        current_stretch = lm
        peak_stretch = float(snap["smax"][r, j])
        if idx >= EXHAUST_IDX and peak_stretch > 0 and current_stretch < peak_stretch * EXHAUST_DECAY:
            scale = 0.2
            c["exhausted"] = True

        c.update(idx=idx, hist_high=historical_high, scale=scale, band_hist=list(bh))
        rec["candidates"].append(c)
        scaled[s] = float(mom[j]) * scale

    # -------- FINAL WEIGHTING --------
    if not scaled:
        rec["status"] = "no scaled assets (liquidate)"
        return rec
    total_scaled = sum(scaled.values())
    if total_scaled == 0:
        # QC: v / total_scaled -> ZeroDivisionError -> algorithm stops.
        rec["status"] = "QC ZeroDivisionError (all scales 0)"
        rec["crash"] = True
        return rec
    raw_weights = {s: v / total_scaled for s, v in scaled.items()}
    capped = {s: min(max_weight, w) for s, w in raw_weights.items()}
    current_sum = sum(capped.values())
    final = {s: w / current_sum for s, w in capped.items()} if current_sum > 0 else {}

    rec["final_weights"] = final
    rec["holdings"] = {s: w for s, w in final.items() if w > 0}
    rec["status"] = "invested"
    for c in rec["candidates"]:
        c["weight"] = final.get(c["ticker"], 0.0)
    return rec


def run_engine(prices, tickers, algo_start, stock_count, max_weight,
               total_return, quirk) -> dict:
    print("  Replaying indicators…")
    panel = build_panel(prices, tickers, algo_start, total_return, quirk)
    cal, T = panel["cal"], panel["T"]

    state = dict(allow=True, was_risk_off=False, max_stress=0.0, band_hist={})
    holdings = {}
    rebs = []
    for p in panel["reb_pos"]:
        t = p - 1
        rec = qc_rebalance(state, panel, t, stock_count, max_weight)
        rec["date"], rec["asof"] = cal[p], cal[t]
        if rec.get("crash") or rec["status"].startswith("breadth<50"):
            pass                                   # no orders placed
        else:
            holdings = rec["holdings"]
        rec["holdings_after"] = dict(holdings)
        rebs.append(rec)

    preview_state = copy.deepcopy(state)
    preview = qc_rebalance(preview_state, panel, T - 1, stock_count, max_weight)
    preview["asof"] = cal[T - 1]

    B = panel["BIDX"]
    nvalid = (B >= 0).sum(axis=1)
    nbot = ((B >= 0) & (B <= 4)).sum(axis=1)
    breadth = pd.DataFrame(dict(
        bottom_frac=np.where(nvalid > 0, nbot / np.maximum(nvalid, 1), np.nan),
        n=nvalid), index=cal)
    breadth = breadth.iloc[panel["warm_idx"]:]

    return dict(panel=panel, rebs=rebs, state=state, holdings=holdings,
                preview=preview, preview_state=preview_state, breadth=breadth)


def write_rebalance_log(rebs: list) -> None:
    def fw(d):
        return "; ".join(f"{s}:{w*100:.2f}%" for s, w in
                         sorted(d.items(), key=lambda kv: -kv[1]))
    rows = []
    for r in rebs:
        rows.append(dict(
            rebalance_date=r["date"].date(), data_asof=r["asof"].date(),
            status=r["status"], regime=r["regime"], n_breadth=r["n_breadth"],
            bottom_frac=r["bottom_frac"], max_stress=r["max_stress"],
            improvement=r["improvement"], ceiling_reset=r["reset"],
            top=" | ".join(
                f"{c['ticker']} mom={c['mom']:.4f} idx={c['idx']} hi={c['hist_high']} "
                f"scale={c['scale']}" + (" EXH" if c["exhausted"] else "")
                + (f" [{c['note']}]" if c["note"] else "")
                for c in r["candidates"]),
            weights=fw(r["holdings"]),
            holdings_after=fw(r["holdings_after"]),
        ))
    try:
        pd.DataFrame(rows).to_csv(REBAL_LOG_CSV, index=False)
    except Exception as e:
        log_error("write_rebalance_log", e)


# ══════════════════════════════════════════════════════════════════════════════
# UI helpers
# ══════════════════════════════════════════════════════════════════════════════
def card(children, extra=None):
    style = dict(background=C["card"], borderRadius="10px",
                 border=f"1px solid {C['border']}", padding="18px 22px",
                 marginBottom="18px")
    if extra:
        style.update(extra)
    return html.Div(children, style=style)


def section_title(text):
    return html.Div(text, style=dict(fontSize="11px", color=C["muted"],
                                     textTransform="uppercase", letterSpacing=".07em",
                                     marginBottom="14px"))


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
        dcc.Graph(figure=fig, config=dict(displayModeBar=False), style=dict(width="100%")),
        style=dict(flex="1", background=C["card"], borderRadius="10px",
                   border=f"1px solid {C['border']}", padding="8px", minWidth="320px"),
    )


def flex_row(*children):
    return html.Div(list(children),
                    style=dict(display="flex", gap="16px", flexWrap="wrap",
                               marginBottom="18px"))


def btn(label, id_, color=None):
    return html.Button(label, id=id_, n_clicks=0, style=dict(
        background=color or C["blue"], color="#fff", border="none",
        borderRadius="8px", padding="10px 22px", fontSize="13px",
        fontWeight="600", cursor="pointer", whiteSpace="nowrap",
    ))


def status_line(text, color=None):
    return html.Div(text, style=dict(fontSize="11px", color=color or C["muted"],
                                     marginTop="6px", lineHeight="1.6"))


def error_banner(context: str, exc: Exception) -> html.Div:
    tb = traceback.format_exc()
    return html.Div([
        html.Div(f"⚠ Error in {context}: {exc}",
                 style=dict(color=C["red"], fontWeight="600", marginBottom="8px")),
        html.Pre(tb, style=dict(fontSize="10px", color=C["muted"], whiteSpace="pre-wrap",
                                wordBreak="break-all", background=C["bg"], padding="10px",
                                borderRadius="6px", maxHeight="300px", overflow="auto")),
        html.Div(f"Full traceback also saved to: {ERROR_LOG}",
                 style=dict(fontSize="10px", color=C["muted"], marginTop="6px")),
    ], style=dict(background=C["card"], border=f"1px solid {C['red']}44",
                  borderRadius="10px", padding="18px 22px", marginBottom="18px"))


def dark_table(rows, page_size=25, cond=None):
    return dash_table.DataTable(
        data=rows,
        columns=[{"name": c, "id": c} for c in (rows[0].keys() if rows else [])],
        style_table={"overflowX": "auto", "border": f"1px solid {C['border']}",
                     "borderRadius": "8px"},
        style_cell=dict(background=C["card"], color=C["text"],
                        border=f"1px solid {C['border']}", padding="8px 14px",
                        fontSize="12px", fontFamily="inherit", textAlign="left",
                        whiteSpace="nowrap", overflow="hidden",
                        textOverflow="ellipsis", maxWidth="420px"),
        style_header=dict(background=C["surface"], color=C["muted"], fontWeight="500",
                          fontSize="11px", textTransform="uppercase",
                          letterSpacing=".05em", border=f"1px solid {C['border']}"),
        style_data_conditional=cond or [],
        sort_action="native", filter_action="native", page_size=page_size,
        style_filter=dict(background=C["surface"], color=C["text"],
                          border=f"1px solid {C['border']}"),
    )


# ── Startup ────────────────────────────────────────────────────────────────────
print(f"\nyfinance version: {yf.__version__}")
print(f"Loading universe from: {STOCK_FILES_DIR}")
UNIVERSE_DF, STARTUP_LOG = load_universe(TOP_PER_SECTOR)
for l in STARTUP_LOG:
    print(l)
print()

app = dash.Dash(__name__, title="Momentum Dashboard", suppress_callback_exceptions=True)
server = app.server


def make_universe_block(df: pd.DataFrame):
    if df.empty:
        return html.Div("No universe loaded  (source: stock_files/)",
                        style=dict(color=C["red"], fontSize="13px"))
    return html.Div([
        html.Div(f"{df['sector'].nunique()} sectors  |  {len(df)} tickers  |  source: stock_files/",
                 style=dict(color=C["green"], fontSize="13px", marginBottom="4px")),
        html.Div("  ·  ".join(sorted(df["sector"].unique())),
                 style=dict(fontSize="11px", color=C["muted"])),
    ])


def make_cache_block():
    m = cache_meta()
    if not m:
        return html.Div("Prices:  no cache yet", style=dict(fontSize="11px", color=C["muted"]))
    return html.Div([
        html.Div(f"Prices:  {m['tickers']} tickers × {m['rows']} rows  "
                 f"({m['first']} → {m['last']})",
                 style=dict(fontSize="11px", color=C["green"], marginBottom="2px")),
        html.Div(f"Fetched {m['fetched']} for algo start {m['algo_start']}",
                 style=dict(fontSize="11px", color=C["muted"])),
    ])


def param_label(text):
    return html.Div(text, style=dict(fontSize="11px", color=C["muted"], marginBottom="4px"))


app.layout = html.Div(style=dict(
    background=C["bg"], minHeight="100vh", color=C["text"],
    fontFamily="Inter, system-ui, sans-serif", padding="24px 32px",
), children=[
    html.Div([
        html.H1("Momentum · Band Ceiling Dashboard",
                style=dict(fontSize="22px", fontWeight="600", margin="0")),
        html.Div("Bar-by-bar replay of StockOnlyMomentum (QuantConnect) · yfinance",
                 style=dict(fontSize="13px", color=C["muted"], marginTop="4px")),
    ], style=dict(marginBottom="28px")),

    card([
        section_title("Data & Configuration"),
        html.Div([
            html.Div([
                html.Div("Universe", style=dict(fontSize="12px", color=C["muted"],
                                                marginBottom="8px", fontWeight="500")),
                html.Div(id="universe-block", children=make_universe_block(UNIVERSE_DF)),
                html.Div([btn("Reload from stock_files/", "reload-universe-excel-btn", C["muted"])],
                         style=dict(display="flex", gap="10px", marginTop="10px")),
                html.Div(id="universe-status", style=dict(fontSize="11px", color=C["muted"],
                                                          marginTop="6px", minHeight="18px")),
            ], style=dict(minWidth="300px", maxWidth="420px")),

            html.Div(style=dict(width="1px", background=C["border"], margin="0 20px",
                                alignSelf="stretch")),

            html.Div([
                html.Div("Cached data", style=dict(fontSize="12px", color=C["muted"],
                                                   marginBottom="8px", fontWeight="500")),
                html.Div(id="cache-status-block", children=make_cache_block()),
                param_label("Algo start date (QC SetStartDate)"),
                dcc.Input(id="algo-start", type="text", value=DEFAULT_ALGO_START,
                          debounce=True, style=dict(background=C["surface"], color=C["text"],
                                                    border=f"1px solid {C['border']}",
                                                    borderRadius="6px", padding="6px 10px",
                                                    width="140px")),
                html.Div([btn("Fetch Prices", "fetch-prices-btn", C["purple"])],
                         style=dict(display="flex", gap="10px", marginTop="12px")),
                html.Div(id="fetch-status", style=dict(fontSize="11px", color=C["muted"],
                                                       marginTop="8px", minHeight="18px")),
            ], style=dict(minWidth="340px")),

            html.Div(style=dict(width="1px", background=C["border"], margin="0 20px",
                                alignSelf="stretch")),

            html.Div([
                html.Div("Parameters", style=dict(fontSize="12px", color=C["muted"],
                                                  marginBottom="14px", fontWeight="500")),
                html.Div([
                    html.Div([
                        param_label("Top N per sector (QC: 100)"),
                        dcc.Slider(id="top-per-file", min=10, max=100, step=10,
                                   value=TOP_PER_SECTOR,
                                   marks={10: "10", 25: "25", 50: "50", 75: "75", 100: "100"},
                                   tooltip=dict(placement="bottom", always_visible=True)),
                    ], style=dict(minWidth="240px")),
                    html.Div([
                        param_label("stock_count (QC: 10)"),
                        dcc.Slider(id="top-n", min=3, max=20, step=1, value=STOCK_COUNT,
                                   marks={3: "3", 5: "5", 10: "10", 15: "15", 20: "20"},
                                   tooltip=dict(placement="bottom", always_visible=True)),
                    ], style=dict(minWidth="240px")),
                    html.Div([
                        param_label("max_weight % (QC: 20)"),
                        dcc.Slider(id="max-weight", min=5, max=40, step=5,
                                   value=int(MAX_WEIGHT * 100),
                                   marks={5: "5", 10: "10", 20: "20", 30: "30", 40: "40"},
                                   tooltip=dict(placement="bottom", always_visible=True)),
                    ], style=dict(minWidth="240px")),
                    dcc.Checklist(
                        id="engine-opts",
                        options=[
                            {"label": " stretch_ema double-feed (experimental — QC does not do this)",
                             "value": "quirk"},
                            {"label": " TotalReturn normalization (QC setting)",
                             "value": "tr"},
                        ],
                        value=["tr"],
                        style=dict(fontSize="12px", color=C["text"]),
                        labelStyle=dict(display="block", marginBottom="4px"),
                    ),
                ], style=dict(display="flex", flexDirection="column", gap="18px")),
            ]),

            html.Div(style=dict(width="1px", background=C["border"], margin="0 20px",
                                alignSelf="stretch")),

            html.Div([
                btn("Run Analysis", "run-btn"),
                html.Div(id="run-status", style=dict(fontSize="11px", color=C["muted"],
                                                     marginTop="8px", textAlign="center")),
            ], style=dict(display="flex", flexDirection="column", alignItems="center",
                          justifyContent="center", minWidth="140px")),
        ], style=dict(display="flex", alignItems="flex-start", flexWrap="wrap", gap="8px")),
    ]),

    dcc.Loading(id="loading-fetch", type="dot", color=C["purple"],
                children=html.Div(id="fetch-spinner")),
    dcc.Loading(id="loading-output", type="circle", color=C["blue"],
                children=html.Div(id="dashboard-output")),
])


# ── Callbacks ─────────────────────────────────────────────────────────────────
@app.callback(
    Output("universe-block", "children"),
    Output("universe-status", "children"),
    Input("reload-universe-excel-btn", "n_clicks"),
    State("top-per-file", "value"),
    prevent_initial_call=True,
)
def do_reload_universe(n_excel, top_per_file):
    global UNIVERSE_DF
    try:
        df, log = load_universe(top_per_file or TOP_PER_SECTOR)
        if df.empty:
            return (make_universe_block(UNIVERSE_DF),
                    status_line(log[-1] if log else "No tickers returned.", C["red"]))
        UNIVERSE_DF = df
        return (make_universe_block(UNIVERSE_DF),
                status_line(f"Loaded {len(df)} tickers from stock_files/.", C["green"]))
    except Exception as e:
        log_error("do_reload_universe", e)
        return make_universe_block(UNIVERSE_DF), status_line(f"Error: {e}", C["red"])


@app.callback(
    Output("fetch-status", "children"),
    Output("cache-status-block", "children"),
    Input("fetch-prices-btn", "n_clicks"),
    State("algo-start", "value"),
    prevent_initial_call=True,
)
def do_fetch_prices(n_clicks, algo_start):
    try:
        if UNIVERSE_DF.empty:
            return status_line("No universe loaded.", C["red"]), make_cache_block()
        algo_start = str(pd.Timestamp(algo_start or DEFAULT_ALGO_START).date())
        payload = fetch_prices(UNIVERSE_DF["ticker"].tolist(), algo_start)
        if not payload:
            return (status_line(f"Fetch returned no data. Check console and {ERROR_LOG}",
                                C["red"]), make_cache_block())
        n = payload["fields"]["Close"].shape[1]
        return status_line(f"Prices saved: {n} tickers → cache/prices.pkl",
                           C["green"]), make_cache_block()
    except Exception as e:
        log_error("do_fetch_prices", e)
        return status_line(f"Error: {e}", C["red"]), make_cache_block()


@app.callback(
    Output("dashboard-output", "children"),
    Output("run-status", "children"),
    Input("run-btn", "n_clicks"),
    State("top-per-file", "value"),
    State("top-n", "value"),
    State("max-weight", "value"),
    State("algo-start", "value"),
    State("engine-opts", "value"),
    prevent_initial_call=True,
)
def run_analysis(n_clicks, top_per_file, top_n, max_weight_pct, algo_start, opts):
    try:
        return _run_analysis_inner(top_per_file, top_n, max_weight_pct, algo_start, opts)
    except Exception as e:
        log_error("run_analysis", e)
        return error_banner("run_analysis", e), f"Error: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# Analysis view
# ══════════════════════════════════════════════════════════════════════════════
def _run_analysis_inner(top_per_file, top_n, max_weight_pct, algo_start, opts):
    if UNIVERSE_DF.empty:
        return html.Div("No universe loaded.", style=dict(color=C["red"], padding="2rem")), ""

    prices = load_prices_cache()
    if prices is None:
        return html.Div([
            html.Div("No price data cached.", style=dict(color=C["amber"])),
            html.Div("Click  Fetch Prices  first.",
                     style=dict(color=C["muted"], fontSize="12px", marginTop="4px")),
        ], style=dict(padding="2rem")), ""

    algo_start = str(pd.Timestamp(algo_start or DEFAULT_ALGO_START).date())
    opts = opts or []
    quirk, total_return = "quirk" in opts, "tr" in opts
    top_n = int(top_n or STOCK_COUNT)
    max_weight = (max_weight_pct or 20) / 100.0

    universe = (UNIVERSE_DF.sort_values("mktcap", ascending=False)
                .groupby("sector", group_keys=False).head(top_per_file or TOP_PER_SECTOR)
                .drop_duplicates(subset="ticker").reset_index(drop=True))
    meta = universe.set_index("ticker").to_dict("index")

    eng = run_engine(prices, universe["ticker"].tolist(), algo_start,
                     top_n, max_weight, total_return, quirk)
    panel, rebs, preview = eng["panel"], eng["rebs"], eng["preview"]
    holdings, breadth = eng["holdings"], eng["breadth"]
    tickers, cal, T = panel["tickers"], panel["cal"], panel["T"]
    write_rebalance_log(rebs)

    notes = list(panel["notes"])
    if quirk:
        notes.append("stretch_ema double-feed ON — experimental; QC output does not show this.")
    crashes = [r for r in rebs if r.get("crash")]
    if crashes:
        notes.append(f"QC would raise ZeroDivisionError and stop on "
                     f"{crashes[0]['date'].date()} ({len(crashes)} such rebalance(s)). "
                     f"The replay keeps prior holdings and continues.")

    last_exec = rebs[-1] if rebs else None
    r_last = panel["row_of"][T - 1]
    snap, snapb, MOM = panel["snap"], panel["snapb"], panel["MOM"]
    fed_now = snapb["fed"][r_last]
    n_resets = sum(1 for r in rebs if r["reset"])
    bf_now = float(breadth["bottom_frac"].iloc[-1]) if len(breadth) else float("nan")

    state = eng["state"]
    if not state["allow"]:
        regime_color, regime_label = C["red"], "Risk-Off — liquidated at last rebalance"
    elif last_exec and last_exec["reset"]:
        regime_color, regime_label = C["amber"], "Recovery — ceilings reset at last rebalance"
    elif last_exec and last_exec["status"].startswith("no"):
        regime_color, regime_label = C["amber"], "Risk-On — but in cash (no qualifying names)"
    else:
        regime_color, regime_label = C["green"], "Risk-On — invested"

    prev_hold = preview.get("holdings", {})
    prev_cands = preview.get("candidates", [])

    # ── Breadth chart ──
    fig_breadth = go.Figure()
    if len(breadth):
        bd = breadth[breadth["n"] >= MIN_BREADTH_SAMPLE]
        shapes = []
        ends = [r["date"] for r in rebs[1:]] + [cal[-1]]
        for r, x1 in zip(rebs, ends):
            if r["regime"] == "risk_off":
                shapes.append(dict(type="rect", xref="x", yref="paper", x0=r["date"], x1=x1,
                                   y0=0, y1=1, fillcolor=C["red"], opacity=0.12, line_width=0))
        fig_breadth.add_trace(go.Scatter(
            x=bd.index, y=bd["bottom_frac"] * 100, mode="lines", name="Daily bottom-band %",
            line=dict(color=C["blue"], width=1.2),
            hovertemplate="%{x|%Y-%m-%d}<br>Stress: %{y:.1f}%<extra></extra>"))
        rv = [r for r in rebs if not np.isnan(r["bottom_frac"])]
        if rv:
            col = [C["red"] if r["regime"] == "risk_off" else
                   C["green"] if r["reset"] else C["text"] for r in rv]
            fig_breadth.add_trace(go.Scatter(
                x=[r["date"] for r in rv], y=[r["bottom_frac"] * 100 for r in rv],
                mode="markers", name="Rebalance evaluation",
                marker=dict(size=[11 if r["reset"] else 5 for r in rv], color=col,
                            symbol=["triangle-up" if r["reset"] else "circle" for r in rv]),
                customdata=[(r["status"], str(r["asof"].date())) for r in rv],
                hovertemplate=("%{x|%Y-%m-%d} (data as of %{customdata[1]})<br>"
                               "Stress: %{y:.1f}%<br>%{customdata[0]}<extra></extra>")))
        for y, colr, txt in ((45, C["red"], "Risk-off (45%)"),
                             (15, C["green"], "Recovery floor (15%)")):
            fig_breadth.add_hline(y=y, line=dict(color=colr, dash="dot", width=1),
                                  annotation_text=txt, annotation_font=dict(color=colr, size=10))
        ymax = max(55, float(np.nanmax(bd["bottom_frac"])) * 110) if len(bd) else 55
        fig_breadth.update_layout(**layout(
            height=300,
            title=dict(text="Breadth stress — daily (line) vs the monthly values QC actually "
                            "acts on (dots; ▲ = ceiling reset)", font=dict(size=13), x=0),
            yaxis=dict(gridcolor=C["grid"], linecolor=C["border"], ticksuffix="%",
                       range=[0, ymax], title="% in bands 0–4"),
            shapes=shapes,
            legend=dict(orientation="h", y=1.08, x=0, font=dict(size=10),
                        bgcolor="rgba(0,0,0,0)")))

    # ── Weights: current vs preview ──
    names = sorted(set(holdings) | set(prev_hold),
                   key=lambda s: (prev_hold.get(s, 0), holdings.get(s, 0)))
    if names:
        fig_w = go.Figure()
        fig_w.add_trace(go.Bar(y=names, x=[holdings.get(s, 0) * 100 for s in names],
                               orientation="h", name="Current (last rebalance)",
                               marker_color=C["muted"]))
        fig_w.add_trace(go.Bar(y=names, x=[prev_hold.get(s, 0) * 100 for s in names],
                               orientation="h", name="Next-rebalance preview",
                               marker_color=C["blue"],
                               text=[f"{prev_hold.get(s, 0)*100:.1f}%" for s in names],
                               textposition="outside", textfont=dict(size=10)))
        fig_w.update_layout(**layout(
            barmode="group", height=max(220, len(names) * 44 + 90),
            title=dict(text="Weights — current vs preview", font=dict(size=13), x=0),
            xaxis_title="Weight %", margin=dict(l=12, r=60, t=40, b=12),
            legend=dict(orientation="h", y=1.1, x=0, font=dict(size=10))))
    else:
        fig_w = go.Figure()
        fig_w.update_layout(**layout(height=180, title="Weights",
                                     annotations=[dict(text="No positions", x=.5, y=.5,
                                                       showarrow=False,
                                                       font=dict(color=C["muted"], size=14))]))

    # ── Breadth band distribution (latest bar) ──
    bl = panel["BIDX"][T - 1]
    bl = bl[bl >= 0]
    counts = [int((bl == b).sum()) for b in range(12)]
    fig_b = go.Figure(go.Bar(
        x=list(range(12)), y=counts,
        marker_color=[C["red"] if b in BOTTOM_LEVELS else
                      (C["amber"] if b >= 10 else C["teal"]) for b in range(12)],
        text=counts, textposition="outside", textfont=dict(size=10, color=C["text"])))
    fig_b.update_layout(**layout(
        height=260, title=dict(text="Breadth band distribution (latest bar, fixed bands)",
                               font=dict(size=13), x=0),
        xaxis_title="Band index  (red = counted as stress 0–4)", yaxis_title="# stocks"))

    # ── Preview candidates: momentum vs scale ──
    pc = [c for c in prev_cands if c["scale"] is not None]
    fig_s = go.Figure(go.Scatter(
        x=[c["mom"] * 100 for c in pc], y=[c["scale"] * 100 for c in pc],
        mode="markers+text", text=[c["ticker"] for c in pc], textposition="top center",
        textfont=dict(size=10, color=C["muted"]),
        marker=dict(size=12, color=[C["green"] if c["ticker"] in prev_hold else C["red"]
                                    for c in pc]),
        customdata=[(c["idx"], c["hist_high"], "yes" if c["exhausted"] else "no") for c in pc],
        hovertemplate=("<b>%{text}</b><br>Momentum: %{x:.2f}%<br>Scale: %{y:.0f}%<br>"
                       "idx/high: %{customdata[0]}/%{customdata[1]}<br>"
                       "Exhausted: %{customdata[2]}<extra></extra>")))
    fig_s.update_layout(**layout(
        height=320, title=dict(text=f"Preview top-{top_n}: momentum vs scale "
                                    "(red = scaled to 0 → not held)",
                               font=dict(size=13), x=0),
        xaxis_title="Momentum (%)", yaxis_title="Scale (%)"))

    # ── Sector donut (preview) ──
    sec_w = {}
    for s, w in prev_hold.items():
        sec = meta.get(s, {}).get("sector") or "Unknown"
        sec_w[sec] = sec_w.get(sec, 0) + w * 100
    fig_sec = go.Figure()
    if sec_w:
        fig_sec.add_trace(go.Pie(labels=list(sec_w), values=[round(v, 1) for v in sec_w.values()],
                                 hole=0.5, textinfo="label+percent", textfont=dict(size=11),
                                 marker_colors=[C["blue"], C["teal"], C["purple"], C["amber"],
                                                C["green"], C["red"], C["muted"]]))
    fig_sec.update_layout(**layout(height=300, showlegend=False,
                                   title=dict(text="Sector allocation (preview)",
                                              font=dict(size=13), x=0)))

    # ── band_hist heatmap (monthly ceiling entries) ──
    fig_heat = None
    hm = [c for c in prev_cands if c.get("band_hist")]
    if hm:
        z = [c["band_hist"][-40:] for c in hm]
        ml = max(len(v) for v in z)
        z = [[None] * (ml - len(v)) + v for v in z]
        fig_heat = go.Figure(go.Heatmap(
            z=z, y=[c["ticker"] for c in hm], zmin=0, zmax=11,
            colorscale=[[0, C["green"]], [0.4, C["teal"]], [0.75, C["amber"]], [1, C["red"]]],
            colorbar=dict(title="Band", tickfont=dict(size=10), len=0.8)))
        fig_heat.update_layout(**layout(
            height=320, title=dict(text="band_hist — ceiling band per monthly rebalance "
                                        "(preview top names, last 40)",
                                   font=dict(size=13), x=0),
            xaxis_title="<- older rebalances  |  preview ->",
            margin=dict(l=12, r=70, t=40, b=12)))

    # ── Rebalance history table ──
    reb_rows = []
    for r in reversed(rebs):
        reb_rows.append({
            "Rebalance": str(r["date"].date()), "Data as of": str(r["asof"].date()),
            "Status": r["status"], "Regime": r["regime"],
            "Stress": "" if np.isnan(r["bottom_frac"]) else f"{r['bottom_frac']*100:.1f}%",
            "Max stress": "" if np.isnan(r["max_stress"]) else f"{r['max_stress']*100:.1f}%",
            "Reset": "yes" if r["reset"] else "",
            "Held": len(r["holdings_after"]),
            "Weights": ", ".join(f"{s} {w*100:.1f}%" for s, w in
                                 sorted(r["holdings_after"].items(), key=lambda kv: -kv[1])),
        })

    # ── Universe table (latest bar) ──
    prev_c = {c["ticker"]: c for c in prev_cands}
    rows = []
    for j, s in enumerate(tickers):
        if not fed_now[j]:
            continue
        m = meta.get(s, {})
        px, ma = snap["price"][r_last, j], snap["ma"][r_last, j]
        mom, adx = MOM[r_last, j], snap["adx"][r_last, j]
        above = bool(snapb["ma_ready"][r_last, j] and px > ma)
        elig = (snapb["adx_ready"][r_last, j] and not adx > ADX_LIMIT and above
                and not np.isnan(mom) and mom > 0)
        c = prev_c.get(s)
        bidx = int(panel["BIDX"][T - 1, j])
        rows.append({
            "Ticker": s, "Company": m.get("company", ""), "Sector": m.get("sector", ""),
            "Feed px": f"{px:.2f}",
            "Momentum": "" if np.isnan(mom) else f"{mom*100:+.2f}%",
            "ADX": "" if np.isnan(adx) else f"{adx:.1f}" + ("" if snapb["adx_ready"][r_last, j] else "*"),
            "Above EMA": "yes" if above else "no",
            "Eligible": "yes" if elig else "no",
            "In top": "yes" if c else "",
            "Ceil idx/high": f"{c['idx']} / {c['hist_high']}" if c and c["idx"] is not None else "",
            "Scale": f"{c['scale']*100:.0f}%" if c and c["scale"] is not None else "",
            "lm (stretch EMA)": "" if np.isnan(snap["sema"][r_last, j]) else f"{snap['sema'][r_last, j]:.2f}",
            "Peak stretch": f"{snap['smax'][r_last, j]:.2f}",
            "Breadth band": bidx if bidx >= 0 else "",
            "Current wt": f"{holdings[s]*100:.1f}%" if s in holdings else "—",
            "Preview wt": f"{prev_hold[s]*100:.1f}%" if s in prev_hold else "—",
        })
    rows.sort(key=lambda x: (x["Preview wt"] == "—", x["In top"] != "yes",
                             -(float(x["Momentum"].rstrip("%")) if x["Momentum"] else -1e9)))
    cond = [
        {"if": {"filter_query": '{Momentum} contains "+"', "column_id": "Momentum"}, "color": C["green"]},
        {"if": {"filter_query": '{Momentum} contains "-"', "column_id": "Momentum"}, "color": C["red"]},
        {"if": {"filter_query": '{Eligible} = "yes"', "column_id": "Eligible"}, "color": C["green"]},
        {"if": {"filter_query": '{Preview wt} != "—"', "column_id": "Preview wt"},
         "color": C["blue"], "fontWeight": "600"},
        {"if": {"filter_query": '{Current wt} != "—"', "column_id": "Current wt"},
         "color": C["teal"], "fontWeight": "600"},
    ]

    note_divs = [html.Div(f"• {n}", style=dict(fontSize="12px", color=C["amber"],
                                               marginBottom="4px")) for n in notes]

    last_label = (f"{last_exec['date'].date()} · {last_exec['status']}" if last_exec
                  else "none yet")
    status_msg = (f"{int(fed_now.sum())} symbols replayed · {len(rebs)} rebalances · "
                  f"{len(holdings)} held · preview {len(prev_hold)} · "
                  f"{n_resets} ceiling reset(s)")

    return html.Div([
        html.Div([
            html.Div(style=dict(width="10px", height="10px", borderRadius="50%",
                                background=regime_color, flexShrink="0")),
            html.Span(regime_label, style=dict(fontWeight="600", fontSize="14px")),
            html.Div([
                html.Span(f"Last rebalance: {last_label}",
                          style=dict(fontSize="12px", color=C["muted"])),
                html.Span(" · ", style=dict(color=C["border"], padding="0 6px")),
                html.Span(f"Preview uses data through {preview['asof'].date()} → "
                          f"{preview['status']}",
                          style=dict(fontSize="12px", color=C["muted"])),
            ], style=dict(marginLeft="auto", display="flex", alignItems="center",
                          flexWrap="wrap", gap="2px")),
        ], style=dict(display="flex", alignItems="center", gap="12px",
                      background=regime_color + "18", border=f"1px solid {regime_color}44",
                      borderRadius="10px", padding="14px 20px", marginBottom="18px")),

        card([section_title("Replay notes"), *note_divs]) if note_divs else html.Div(),

        card([
            section_title("Summary"),
            html.Div([
                metric_box("Symbols", str(int(fed_now.sum())), "subscribed at latest bar"),
                metric_box("Held now", str(len(holdings)), "targets from last rebalance"),
                metric_box("Preview held", str(len(prev_hold)),
                           f"of {len(prev_cands)} top (cap {top_n})"),
                metric_box("Eligible now", str(preview.get("n_eligible", 0)),
                           "ADX ≤ 35, > EMA, mom > 0"),
                metric_box("Breadth stress", f"{bf_now*100:.1f}%", "latest bar",
                           C["red"] if bf_now >= RISK_OFF_FRAC else
                           (C["amber"] if bf_now > 0.3 else C["green"])),
                metric_box("Max stress", f"{state['max_stress']*100:.1f}%",
                           "QC max_stress_level",
                           C["amber"] if state["was_risk_off"] else None),
                metric_box("Ceiling resets", str(n_resets), "recovery events"),
                metric_box("QC crashes", str(len(crashes)), "ZeroDivisionError rebalances",
                           C["red"] if crashes else None),
            ], style=dict(display="flex", gap="10px", flexWrap="wrap")),
        ]),

        card([section_title("Breadth History"),
              dcc.Graph(figure=fig_breadth, config=dict(displayModeBar=False))]),

        flex_row(chart_card(fig_w), chart_card(fig_b, height=260)),
        flex_row(chart_card(fig_s, height=320), chart_card(fig_sec, height=300)),
        (flex_row(chart_card(fig_heat, height=320)) if fig_heat else html.Div()),

        card([section_title(f"Rebalance history · {len(rebs)} month-ends · "
                            f"full log: cache/rebalance_log.csv"),
              dark_table(reb_rows, page_size=12)]),

        card([section_title(f"Universe at latest bar · {len(rows)} symbols "
                            f"(* = ADX not ready)"),
              dark_table(rows, page_size=30, cond=cond)]),
    ]), status_msg


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8050)
