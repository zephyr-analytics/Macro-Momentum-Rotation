"""
ETF Momentum & Band Ceiling — Live Signals Dashboard
=====================================================
Plotly Dash app that replicates the QuantConnect algorithm's signal
logic against live (yfinance) price data.

Run:
    python etf_dashboard.py
Then open http://127.0.0.1:8050 in your browser.
"""

import dash
from dash import dcc, html, dash_table, Input, Output, callback
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────────────────────
# Universe & constants  (mirrors the QC file exactly)
# ──────────────────────────────────────────────────────────────────────────────

ETF_UNIVERSE = [
    "VTI", "VB", "VTV", "VUG", "MTUM", "QUAL", "DGRO",
    "VEA", "VWO", "EEMS", "SCZ", "EFV", "EFG", "IQLT", "IMTM", "IGRO",
    "IXN", "GXI", "IXC", "IXJ", "EXI", "KXI", "RXI", "JXI", "MXI",
    "VGSH", "VGIT", "BND", "BNDX",
    "GLD", "PDBC",
    "VNQ", "VNQI",
    "BIL",
]
CASH_ETF = "BIL"
NON_CASH = [t for t in ETF_UNIVERSE if t != CASH_ETF]

LOOKBACKS   = [21, 63, 126, 189, 252]
BAND_LEN    = 168
HIST_LEN    = 126
STOCK_COUNT = 5
MAX_WEIGHT  = 0.30

RISK_OFF_THRESHOLD   = 0.35
RECOVERY_FLOOR       = 0.12
BOTTOM_LEVELS        = {0, 1, 2, 3, 4}

SECTOR_MAP = {
    "VTI":"Broad Equity","VB":"Broad Equity","VTV":"Broad Equity",
    "VUG":"Broad Equity","MTUM":"Broad Equity","QUAL":"Broad Equity","DGRO":"Broad Equity",
    "VEA":"Intl Equity","VWO":"Intl Equity","EEMS":"Intl Equity","SCZ":"Intl Equity",
    "EFV":"Intl Equity","EFG":"Intl Equity","IQLT":"Intl Equity","IMTM":"Intl Equity","IGRO":"Intl Equity",
    "IXN":"Sector","GXI":"Sector","IXC":"Sector","IXJ":"Sector","EXI":"Sector",
    "KXI":"Sector","RXI":"Sector","JXI":"Sector","MXI":"Sector",
    "VGSH":"Fixed Income","VGIT":"Fixed Income","BND":"Fixed Income","BNDX":"Fixed Income",
    "GLD":"Commodities","PDBC":"Commodities",
    "VNQ":"Real Estate","VNQI":"Real Estate",
    "BIL":"Cash",
}

PALETTE = {
    "bg":           "#0c0f14",
    "panel":        "#131820",
    "border":       "#1e2733",
    "accent":       "#00d4ff",
    "accent2":      "#7b61ff",
    "green":        "#00e5a0",
    "red":          "#ff4d6d",
    "amber":        "#ffb340",
    "text":         "#e8edf5",
    "muted":        "#5a6a7e",
    "grid":         "#1a2130",
}


# ──────────────────────────────────────────────────────────────────────────────
# Signal engine  (pure-Python replica of the QC algorithm)
# ──────────────────────────────────────────────────────────────────────────────

def build_bands(mid, dev, m=1.0):
    fibs = [1.618, 1.382, 1.0, 0.809, 0.5, 0.382]
    levels = [mid - dev * f * m for f in reversed(fibs)] + [mid] + [mid + dev * f * m for f in fibs]
    return levels


def build_bands_adaptive(mid, dev, lm):
    lm2 = lm / 2.0
    lm3 = lm2 * 0.38196601
    lm4 = lm * 1.38196601
    lm5 = lm * 1.61803399
    lm6 = (lm + lm2) / 2.0
    return [
        mid - dev*lm5, mid - dev*lm4, mid - dev*lm,
        mid - dev*lm6, mid - dev*lm2, mid - dev*lm3,
        mid,
        mid + dev*lm3, mid + dev*lm2, mid + dev*lm6,
        mid + dev*lm, mid + dev*lm4, mid + dev*lm5,
    ]


def band_index(price, bands):
    for i in range(len(bands) - 1):
        if bands[i] <= price < bands[i+1]:
            return i
    return len(bands) - 2


def compute_signals(prices: pd.DataFrame, regime_state: dict | None = None) -> dict:
    """
    Run the full algorithm logic on historical price data.

    Parameters
    ----------
    prices : pd.DataFrame
        Historical close prices, one column per ticker.
    regime_state : dict | None
        Persisted state from the previous run:
          {
            "was_risk_off":    bool,
            "max_stress":      float,
            "band_hist_high":  {ticker: int},   # per-ticker historical-high band index
                                                 # reset to 0 on recovery, exactly as QC does
          }
        None → treat as first run (no prior state).

    Returns
    -------
    dict with keys:
        ticker_data, weights, bottom_frac, risk_off, as_of,
        new_regime_state   ← caller must persist this for the next run
    """
    # ── Unpack or initialise regime state ──────────────────────────────────
    if regime_state is None:
        regime_state = {}

    was_risk_off   = regime_state.get("was_risk_off", False)
    max_stress     = regime_state.get("max_stress", 0.0)
    # band_hist_high: maps ticker → highest band index seen since last reset
    # A value of None means "never observed" — will be filled from rolling history
    # on this run and then accumulated going forward.
    band_hist_high = dict(regime_state.get("band_hist_high", {}))

    results  = {}
    tickers  = [t for t in ETF_UNIVERSE if t in prices.columns]
    non_cash = [t for t in tickers if t != CASH_ETF]

    # ── Per-ticker metrics ─────────────────────────────────────────────────
    ticker_data = {}
    for t in tickers:
        px = prices[t].dropna()
        if len(px) < BAND_LEN + 10:
            continue

        closes  = px.values
        sma     = pd.Series(closes).rolling(BAND_LEN).mean().values
        std_arr = pd.Series(closes).rolling(BAND_LEN).std().values

        # Current bar
        close = closes[-1]
        mid   = sma[-1]
        dev   = std_arr[-1]
        if np.isnan(mid) or np.isnan(dev) or dev == 0:
            continue

        # Stretch (current Z-score from SMA)
        stretch = abs(close - mid) / dev

        # SMA-smoothed stretch — mean of |z| over last BAND_LEN bars
        z_series = np.abs((closes - sma) / np.where(std_arr > 0, std_arr, np.nan))
        lm = float(np.nanmean(z_series[-BAND_LEN:]))

        # ── Breadth band index: fixed bands, multiplier=1.0  (mirrors OnData) ──
        fixed_bands = build_bands(mid, dev, m=1.0)
        breadth_idx = band_index(close, fixed_bands)

        # ── Sizing band index: adaptive bands  (mirrors Rebalance) ──
        lm_safe    = lm if lm > 0 else 1.0
        adap_bands = build_bands_adaptive(mid, dev, lm_safe)
        sizing_idx = band_index(close, adap_bands)

        # ── Rolling hist-high over last HIST_LEN bars (adaptive bands) ──
        rolling_hist_indices = []
        for i in range(max(0, len(closes) - HIST_LEN), len(closes)):
            m2 = sma[i]; d2 = std_arr[i]
            if np.isnan(m2) or np.isnan(d2) or d2 == 0:
                continue
            z2 = float(np.nanmean(z_series[max(0, i - BAND_LEN):i])) or 1.0
            b2 = build_bands_adaptive(m2, d2, z2)
            rolling_hist_indices.append(band_index(closes[i], b2))

        rolling_high = max(rolling_hist_indices) if rolling_hist_indices else sizing_idx

        # ── Accumulate the persisted historical high ──
        # The QC algorithm accumulates band_hist indefinitely (until a recovery
        # reset clears it).  We mimic this by taking the max of:
        #   (a) whatever was stored from previous runs
        #   (b) the rolling high computed from the data we have now
        prev_high     = band_hist_high.get(t, 0)
        hist_high     = max(prev_high, rolling_high)
        band_hist_high[t] = hist_high          # will be saved back into regime state

        # Momentum
        mom = np.nan
        if len(px) >= max(LOOKBACKS) + 1:
            mom = float(np.mean([
                closes[-1] / closes[-lb - 1] - 1
                for lb in LOOKBACKS
                if len(closes) > lb
            ]))

        # Uptrend filter
        uptrend = close > mid

        # Band ceiling scale (uses sizing_idx vs accumulated hist_high)
        if hist_high <= 0:
            scale = 1.0
        elif sizing_idx >= hist_high:
            scale = 0.0
        else:
            scale = max(0.2, 1.0 - sizing_idx / hist_high)

        # Anticipatory exhaustion
        peak_stretch = float(np.nanmax(z_series[-252:])) if len(z_series) >= 10 else stretch
        exhausted    = False
        if sizing_idx >= 10 and peak_stretch > 0 and stretch < (peak_stretch * 0.80):
            scale     = 0.2
            exhausted = True

        # 52-week % from high/low
        w52           = closes[-min(252, len(closes)):]
        pct_from_high = (close - w52.max()) / w52.max() * 100
        pct_from_low  = (close - w52.min()) / w52.min() * 100

        ticker_data[t] = {
            "ticker":        t,
            "price":         round(close, 2),
            "sma168":        round(mid, 2),
            "dev":           round(dev, 4),
            "stretch":       round(stretch, 3),
            "lm":            round(lm, 3),
            "band_idx":      sizing_idx,        # adaptive — used for sizing & display
            "breadth_idx":   breadth_idx,       # fixed — used for regime breadth check
            "hist_high":     hist_high,
            "scale":         round(scale, 3),
            "momentum":      round(mom * 100, 2) if not np.isnan(mom) else None,
            "uptrend":       uptrend,
            "exhausted":     exhausted,
            "pct_from_high": round(pct_from_high, 1),
            "pct_from_low":  round(pct_from_low, 1),
            "sector":        SECTOR_MAP.get(t, "Other"),
        }

    results["ticker_data"] = ticker_data

    # ── Regime / breadth  (uses breadth_idx — fixed bands, mirrors OnData) ──
    breadth_idxs = [ticker_data[t]["breadth_idx"] for t in non_cash if t in ticker_data]
    bottom_frac  = (
        sum(i in BOTTOM_LEVELS for i in breadth_idxs) / len(breadth_idxs)
        if breadth_idxs else 0.0
    )

    max_stress = max(max_stress, bottom_frac)

    # ── Hysteresis / recovery gate  (mirrors QC Rebalance exactly) ──
    allow_universe = True

    if bottom_frac >= RISK_OFF_THRESHOLD:
        allow_universe = False
        was_risk_off   = True

    elif was_risk_off:
        denominator  = max(max_stress, 0.10)
        improvement  = (max_stress - bottom_frac) / denominator

        if improvement >= RECOVERY_IMPROVEMENT or bottom_frac < RECOVERY_FLOOR:
            # ── Recovery: reset band_hist_high for all tickers ──
            band_hist_high = {t: 0 for t in band_hist_high}
            allow_universe = True
            was_risk_off   = False
            max_stress     = 0.0
        else:
            # Still in recovery wait — stay risk-off
            allow_universe = False

    risk_off = not allow_universe
    results["bottom_frac"] = bottom_frac
    results["risk_off"]    = risk_off
    results["allow_universe"] = allow_universe

    # ── Persist updated regime state ───────────────────────────────────────
    results["new_regime_state"] = {
        "was_risk_off":   was_risk_off,
        "max_stress":     max_stress,
        "band_hist_high": band_hist_high,
    }

    # ── Top picks (only when risk-on) ──────────────────────────────────────
    if not allow_universe:
        final_w = {CASH_ETF: 1.0}
        results["weights"] = final_w
        results["as_of"]   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return results

    candidates = {
        t: d for t, d in ticker_data.items()
        if t != CASH_ETF
        and d["uptrend"]
        and d["momentum"] is not None
        and d["momentum"] > 0
    }
    top_raw = sorted(candidates, key=lambda t: candidates[t]["momentum"], reverse=True)[:STOCK_COUNT]

    scaled = {}
    for t in top_raw:
        d   = ticker_data[t]
        raw = d["momentum"] * d["scale"] / 100
        scaled[t] = max(raw, 0)

    if scaled:
        total = sum(scaled.values())
        raw_w = {t: v / total for t, v in scaled.items()}
        capped = {t: min(MAX_WEIGHT, w) for t, w in raw_w.items()}
        csum   = sum(capped.values())
        final_w = {t: w / csum for t, w in capped.items()} if csum > 0 else {}
    else:
        final_w = {}

    etf_total = sum(final_w.values())
    remainder = max(0.0, 1.0 - etf_total)
    if remainder > 0.01:
        final_w[CASH_ETF] = remainder

    results["weights"] = final_w
    results["as_of"]   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Data fetcher
# ──────────────────────────────────────────────────────────────────────────────

_cache = {"ts": None, "prices": None, "signals": None}

def fetch_and_compute(regime_state=None):
    """
    Download prices (cached for 10 min) and run compute_signals.

    regime_state is the persisted hysteresis dict from the previous run.
    It is passed straight through to compute_signals so that
    was_risk_off, max_stress, and band_hist_high all accumulate correctly
    across dashboard refreshes.
    """
    now = datetime.now()
    # Re-use cached prices if fresh enough, but always recompute signals
    # with the latest regime_state so hysteresis is applied correctly.
    if _cache["prices"] is None or not _cache["ts"] or (now - _cache["ts"]).seconds >= 600:
        end   = now
        start = end - timedelta(days=520)
        raw   = yf.download(ETF_UNIVERSE, start=start, end=end,
                            auto_adjust=True, progress=False, threads=True)
        if isinstance(raw.columns, pd.MultiIndex):
            prices = raw["Close"]
        else:
            prices = raw[["Close"]].rename(columns={"Close": ETF_UNIVERSE[0]})

        prices = prices.ffill().dropna(how="all")
        _cache.update({"ts": now, "prices": prices})

    sigs = compute_signals(_cache["prices"], regime_state)
    _cache["signals"] = sigs
    return sigs


# ──────────────────────────────────────────────────────────────────────────────
# Layout helpers
# ──────────────────────────────────────────────────────────────────────────────

def kpi_card(label, value, subtitle="", color=None):
    color = color or PALETTE["accent"]
    return html.Div([
        html.Div(label, style={
            "font": "10px/1 'Courier New', monospace",
            "letterSpacing": "0.15em",
            "textTransform": "uppercase",
            "color": PALETTE["muted"],
            "marginBottom": "6px",
        }),
        html.Div(value, style={
            "font": "28px/1 'Courier New', monospace",
            "fontWeight": "700",
            "color": color,
            "marginBottom": "4px",
        }),
        html.Div(subtitle, style={
            "font": "11px/1 'Courier New', monospace",
            "color": PALETTE["muted"],
        }),
    ], style={
        "background": PALETTE["panel"],
        "border": f"1px solid {PALETTE['border']}",
        "borderTop": f"2px solid {color}",
        "padding": "18px 20px",
        "borderRadius": "4px",
        "minWidth": "160px",
        "flex": "1",
    })


# ──────────────────────────────────────────────────────────────────────────────
# App
# ──────────────────────────────────────────────────────────────────────────────

app = dash.Dash(__name__, title="ETF Momentum Signals")

app.layout = html.Div([

    # ── Header ──────────────────────────────────────────────────────────────
    html.Div([
        html.Div([
            html.Span("◈ ", style={"color": PALETTE["accent"], "fontSize": "22px"}),
            html.Span("ETF MOMENTUM", style={
                "font": "bold 20px/1 'Courier New', monospace",
                "letterSpacing": "0.25em",
                "color": PALETTE["text"],
            }),
            html.Span(" // BAND CEILING SIGNALS", style={
                "font": "13px/1 'Courier New', monospace",
                "letterSpacing": "0.15em",
                "color": PALETTE["muted"],
                "marginLeft": "12px",
            }),
        ], style={"display": "flex", "alignItems": "center"}),
        html.Div([
            html.Span(id="last-update", style={
                "font": "11px 'Courier New', monospace",
                "color": PALETTE["muted"],
                "marginRight": "16px",
            }),
            html.Button("⟳ REFRESH", id="refresh-btn", n_clicks=0, style={
                "background": "transparent",
                "border": f"1px solid {PALETTE['accent']}",
                "color": PALETTE["accent"],
                "font": "11px 'Courier New', monospace",
                "letterSpacing": "0.1em",
                "padding": "6px 14px",
                "cursor": "pointer",
                "borderRadius": "2px",
            }),
        ], style={"display": "flex", "alignItems": "center"}),
    ], style={
        "background": PALETTE["panel"],
        "borderBottom": f"1px solid {PALETTE['border']}",
        "padding": "16px 28px",
        "display": "flex",
        "justifyContent": "space-between",
        "alignItems": "center",
        "position": "sticky",
        "top": "0",
        "zIndex": "100",
    }),

    # ── KPI row ─────────────────────────────────────────────────────────────
    html.Div(id="kpi-row", style={
        "display": "flex",
        "gap": "12px",
        "padding": "20px 28px 0",
        "flexWrap": "wrap",
    }),

    # ── Regime bar ──────────────────────────────────────────────────────────
    html.Div([
        html.Div([
            html.Span("REGIME  ", style={
                "font": "10px 'Courier New', monospace",
                "letterSpacing": "0.15em",
                "color": PALETTE["muted"],
            }),
            html.Span(id="regime-badge"),
        ], style={"display": "flex", "alignItems": "center", "marginBottom": "8px"}),
        html.Div([
            html.Div(id="stress-bar-fill", style={
                "height": "6px",
                "borderRadius": "3px",
                "transition": "width 0.5s ease",
            }),
        ], style={
            "background": PALETTE["border"],
            "borderRadius": "3px",
            "height": "6px",
            "position": "relative",
            "marginBottom": "4px",
        }),
        html.Div([
            html.Span(id="stress-label", style={
                "font": "11px 'Courier New', monospace",
                "color": PALETTE["muted"],
            }),
            html.Span(f"  RISK-OFF THRESHOLD: {RISK_OFF_THRESHOLD*100:.0f}%", style={
                "font": "11px 'Courier New', monospace",
                "color": PALETTE["muted"],
            }),
        ]),
    ], style={
        "background": PALETTE["panel"],
        "border": f"1px solid {PALETTE['border']}",
        "padding": "16px 20px",
        "margin": "16px 28px 0",
        "borderRadius": "4px",
    }),

    # ── Charts row ──────────────────────────────────────────────────────────
    html.Div([
        # Portfolio weights
        html.Div([
            html.Div("SIGNAL WEIGHTS", style={
                "font": "10px 'Courier New', monospace",
                "letterSpacing": "0.15em",
                "color": PALETTE["muted"],
                "marginBottom": "12px",
            }),
            dcc.Graph(id="weights-chart", config={"displayModeBar": False},
                      style={"height": "300px"}),
        ], style={
            "background": PALETTE["panel"],
            "border": f"1px solid {PALETTE['border']}",
            "padding": "18px",
            "borderRadius": "4px",
            "flex": "1",
            "minWidth": "280px",
        }),

        # Band index heatmap
        html.Div([
            html.Div("BAND INDEX HEATMAP — UNIVERSE", style={
                "font": "10px 'Courier New', monospace",
                "letterSpacing": "0.15em",
                "color": PALETTE["muted"],
                "marginBottom": "12px",
            }),
            dcc.Graph(id="heatmap-chart", config={"displayModeBar": False},
                      style={"height": "300px"}),
        ], style={
            "background": PALETTE["panel"],
            "border": f"1px solid {PALETTE['border']}",
            "padding": "18px",
            "borderRadius": "4px",
            "flex": "2",
            "minWidth": "340px",
        }),

        # Sector momentum
        html.Div([
            html.Div("MOMENTUM BY SECTOR", style={
                "font": "10px 'Courier New', monospace",
                "letterSpacing": "0.15em",
                "color": PALETTE["muted"],
                "marginBottom": "12px",
            }),
            dcc.Graph(id="sector-chart", config={"displayModeBar": False},
                      style={"height": "300px"}),
        ], style={
            "background": PALETTE["panel"],
            "border": f"1px solid {PALETTE['border']}",
            "padding": "18px",
            "borderRadius": "4px",
            "flex": "1",
            "minWidth": "260px",
        }),
    ], style={
        "display": "flex",
        "gap": "12px",
        "padding": "16px 28px 0",
        "flexWrap": "wrap",
    }),

    # ── Signal price chart ───────────────────────────────────────────────────
    html.Div([
        html.Div([
            html.Span("PRICE + BANDS  — ", style={
                "font": "10px 'Courier New', monospace",
                "letterSpacing": "0.15em",
                "color": PALETTE["muted"],
            }),
            dcc.Dropdown(
                id="ticker-select",
                options=[{"label": t, "value": t} for t in NON_CASH],
                value="VTI",
                clearable=False,
                style={
                    "display": "inline-block",
                    "width": "120px",
                    "verticalAlign": "middle",
                    "background": PALETTE["panel"],
                },
            ),
        ], style={"marginBottom": "12px", "display": "flex", "alignItems": "center", "gap": "8px"}),
        dcc.Graph(id="price-chart", config={"displayModeBar": False},
                  style={"height": "340px"}),
    ], style={
        "background": PALETTE["panel"],
        "border": f"1px solid {PALETTE['border']}",
        "padding": "18px",
        "margin": "16px 28px 0",
        "borderRadius": "4px",
    }),

    # ── Rankings table ───────────────────────────────────────────────────────
    html.Div([
        html.Div("FULL UNIVERSE RANKINGS", style={
            "font": "10px 'Courier New', monospace",
            "letterSpacing": "0.15em",
            "color": PALETTE["muted"],
            "marginBottom": "14px",
        }),
        html.Div(id="rankings-table"),
    ], style={
        "background": PALETTE["panel"],
        "border": f"1px solid {PALETTE['border']}",
        "padding": "18px 20px",
        "margin": "16px 28px 24px",
        "borderRadius": "4px",
    }),

    # Auto-refresh interval
    dcc.Interval(id="auto-refresh", interval=10 * 60 * 1000, n_intervals=0),
    # Transient signals (rebuilt each refresh)
    dcc.Store(id="signals-store"),
    # Persisted regime state — survives page reloads via browser localStorage.
    # Stores: was_risk_off (bool), max_stress (float), band_hist_high (dict).
    dcc.Store(id="regime-store", storage_type="local"),

], style={
    "background": PALETTE["bg"],
    "minHeight": "100vh",
    "fontFamily": "'Courier New', monospace",
    "color": PALETTE["text"],
})


# ──────────────────────────────────────────────────────────────────────────────
# Callbacks
# ──────────────────────────────────────────────────────────────────────────────

@app.callback(
    Output("signals-store", "data"),
    Output("regime-store", "data"),
    Input("refresh-btn", "n_clicks"),
    Input("auto-refresh", "n_intervals"),
    dash.dependencies.State("regime-store", "data"),
)
def load_signals(n_clicks, n_intervals, regime_state):
    """
    Loads prices, runs signal engine with persisted regime state,
    writes updated regime state back to localStorage.
    """
    sigs = fetch_and_compute(regime_state or {})
    signals_out = {
        "ticker_data": sigs["ticker_data"],
        "weights":     {t: round(w, 4) for t, w in sigs["weights"].items()},
        "bottom_frac": sigs["bottom_frac"],
        "risk_off":    sigs["risk_off"],
        "as_of":       sigs["as_of"],
    }
    # Persist the updated regime state (was_risk_off, max_stress, band_hist_high).
    # Dash serialises this to browser localStorage automatically.
    new_regime = sigs.get("new_regime_state", regime_state or {})
    return signals_out, new_regime


@app.callback(
    Output("last-update", "children"),
    Output("kpi-row", "children"),
    Output("regime-badge", "children"),
    Output("regime-badge", "style"),
    Output("stress-bar-fill", "style"),
    Output("stress-label", "children"),
    Input("signals-store", "data"),
    dash.dependencies.State("regime-store", "data"),
)
def update_kpis(data, regime_state):
    if not data:
        return "Loading…", [], "", {}, {}, ""

    td    = data["ticker_data"]
    wts   = data["weights"]
    bf    = data["bottom_frac"]
    ro    = data["risk_off"]
    as_of = data["as_of"]
    rs    = regime_state or {}
    was_ro     = rs.get("was_risk_off", False)
    peak_stress = rs.get("max_stress", 0.0)

    uptrend_pct = sum(1 for d in td.values() if d["uptrend"] and d["ticker"] != CASH_ETF) / max(len(td)-1, 1)
    top_ticker  = max((t for t in wts if t != CASH_ETF), key=lambda t: wts[t], default=CASH_ETF)
    top_wt      = wts.get(top_ticker, 0)
    n_pos       = sum(1 for t in wts if t != CASH_ETF and wts[t] > 0)
    bil_pct     = wts.get(CASH_ETF, 0)

    regime_txt   = "◉ RISK-OFF — PARKED IN BIL" if ro else "◉ RISK-ON — ACTIVE"
    regime_color = PALETTE["red"] if ro else PALETTE["green"]

    bar_pct  = min(bf / RISK_OFF_THRESHOLD, 1.0) * 100
    bar_color = PALETTE["red"] if ro else (PALETTE["amber"] if bf > 0.2 else PALETTE["green"])

    kpis = [
        kpi_card("Stress Level", f"{bf*100:.1f}%",
                 f"threshold {RISK_OFF_THRESHOLD*100:.0f}%",
                 bar_color),
        kpi_card("Peak Stress", f"{peak_stress*100:.1f}%",
                 "since last reset", PALETTE["red"] if peak_stress >= RISK_OFF_THRESHOLD else PALETTE["muted"]),
        kpi_card("Recovery Gate", "OPEN" if not was_ro else "LOCKED",
                 "was_risk_off state", PALETTE["green"] if not was_ro else PALETTE["amber"]),
        kpi_card("Uptrend %", f"{uptrend_pct*100:.0f}%",
                 "vs SMA-168", PALETTE["accent"]),
        kpi_card("Open Positions", str(n_pos),
                 f"max {STOCK_COUNT}", PALETTE["accent2"]),
        kpi_card("Top Signal", top_ticker,
                 f"{top_wt*100:.1f}% weight", PALETTE["amber"]),
        kpi_card("BIL / Cash", f"{bil_pct*100:.1f}%",
                 "risk-off buffer", PALETTE["muted"]),
    ]

    return (
        f"AS OF  {as_of}",
        kpis,
        regime_txt,
        {"font": "11px 'Courier New', monospace",
         "letterSpacing": "0.12em", "color": regime_color,
         "border": f"1px solid {regime_color}",
         "padding": "4px 10px", "borderRadius": "2px"},
        {"height": "6px", "borderRadius": "3px",
         "width": f"{bar_pct:.1f}%", "background": bar_color,
         "transition": "width 0.5s ease"},
        f"BOTTOM-BAND FRACTION: {bf*100:.1f}%  |  {sum(1 for d in td.values() if d['band_idx'] in BOTTOM_LEVELS and d['ticker'] != CASH_ETF)} / {len(td)-1} ETFs in bands 0–4",
    )


@app.callback(
    Output("weights-chart", "figure"),
    Input("signals-store", "data"),
)
def update_weights(data):
    if not data:
        return go.Figure()

    wts = {k: v for k, v in data["weights"].items() if v > 0.001}
    tickers = list(wts.keys())
    values  = [wts[t] * 100 for t in tickers]
    colors  = [PALETTE["amber"] if t == CASH_ETF else PALETTE["accent"] for t in tickers]

    fig = go.Figure(go.Bar(
        x=values, y=tickers,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
        textfont={"family": "Courier New", "size": 11, "color": PALETTE["text"]},
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 60, "r": 40, "t": 10, "b": 30},
        xaxis=dict(showgrid=True, gridcolor=PALETTE["grid"],
                   tickfont={"family": "Courier New", "size": 10, "color": PALETTE["muted"]},
                   range=[0, max(values) * 1.25 if values else 35]),
        yaxis=dict(showgrid=False,
                   tickfont={"family": "Courier New", "size": 11, "color": PALETTE["text"]}),
        showlegend=False,
        bargap=0.35,
    )
    return fig


@app.callback(
    Output("heatmap-chart", "figure"),
    Input("signals-store", "data"),
)
def update_heatmap(data):
    if not data:
        return go.Figure()

    td = data["ticker_data"]
    sectors  = ["Broad Equity","Intl Equity","Sector","Fixed Income","Commodities","Real Estate"]
    s_tickers = {s: [t for t in NON_CASH if SECTOR_MAP.get(t) == s and t in td] for s in sectors}

    rows, cols, vals, texts = [], [], [], []
    for sec, tkrs in s_tickers.items():
        for t in tkrs:
            d = td[t]
            rows.append(sec)
            cols.append(t)
            vals.append(d["band_idx"])
            mom = d["momentum"] or 0
            texts.append(
                f"<b>{t}</b><br>Band: {d['band_idx']}/12<br>"
                f"Momentum: {mom:+.1f}%<br>Scale: {d['scale']:.2f}<br>"
                f"Uptrend: {'✓' if d['uptrend'] else '✗'}"
            )

    # Build z-matrix
    unique_sectors = [s for s in sectors if s_tickers[s]]
    all_tkrs_ordered = [t for s in unique_sectors for t in s_tickers[s]]
    z_mat = [[td[t]["band_idx"] if t in td else 0 for t in all_tkrs_ordered]]
    text_mat = [[texts[all_tkrs_ordered.index(t)] for t in all_tkrs_ordered]]

    fig = go.Figure(go.Heatmap(
        z=z_mat,
        x=all_tkrs_ordered,
        y=["Band Index"],
        text=text_mat,
        hoverinfo="text",
        colorscale=[
            [0.0,  "#ff4d6d"],
            [0.35, "#ffb340"],
            [0.55, "#1e2733"],
            [0.75, "#00d4ff"],
            [1.0,  "#00e5a0"],
        ],
        zmin=0, zmax=12,
        showscale=True,
        colorbar=dict(
            thickness=12,
            tickfont={"family": "Courier New", "size": 9, "color": PALETTE["muted"]},
            bgcolor="rgba(0,0,0,0)",
            outlinewidth=0,
        ),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 60, "r": 60, "t": 10, "b": 60},
        xaxis=dict(
            tickfont={"family": "Courier New", "size": 9, "color": PALETTE["text"]},
            tickangle=45,
        ),
        yaxis=dict(showticklabels=False),
        height=300,
    )
    return fig


@app.callback(
    Output("sector-chart", "figure"),
    Input("signals-store", "data"),
)
def update_sector(data):
    if not data:
        return go.Figure()

    td = data["ticker_data"]
    sector_mom = {}
    for t, d in td.items():
        if t == CASH_ETF or d["momentum"] is None:
            continue
        sec = d["sector"]
        sector_mom.setdefault(sec, []).append(d["momentum"])

    sec_avg = {s: np.mean(v) for s, v in sector_mom.items()}
    sec_sorted = sorted(sec_avg.items(), key=lambda x: x[1], reverse=True)

    labels = [s for s, _ in sec_sorted]
    values = [v for _, v in sec_sorted]
    colors = [PALETTE["green"] if v >= 0 else PALETTE["red"] for v in values]

    fig = go.Figure(go.Bar(
        x=values, y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.1f}%" for v in values],
        textposition="outside",
        textfont={"family": "Courier New", "size": 10, "color": PALETTE["text"]},
        hovertemplate="%{y}: %{x:+.2f}%<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 90, "r": 50, "t": 10, "b": 30},
        xaxis=dict(
            showgrid=True, gridcolor=PALETTE["grid"],
            zeroline=True, zerolinecolor=PALETTE["border"],
            tickfont={"family": "Courier New", "size": 9, "color": PALETTE["muted"]},
        ),
        yaxis=dict(
            showgrid=False,
            tickfont={"family": "Courier New", "size": 10, "color": PALETTE["text"]},
        ),
        showlegend=False,
        bargap=0.35,
    )
    return fig


@app.callback(
    Output("price-chart", "figure"),
    Input("ticker-select", "value"),
    Input("signals-store", "data"),
)
def update_price_chart(ticker, data):
    if not ticker or not data:
        return go.Figure()

    prices_df = _cache.get("prices")
    if prices_df is None or ticker not in prices_df.columns:
        return go.Figure()

    px_series = prices_df[ticker].dropna().tail(300)
    closes    = px_series.values
    dates     = px_series.index

    sma    = pd.Series(closes).rolling(BAND_LEN).mean().values
    std_v  = pd.Series(closes).rolling(BAND_LEN).std().values

    td = data["ticker_data"].get(ticker, {})
    lm = td.get("lm", 1.0) or 1.0

    fig = go.Figure()

    # Band fills
    valid = ~(np.isnan(sma) | np.isnan(std_v))
    if valid.any():
        d_valid = dates[valid]
        m_valid = sma[valid]
        s_valid = std_v[valid]

        upper = m_valid + s_valid * lm * 1.618
        lower = m_valid - s_valid * lm * 1.618

        fig.add_trace(go.Scatter(
            x=np.concatenate([d_valid, d_valid[::-1]]),
            y=np.concatenate([upper, lower[::-1]]),
            fill="toself",
            fillcolor="rgba(0,212,255,0.04)",
            line={"width": 0},
            showlegend=False, hoverinfo="skip",
        ))

        for mult, alpha in [(1.0, 0.12), (0.5, 0.08)]:
            u = m_valid + s_valid * lm * mult
            l = m_valid - s_valid * lm * mult
            fig.add_trace(go.Scatter(
                x=np.concatenate([d_valid, d_valid[::-1]]),
                y=np.concatenate([u, l[::-1]]),
                fill="toself",
                fillcolor=f"rgba(0,212,255,{alpha})",
                line={"width": 0},
                showlegend=False, hoverinfo="skip",
            ))

        # SMA line
        fig.add_trace(go.Scatter(
            x=d_valid, y=m_valid,
            line={"color": PALETTE["accent2"], "width": 1.2, "dash": "dot"},
            name="SMA-168", hoverinfo="skip",
        ))

    # Price line
    fig.add_trace(go.Scatter(
        x=dates, y=closes,
        line={"color": PALETTE["accent"], "width": 1.8},
        name=ticker,
        hovertemplate="%{x|%Y-%m-%d}: $%{y:.2f}<extra></extra>",
    ))

    # Annotation
    band_idx = td.get("band_idx", "—")
    mom      = td.get("momentum")
    ann_text = f"Band {band_idx}/12 | Mom {mom:+.1f}%" if mom is not None else f"Band {band_idx}/12"

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 50, "r": 20, "t": 30, "b": 30},
        xaxis=dict(
            showgrid=True, gridcolor=PALETTE["grid"],
            tickfont={"family": "Courier New", "size": 10, "color": PALETTE["muted"]},
        ),
        yaxis=dict(
            showgrid=True, gridcolor=PALETTE["grid"],
            tickfont={"family": "Courier New", "size": 10, "color": PALETTE["muted"]},
            tickprefix="$",
        ),
        legend=dict(
            font={"family": "Courier New", "size": 10, "color": PALETTE["muted"]},
            bgcolor="rgba(0,0,0,0)", borderwidth=0,
        ),
        title=dict(
            text=ann_text,
            font={"family": "Courier New", "size": 11, "color": PALETTE["muted"]},
            x=0.01,
        ),
    )
    return fig


@app.callback(
    Output("rankings-table", "children"),
    Input("signals-store", "data"),
)
def update_table(data):
    if not data:
        return html.Div("Loading…")

    td   = data["ticker_data"]
    wts  = data["weights"]
    rows = []

    for t in NON_CASH:
        if t not in td:
            continue
        d     = td[t]
        weight = wts.get(t, 0.0)
        mom    = d["momentum"]
        mom_s  = f"{mom:+.1f}%" if mom is not None else "—"
        ut_col = PALETTE["green"] if d["uptrend"] else PALETTE["red"]
        ex_col = PALETTE["amber"] if d["exhausted"] else PALETTE["muted"]
        rows.append({
            "__ticker":    t,
            "__sector":    d["sector"],
            "__uptrend":   d["uptrend"],
            "__exhausted": d["exhausted"],
            "TICKER":      t,
            "SECTOR":      d["sector"],
            "PRICE":       f"${d['price']:,.2f}",
            "SMA-168":     f"${d['sma168']:,.2f}",
            "MOM":         mom_s,
            "BAND":        f"{d['band_idx']} / 12",
            "HIST HIGH":   str(d["hist_high"]),
            "SCALE":       f"{d['scale']:.2f}",
            "52W HIGH":    f"{d['pct_from_high']:+.1f}%",
            "52W LOW":     f"{d['pct_from_low']:+.1f}%",
            "WEIGHT":      f"{weight*100:.1f}%" if weight > 0 else "—",
        })

    # Sort: weighted first, then by momentum
    rows.sort(key=lambda r: (
        -1 if wts.get(r["TICKER"], 0) > 0 else 0,
        -(r["__uptrend"]),
        -(td[r["TICKER"]]["momentum"] or -999),
    ))

    header_style = {
        "font": "10px 'Courier New', monospace",
        "letterSpacing": "0.12em",
        "color": PALETTE["muted"],
        "borderBottom": f"1px solid {PALETTE['border']}",
        "padding": "6px 10px",
        "textAlign": "right",
        "whiteSpace": "nowrap",
    }
    header_style_l = {**header_style, "textAlign": "left"}

    columns = ["TICKER","SECTOR","PRICE","SMA-168","MOM","BAND","HIST HIGH","SCALE","52W HIGH","52W LOW","WEIGHT"]
    left_cols = {"TICKER","SECTOR"}

    thead = html.Tr([
        html.Th(c, style=header_style_l if c in left_cols else header_style)
        for c in columns
    ])

    tbody_rows = []
    for r in rows:
        t     = r["TICKER"]
        wt    = wts.get(t, 0)
        is_wt = wt > 0
        is_ex = r["__exhausted"]
        ut    = r["__uptrend"]

        row_bg = "rgba(0,212,255,0.05)" if is_wt else "transparent"

        cells = []
        for c in columns:
            val = r[c]
            style = {
                "padding": "7px 10px",
                "borderBottom": f"1px solid {PALETTE['grid']}",
                "font": "11px 'Courier New', monospace",
                "textAlign": "left" if c in left_cols else "right",
                "color": PALETTE["text"],
                "whiteSpace": "nowrap",
            }
            if c == "TICKER":
                style["color"] = PALETTE["accent"] if is_wt else PALETTE["text"]
                style["fontWeight"] = "bold" if is_wt else "normal"
            elif c == "MOM":
                v = td[t]["momentum"]
                if v is not None:
                    style["color"] = PALETTE["green"] if v >= 0 else PALETTE["red"]
            elif c == "BAND":
                bidx = td[t]["band_idx"]
                style["color"] = PALETTE["red"] if bidx in BOTTOM_LEVELS else (
                    PALETTE["green"] if bidx >= 9 else PALETTE["text"])
            elif c == "SCALE":
                style["color"] = PALETTE["amber"] if is_ex else PALETTE["text"]
            elif c == "WEIGHT":
                style["color"] = PALETTE["amber"] if is_wt else PALETTE["muted"]
            elif c in ("52W HIGH",):
                v_s = val.replace("%","").replace("+","")
                try:
                    v_f = float(v_s)
                    style["color"] = PALETTE["green"] if v_f > -5 else PALETTE["red"]
                except:
                    pass

            cells.append(html.Td(val, style=style))

        tbody_rows.append(html.Tr(cells, style={"background": row_bg}))

    return html.Table(
        [html.Thead(thead), html.Tbody(tbody_rows)],
        style={"width": "100%", "borderCollapse": "collapse"},
    )


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Starting ETF Momentum Dashboard…")
    print("Open http://127.0.0.1:8050 in your browser")
    app.run(debug=False, host="0.0.0.0", port=8050)
