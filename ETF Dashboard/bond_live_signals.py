"""
Live Bond / Cash / Duration Rotation Signals Dashboard
-------------------------------------------------------
Mirrors the exact selection/weighting logic from BondCashDurationRotation.py
using current market data (via yfinance), so you can see what the algo would
signal right now without waiting for a QuantConnect backtest or live
deployment.

Shows, per ticker: current price, whether it's above its 147-day SMA, its
vol-adjusted momentum score, whether that score beats cash (SHV) over the
same lookbacks, whether it passes the 63-day Chaikin Money Flow directional
volume confirmation, which family it belongs to (Treasury vs. IG corporate,
shown for reference only), and the resulting target weight if a rebalance
happened today.

Universe mirrors the algo's TICKERS list exactly - HY_CORP (HYG/SHYG) is
commented out in the algo's source, so it's excluded here too:
    TREASURIES = SHV, SHY, IEI, IEF, TLH, TLT
    IG_CORP    = IGSB, IGIB, IGLB

There is no family-level gate: every ticker in every family is checked on
its own merits, so a strong IG corporate and a strong Treasury can both
qualify and be held at once (the family gate that used to lock out one
whole family based on its representative's momentum has been removed).

Constants below match what's actually set in the algo's code (not its
docstring prose, which is stale on the SMA and CMF window lengths):
    SMA_PERIOD  = 147   (not the 168 the docstring mentions)
    CMF_PERIOD  = 63    (not the 20 the docstring mentions)
    MOM_LOOKBACKS = [21, 63, 126]  (~1mo / 3mo / 6mo)
    VOL_LOOKBACK  = 126  (single vol window)

Momentum score = average of the raw returns over MOM_LOOKBACKS, divided
once by one annualized volatility figure computed over VOL_LOOKBACK days -
not return/vol computed separately per lookback and then averaged.

If nothing in the universe clears all four per-ticker checks, the algo
parks 100% in SHV - this dashboard's fallback does the same.

IMPORTANT: this was built and syntax-tested without live network access -
the sandbox this was authored in cannot reach Yahoo Finance. Run this on a
machine with normal internet access; yfinance requires no API key.

Run with:
    pip install dash yfinance pandas numpy
    python bond_cash_duration_dashboard.py
Then open http://127.0.0.1:8050
"""

from datetime import datetime, timezone

import dash
from dash import dcc, html, dash_table, Input, Output
import pandas as pd
import numpy as np
import yfinance as yf

# ---------------------------------------------------------------------------
# Constants mirrored exactly from BondCashDurationRotation.py
# ---------------------------------------------------------------------------

TREASURIES = ["SHV", "SHY", "IEI", "IEF", "TLH", "TLT"]
IG_CORP = ["IGSB", "IGIB", "IGLB"]
TICKERS = TREASURIES + IG_CORP
CASH_PROXY = "SHV"

SMA_PERIOD = 147
MOM_LOOKBACKS = [21, 63, 126]
VOL_LOOKBACK = 126        # single vol window momentum is divided by
CMF_PERIOD = 63
TRADING_DAYS_PER_YEAR = 252

MAX_LOOKBACK_NEEDED = max(max(MOM_LOOKBACKS), VOL_LOOKBACK, SMA_PERIOD, CMF_PERIOD) + 10

app = dash.Dash(__name__)
app.title = "Live Bond/Cash/Duration Signals"

# ---------------------------------------------------------------------------
# Dark theme / lavender text palette
# ---------------------------------------------------------------------------

COLORS = {
    "background": "#121016",
    "surface": "#1c1a24",
    "border": "#3a3548",
    "text": "#E6E6FA",       # lavender
    "text_dim": "#B9A9D9",   # dimmer lavender for secondary text
    "accent": "#C8A2C8",     # muted lavender-pink for buttons/highlights
    "qualify_bg": "#2a1f3d",
}

app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body { background-color: """ + COLORS["background"] + """; margin: 0; }
            input::placeholder { color: """ + COLORS["text_dim"] + """; opacity: 0.7; }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""


# ---------------------------------------------------------------------------
# Data + scoring (mirrors momentum_score / passes_trend_filter /
# passes_cmf_filter / get_family_gate / rebalance in the algo exactly)
# ---------------------------------------------------------------------------

def fetch_ohlcv(tickers):
    """
    Fetches daily OHLCV for the given tickers, enough history for the
    147-day SMA / 126-day momentum lookback / 63-day CMF window the algo
    needs. Returns a dict of DataFrames keyed by 'close', 'high', 'low',
    'volume', each with one column per ticker.
    """
    empty_result = {key: pd.DataFrame() for key in ("close", "high", "low", "volume")}
    if not tickers:
        return empty_result

    data = yf.download(tickers, period=f"{MAX_LOOKBACK_NEEDED + 30}d", progress=False, auto_adjust=True)

    if len(tickers) == 1:
        # yfinance returns a flat (single-level) frame for a single ticker
        ohlcv = {
            "close": data["Close"].to_frame(tickers[0]),
            "high": data["High"].to_frame(tickers[0]),
            "low": data["Low"].to_frame(tickers[0]),
            "volume": data["Volume"].to_frame(tickers[0]),
        }
    else:
        ohlcv = {
            "close": data["Close"],
            "high": data["High"],
            "low": data["Low"],
            "volume": data["Volume"],
        }

    return {key: df.dropna(how="all") for key, df in ohlcv.items()}


def score_group(ohlcv, tickers):
    """
    Mirrors momentum_score / passes_trend_filter / passes_cmf_filter: for
    each ticker, average the raw returns across MOM_LOOKBACKS first, then
    divide that single average by one annualized volatility figure
    (computed over VOL_LOOKBACK days) - one division, not one per lookback.
    Also computes the above-147-day-SMA flag and the 63-day Chaikin Money
    Flow (CMF) directional volume confirmation.
    """
    scores, above_sma, current_prices, positive_volume_flow = {}, {}, {}, {}
    closes, highs, lows, volumes = ohlcv["close"], ohlcv["high"], ohlcv["low"], ohlcv["volume"]

    for ticker in tickers:
        if ticker not in closes.columns:
            continue
        series = closes[ticker].dropna()

        max_needed = max(max(MOM_LOOKBACKS), VOL_LOOKBACK) + 1
        if len(series) < max_needed:
            continue

        current_price = series.iloc[-1]
        current_prices[ticker] = current_price

        returns = []
        for lookback in MOM_LOOKBACKS:
            if len(series) < lookback + 1:
                continue
            past_price = series.iloc[-1 - lookback]
            if past_price <= 0:
                continue
            returns.append(current_price / past_price - 1)
        if not returns:
            continue
        raw_momentum = float(np.mean(returns))

        vol_window = series.iloc[-(VOL_LOOKBACK + 1):]
        daily_rets = vol_window.pct_change().dropna()
        if len(daily_rets) == 0:
            continue
        vol = daily_rets.std() * (TRADING_DAYS_PER_YEAR ** 0.5)
        if not vol or pd.isna(vol):
            continue

        scores[ticker] = raw_momentum / vol

        if len(series) >= SMA_PERIOD:
            sma_window = series.iloc[-SMA_PERIOD:]
            above_sma[ticker] = current_price > sma_window.mean()
        else:
            above_sma[ticker] = False

        # Chaikin Money Flow (CMF) directional volume confirmation, mirroring
        # the algo exactly: money flow multiplier per day = ((Close-Low)-(High-Close))/(High-Low),
        # weighted by that day's volume, summed over CMF_PERIOD days.
        if ticker not in highs.columns or ticker not in lows.columns or ticker not in volumes.columns:
            positive_volume_flow[ticker] = False
            continue

        high_series = highs[ticker].dropna()
        low_series = lows[ticker].dropna()
        volume_series = volumes[ticker].dropna()

        recent_close = series.iloc[-CMF_PERIOD:]
        recent_high = high_series.iloc[-CMF_PERIOD:]
        recent_low = low_series.iloc[-CMF_PERIOD:]
        recent_volume = volume_series.iloc[-CMF_PERIOD:]

        if (
            len(recent_close) < CMF_PERIOD
            or len(recent_high) < CMF_PERIOD
            or len(recent_low) < CMF_PERIOD
            or len(recent_volume) < CMF_PERIOD
        ):
            positive_volume_flow[ticker] = False
            continue

        high_low_range = recent_high - recent_low
        money_flow_multiplier = np.where(
            high_low_range.values > 0,
            ((recent_close.values - recent_low.values) - (recent_high.values - recent_close.values))
            / np.where(high_low_range.values > 0, high_low_range.values, np.nan),
            0.0,
        )
        money_flow_multiplier = pd.Series(money_flow_multiplier, index=recent_close.index).fillna(0.0)

        money_flow_volume = money_flow_multiplier * recent_volume.values
        total_volume = recent_volume.sum()

        if total_volume <= 0:
            positive_volume_flow[ticker] = False
            continue

        cmf = money_flow_volume.sum() / total_volume
        positive_volume_flow[ticker] = cmf > 0

    return scores, above_sma, current_prices, positive_volume_flow


def compute_signals():
    """Runs the full pipeline and returns a DataFrame of per-ticker signals
    and the resulting target weights dict. No family gate: every ticker in
    the universe is checked on its own merits."""
    ohlcv = fetch_ohlcv(TICKERS)
    scores, above_sma, current_prices, positive_volume_flow = score_group(ohlcv, TICKERS)

    cash_score = scores.get(CASH_PROXY)

    # Per-ticker checks across the whole universe - no family-level gate
    positive_scores = {}
    for ticker in TICKERS:
        if ticker == CASH_PROXY:
            continue
        score = scores.get(ticker)
        if score is None or score <= 0:
            continue
        if cash_score is not None and score <= cash_score:
            continue
        if not above_sma.get(ticker, False):
            continue
        if not positive_volume_flow.get(ticker, False):
            continue
        positive_scores[ticker] = score

    num_selected = len(positive_scores)

    if num_selected == 0:
        target_weights = {CASH_PROXY: 1.0}
    else:
        score_sum = sum(positive_scores.values())
        target_weights = {t: s / score_sum for t, s in positive_scores.items()}

    # Build the display table - every ticker in the universe, not just qualifiers
    rows = []
    for ticker in TICKERS:
        family = "IG" if ticker in IG_CORP else "TREASURY"
        rows.append({
            "Symbol": ticker,
            "Family": family,
            "Price": current_prices.get(ticker, np.nan),
            "Above SMA": above_sma.get(ticker, False),
            "Momentum Score": scores.get(ticker, np.nan),
            "Beats Cash": (
                np.nan if ticker == CASH_PROXY or ticker not in scores or cash_score is None
                else scores[ticker] > cash_score
            ),
            "Positive CMF": positive_volume_flow.get(ticker, False),
            "Qualifies": ticker in positive_scores,
            "Target Weight": target_weights.get(ticker, 0.0),
        })

    signals_df = pd.DataFrame(rows).sort_values("Target Weight", ascending=False).reset_index(drop=True)
    return signals_df, num_selected, target_weights


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

app.layout = html.Div(
    style={
        "fontFamily": "Arial, sans-serif", "maxWidth": "1100px", "margin": "0 auto", "padding": "20px",
        "backgroundColor": COLORS["background"], "color": COLORS["text"], "minHeight": "100vh",
    },
    children=[
        html.H2("Live Bond / Cash / Duration Rotation Signals", style={"color": COLORS["text"]}),
        html.P(
            "Mirrors BondCashDurationRotation.py's family-gate/selection/weighting logic against "
            "current market data. Not connected to a brokerage - this shows what the algo would "
            "signal if a rebalance ran today.",
            style={"color": COLORS["text_dim"]},
        ),

        html.Button(
            "Refresh Now", id="refresh-btn", n_clicks=0,
            style={
                "marginBottom": "10px", "backgroundColor": COLORS["accent"], "color": COLORS["background"],
                "border": "none", "borderRadius": "6px", "padding": "8px 16px", "fontWeight": "bold",
                "cursor": "pointer",
            },
        ),
        html.Div(id="last-updated", style={"fontStyle": "italic", "marginBottom": "20px", "color": COLORS["text_dim"]}),

        html.Div(id="summary-row", style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "20px"}),

        html.H3("Signal Table", style={"color": COLORS["text"]}),
        dash_table.DataTable(
            id="signals-table",
            columns=[
                {"name": "Symbol", "id": "Symbol"},
                {"name": "Family", "id": "Family"},
                {"name": "Price", "id": "Price", "type": "numeric", "format": {"specifier": ",.2f"}},
                {"name": "Above SMA", "id": "Above SMA"},
                {"name": "Momentum Score", "id": "Momentum Score", "type": "numeric", "format": {"specifier": ".3f"}},
                {"name": "Beats Cash", "id": "Beats Cash"},
                {"name": "Positive CMF", "id": "Positive CMF"},
                {"name": "Qualifies", "id": "Qualifies"},
                {"name": "Target Weight", "id": "Target Weight", "type": "numeric", "format": {"specifier": ".2%"}},
            ],
            sort_action="native",
            style_table={"overflowX": "auto"},
            style_cell={
                "padding": "8px", "textAlign": "right",
                "backgroundColor": COLORS["surface"], "color": COLORS["text"],
                "border": f"1px solid {COLORS['border']}",
            },
            style_cell_conditional=[{"if": {"column_id": "Symbol"}, "textAlign": "left"}],
            style_header={
                "fontWeight": "bold", "backgroundColor": COLORS["background"], "color": COLORS["text"],
                "border": f"1px solid {COLORS['border']}",
            },
            style_data_conditional=[
                {"if": {"filter_query": "{Qualifies} = true"}, "backgroundColor": COLORS["qualify_bg"]},
            ],
        ),
    ],
)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

@app.callback(
    Output("last-updated", "children"),
    Output("summary-row", "children"),
    Output("signals-table", "data"),
    Input("refresh-btn", "n_clicks"),
)
def refresh_signals(n_clicks):
    try:
        signals_df, num_selected, target_weights = compute_signals()
    except Exception as e:
        return f"Error fetching/computing signals: {e}", [], []

    def card(label, value):
        return html.Div(
            style={
                "border": f"1px solid {COLORS['border']}", "borderRadius": "8px", "padding": "12px 16px",
                "minWidth": "160px", "textAlign": "center", "backgroundColor": COLORS["surface"],
            },
            children=[
                html.Div(label, style={"fontSize": "12px", "color": COLORS["text_dim"]}),
                html.Div(value, style={"fontSize": "18px", "fontWeight": "bold", "color": COLORS["text"]}),
            ],
        )

    shv_weight = target_weights.get(CASH_PROXY, 0.0)
    ig_weight = sum(w for t, w in target_weights.items() if t in IG_CORP)
    treasury_weight = sum(w for t, w in target_weights.items() if t in TREASURIES and t != CASH_PROXY)
    summary = [
        card("Tickers Qualifying", f"{num_selected} / {len(TICKERS) - 1}"),
        card("IG Corp Weight", f"{ig_weight:.1%}"),
        card("Treasury Weight (ex-SHV)", f"{treasury_weight:.1%}"),
        card("SHV (Cash) Weight", f"{shv_weight:.1%}"),
    ]

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return f"Last updated: {timestamp}", summary, signals_df.to_dict("records")


if __name__ == "__main__":
    app.run(debug=True)
