"""
Live Momentum Signals Dashboard
--------------------------------
Mirrors the exact scoring/weighting logic from MultiAssetMomentumAlgo.py using
current market data (via yfinance), so you can see what the algo would signal
right now without waiting for a QuantConnect backtest or live deployment.

Shows, per asset: current price, raw momentum, volatility-adjusted momentum
score, whether it's above its SMA (bond sleeve uses its own SMA period),
whether it passes the Chaikin Money Flow directional volume confirmation, and
the resulting target weight if a rebalance happened today - including the
MIN_POSITIONS/SHV gap-fill and the MAX_ASSET_WEIGHT cap.

Crypto (BTCUSD, ETHUSD) is included and dampened by CRYPTO_MOMENTUM_SCALE,
matching the algo whenever its CRYPTO_TICKERS is populated. Set
CRYPTO_TICKERS = [] above to match if the algo disables crypto again.

The algo's portfolio-level drawdown circuit breaker was removed, so this
dashboard no longer tracks/simulates one either - SHV's cap exemption here
now matches the algo exactly (MIN_POSITIONS gap-filler only).

Cap-routing update: when num_selected <= MIN_POSITIONS, excess above
MAX_ASSET_WEIGHT is now routed straight to SHV (route_excess_to_exempt),
instead of being redistributed across the other thin-breadth real assets
first - matching the algo's _apply_weight_cap exactly. This avoids
distorting the other qualifiers' momentum-earned proportions (or collapsing
them to equal-weight when MAX_ASSET_WEIGHT happens to equal 1/MIN_POSITIONS).

IMPORTANT: this was built and syntax-tested without live network access -
the sandbox this was authored in cannot reach Yahoo Finance. Run this on a
machine with normal internet access; yfinance requires no API key.

Run with:
    pip install -r requirements_live.txt
    python live_signals.py
Then open http://127.0.0.1:8050
"""

from datetime import datetime, timezone

import dash
from dash import dcc, html, dash_table, Input, Output
import pandas as pd
import numpy as np
import yfinance as yf

# ---------------------------------------------------------------------------
# Constants mirrored exactly from MultiAssetMomentumAlgo.py
# ---------------------------------------------------------------------------

LOOKBACKS = [21, 63, 126, 189, 252]
VOL_LOOKBACK = 126
TRADING_DAYS_PER_YEAR = 252
SMA_PERIOD = 168
BOND_SMA_PERIOD = 126
MIN_POSITIONS = 3  # matches algo (was 4 - out of sync)
CRYPTO_MOMENTUM_SCALE = 0.20
MAX_ASSET_WEIGHT = 0.25
DIRECTIONAL_VOLUME_LOOKBACK = 63

EQUITY_TICKERS = ["VTI", "VEA", "VWO", "IXN", "GLD", "PDBC"]
# SHV lives in the bond sleeve now, matching the algo: it's scored, SMA-filtered
# (using BOND_SMA_PERIOD), and CMF-filtered exactly like BND/BNDX/EMB, and can
# win a momentum-earned weight on its own. SHV_TICKER below is kept only as a
# reference for the gap-filler/no-qualifiers fallback logic.
BOND_TICKERS = ["BND", "BNDX", "EMB", "SHV"]
CRYPTO_TICKERS = ["BTC-USD", "ETH-USD"]  # yfinance's naming for BTCUSD/ETHUSD
SHV_TICKER = "SHV"

# yfinance crypto tickers -> display name matching the algo's QC symbols
# (kept for when crypto is re-enabled; use "BTC-USD"/"ETH-USD" in CRYPTO_TICKERS above)
CRYPTO_DISPLAY_NAMES = {"BTC-USD": "BTCUSD", "ETH-USD": "ETHUSD"}

MAX_LOOKBACK_NEEDED = max(
    max(LOOKBACKS), SMA_PERIOD, BOND_SMA_PERIOD, VOL_LOOKBACK, DIRECTIONAL_VOLUME_LOOKBACK
) + 10

app = dash.Dash(__name__)
app.title = "Live Momentum Signals"

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
# Data + scoring (mirrors _score_symbol_group in the algo exactly)
# ---------------------------------------------------------------------------

def fetch_ohlcv(tickers):
    """
    Fetches daily OHLCV for the given tickers, enough history for the longest
    lookback/SMA/vol/CMF window the algo needs. Returns a dict of DataFrames
    keyed by 'close', 'high', 'low', 'volume', each with one column per ticker.

    Called once per asset group (equity, bond, crypto) rather than once for
    the whole universe - see the call sites in compute_signals() for why.
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


def score_group(ohlcv, tickers, sma_period, crypto_scale=1.0):
    """
    Mirrors _score_symbol_group: for each ticker, compute the vol-adjusted
    momentum score, the above-SMA flag using this group's sma_period, and the
    Chaikin Money Flow (CMF) directional volume confirmation.
    """
    scores, above_sma, raw_scores, current_prices, positive_volume_flow = {}, {}, {}, {}, {}
    closes, highs, lows, volumes = ohlcv["close"], ohlcv["high"], ohlcv["low"], ohlcv["volume"]

    for ticker in tickers:
        if ticker not in closes.columns:
            continue
        series = closes[ticker].dropna()

        max_lookback = max(max(LOOKBACKS), sma_period, VOL_LOOKBACK, DIRECTIONAL_VOLUME_LOOKBACK)
        if len(series) < max_lookback + 1:
            continue

        current_price = series.iloc[-1]
        current_prices[ticker] = current_price

        returns = []
        for lookback in LOOKBACKS:
            past_price = series.iloc[-1 - lookback]
            if past_price <= 0:
                continue
            returns.append((current_price / past_price) - 1.0)

        if len(returns) != len(LOOKBACKS):
            continue

        raw_momentum = sum(returns) / len(returns)
        raw_scores[ticker] = raw_momentum

        vol_window = series.iloc[-(VOL_LOOKBACK + 1):]
        daily_returns = vol_window.pct_change().dropna()
        daily_vol = daily_returns.std()
        annualized_vol = daily_vol * (TRADING_DAYS_PER_YEAR ** 0.5)

        if annualized_vol is None or annualized_vol <= 0 or pd.isna(annualized_vol):
            continue

        scores[ticker] = (raw_momentum / annualized_vol) * crypto_scale

        sma_window = series.iloc[-sma_period:]
        above_sma[ticker] = current_price > sma_window.mean()

        # Chaikin Money Flow (CMF) directional volume confirmation, mirroring
        # the algo exactly: money flow multiplier per day = ((Close-Low)-(High-Close))/(High-Low),
        # weighted by that day's volume, summed over DIRECTIONAL_VOLUME_LOOKBACK days.
        if ticker not in highs.columns or ticker not in lows.columns or ticker not in volumes.columns:
            positive_volume_flow[ticker] = False
            continue

        high_series = highs[ticker].dropna()
        low_series = lows[ticker].dropna()
        volume_series = volumes[ticker].dropna()

        vol_lookback_window = DIRECTIONAL_VOLUME_LOOKBACK
        recent_close = series.iloc[-vol_lookback_window:]
        recent_high = high_series.iloc[-vol_lookback_window:]
        recent_low = low_series.iloc[-vol_lookback_window:]
        recent_volume = volume_series.iloc[-vol_lookback_window:]

        if (
            len(recent_close) < vol_lookback_window
            or len(recent_high) < vol_lookback_window
            or len(recent_low) < vol_lookback_window
            or len(recent_volume) < vol_lookback_window
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

    return scores, above_sma, raw_scores, current_prices, positive_volume_flow


def apply_weight_cap(weights, max_weight, exempt_keys=None, route_excess_to_exempt=False):
    """
    Mirrors _apply_weight_cap exactly, including the route_excess_to_exempt
    behavior: when True, excess above the cap is routed straight to the
    exempt keys (SHV) instead of being redistributed across other uncapped,
    non-exempt entries first. Used when breadth is at or below
    MIN_POSITIONS, where inflating the other thin-breadth real assets to
    fill the gap would distort their momentum-earned proportions (or, in the
    worst case, force them all the way to equal-weight when MAX_ASSET_WEIGHT
    happens to equal 1/MIN_POSITIONS).
    """
    exempt_keys = exempt_keys or set()
    weights = dict(weights)

    for _ in range(10):
        capped_keys = [k for k, w in weights.items() if k not in exempt_keys and w > max_weight]
        if not capped_keys:
            break
        excess = sum(weights[k] - max_weight for k in capped_keys)
        for k in capped_keys:
            weights[k] = max_weight

        if route_excess_to_exempt:
            # Skip redistributing to other real assets - go straight to the
            # exempt fallback below.
            redistribute_keys = []
            redistribute_sum = 0
        else:
            redistribute_keys = [k for k in weights if k not in capped_keys and k not in exempt_keys]
            redistribute_sum = sum(weights[k] for k in redistribute_keys)

        if redistribute_sum > 0:
            for k in redistribute_keys:
                weights[k] += excess * (weights[k] / redistribute_sum)
        else:
            exempt_present = [k for k in weights if k in exempt_keys]
            if exempt_present:
                exempt_sum = sum(weights[k] for k in exempt_present)
                for k in exempt_present:
                    share = (weights[k] / exempt_sum) if exempt_sum > 0 else (1.0 / len(exempt_present))
                    weights[k] += excess * share
            # else: nowhere to put it - excess just isn't allocated (uninvested cash)
    return weights


def compute_signals():
    """Runs the full pipeline and returns a DataFrame of per-asset signals
    plus the resulting target weights dict."""
    # Fetched as three separate yf.download() calls - one per asset group -
    # mirroring the algo's three separate History() requests (equity, bond,
    # crypto) in _score_symbol_group/CalculateMomentumScores. Crypto trades
    # on a 24/7 calendar (a bar every calendar day) while equities and bonds
    # only produce bars on trading days; pulling them together into one
    # yf.download call would build a single shared date index across both
    # calendars, which is exactly the misaligned-index/bar-count risk the
    # algo's own comments call out for keeping crypto history separate. SHV
    # is fetched alongside the bond group since it trades on the same
    # calendar as the other bonds and is scored as part of that group.
    equity_ohlcv = fetch_ohlcv(EQUITY_TICKERS)
    bond_ohlcv = fetch_ohlcv(BOND_TICKERS)  # BOND_TICKERS already includes SHV
    crypto_ohlcv = fetch_ohlcv(CRYPTO_TICKERS)

    equity_scores, equity_sma, equity_raw, equity_px, equity_vol_flow = score_group(
        equity_ohlcv, EQUITY_TICKERS, SMA_PERIOD
    )
    bond_scores, bond_sma, bond_raw, bond_px, bond_vol_flow = score_group(
        bond_ohlcv, BOND_TICKERS, BOND_SMA_PERIOD
    )
    crypto_scores, crypto_sma, crypto_raw, crypto_px, crypto_vol_flow = score_group(
        crypto_ohlcv, CRYPTO_TICKERS, SMA_PERIOD, crypto_scale=CRYPTO_MOMENTUM_SCALE
    )

    scores = {**equity_scores, **bond_scores, **crypto_scores}
    above_sma = {**equity_sma, **bond_sma, **crypto_sma}
    raw_scores = {**equity_raw, **bond_raw, **crypto_raw}
    current_prices = {**equity_px, **bond_px, **crypto_px}
    positive_volume_flow = {**equity_vol_flow, **bond_vol_flow, **crypto_vol_flow}

    # Requires momentum > 0, above the trend-filter SMA, AND positive Chaikin
    # Money Flow (net buying pressure) - matches the algo's three-gate filter
    positive_scores = {
        t: s for t, s in scores.items()
        if s > 0 and above_sma.get(t, False) and positive_volume_flow.get(t, False)
    }
    num_selected = len(positive_scores)

    if num_selected == 0:
        target_weights = {SHV_TICKER: 1.0}
    elif num_selected < MIN_POSITIONS:
        score_sum = sum(positive_scores.values())
        occupied_fraction = num_selected / MIN_POSITIONS
        shv_fraction = 1.0 - occupied_fraction
        target_weights = {t: (s / score_sum) * occupied_fraction for t, s in positive_scores.items()}
        # Add the gap-filler share on top of whatever SHV may have already
        # earned above as one of the qualifying positive_scores (SHV can be
        # in there now that it's part of BOND_TICKERS) - never overwrite it.
        target_weights[SHV_TICKER] = target_weights.get(SHV_TICKER, 0.0) + shv_fraction
    else:
        score_sum = sum(positive_scores.values())
        target_weights = {t: s / score_sum for t, s in positive_scores.items()}
        # SHV can be one of these positive_scores now that it's part of
        # BOND_TICKERS - if so it just gets an ordinary momentum-proportional
        # weight here like any other qualifier, same as the algo.

    # Matches the algo exactly: when breadth is at or below MIN_POSITIONS,
    # the real (non-SHV) qualifying assets are still capped at
    # MAX_ASSET_WEIGHT, but any excess from that capping is routed directly
    # to SHV instead of being redistributed across the other thin-breadth
    # real assets - preserves their relative momentum proportions rather
    # than inflating them to fill the gap. SHV needs an entry in
    # target_weights to receive this even when it isn't independently
    # qualifying and isn't acting as a fractional gap-filler (num_selected
    # == MIN_POSITIONS exactly, where shv_fraction above is 0).
    if num_selected <= MIN_POSITIONS:
        if SHV_TICKER not in target_weights:
            target_weights[SHV_TICKER] = 0.0
        exempt_keys = {SHV_TICKER}
        route_excess_to_exempt = True
    else:
        # Enough real breadth qualifies on its own - cap everyone (including
        # SHV, if it's one of them) and let excess redistribute across other
        # uncapped candidates first, same as always.
        exempt_keys = set()
        route_excess_to_exempt = False

    target_weights = apply_weight_cap(
        target_weights, MAX_ASSET_WEIGHT,
        exempt_keys=exempt_keys, route_excess_to_exempt=route_excess_to_exempt
    )

    # Build the display table - every asset in the universe, not just qualifiers
    rows = []
    for ticker in EQUITY_TICKERS + BOND_TICKERS + CRYPTO_TICKERS:
        display_name = CRYPTO_DISPLAY_NAMES.get(ticker, ticker)
        qualifies = ticker in positive_scores
        rows.append({
            "Symbol": display_name,
            "Price": current_prices.get(ticker, np.nan),
            "Raw Momentum": raw_scores.get(ticker, np.nan),
            "Vol-Adj Score": scores.get(ticker, np.nan),
            "Above SMA": above_sma.get(ticker, False),
            "Positive CMF": positive_volume_flow.get(ticker, False),
            "Qualifies": qualifies,
            "Target Weight": target_weights.get(ticker, 0.0),
        })
    # No separate SHV row needed - it's already covered by the loop above
    # since BOND_TICKERS includes it, with real scores/SMA/CMF like any other
    # bond. Its "Qualifies" reflects whether it actually passed the filters;
    # its "Target Weight" reflects momentum-earned share plus any fallback
    # share, which can be nonzero even when Qualifies is False.

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
        html.H2("Live Momentum Signals", style={"color": COLORS["text"]}),
        html.P(
            "Mirrors MultiAssetMomentumAlgo.py's scoring/weighting logic against current market data. "
            "Not connected to a brokerage - this shows what the algo would signal if a rebalance ran today.",
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
                {"name": "Price", "id": "Price", "type": "numeric", "format": {"specifier": ",.2f"}},
                {"name": "Raw Momentum", "id": "Raw Momentum", "type": "numeric", "format": {"specifier": ".2%"}},
                {"name": "Vol-Adj Score", "id": "Vol-Adj Score", "type": "numeric", "format": {"specifier": ".3f"}},
                {"name": "Above SMA", "id": "Above SMA"},
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

    shv_weight = target_weights.get(SHV_TICKER, 0.0)
    summary = [
        card("Assets Qualifying", f"{num_selected} / {len(EQUITY_TICKERS + BOND_TICKERS + CRYPTO_TICKERS)}"),
        card("SHV Weight", f"{shv_weight:.1%}"),
        card("Min Positions Rule", "Active" if num_selected < MIN_POSITIONS else "Not needed"),
    ]

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return f"Last updated: {timestamp}", summary, signals_df.to_dict("records")


if __name__ == "__main__":
    app.run(debug=True)
