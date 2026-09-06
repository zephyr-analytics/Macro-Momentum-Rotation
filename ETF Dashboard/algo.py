# region imports
from AlgorithmImports import *
import pandas as pd
import numpy as np
# endregion

class MultiAssetMomentumAlgorithm(QCAlgorithm):
    """
    Momentum-weighted allocation across a fixed multi-asset ETF + crypto universe.

    Universe:
        VTI    - US Total Market Equity
        VEA    - Developed ex-US Equity
        VWO    - Emerging Markets Equity
        IXN    - Global Technology Equity
        BND    - US Total Bond Market
        BNDX   - International Bonds (Hedged)
        EMB    - Emerging Market USD Bonds
        SHV    - Short-term Treasury Bills (regular bond-sleeve asset; also
                 serves as the cash-like fallback for the MIN_POSITIONS
                 gap-filler and the no-qualifiers case - see "SHV's dual
                 role" below)
        GLD    - Gold
        PDBC   - Broad Commodities
        BTCUSD - Bitcoin (spot crypto pair)
        ETHUSD - Ethereum (spot crypto pair)

    Momentum definition:
        For each asset, compute total return over 21, 63, 126, 189, and 252
        trading days. Momentum score = simple average of those 5 returns,
        then divided by the asset's trailing realized volatility (annualized
        std dev of daily returns over VOL_LOOKBACK days) to produce a
        volatility-adjusted (risk-adjusted) momentum score. This keeps a
        low-vol bond ETF from being automatically outranked by a high-vol
        asset (crypto, gold, tech) that merely moved more in raw terms.
        Crypto (BTCUSD, ETHUSD) scores are then further multiplied by
        CRYPTO_MOMENTUM_SCALE (0.20x) to dampen their still-larger swings
        even after volatility adjustment.

    Trend filter:
        In addition to the momentum score, an asset must be trading above
        its own simple moving average to be eligible - SMA_PERIOD (168 days)
        for equities/crypto, BOND_SMA_PERIOD for the bond sleeve (BND, BNDX,
        EMB, SHV), since bonds trend more slowly and a single shared SMA
        period may not suit their behavior as well as it suits
        equities/crypto. This acts as a trend/regime confirmation on top of
        the momentum ranking.

    Weighting:
        - Any asset with momentum score <= 0 gets zero weight (excluded).
        - Any asset trading below its own trend-filter SMA gets zero weight
          (excluded), regardless of momentum score.
        - Any asset without net positive Chaikin Money Flow (CMF > 0) over
          the trailing DIRECTIONAL_VOLUME_LOOKBACK days gets zero weight
          (excluded), regardless of momentum/SMA.
        - Assets that pass both filters are weighted proportionally to
          their momentum score (score / sum of positive scores), i.e.
          stronger momentum gets more weight.
        - Minimum breadth rule: if fewer than MIN_POSITIONS assets pass both
          filters, SHV is used as a cash-like fill so the portfolio still
          behaves as if it held MIN_POSITIONS equal "slots" - the qualifying
          assets keep their relative momentum weighting but only occupy
          (N / MIN_POSITIONS) of the portfolio, and SHV takes the remaining
          (MIN_POSITIONS - N) / MIN_POSITIONS.
        - If no assets pass both filters, the portfolio goes 100% to SHV.

    SHV's dual role:
        SHV is scored, SMA-filtered, and CMF-filtered exactly like the other
        bond-sleeve assets (BND, BNDX, EMB) and can win weight purely on its
        own momentum like any other candidate. Independently of that, SHV
        also remains the designated cash-like target for two fallback
        mechanisms: the MIN_POSITIONS gap-filler and the "nothing qualifies"
        case. When both apply at once - e.g. SHV qualifies on its own
        momentum AND the gap-filler is also active - its final weight is the
        SUM of its momentum-earned share and its fallback share, never one
        overwriting the other.

    Weight cap:
        A hard MAX_ASSET_WEIGHT cap is applied to every real (non-SHV)
        scored asset after weighting, preventing any single score from
        taking an outsized share of the portfolio. When num_selected >
        MIN_POSITIONS (enough real breadth qualifies on its own), any excess
        above the cap is redistributed proportionally to the other scored
        assets first, and only overflows into SHV if no scored asset has
        room left. When num_selected <= MIN_POSITIONS, the real assets are
        still capped, but excess is routed directly to SHV instead of being
        spread across the other thin-breadth real assets - redistributing it
        to them would distort their momentum-earned proportions (or, e.g.
        when MAX_ASSET_WEIGHT happens to equal 1/MIN_POSITIONS, collapse the
        whole allocation to equal-weight regardless of the actual scores).
        SHV itself is exempt from the cap in this low-breadth regime, same
        as before, and keeps its existing dual role - its own momentum-
        earned share, plus any gap-filler share, plus any cap overflow, all
        summed rather than overwriting one another.


    Rebalance:
        Weekly, on week-end (equity calendar).

    Brokerage:
        No explicit brokerage/margin model is set - equities/SHV and
        BTCUSD/ETHUSD still use QuantConnect's default fill/margin models
        per security type (equity margin behavior vs. crypto cash-account,
        no-leverage behavior, handled independently since there's no single
        unified brokerage spanning both asset classes). Fees, however, are
        explicitly zeroed out on every security (see "Fee override" below)
        rather than left on QC's default per-asset-class fee model.

    Fee override:
        Every security added in Initialize (equities, bonds/SHV, and crypto)
        has its fee model explicitly set to ConstantFeeModel(0) - a flat $0
        commission on every fill, replacing whatever QC's default
        equity/crypto fee model would otherwise charge. This isolates the
        strategy's signal/weighting performance from transaction-cost
        assumptions; re-enable a nonzero fee model before treating results
        as a realistic estimate of live trading costs.
    """

    # Lookback periods (in trading days) used to build the momentum score
    LOOKBACKS = [21, 63, 126, 189, 252]

    # Trailing window (in trading days) used to measure realized volatility
    # for risk-adjusting the momentum score
    VOL_LOOKBACK = 126

    # Trading days per year, used to annualize the realized volatility
    TRADING_DAYS_PER_YEAR = 252

    # SMA trend-filter period (in trading days) for equities (ex-bonds) and crypto
    SMA_PERIOD = 168

    # SMA trend-filter period (in trading days) for the bond sleeve (BND, BNDX, EMB, SHV) -
    # kept independent since bonds trend more slowly than equities/crypto, and a
    # single shared SMA period doesn't necessarily fit both well. Defaulted to a
    # longer window than SMA_PERIOD as a starting point - tune as needed.
    BOND_SMA_PERIOD = 126

    # Minimum number of qualifying assets before SHV is used to fill the gap
    MIN_POSITIONS = 3

    # Crypto momentum scores are multiplied by this before ranking/weighting
    # (dampens crypto's typically much larger raw momentum swings vs. the ETFs)
    CRYPTO_MOMENTUM_SCALE = 0.20

    # Hard cap on how much of the portfolio any single asset can take, applied
    # after weighting - prevents any one score from structurally dominating
    # the allocation
    MAX_ASSET_WEIGHT = 0.25

    # --- Directional volume confirmation (Chaikin Money Flow) ---
    # Trailing window (in trading days) over which Chaikin Money Flow (CMF) is
    # computed. CMF captures where each day's close falls within its high-low
    # range, weighted by that day's volume - richer than a plain up/down-day
    # volume split, since it also reflects how strong each day's close was,
    # not just its direction. An asset must show CMF > 0 (net buying
    # pressure) over this window to qualify, on top of the existing
    # momentum/SMA filters.
    DIRECTIONAL_VOLUME_LOOKBACK = 63

    TICKERS = ["VTI", "VEA", "VWO", "IXN", "GLD", "PDBC"]
    BOND_TICKERS = ["BND", "BNDX", "EMB", "SHV"]
    CRYPTO_TICKERS = ["BTCUSD", "ETHUSD"]
    SHV_TICKER = "SHV"

    def Initialize(self):
        self.SetStartDate(2012, 1, 1)
        self.SetEndDate(2026, 9, 1)
        self.SetCash(100000)

        self.equity_symbols = []
        for ticker in self.TICKERS:
            equity = self.AddEquity(ticker, Resolution.Daily)
            equity.SetDataNormalizationMode(DataNormalizationMode.TotalReturn)
            # Zero out fees so backtest performance isolates the signal/weighting
            # logic from transaction-cost assumptions - see "Fee override" above.
            equity.SetFeeModel(ConstantFeeModel(0))
            self.equity_symbols.append(equity.Symbol)

        # Bonds are kept in their own list/history requests so they can use a
        # different SMA trend-filter period (BOND_SMA_PERIOD) than the rest of
        # the equity sleeve (SMA_PERIOD). SHV lives here too - it's scored and
        # filtered exactly like BND/BNDX/EMB (see "SHV's dual role" above).
        self.bond_symbols = []
        for ticker in self.BOND_TICKERS:
            bond = self.AddEquity(ticker, Resolution.Daily)
            bond.SetDataNormalizationMode(DataNormalizationMode.TotalReturn)
            # Zero out fees - see "Fee override" above.
            bond.SetFeeModel(ConstantFeeModel(0))
            self.bond_symbols.append(bond.Symbol)

        # Actual spot crypto pairs - kept in their own list/history requests since
        # crypto trades on a 24/7 calendar while equities only have trading-day bars.
        # No brokerage model set, so these use QC's default crypto fill/margin models
        # (cash-account, no leverage); fees are zeroed out below regardless - see
        # "Fee override" above.
        self.crypto_symbols = []
        if self.CRYPTO_TICKERS is not None:
            for ticker in self.CRYPTO_TICKERS:
                crypto = self.AddCrypto(ticker, Resolution.Daily)
                crypto.SetFeeModel(ConstantFeeModel(0))
                self.crypto_symbols.append(crypto.Symbol)

        # Combined ranked universe (equities + bonds + crypto), used for weighting/liquidation logic.
        # SHV is included here (via bond_symbols) since it's a regular scored candidate now.
        self.symbols = self.equity_symbols + self.bond_symbols + self.crypto_symbols

        # SHV also keeps a dedicated reference (self.shv_symbol) purely so the
        # gap-filler / no-qualifiers logic has something to target - it is NOT
        # a separately-added security. Pulled from bond_symbols rather than
        # re-added, since it's already part of the bond sleeve above.
        self.shv_symbol = next(
            symbol for symbol in self.bond_symbols if symbol.Value == self.SHV_TICKER
        )

        # Need enough history for the longest lookback / SMA / vol window plus a small buffer
        self.SetWarmUp(
            max(max(self.LOOKBACKS), self.SMA_PERIOD, self.BOND_SMA_PERIOD, self.VOL_LOOKBACK) + 5,
            Resolution.Daily
        )

        # Rebalance weekly, on week-end (driven off the equity calendar; crypto
        # trades 24/7 but this keeps the schedule aligned to the ETF rebalance day)
        self.Schedule.On(
            self.DateRules.MonthEnd(self.equity_symbols[0]),
            self.TimeRules.BeforeMarketClose(self.equity_symbols[0], 30),
            self.Rebalance
        )

        self.rebalance_flag = False

    def OnData(self, data: Slice):
        pass

    def Rebalance(self):
        if self.IsWarmingUp:
            return

        scores, above_sma, positive_volume_flow = self.CalculateMomentumScores()

        # Keep only assets with strictly positive momentum, trading above their
        # own trend-filter SMA, AND showing net positive Chaikin Money Flow
        # over DIRECTIONAL_VOLUME_LOOKBACK - real buying pressure behind the trend.
        # SHV is scored alongside the other bonds, so it can appear here too if
        # it independently qualifies.
        positive_scores = {
            symbol: score for symbol, score in scores.items()
            if score > 0.0 and above_sma.get(symbol, False) and positive_volume_flow.get(symbol, False)
        }

        num_selected = len(positive_scores)

        if num_selected == 0:
            # Nothing qualifies - go fully into SHV instead of sitting in raw cash
            self.Log("No assets qualified (momentum/SMA/volume) - moving 100% to SHV.")
            target_weights = {self.shv_symbol: 1.0}
        else:
            score_sum = sum(positive_scores.values())

            if num_selected < self.MIN_POSITIONS:
                # Scale the qualifying assets down to occupy only their share of
                # MIN_POSITIONS "slots"; SHV fills the remaining slots as cash-like exposure
                occupied_fraction = num_selected / self.MIN_POSITIONS
                shv_fraction = 1.0 - occupied_fraction

                target_weights = {
                    symbol: (score / score_sum) * occupied_fraction
                    for symbol, score in positive_scores.items()
                }

                # Add the gap-filler share on top of whatever SHV may have
                # already earned above as one of the qualifying positive_scores
                # (rather than overwriting it) - SHV's total weight is the sum
                # of its own momentum share plus the filler share.
                target_weights[self.shv_symbol] = target_weights.get(self.shv_symbol, 0.0) + shv_fraction

                self.Log(
                    f"Only {num_selected}/{self.MIN_POSITIONS} assets qualified - "
                    f"filling {shv_fraction:.2%} with SHV."
                )
            else:
                target_weights = {symbol: score / score_sum for symbol, score in positive_scores.items()}

        # When breadth is at or below MIN_POSITIONS, the real (non-SHV)
        # qualifying assets are still capped at MAX_ASSET_WEIGHT like normal -
        # but any excess from that capping is routed directly to SHV instead
        # of being redistributed across the other thin-breadth real assets.
        # Redistributing to the others would just inflate weak performers to
        # fill the gap (and, e.g. when MAX_ASSET_WEIGHT == 1/MIN_POSITIONS,
        # collapses the whole allocation to equal-weight regardless of how
        # skewed the actual momentum scores are); sending it to SHV instead
        # preserves the real assets' relative momentum proportions and just
        # parks the slack in cash-like exposure, same as the gap-filler share
        # already does. SHV needs an entry in target_weights to receive this
        # even when it isn't independently qualifying and isn't acting as a
        # fractional gap-filler (num_selected == MIN_POSITIONS exactly, where
        # shv_fraction above is 0).
        if num_selected <= self.MIN_POSITIONS:
            if self.shv_symbol not in target_weights:
                target_weights[self.shv_symbol] = 0.0
            exempt_keys = {self.shv_symbol}
            route_excess_to_exempt = True
        else:
            # Enough real breadth qualifies on its own - cap everyone
            # (including SHV, if it's one of them) and let excess
            # redistribute across other uncapped candidates first, same as
            # always.
            exempt_keys = set()
            route_excess_to_exempt = False

        # Apply the hard per-asset weight cap to the scored assets (and to SHV
        # too, once it's no longer acting as the gap-filler)
        target_weights = self._apply_weight_cap(
            target_weights, self.MAX_ASSET_WEIGHT,
            exempt_keys=exempt_keys, route_excess_to_exempt=route_excess_to_exempt
        )

        # Log the allocation for transparency
        weight_str = ", ".join([f"{symbol.Value}: {weight:.2%}" for symbol, weight in target_weights.items()])
        self.Log(f"Rebalance {self.Time.date()} -> {weight_str}")

        # Debug line: assets in the portfolio and their target weight
        self.Debug(f"{self.Time.date()} PORTFOLIO -> " + weight_str)

        targets = [PortfolioTarget(symbol, weight) for symbol, weight in target_weights.items()]

        # Liquidate anything currently held that's no longer in the target set.
        # self.symbols already includes SHV (via bond_symbols), so no need to
        # append self.shv_symbol separately here.
        for symbol in self.symbols:
            if symbol not in target_weights and self.Portfolio[symbol].Invested:
                self.Liquidate(symbol)

        self.SetHoldings(targets)

    def _apply_weight_cap(self, weights, max_weight, exempt_keys=None, route_excess_to_exempt=False):
        """
        Caps each entry in `weights` at `max_weight`. Keys in `exempt_keys`
        (e.g. SHV) are never capped themselves.

        By default (route_excess_to_exempt=False), excess above the cap is
        redistributed proportionally across the other uncapped, non-exempt
        entries first, and only overflows into the exempt keys if no
        non-exempt entry has room left - this avoids an infeasible/
        oscillating result when there are too few scored assets to absorb
        the excess while everyone stays under the cap (e.g. only 1 real
        asset qualifying plus SHV: capping SHV too would force its excess
        back onto the single real asset, blowing it past the cap, which
        would then bounce back onto SHV next iteration, and so on).

        When route_excess_to_exempt=True, excess above the cap is routed
        straight to the exempt keys instead, skipping redistribution to
        other non-exempt entries entirely - used when breadth is at or
        below MIN_POSITIONS, where inflating the other thin-breadth real
        assets to fill the gap would distort their momentum-earned
        proportions (or, in the worst case, force them all the way to
        equal-weight); sending the excess to SHV instead preserves those
        proportions and just parks the slack in cash-like exposure.
        """
        exempt_keys = exempt_keys or set()
        weights = dict(weights)

        for _ in range(10):
            capped_keys = [
                key for key, weight in weights.items()
                if key not in exempt_keys and weight > max_weight
            ]
            if not capped_keys:
                break

            excess = sum(weights[key] - max_weight for key in capped_keys)
            for key in capped_keys:
                weights[key] = max_weight

            if route_excess_to_exempt:
                # Skip redistributing to other real assets - go straight to
                # the exempt fallback below.
                redistribute_keys = []
                redistribute_sum = 0
            else:
                # Redistribute into other uncapped, non-exempt assets first
                redistribute_keys = [
                    key for key in weights
                    if key not in capped_keys and key not in exempt_keys
                ]
                redistribute_sum = sum(weights[key] for key in redistribute_keys)

            if redistribute_sum > 0:
                for key in redistribute_keys:
                    weights[key] += excess * (weights[key] / redistribute_sum)
            else:
                # Nothing left to redistribute into (or route_excess_to_exempt
                # skipped that step) - dump the excess into the exempt keys
                # (SHV), split proportionally, since that's the flexible
                # catch-all bucket
                exempt_present = [key for key in weights if key in exempt_keys]
                if exempt_present:
                    exempt_sum = sum(weights[key] for key in exempt_present)
                    for key in exempt_present:
                        share = (weights[key] / exempt_sum) if exempt_sum > 0 else (1.0 / len(exempt_present))
                        weights[key] += excess * share
                # else: nowhere to put it - excess just isn't allocated (uninvested cash)

        return weights

    def CalculateMomentumScores(self):
        """
        Returns a tuple (scores, above_sma, positive_volume_flow):
            scores               - dict {symbol: momentum_score}, the
                                    volatility-adjusted momentum (raw average
                                    return over self.LOOKBACKS periods,
                                    divided by trailing annualized realized
                                    volatility).
            above_sma            - dict {symbol: bool}, True if the current
                                    price is above the trailing SMA -
                                    SMA_PERIOD for equities and crypto,
                                    BOND_SMA_PERIOD for the bond sleeve
                                    (including SHV).
            positive_volume_flow - dict {symbol: bool}, True if Chaikin Money
                                    Flow (CMF) over the trailing
                                    DIRECTIONAL_VOLUME_LOOKBACK days is
                                    positive (net buying pressure).

        Equities, bonds, and crypto are pulled via separate History() requests
        and scored independently: crypto trades on a 24/7 calendar (daily bars
        every calendar day) while equities/bonds only produce bars on trading
        days, so mixing them into a single request risks misaligned indices
        and bar counts; bonds (including SHV) are split out from the rest of
        the equity sleeve so they can use their own SMA_PERIOD
        (BOND_SMA_PERIOD) independent of the equities/crypto trend-filter
        window.
        """
        scores = {}
        above_sma = {}
        positive_volume_flow = {}

        equity_scores, equity_above_sma, equity_volume_flow = self._score_symbol_group(
            self.equity_symbols, self.SMA_PERIOD
        )
        scores.update(equity_scores)
        above_sma.update(equity_above_sma)
        positive_volume_flow.update(equity_volume_flow)

        bond_scores, bond_above_sma, bond_volume_flow = self._score_symbol_group(
            self.bond_symbols, self.BOND_SMA_PERIOD
        )
        scores.update(bond_scores)
        above_sma.update(bond_above_sma)
        positive_volume_flow.update(bond_volume_flow)

        crypto_scores, crypto_above_sma, crypto_volume_flow = self._score_symbol_group(
            self.crypto_symbols, self.SMA_PERIOD
        )

        # Dampen crypto's momentum score by CRYPTO_MOMENTUM_SCALE (0.20x) before it
        # competes with equities for weight - crypto momentum tends to run much
        # hotter/more volatile than the ETF sleeve, so this keeps it from
        # dominating the ranking purely on raw magnitude.
        crypto_scores = {symbol: score * self.CRYPTO_MOMENTUM_SCALE for symbol, score in crypto_scores.items()}

        scores.update(crypto_scores)
        above_sma.update(crypto_above_sma)
        positive_volume_flow.update(crypto_volume_flow)

        return scores, above_sma, positive_volume_flow

    def _score_symbol_group(self, symbols, sma_period):
        """
        Runs a single History() request for the given symbol group and computes
        volatility-adjusted momentum score + SMA trend flag (using the given
        sma_period) + directional volume confirmation for each. Used
        separately for the equity, bond, and crypto groups so their history
        requests never mix and each can use its own SMA trend-filter window.
        """
        scores = {}
        above_sma = {}
        positive_volume_flow = {}

        if len(symbols) == 0:
            return scores, above_sma, positive_volume_flow

        max_lookback = max(max(self.LOOKBACKS), sma_period, self.VOL_LOOKBACK, self.DIRECTIONAL_VOLUME_LOOKBACK)
        history = self.History(symbols, max_lookback + 1, Resolution.Daily)

        if history.empty:
            self.Log("History request returned empty for symbol group - skipping.")
            return scores, above_sma, positive_volume_flow

        for symbol in symbols:
            try:
                close_prices = history.loc[symbol]["close"]
            except KeyError:
                self.Log(f"No history for {symbol.Value} - excluding from scoring.")
                continue

            if len(close_prices) < max_lookback + 1:
                self.Log(f"Insufficient history for {symbol.Value} ({len(close_prices)} bars) - excluding.")
                continue

            current_price = close_prices.iloc[-1]

            # Raw momentum score - simple average of trailing returns
            returns = []
            for lookback in self.LOOKBACKS:
                past_price = close_prices.iloc[-1 - lookback]
                if past_price <= 0:
                    continue
                ret = (current_price / past_price) - 1.0
                returns.append(ret)

            if len(returns) != len(self.LOOKBACKS):
                continue

            raw_momentum = sum(returns) / len(returns)

            # Realized volatility over the trailing VOL_LOOKBACK days, annualized
            vol_window = close_prices.iloc[-(self.VOL_LOOKBACK + 1):]
            daily_returns = vol_window.pct_change().dropna()
            daily_vol = daily_returns.std()
            annualized_vol = daily_vol * (self.TRADING_DAYS_PER_YEAR ** 0.5)

            if annualized_vol is None or annualized_vol <= 0 or pd.isna(annualized_vol):
                self.Log(f"Zero/invalid volatility for {symbol.Value} - excluding from scoring.")
                continue

            # Volatility-adjusted momentum: raw momentum scaled down for how much
            # the asset actually moved to get there - keeps low-vol bonds from
            # being structurally outranked by high-vol crypto/gold/tech
            scores[symbol] = raw_momentum / annualized_vol

            # SMA trend filter (still based on raw price, unaffected by vol adjustment) -
            # uses this group's sma_period (SMA_PERIOD for equities/crypto, BOND_SMA_PERIOD for bonds)
            sma_window = close_prices.iloc[-sma_period:]
            sma_value = sma_window.mean()
            above_sma[symbol] = current_price > sma_value

            # Chaikin Money Flow (CMF) directional volume confirmation, over
            # DIRECTIONAL_VOLUME_LOOKBACK trading days. For each day, the money
            # flow multiplier ((Close-Low)-(High-Close))/(High-Low) captures
            # where the close landed within that day's range (+1 = closed at
            # the high, -1 = closed at the low) - richer than a plain up/down
            # day split, since it also reflects how strong the close was, not
            # just its direction. Money flow volume = multiplier * volume;
            # CMF = sum(money flow volume) / sum(volume) over the window.
            # Requires CMF > 0 (net buying pressure) to qualify.
            try:
                volume = history.loc[symbol]["volume"]
                high_prices = history.loc[symbol]["high"]
                low_prices = history.loc[symbol]["low"]
            except KeyError:
                self.Log(f"No volume/high/low data for {symbol.Value} - excluding from volume confirmation.")
                positive_volume_flow[symbol] = False
                continue

            vol_lookback_window = self.DIRECTIONAL_VOLUME_LOOKBACK
            recent_close = close_prices.iloc[-vol_lookback_window:]
            recent_high = high_prices.iloc[-vol_lookback_window:]
            recent_low = low_prices.iloc[-vol_lookback_window:]
            recent_volume = volume.iloc[-vol_lookback_window:]

            if (
                len(recent_close) < vol_lookback_window
                or len(recent_high) < vol_lookback_window
                or len(recent_low) < vol_lookback_window
                or len(recent_volume) < vol_lookback_window
            ):
                positive_volume_flow[symbol] = False
                continue

            high_low_range = recent_high - recent_low
            # A day with High == Low (no range - e.g. a halted/illiquid bar) has an
            # undefined multiplier; treat its money flow multiplier as 0 rather than
            # dividing by zero, so it contributes volume to the denominator but no
            # directional signal to the numerator
            money_flow_multiplier = np.where(
                high_low_range > 0,
                ((recent_close - recent_low) - (recent_high - recent_close)) / high_low_range.replace(0, np.nan),
                0.0
            )
            money_flow_multiplier = pd.Series(money_flow_multiplier, index=recent_close.index).fillna(0.0)

            money_flow_volume = money_flow_multiplier * recent_volume
            total_volume = recent_volume.sum()

            if total_volume <= 0:
                positive_volume_flow[symbol] = False
                continue

            cmf = money_flow_volume.sum() / total_volume
            positive_volume_flow[symbol] = cmf > 0.0

        return scores, above_sma, positive_volume_flow
