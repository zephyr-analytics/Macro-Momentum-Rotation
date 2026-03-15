from AlgorithmImports import *
import numpy as np


class AbsoluteRelativeMomentum(QCAlgorithm):
    """Absolute + relative momentum strategy with volatility targeting.

    Ranks a universe of ETFs and crypto by composite momentum, applies an
    absolute-return filter against cash (BIL), gates entries with a
    simple-moving-average trend filter, and sizes the top-2 winning assets via
    realized-volatility targeting. Any unallocated weight is held in BIL.
    """

    def initialize(self) -> None:
        """Set up the algorithm: universe, config, schedule, and warm-up."""
        self.set_start_date(2012, 1, 1)
        self.set_cash(100_000)

        # ------------------------------------------------------------------ #
        # Universe
        # ------------------------------------------------------------------ #
        self._vti   = self.add_equity("VTI",   Resolution.DAILY).symbol
        self._vxus   = self.add_equity("VEU",   Resolution.DAILY).symbol
        self._vgit = self.add_equity("VGIT", Resolution.DAILY).symbol
        self._bndx = self.add_equity("BNDX", Resolution.DAILY).symbol
        self._gld  = self.add_equity("GLD",  Resolution.DAILY).symbol
        self._dbc  = self.add_equity("DBC",  Resolution.DAILY).symbol
        self._bil  = self.add_equity("BIL",  Resolution.DAILY).symbol
        self._bnd = self.add_equity("BND", Resolution.DAILY).symbol
        self._btc  = self.add_crypto("BTCUSD", Resolution.DAILY).symbol

        self._etfs: list = [
            self._vti, self._bnd, self._bndx, self._vgit, self._vxus,
            self._gld, self._dbc, self._bil
        ]

        self._assets: list = self._etfs + [self._btc]

        for symbol in self._assets:
            self.securities[symbol].set_fee_model(ConstantFeeModel(0))

        # ------------------------------------------------------------------ #
        # Hyperparameters
        # ------------------------------------------------------------------ #
        self._momentum_lookbacks: list[int] = [21, 63, 126, 189, 252]
        self._max_lookback: int = max(self._momentum_lookbacks)

        self._sma_period:   int   = 168
        self._sma_overrides: dict = {
            self._bnd:  126,
            self._bndx: 126,
            self._vgit: 126,
        }
        self._vol_lookback: int   = 63
        self._target_vol:   float = 0.20
        self._max_weight:   float = 1.0
        self._top_n:        int   = 2

        # ------------------------------------------------------------------ #
        # Schedule
        # ------------------------------------------------------------------ #
        self.schedule.on(
            self.date_rules.month_start(self._vti),
            self.time_rules.after_market_open(self._vti, 120),
            self._rebalance,
        )

        self.set_warmup(
            max(self._max_lookback + 1, self._sma_period, self._vol_lookback + 1),
            Resolution.DAILY,
        )

    # ---------------------------------------------------------------------- #
    # Signal helpers
    # ---------------------------------------------------------------------- #

    def _momentum(self, symbol, closes) -> float:
        """Compute composite momentum as the mean return over multiple lookbacks.

        Parameters
        ----------
        symbol : Symbol
            The asset whose momentum is being computed.
        closes : pd.DataFrame
            Wide-format DataFrame of close prices; columns are Symbol objects.

        Returns
        -------
        float
            Mean total return across all ``_momentum_lookbacks``.
            Returns ``-np.inf`` when data are insufficient.
        """
        if symbol not in closes.columns:
            return -np.inf

        px = closes[symbol].dropna()
        if len(px) < self._max_lookback + 1:
            return -np.inf

        return float(np.mean([
            px.iloc[-1] / px.iloc[-(lb + 1)] - 1
            for lb in self._momentum_lookbacks
        ]))

    def _passes_sma_gate(self, symbol, closes) -> bool:
        """Check whether the asset's last close is above its long-run SMA.

        BIL is unconditionally exempt from this filter. BND, BNDX, and VGIT
        use a 126-day SMA; all other assets use the default ``_sma_period``.

        Parameters
        ----------
        symbol : Symbol
            The asset to test.
        closes : pd.DataFrame
            Wide-format DataFrame of close prices.

        Returns
        -------
        bool
            ``True`` if the asset passes (or is BIL), ``False`` otherwise.
        """
        if symbol == self._bil:
            return True

        if symbol not in closes.columns:
            return False

        px = closes[symbol].dropna()
        sma_period = self._sma_overrides.get(symbol, self._sma_period)

        if len(px) < sma_period:
            return False

        sma = px.rolling(sma_period).mean().iloc[-1]
        if np.isnan(sma):
            return False

        return bool(px.iloc[-1] > sma)

    def _realized_vol(self, symbol, closes) -> float:
        """Estimate annualised realised volatility from log returns.

        Parameters
        ----------
        symbol : Symbol
            The asset to evaluate.
        closes : pd.DataFrame
            Wide-format DataFrame of close prices.

        Returns
        -------
        float
            Annualised volatility (σ × √252).
            Returns ``np.nan`` when data are insufficient.
        """
        if symbol not in closes.columns:
            return np.nan

        px = closes[symbol].dropna()
        if len(px) < self._vol_lookback + 1:
            return np.nan

        log_rets = np.log(px / px.shift(1)).dropna()
        if len(log_rets) < self._vol_lookback:
            return np.nan

        vol = log_rets.tail(self._vol_lookback).std()
        if vol is None or np.isnan(vol):
            return np.nan

        return float(vol * np.sqrt(252))

    def _absolute_return_6m(self, symbol, closes) -> float:
        """Compute the trailing six-month (126-day) total return.

        Parameters
        ----------
        symbol : Symbol
            The asset to evaluate.
        closes : pd.DataFrame
            Wide-format DataFrame of close prices.

        Returns
        -------
        float
            Six-month return as a decimal.
            Returns ``-np.inf`` when data are insufficient.
        """
        if symbol not in closes.columns:
            return -np.inf

        px = closes[symbol].dropna()
        if len(px) < 127:
            return -np.inf

        return float(px.iloc[-1] / px.iloc[-127] - 1)

    # ---------------------------------------------------------------------- #
    # Rebalance
    # ---------------------------------------------------------------------- #

    def _rebalance(self) -> None:
        """Monthly rebalance: filter, rank, size, and trade.

        ETFs and BTC are evaluated entirely independently — separate history
        pulls, separate close DataFrames, and separate eligibility checks —
        before being combined only at the final ranking step.

        Steps
        -----
        1. Pull total-return closes for ETFs; pull raw closes for BTC.
        2. Screen each universe independently via SMA gate and absolute-return
           vs. cash filter.
        3. Merge the two eligible lists, rank by composite momentum, and select
           the top ``_top_n`` assets.
        4. Size each winner using volatility targeting (cap per slot =
           ``_max_weight / _top_n``); park remainder in BIL.
        """
        if self.is_warming_up:
            return

        n_bars = self._max_lookback + 1

        # ------------------------------------------------------------------ #
        # Step 1 — separate history pulls
        # ------------------------------------------------------------------ #
        etf_history = self.history(
            self._etfs,
            n_bars,
            Resolution.DAILY,
            data_normalization_mode=DataNormalizationMode.TOTAL_RETURN,
        )
        btc_history = self.history(
            [self._btc],
            n_bars,
            Resolution.DAILY,
        )

        if etf_history.empty:
            return

        etf_closes = etf_history["close"].unstack(0)
        btc_closes = None
        if not btc_history.empty and "close" in btc_history.columns:
            btc_closes = btc_history["close"].unstack(0)

        # ------------------------------------------------------------------ #
        # Step 2 — eligibility screen (ETFs and BTC evaluated independently)
        # ------------------------------------------------------------------ #
        bil_ret_6m = self._absolute_return_6m(self._bil, etf_closes)

        eligible: list = []

        for symbol in self._etfs:
            if symbol not in etf_closes.columns:
                continue
            if not self._passes_sma_gate(symbol, etf_closes):
                continue
            if symbol == self._bil or self._absolute_return_6m(symbol, etf_closes) > bil_ret_6m:
                eligible.append((symbol, etf_closes))

        if (
            btc_closes is not None
            and self._btc in btc_closes.columns
            and self._passes_sma_gate(self._btc, btc_closes)
            and self._absolute_return_6m(self._btc, btc_closes) > bil_ret_6m
        ):
            eligible.append((self._btc, btc_closes))

        if not eligible:
            self.liquidate()
            self.set_holdings(self._bil, 1.0)
            return

        # ------------------------------------------------------------------ #
        # Step 3 — momentum ranking; select top N
        # ------------------------------------------------------------------ #
        scores: dict = {s: self._momentum(s, closes) for s, closes in eligible}

        for symbol, score in scores.items():
            self.log(f"{symbol.value} momentum={score:.2%}")

        sorted_eligible = sorted(eligible, key=lambda pair: scores[pair[0]], reverse=True)
        top_n = sorted_eligible[:self._top_n]

        # ------------------------------------------------------------------ #
        # Step 4 & 5 — volatility-targeted sizing and execution
        # ------------------------------------------------------------------ #
        # Per-slot weight cap scales with number of winners so combined
        # exposure stays within _max_weight.
        per_slot_cap = self._max_weight / len(top_n)

        self.liquidate()

        total_allocated = 0.0

        for winner, winner_closes in top_n:
            if winner == self._bil:
                weight = per_slot_cap
            else:
                vol = self._realized_vol(winner, winner_closes)
                weight = (
                    min(per_slot_cap, self._target_vol / vol)
                    if vol and not np.isnan(vol)
                    else 0.0
                )

            total_allocated += weight
            self.log(f"Selected={winner.value} | weight={weight:.2f}")
            self.set_holdings(winner, weight)

        cash_weight = max(0.0, 1.0 - total_allocated)
        self.log(f"Cash (BIL) weight={cash_weight:.2f}")
        self.set_holdings(self._bil, cash_weight)
