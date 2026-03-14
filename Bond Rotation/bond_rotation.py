from AlgorithmImports import *
import numpy as np


class CashBondMomentum(QCAlgorithm):
    """Cash & bond portfolio weighted by composite momentum × win rate.

    Universe : BIL, BND, BNDX, VGIT, ICSH
    Scoring  : composite_score = mean_momentum × mean_win_rate
                 where both metrics are averaged across the same
                 lookback periods [21, 63, 126, 189, 252].
    Sizing   : scores are normalised to sum to 1.0 (long-only).
                 Any asset with a non-positive composite score is
                 excluded and its weight is redistributed to BIL.
    """

    def initialize(self) -> None:
        self.set_start_date(2014, 1, 1)
        self.set_cash(100_000)

        # ------------------------------------------------------------------ #
        # Universe
        # ------------------------------------------------------------------ #
        self._bil  = self.add_equity("SHV",  Resolution.DAILY).symbol
        self._sgov  = self.add_equity("SGOV",  Resolution.DAILY).symbol
        self._bnd  = self.add_equity("BND",  Resolution.DAILY).symbol
        self._bndx = self.add_equity("BNDX", Resolution.DAILY).symbol
        self._vgit = self.add_equity("VGIT", Resolution.DAILY).symbol
        self._icsh = self.add_equity("ICSH", Resolution.DAILY).symbol

        self._assets: list = [
            self._bil, self._bnd, self._bndx, self._vgit, self._icsh, self._sgov
        ]

        for symbol in self._assets:
            self.securities[symbol].set_fee_model(ConstantFeeModel(0))

        # ------------------------------------------------------------------ #
        # Hyperparameters
        # ------------------------------------------------------------------ #
        self._lookbacks: list[int] = [21, 63, 126, 189, 252]
        self._max_lookback: int    = max(self._lookbacks)

        # ------------------------------------------------------------------ #
        # Schedule — monthly rebalance
        # ------------------------------------------------------------------ #
        self.schedule.on(
            self.date_rules.month_start(self._bil),
            self.time_rules.after_market_open(self._bil, 120),
            self._rebalance,
        )

        self.set_warmup(self._max_lookback + 1, Resolution.DAILY)

    # ---------------------------------------------------------------------- #
    # Signal helpers
    # ---------------------------------------------------------------------- #

    def _composite_momentum(self, px: "pd.Series") -> float:
        """Mean total return across all lookback periods.

        Returns
        -------
        float
            Mean return, or ``-np.inf`` if data are insufficient.
        """
        if len(px) < self._max_lookback + 1:
            return -np.inf

        return float(np.mean([
            px.iloc[-1] / px.iloc[-(lb + 1)] - 1
            for lb in self._lookbacks
        ]))

    def _composite_win_rate(self, px: "pd.Series") -> float:
        """Mean win rate (% of daily returns > 0) across all lookback periods.

        Returns
        -------
        float
            Mean win rate in [0, 1], or ``-np.inf`` if data are insufficient.
        """
        if len(px) < self._max_lookback + 1:
            return -np.inf

        daily_rets = px.pct_change().dropna()

        rates = []
        for lb in self._lookbacks:
            window = daily_rets.tail(lb)
            if len(window) < lb:
                continue
            rates.append(float((window > 0).mean()))

        if not rates:
            return -np.inf

        return float(np.mean(rates))

    def _passes_sma_gate(self, symbol, px: "pd.Series") -> bool:
        """Check whether the asset's last close is above its 126-day SMA.

        BIL/ICSH (cash proxies) are unconditionally exempt.

        Returns
        -------
        bool
            ``True`` if the asset passes or is a cash proxy.
        """
        if symbol in (self._bil, self._sgov):
            return True

        if len(px) < 126:
            return False

        sma = float(px.rolling(126).mean().iloc[-1])
        if np.isnan(sma):
            return False

        return bool(px.iloc[-1] > sma)

    def _absolute_return_6m(self, px: "pd.Series") -> float:
        """Trailing 126-day total return.

        Returns
        -------
        float
            Six-month return, or ``-np.inf`` if data are insufficient.
        """
        if len(px) < 127:
            return -np.inf
        return float(px.iloc[-1] / px.iloc[-127] - 1)

    # ---------------------------------------------------------------------- #
    # Rebalance
    # ---------------------------------------------------------------------- #

    def _rebalance(self) -> None:
        """Score every asset, normalise weights, and trade."""
        if self.is_warming_up:
            return

        n_bars = self._max_lookback + 1

        history = self.history(
            self._assets,
            n_bars,
            Resolution.DAILY,
            data_normalization_mode=DataNormalizationMode.TOTAL_RETURN,
        )

        if history.empty:
            return

        closes = history["close"].unstack(0)

        # ------------------------------------------------------------------ #
        # Absolute return filter — must beat BIL's 6m return to be scored
        # ------------------------------------------------------------------ #
        bil_px     = closes[self._bil].dropna() if self._bil in closes.columns else None
        bil_ret_6m = self._absolute_return_6m(bil_px) if bil_px is not None else -np.inf

        # ------------------------------------------------------------------ #
        # Score each asset
        # ------------------------------------------------------------------ #
        scores: dict = {}
        for symbol in self._assets:
            if symbol not in closes.columns:
                scores[symbol] = 0.0
                continue

            px = closes[symbol].dropna()

            # BIL always passes its own absolute-return check
            if symbol != self._bil:
                if not self._passes_sma_gate(symbol, px):
                    self.log(f"{symbol.value} | FILTERED (below 126d SMA)")
                    scores[symbol] = 0.0
                    continue

                ret_6m = self._absolute_return_6m(px)
                if ret_6m <= bil_ret_6m:
                    self.log(f"{symbol.value} | FILTERED (6m ret {ret_6m:.4f} <= BIL {bil_ret_6m:.4f})")
                    scores[symbol] = 0.0
                    continue

            mom      = self._composite_momentum(px)
            win_rate = self._composite_win_rate(px)

            if mom == -np.inf or win_rate == -np.inf:
                scores[symbol] = 0.0
                continue

            # Multiplicative combination; clip at zero so weak assets
            # don't receive allocation
            raw = mom * win_rate
            scores[symbol] = max(raw, 0.0)

            self.debug(
                f"{symbol.value} | mom={mom:.4f} "
                f"win_rate={win_rate:.4f} score={raw:.6f}"
            )

        # ------------------------------------------------------------------ #
        # Normalise to weights
        # ------------------------------------------------------------------ #
        total = sum(scores.values())

        if total <= 0:
            # Everything scored zero — park entirely in BIL
            self.log("All scores <= 0, holding BIL 100%")
            self.liquidate()
            self.set_holdings(self._bil, 1.0)
            return

        weights: dict = {s: v / total for s, v in scores.items()}

        # ------------------------------------------------------------------ #
        # Execute
        # ------------------------------------------------------------------ #
        self.liquidate()
        for symbol, weight in weights.items():
            if weight > 0.0:
                self.set_holdings(symbol, weight)
                self.log(f"  → {symbol.value} {weight:.2%}")
