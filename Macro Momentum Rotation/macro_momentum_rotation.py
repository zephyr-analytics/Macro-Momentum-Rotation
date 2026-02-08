from AlgorithmImports import *
import numpy as np


class AbsoluteRelativeMomentum(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2012, 1, 1)
        self.SetCash(1_000_000)
        self.Settings.SeedInitialPrices = True

        # ============================
        # Assets
        # ============================
        self.vt   = self.AddEquity("VT",   Resolution.Daily).Symbol
        self.vglt = self.AddEquity("VGLT", Resolution.Daily).Symbol
        self.vgit = self.AddEquity("VGIT", Resolution.Daily).Symbol
        self.gld  = self.AddEquity("GLD",  Resolution.Daily).Symbol
        self.dbc  = self.AddEquity("DBC",  Resolution.Daily).Symbol
        self.bil  = self.AddEquity("BIL",  Resolution.Daily).Symbol

        self.hyg  = self.AddEquity("HYG",  Resolution.Daily).Symbol
        self.vcit = self.AddEquity("VCIT", Resolution.Daily).Symbol
        self.vclt = self.AddEquity("VCLT", Resolution.Daily).Symbol
        self.vgsh = self.AddEquity("VGSH", Resolution.Daily).Symbol
        self.vcsh = self.AddEquity("VCSH", Resolution.Daily).Symbol

        self.btc = self.AddCrypto("BTCUSD", Resolution.Daily).Symbol

        self.etfs = [
            self.vt, self.vglt, self.vgit, self.gld, self.dbc,
            self.bil, self.hyg, self.vcit, self.vclt, self.vcsh, self.vgsh
        ]

        # All tradable assets
        self.assets = self.etfs + [self.btc]
        for symbol in self.assets:
            self.Securities[symbol].SetFeeModel(ConstantFeeModel(0))

        # ============================
        # Config
        # ============================
        self.momentum_lookbacks = [21, 63, 126, 189, 252]
        self.max_lookback = max(self.momentum_lookbacks)

        self.sma_period = 168
        self.vol_lookback = 252
        self.target_vol = 0.12
        self.max_weight = 1.0

        self.Schedule.On(
            self.DateRules.MonthEnd(self.vt),
            self.TimeRules.BeforeMarketClose(self.vt, 30),
            self.Rebalance
        )

        self.SetWarmup(
            max(self.max_lookback + 1, self.sma_period, self.vol_lookback + 1),
            Resolution.Daily
        )

    # ====================================================
    # Momentum
    # ====================================================
    def Momentum(self, symbol, closes):
        if symbol not in closes.columns:
            return -np.inf

        px = closes[symbol].dropna()

        max_lb = max(self.momentum_lookbacks)
        if len(px) < max_lb + 1:
            return -np.inf

        return float(np.mean([
            px.iloc[-1] / px.iloc[-(lb + 1)] - 1
            for lb in self.momentum_lookbacks
        ]))


    # ====================================================
    # SMA gate
    # ====================================================
    def PassesSmaGate(self, symbol, closes):
        if symbol == self.bil:
            return True

        if symbol not in closes.columns:
            return False

        px = closes[symbol].dropna()

        if len(px) < self.sma_period:
            return False

        sma = px.rolling(self.sma_period).mean().iloc[-1]
        if np.isnan(sma):
            return False

        return px.iloc[-1] > sma


    # ====================================================
    # Realized volatility
    # ====================================================
    def RealizedVol(self, symbol, closes):
        if symbol not in closes.columns:
            return np.nan

        px = closes[symbol].dropna()

        if len(px) < self.vol_lookback + 1:
            return np.nan

        rets = np.log(px / px.shift(1)).dropna()
        if len(rets) < self.vol_lookback:
            return np.nan

        vol = rets.tail(self.vol_lookback).std()
        if vol is None or np.isnan(vol):
            return np.nan

        return float(vol * np.sqrt(252))


    # ====================================================
    # Absolute 6-month return
    # ====================================================
    def AbsoluteReturn6M(self, symbol, closes):
        if symbol not in closes.columns:
            return -np.inf

        px = closes[symbol].dropna()
        if len(px) < 126 + 1:
            return -np.inf

        return float(px.iloc[-1] / px.iloc[-127] - 1)


    # ====================================================
    # Rebalance
    # ====================================================
    def Rebalance(self):
        if self.IsWarmingUp:
            return

        history = self.history(
            self.assets,
            self.max_lookback + 1,
            Resolution.Daily,
            data_normalization_mode=DataNormalizationMode.TotalReturn
        )

        if history.empty:
            return

        closes = history["close"].unstack(0)

        # ----------------------------
        # Eligibility (SMA + absolute return > cash)
        # ----------------------------
        bil_ret_12m = self.AbsoluteReturn6M(self.bil, closes)

        eligible = []
        for s in self.assets:
            if s not in closes.columns:
                continue

            if not self.PassesSmaGate(s, closes):
                continue

            # BIL always allowed
            if s == self.bil:
                eligible.append(s)
                continue

            # Absolute return filter: must beat cash
            if self.AbsoluteReturn6M(s, closes) > bil_ret_12m:
                eligible.append(s)

        if not eligible:
            self.Liquidate()
            self.SetHoldings(self.bil, 1.0)
            return

        # ----------------------------
        # Momentum ranking
        # ----------------------------
        scores = {s: self.Momentum(s, closes) for s in eligible}
        winner = max(scores, key=scores.get)

        for s, m in scores.items():
            self.Log(f"{s.Value} momentum: {m:.2%}")

        # ----------------------------
        # Vol targeting
        # ----------------------------
        if winner == self.bil:
            weight = 1.0
        else:
            vol = self.RealizedVol(winner, closes)
            if not vol or np.isnan(vol):
                weight = 0.0
            else:
                weight = min(self.max_weight, self.target_vol / vol)

        cash_weight = 1.0 - weight

        self.Log(
            f"Selected: {winner.Value} | "
            f"weight={weight:.2f}, cash={cash_weight:.2f}"
        )

        self.Liquidate()
        self.SetHoldings(winner, weight)
        self.SetHoldings(self.bil, cash_weight)
