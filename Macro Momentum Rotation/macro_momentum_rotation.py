from AlgorithmImports import *
import numpy as np

class AbsoluteRelativeMomentum(QCAlgorithm):
    """
    """
    def Initialize(self):
        """
        """
        self.SetStartDate(2012, 1, 1)
        self.SetCash(1_000_000)
        self.Settings.SeedInitialPrices = True

        # ============================
        # Core Assets & Benchmarks
        # ============================
        self.vt   = self.AddEquity("VTI",   Resolution.Daily).Symbol
        self.vglt = self.AddEquity("VGLT", Resolution.Daily).Symbol
        self.vgit = self.AddEquity("VGIT", Resolution.Daily).Symbol
        self.gld  = self.AddEquity("GLD",  Resolution.Daily).Symbol
        self.dbc  = self.AddEquity("DBC",  Resolution.Daily).Symbol
        self.bil  = self.AddEquity("SHV",  Resolution.Daily).Symbol
        self.vea  = self.AddEquity("VEA",  Resolution.Daily).Symbol
        self.vwo  = self.AddEquity("VWO",  Resolution.Daily).Symbol
        self.bnd  = self.AddEquity("BND",  Resolution.Daily).Symbol
        self.bndx = self.AddEquity("BNDX", Resolution.Daily).Symbol
        self.emb  = self.AddEquity("EMB",  Resolution.Daily).Symbol
        self.btc  = self.AddCrypto("BTCUSD", Resolution.Daily).Symbol

        # ============================
        # iShares Global Sector ETFs
        # ============================
        self.sectors = [
            self.AddEquity("IXP", Resolution.Daily).Symbol, # Comm Services
            self.AddEquity("RXI", Resolution.Daily).Symbol, # Consumer Disc
            self.AddEquity("KXI", Resolution.Daily).Symbol, # Consumer Staples
            self.AddEquity("IXC", Resolution.Daily).Symbol, # Energy
            self.AddEquity("IXG", Resolution.Daily).Symbol, # Financials
            self.AddEquity("IXJ", Resolution.Daily).Symbol, # Healthcare
            self.AddEquity("EXI", Resolution.Daily).Symbol, # Industrials
            self.AddEquity("IXN", Resolution.Daily).Symbol, # Technology
            self.AddEquity("MXI", Resolution.Daily).Symbol, # Materials
            self.AddEquity("REET", Resolution.Daily).Symbol,# Real Estate
            self.AddEquity("JXI", Resolution.Daily).Symbol  # Utilities
        ]

        self.etfs = [
            self.vt, self.vglt, self.vgit, self.gld, self.dbc,
            self.bil, self.vea, self.vwo, self.bnd, self.bndx, self.emb
        ] + self.sectors

        self.bond_assets = [self.vgit, self.bnd, self.bndx]
        self.assets = self.etfs + [self.btc]

        for symbol in self.assets:
            self.Securities[symbol].SetFeeModel(ConstantFeeModel(0))

        # ============================
        # Config
        # ============================
        self.top_n = 3
        self.momentum_lookbacks = [21, 63, 126, 189, 252]
        self.sma_period = 168
        self.bond_sma_period = 126
        self.cvar_lookback = 756
        self.target_cvar = 0.03
        self.max_weight = 1.0

        self.Schedule.On(
            self.DateRules.MonthStart(self.vt),
            self.TimeRules.BeforeMarketClose(self.vt, 120),
            self.Rebalance
        )

        warmup_days = max(max(self.momentum_lookbacks) + 1, self.sma_period, self.cvar_lookback + 1)
        self.SetWarmup(warmup_days, Resolution.Daily)


    def Momentum(self, symbol, closes):
        """
        """
        if symbol not in closes.columns: return -np.inf
        px = closes[symbol].dropna()
        if len(px) < max(self.momentum_lookbacks) + 1: return -np.inf

        return float(np.mean([
            px.iloc[-1] / px.iloc[-(lb + 1)] - 1
            for lb in self.momentum_lookbacks
        ]))


    def PassesTrendFilter(self, symbol, closes):
        """
        """
        if symbol == self.bil: return True
        if symbol not in closes.columns: return False
        px = closes[symbol].dropna()
        period = self.bond_sma_period if symbol in self.bond_assets else self.sma_period
        if len(px) < period: return False
        sma = px.iloc[-period:].mean()
        return px.iloc[-1] > sma


    def GetCVaR(self, symbol, closes, alpha=0.95):
        """
        """
        if symbol not in closes.columns: return np.nan
        px = closes[symbol].dropna()
        if len(px) < 252: return np.nan
        rets = px.pct_change().dropna().tail(self.cvar_lookback)
        var_threshold = np.percentile(rets, (1 - alpha) * 100)
        tail_losses = rets[rets <= var_threshold]
        if tail_losses.empty: return np.nan
        return float(abs(tail_losses.mean()))


    def Rebalance(self):
        """
        """
        if self.IsWarmingUp: return

        etf_history = self.history(self.etfs, max(self.cvar_lookback + 1, self.sma_period), 
                                   Resolution.Daily, data_normalization_mode=DataNormalizationMode.TotalReturn)
        if etf_history.empty: return
        etf_closes = etf_history["close"].unstack(0)

        btc_history = self.history([self.btc], max(self.cvar_lookback + 1, self.sma_period), Resolution.Daily)
        btc_closes = btc_history["close"].unstack(0) if not btc_history.empty else None

        # Eligibility
        eligible = [s for s in self.etfs if s in etf_closes.columns and self.PassesTrendFilter(s, etf_closes)]
        if btc_closes is not None and self.btc in btc_closes.columns and self.PassesTrendFilter(self.btc, btc_closes):
            eligible.append(self.btc)

        num_eligible = len(eligible)

        if num_eligible < 10:
            self.Debug(f"Low Breadth: {num_eligible} assets. Moving to BIL.")
            self.Liquidate()
            self.SetHoldings(self.bil, 1.0)
            return

        # Momentum Ranking
        scores = {}
        for s in eligible:
            hist_to_use = btc_closes if s == self.btc else etf_closes
            scores[s] = self.Momentum(s, hist_to_use)

        # Select Top N
        sorted_assets = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        winners = [x[0] for x in sorted_assets[:self.top_n]]

        self.Liquidate()

        total_allocated_weight = 0
        debug_msg = f"DATE: {self.Time.strftime('%Y-%m-%d')} | BREADTH: {num_eligible} | SELECTED: "

        for winner in winners:
            if winner == self.bil:
                indiv_weight = 1.0 / self.top_n
            else:
                hist_to_use = btc_closes if winner == self.btc else etf_closes
                asset_cvar = self.GetCVaR(winner, hist_to_use)

                if not asset_cvar or np.isnan(asset_cvar) or asset_cvar == 0:
                    indiv_weight = 0.0
                else:
                    slot_target = self.target_cvar / self.top_n
                    raw_weight = min(self.max_weight / self.top_n, slot_target / asset_cvar)
                    indiv_weight = min(raw_weight, 0.15) if winner == self.btc else raw_weight

            if indiv_weight > 0:
                self.SetHoldings(winner, indiv_weight)
                total_allocated_weight += indiv_weight
                debug_msg += f"{winner.Value}({indiv_weight:.2f}) "

        # Fill remainder with BIL
        cash_weight = 1.0 - total_allocated_weight
        if cash_weight > 0.01:
            self.SetHoldings(self.bil, cash_weight)
            debug_msg += f"| CASH: {cash_weight:.2f}"

        self.Debug(debug_msg)
