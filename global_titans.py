"""
Dow Global Titans 50 — Monthly Rebalancing Algo (QuantConnect / Lean)
=====================================================================
Universe  : Dow Jones Global Titans 50 (hardcoded inside the class)
Entry Rule: Price must be ABOVE its 200-day EMA to be eligible.
Portfolio : Equal-weight across all eligible stocks, rebalanced monthly.
            Stocks that fall back below the EMA are sold at the next rebalance.
Optional  : Multi-filter screener gate (USE_SCREENER_FILTERS = True):
              ① 6m return > SHV  (cash proxy)
              ② 6m return > VGIT (intermediate Treasury)
              ③ 3m, 6m AND 1y returns each > ACWI (global market)
Schedule  : Last trading day of each calendar month, 30 min after open.
"""

from AlgorithmImports import *


class DowTitansEMAAlgo(QCAlgorithm):

    # ── CONFIG ────────────────────────────────────────────────────────────────
    START_DATE           = (2010, 1, 1)
    END_DATE             = (2026, 1, 1)
    CASH                 = 100_000
    EMA_PERIOD           = 200          # 200-day EMA
    USE_SCREENER_FILTERS = True
    CASH_ETF             = "SHV"
    TREASURY_ETF         = "VGIT"
    ACWI_ETF             = "ACWI"

    DOW_TITANS = [
        # Information Technology
        "NVDA", "AAPL", "MSFT", "AVGO", "TSM",  "CSCO", "ORCL", "IBM",  "SAP",  "CRM",  "ACN",
        # Communication
        "META", "GOOGL", "NFLX",
        # Consumer Discretionary
        "AMZN", "TSLA", "TM",   "MCD",
        # Consumer Staples
        "PG",   "KO",   "PM",   "PEP",  "UL",
        # Health Care
        "LLY",  "JNJ",  "ABBV", "AZN",  "NVS",  "MRK",  "ABT",  "TMO",  "PFE",  "NVO",
        # Financials
        "JPM",  "V",    "MA",   "HSBC", "GS",   "RY",
        # Energy
        "XOM",  "CVX",  "SHEL",
        # Industrials
        "CAT",  "GE",
        # Materials
        "LIN",
        # Semiconductors
        "ASML",
    ]

    # ── INITIALIZE ────────────────────────────────────────────────────────────

    def Initialize(self):
        self.SetStartDate(*self.START_DATE)
        self.SetEndDate(*self.END_DATE)
        self.SetCash(self.CASH)
        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage,
                               AccountType.Margin)
        self.SetSecurityInitializer(lambda s: s.SetFeeModel(ConstantFeeModel(0)))

        resolution = Resolution.Daily

        # -- Universe & EMA indicators ----------------------------------------
        self._symbols = {}   # ticker str → Symbol
        self._ema     = {}   # Symbol     → EMA indicator

        for ticker in self.DOW_TITANS:
            try:
                eq = self.AddEquity(ticker, resolution)
                eq.SetDataNormalizationMode(DataNormalizationMode.Adjusted)
                sym = eq.Symbol
                self._symbols[ticker] = sym
                self._ema[sym] = self.EMA(sym, self.EMA_PERIOD, resolution)
            except Exception as e:
                self.Debug(f"Could not add {ticker}: {e}")

        # -- Benchmark instruments (screener filters only) --------------------
        self._bench_syms = {}
        if self.USE_SCREENER_FILTERS:
            for b in [self.CASH_ETF, self.TREASURY_ETF, self.ACWI_ETF]:
                try:
                    eq = self.AddEquity(b, resolution)
                    eq.SetDataNormalizationMode(DataNormalizationMode.Adjusted)
                    self._bench_syms[b] = eq.Symbol
                except Exception as e:
                    self.Debug(f"Could not add benchmark {b}: {e}")

        # -- SPY: scheduling anchor only (never traded) -----------------------
        self.AddEquity("SPY", resolution)

        # -- Warm-up ----------------------------------------------------------
        self.SetWarmUp(int(self.EMA_PERIOD * 1.5), Resolution.Daily)

        # -- Monthly rebalance schedule ---------------------------------------
        self.Schedule.On(
            self.DateRules.MonthEnd("SPY"),
            self.TimeRules.AfterMarketOpen("SPY", 60),
            self._rebalance
        )

        # -- State ------------------------------------------------------------
        self._current_holdings = set()
        self._bench_cache      = {}

        self.Debug(f"Universe loaded : {len(self._symbols)} tickers")
        self.Debug(f"EMA filter      : {self.EMA_PERIOD}-day EMA")
        self.Debug(f"Screener filters: {'ON' if self.USE_SCREENER_FILTERS else 'OFF'}")

    # ── HELPERS ───────────────────────────────────────────────────────────────

    def _period_return(self, sym, days):
        """Trailing calendar-day total return (%). Returns None on insufficient data."""
        try:
            hist = self.History(sym, days + 5, Resolution.Daily)
            if hist.empty or "close" not in hist.columns:
                return None
            closes = hist["close"].dropna()
            if len(closes) < max(10, days // 10):
                return None
            return (closes.iloc[-1] / closes.iloc[0] - 1) * 100
        except Exception:
            return None

    def _populate_bench_cache(self):
        """Fetch benchmark returns once per rebalance cycle."""
        self._bench_cache = {}
        if not self.USE_SCREENER_FILTERS:
            return
        for label, ticker, days in [
            ("shv_6m",  self.CASH_ETF,     126),
            ("vgit_6m", self.TREASURY_ETF, 126),
            ("acwi_3m", self.ACWI_ETF,      63),
            ("acwi_6m", self.ACWI_ETF,     126),
            ("acwi_1y", self.ACWI_ETF,     252),
        ]:
            sym = self._bench_syms.get(ticker)
            if sym:
                self._bench_cache[label] = self._period_return(sym, days)

    def _screener_pass(self, sym):
        """True if sym clears all three screener filters."""
        r3m = self._period_return(sym, 63)
        r6m = self._period_return(sym, 126)
        r1y = self._period_return(sym, 252)
        shv_6m  = self._bench_cache.get("shv_6m")
        vgit_6m = self._bench_cache.get("vgit_6m")
        acwi_3m = self._bench_cache.get("acwi_3m")
        acwi_6m = self._bench_cache.get("acwi_6m")
        acwi_1y = self._bench_cache.get("acwi_1y")
        f1 = r6m is not None and shv_6m  is not None and r6m > shv_6m
        f2 = r6m is not None and vgit_6m is not None and r6m > vgit_6m
        f3 = (r3m is not None and acwi_3m is not None and r3m > acwi_3m and
              r6m is not None and acwi_6m is not None and r6m > acwi_6m and
              r1y is not None and acwi_1y is not None and r1y > acwi_1y)
        return f1 and f2 and f3

    def _above_ema(self, sym):
        """True if current price is strictly above the EMA and the indicator is ready."""
        ind = self._ema.get(sym)
        if ind is None or not ind.IsReady:
            return False
        price = self.Securities[sym].Price
        return price > 0 and price > ind.Current.Value

    # ── MONTHLY REBALANCE ────────────────────────────────────────────────────

    def _rebalance(self):
        if self.IsWarmingUp:
            return

        self._populate_bench_cache()

        # Build eligible list
        eligible = []
        for ticker, sym in self._symbols.items():
            sec = self.Securities.get(sym)
            if sec is None or not sec.IsTradable or sec.Price <= 0:
                continue
            if not self._above_ema(sym):
                continue
            if self.USE_SCREENER_FILTERS and not self._screener_pass(sym):
                continue
            eligible.append(sym)

        n            = len(eligible)
        eligible_set = set(eligible)

        self.Debug(f"{self.Time.date()} | Eligible: {n}/{len(self._symbols)}")

        # Go to cash if nothing qualifies
        if n == 0:
            for sym in list(self._current_holdings):
                self.Liquidate(sym)
                self.Debug(f"  SELL (no eligible): {sym.Value}")
            self._current_holdings.clear()
            return

        target_weight = 1.0 / n

        # Sell positions that dropped out
        for sym in list(self._current_holdings):
            if sym not in eligible_set:
                self.Liquidate(sym)
                self._current_holdings.discard(sym)
                self.Debug(f"  SELL (below EMA): {sym.Value}")

        # Buy / rebalance eligible positions
        for sym in eligible:
            action = "HOLD/resize" if self.Portfolio[sym].Invested else "BUY"
            self.SetHoldings(sym, target_weight)
            self._current_holdings.add(sym)
            self.Debug(f"  {action}: {sym.Value}  wt={target_weight:.2%}  "
                       f"EMA={self._ema[sym].Current.Value:.2f}  "
                       f"px={self.Securities[sym].Price:.2f}")

    # ── CHARTING & SUMMARY ────────────────────────────────────────────────────

    def OnEndOfMonth(self):
        self.Plot("Portfolio", "Value ($)", self.Portfolio.TotalPortfolioValue)
        self.Plot("Portfolio", "# Holdings",
                  sum(1 for s in self._current_holdings if self.Portfolio[s].Invested))

    def OnEndOfAlgorithm(self):
        total   = self.Portfolio.TotalPortfolioValue
        ret_pct = (total / self.CASH - 1) * 100
        self.Debug("=" * 60)
        self.Debug(f"Final value  : ${total:,.2f}")
        self.Debug(f"Total return : {ret_pct:.1f}%")
        self.Debug("=" * 60)
