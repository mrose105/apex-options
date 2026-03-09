"""
APEX 0DTE — Data Provider Layer
================================
Abstract interface + concrete implementations for:
  - Tradier (recommended: real options chains + equity data)
  - Alpaca  (equity bars; options via partner feeds)
  - CBOE    (historical options data downloads)
  - Synthetic (fallback for testing)

All providers return standardised DataFrames so the strategy engine
never touches provider-specific code.

Usage:
    from data_providers import TradierProvider, AlpacaProvider, SyntheticProvider

    # Live
    provider = TradierProvider(api_key="YOUR_KEY", sandbox=False)

    # Paper / testing
    provider = TradierProvider(api_key="YOUR_SANDBOX_KEY", sandbox=True)

    # Backtesting with no API
    provider = SyntheticProvider()
"""

from __future__ import annotations
import os
import time
import json
import logging
from abc import ABC, abstractmethod
from datetime import date, datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

logger = logging.getLogger("apex.data")


# ─────────────────────────────────────────────────────────────────────────────
# STANDARD OUTPUT SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────

class DailyBar:
    """Standardised daily OHLCV + gap."""
    COLS = ["date", "open", "high", "low", "close", "volume", "gap_pct"]

class IntradayBar:
    """5-min OHLCV bar."""
    COLS = ["datetime", "open", "high", "low", "close", "volume"]

class OptionQuote:
    """
    Single option quote at a point in time.
    All Greeks are model-computed if the provider doesn't supply them.
    """
    COLS = [
        "datetime", "underlying_price", "symbol",
        "expiry", "strike", "option_type",   # 'call' or 'put'
        "bid", "ask", "mid",
        "iv", "delta", "gamma", "theta", "vega",
        "open_interest", "volume",
    ]


# ─────────────────────────────────────────────────────────────────────────────
# BLACK-SCHOLES UTILITY  (used by all providers for missing Greeks)
# ─────────────────────────────────────────────────────────────────────────────

def bsm_greeks(S: float, K: float, T: float, r: float, sigma: float,
               call: bool = True) -> dict:
    """Full BSM price + Greeks. T in years."""
    if T < 1e-9 or sigma <= 0:
        intrinsic = max(S - K, 0) if call else max(K - S, 0)
        return dict(price=intrinsic, delta=(1.0 if S > K else 0.0) if call
                    else (-1.0 if S < K else 0.0),
                    gamma=0.0, theta=0.0, vega=0.0, iv=sigma)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * T ** 0.5)
    d2 = d1 - sigma * T ** 0.5
    nd1 = norm.pdf(d1)
    if call:
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = float(norm.cdf(d1))
        theta = ((-S * nd1 * sigma / (2 * T ** 0.5)
                  - r * K * np.exp(-r * T) * norm.cdf(d2)) / 252)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = float(norm.cdf(d1) - 1)
        theta = ((-S * nd1 * sigma / (2 * T ** 0.5)
                  + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 252)
    gamma = nd1 / (S * sigma * T ** 0.5)
    vega  = S * nd1 * T ** 0.5 / 100
    return dict(price=float(price), delta=delta, gamma=float(gamma),
                theta=float(theta), vega=float(vega), iv=sigma)


def implied_vol(market_price: float, S: float, K: float, T: float,
                r: float, call: bool = True,
                tol: float = 1e-5, max_iter: int = 100) -> float:
    """Newton-Raphson IV solver."""
    if market_price <= 0 or T < 1e-9:
        return 0.0
    sig = 0.25
    for _ in range(max_iter):
        g = bsm_greeks(S, K, T, r, sig, call)
        diff = g["price"] - market_price
        if abs(diff) < tol:
            break
        vega = g["vega"] * 100       # vega is per 1% move; un-normalise
        if abs(vega) < 1e-9:
            break
        sig = max(0.01, sig - diff / vega)
    return float(sig)


# ─────────────────────────────────────────────────────────────────────────────
# EXECUTION MODEL  (slippage + transaction costs)
# ─────────────────────────────────────────────────────────────────────────────

class ExecutionModel:
    """
    Models real execution costs for 0DTE options.

    Slippage on 0DTE ATM QQQ options:
    - Bid-ask spread typically 2–8% of mid at open (wider in high vol)
    - Market impact negligible below 50 contracts; grows ~sqrt above
    - Commission: $0.65/contract (TD/Schwab/Tradier standard)
    """

    def __init__(self,
                 spread_model: str = "dynamic",   # 'dynamic' | 'fixed'
                 fixed_spread_pct: float = 0.04,  # used if spread_model='fixed'
                 commission_per_contract: float = 0.65,
                 contracts_per_1k_risk: float = None):
        self.spread_model   = spread_model
        self.fixed_spread   = fixed_spread_pct
        self.commission     = commission_per_contract

    def spread_pct(self, iv: float, time_to_expiry: float,
                   moneyness: float = 0.0) -> float:
        """
        Dynamic spread as function of IV and time remaining.
        At open (T=1/252): spread ~4-6% of mid in normal vol (IV~20%)
        In high vol (IV>35%): spread ~8-12% (MM widens for inventory risk)
        OTM options are proportionally wider.
        """
        if self.spread_model == "fixed":
            return self.fixed_spread
        # Base spread: linear in IV (empirically fits QQQ market data)
        base = 0.03 + iv * 0.12
        # Moneyness penalty: OTM options have wider relative spreads
        otm_penalty = max(0.0, abs(moneyness) * 2.0)
        # Time penalty: near-expiry < 30min gets much wider
        time_penalty = max(0.0, (0.1 - time_to_expiry * 252) * 0.5)
        return min(base + otm_penalty + time_penalty, 0.20)   # cap at 20%

    def entry_price(self, mid: float, iv: float, T: float,
                    moneyness: float = 0.0, n_contracts: int = 1) -> float:
        """
        Realistic entry price (you pay the ask + half-spread impact).
        For buys: pay mid + half_spread
        """
        half_spread = mid * self.spread_pct(iv, T, moneyness) / 2
        impact = mid * max(0.0, max(n_contracts - 50, 0) ** 0.5 * 0.0002)
        return mid + half_spread + impact

    def exit_price(self, mid: float, iv: float, T: float,
                   moneyness: float = 0.0, n_contracts: int = 1) -> float:
        """
        Realistic exit price (you sell at bid = mid - half_spread).
        """
        half_spread = mid * self.spread_pct(iv, T, moneyness) / 2
        impact = mid * max(0.0, max(n_contracts - 50, 0) ** 0.5 * 0.0002)
        return max(0.0, mid - half_spread - impact)

    def commission_cost(self, n_contracts: int) -> float:
        return n_contracts * self.commission


# ─────────────────────────────────────────────────────────────────────────────
# ABSTRACT BASE PROVIDER
# ─────────────────────────────────────────────────────────────────────────────

class BaseProvider(ABC):
    """
    All providers must implement these methods.
    The strategy engine calls ONLY these — never provider-specific code.
    """

    def __init__(self, symbol: str = "QQQ", risk_free_rate: float = 0.05):
        self.symbol  = symbol
        self.rf      = risk_free_rate
        self.exec_model = ExecutionModel()

    @abstractmethod
    def get_daily_bars(self, start: date, end: date) -> pd.DataFrame:
        """
        Returns daily OHLCV with gap_pct column.
        gap_pct = (open - prev_close) / prev_close
        """

    @abstractmethod
    def get_intraday_bars(self, trading_date: date,
                          interval_min: int = 5) -> pd.DataFrame:
        """Returns 5-min bars for a single trading day."""

    @abstractmethod
    def get_option_chain_at_open(self, trading_date: date,
                                  expiry: date) -> pd.DataFrame:
        """
        Returns full 0DTE option chain as of market open (~9:31am).
        Columns: OptionQuote.COLS
        """

    def get_atm_quote(self, trading_date: date, expiry: date,
                      option_type: str = "call") -> Optional[dict]:
        """Helper: returns the ATM option quote for the given date."""
        chain = self.get_option_chain_at_open(trading_date, expiry)
        if chain is None or len(chain) == 0:
            return None
        chain = chain[chain["option_type"] == option_type]
        if len(chain) == 0:
            return None
        # ATM = strike closest to underlying price
        chain = chain.copy()
        chain["_dist"] = (chain["strike"] - chain["underlying_price"]).abs()
        return chain.sort_values("_dist").iloc[0].to_dict()


# ─────────────────────────────────────────────────────────────────────────────
# TRADIER PROVIDER  (live + paper trading)
# ─────────────────────────────────────────────────────────────────────────────

class TradierProvider(BaseProvider):
    """
    Tradier brokerage API.
    Paper:  https://sandbox.tradier.com/v1
    Live:   https://api.tradier.com/v1

    Get API key at: https://brokerage.tradier.com/settings/api
    Sandbox key:    https://developer.tradier.com/

    Rate limits: 120 req/min (live), 60 req/min (sandbox)
    """

    LIVE_URL    = "https://api.tradier.com/v1"
    SANDBOX_URL = "https://sandbox.tradier.com/v1"

    def __init__(self, api_key: str, sandbox: bool = False,
                 symbol: str = "QQQ", risk_free_rate: float = 0.05):
        super().__init__(symbol, risk_free_rate)
        self.api_key  = api_key
        self.base_url = self.SANDBOX_URL if sandbox else self.LIVE_URL
        self.sandbox  = sandbox
        self._session_headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        }
        logger.info(f"TradierProvider init  sandbox={sandbox}  symbol={symbol}")

    def _get(self, endpoint: str, params: dict = None) -> dict:
        """HTTP GET with retry logic."""
        try:
            import requests
        except ImportError:
            raise ImportError("pip install requests")
        url = f"{self.base_url}/{endpoint}"
        for attempt in range(3):
            try:
                r = requests.get(url, headers=self._session_headers,
                                 params=params, timeout=10)
                r.raise_for_status()
                return r.json()
            except Exception as e:
                if attempt == 2:
                    raise
                logger.warning(f"Tradier request failed (attempt {attempt+1}): {e}")
                time.sleep(1.5 ** attempt)

    def get_daily_bars(self, start: date, end: date) -> pd.DataFrame:
        """
        GET /markets/history
        Returns adjusted OHLCV daily bars.
        """
        data = self._get("markets/history", {
            "symbol":   self.symbol,
            "interval": "daily",
            "start":    start.strftime("%Y-%m-%d"),
            "end":      end.strftime("%Y-%m-%d"),
        })
        rows = data.get("history", {}).get("day", [])
        if not rows:
            logger.warning(f"No daily bars returned for {self.symbol} {start}→{end}")
            return pd.DataFrame(columns=DailyBar.COLS)
        df = pd.DataFrame(rows)
        df["date"]   = pd.to_datetime(df["date"])
        df           = df.rename(columns={"open":"open","high":"high",
                                           "low":"low","close":"close",
                                           "volume":"volume"})
        df           = df.sort_values("date").reset_index(drop=True)
        df["gap_pct"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)
        return df[DailyBar.COLS]

    def get_intraday_bars(self, trading_date: date,
                          interval_min: int = 5) -> pd.DataFrame:
        """
        GET /markets/timesales
        Returns 5-min bars for a single trading session.
        """
        start_dt = datetime.combine(trading_date, datetime.min.time()).replace(
            hour=9, minute=30)
        end_dt   = start_dt.replace(hour=16, minute=0)
        data = self._get("markets/timesales", {
            "symbol":   self.symbol,
            "interval": f"{interval_min}min",
            "start":    start_dt.strftime("%Y-%m-%d %H:%M"),
            "end":      end_dt.strftime("%Y-%m-%d %H:%M"),
            "session_filter": "open",
        })
        rows = data.get("series", {}).get("data", [])
        if not rows:
            return pd.DataFrame(columns=IntradayBar.COLS)
        df = pd.DataFrame(rows)
        df["datetime"] = pd.to_datetime(df["time"])
        df = df.rename(columns={"open":"open","high":"high",
                                  "low":"low","close":"close",
                                  "volume":"volume"})
        return df[IntradayBar.COLS].sort_values("datetime").reset_index(drop=True)

    def get_option_chain_at_open(self, trading_date: date,
                                  expiry: date) -> pd.DataFrame:
        """
        GET /markets/options/chains
        Returns full chain for the given expiry.
        """
        data = self._get("markets/options/chains", {
            "symbol":     self.symbol,
            "expiration": expiry.strftime("%Y-%m-%d"),
            "greeks":     "true",
        })
        options = data.get("options", {}).get("option", [])
        if not options:
            return pd.DataFrame(columns=OptionQuote.COLS)

        # Get underlying price at open
        intra = self.get_intraday_bars(trading_date)
        S = intra.iloc[0]["open"] if len(intra) > 0 else None

        rows = []
        T = max((datetime.combine(expiry, datetime.min.time()) -
                 datetime.combine(trading_date, datetime.min.time())).days / 365, 1e-8)

        for o in options:
            mid  = (float(o.get("bid", 0) or 0) +
                    float(o.get("ask", 0) or 0)) / 2
            g    = o.get("greeks") or {}
            iv_v = float(g.get("mid_iv", 0) or 0)
            if iv_v == 0 and mid > 0 and S:
                iv_v = implied_vol(mid, S, float(o["strike"]), T, self.rf,
                                   call=(o["option_type"] == "call"))
            greeks_calc = bsm_greeks(S or float(o["strike"]),
                                      float(o["strike"]), T, self.rf, iv_v,
                                      call=(o["option_type"] == "call"))
            rows.append({
                "datetime":        pd.Timestamp(f"{trading_date} 09:31:00"),
                "underlying_price": S,
                "symbol":          o.get("symbol", ""),
                "expiry":          pd.Timestamp(expiry),
                "strike":          float(o["strike"]),
                "option_type":     o["option_type"],
                "bid":             float(o.get("bid", 0) or 0),
                "ask":             float(o.get("ask", 0) or 0),
                "mid":             mid,
                "iv":              iv_v,
                "delta":           float(g.get("delta", greeks_calc["delta"]) or greeks_calc["delta"]),
                "gamma":           float(g.get("gamma", greeks_calc["gamma"]) or greeks_calc["gamma"]),
                "theta":           float(g.get("theta", greeks_calc["theta"]) or greeks_calc["theta"]),
                "vega":            float(g.get("vega",  greeks_calc["vega"])  or greeks_calc["vega"]),
                "open_interest":   int(o.get("open_interest", 0) or 0),
                "volume":          int(o.get("volume", 0) or 0),
            })
        return pd.DataFrame(rows, columns=OptionQuote.COLS)


# ─────────────────────────────────────────────────────────────────────────────
# ALPACA PROVIDER  (equity bars; options via Alpaca options feed)
# ─────────────────────────────────────────────────────────────────────────────

class AlpacaProvider(BaseProvider):
    """
    Alpaca Markets API.
    Docs: https://docs.alpaca.markets/reference/stockbars

    Requires ALPACA_API_KEY + ALPACA_SECRET_KEY env vars
    or pass directly to constructor.

    Plan matrix:
      Free / Paper : equity bars (delayed 15min on free, real-time on paper) ✓
                     options chain via snapshot endpoint                       ✓ (paper)
      Live         : real-time equity bars + real options chain                ✓

    Equity bars:   https://data.alpaca.markets/v2/stocks/{sym}/bars
    Options chain: https://data.alpaca.markets/v1beta1/options/snapshots/{sym}
    Account info:  https://paper-api.alpaca.markets/v2/account  (paper)
                   https://api.alpaca.markets/v2/account        (live)
    """

    DATA_URL    = "https://data.alpaca.markets/v2"
    OPTION_URL  = "https://data.alpaca.markets/v1beta1"
    BROKER_PAPER = "https://paper-api.alpaca.markets/v2"
    BROKER_LIVE  = "https://api.alpaca.markets/v2"

    def __init__(self, api_key: str = None, secret_key: str = None,
                 symbol: str = "QQQ", paper: bool = True,
                 risk_free_rate: float = 0.05):
        super().__init__(symbol, risk_free_rate)
        self.api_key    = api_key    or os.environ.get("ALPACA_API_KEY", "")
        self.secret_key = secret_key or os.environ.get("ALPACA_API_SECRET", os.environ.get("ALPACA_SECRET_KEY", ""))
        self.paper      = paper
        self.broker_url = self.BROKER_PAPER if paper else self.BROKER_LIVE
        self._headers   = {
            "APCA-API-KEY-ID":     self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
        }
        self._has_options: Optional[bool] = None  # lazy-detected

    def _get(self, base: str, endpoint: str, params: dict = None) -> dict:
        import requests
        url = f"{base}/{endpoint}"
        for attempt in range(3):
            try:
                r = requests.get(url, headers=self._headers,
                                 params=params, timeout=15)
                r.raise_for_status()
                return r.json()
            except Exception as e:
                if attempt == 2:
                    raise
                time.sleep(1.5 ** attempt)

    def _get_paginated(self, base: str, endpoint: str,
                        params: dict, key: str) -> list:
        """Fetch all pages from a paginated Alpaca endpoint."""
        import requests
        results = []
        p = dict(params)
        while True:
            data = self._get(base, endpoint, p)
            items = data.get(key, []) or []
            if isinstance(items, dict):
                items = list(items.values())
            results.extend(items)
            next_token = data.get("next_page_token")
            if not next_token:
                break
            p["page_token"] = next_token
        return results

    def get_account(self) -> dict:
        """Return account info — useful for verifying credentials."""
        return self._get(self.broker_url, "account")

    def get_daily_bars(self, start: date, end: date) -> pd.DataFrame:
        bars = self._get_paginated(self.DATA_URL,
            f"stocks/{self.symbol}/bars",
            {"timeframe": "1Day",
             "start":     start.isoformat(),
             "end":       end.isoformat(),
             "adjustment": "all",
             "feed":      "iex",    # iex = free; sip = paid (more accurate)
             "limit":     1000},
            key="bars")
        if not bars:
            return pd.DataFrame(columns=DailyBar.COLS)
        df = pd.DataFrame(bars)
        df["date"]    = pd.to_datetime(df["t"]).dt.date
        df = df.rename(columns={"o":"open","h":"high","l":"low",
                                  "c":"close","v":"volume"})
        df = df.sort_values("date").reset_index(drop=True)
        df["gap_pct"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)
        df["date"]    = pd.to_datetime(df["date"])
        return df[DailyBar.COLS]

    def get_intraday_bars(self, trading_date: date,
                          interval_min: int = 5) -> pd.DataFrame:
        bars = self._get_paginated(self.DATA_URL,
            f"stocks/{self.symbol}/bars",
            {"timeframe": f"{interval_min}Min",
             "start":     f"{trading_date}T09:30:00-04:00",
             "end":       f"{trading_date}T16:00:00-04:00",
             "feed":      "iex",
             "limit":     1000},
            key="bars")
        if not bars:
            return pd.DataFrame(columns=IntradayBar.COLS)
        df = pd.DataFrame(bars)
        df["datetime"] = pd.to_datetime(df["t"])
        df = df.rename(columns={"o":"open","h":"high","l":"low",
                                  "c":"close","v":"volume"})
        return df[IntradayBar.COLS].sort_values("datetime").reset_index(drop=True)

    def get_option_chain_at_open(self, trading_date: date,
                                  expiry: date) -> pd.DataFrame:
        """
        Fetch options chain from Alpaca snapshot endpoint.
        Works on paper accounts.
        Falls back to BSM-priced chain if unavailable (e.g. free data plan).

        Alpaca option symbol format: SPXW240315C05200000
          → QQQ + YYMMDD + C/P + 8-digit strike (×1000)
        """
        # T floor: for 0DTE (same calendar day), use 1 full trading day
        cal_days = (datetime.combine(expiry, datetime.min.time()) -
                    datetime.combine(trading_date, datetime.min.time())).days
        T = max(cal_days / 365, 1 / 252)

        try:
            data = self._get(self.OPTION_URL,
                             f"options/snapshots/{self.symbol}", {
                "expiration_date_gte": expiry.isoformat(),
                "expiration_date_lte": expiry.isoformat(),
                "feed":  "indicative",
                "limit": 1000,
            })
            snapshots = data.get("snapshots", {})
            if not snapshots:
                raise ValueError("empty")
            self._has_options = True
        except Exception as e:
            if self._has_options is None:
                logger.info(f"Alpaca options unavailable ({type(e).__name__}), "
                            f"using BSM chain — upgrade plan for real quotes")
                self._has_options = False
            return self._bsm_chain(trading_date, expiry, T)

        # Get underlying price from first intraday bar
        intra = self.get_intraday_bars(trading_date)
        S = float(intra.iloc[0]["open"]) if len(intra) > 0 else None

        rows = []
        for sym, snap in snapshots.items():
            try:
                # Parse OCC symbol: QQQ + YYMMDD + C/P + 00000000 (strike×1000)
                suffix = sym[len(self.symbol):]          # e.g. "240315C00450000"
                opt_type = "call" if "C" in suffix else "put"
                strike_str = suffix[-8:]
                K = float(strike_str) / 1000

                q   = snap.get("latestQuote", {})
                bid = float(q.get("bp", 0) or 0)
                ask = float(q.get("ap", 0) or 0)
                mid = (bid + ask) / 2
                if mid <= 0:
                    continue

                d    = snap.get("greeks", snap.get("Greeks", {})) or {}
                iv_v = float(d.get("impliedVolatility", 0) or 0)
                if iv_v == 0 and S:
                    iv_v = implied_vol(mid, S, K, T, self.rf,
                                       call=(opt_type == "call"))
                iv_v = max(iv_v, 0.05)

                g_bsm = bsm_greeks(S or K, K, T, self.rf, iv_v,
                                    call=(opt_type == "call"))
                rows.append({
                    "datetime":         pd.Timestamp(f"{trading_date} 09:31:00"),
                    "underlying_price": S,
                    "symbol":           sym,
                    "expiry":           pd.Timestamp(expiry),
                    "strike":           K,
                    "option_type":      opt_type,
                    "bid":              bid, "ask": ask, "mid": mid,
                    "iv":               iv_v,
                    "delta": float(d.get("delta",  g_bsm["delta"])  or g_bsm["delta"]),
                    "gamma": float(d.get("gamma",  g_bsm["gamma"])  or g_bsm["gamma"]),
                    "theta": float(d.get("theta",  g_bsm["theta"])  or g_bsm["theta"]),
                    "vega":  float(d.get("vega",   g_bsm["vega"])   or g_bsm["vega"]),
                    "open_interest": int(snap.get("openInterest", 0) or 0),
                    "volume":        int(snap.get("dailyBar", {}).get("v", 0) or 0),
                })
            except Exception:
                continue

        if not rows:
            logger.warning(f"{trading_date}: parsed 0 rows from Alpaca chain, "
                           f"falling back to BSM")
            return self._bsm_chain(trading_date, expiry, T)
        return pd.DataFrame(rows, columns=OptionQuote.COLS)

    def _bsm_chain(self, trading_date: date, expiry: date,
                    T: float = None) -> pd.DataFrame:
        """
        BSM-priced ATM chain using real QQQ open price + trailing HV for IV.
        Used when Alpaca options data is unavailable.
        The IV is calibrated from trailing 21-day realised vol × open-spike factor.
        """
        intra = self.get_intraday_bars(trading_date)
        if len(intra) == 0:
            return pd.DataFrame(columns=OptionQuote.COLS)
        S = float(intra.iloc[0]["open"])

        if T is None:
            cal_days = (datetime.combine(expiry, datetime.min.time()) -
                        datetime.combine(trading_date, datetime.min.time())).days
            T = max(cal_days / 365, 1 / 252)

        # IV from trailing 21-day HV × gap-scaled open spike
        # On gap days, market makers price in higher IV because the gap itself
        # signals elevated intraday uncertainty. Larger gap = higher IV premium.
        # Empirically: small gap (+0.4%) → ~1.3x HV, large gap (+1%+) → ~1.8x HV
        hist = self.get_daily_bars(
            trading_date - timedelta(days=35), trading_date)
        if len(hist) > 5:
            lr = np.diff(np.log(hist["close"].astype(float).values))
            hv = float(np.std(lr, ddof=1) * 252 ** 0.5)
        else:
            hv = 0.20

        # Gap-scaled IV multiplier: 1.3 at 0% gap, scales to 1.8 at 1%+ gap
        if len(hist) > 0:
            last_close = float(hist["close"].iloc[-1])
            gap_mag = abs(S - last_close) / last_close if last_close > 0 else 0.005
        else:
            gap_mag = 0.005
        iv_mult = 1.30 + min(gap_mag / 0.01, 1.0) * 0.50   # 1.30 → 1.80 as gap 0→1%
        iv_open = float(np.clip(hv * iv_mult, 0.10, 1.50))

        # Build strikes ±4% around ATM in $1 increments
        strikes = np.arange(round(S * 0.96), round(S * 1.04) + 1, 1.0)
        rows = []
        for K in strikes:
            for call in [True, False]:
                g = bsm_greeks(S, K, T, self.rf, iv_open, call=call)
                if g["price"] < 0.05:
                    continue
                spread = self.exec_model.spread_pct(iv_open, T,
                                                     (K - S) / S) * g["price"]
                rows.append({
                    "datetime":         pd.Timestamp(f"{trading_date} 09:31:00"),
                    "underlying_price": S,
                    "symbol":           f"{trading_date}_{K}_{'C' if call else 'P'}_BSM",
                    "expiry":           pd.Timestamp(expiry),
                    "strike":           K,
                    "option_type":      "call" if call else "put",
                    "bid":              max(0.0, g["price"] - spread / 2),
                    "ask":              g["price"] + spread / 2,
                    "mid":              g["price"],
                    "iv":               iv_open,
                    "delta":            g["delta"], "gamma": g["gamma"],
                    "theta":            g["theta"], "vega":  g["vega"],
                    "open_interest":    0, "volume": 0,
                })
        return pd.DataFrame(rows, columns=OptionQuote.COLS)


# ─────────────────────────────────────────────────────────────────────────────
# CBOE PROVIDER  (historical data for backtesting — free downloads)
# ─────────────────────────────────────────────────────────────────────────────

class CBOEProvider(BaseProvider):
    """
    CBOE historical data loader for backtesting.

    Download free historical 0DTE data from:
    https://datashop.cboe.com/option-quotes  (intraday quotes, paid)
    https://www.cboe.com/us/equities/market_statistics/historical_data/
    (free daily; intraday requires subscription)

    Usage:
        provider = CBOEProvider(
            daily_csv_path="data/QQQ_daily_2016_2026.csv",
            options_parquet_dir="data/qqq_options/"
        )

    Expected CSV columns: Date, Open, High, Low, Close, Adj_Close, Volume
    Options parquet: one file per day named YYYY-MM-DD.parquet
    with columns matching OptionQuote.COLS
    """

    def __init__(self, daily_csv_path: str,
                 options_parquet_dir: str = None,
                 symbol: str = "QQQ", risk_free_rate: float = 0.05):
        super().__init__(symbol, risk_free_rate)
        self.daily_csv   = daily_csv_path
        self.options_dir = options_parquet_dir
        self._daily_cache: Optional[pd.DataFrame] = None

    def _load_daily(self) -> pd.DataFrame:
        if self._daily_cache is not None:
            return self._daily_cache
        df = pd.read_csv(self.daily_csv, parse_dates=["Date"])
        df.columns = [c.lower().strip() for c in df.columns]
        df = df.rename(columns={"date":"date","adj_close":"close",
                                  "adj close":"close"})
        for col in ["open","high","low","close","volume"]:
            if col not in df.columns:
                raise ValueError(f"Missing column '{col}' in {self.daily_csv}")
        df = df.sort_values("date").reset_index(drop=True)
        df["gap_pct"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)
        self._daily_cache = df
        return df

    def get_daily_bars(self, start: date, end: date) -> pd.DataFrame:
        df = self._load_daily()
        mask = (df["date"].dt.date >= start) & (df["date"].dt.date <= end)
        return df[mask][DailyBar.COLS].reset_index(drop=True)

    def get_intraday_bars(self, trading_date: date,
                          interval_min: int = 5) -> pd.DataFrame:
        """
        If options_parquet_dir contains intraday files: load them.
        Otherwise synthesise from daily OHLC via the SyntheticProvider method.
        """
        if self.options_dir:
            path = os.path.join(self.options_dir, f"{trading_date}.parquet")
            if os.path.exists(path):
                return pd.read_parquet(path)
        # Fallback: synthesise intraday bars from daily OHLC
        daily = self.get_daily_bars(trading_date, trading_date)
        if len(daily) == 0:
            return pd.DataFrame(columns=IntradayBar.COLS)
        row  = daily.iloc[0]
        return SyntheticProvider._gen_intraday_from_daily(
            row["open"], row["high"], row["low"], row["close"],
            trading_date, interval_min)

    def get_option_chain_at_open(self, trading_date: date,
                                  expiry: date) -> pd.DataFrame:
        if not self.options_dir:
            # No options data — use BSM with IV from trailing HV
            intra = self.get_intraday_bars(trading_date)
            if len(intra) == 0:
                return pd.DataFrame(columns=OptionQuote.COLS)
            daily = self.get_daily_bars(
                trading_date - timedelta(days=30), trading_date)
            if len(daily) > 5:
                lr = np.diff(np.log(daily["close"].values))
                iv = float(np.std(lr, ddof=1) * 252 ** 0.5)
                iv = np.clip(iv, 0.10, 1.5) * 1.38  # open IV spike
            else:
                iv = 0.25
            S = intra.iloc[0]["open"]
            return AlpacaProvider(symbol=self.symbol, risk_free_rate=self.rf) \
                ._synthetic_chain.__func__(
                    type('obj', (object,), {
                        'exec_model': self.exec_model, 'rf': self.rf,
                        'symbol': self.symbol,
                        'get_intraday_bars': lambda d: intra
                    })(), trading_date, expiry)

        path = os.path.join(self.options_dir, f"chain_{trading_date}.parquet")
        if os.path.exists(path):
            return pd.read_parquet(path)
        return pd.DataFrame(columns=OptionQuote.COLS)


# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC PROVIDER  (testing / CI)
# ─────────────────────────────────────────────────────────────────────────────

class SyntheticProvider(BaseProvider):
    """
    Deterministic synthetic data — used when no real data is available.
    This is what the backtest currently uses.
    Switching to any real provider = drop-in replacement.
    """

    def __init__(self, symbol: str = "QQQ", seed: int = 2024,
                 risk_free_rate: float = 0.05,
                 start_date: date = date(2016, 1, 4),
                 start_price: float = 100.0):
        super().__init__(symbol, risk_free_rate)
        self.seed        = seed
        self.start_date  = start_date
        self.start_price = start_price
        self._daily_cache: Optional[pd.DataFrame] = None
        self._build_daily()

    def _build_daily(self):
        from math import log
        def rmu(s, e, n): return log(e / s) / (n / 252)
        segs = [
            (12,   100,  93,  0.45),
            (247,   93, 118,  0.17),
            (260,  118, 155,  0.16),
            (29,   155, 145,  0.30),
            (231,  145, 132,  0.20),
            (262,  132, 210,  0.18),
            (59,   210, 174,  0.75),
            (203,  174, 310,  0.40),
            (261,  310, 400,  0.17),
            (260,  400, 265,  0.35),
            (260,  265, 430,  0.20),
            (310,  430, 510,  0.22),
        ]
        N = sum(s[0] for s in segs)
        rng   = np.random.default_rng(self.seed)
        mu_d  = np.concatenate([np.full(s[0], rmu(s[1], s[2], s[0]) / 252)
                                 for s in segs])[:N]
        sig_d = np.concatenate([np.full(s[0], s[3] / 252 ** 0.5)
                                 for s in segs])[:N]
        gap_n = rng.normal(0, .0065, N)
        close = np.zeros(N); openp = np.zeros(N)
        close[0] = openp[0] = self.start_price
        for k in range(1, N):
            openp[k] = close[k - 1] * np.exp(gap_n[k])
            g = gap_n[k]
            if .0025 < abs(g) < .015:
                close[k] = openp[k] * np.exp(
                    g * .55 + mu_d[k] + rng.normal(0, .0038))
            else:
                close[k] = openp[k] * np.exp(rng.normal(mu_d[k], sig_d[k]))
        dr   = np.abs(rng.normal(0, .006, N)) + .003
        high = np.maximum(close, openp) * (1 + dr * .55)
        low  = np.minimum(close, openp) * (1 - dr * .55)
        vol  = rng.lognormal(np.log(6e7), .3, N)
        dates = pd.bdate_range(self.start_date.strftime("%Y-%m-%d"), periods=N)
        df = pd.DataFrame({
            "date": dates, "open": openp, "high": high,
            "low": low, "close": close, "volume": vol})
        df["gap_pct"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)
        self._daily_cache = df

    def get_daily_bars(self, start: date, end: date) -> pd.DataFrame:
        df   = self._daily_cache
        mask = (df["date"].dt.date >= start) & (df["date"].dt.date <= end)
        return df[mask][DailyBar.COLS].reset_index(drop=True)

    def get_intraday_bars(self, trading_date: date,
                          interval_min: int = 5) -> pd.DataFrame:
        df = self._daily_cache
        row = df[df["date"].dt.date == trading_date]
        if len(row) == 0:
            return pd.DataFrame(columns=IntradayBar.COLS)
        row = row.iloc[0]
        return self._gen_intraday_from_daily(
            row["open"], row["high"], row["low"], row["close"],
            trading_date, interval_min,
            seed=int(abs(hash(str(trading_date))) % 2**32))

    @staticmethod
    def _gen_intraday_from_daily(open_p: float, high: float, low: float,
                                  close: float, trading_date: date,
                                  interval_min: int = 5,
                                  seed: int = 42) -> pd.DataFrame:
        BARS = 390 // interval_min
        rng  = np.random.default_rng(seed)
        gap  = (open_p - close) / close if close > 0 else 0.0
        daily_range = (high - low) / open_p
        vol_sched = np.concatenate([
            np.linspace(1.8, 1.2, 12), np.linspace(1.2, 0.9, 18),
            np.full(30, 0.9), np.linspace(0.9, 1.3, 18),
        ])[:BARS]
        bar_sig = (daily_range / 2) / BARS ** 0.5 * vol_sched
        cont    = np.zeros(BARS)
        if .0025 < abs(gap) < .015:
            cont[:18] = np.sign(gap) * min(abs(gap) / .005, 1.0) * .55 * \
                        bar_sig[:18] * np.exp(-np.arange(18) / 10)
        prices  = [open_p]
        for b in range(BARS):
            prices.append(prices[-1] * np.exp(cont[b] + rng.normal(0, bar_sig[b])))
        times  = pd.date_range(f"{trading_date} 09:35",
                                periods=BARS, freq=f"{interval_min}min")
        bars   = np.array(prices[1:])
        return pd.DataFrame({
            "datetime": times, "open": bars, "high": bars * (1 + bar_sig * .3),
            "low": bars * (1 - bar_sig * .3),
            "close": bars, "volume": rng.lognormal(np.log(6e7/BARS), .4, BARS)
        })

    def get_option_chain_at_open(self, trading_date: date,
                                  expiry: date) -> pd.DataFrame:
        intra = self.get_intraday_bars(trading_date)
        if len(intra) == 0:
            return pd.DataFrame(columns=OptionQuote.COLS)
        S  = intra.iloc[0]["open"]
        # For 0DTE (same-day expiry), use exactly 1 trading day remaining.
        # Calendar days = 0 would give T→0 (wrong). T = 1/252 is correct for
        # an option at market open with one full trading session remaining.
        cal_days = (datetime.combine(expiry, datetime.min.time()) -
                    datetime.combine(trading_date, datetime.min.time())).days
        T  = max(cal_days / 365, 1/252)   # floor at 1 trading day for 0DTE
        # IV: 21-day trailing HV × open spike factor
        daily = self.get_daily_bars(
            trading_date - timedelta(days=30), trading_date)
        if len(daily) > 5:
            lr = np.diff(np.log(daily["close"].values))
            hv = float(np.std(lr, ddof=1) * 252 ** 0.5)
        else:
            hv = 0.25
        iv_open = np.clip(hv, 0.10, 1.5) * 1.38   # open spike
        strikes = np.arange(round(S * 0.96), round(S * 1.04) + 1, 1.0)
        rows    = []
        for K in strikes:
            for call in [True, False]:
                g = bsm_greeks(S, K, T, self.rf, iv_open, call=call)
                if g["price"] < 0.01:
                    continue
                spread = self.exec_model.spread_pct(
                    iv_open, T, (K - S) / S) * g["price"]
                rows.append({
                    "datetime":         pd.Timestamp(f"{trading_date} 09:31:00"),
                    "underlying_price": S,
                    "symbol":           f"{trading_date}_{K}_{'C' if call else 'P'}",
                    "expiry":           pd.Timestamp(expiry),
                    "strike":           K,
                    "option_type":      "call" if call else "put",
                    "bid":              max(0, g["price"] - spread / 2),
                    "ask":              g["price"] + spread / 2,
                    "mid":              g["price"],
                    "iv":               iv_open,
                    "delta":            g["delta"],
                    "gamma":            g["gamma"],
                    "theta":            g["theta"],
                    "vega":             g["vega"],
                    "open_interest":    0,
                    "volume":           0,
                })
        return pd.DataFrame(rows, columns=OptionQuote.COLS)
