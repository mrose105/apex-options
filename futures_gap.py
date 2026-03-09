"""
APEX 0DTE — NQ Futures Gap Provider
=====================================
Replaces the QQQ RTH open-to-prev-close gap signal with a superior
overnight NQ/MNQ futures gap signal.

WHY FUTURES GAPS ARE BETTER THAN QQQ GAPS
------------------------------------------
QQQ gap (what we had):
    gap = (QQQ_open_9:30 - QQQ_prev_close_16:00) / QQQ_prev_close
    Problem: only captures the 4pm→9:30am window. Noisy. No pre-market context.

NQ futures gap (what this module provides):
    overnight_gap = (NQ_settlement_at_9:29 - NQ_prev_RTH_close_16:15) / NQ_prev_RTH_close
    Advantages:
    - NQ trades 23hrs/day → true overnight price discovery
    - Can measure PERSISTENCE: did the gap hold all night or just form at 9:25am?
    - Tighter bid/ask → cleaner signal
    - NQ leads QQQ by 1–5 minutes → earlier signal
    - Historical data free via yfinance back to 2002

NQ → QQQ CONVERSION
---------------------
NQ is quoted in index points. QQQ tracks the Nasdaq-100 at ~0.01× NQ price.
    Approximate conversion: QQQ ≈ NQ / 40  (varies slightly over time)
    NQ gap_pct ≈ QQQ gap_pct (both track Nasdaq-100, percentage moves identical)

So the signal gap_pct from NQ is directly usable as-is. No conversion needed
for the percentage gap — only if you need the dollar option strike.

DATA SOURCES
-------------
Primary (yfinance, FREE):
    - /NQ=F  : current NQ front-month (daily OHLCV)
    - NQH25.CME, NQM25.CME... : specific quarterly contracts
    - QQQ    : underlying for option strike computation

    pip install yfinance
    import yfinance as yf
    nq = yf.download("/NQ=F", start="2016-01-01", end="2026-01-01", interval="1d")

Alternative (if yfinance unavailable):
    - Barchart.com free daily download: https://www.barchart.com/futures/quotes/NQ*0/historical-download
    - Quandl/Nasdaq Data Link: CME_NQ1  (requires free account)
    - IBKR TWS: historical data API (free with account)
    - NinjaTrader: free historical futures data

INTRADAY NQ DATA (for gap persistence, optional)
-------------------------------------------------
    yf.download("/NQ=F", interval="5m", period="60d")  # last 60 days only, free
    For longer history: IBKR TWS, Rithmic, or Kinetick (~$20/mo)

USAGE
------
    from futures_gap import NQGapProvider, load_nq_from_yfinance, NQGapSignal

    # With yfinance (requires internet)
    nq_df, qqq_df = load_nq_from_yfinance("2016-01-01", "2026-01-01")
    provider = NQGapProvider(nq_df, qqq_df)

    # With local CSV (e.g. downloaded from Barchart)
    nq_df  = pd.read_csv("NQ_daily.csv", parse_dates=["Date"])
    qqq_df = pd.read_csv("QQQ_daily.csv", parse_dates=["Date"])
    provider = NQGapProvider(nq_df, qqq_df)

    # Get gap signal for a specific date
    signal = provider.gap_signal(date(2024, 3, 15))
    print(signal.gap_pct, signal.quality_score, signal.direction)
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("apex.futures")


# ─────────────────────────────────────────────────────────────────────────────
# NQ GAP SIGNAL  (richer than raw gap_pct)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class NQGapSignal:
    """
    All information extracted from the overnight NQ futures gap.
    The strategy engine only needs gap_pct and direction, but the
    richer fields enable more selective filtering.
    """
    trade_date:         date  = None

    # Core gap metrics
    gap_pct:            float = 0.0    # (NQ_open - NQ_prev_close) / NQ_prev_close
    gap_pts:            float = 0.0    # gap in NQ index points
    direction:          str   = ""     # 'call' or 'put'

    # NQ price context
    nq_prev_close:      float = 0.0   # NQ RTH settle prior day
    nq_open:            float = 0.0   # NQ open (or 9:29am if intraday available)

    # QQQ context (for option strike)
    qqq_prev_close:     float = 0.0
    qqq_open:           float = 0.0   # actual QQQ RTH open (for option entry)
    qqq_gap_pct:        float = 0.0   # QQQ's own gap (should ≈ nq gap_pct)

    # Signal quality metrics
    nq_qqq_gap_diff:    float = 0.0   # |NQ_gap% - QQQ_gap%|  (divergence = noise)
    gap_z_score:        float = 0.0   # gap / trailing_vol (how extreme vs history)
    vol_regime:         float = 0.0   # trailing 21-day NQ vol (annualised)
    vol_adjusted_gap:   float = 0.0   # gap_pct / vol_regime (comparable across regimes)

    # Calibration context
    gap_quantile:       float = 0.0   # where this gap falls in trailing distribution
    in_moderate_band:   bool  = False  # True if gap in 20th–80th percentile
    is_strong_gap:      bool  = False  # True if gap above 50th percentile of band

    # Derived
    quality_score:      float = 0.0   # 0–1 composite quality (higher = better signal)
    tradeable:          bool  = False  # passes all filters

    def __repr__(self):
        return (f"NQGapSignal({self.trade_date} {self.direction} "
                f"gap={self.gap_pct:+.4f} z={self.gap_z_score:+.2f} "
                f"quality={self.quality_score:.2f} tradeable={self.tradeable})")


# ─────────────────────────────────────────────────────────────────────────────
# NQ/QQQ DATA LOADER  (yfinance)
# ─────────────────────────────────────────────────────────────────────────────

def load_nq_from_yfinance(start: str = "2016-01-01",
                           end: str = None,
                           cache_path: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Download NQ futures (/NQ=F) and QQQ daily OHLCV from Yahoo Finance.

    Returns:
        nq_df:  DataFrame with columns [date, open, high, low, close, volume]
        qqq_df: DataFrame with columns [date, open, high, low, close, volume]

    Cache: if cache_path provided, saves/loads from CSV to avoid re-downloading.

    Example:
        nq_df, qqq_df = load_nq_from_yfinance("2016-01-01", "2026-01-01")
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError(
            "yfinance required: pip install yfinance\n"
            "Or load your own CSV: pd.read_csv('NQ_daily.csv', parse_dates=['date'])"
        )

    if end is None:
        import datetime
        end = datetime.date.today().isoformat()

    nq_cache  = (cache_path + "_nq.csv")  if cache_path else None
    qqq_cache = (cache_path + "_qqq.csv") if cache_path else None

    def _download(ticker, cache):
        if cache:
            try:
                df = pd.read_csv(cache, parse_dates=["date"])
                logger.info(f"Loaded {ticker} from cache: {cache}")
                return df
            except FileNotFoundError:
                pass
        logger.info(f"Downloading {ticker} {start} → {end}")
        raw = yf.download(ticker, start=start, end=end,
                          auto_adjust=True, progress=False)
        if raw.empty:
            raise ValueError(f"No data returned for {ticker}. "
                             "Check ticker, dates, and internet connection.")
        raw = raw.reset_index()
        # Handle MultiIndex columns from yfinance
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [c[0].lower() for c in raw.columns]
        else:
            raw.columns = [c.lower() for c in raw.columns]
        raw = raw.rename(columns={"date": "date", "open": "open",
                                   "high": "high", "low": "low",
                                   "close": "close", "volume": "volume"})
        raw["date"] = pd.to_datetime(raw["date"]).dt.date
        df = raw[["date","open","high","low","close","volume"]].dropna(
            subset=["close"]).sort_values("date").reset_index(drop=True)
        if cache:
            df.to_csv(cache, index=False)
            logger.info(f"Cached {ticker} → {cache}")
        return df

    nq_df  = _download("/NQ=F", nq_cache)
    qqq_df = _download("QQQ",   qqq_cache)
    logger.info(f"NQ:  {len(nq_df)} days  {nq_df['date'].min()} → {nq_df['date'].max()}")
    logger.info(f"QQQ: {len(qqq_df)} days  {qqq_df['date'].min()} → {qqq_df['date'].max()}")
    return nq_df, qqq_df


def load_nq_from_csv(nq_csv: str, qqq_csv: str,
                      date_col: str = "Date") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load NQ and QQQ data from local CSVs (e.g. downloaded from Barchart).

    Barchart CSV format: Date, Open, High, Low, Last, Change, %Chg, Volume, ...
    Yahoo Finance CSV:   Date, Open, High, Low, Close, Adj Close, Volume

    Both are handled automatically.
    """
    def _clean(path):
        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]
        # Handle various column names
        renames = {"last": "close", "adj close": "close",
                   "adj_close": "close", "settle": "close"}
        df = df.rename(columns={k: v for k, v in renames.items()
                                  if k in df.columns})
        df["date"] = pd.to_datetime(df[date_col.lower()]).dt.date
        df = df[["date","open","high","low","close","volume"]].dropna(
            subset=["close"])
        for col in ["open","high","low","close","volume"]:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(",",""),
                                     errors="coerce")
        return df.sort_values("date").reset_index(drop=True)

    return _clean(nq_csv), _clean(qqq_csv)


# ─────────────────────────────────────────────────────────────────────────────
# NQ GAP PROVIDER  (main class)
# ─────────────────────────────────────────────────────────────────────────────

class NQGapProvider:
    """
    Computes rich overnight NQ gap signals aligned to QQQ trading dates.

    After instantiation, call gap_signal(trade_date) for each day
    or get_all_signals(start, end) for a full date range.

    The signals replace gap_pct in the existing strategy engine.
    Everything else (options pricing, exit logic, portfolio management)
    is unchanged.
    """

    def __init__(self,
                 nq_df:  pd.DataFrame,
                 qqq_df: pd.DataFrame,
                 # Gap band calibration (percentiles of daily gaps)
                 band_lo_pct: float = 0.20,   # 20th percentile (lower edge of moderate gap)
                 band_hi_pct: float = 0.80,   # 80th percentile (upper edge)
                 band_mid_pct: float = 0.50,  # tier-1 threshold
                 # Rolling calibration window
                 cal_window:   int   = 252,   # 1-year rolling
                 # Gap quality filters
                 max_gap_z:    float = 3.5,   # reject extreme gap > 3.5σ (news events)
                 min_gap_z:    float = 0.3,   # reject tiny gap < 0.3σ (noise)
                 max_nq_qqq_div: float = 0.0015,  # reject if NQ/QQQ diverge >0.15%
                 ):
        self.nq  = nq_df.copy().set_index("date")
        self.qqq = qqq_df.copy().set_index("date")
        self.band_lo  = band_lo_pct
        self.band_hi  = band_hi_pct
        self.band_mid = band_mid_pct
        self.cal_win  = cal_window
        self.max_z    = max_gap_z
        self.min_z    = min_gap_z
        self.max_div  = max_nq_qqq_div

        # Pre-compute NQ daily gaps aligned to trading calendar
        self._build_nq_gaps()
        logger.info(f"NQGapProvider ready: {len(self._gaps)} trading days with NQ data")

    def _build_nq_gaps(self):
        """Pre-compute overnight NQ gap for every trading day in the dataset."""
        nq = self.nq.copy().sort_index()
        nq["nq_gap_pct"] = (nq["open"] - nq["close"].shift(1)) / nq["close"].shift(1)
        nq["nq_prev_close"] = nq["close"].shift(1)

        # 21-day realised vol (NQ daily log returns, annualised)
        nq["nq_lr"]   = np.log(nq["close"] / nq["close"].shift(1))
        nq["nq_rvol"] = nq["nq_lr"].rolling(21, min_periods=10).std() * 252**0.5

        self._gaps = nq[["open","close","nq_prev_close","nq_gap_pct","nq_rvol"]].dropna()

    def gap_signal(self, trade_date: date,
                   cal_end: date = None) -> Optional[NQGapSignal]:
        """
        Compute the full NQ gap signal for one trading day.

        trade_date: the day you want to trade (gaps are computed as of this open)
        cal_end:    last date to use for calibration (default = day before trade_date,
                    ensuring zero lookahead)

        Returns NQGapSignal or None if data unavailable.
        """
        if trade_date not in self._gaps.index:
            return None
        if trade_date not in self.qqq.index:
            return None

        cal_end = cal_end or (trade_date - timedelta(days=1))

        row_nq  = self._gaps.loc[trade_date]
        row_qqq = self.qqq.loc[trade_date]

        nq_gap  = float(row_nq["nq_gap_pct"])
        nq_open = float(row_nq["open"])
        nq_prev = float(row_nq["nq_prev_close"])
        rvol    = float(row_nq["nq_rvol"])
        qqq_open  = float(row_qqq["open"])

        # QQQ prev close
        qqq_dates = sorted(self.qqq.index)
        i = qqq_dates.index(trade_date)
        qqq_prev_close = float(self.qqq.loc[qqq_dates[i-1]]["close"]) if i > 0 else np.nan
        qqq_gap = (qqq_open - qqq_prev_close) / qqq_prev_close if qqq_prev_close else 0.0

        # Build calibration window (trailing cal_win bars, no lookahead)
        hist_gaps = self._gaps.loc[
            self._gaps.index <= cal_end
        ]["nq_gap_pct"].dropna().tail(self.cal_win)

        if len(hist_gaps) < 40:
            logger.debug(f"{trade_date}: insufficient history for calibration")
            return None

        # Split positive and negative gaps
        pos_gaps = hist_gaps[hist_gaps > 0]
        neg_gaps = hist_gaps[hist_gaps < 0]
        if len(pos_gaps) < 10 or len(neg_gaps) < 10:
            return None

        # Z-score of today's gap
        gap_std = float(hist_gaps.std(ddof=1))
        z_score = nq_gap / gap_std if gap_std > 0 else 0.0

        sig = NQGapSignal(trade_date=trade_date)
        sig.gap_pct         = nq_gap
        sig.gap_pts         = nq_open - nq_prev
        sig.nq_prev_close   = nq_prev
        sig.nq_open         = nq_open
        sig.qqq_prev_close  = qqq_prev_close
        sig.qqq_open        = qqq_open
        sig.qqq_gap_pct     = qqq_gap
        sig.nq_qqq_gap_diff = abs(nq_gap - qqq_gap)
        sig.gap_z_score     = z_score
        sig.vol_regime      = rvol
        sig.vol_adjusted_gap = nq_gap / rvol if rvol > 0 else 0.0

        # Calibrated band thresholds
        if nq_gap > 0:
            lo = float(np.quantile(pos_gaps, self.band_lo))
            hi = float(np.quantile(pos_gaps, self.band_hi))
            mid = float(np.quantile(pos_gaps, self.band_mid))
            in_band = (lo <= nq_gap <= hi)
            sig.direction = "call"
        else:
            # For negative gaps, lo < hi < 0 (lo is more negative)
            lo = float(np.quantile(neg_gaps, 1 - self.band_hi))  # most negative end
            hi = float(np.quantile(neg_gaps, 1 - self.band_lo))  # least negative end
            mid = float(np.quantile(neg_gaps, 0.50))
            in_band = (lo <= nq_gap <= hi)
            sig.direction = "put"

        sig.gap_quantile    = self._gap_quantile(nq_gap, hist_gaps)
        sig.in_moderate_band = in_band
        sig.is_strong_gap   = in_band and (
            (nq_gap > 0 and nq_gap >= mid) or
            (nq_gap < 0 and nq_gap <= mid)
        )

        # Quality filters
        nq_qqq_ok  = sig.nq_qqq_gap_diff <= self.max_div
        z_range_ok = (self.min_z <= abs(z_score) <= self.max_z)
        gap_ok     = in_band

        sig.quality_score = self._quality_score(sig, nq_qqq_ok, z_range_ok)
        sig.tradeable     = gap_ok and nq_qqq_ok and z_range_ok

        return sig

    def get_all_signals(self, start: date, end: date,
                        min_train_days: int = 252) -> pd.DataFrame:
        """
        Compute gap signals for every trading day in [start, end].
        Returns DataFrame with all NQGapSignal fields as columns.

        min_train_days: minimum history before first signal (prevents cold-start bias)
        """
        rows = []
        all_dates = sorted(d for d in self._gaps.index
                           if start <= d <= end
                           and d in self.qqq.index)

        for d in all_dates:
            # Check min training history
            hist = self._gaps.loc[self._gaps.index < d]
            if len(hist) < min_train_days:
                continue
            sig = self.gap_signal(d)
            if sig is not None:
                rows.append(vars(sig))

        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        return df.sort_values("trade_date").reset_index(drop=True)

    def calibration_summary(self, as_of: date, window: int = None) -> dict:
        """
        Show calibration thresholds as of a given date.
        Useful for understanding what gaps the strategy was targeting.
        """
        w = window or self.cal_win
        hist = self._gaps.loc[self._gaps.index < as_of]["nq_gap_pct"].tail(w).dropna()
        pos  = hist[hist > 0]; neg = hist[hist < 0]
        return {
            "as_of": as_of, "n_days": len(hist),
            "call_lo":  float(np.quantile(pos, self.band_lo)),
            "call_hi":  float(np.quantile(pos, self.band_hi)),
            "call_mid": float(np.quantile(pos, self.band_mid)),
            "put_lo":   float(np.quantile(neg, 1-self.band_hi)),
            "put_hi":   float(np.quantile(neg, 1-self.band_lo)),
            "put_mid":  float(np.quantile(neg, 0.50)),
            "rvol_21d": float(hist.std(ddof=1) * 252**0.5),
            "pct_gaps": float((hist != 0).mean()),
        }

    # ── helpers ──────────────────────────────────────────────────────────────

    def _gap_quantile(self, gap: float, hist: pd.Series) -> float:
        """Percentile of today's gap within same-sign historical gaps."""
        if gap > 0:
            s = hist[hist > 0]
        else:
            s = hist[hist < 0]
        if len(s) == 0:
            return 0.5
        return float((s <= gap).mean())

    def _quality_score(self, sig: NQGapSignal,
                        nq_qqq_ok: bool, z_range_ok: bool) -> float:
        """
        Composite 0–1 quality score.
        Components:
          - band position (0.4 weight): how central is gap in moderate band?
          - z-score range (0.3 weight): away from extremes = cleaner signal
          - nq/qqq alignment (0.3 weight): both futures and equity agree
        """
        # Band score: 1.0 at centre of band, 0 at edges
        q = sig.gap_quantile
        band_score = 1.0 - 2 * abs(q - 0.50)  # peaks at quantile=0.50

        # Z-score score: best at z=1.0–2.0, worse at extremes
        z = abs(sig.gap_z_score)
        z_score_v = max(0.0, min(1.0, 1.0 - abs(z - 1.5) / 1.5))

        # NQ/QQQ alignment
        align_score = max(0.0, 1.0 - sig.nq_qqq_gap_diff / 0.002)

        return 0.4 * band_score + 0.3 * z_score_v + 0.3 * align_score


# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC NQ DATA  (for testing without internet)
# ─────────────────────────────────────────────────────────────────────────────

def build_synthetic_nq(start_date: str = "2016-01-01",
                        n_days: int = 2500,
                        seed: int = 2024) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate synthetic NQ and QQQ daily bars calibrated to real NQ price history.

    NQ anchors (real approximate annual closes):
        2016: 4,850  2017: 6,430  2018: 6,100  2019: 8,510
        2020: 12,900 2021: 16,400 2022: 11,100 2023: 16,800
        2024: 21,800 2025: 21,200 2026: 24,400 (est.)

    QQQ ≈ NQ / 40  (rough constant, varies slightly)

    Key differences vs synthetic QQQ:
    - NQ gaps are larger in absolute points but similar in pct
    - NQ has slightly higher overnight volatility (futures trade 23hrs)
    - NQ/QQQ gap occasionally diverge 0.05-0.15% (this is real)
    """
    from math import log
    rng = np.random.default_rng(seed)

    # ── NQ regime segments (calibrated to real NQ history) ────────────────
    def rmu(s, e, n): return log(e/s) / (n/252)
    segs = [
        (12,   4850,  4510,  0.45),   # Jan 2016 selloff
        (247,  4510,  4870,  0.17),   # 2016 recovery
        (260,  4870,  6430,  0.16),   # 2017 melt-up
        (29,   6430,  5960,  0.30),   # Feb 2018 vol spike
        (231,  5960,  5420,  0.20),   # 2018 chop/Q4 crash
        (262,  5420,  8510,  0.18),   # 2019 melt-up
        (59,   8510,  7050,  0.75),   # COVID crash
        (203,  7050, 12900,  0.40),   # 2020 V-recovery
        (261, 12900, 16400,  0.17),   # 2021 bull
        (260, 16400, 11100,  0.35),   # 2022 bear
        (260, 11100, 16800,  0.20),   # 2023-24 AI bull
        (310, 16800, 21200,  0.22),   # 2025 chop/tariff
    ]
    N = sum(s[0] for s in segs)

    mu_d  = np.concatenate([np.full(s[0], rmu(s[1], s[2], s[0])/252) for s in segs])[:N]
    sig_d = np.concatenate([np.full(s[0], s[3]/252**0.5)             for s in segs])[:N]

    # NQ-specific: slightly wider overnight gap noise (futures trade 23hrs)
    nq_gap_sig = 0.0085   # vs QQQ 0.0065 (overnight futures more volatile)
    gap_n      = rng.normal(0, nq_gap_sig, N)

    # NQ prices
    nq_close = np.zeros(N); nq_open = np.zeros(N)
    nq_close[0] = 4850.0; nq_open[0] = 4850.0
    for k in range(1, N):
        nq_open[k]  = nq_close[k-1] * np.exp(gap_n[k])
        nq_close[k] = nq_open[k]   * np.exp(rng.normal(mu_d[k], sig_d[k]))

    dr       = np.abs(rng.normal(0, .006, N)) + .003
    nq_high  = np.maximum(nq_close, nq_open) * (1 + dr * .55)
    nq_low   = np.minimum(nq_close, nq_open) * (1 - dr * .55)
    nq_vol   = rng.lognormal(np.log(25e3), .3, N).astype(int)  # NQ volume ~25k/day

    dates = pd.bdate_range(start_date, periods=N)

    nq_df = pd.DataFrame({
        "date":   dates.date, "open": nq_open.round(2),
        "high":   nq_high.round(2), "low": nq_low.round(2),
        "close":  nq_close.round(2), "volume": nq_vol
    })

    # ── QQQ: NQ/40 with small idiosyncratic noise ──────────────────────────
    # QQQ tracks NQ-100 identically in pct, minor basis variation
    scale    = 1 / 40.0
    basis_n  = rng.normal(0, 0.0002, N)  # tiny NQ/QQQ basis noise
    qqq_close = nq_close * scale * np.exp(np.cumsum(basis_n - basis_n.mean()))
    qqq_open  = nq_open  * scale * (1 + rng.normal(0, 0.0003, N))  # tiny open divergence

    qqq_df = pd.DataFrame({
        "date":   dates.date, "open": qqq_open.round(3),
        "high":   (qqq_close * (1 + dr*.55)).round(3),
        "low":    (qqq_close * (1 - dr*.55)).round(3),
        "close":  qqq_close.round(3),
        "volume": (nq_vol * 40).astype(int)
    })

    return nq_df, qqq_df


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL ANALYSIS UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def analyse_gap_quality(signals_df: pd.DataFrame) -> dict:
    """
    Summarise quality metrics across all signals.
    Useful for understanding what the NQ gap signal looks like historically.
    """
    df = signals_df
    all_gaps = df[df["tradeable"] | ~df["tradeable"]]  # all days with gap data
    tradeable = df[df["tradeable"]]

    return {
        "total_days":          len(df),
        "tradeable_days":      len(tradeable),
        "trade_pct":           len(tradeable) / max(len(df), 1),
        "call_pct":            (tradeable["direction"] == "call").mean(),
        "put_pct":             (tradeable["direction"] == "put").mean(),
        "avg_gap_z":           tradeable["gap_z_score"].abs().mean(),
        "avg_quality_score":   tradeable["quality_score"].mean(),
        "nq_qqq_diverge_pct":  (~df["nq_qqq_gap_diff"].le(0.0015)).mean(),
        "extreme_gap_pct":     (df["gap_z_score"].abs() > 3.5).mean(),
        "tiny_gap_pct":        (df["gap_z_score"].abs() < 0.3).mean(),
        "avg_rvol":            df["vol_regime"].mean(),
    }


def plot_gap_analysis(signals_df: pd.DataFrame, nq_df: pd.DataFrame,
                       save_path: str = None):
    """
    Visualise gap signal quality and distribution.
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    df = signals_df.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    tradeable = df[df["tradeable"]]
    non_trade = df[~df["tradeable"]]

    fig, axes = plt.subplots(2, 3, figsize=(18, 9), facecolor="#0d1117")
    fig.suptitle("NQ Futures Gap Signal Analysis", color="white",
                 fontsize=14, fontweight="bold")

    RD="#ef4444"; GR="#22c55e"; BL="#3b82f6"; YL="#eab308"; MU="#94a3b8"; CY="#06b6d4"

    def style(ax, title):
        ax.set_facecolor("#161b22")
        for sp in ax.spines.values(): sp.set_color("#2d3748")
        ax.tick_params(colors=MU, labelsize=8)
        ax.set_title(title, color="white", fontsize=9, pad=6)
        ax.grid(True, color="#2d3748", lw=0.4, alpha=0.7)

    # 1. NQ price + gap events
    ax = axes[0, 0]
    style(ax, "NQ Futures + Trade Signals")
    nq = nq_df.copy(); nq["date"] = pd.to_datetime(nq["date"])
    nq = nq[nq["date"] >= df["trade_date"].min()]
    ax.plot(nq["date"], nq["close"]/1000, color=MU, lw=0.8, label="NQ (000s)")
    calls = tradeable[tradeable["direction"]=="call"]
    puts  = tradeable[tradeable["direction"]=="put"]
    for d, g in zip(calls["trade_date"], calls["gap_pct"]):
        ax.axvline(d, color=GR, alpha=0.15, lw=0.6)
    for d, g in zip(puts["trade_date"], puts["gap_pct"]):
        ax.axvline(d, color=RD, alpha=0.15, lw=0.6)
    ax.set_ylabel("NQ Price (thousands)", color=MU)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(fontsize=7, facecolor="#1e2530", labelcolor="white", edgecolor="#2d3748")

    # 2. Gap distribution
    ax = axes[0, 1]
    style(ax, f"Gap Distribution  ({len(tradeable)} tradeable days)")
    bins = np.linspace(-0.025, 0.025, 60)
    ax.hist(non_trade["gap_pct"], bins=bins, color=MU, alpha=0.5, label="Filtered out")
    ax.hist(tradeable[tradeable["direction"]=="call"]["gap_pct"],
            bins=bins, color=GR, alpha=0.8, label="Call trades")
    ax.hist(tradeable[tradeable["direction"]=="put"]["gap_pct"],
            bins=bins, color=RD, alpha=0.8, label="Put trades")
    ax.axvline(0, color="white", lw=0.8, ls="--")
    ax.set_xlabel("NQ Gap %", color=MU)
    ax.legend(fontsize=7, facecolor="#1e2530", labelcolor="white", edgecolor="#2d3748")

    # 3. Z-score distribution
    ax = axes[0, 2]
    style(ax, "Gap Z-Score (vol-adjusted)")
    bins_z = np.linspace(-5, 5, 50)
    ax.hist(df["gap_z_score"], bins=bins_z, color=BL, alpha=0.6, label="All gaps")
    ax.hist(tradeable["gap_z_score"], bins=bins_z, color=YL, alpha=0.8, label="Tradeable")
    ax.axvline(0.3, color=GR, lw=1, ls="--", label="Min z=0.3")
    ax.axvline(-0.3, color=GR, lw=1, ls="--")
    ax.axvline(3.5, color=RD, lw=1, ls="--", label="Max z=3.5")
    ax.axvline(-3.5, color=RD, lw=1, ls="--")
    ax.set_xlabel("Gap Z-Score", color=MU)
    ax.legend(fontsize=7, facecolor="#1e2530", labelcolor="white", edgecolor="#2d3748")

    # 4. NQ vs QQQ gap divergence
    ax = axes[1, 0]
    style(ax, "NQ vs QQQ Gap Divergence (should be ≈0)")
    ax.scatter(df["gap_pct"], df["qqq_gap_pct"],
               c=df["nq_qqq_gap_diff"].clip(0, 0.003),
               cmap="RdYlGn_r", alpha=0.4, s=8)
    lim = max(abs(df["gap_pct"]).max(), abs(df["qqq_gap_pct"]).max()) * 1.1
    ax.plot([-lim, lim], [-lim, lim], color=GR, lw=0.8, ls="--", label="NQ=QQQ")
    ax.set_xlabel("NQ Gap %", color=MU)
    ax.set_ylabel("QQQ Gap %", color=MU)
    ax.legend(fontsize=7, facecolor="#1e2530", labelcolor="white", edgecolor="#2d3748")

    # 5. Quality score distribution
    ax = axes[1, 1]
    style(ax, "Signal Quality Score")
    ax.hist(tradeable["quality_score"], bins=30, color=CY, alpha=0.8)
    ax.axvline(tradeable["quality_score"].mean(), color=YL, lw=1.5, ls="--",
               label=f"Mean {tradeable['quality_score'].mean():.2f}")
    ax.set_xlabel("Quality Score (0–1)", color=MU)
    ax.legend(fontsize=7, facecolor="#1e2530", labelcolor="white", edgecolor="#2d3748")

    # 6. Vol regime over time
    ax = axes[1, 2]
    style(ax, "Vol Regime (21d NQ rvol)")
    ax.fill_between(df["trade_date"], df["vol_regime"]*100,
                    color=BL, alpha=0.5, label="NQ rvol %")
    ax.plot(df["trade_date"], df["vol_regime"]*100, color=BL, lw=0.8)
    ax.axhline(20, color=YL, lw=0.8, ls="--", label="20% (normal)")
    ax.axhline(35, color=RD, lw=0.8, ls="--", label="35% (elevated)")
    ax.set_ylabel("Annualised Vol %", color=MU)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(fontsize=7, facecolor="#1e2530", labelcolor="white", edgecolor="#2d3748")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        logger.info(f"Gap analysis saved → {save_path}")
    plt.close(fig)
    return fig
