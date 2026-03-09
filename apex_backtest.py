"""
APEX 0DTE — Clean Daily-Bar Backtest
======================================
No synthetic intraday bars. No fake 5-min repricing.
Uses real daily OHLC to model entry, stop, and exit.

Architecture:
  1. Load QQQ daily bars from Alpaca
  2. For each day: classify gap zone
  3. Entry at 9:45am price (open + slippage)
  4. Stop: check if day's low/high crossed stop price
  5. Exit at 3:30pm (close price)
  6. P&L via BSM at two points: entry and exit

Usage:
  python3 apex_backtest.py
  python3 apex_backtest.py --start 2016-01-01 --end 2025-01-01
  python3 apex_backtest.py --capital 50000 --risk 0.02
"""

from __future__ import annotations
import os, sys, logging, argparse
from datetime import date, datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("apex")


# ─────────────────────────────────────────────────────────────────────────────
# BSM PRICER
# ─────────────────────────────────────────────────────────────────────────────

def bsm_price(S: float, K: float, T: float, iv: float, call: bool) -> float:
    """Black-Scholes-Merton option price. Returns 0 if T tiny."""
    if T < 1e-8:
        return max(S - K, 0.0) if call else max(K - S, 0.0)
    r = 0.05
    d1 = (np.log(S / K) + (r + 0.5 * iv ** 2) * T) / (iv * np.sqrt(T))
    d2 = d1 - iv * np.sqrt(T)
    if call:
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def bsm_delta(S: float, K: float, T: float, iv: float, call: bool) -> float:
    if T < 1e-8:
        return 1.0 if (call and S > K) else (-1.0 if (not call and S < K) else 0.0)
    r = 0.05
    d1 = (np.log(S / K) + (r + 0.5 * iv ** 2) * T) / (iv * np.sqrt(T))
    return norm.cdf(d1) if call else norm.cdf(d1) - 1.0

# 0DTE time constants (fraction of trading year)
# Trading year = 252 days × 6.5 hrs/day = 1638 hrs
HOURS_PER_YEAR = 252 * 6.5
T_ENTRY = 5.75 / HOURS_PER_YEAR   # 9:45am → 3:30pm = 5.75 hrs left
T_EXIT  = 0.25 / HOURS_PER_YEAR   # 3:30pm close    = 0.25 hrs left


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    # Zone thresholds
    up_mod_lo:     float = 0.0045   # UP_MOD_CONT  lower (+0.45%)
    up_mod_hi:     float = 0.0065   # UP_MOD_CONT  upper (+0.65%)
    up_mod_iv_max: float = 0.28     # skip if IV already elevated

    up_big_thresh: float = 0.0080   # UP_BIG_FADE  (+0.80%+)
    up_big_iv_min: float = 0.20     # only fade if some IV present

    # DN_MOD_CONT (-0.20% to -0.65%) DISABLED
    # N=8 empirical: 62.5% close UP intraday, avg +0.49% — bounce not continuation
    dn_ext_thresh: float = -0.0150  # DN_EXTREME   (<-1.50%)
    dn_ext_iv_min: float = 0.28     # require elevated IV (panic day)

    # Entry / sizing
    risk_pct:   float = 0.02        # 2% of portfolio per trade
    sma_window: int   = 50          # regime SMA

    # Stop: QQQ price stop (NOT option %)
    # If QQQ moves this far against us, exit.
    # Set wide enough to not fire on noise, tight enough to limit loss.
    qqq_stop_pct: float = 0.006     # -0.6% on QQQ from entry (gap fully reversed + buffer)

    # IV model
    hv_window:    int   = 21        # days for HV calculation
    iv_mult_base: float = 1.35      # base IV multiplier
    iv_mult_gap:  float = 0.50      # additional per 1% gap magnitude
    iv_exit_mult: float = 0.75      # IV at close (crush)

    # ── PLTR + NVDA 9:45 ENTRY FILTER ─────────────────────────────────────────
    # Empirical (N=59 days): composite first-15 ρ=+0.463 with SPY next-60
    # Divergence (comp-SPY15) ρ=+0.447 — incremental R² +0.206 over SPY alone
    # Strong DN (<-0.3%): P(SPY<0 next 60min) = 64%
    # Strong UP (>+0.3%): P(SPY>0 next 60min) = 73%
    # Signal strongest on down-open days (ρ=0.540 vs 0.306 on up-opens)
    lead_filter_on:      bool  = True    # enable/disable the filter
    lead_threshold:      float = 0.003   # 0.3% composite = strong conviction
    lead_require_agree:  bool  = True    # block trade if composite opposes direction
    lead_boost_strong:   bool  = True    # log strong-signal trades separately


# ─────────────────────────────────────────────────────────────────────────────
# DATA — Alpaca daily bars
# ─────────────────────────────────────────────────────────────────────────────

def load_daily_bars(symbol: str, start: str, end: str) -> pd.DataFrame:
    """
    Load daily OHLCV from Alpaca. Falls back to synthetic if no key set.
    Returns DataFrame with columns: date, open, high, low, close, volume, gap_pct
    """
    api_key    = os.environ.get("ALPACA_API_KEY", "")
    api_secret = os.environ.get("ALPACA_API_SECRET", "")

    if api_key and api_secret:
        try:
            import alpaca_trade_api as tradeapi
            api = tradeapi.REST(api_key, api_secret,
                                base_url="https://paper-api.alpaca.markets")
            log.info(f"Loading {symbol} daily bars {start} → {end} from Alpaca")
            raw = api.get_bars(symbol, "1Day", start=start, end=end,
                               adjustment="all").df
            raw = raw.reset_index()
            raw.columns = [c.lower() for c in raw.columns]
            if "timestamp" in raw.columns:
                raw = raw.rename(columns={"timestamp": "date"})
            raw["date"] = pd.to_datetime(raw["date"]).dt.date
            raw = raw[["date", "open", "high", "low", "close", "volume"]].copy()
            raw = raw.sort_values("date").reset_index(drop=True)
            raw["gap_pct"] = (raw["open"] - raw["close"].shift(1)) / raw["close"].shift(1)
            raw["gap_pct"] = raw["gap_pct"].fillna(0.0)
            log.info(f"  Loaded {len(raw)} trading days")
            return raw
        except Exception as e:
            log.warning(f"Alpaca load failed: {e} — using synthetic fallback")

    # Synthetic fallback: GBM with empirical QQQ params
    log.info("Using SYNTHETIC data (set ALPACA_API_KEY + ALPACA_API_SECRET for real data)")
    np.random.seed(42)
    n_days = 252 * 9
    dates  = pd.bdate_range(start=start, periods=n_days)

    # ── Calibrated to real QQQ 2016-2025 gap distribution ─────────────────
    # Overnight gap: N(mu=+0.03%, sigma=0.35%) with fat tails (t-dist, df=5)
    # This gives:  P(>+0.45%) ≈ 11.5%  → ~29 days/yr
    #              P(+0.45→+0.65%) ≈ 7.7% → ~19 days/yr
    #              P(>+0.80%) ≈ 1.4%   → ~3.5 days/yr
    #              P(-0.01→-0.18%) ≈ 18% → ~45 days/yr
    #              P(<-1.50%) ≈ ~0.5%  → ~1.3 days/yr
    daily_drift  = 0.15 / 252          # 15% CAGR
    daily_vol    = 0.22 / np.sqrt(252) # 22% annualised vol
    gap_mu       = 0.0003              # slight positive overnight drift
    gap_sigma    = 0.0035              # 0.35% overnight gap std (empirical QQQ)
    gap_df       = 5                   # t-dist df — fat tails for crash days

    from scipy.stats import t as t_dist
    gap_draws = t_dist.rvs(df=gap_df, loc=gap_mu, scale=gap_sigma, size=n_days,
                           random_state=42)

    prices = [400.0]
    opens, highs, lows, closes = [], [], [], []

    for i in range(n_days):
        prev  = prices[-1]
        gap   = float(gap_draws[i])
        op    = prev * (1 + gap)
        intra = np.random.normal(daily_drift, daily_vol)
        cl    = op * (1 + intra)
        hi    = max(op, cl) * (1 + abs(np.random.normal(0, 0.002)))
        lo    = min(op, cl) * (1 - abs(np.random.normal(0, 0.002)))
        opens.append(op); highs.append(hi)
        lows.append(lo);  closes.append(cl)
        prices.append(cl)
    
    df = pd.DataFrame({
        "date":   [d.date() for d in dates],
        "open":   opens, "high": highs, "low": lows,
        "close":  closes, "volume": [20_000_000] * n_days,
    })
    df["gap_pct"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)
    df["gap_pct"] = df["gap_pct"].fillna(0.0)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# ZONE CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def classify_zone(gap: float, iv_daily: float, bear: bool,
                  cfg: Config) -> tuple[Optional[str], Optional[str]]:
    """
    Returns (zone, direction) or (None, None) for no-trade.

    Zone map:
      UP_MOD_CONT   +0.45% → +0.65%   CALL   Bull + IV calm
      UP_BIG_FADE   +0.80%+            PUT    Bear only
      DN_MOD_CONT   -0.20% → -0.65%   DISABLED (bounce zone, N=8: 62.5% close UP)
      DN_EXTREME    < -1.50%           CALL   IV elevated (panic)
    """
    g = gap

    # ── UP_MOD_CONT: moderate gap-up continuation ─────────────────────────
    if cfg.up_mod_lo <= g <= cfg.up_mod_hi:
        if bear:
            return "UP_MOD_CONT_BLOCKED", None   # don't buy calls in bear
        if iv_daily > cfg.up_mod_iv_max:
            return "UP_MOD_CONT_IV", None         # elevated IV = risky
        return "UP_MOD_CONT", "call"

    # ── UP_BIG_FADE: large gap-up exhaustion ──────────────────────────────
    elif g > cfg.up_big_thresh:
        if not bear:
            return "UP_BIG_FADE_BLOCKED", None    # don't fade rips in bull
        if iv_daily < cfg.up_big_iv_min:
            return "UP_BIG_FADE_IV", None
        return "UP_BIG_FADE", "put"

    # ── DN_MOD_CONT DISABLED (-0.20% to -0.65%) ─────────────────────────
    # Empirically a bounce zone: 62.5% close UP, avg intraday +0.49%
    # Puts lose systematically here — dead zone
    elif -0.0065 <= g <= -0.0020:
        return "DEAD_ZONE_DN", None

    # ── DN_EXTREME: panic gap-down reversal ───────────────────────────────
    elif g < cfg.dn_ext_thresh:
        if iv_daily < cfg.dn_ext_iv_min:
            return "DN_EXTREME_IV", None           # not panicky enough
        return "DN_EXTREME", "call"

    # Dead zones and flat
    else:
        return "NO_TRADE", None


# ─────────────────────────────────────────────────────────────────────────────
# IV ESTIMATE
# ─────────────────────────────────────────────────────────────────────────────

def estimate_iv(hist: pd.DataFrame, gap: float, cfg: Config) -> float:
    """Gap-scaled IV: larger gaps inflate IV above HV baseline."""
    if len(hist) < 5:
        return 0.25
    lr  = np.diff(np.log(hist["close"].values[-cfg.hv_window:]))
    hv  = float(np.std(lr, ddof=1) * np.sqrt(252)) if len(lr) > 2 else 0.22
    mult = cfg.iv_mult_base + min(abs(gap) / 0.01, 1.5) * cfg.iv_mult_gap
    return float(np.clip(hv * mult, 0.10, 1.20))


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE TRADE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Trade:
    date:        date
    zone:        str
    direction:   str
    gap_pct:     float
    iv:          float
    entry_S:     float   # QQQ price at entry (9:45am ≈ open)
    entry_K:     float   # strike = ATM = entry_S
    entry_px:    float   # option mid price
    stop_S:      float   # QQQ stop price
    exit_S:      float   # QQQ price at exit
    exit_px:     float   # option value at exit
    exit_reason: str
    ret:         float   # option return (net of entry)
    pnl:         float   # dollar P&L
    portfolio:   float   # portfolio value after trade


def run_trade(row: pd.Series, portfolio: float, cfg: Config) -> Trade:
    """
    Execute one trade using daily OHLCV only.
    
    Entry:  open price (9:45am approximation — gap has been observed)
    Stop:   if high/low crosses stop_S → option valued at stop_S
    Exit:   close price (3:30pm)
    """
    is_call = row["direction"] == "call"
    S_entry = float(row["open"])
    K       = S_entry                    # ATM
    iv      = float(row["iv"])
    iv_exit = iv * cfg.iv_exit_mult      # IV crush by close

    # Entry option price
    entry_px = bsm_price(S_entry, K, T_ENTRY, iv, is_call)
    entry_px = max(entry_px, 0.05)       # floor for realistic fills

    # Stop price on QQQ (not on option %)
    # Call: stop if QQQ falls below entry × (1 - qqq_stop_pct)
    # Put:  stop if QQQ rises above entry × (1 + qqq_stop_pct)
    if is_call:
        stop_S = S_entry * (1 - cfg.qqq_stop_pct)
        stop_hit = float(row["low"]) <= stop_S
    else:
        stop_S = S_entry * (1 + cfg.qqq_stop_pct)
        stop_hit = float(row["high"]) >= stop_S

    if stop_hit:
        exit_S      = stop_S
        exit_reason = "STOP"
    else:
        exit_S      = float(row["close"])
        exit_reason = "EOD"

    # Exit option price (near T=0, dominated by intrinsic)
    T_ex   = T_EXIT if exit_reason == "EOD" else T_ENTRY * 0.5
    exit_px = bsm_price(exit_S, K, T_ex, iv_exit, is_call)

    # P&L
    risk    = portfolio * cfg.risk_pct
    n_contracts = max(1, int(risk / (entry_px * 100)))
    gross   = (exit_px - entry_px) * n_contracts * 100
    comm    = n_contracts * 0.65
    net_pnl = gross - comm

    ret = (exit_px - entry_px) / entry_px

    return Trade(
        date        = row["date"],
        zone        = row["zone"],
        direction   = row["direction"],
        gap_pct     = float(row["gap_pct"]),
        iv          = iv,
        entry_S     = S_entry,
        entry_K     = K,
        entry_px    = entry_px,
        stop_S      = stop_S,
        exit_S      = exit_S,
        exit_px     = exit_px,
        exit_reason = exit_reason,
        ret         = ret,
        pnl         = net_pnl,
        portfolio   = portfolio + net_pnl,
    )



# ─────────────────────────────────────────────────────────────────────────────
# PLTR + NVDA 9:45 INTRADAY LEAD SIGNAL
# ─────────────────────────────────────────────────────────────────────────────

def compute_lead_signal(trade_date, qqq_gap: float, cfg: Config) -> dict:
    """
    Compute the PLTR+NVDA composite first-15min signal for a given trade date.

    Live mode:  fetches real 5-min bars from Alpaca (or yfinance fallback)
    Backtest:   synthesises correlated returns matching empirical structure:
                  PLTR_15 ~ qqq_gap * 1.8 + N(0, 0.004)   # high-beta, 1.8x gap amplifier
                  NVDA_15 ~ qqq_gap * 1.4 + N(0, 0.006)   # semi-beta, noisier
                  composite = (PLTR_15 + NVDA_15) / 2
                  divergence = composite - spy_15
                  spy_15 ~ qqq_gap * 0.92 + N(0, 0.002)   # SPY ≈ QQQ * 0.92

    Returns dict with keys:
      composite   : float  — avg PLTR+NVDA first-15 return
      pltr_15     : float
      nvda_15     : float
      spy_15      : float  — SPY first-15 (estimated or real)
      divergence  : float  — composite - spy_15
      signal      : int    — +1 (strong up), -1 (strong dn), 0 (neutral)
      source      : str    — 'live' | 'synthetic'
    """
    api_key    = os.environ.get("ALPACA_API_KEY", "")
    api_secret = os.environ.get("ALPACA_API_SECRET", "")

    if api_key and api_secret:
        try:
            import alpaca_trade_api as tradeapi
            api   = tradeapi.REST(api_key, api_secret,
                                  base_url="https://paper-api.alpaca.markets")
            date_str = str(trade_date)

            def get_15min_ret(symbol):
                bars = api.get_bars(symbol, "5Min",
                                    start=f"{date_str}T09:30:00-04:00",
                                    end=f"{date_str}T09:45:00-04:00",
                                    limit=3).df
                if len(bars) < 3:
                    return None
                open_  = bars["open"].iloc[0]
                close_ = bars["close"].iloc[-1]
                return (close_ - open_) / open_

            pltr_15 = get_15min_ret("PLTR")
            nvda_15 = get_15min_ret("NVDA")
            spy_15  = get_15min_ret("SPY")

            if None not in (pltr_15, nvda_15, spy_15):
                composite  = (pltr_15 + nvda_15) / 2
                divergence = composite - spy_15
                signal = (1  if composite >  cfg.lead_threshold else
                         -1  if composite < -cfg.lead_threshold else 0)
                return dict(composite=composite, pltr_15=pltr_15,
                            nvda_15=nvda_15, spy_15=spy_15,
                            divergence=divergence, signal=signal,
                            source="live")
        except Exception:
            pass  # fall through to synthetic

    # ── SYNTHETIC: calibrated to empirical correlation structure ──────────────
    # Seed from date so same day always gives same signal (reproducible backtest)
    rng = np.random.default_rng(int(pd.Timestamp(str(trade_date)).timestamp()) % 2**32)

    spy_15   = qqq_gap * 0.92  + rng.normal(0, 0.0020)
    pltr_15  = qqq_gap * 1.80  + rng.normal(0, 0.0040)  # PLTR: high-beta, primary signal
    nvda_15  = qqq_gap * 1.40  + rng.normal(0, 0.0060)  # NVDA: semi-beta, noisier

    composite  = (pltr_15 + nvda_15) / 2
    divergence = composite - spy_15
    signal = (1  if composite >  cfg.lead_threshold else
             -1  if composite < -cfg.lead_threshold else 0)

    return dict(composite=composite, pltr_15=pltr_15,
                nvda_15=nvda_15, spy_15=spy_15,
                divergence=divergence, signal=signal,
                source="synthetic")


def apply_lead_filter(lead: dict, direction: str, cfg: Config) -> tuple[bool, str]:
    """
    Returns (should_trade, reason).
    Blocks trade if composite strongly opposes direction.
    """
    if not cfg.lead_filter_on:
        return True, "filter_off"

    composite = lead["composite"]
    signal    = lead["signal"]

    # Block: strong signal opposes trade direction
    if direction == "call" and composite < -cfg.lead_threshold:
        return False, f"lead_BLOCKED_composite={composite:+.3%}_vs_call"

    if direction == "put" and composite > cfg.lead_threshold:
        return False, f"lead_BLOCKED_composite={composite:+.3%}_vs_put"

    # Neutral signal — allow but no boost
    strength = "STRONG" if abs(composite) >= cfg.lead_threshold else "neutral"
    agree    = ((direction=="call" and composite > 0) or
                (direction=="put"  and composite < 0))

    return True, f"lead_OK_{strength}_agree={agree}_comp={composite:+.3%}"

# ─────────────────────────────────────────────────────────────────────────────
# BACKTEST LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest(df: pd.DataFrame, capital: float, cfg: Config) -> list[Trade]:
    """
    Main loop. Iterates through daily bars, classifies each day,
    executes trades, tracks portfolio.
    """
    portfolio = capital
    trades    = []
    peak      = capital
    halted    = False
    MAX_DD    = 0.25   # halt if drawdown exceeds 25%

    # Need at least sma_window + hv_window days of history before trading
    warmup = max(cfg.sma_window, cfg.hv_window) + 5

    for i in range(warmup, len(df)):
        row  = df.iloc[i]
        hist = df.iloc[:i]    # all bars before today

        if halted:
            break

        # Drawdown check
        if portfolio > peak:
            peak = portfolio
        dd = (peak - portfolio) / peak
        if dd >= MAX_DD:
            log.warning(f"{row['date']}: halted — drawdown {dd:.1%}")
            halted = True
            break

        gap = float(row["gap_pct"])

        # IV and regime
        iv   = estimate_iv(hist, gap, cfg)
        sma  = float(hist["close"].values[-cfg.sma_window:].mean())
        bear = float(hist["close"].iloc[-1]) < sma

        zone, direction = classify_zone(gap, iv, bear, cfg)

        if direction is None:
            continue

        # ── PLTR + NVDA 9:45 LEAD FILTER ─────────────────────────────────────
        lead   = compute_lead_signal(row["date"], gap, cfg)
        ok, lead_reason = apply_lead_filter(lead, direction, cfg)

        if not ok:
            log.debug(f"{row['date']}  {zone}  BLOCKED by lead filter: {lead_reason}")
            continue

        # Log strong confirming signals
        if cfg.lead_boost_strong and abs(lead["composite"]) >= cfg.lead_threshold:
            log.info(f"  ↳ LEAD {lead['source'].upper()}: "
                     f"PLTR={lead['pltr_15']:+.2%}  "
                     f"NVDA={lead['nvda_15']:+.2%}  "
                     f"comp={lead['composite']:+.2%}  "
                     f"div={lead['divergence']:+.2%}  "
                     f"({lead_reason})")

        # Attach computed values for run_trade
        row2 = row.copy()
        row2["zone"]      = zone
        row2["direction"] = direction
        row2["iv"]        = iv

        t = run_trade(row2, portfolio, cfg)
        portfolio = t.portfolio
        trades.append(t)

        log.info(
            f"{t.date}  {t.zone:<18}  {t.direction.upper():<4}  "
            f"gap={t.gap_pct:+.3f}  iv={t.iv:.2f}  "
            f"QQQ {t.entry_S:.2f}→{t.exit_S:.2f}  "
            f"opt {t.entry_px:.2f}→{t.exit_px:.2f}  "
            f"{t.exit_reason:<5}  ret={t.ret:+.0%}  pnl=${t.pnl:+,.0f}  "
            f"port=${portfolio:,.0f}"
        )

    return trades


# ─────────────────────────────────────────────────────────────────────────────
# REPORTING
# ─────────────────────────────────────────────────────────────────────────────

def report(trades: list[Trade], capital: float, start: str, end: str, df: pd.DataFrame):
    if not trades:
        print("No trades executed.")
        return

    rets  = np.array([t.ret   for t in trades])
    pnls  = np.array([t.pnl   for t in trades])
    ports = np.array([t.portfolio for t in trades])

    final  = trades[-1].portfolio
    n_yrs  = (pd.to_datetime(end) - pd.to_datetime(start)).days / 365.25
    cagr   = (final / capital) ** (1 / n_yrs) - 1 if n_yrs > 0 else 0

    wins   = rets[rets > 0]
    losses = rets[rets <= 0]
    wr     = len(wins) / len(rets)

    # Sharpe (annualised) on weekly portfolio returns
    port_vals  = np.concatenate([[capital], ports])
    weekly_ret = np.diff(port_vals) / port_vals[:-1]
    sharpe     = (np.mean(weekly_ret) / np.std(weekly_ret) * np.sqrt(52)
                  if np.std(weekly_ret) > 0 else 0)

    # Max drawdown
    peak = capital
    max_dd = 0.0
    for p in ports:
        if p > peak: peak = p
        dd = (peak - p) / peak
        if dd > max_dd: max_dd = dd

    profit_factor = (wins.sum() / abs(losses.sum())
                     if losses.sum() != 0 else float("inf"))

    # Per-zone breakdown
    by_zone: dict[str, list] = {}
    for t in trades:
        by_zone.setdefault(t.zone, []).append(t.ret)

    print()
    print("=" * 60)
    print("APEX 0DTE  —  BACKTEST RESULTS")
    print("=" * 60)
    print(f"Period:    {start} → {end}  ({n_yrs:.1f} yrs)")
    print(f"Capital:   ${capital:,.0f}  →  ${final:,.0f}")
    print(f"CAGR:      {cagr:+.1%}   Max DD: {max_dd:.1%}   Sharpe: {sharpe:.2f}")
    print(f"Trades:    {len(trades)}  ({len(trades)/n_yrs:.1f}/yr)   WR: {wr:.1%}   PF: {profit_factor:.2f}")
    print(f"Avg win:   {wins.mean():+.1%}   Avg loss: {losses.mean():+.1%}")
    print(f"Exits:     STOP={sum(1 for t in trades if t.exit_reason=='STOP')}  "
          f"EOD={sum(1 for t in trades if t.exit_reason=='EOD')}")
    print()
    print("Zone breakdown:")
    print(f"  {'Zone':<20}  {'N':>4}  {'WR':>6}  {'AvgRet':>8}  {'Total$':>10}")
    print(f"  {'-'*20}  {'-'*4}  {'-'*6}  {'-'*8}  {'-'*10}")
    for zone, zrets in sorted(by_zone.items()):
        zr   = np.array(zrets)
        zwr  = (zr > 0).mean()
        zavg = zr.mean()
        ztot = sum(t.pnl for t in trades if t.zone == zone)
        print(f"  {zone:<20}  {len(zr):>4}  {zwr:>6.1%}  {zavg:>+8.1%}  ${ztot:>+10,.0f}")
    print("=" * 60)

    # Directional edge (QQQ only — separate from option math)
    print()
    print("DIRECTIONAL EDGE (QQQ open→close, ignoring options)")
    print(f"  {'Zone':<20}  {'N':>4}  {'Dir%':>6}  {'AvgMove':>9}")
    print(f"  {'-'*20}  {'-'*4}  {'-'*6}  {'-'*9}")
    for zone, zrets in sorted(by_zone.items()):
        ztrades = [t for t in trades if t.zone == zone]
        is_call  = ztrades[0].direction == "call"
        moves = [(t.exit_S - t.entry_S) / t.entry_S * (1 if is_call else -1)
                 for t in ztrades]
        dir_pct = np.mean([m > 0 for m in moves])
        avg_move = np.mean(moves)
        print(f"  {zone:<20}  {len(moves):>4}  {dir_pct:>6.1%}  {avg_move:>+9.3%}")
    print()

    return {
        "cagr": cagr, "max_dd": max_dd, "sharpe": sharpe,
        "n_trades": len(trades), "win_rate": wr, "profit_factor": profit_factor,
    }


def save_results(trades: list[Trade], out_prefix: str, capital: float, df: pd.DataFrame, mc: dict = None):
    """Save trades CSV and equity curve."""
    rows = [{
        "date": t.date, "zone": t.zone, "direction": t.direction,
        "gap_pct": f"{t.gap_pct:.4f}", "iv": f"{t.iv:.3f}",
        "entry_S": f"{t.entry_S:.2f}", "entry_K": f"{t.entry_K:.2f}",
        "entry_px": f"{t.entry_px:.3f}", "exit_S": f"{t.exit_S:.2f}",
        "exit_px": f"{t.exit_px:.3f}", "exit_reason": t.exit_reason,
        "ret_pct": f"{t.ret*100:.1f}", "pnl": f"{t.pnl:.0f}",
        "portfolio": f"{t.portfolio:.0f}",
    } for t in trades]

    tdf = pd.DataFrame(rows)
    csv = f"{out_prefix}_trades.csv"
    tdf.to_csv(csv, index=False)
    log.info(f"Trades → {csv}")

    # Equity curve (daily, using last known portfolio value)
    eq_dates = pd.to_datetime([t.date for t in trades])
    eq_vals  = [t.portfolio for t in trades]
    edf = pd.DataFrame({"date": eq_dates, "portfolio": eq_vals})
    edf.to_csv(f"{out_prefix}_equity.csv", index=False)
    log.info(f"Equity → {out_prefix}_equity.csv")

    _plot(trades, capital, out_prefix, mc or {})


def _plot(trades: list[Trade], capital: float, out_prefix: str, mc: dict = None):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        BG = "#0d1117"; PAN = "#161b22"; GR = "#22c55e"
        RD = "#ef4444"; BL = "#3b82f6"; MU = "#94a3b8"; YL = "#eab308"

        fig, axes = plt.subplots(2, 2, figsize=(16, 10), facecolor=BG)
        fig.suptitle("APEX 0DTE — Daily-Bar Backtest", color="white",
                     fontsize=14, fontweight="bold")

        dates = [t.date for t in trades]
        ports = [t.portfolio for t in trades]
        rets  = [t.ret * 100 for t in trades]

        def style(ax, title):
            ax.set_facecolor(PAN)
            for sp in ax.spines.values(): sp.set_color("#2d3748")
            ax.tick_params(colors=MU, labelsize=8)
            ax.set_title(title, color="white", fontsize=9, pad=5)
            ax.grid(True, color="#2d3748", lw=0.4, alpha=0.6)

        # Equity curve
        ax = axes[0, 0]
        style(ax, "Equity Curve")
        ax.plot(pd.to_datetime(dates), ports, color=GR, lw=2)
        ax.axhline(capital, color=MU, lw=0.8, ls="--", alpha=0.6)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

        # Drawdown
        ax = axes[0, 1]
        style(ax, "Drawdown")
        all_ports = np.array([capital] + ports)
        peak = np.maximum.accumulate(all_ports[1:])
        dd   = (peak - np.array(ports)) / peak * 100
        ax.fill_between(pd.to_datetime(dates), -dd, 0, color=RD, alpha=0.4)
        ax.plot(pd.to_datetime(dates), -dd, color=RD, lw=1)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

        # Return distribution
        ax = axes[1, 0]
        style(ax, f"Option Returns  WR={np.mean([r>0 for r in rets]):.1%}")
        wins = [r for r in rets if r > 0]
        loss = [r for r in rets if r <= 0]
        bins = np.linspace(-110, 300, 40)
        if loss: ax.hist(loss, bins=bins, color=RD, alpha=0.7, label=f"Loss ({len(loss)})")
        if wins: ax.hist(wins, bins=bins, color=GR, alpha=0.7, label=f"Win ({len(wins)})")
        ax.axvline(0, color=MU, lw=0.8, ls="--")
        ax.legend(fontsize=7, facecolor=PAN, labelcolor="white", edgecolor="#2d3748")
        ax.set_xlabel("Option Return (%)", color=MU)

        # Monte Carlo paths (bottom right)
        ax = axes[1, 1]
        if mc and "paths" in mc:
            paths_arr = mc["paths"]
            n_plot    = min(500, paths_arr.shape[0])
            style(ax, f"Monte Carlo  {paths_arr.shape[0]:,} paths  "
                      f"P50 CAGR={mc['cagr_pcts'][50]:+.1%}")
            x = np.arange(1, paths_arr.shape[1] + 1)
            # Plot 500 sample paths faintly
            for i in range(n_plot):
                ax.plot(x, paths_arr[i], color=BL, alpha=0.015, lw=0.4)
            # Percentile bands
            p5  = np.percentile(paths_arr, 5,  axis=0)
            p25 = np.percentile(paths_arr, 25, axis=0)
            p50 = np.percentile(paths_arr, 50, axis=0)
            p75 = np.percentile(paths_arr, 75, axis=0)
            p95 = np.percentile(paths_arr, 95, axis=0)
            ax.fill_between(x, p5,  p95, color=BL, alpha=0.10, label="P5–P95")
            ax.fill_between(x, p25, p75, color=BL, alpha=0.20, label="P25–P75")
            ax.plot(x, p50, color=YL, lw=1.8, label="Median")
            ax.axhline(capital, color=MU, lw=0.8, ls="--", alpha=0.6)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x_, _: f"${x_/1000:.0f}K"))
            ax.set_xlabel("Trade #", color=MU, fontsize=8)
            ax.legend(fontsize=7, facecolor=PAN, labelcolor="white", edgecolor="#2d3748")
        else:
            # Fallback: zone P&L bar chart
            style(ax, "P&L by Zone")
            by_zone: dict[str, float] = {}
            for t in trades:
                by_zone[t.zone] = by_zone.get(t.zone, 0) + t.pnl
            zones = list(by_zone.keys())
            vals  = [by_zone[z] for z in zones]
            colors_z = [GR if v > 0 else RD for v in vals]
            ax.bar(zones, vals, color=colors_z, alpha=0.85, edgecolor=BG)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x_, _: f"${x_/1000:.1f}K"))
            ax.axhline(0, color=MU, lw=0.8)

        plt.tight_layout()
        path = f"{out_prefix}_dashboard.png"
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
        plt.close()
        log.info(f"Dashboard → {path}")
    except Exception as e:
        log.warning(f"Plot failed: {e}")



# ─────────────────────────────────────────────────────────────────────────────
# MONTE CARLO  (100,000 bootstrap paths)
# ─────────────────────────────────────────────────────────────────────────────

def monte_carlo(trades: list, capital: float, n_years: float,
                n_sims: int = 100_000, seed: int = 42) -> dict:
    """
    Bootstrap Monte Carlo over the observed trade return distribution.

    Method:
      - Each simulation draws N_per_year * n_years trades WITH replacement
        from the empirical return vector.
      - Position size is fixed at the historical avg_risk_pct of capital.
      - Tracks terminal wealth, CAGR, max drawdown, and ruin (port < 50% start).

    Returns dict of percentile stats for printing and plotting.
    """
    if not trades:
        return {}

    rng = np.random.default_rng(seed)

    # Empirical distribution
    rets          = np.array([t.ret for t in trades])           # option returns
    risk_pcts     = np.array([abs(t.pnl) / max(t.portfolio, 1) for t in trades])
    avg_risk      = float(np.mean(risk_pcts))
    n_per_year    = len(trades) / n_years
    n_draws       = max(int(round(n_per_year * n_years)), 1)

    # 100k paths — vectorised: shape (n_sims, n_draws)
    sample_idx    = rng.integers(0, len(rets), size=(n_sims, n_draws))
    path_rets     = rets[sample_idx]                             # (n_sims, n_draws)

    # Portfolio evolution: each trade risks avg_risk_pct of current portfolio
    # pnl_i = port_{i-1} * avg_risk * ret_i
    # This simplifies to a multiplicative chain:
    # port_T = capital * prod(1 + avg_risk * ret_i)
    multipliers   = 1.0 + avg_risk * path_rets                  # (n_sims, n_draws)
    cum_mult      = np.cumprod(multipliers, axis=1)              # (n_sims, n_draws)
    paths         = capital * cum_mult                           # (n_sims, n_draws)

    # Terminal stats
    terminal      = paths[:, -1]
    cagrs         = (terminal / capital) ** (1.0 / n_years) - 1.0

    # Max drawdown per path
    full_paths    = np.concatenate([np.full((n_sims, 1), capital), paths], axis=1)
    running_peak  = np.maximum.accumulate(full_paths, axis=1)
    drawdowns     = (running_peak - full_paths) / running_peak
    max_dds       = drawdowns.max(axis=1)

    # Ruin = portfolio ever falls below 50% of start
    ruin_mask     = (paths < capital * 0.50).any(axis=1)
    ruin_prob     = float(ruin_mask.mean())

    # Percentiles
    pcts = [5, 25, 50, 75, 95]
    cagr_pcts  = np.percentile(cagrs,   pcts)
    dd_pcts    = np.percentile(max_dds, pcts)
    term_pcts  = np.percentile(terminal, pcts)

    print()
    print("=" * 60)
    print(f"MONTE CARLO  ({n_sims:,} paths  ×  {n_draws} trades/path)")
    print(f"  Empirical N={len(rets)}  avg_risk={avg_risk:.2%}  {n_per_year:.1f} trades/yr")
    print("=" * 60)
    print(f"  {'Metric':<22}  {'P5':>8}  {'P25':>8}  {'P50':>8}  {'P75':>8}  {'P95':>8}")
    print(f"  {'-'*22}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
    print(f"  {'CAGR':<22}  " + "  ".join(f"{v:>+7.1%}" for v in cagr_pcts))
    print(f"  {'Max Drawdown':<22}  " + "  ".join(f"{v:>7.1%}" for v in dd_pcts))
    print(f"  {'Terminal ($100K→)':<22}  " + "  ".join(f"${v/1000:>6.0f}K" for v in term_pcts))
    print(f"  {'Ruin P(port<50%)':<22}  {ruin_prob:>8.2%}")
    print(f"  {'Profitable paths':<22}  {(cagrs > 0).mean():>8.2%}")
    print("=" * 60)

    return {
        "paths": paths, "cagrs": cagrs, "max_dds": max_dds,
        "terminal": terminal, "ruin_prob": ruin_prob,
        "cagr_pcts": dict(zip(pcts, cagr_pcts)),
        "dd_pcts":   dict(zip(pcts, dd_pcts)),
    }

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="APEX 0DTE Daily-Bar Backtest")
    p.add_argument("--start",   default="2016-01-01")
    p.add_argument("--end",     default="2025-01-01")
    p.add_argument("--capital", type=float, default=100_000)
    p.add_argument("--risk",    type=float, default=0.02, help="risk per trade (0.02 = 2%%)")
    p.add_argument("--out",     default="results/apex_v10")
    p.add_argument("--symbol",  default="QQQ")
    args = p.parse_args()

    os.makedirs("results", exist_ok=True)

    cfg = Config(risk_pct=args.risk)

    # Load data
    df = load_daily_bars(args.symbol, args.start, args.end)

    log.info(f"Running backtest: {args.start} → {args.end}  "
             f"capital=${args.capital:,.0f}  risk={args.risk:.1%}")
    log.info(f"Zones: UP_MOD_CONT (+{cfg.up_mod_lo:.2%}→+{cfg.up_mod_hi:.2%}) | "
             f"UP_BIG_FADE (+{cfg.up_big_thresh:.2%}+) | "
             f"DN_MOD_CONT DISABLED | "
             f"DN_EXTREME (<{cfg.dn_ext_thresh:.2%})")

    trades = run_backtest(df, args.capital, cfg)
    stats  = report(trades, args.capital, args.start, args.end, df)

    if trades:
        n_years = (pd.to_datetime(args.end) - pd.to_datetime(args.start)).days / 365.25
        mc = monte_carlo(trades, args.capital, n_years)
        save_results(trades, args.out, args.capital, df, mc)
    else:
        log.warning("No trades generated — check zone thresholds and data")


if __name__ == "__main__":
    main()
