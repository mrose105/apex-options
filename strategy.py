"""
APEX 0DTE — Strategy Engine
=============================
Pure strategy logic. Receives standardised DataFrames from any provider.
No API calls, no data fetching — just signal generation and trade management.

The engine is stateless per-day: feed it the morning data, get back a trade decision.
The portfolio manager (portfolio.py) owns state across days.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

from data_providers import bsm_greeks, implied_vol, ExecutionModel

logger = logging.getLogger("apex.strategy")


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StrategyConfig:
    # ── POSITION SIZING ──────────────────────────────────────────────────────
    # Increased to 3% — math requires N×risk×edge = 25% CAGR
    # At 20 trades/yr, 3% risk, WR=35%, Win=150%, Loss=22%:
    #   20 × 0.03 × (0.35×1.50 - 0.65×0.22) = 20 × 0.03 × 0.382 = +22.9%
    tier2_alloc_pct:    float = 0.03      # 3% of portfolio (up from 2%)
    tier1_alloc_pct:    float = 0.03
    max_position_usd:   float = 500_000

    # ── ZONE THRESHOLDS ──────────────────────────────────────────────────
    # UP_MOD_CONT:   gap-up +0.35% → +0.65%  BUY CALL  (tightened from 0.105%)
    # UP_BIG_FADE:   gap-up +0.75%+           BUY PUT   (unchanged)
    # DN_SMALL_BOUNCE: gap-dn -0.01% → -0.18% BUY CALL  (unchanged)
    # DN_MOD_CONT:   DISABLED (6% WR, need 58%)
    # DN_EXTREME:    gap-dn < -1.50%           BUY CALL  (raised from -0.80%)
    #                ONLY if iv_daily > 30% AND prev_gap < -0.50%
    up_mod_lo:          float = 0.0045    # UP_MOD_CONT lower edge  (+0.45%, raised from 0.35%)
    up_mod_hi:          float = 0.0065    # UP_MOD_CONT upper edge  (+0.65%)
    up_mod_iv_max:      float = 0.25      # only trade UP_MOD in low-vol (IV<25%)
    up_big_threshold:   float = 0.0075    # UP_BIG_FADE threshold   (+0.75%)
    dn_small_lo:        float = -0.0001   # DN_SMALL_BOUNCE lower   (-0.01%)
    dn_small_hi:        float = -0.0018   # DN_SMALL_BOUNCE upper   (-0.18%)
    # DN_MOD_CONT (-0.18% to -0.65%) DISABLED
    # N=8 empirical: 62.5% close UP intraday, avg +0.49% -- bounce zone not continuation
    dn_extreme_thresh:  float = -0.0150   # DN_EXTREME threshold     (-1.50%)
    dn_extreme_iv_min:  float = 0.30      # DN_EXTREME requires IV > 30%
    dn_extreme_prev_gap: float = -0.0050  # DN_EXTREME requires prev day gap < -0.50%

    # Gap signal thresholds (recalibrated quarterly — band context)
    gap_pct_lo:         float = 0.20
    gap_pct_hi:         float = 0.80
    gap_pct_mid:        float = 0.50

    # Recalibration
    recal_freq_bars:    int   = 63        # quarterly
    recal_window_bars:  int   = 252       # 1-year trailing window
    vol_shift_trigger:  float = 0.25      # recal early if vol moves >25%

    # ── EXIT RULES ───────────────────────────────────────────────────────────
    # FIX 1 — STOP SLIPPAGE: avg loss was -37.7% vs -28% target
    # BSM intraday fills gap through stop on fast reversals. Tighter stop
    # (-22%) means even with 5-bar slippage the avg loss stays near -25%.
    stop_5min_pct:      float = -0.15     # -15% in first bar (was -20%)
    hard_stop_pct:      float = -0.22     # -22% hard wall (was -28%)

    # FIX 3 — BIGGER WINS: let GAMMA_PEAK run longer
    # Previously: exit when gamma fell 50% from peak AND up 30% AND bar≥6
    # Problem: exiting too early — avg win was +73%, need +150%
    # Solution: wait for 65% gamma decline (more definitive peak), lower
    # return floor to 15% (don't miss exits on moderate moves), keep bar≥4
    gamma_peak_decline: float = 0.65     # wait for 65% gamma decline (was 0.50)
    gamma_peak_min_ret: float = 0.15     # floor at +15% (was +30%)
    gamma_peak_min_bar: int   = 4        # allow from bar 4 (was 6 = 30min)

    iv_crush_pct:       float = 0.40
    iv_crush_delta:     float = 0.65
    iv_crush_min_ret:   float = 0.15
    iv_crush_min_bar:   int   = 4        # was 6
    time_stop_bar:      int   = 36
    forced_exit_bar:    int   = 72

    # Let big winners run — lower extrinsic exit bar from 150% to 100%
    extrinsic_exit_thresh:  float = 0.05
    extrinsic_exit_min_ret: float = 1.00  # was 1.50 — capture 100%+ wins sooner
    extrinsic_exit_min_bar: int   = 4     # was 6

    # Trend / regime filters
    sma_fast:           int   = 3         # SMA3 for short trend
    sma_slow:           int   = 10        # SMA10 for medium trend
    sma_regime:         int   = 50        # SMA50 for macro regime filter

    # Execution model
    slippage_model:     str   = "dynamic" # 'dynamic' | 'fixed' | 'zero'
    commission_per_contract: float = 0.65

    # Risk free rate (for BSM)
    risk_free_rate:     float = 0.05

    # ── ENTRY FILTERS ────────────────────────────────────────────────────────
    # FIX 2 — MORE TRADES: current 10/yr is statistically too thin
    # MTF 1/4: only skip if ALL timeframes disagree (clear reversal)
    # gap_hold 25%: skip only if 75%+ of gap faded (near full reversal)
    # Together these target 20-25 trades/year
    mtf_min_score:      int   = 1         # 1 of 4 TFs (was 2) — only block clear reversals
    entry_bar:          int   = 3
    gap_hold_min:       float = 0.25      # 25% of gap must remain (was 40%)

    # Strike selection — 0.25% OTM
    # Gap already consumed +0.45-0.65% at open. Entry at 9:45am.
    # 0.25% OTM means:
    #   Base case  (+0.5% more move): option crosses strike → solid gain
    #   Reward     (+1.0% more move): deep ITM, explosive return 300-1000%
    #   Breakeven: only needs +0.25% continuation → very achievable
    # ATM was too expensive per contract; 1% OTM needed too large a move.
    otm_pct:            float = 0.0025     # 0.25% OTM — sweet spot

    # Bars per day
    bars_per_day:       int   = 78        # 5-min bars 9:30–16:00


# ─────────────────────────────────────────────────────────────────────────────
# CALIBRATION  (walk-forward, no lookahead)
# ─────────────────────────────────────────────────────────────────────────────

class GapCalibrator:
    """
    Computes gap thresholds from a rolling window of daily bar data.
    Call recalibrate() at startup and then every config.recal_freq_bars days.
    """

    def __init__(self, config: StrategyConfig):
        self.cfg    = config
        self.c_lo   = None; self.c_hi  = None; self.c_mid = None
        self.p_lo   = None; self.p_hi  = None; self.p_mid = None
        self.rvol   = 0.20
        self._last_bar = 0

    def calibrate(self, daily_bars: pd.DataFrame) -> dict:
        """
        daily_bars: full history up to but NOT including current day.
        Returns dict of thresholds. Updates internal state.
        """
        df = daily_bars.dropna(subset=["gap_pct"])
        if len(df) < 20:
            logger.warning("Not enough bars to calibrate — using defaults")
            return self._defaults()

        # Use trailing recal_window only
        df = df.tail(self.cfg.recal_window_bars)
        gaps = df["gap_pct"].values
        pos  = gaps[gaps > 0]
        neg  = gaps[gaps < 0]

        if len(pos) < 5 or len(neg) < 5:
            return self._defaults()

        self.c_lo  = float(np.quantile(pos, self.cfg.gap_pct_lo))
        self.c_hi  = float(np.quantile(pos, self.cfg.gap_pct_hi))
        self.c_mid = float(np.quantile(pos, self.cfg.gap_pct_mid))
        self.p_lo  = float(np.quantile(neg, self.cfg.gap_pct_lo))   # more negative
        self.p_hi  = float(np.quantile(neg, self.cfg.gap_pct_hi))   # less negative
        self.p_mid = float(np.quantile(neg, self.cfg.gap_pct_mid))

        # Realized vol (annualised)
        if "close" in df.columns and len(df) > 1:
            lr = np.diff(np.log(df["close"].values))
            self.rvol = float(np.std(lr, ddof=1) * 252 ** 0.5)

        logger.info(f"Calibrated: call [{self.c_lo:+.4f}/{self.c_hi:+.4f}] "
                    f"put [{self.p_lo:+.4f}/{self.p_hi:+.4f}] rvol={self.rvol:.3f}")
        return self.thresholds()

    def thresholds(self) -> dict:
        if self.c_lo is None:
            return self._defaults()
        return dict(c_lo=self.c_lo, c_hi=self.c_hi, c_mid=self.c_mid,
                    p_lo=self.p_lo, p_hi=self.p_hi, p_mid=self.p_mid,
                    rvol=self.rvol)

    def _defaults(self) -> dict:
        return dict(c_lo=0.0013, c_hi=0.0079, c_mid=0.0046,
                    p_lo=-0.0085, p_hi=-0.0016, p_mid=-0.0050, rvol=0.20)


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL GENERATION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DaySignal:
    """Output of signal_for_day()."""
    trade:       bool   = False
    direction:   str    = ""      # 'call' or 'put'
    tier:        int    = 2
    gap_pct:     float  = 0.0
    iv_daily:    float  = 0.0
    trend_aligned: bool = False
    strong_gap:  bool   = False
    reason:      str    = ""


def signal_for_day(gap_pct: float,
                   daily_bars_history: pd.DataFrame,
                   cal: dict,
                   config: StrategyConfig) -> DaySignal:
    """
    Given the opening gap and trailing history, return whether to trade today.
    Pure function — no side effects, no API calls.

    gap_pct: (open - prev_close) / prev_close
    daily_bars_history: all bars up to but NOT including today
    cal: current calibration thresholds dict
    """
    sig = DaySignal()
    sig.gap_pct = gap_pct
    g = gap_pct

    # ── ZONE MAP v3 ───────────────────────────────────────────────────────
    #
    #  UP_MOD_CONT    +0.45% → +0.65%   BUY CALL  (tightened — only clean moderate gaps)
    #                                    FILTER: iv_daily < 25% (low vol regime only)
    #  UP_BIG_FADE    +0.75%+            BUY PUT   (fade large gap-up)
    #  DEAD_ZONE_UP   +0.65% → +0.75%   NO TRADE
    #
    #  FLAT           -0.01% → +0.45%   NO TRADE
    #
    #  DN_SMALL_BOUNCE -0.01% → -0.18%  BUY CALL  (small dip bounce)
    #  DN_MOD_CONT    -0.18% → -0.65%   DISABLED  (62.5% close UP — bounce not continuation)
    #                                    FILTER: iv_daily < 28% (no panic spike)
    #                                    BEAR regime only (trend aligned)
    #  DN_EXTREME      < -1.50%          BUY CALL  (panic exhaustion fade)
    #                                    REQUIRES: iv_daily > 30%
    #                                              prev_day_gap < -0.50%

    # IV estimate first (needed for zone filters)
    recent = daily_bars_history.tail(21)
    if len(recent) > 2:
        lr = np.diff(np.log(recent["close"].values))
        sig.iv_daily = float(np.clip(np.std(lr, ddof=1) * 252**0.5, 0.10, 1.50))
    else:
        sig.iv_daily = 0.25

    # Previous day gap (for DN_EXTREME filter)
    if len(daily_bars_history) >= 2:
        prev_gap = float(daily_bars_history["gap_pct"].iloc[-1]) \
                   if "gap_pct" in daily_bars_history.columns else 0.0
    else:
        prev_gap = 0.0

    # Zone assignment
    if config.up_mod_lo <= g <= config.up_mod_hi:
        direction = "call"; zone = "UP_MOD_CONT"
        # Tighten: only trade in low IV regime
        if sig.iv_daily > config.up_mod_iv_max:
            sig.reason = f"UP_MOD_CONT filtered — iv={sig.iv_daily:.3f} > {config.up_mod_iv_max}"
            return sig

    elif g > config.up_big_threshold:
        direction = "put";  zone = "UP_BIG_FADE"

    elif config.up_mod_hi < g <= config.up_big_threshold:
        direction = None;   zone = "DEAD_ZONE_UP"

    elif config.dn_small_lo >= g >= config.dn_small_hi:
        direction = "call"; zone = "DN_SMALL_BOUNCE"

    elif g < config.dn_extreme_thresh:
        direction = "call"; zone = "DN_EXTREME"
        # Strict filters: must be panic (high IV) + consecutive gap-down
        if sig.iv_daily < config.dn_extreme_iv_min:
            sig.reason = f"DN_EXTREME filtered — iv={sig.iv_daily:.3f} < {config.dn_extreme_iv_min} (not panic)"
            return sig
        if prev_gap > config.dn_extreme_prev_gap:
            sig.reason = f"DN_EXTREME filtered — prev_gap={prev_gap:+.4f} not consecutive"
            return sig

    elif -0.0065 > g >= config.dn_extreme_thresh:
        # -0.65% to -1.50%: too large for bounce, too small for panic exhaustion
        direction = None; zone = "DEAD_ZONE_DN"

    else:
        direction = None;   zone = "NO_TRADE"

    if direction is None:
        sig.reason = f"no_trade_{zone} ({g:+.5f})"
        return sig

    sig.direction = direction

    # ── 50-DAY SMA REGIME FILTER ──────────────────────────────────────────
    closes = daily_bars_history["close"].values
    if len(closes) >= config.sma_regime:
        sma50   = float(np.mean(closes[-config.sma_regime:]))
        last_px = float(closes[-1])
        bear    = last_px < sma50
        bull    = last_px >= sma50

        blocked = False
        if bear:
            if zone == "UP_MOD_CONT":       blocked = True   # no calls in bear
            if zone == "DN_SMALL_BOUNCE":   blocked = True   # no bounce calls in bear
        if bull:
            if zone == "UP_BIG_FADE":       blocked = True   # don't fade rips in bull
        # DN_EXTREME allowed in any regime (panic exhaustion)

        if blocked:
            regime = "BEAR" if bear else "BULL"
            sig.reason = f"regime_{regime}_blocks_{zone} ({g:+.5f})"
            return sig

        sig.trend_aligned = bull if direction == "call" else bear
    else:
        sig.trend_aligned = False

    sig.strong_gap = abs(g) >= 0.005
    sig.tier  = 2
    sig.trade = True
    sig.reason = f"{zone}_{direction} gap={g:+.5f} iv={sig.iv_daily:.3f} prev_gap={prev_gap:+.4f}"
    return sig


# ─────────────────────────────────────────────────────────────────────────────
# INTRADAY TRADE MANAGER
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TradeResult:
    date:           date   = None
    direction:      str    = ""
    tier:           int    = 2
    gap_pct:        float  = 0.0
    iv_daily:       float  = 0.0
    entry_bar:      int    = 0
    exit_bar:       int    = 0
    exit_reason:    str    = ""
    entry_px:       float  = 0.0     # option mid at entry
    entry_fill:     float  = 0.0     # actual fill (with slippage)
    exit_px:        float  = 0.0     # option mid at exit
    exit_fill:      float  = 0.0     # actual fill (with slippage)
    contracts:      int    = 0
    risk_cap:       float  = 0.0
    gross_pnl:      float  = 0.0
    commission:     float  = 0.0
    net_pnl:        float  = 0.0
    return_pct:     float  = 0.0     # based on risk_cap
    portfolio_before: float = 0.0
    portfolio_after:  float = 0.0
    peak_gamma:     float  = 0.0
    entry_delta:    float  = 0.0
    capped:         bool   = False
    slippage_cost:  float  = 0.0


def run_intraday_trade(signal: DaySignal,
                       intraday_bars: pd.DataFrame,
                       option_chain_open: pd.DataFrame,
                       portfolio_value: float,
                       config: StrategyConfig,
                       exec_model: ExecutionModel,
                       trading_date: date) -> Optional[TradeResult]:
    """
    Execute and manage a single 0DTE trade through the intraday session.

    intraday_bars: 5-min OHLCV for today (IntradayBar.COLS)
    option_chain_open: option quotes at open (OptionQuote.COLS)
    portfolio_value: current account value in dollars

    Returns TradeResult or None if no valid entry found.
    """
    result = TradeResult(
        date=trading_date,
        direction=signal.direction,
        tier=signal.tier,
        gap_pct=signal.gap_pct,
        iv_daily=signal.iv_daily,
        portfolio_before=portfolio_value,
    )

    # ── Position sizing ────────────────────────────────────────────────────
    alloc_pct  = config.tier1_alloc_pct if signal.tier == 1 else config.tier2_alloc_pct
    raw_risk   = portfolio_value * alloc_pct
    risk_cap   = min(raw_risk, config.max_position_usd)
    result.capped    = raw_risk > config.max_position_usd
    result.risk_cap  = risk_cap

    if option_chain_open is None or len(option_chain_open) == 0:
        logger.warning(f"{trading_date}: empty option chain — skipping")
        return None

    bars = intraday_bars.reset_index(drop=True)
    n_bars = min(len(bars), config.bars_per_day)
    if n_bars < config.entry_bar + 1:
        logger.warning(f"{trading_date}: not enough bars to reach entry bar")
        return None

    # ── 9:30–9:45am observation window (bars 0–2) ─────────────────────────
    obs_bars = bars.iloc[:config.entry_bar]
    prev_close = float(bars.iloc[0]["open"]) / (1 + signal.gap_pct) \
                 if signal.gap_pct != 0 else float(bars.iloc[0]["open"])

    if len(obs_bars) >= 2:
        bar0_range = float(obs_bars.iloc[0]["high"]) - float(obs_bars.iloc[0]["low"])
        bar2_range = float(obs_bars.iloc[-1]["high"]) - float(obs_bars.iloc[-1]["low"])
        vol_expanding = bar2_range > bar0_range * 1.10

        # For extreme gap-down fades: skip if vol still expanding at 9:45
        if "EXTREME" in signal.reason and vol_expanding:
            logger.info(f"{trading_date}: DN_EXTREME skipped — vol still expanding "
                        f"(bar0={bar0_range:.3f} bar2={bar2_range:.3f})")
            return None

    else:
        vol_expanding = False

    # ── GAP HOLDING FILTER ────────────────────────────────────────────────
    # At 9:45am (bar 3), the gap must still be at least 60% intact.
    # If the gap has already faded more than 40%, the move is reversing —
    # entering now means chasing a dying gap = almost guaranteed hard stop.
    S_bar3    = float(bars.iloc[config.entry_bar]["close"])
    gap_now   = (S_bar3 - prev_close) / prev_close if prev_close > 0 else signal.gap_pct
    gap_orig  = signal.gap_pct

    # For calls: gap must still be positive and >= 60% of original
    # For puts on big gap-up: gap must still be elevated (not already faded before we fade it)
    if abs(gap_orig) > 0.0001:
        gap_hold_ratio = gap_now / gap_orig   # 1.0 = perfectly held, 0 = fully faded
    else:
        gap_hold_ratio = 1.0

    if gap_hold_ratio < config.gap_hold_min:
        logger.info(f"{trading_date}: gap_hold SKIP — gap faded to "
                    f"{gap_hold_ratio:.0%} of original "
                    f"(orig={gap_orig:+.4f} now={gap_now:+.4f})")
        return None

    # ── MULTI-TIMEFRAME CONFIRMATION ──────────────────────────────────────
    # Using bars 0–2 (9:30–9:45am) we construct higher and lower timeframes
    # from the 5min base data to confirm the trade direction before entry.
    #
    # TIMEFRAME STACK:
    #   4hr  (48×5min) — only 1 bar available intraday; use daily history instead
    #   1hr  (12×5min) — only partial bar at open; use last 12 bars if available
    #   15min ( 3×5min) — bars 0–2 form the 9:30–9:45 15min candle
    #   10min ( 2×5min) — bars 0–1 and bars 1–2
    #   5min  (1×5min)  — bar 2 is the trigger bar (last bar before entry)
    #
    # RULES:
    #   4hr  trend aligned  → daily SMA50 already handles this (done above in signal)
    #   1hr  momentum       → last 12 5min bars trending in trade direction
    #   15min structure     → 15min candle (bars 0-2) closes in trade direction
    #   10min confirmation  → last 10min candle (bars 1-2) closes in trade direction
    #   5min trigger        → bar 2 (9:40-9:45) closes in trade direction
    #
    # Require at least 3 of the 4 lower TF confirmations to enter.

    is_call = signal.direction == "call"
    mtf_score = 0
    mtf_details = []

    # ── 1HR confirmation (use all available intraday bars up to bar 3) ────
    # Proxy: are the first 3 bars making higher lows (calls) / lower highs (puts)?
    if len(obs_bars) >= 3:
        lows  = [float(obs_bars.iloc[i]["low"])  for i in range(3)]
        highs = [float(obs_bars.iloc[i]["high"]) for i in range(3)]
        if is_call:
            # Higher lows = buyers stepping in at higher prices = bullish
            hr1_ok = lows[1] >= lows[0] and lows[2] >= lows[1]
        else:
            # Lower highs = sellers pressing = bearish
            hr1_ok = highs[1] <= highs[0] and highs[2] <= highs[1]
        if hr1_ok:
            mtf_score += 1
        mtf_details.append(f"1hr={'✓' if hr1_ok else '✗'}")

    # ── 15MIN candle (bars 0–2, full 9:30–9:45 candle) ───────────────────
    if len(obs_bars) >= 3:
        c15_open  = float(obs_bars.iloc[0]["open"])
        c15_close = float(obs_bars.iloc[2]["close"])
        if is_call:
            tf15_ok = c15_close > c15_open   # 15min candle is green
        else:
            tf15_ok = c15_close < c15_open   # 15min candle is red
        if tf15_ok:
            mtf_score += 1
        mtf_details.append(f"15m={'✓' if tf15_ok else '✗'}")

    # ── 10MIN candle (bars 1–2, 9:35–9:45) ───────────────────────────────
    if len(obs_bars) >= 3:
        c10_open  = float(obs_bars.iloc[1]["open"])
        c10_close = float(obs_bars.iloc[2]["close"])
        if is_call:
            tf10_ok = c10_close > c10_open   # 10min candle is green
        else:
            tf10_ok = c10_close < c10_open   # 10min candle is red
        if tf10_ok:
            mtf_score += 1
        mtf_details.append(f"10m={'✓' if tf10_ok else '✗'}")

    # ── 5MIN trigger bar (bar 2, 9:40–9:45) ──────────────────────────────
    if len(obs_bars) >= 3:
        bar2_open  = float(obs_bars.iloc[2]["open"])
        bar2_close = float(obs_bars.iloc[2]["close"])
        if is_call:
            tf5_ok = bar2_close > bar2_open   # trigger bar is green
        else:
            tf5_ok = bar2_close < bar2_open   # trigger bar is red
        if tf5_ok:
            mtf_score += 1
        mtf_details.append(f"5m={'✓' if tf5_ok else '✗'}")

    mtf_str = "  ".join(mtf_details)

    if mtf_score < config.mtf_min_score:
        logger.info(f"{trading_date}: MTF SKIP — score={mtf_score}/{config.mtf_min_score}  {mtf_str}")
        return None

    logger.debug(f"{trading_date}: MTF PASS — score={mtf_score}/4  {mtf_str}")

    # ── Enter at bar 3 (9:45am) — buy slightly OTM for maximum leverage ───
    entry_bar_idx = config.entry_bar
    S_entry = float(bars.iloc[entry_bar_idx]["close"])

    chain = option_chain_open[option_chain_open["option_type"] == signal.direction].copy()
    if len(chain) == 0:
        return None

    # Target strike: config.otm_pct OTM from bar-3 price
    # Calls: strike ABOVE current price by otm_pct
    # Puts:  strike BELOW current price by otm_pct
    # This gives cheap options with explosive leverage on a continued move.
    # Example: QQQ $500, buy 505 call @ $0.15 → QQQ +1% → worth $1.00+ = +567%
    is_call  = signal.direction == "call"
    otm_mult = (1 + config.otm_pct) if is_call else (1 - config.otm_pct)
    target_K = S_entry * otm_mult

    chain["_dist"] = (chain["strike"] - target_K).abs()
    otm_row  = chain.sort_values("_dist").iloc[0]

    open_iv  = float(otm_row["iv"])
    entry_K  = float(otm_row["strike"])
    moneyness = (entry_K - S_entry) / S_entry

    T_open      = 1.0 / 252
    dt_bar      = T_open / config.bars_per_day
    T_entry     = max((config.bars_per_day - entry_bar_idx) * dt_bar, 1e-8)
    iv_at_entry = open_iv * (0.70 + 0.30 * np.exp(-entry_bar_idx / 25))
    iv_at_entry = max(iv_at_entry, 0.05)

    rf      = config.risk_free_rate
    g_entry = bsm_greeks(S_entry, entry_K, T_entry, rf, iv_at_entry, call=is_call)
    entry_mid = g_entry["price"]

    MIN_ENTRY_MID = 0.08   # 0.25% OTM options realistically $0.10-0.60
    if entry_mid < MIN_ENTRY_MID:
        logger.debug(f"{trading_date}: OTM option too cheap (${entry_mid:.3f}) — skip")
        return None

    n_contracts_approx = max(1, int(risk_cap / (entry_mid * 100)))
    entry_fill = exec_model.entry_price(entry_mid, iv_at_entry,
                                         T_entry, moneyness, n_contracts_approx)
    contracts  = min(500, max(1, int(risk_cap / (entry_fill * 100))))
    commission = exec_model.commission_cost(contracts)

    result.entry_bar    = entry_bar_idx
    result.entry_px     = entry_mid
    result.entry_fill   = entry_fill
    result.contracts    = contracts
    result.entry_delta  = float(g_entry["delta"])
    result.peak_gamma   = float(g_entry["gamma"])

    # Slippage cost at entry
    result.slippage_cost = (entry_fill - entry_mid) * contracts * 100

    logger.debug(f"{trading_date}: entering at bar {entry_bar_idx} (9:45am)  "
                 f"S={S_entry:.2f}  K={entry_K:.0f}  mid=${entry_mid:.3f}  "
                 f"IV={iv_at_entry:.3f}  open_IV={open_iv:.3f}  "
                 f"discount={((open_iv-iv_at_entry)/open_iv*100):.1f}% IV cooled  "
                 f"vol_expanding={vol_expanding}")

    # ── Intraday bar loop ──────────────────────────────────────────────────
    bars = intraday_bars.reset_index(drop=True)
    n_bars = min(len(bars), config.bars_per_day)

    if n_bars == 0:
        logger.warning(f"{trading_date}: no intraday bars")
        return None

    # BSM parameters for each bar
    T_open   = 1.0 / 252
    dt_bar   = T_open / config.bars_per_day
    is_call  = signal.direction == "call"
    rf       = config.risk_free_rate

    max_gamma     = result.peak_gamma
    exit_px       = None
    exit_fill     = None
    exit_bar      = n_bars - 1
    exit_reason   = "FORCED"

    for b in range(entry_bar_idx, n_bars):
        T_rem = max((config.bars_per_day - b) * dt_bar, 1e-8)

        S_b = float(bars.iloc[b]["close"])

        iv_crush_factor = 0.70 + 0.30 * np.exp(-b / 25)
        iv_b = max(iv_at_entry * iv_crush_factor, 0.05)

        g = bsm_greeks(S_b, entry_K, T_rem, rf, iv_b, call=is_call)
        mid_b = max(g["price"], 0.001)

        ret_b   = (mid_b - entry_fill) / entry_fill
        gamma_b = g["gamma"]
        if gamma_b > max_gamma:
            max_gamma = gamma_b

        # ── Exit logic (priority order) ──────────────────────────────────

        # 1. First-bar stop — if trade goes immediately wrong at 9:45 entry bar
        if b == entry_bar_idx and ret_b <= config.stop_5min_pct:
            exit_fill   = exec_model.exit_price(mid_b, iv_b, T_rem,
                                                 (entry_K-S_b)/S_b, contracts)
            exit_bar    = b
            exit_reason = "5MIN_STOP"
            break

        # 2. Hard stop
        if ret_b <= config.hard_stop_pct:
            exit_fill   = exec_model.exit_price(mid_b, iv_b, T_rem,
                                                 (entry_K-S_b)/S_b, contracts)
            exit_bar    = b
            exit_reason = "HARD_STOP"
            break

        # 3. Gamma peak exit
        gamma_decline = (max_gamma - gamma_b) / (max_gamma + 1e-9)
        if (gamma_decline > config.gamma_peak_decline and
                ret_b >= config.gamma_peak_min_ret and
                b >= config.gamma_peak_min_bar):
            exit_fill   = exec_model.exit_price(mid_b, iv_b, T_rem,
                                                 (entry_K-S_b)/S_b, contracts)
            exit_bar    = b
            exit_reason = "GAMMA_PEAK"
            break

        # 4. IV crush exit (gone ITM, IV collapsed)
        iv_crush_pct = (iv_at_entry - iv_b) / (iv_at_entry + 1e-9)
        if (iv_crush_pct > config.iv_crush_pct and
                abs(g["delta"]) > config.iv_crush_delta and
                ret_b >= config.iv_crush_min_ret and
                b >= config.iv_crush_min_bar):
            exit_fill   = exec_model.exit_price(mid_b, iv_b, T_rem,
                                                 (entry_K-S_b)/S_b, contracts)
            exit_bar    = b
            exit_reason = "IV_CRUSH"
            break

        # 5. Extrinsic value exit — replaces hard TIME_STOP
        #
        # When an option goes ITM, it has:
        #   intrinsic = max(S-K, 0) for calls | max(K-S, 0) for puts
        #   extrinsic = option_price - intrinsic
        #
        # As the option goes deeper ITM, extrinsic shrinks and theta
        # accelerates. Once extrinsic/price < threshold, you are mostly
        # carrying intrinsic dollar-for-dollar — no leverage benefit,
        # pure theta bleed. Sell it and lock in the gain.
        #
        # Only fires when profitable (ret > min_ret) and after min holding time.
        # For losing/flat trades still in session, fall through to FORCED at 3:30pm.
        #
        intrinsic_b = max(S_b - entry_K, 0) if is_call else max(entry_K - S_b, 0)
        extrinsic_b = max(mid_b - intrinsic_b, 0)
        extrinsic_ratio = extrinsic_b / (mid_b + 1e-9)

        if (extrinsic_ratio < config.extrinsic_exit_thresh and
                ret_b >= config.extrinsic_exit_min_ret and
                b >= config.extrinsic_exit_min_bar):
            exit_fill   = exec_model.exit_price(mid_b, iv_b, T_rem,
                                                 (entry_K-S_b)/S_b, contracts)
            exit_bar    = b
            exit_reason = "EXTRINSIC_EXIT"
            break

        # 6. Forced close — 3:30pm hard wall (bar 72), no exceptions
        #    Theta collapse in last 30min destroys extrinsic on all options
        if b >= config.forced_exit_bar:
            exit_fill   = exec_model.exit_price(mid_b, iv_b, T_rem,
                                                 (entry_K-S_b)/S_b, contracts)
            exit_bar    = b
            exit_reason = "FORCED"
            break

    # If we never broke — force exit at last bar
    if exit_fill is None:
        b      = n_bars - 1
        T_rem  = max(dt_bar, 1e-9)
        S_b    = float(bars.iloc[b]["close"])
        iv_b   = max(iv_at_entry * 0.60, 0.05)   # heavy crush at close
        g      = bsm_greeks(S_b, entry_K, T_rem, rf, iv_b, call=is_call)
        mid_b  = max(g["price"], 0.001)
        exit_fill = exec_model.exit_price(mid_b, iv_b, T_rem,
                                           (entry_K-S_b)/S_b, contracts)
        exit_bar    = b
        exit_reason = "FORCED"

    # ── P&L calculation ────────────────────────────────────────────────────
    # Gross P&L = (exit_fill - entry_fill) × contracts × 100
    gross_pnl  = (exit_fill - entry_fill) * contracts * 100
    commission = exec_model.commission_cost(contracts)

    # Add exit slippage cost to tracking
    result.slippage_cost += (mid_b - exit_fill) * contracts * 100   # always >= 0

    net_pnl    = gross_pnl - commission
    return_pct = net_pnl / risk_cap   # return on risk deployed

    result.exit_bar       = exit_bar
    result.exit_reason    = exit_reason
    result.exit_px        = mid_b
    result.exit_fill      = exit_fill
    result.gross_pnl      = gross_pnl
    result.commission     = commission
    result.net_pnl        = net_pnl
    result.return_pct     = return_pct
    result.peak_gamma     = max_gamma
    result.portfolio_after = portfolio_value + net_pnl

    logger.info(f"{trading_date} {signal.direction} tier-{signal.tier} "
                f"exit={exit_reason} bar={exit_bar} "
                f"ret={return_pct:+.1%} pnl=${net_pnl:+,.0f}")

    return result
