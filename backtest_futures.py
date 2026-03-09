"""
APEX 0DTE — NQ Futures Gap Backtest
=====================================
Replaces the synthetic QQQ gap signal with real (or synthetic) NQ overnight gap.
Everything else — option pricing, exit logic, position sizing, portfolio management —
is identical to the base engine.

HOW THE SIGNAL CHANGES
------------------------
OLD: gap_pct = (QQQ_open - QQQ_prev_close) / QQQ_prev_close   ← RTH only, noisy
NEW: gap_pct = (NQ_open   - NQ_prev_close) / NQ_prev_close    ← 23hr futures, cleaner

Additionally, signals are filtered by:
  - Gap must be in the 20th–80th percentile of same-sign gaps (moderate band)
  - Gap Z-score must be 0.3–3.5σ (reject noise and news events)
  - NQ/QQQ divergence must be <0.15% (reject dislocations)
  - Quality score computed for optional tier enhancement

QUICK START
-----------
With real data (requires internet):
    python backtest_futures.py --start 2018-01-01 --end 2026-01-01

With synthetic NQ data (no internet, for testing):
    python backtest_futures.py --synthetic --start 2018-01-01 --end 2026-01-01

With local CSV files:
    python backtest_futures.py --nq-csv data/NQ_daily.csv --qqq-csv data/QQQ_daily.csv

GETTING FREE NQ DATA
---------------------
Option A: yfinance (easiest)
    pip install yfinance
    # The script downloads automatically when --start/--end given without --synthetic

Option B: Barchart.com free download
    1. Go to: https://www.barchart.com/futures/quotes/NQ*0/historical-download
    2. Download CSV → pass with --nq-csv

Option C: Yahoo Finance web
    https://finance.yahoo.com/quote/%2FNQ%3DF/history/
    Download → pass with --nq-csv
"""

from __future__ import annotations
import argparse
import logging
import sys
import os
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter

sys.path.insert(0, os.path.dirname(__file__))

from futures_gap import (NQGapProvider, NQGapSignal,
                          build_synthetic_nq, load_nq_from_yfinance,
                          load_nq_from_csv, analyse_gap_quality, plot_gap_analysis)
from data_providers  import SyntheticProvider, ExecutionModel, bsm_greeks
from strategy        import StrategyConfig, signal_for_day, run_intraday_trade, DaySignal
from portfolio       import PortfolioManager, RiskLimits

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("apex.futures_bt")


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL ADAPTER  (NQGapSignal → strategy.DaySignal)
# ─────────────────────────────────────────────────────────────────────────────

def nq_to_day_signal(nq_sig: NQGapSignal,
                      qqq_daily_history: pd.DataFrame,
                      config: StrategyConfig) -> DaySignal:
    """
    Convert an NQGapSignal to the DaySignal that run_intraday_trade() expects.

    Key adaptations:
    - direction and tier come from NQ signal
    - IV estimate comes from QQQ trailing history (needed for options pricing)
    - Tier-1 gets an additional quality_score boost from NQ signal
    """
    sig = DaySignal()
    sig.trade     = nq_sig.tradeable
    sig.direction = nq_sig.direction
    sig.gap_pct   = nq_sig.gap_pct   # NQ gap — used only for logging; QQQ strike set separately

    if not nq_sig.tradeable:
        sig.reason = "nq_gap_filtered"
        return sig

    # IV estimate from QQQ trailing HV (same as before)
    recent = qqq_daily_history.tail(21)
    if len(recent) > 2:
        lr = np.diff(np.log(recent["close"].values))
        sig.iv_daily = float(np.clip(np.std(lr, ddof=1) * 252**0.5, 0.10, 1.50))
    else:
        sig.iv_daily = nq_sig.vol_regime if nq_sig.vol_regime > 0 else 0.25

    # Trend alignment from QQQ SMAs
    closes = qqq_daily_history["close"].values
    if len(closes) >= config.sma_slow:
        sma_fast = float(np.mean(closes[-config.sma_fast:]))
        sma_slow = float(np.mean(closes[-config.sma_slow:]))
        sig.trend_aligned = (sma_fast > sma_slow) if nq_sig.direction == "call" \
                             else (sma_fast < sma_slow)
    else:
        sig.trend_aligned = False

    sig.strong_gap = nq_sig.is_strong_gap

    # Tier: NQ quality score boosts to tier-1 threshold
    # Tier-1 requires: strong gap + trend-aligned + IV>20% + NQ quality > 0.55
    tier1 = (sig.trend_aligned and
              sig.strong_gap and
              sig.iv_daily > config.tier1_iv_min and
              nq_sig.quality_score > 0.55)
    sig.tier   = 1 if tier1 else 2
    sig.reason = (f"NQ_tier{sig.tier}_{sig.direction} "
                  f"gap={nq_sig.gap_pct:+.4f} z={nq_sig.gap_z_score:+.2f} "
                  f"quality={nq_sig.quality_score:.2f}")
    return sig


# ─────────────────────────────────────────────────────────────────────────────
# BACKTEST CORE
# ─────────────────────────────────────────────────────────────────────────────

def run_futures_backtest(nq_df: pd.DataFrame,
                          qqq_df: pd.DataFrame,
                          config: StrategyConfig,
                          limits: RiskLimits,
                          initial_capital: float,
                          start: date,
                          end: date,
                          train_end: date,
                          out_prefix: str = "apex_nq") -> PortfolioManager:
    """
    Full walk-forward backtest using NQ gap as signal and QQQ options as instrument.

    train_end: last date of training period (calibration uses only data up to this date)
    """
    logger.info(f"Building NQ gap provider...")
    gap_prov = NQGapProvider(nq_df, qqq_df)

    logger.info(f"Computing gap signals {start} → {end}...")
    signals_df = gap_prov.get_all_signals(start, end,
                                           min_train_days=252)
    if signals_df.empty:
        logger.error("No signals generated. Check data coverage.")
        return None

    stats = analyse_gap_quality(signals_df)
    logger.info(f"Gap analysis: {stats['tradeable_days']}/{stats['total_days']} "
                f"tradeable ({stats['trade_pct']:.1%})  "
                f"calls={stats['call_pct']:.0%}  puts={stats['put_pct']:.0%}")

    # Build QQQ provider for option chain + intraday bars
    qqq_provider = SyntheticProvider()   # ← swap for TradierProvider when you have real options data

    pm = PortfolioManager(initial_capital=initial_capital,
                           config=config, risk_limits=limits)

    # Initial calibration
    qqq_hist_full = qqq_df.copy()
    qqq_hist_full["date"] = pd.to_datetime(qqq_hist_full["date"])
    qqq_train = qqq_hist_full[qqq_hist_full["date"].dt.date <= train_end].copy()
    qqq_train["gap_pct"] = (qqq_train["open"] - qqq_train["close"].shift(1)) / qqq_train["close"].shift(1)
    pm.calibrator.calibrate(qqq_train)
    pm._cal = pm.calibrator.thresholds()

    logger.info(f"Running {len(signals_df)} signal days in OOS period")

    exec_model = ExecutionModel(spread_model=config.slippage_model,
                                 commission_per_contract=config.commission_per_contract)

    equity_dates = []
    equity_vals  = []

    for _, sig_row in signals_df.iterrows():
        tdate = sig_row["trade_date"].date()
        if tdate < start or tdate > end:
            continue

        # QQQ history up to yesterday (zero lookahead)
        qqq_hist = qqq_hist_full[qqq_hist_full["date"].dt.date < tdate].copy()
        qqq_hist["gap_pct"] = (qqq_hist["open"] - qqq_hist["close"].shift(1)) / \
                               qqq_hist["close"].shift(1)

        # Track equity on non-trade days too
        equity_dates.append(tdate)
        equity_vals.append(pm.portfolio)

        if pm._halted:
            break

        # Build NQGapSignal from the row
        nq_signal = NQGapSignal(
            trade_date        = tdate,
            gap_pct           = float(sig_row["gap_pct"]),
            gap_pts           = float(sig_row["gap_pts"]),
            direction         = str(sig_row["direction"]),
            nq_prev_close     = float(sig_row["nq_prev_close"]),
            nq_open           = float(sig_row["nq_open"]),
            qqq_prev_close    = float(sig_row["qqq_prev_close"]),
            qqq_open          = float(sig_row["qqq_open"]),
            qqq_gap_pct       = float(sig_row["qqq_gap_pct"]),
            nq_qqq_gap_diff   = float(sig_row["nq_qqq_gap_diff"]),
            gap_z_score       = float(sig_row["gap_z_score"]),
            vol_regime        = float(sig_row["vol_regime"]),
            vol_adjusted_gap  = float(sig_row["vol_adjusted_gap"]),
            gap_quantile      = float(sig_row["gap_quantile"]),
            in_moderate_band  = bool(sig_row["in_moderate_band"]),
            is_strong_gap     = bool(sig_row["is_strong_gap"]),
            quality_score     = float(sig_row["quality_score"]),
            tradeable         = bool(sig_row["tradeable"]),
        )

        if not nq_signal.tradeable:
            pm._bar_count += 1
            pm._maybe_recalibrate(qqq_hist)
            continue

        # Convert to DaySignal
        day_sig = nq_to_day_signal(nq_signal, qqq_hist, config)
        if not day_sig.trade:
            pm._bar_count += 1
            pm._maybe_recalibrate(qqq_hist)
            continue

        # Risk gate
        ok, reason = limits.check(pm.portfolio, pm.peak,
                                   pm.daily_start, pm._open_positions)
        if not ok:
            logger.warning(f"{tdate}: risk gate BLOCKED — {reason}")
            pm._halted = True; pm._halt_reason = reason
            break

        pm.daily_start = pm.portfolio

        # Get intraday bars and option chain
        intraday = qqq_provider.get_intraday_bars(tdate)
        chain    = qqq_provider.get_option_chain_at_open(tdate, tdate)

        # Override option chain underlying price with QQQ actual open
        if chain is not None and len(chain) > 0 and not np.isnan(nq_signal.qqq_open):
            chain = chain.copy()
            chain["underlying_price"] = nq_signal.qqq_open

        pm._open_positions += 1
        from strategy import run_intraday_trade
        result = run_intraday_trade(
            signal            = day_sig,
            intraday_bars     = intraday,
            option_chain_open = chain,
            portfolio_value   = pm.portfolio,
            config            = config,
            exec_model        = exec_model,
            trading_date      = tdate,
        )
        pm._open_positions -= 1

        if result is not None:
            pm.portfolio += result.net_pnl
            result.portfolio_after = pm.portfolio
            pm.peak = max(pm.peak, pm.portfolio)
            pm.trade_log.append(result)
            pm.equity_curve.append((tdate, pm.portfolio))
        else:
            pm.equity_curve.append((tdate, pm.portfolio))

        pm._bar_count += 1
        pm._maybe_recalibrate(qqq_hist)

        if len(pm.trade_log) % 100 == 0 and pm.trade_log:
            logger.info(f"Progress: {len(pm.trade_log)} trades  "
                        f"Portfolio: ${pm.portfolio:,.0f}")

    # Final equity point
    if equity_dates:
        pm.equity_curve.append((equity_dates[-1], pm.portfolio))

    # Generate outputs
    pm.print_summary()
    _save_outputs(pm, signals_df, nq_df, qqq_df, out_prefix)

    return pm


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def _save_outputs(pm: PortfolioManager,
                   signals_df: pd.DataFrame,
                   nq_df: pd.DataFrame,
                   qqq_df: pd.DataFrame,
                   out_prefix: str):

    tdf = pm.trade_dataframe()
    edf = pm.equity_dataframe()

    if len(tdf):
        tdf.to_csv(f"{out_prefix}_trades.csv", index=False)
        logger.info(f"Trades → {out_prefix}_trades.csv")
    if len(edf):
        edf.to_csv(f"{out_prefix}_equity.csv", index=False)
        logger.info(f"Equity → {out_prefix}_equity.csv")

    # Gap signal analysis chart
    plot_gap_analysis(signals_df, nq_df,
                       save_path=f"{out_prefix}_gap_analysis.png")

    # Main dashboard
    _plot_dashboard(pm, tdf, edf, signals_df, nq_df, qqq_df, out_prefix)


def _plot_dashboard(pm, tdf, edf, signals_df, nq_df, qqq_df, out_prefix):
    s = pm.summary()
    fig = plt.figure(figsize=(20, 12), facecolor="#0d1117")
    fig.suptitle("APEX 0DTE — NQ Futures Gap Signal  |  QQQ Options Strategy",
                 color="white", fontsize=14, fontweight="bold", y=0.98)

    RD="#ef4444"; GR="#22c55e"; BL="#3b82f6"; MU="#94a3b8"; YL="#eab308"; CY="#06b6d4"

    axes = fig.subplot_mosaic(
        [["eq","eq","eq","ret"],
         ["nq","nq","exit","tier"]],
        gridspec_kw={"hspace": 0.38, "wspace": 0.35}
    )

    def style(ax, title):
        ax.set_facecolor("#161b22")
        for sp in ax.spines.values(): sp.set_color("#2d3748")
        ax.tick_params(colors=MU, labelsize=8)
        ax.set_title(title, color="white", fontsize=9, pad=6)
        ax.grid(True, color="#2d3748", lw=0.4, alpha=0.7)

    # ── Equity curve ──────────────────────────────────────────────────────
    ax = axes["eq"]
    style(ax, f"Equity Curve (log)  |  CAGR {s['cagr']:+.1%}  Sharpe {s['sharpe']:.2f}  "
          f"MaxDD {s['max_drawdown']:.1%}  |  Signal: NQ Overnight Gap")
    if len(edf):
        dates  = pd.to_datetime(edf["date"])
        vals   = edf["portfolio"].values / 1000
        ax.semilogy(dates, vals, color=RD, lw=2.0, label="APEX (NQ signal)", zorder=3)

        # QQQ benchmark
        qqq = qqq_df.copy(); qqq["date"] = pd.to_datetime(qqq["date"])
        qqq = qqq[(qqq["date"] >= dates.min()) & (qqq["date"] <= dates.max())]
        if len(qqq) > 1:
            qqq_bench = pm.initial * qqq["close"].values / qqq["close"].iloc[0] / 1000
            ax.semilogy(qqq["date"], qqq_bench, color=BL, lw=1.2, ls="--",
                        alpha=0.8, label="QQQ B&H", zorder=2)

        start_k = vals[0]
        for mult, lbl in [(2,"2×"),(5,"5×"),(10,"10×"),(50,"50×"),(100,"100×")]:
            t = start_k * mult
            if vals[-1] >= t >= vals[0]:
                ax.axhline(t, color=GR, lw=0.4, ls=":", alpha=0.5)
                ax.text(dates.iloc[2], t*1.05, lbl, color=MU, fontsize=7)
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda x, _: f"${x:,.0f}K" if x<1000 else f"${x/1000:.1f}M"))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(fontsize=8, loc="upper left",
              facecolor="#1e2530", labelcolor="white", edgecolor="#2d3748")

    # ── Return distribution ───────────────────────────────────────────────
    ax = axes["ret"]
    style(ax, f"Option Return Dist  WR={s['win_rate']:.1%}  PF={s['profit_factor']:.2f}")
    if len(tdf):
        rets = tdf["return_pct"].values * 100
        wins = rets[rets > 0]; losses = rets[rets <= 0]
        bins = np.linspace(-100, 300, 50)
        ax.hist(losses, bins=bins, color=RD, alpha=0.7, label=f"Loss ({len(losses)})")
        ax.hist(wins,   bins=bins, color=GR, alpha=0.7, label=f"Win ({len(wins)})")
        ax.axvline(0, color=MU, lw=0.8, ls="--")
        ax.axvline(float(np.mean(rets)), color=YL, lw=1.2, ls="--",
                   label=f"μ={np.mean(rets):+.1f}%")
        ax.set_xlabel("Return on Risk Capital (%)")
        ax.legend(fontsize=7, facecolor="#1e2530", labelcolor="white", edgecolor="#2d3748")

    # ── NQ gap + vol regime ───────────────────────────────────────────────
    ax = axes["nq"]
    style(ax, "NQ Gap Z-Score  |  Grey=filtered  Green=call  Red=put")
    if len(signals_df):
        df = signals_df.copy()
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        tradeable = df[df["tradeable"]]
        non_trade = df[~df["tradeable"]]
        ax.bar(non_trade["trade_date"], non_trade["gap_z_score"],
               color=MU, alpha=0.3, width=1.5, label="Filtered")
        call_t = tradeable[tradeable["direction"]=="call"]
        put_t  = tradeable[tradeable["direction"]=="put"]
        ax.bar(call_t["trade_date"], call_t["gap_z_score"],
               color=GR, alpha=0.8, width=1.5, label="Call")
        ax.bar(put_t["trade_date"],  put_t["gap_z_score"],
               color=RD, alpha=0.8, width=1.5, label="Put")
        ax.axhline(0.3,  color=YL, lw=0.6, ls="--")
        ax.axhline(-0.3, color=YL, lw=0.6, ls="--")
        ax.axhline(3.5,  color=RD, lw=0.6, ls="--")
        ax.axhline(-3.5, color=RD, lw=0.6, ls="--")
        ax.set_ylabel("Gap Z-Score", color=MU)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.legend(fontsize=7, facecolor="#1e2530", labelcolor="white",
                  edgecolor="#2d3748", ncols=3)

    # ── Exit breakdown ────────────────────────────────────────────────────
    ax = axes["exit"]
    style(ax, "Exit Reasons")
    exits  = s["exits"]
    labels = list(exits.keys())
    counts = list(exits.values())
    colors = [GR if "GAMMA" in l or "CRUSH" in l else
              RD if "STOP" in l else MU for l in labels]
    bars = ax.bar(labels, counts, color=colors, alpha=0.85, edgecolor="#0d1117")
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(cnt), ha="center", va="bottom", color="white", fontsize=8)

    # ── Tier cumulative P&L ────────────────────────────────────────────────
    ax = axes["tier"]
    style(ax, "Tier Cumulative P&L")
    if len(tdf):
        t1 = tdf[tdf["tier"]==1]["net_pnl"].cumsum() / 1000
        t2 = tdf[tdf["tier"]==2]["net_pnl"].cumsum() / 1000
        ax.plot(range(len(t1)), t1.values, color=CY, lw=1.5,
                label=f"T1 3% n={s['tier1']['n']} WR={s['tier1']['win_rate']:.0%}")
        ax.plot(range(len(t2)), t2.values, color=YL, lw=1.5,
                label=f"T2 1% n={s['tier2']['n']} WR={s['tier2']['win_rate']:.0%}")
        ax.axhline(0, color=MU, lw=0.6, ls="--")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:.0f}K"))
        ax.legend(fontsize=7, facecolor="#1e2530", labelcolor="white",
                  edgecolor="#2d3748")

    # Stats footer
    fig.text(0.5, 0.01,
             f"Trades: {s['n_trades']}  WR: {s['win_rate']:.1%}  PF: {s['profit_factor']:.2f}  "
             f"Avg Win: {s['avg_win_ret']:+.1%}  Avg Loss: {s['avg_loss_ret']:.1%}  "
             f"Slippage: ${s['total_slippage']:,.0f}  Commission: ${s['total_commission']:,.0f}  "
             f"Recals: {s['n_recals']}",
             ha="center", color=MU, fontsize=8, style="italic")

    path = f"{out_prefix}_dashboard.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"Dashboard → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(
        prog="backtest_futures",
        description="APEX 0DTE — NQ Futures Gap Signal Backtest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Real NQ data from Yahoo Finance (requires internet)
  python backtest_futures.py --start 2018-01-01 --end 2026-01-01

  # Synthetic NQ data (no internet, for testing architecture)
  python backtest_futures.py --synthetic --start 2018-01-01 --end 2026-01-01

  # Local CSV files (e.g. from Barchart)
  python backtest_futures.py \\
      --nq-csv  data/NQ_daily.csv \\
      --qqq-csv data/QQQ_daily.csv \\
      --start 2018-01-01 --end 2026-01-01

  # Cache downloaded data to avoid re-downloading
  python backtest_futures.py --start 2018-01-01 --cache data/nq_cache

  # Conservative sizing, tighter risk limits
  python backtest_futures.py --synthetic --capital 50000 --max-dd 0.10
        """
    )
    p.add_argument("--synthetic", action="store_true",
                   help="Use synthetic NQ/QQQ data (no internet required)")
    p.add_argument("--nq-csv",    default=None, dest="nq_csv",
                   help="Path to NQ daily OHLCV CSV")
    p.add_argument("--qqq-csv",   default=None, dest="qqq_csv",
                   help="Path to QQQ daily OHLCV CSV")
    p.add_argument("--cache",     default=None,
                   help="Prefix for caching downloaded data (e.g. data/cache)")
    p.add_argument("--start",     default="2018-01-01", help="Backtest start (OOS)")
    p.add_argument("--end",       default="2026-01-01", help="Backtest end")
    p.add_argument("--train-end", default="2017-12-31", dest="train_end",
                   help="End of training period (before OOS start)")
    p.add_argument("--capital",   type=float, default=100_000, help="Initial capital")
    p.add_argument("--max-pos",   type=float, default=500_000, dest="max_pos",
                   help="Max position size USD")
    p.add_argument("--max-dd",    type=float, default=0.20, dest="max_dd",
                   help="Max drawdown before halt (default 0.20 = 20%%)")
    p.add_argument("--slippage",  default="dynamic",
                   choices=["dynamic","fixed","zero"],
                   help="Execution slippage model")
    p.add_argument("--out",       default="apex_nq_results",
                   help="Output file prefix")
    p.add_argument("--min-quality", type=float, default=0.0, dest="min_quality",
                   help="Minimum NQ gap quality score to trade (0–1, default 0 = no filter)")
    return p


def main():
    parser = build_parser()
    args   = parser.parse_args()

    # ── Load data ──────────────────────────────────────────────────────────
    if args.synthetic:
        logger.info("Building synthetic NQ/QQQ data...")
        nq_df, qqq_df = build_synthetic_nq()
    elif args.nq_csv and args.qqq_csv:
        logger.info(f"Loading from CSV: {args.nq_csv}, {args.qqq_csv}")
        nq_df, qqq_df = load_nq_from_csv(args.nq_csv, args.qqq_csv)
    else:
        logger.info("Downloading from Yahoo Finance (pip install yfinance if missing)...")
        try:
            nq_df, qqq_df = load_nq_from_yfinance(
                start="2014-01-01",  # extra history for calibration warm-up
                end=args.end,
                cache_path=args.cache,
            )
        except ImportError as e:
            logger.error(str(e))
            logger.error("Run with --synthetic to test without internet, "
                         "or pip install yfinance")
            sys.exit(1)

    logger.info(f"NQ:  {len(nq_df)} days  "
                f"{nq_df['date'].min()} → {nq_df['date'].max()}")
    logger.info(f"QQQ: {len(qqq_df)} days  "
                f"{qqq_df['date'].min()} → {qqq_df['date'].max()}")

    # ── Config ─────────────────────────────────────────────────────────────
    config = StrategyConfig(
        max_position_usd = args.max_pos,
        slippage_model   = args.slippage,
    )
    limits = RiskLimits(max_drawdown_pct=args.max_dd)

    start     = date.fromisoformat(args.start)
    end       = date.fromisoformat(args.end)
    train_end = date.fromisoformat(args.train_end)

    # ── Run ─────────────────────────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info(f"APEX 0DTE  |  NQ Futures Gap Signal  |  QQQ 0DTE Options")
    logger.info(f"OOS: {start} → {end}  |  Capital: ${args.capital:,.0f}")
    logger.info(f"{'='*60}\n")

    pm = run_futures_backtest(
        nq_df          = nq_df,
        qqq_df         = qqq_df,
        config         = config,
        limits         = limits,
        initial_capital = args.capital,
        start          = start,
        end            = end,
        train_end      = train_end,
        out_prefix     = args.out,
    )


if __name__ == "__main__":
    main()
