"""
APEX 0DTE — Main Runner
========================
Single entry point for all modes:

  python apex.py backtest --start 2018-01-01 --end 2026-01-01 --capital 100000
  python apex.py paper    --provider tradier  --key YOUR_KEY
  python apex.py live     --provider tradier  --key YOUR_KEY

Switching from synthetic → real data is a single --provider flag.
"""

from __future__ import annotations
import argparse
import logging
import sys
import os
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

# ── allow running from apex_0dte/ directory ──────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from data_providers import (
    SyntheticProvider, TradierProvider, AlpacaProvider, CBOEProvider
)
from strategy   import StrategyConfig
from portfolio  import PortfolioManager, RiskLimits

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("apex")


# ─────────────────────────────────────────────────────────────────────────────
# PROVIDER FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def make_provider(args):
    prov = getattr(args, "provider", "synthetic")
    sym  = getattr(args, "symbol", "QQQ")

    if prov == "tradier":
        key = getattr(args, "key", None) or os.environ.get("TRADIER_API_KEY", "")
        if not key:
            logger.error("Tradier API key required: --key or TRADIER_API_KEY env var")
            sys.exit(1)
        sandbox = getattr(args, "sandbox", True)
        logger.info(f"Using Tradier {'sandbox' if sandbox else 'live'} API")
        return TradierProvider(api_key=key, sandbox=sandbox, symbol=sym)

    elif prov == "alpaca":
        api_key    = getattr(args, "key", None) or os.environ.get("ALPACA_API_KEY", "")
        secret_key = getattr(args, "secret", None) or os.environ.get("ALPACA_API_SECRET", os.environ.get("ALPACA_SECRET_KEY", ""))
        paper      = getattr(args, "paper", True)
        logger.info(f"Using Alpaca {'paper' if paper else 'live'} API")
        return AlpacaProvider(api_key=api_key, secret_key=secret_key,
                               symbol=sym, paper=paper)

    elif prov == "cboe":
        csv_path = getattr(args, "csv", None)
        opt_dir  = getattr(args, "options_dir", None)
        if not csv_path:
            logger.error("CBOE provider requires --csv path to daily OHLCV file")
            sys.exit(1)
        logger.info(f"Using CBOE historical data: {csv_path}")
        return CBOEProvider(daily_csv_path=csv_path,
                            options_parquet_dir=opt_dir, symbol=sym)

    else:  # synthetic
        logger.info("Using synthetic data provider (no API required)")
        return SyntheticProvider(symbol=sym)


# ─────────────────────────────────────────────────────────────────────────────
# BACKTEST MODE
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest(args):
    provider = make_provider(args)
    config   = StrategyConfig(
        max_position_usd = getattr(args, "max_pos", 500_000),
        slippage_model   = getattr(args, "slippage", "dynamic"),
    )
    limits   = RiskLimits()
    pm       = PortfolioManager(
        initial_capital = getattr(args, "capital", 100_000),
        config          = config,
        risk_limits     = limits,
    )

    start = date.fromisoformat(args.start)
    end   = date.fromisoformat(args.end)

    # Load all daily bars (training + OOS)
    logger.info(f"Loading daily bars {start} → {end}")
    all_daily = provider.get_daily_bars(
        start - timedelta(days=365 * 3),   # 3yr lookback for calibration
        end
    )

    if len(all_daily) == 0:
        logger.error("No daily bars returned — check provider/dates")
        sys.exit(1)

    # Training period: first args.train_bars bars
    train_bars = getattr(args, "train_bars", 500)
    all_daily  = all_daily.reset_index(drop=True)

    # Initial calibration on training data
    pm.calibrator.calibrate(all_daily.iloc[:train_bars])
    pm._cal = pm.calibrator.thresholds()
    logger.info(f"Initial cal: {pm._cal}")

    # OOS loop
    oos_dates = all_daily.iloc[train_bars:]["date"].tolist()
    logger.info(f"Running OOS: {len(oos_dates)} trading days")

    for i, row_date in enumerate(oos_dates):
        bar_idx  = train_bars + i
        row      = all_daily.iloc[bar_idx]
        gap_pct  = float(row["gap_pct"]) if pd.notna(row["gap_pct"]) else 0.0
        tdate    = row_date.date() if hasattr(row_date, 'date') else row_date

        # History = everything before today
        history  = all_daily.iloc[:bar_idx]

        # Get intraday bars + option chain (only if signal likely)
        # Pre-check gap to avoid unnecessary API calls
        cal = pm._cal
        if not ((cal["c_lo"] <= gap_pct <= cal["c_hi"]) or
                (cal["p_hi"] >= gap_pct >= cal["p_lo"])):
            pm.equity_curve.append((tdate, pm.portfolio))
            pm._bar_count += 1
            pm._maybe_recalibrate(history)
            continue

        intraday = provider.get_intraday_bars(tdate)
        chain    = provider.get_option_chain_at_open(tdate, tdate)   # 0DTE: same day expiry

        result = pm.process_day(tdate, history, intraday, chain, gap_pct)

        if pm._halted:
            logger.warning(f"Engine halted at {tdate}: {pm._halt_reason}")
            break

        if (i + 1) % 100 == 0:
            pct_done = (i + 1) / len(oos_dates) * 100
            logger.info(f"Progress {pct_done:.0f}%  "
                        f"Portfolio: ${pm.portfolio:,.0f}  "
                        f"Trades: {len(pm.trade_log)}")

    # Final equity point
    pm.equity_curve.append((oos_dates[-1], pm.portfolio))

    pm.print_summary()
    _save_outputs(pm, getattr(args, "out", "apex_results"))


# ─────────────────────────────────────────────────────────────────────────────
# PAPER / LIVE MODE
# ─────────────────────────────────────────────────────────────────────────────

def run_live(args, paper: bool = True):
    """
    Runs once per day (call at 9:28am EST via cron / Task Scheduler).
    Loads persistent state from disk, processes today, saves state.
    """
    provider   = make_provider(args)
    state_file = getattr(args, "state", "apex_live_state.pkl")
    import pickle

    # Load or init portfolio manager
    if os.path.exists(state_file):
        with open(state_file, "rb") as f:
            pm = pickle.load(f)
        logger.info(f"Loaded state: ${pm.portfolio:,.0f}  "
                    f"{len(pm.trade_log)} trades")
    else:
        pm = PortfolioManager(
            initial_capital = getattr(args, "capital", 100_000),
            config          = StrategyConfig(
                max_position_usd = getattr(args, "max_pos", 500_000),
                slippage_model   = "dynamic",
            ),
        )
        logger.info(f"New portfolio: ${pm.initial:,.0f}")

    today   = date.today()
    capital_arg = getattr(args, "capital", 100_000)

    # Load sufficient history for calibration
    lookback = today - timedelta(days=365 * 2)
    logger.info(f"Fetching daily bars {lookback} → {today}")
    daily = provider.get_daily_bars(lookback, today - timedelta(days=1))

    if len(daily) < 20:
        logger.error("Insufficient history — cannot calibrate")
        return

    gap_pct = float(daily["gap_pct"].iloc[-1]) if len(daily) > 1 else 0.0

    # Fetch intraday + chain only if likely to trade
    pm._maybe_recalibrate(daily)
    cal = pm._cal
    will_signal = ((cal["c_lo"] <= gap_pct <= cal["c_hi"]) or
                   (cal["p_hi"] >= gap_pct >= cal["p_lo"]))

    if not will_signal:
        logger.info(f"No signal today (gap={gap_pct:+.4f}) — no trade")
        pm.equity_curve.append((today, pm.portfolio))
        _save_state(pm, state_file, paper)
        return

    logger.info(f"Signal! gap={gap_pct:+.4f}  Fetching intraday + chain...")
    intraday = provider.get_intraday_bars(today)
    chain    = provider.get_option_chain_at_open(today, today)

    result = pm.process_day(today, daily, intraday, chain, gap_pct)

    if result:
        mode = "PAPER" if paper else "LIVE"
        print(f"\n{'='*50}")
        print(f"[{mode}] {today}  {result.direction.upper()} T{result.tier}")
        print(f"  Gap:      {result.gap_pct:+.4f}")
        print(f"  Entry:    ${result.entry_fill:.3f}  ×  {result.contracts} contracts")
        print(f"  Exit:     ${result.exit_fill:.3f}  ({result.exit_reason}  bar {result.exit_bar})")
        print(f"  Net P&L:  ${result.net_pnl:>+,.0f}  ({result.return_pct:+.1%} on risk)")
        print(f"  Portfolio: ${result.portfolio_after:>,.0f}")
        print(f"{'='*50}\n")

        # In live mode: place actual order via broker API here
        if not paper:
            logger.warning("LIVE ORDER EXECUTION not yet implemented — "
                           "manual execution required")

    _save_state(pm, state_file, paper)
    pm.print_summary()


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT & PERSISTENCE
# ─────────────────────────────────────────────────────────────────────────────

def _save_state(pm: PortfolioManager, path: str, paper: bool):
    import pickle
    with open(path, "wb") as f:
        pickle.dump(pm, f)
    logger.info(f"State saved → {path}")


def _save_outputs(pm: PortfolioManager, out_prefix: str):
    """Save trade log, equity curve, and dashboard."""
    tdf = pm.trade_dataframe()
    edf = pm.equity_dataframe()

    if len(tdf):
        csv_path = f"{out_prefix}_trades.csv"
        tdf.to_csv(csv_path, index=False)
        logger.info(f"Trades saved → {csv_path}")

    if len(edf):
        eq_path = f"{out_prefix}_equity.csv"
        edf.to_csv(eq_path, index=False)
        logger.info(f"Equity curve saved → {eq_path}")

    # Generate dashboard
    try:
        _plot_dashboard(pm, tdf, edf, out_prefix)
    except Exception as e:
        logger.warning(f"Dashboard plot failed: {e}")


def _plot_dashboard(pm: PortfolioManager, tdf: pd.DataFrame,
                    edf: pd.DataFrame, out_prefix: str):
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.ticker import FuncFormatter

    s   = pm.summary()
    fig = plt.figure(figsize=(18, 11), facecolor="#0d1117")
    fig.suptitle("APEX 0DTE QQQ  —  Strategy Performance",
                 color="white", fontsize=15, fontweight="bold", y=0.98)

    RD = "#ef4444"; GR = "#22c55e"; BL = "#3b82f6"
    MU = "#94a3b8"; YL = "#eab308"; CY = "#06b6d4"
    ax_kw = dict(facecolor="#161b22",
                 **{f"{s}color": "#1e2530" for s in ["left","right","bottom","top"]})

    axes = fig.subplot_mosaic(
        [["eq","eq","eq","ret"],
         ["dd","dd","exit","tier"]],
        gridspec_kw={"hspace": 0.38, "wspace": 0.35}
    )

    def style(ax, title):
        ax.set_facecolor("#161b22")
        for sp in ax.spines.values(): sp.set_color("#2d3748")
        ax.tick_params(colors=MU, labelsize=8)
        ax.set_title(title, color="white", fontsize=9, pad=6)
        ax.yaxis.label.set_color(MU); ax.xaxis.label.set_color(MU)
        ax.grid(True, color="#2d3748", lw=0.4, alpha=0.7)

    # ── Equity curve (log) ─────────────────────────────────────────────
    ax = axes["eq"]
    style(ax, f"Equity Curve (log scale)  |  CAGR {s['cagr']:+.1%}  "
          f"Sharpe {s['sharpe']:.2f}  MaxDD {s['max_drawdown']:.1%}")
    if len(edf):
        dates = pd.to_datetime(edf["date"])
        vals  = edf["portfolio"].values / 1000
        ax.semilogy(dates, vals, color=RD, lw=1.8, label="APEX Strategy", zorder=3)
        # Milestones
        start_k = vals[0]
        for mult, lbl in [(2,"2×"),(5,"5×"),(10,"10×"),(50,"50×"),(100,"100×")]:
            target = start_k * mult
            if vals[-1] >= target >= vals[0]:
                ax.axhline(target, color=GR, lw=0.5, ls=":", alpha=0.6)
                ax.text(dates.iloc[2], target * 1.05, lbl, color=MU, fontsize=7)
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda x, _: f"${x:,.0f}K" if x < 1000 else f"${x/1000:.1f}M"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(fontsize=8, loc="upper left",
              labelcolor="white", facecolor="#1e2530", edgecolor="#2d3748")

    # ── Return distribution ────────────────────────────────────────────
    ax = axes["ret"]
    style(ax, f"Return Dist  WR={s['win_rate']:.1%}  PF={s['profit_factor']:.2f}")
    if len(tdf):
        rets = tdf["return_pct"].values * 100
        wins = rets[rets > 0]; losses = rets[rets <= 0]
        bins = np.linspace(-50, 200, 40)
        ax.hist(losses, bins=bins, color=RD, alpha=0.7, label=f"Losses ({len(losses)})")
        ax.hist(wins,   bins=bins, color=GR, alpha=0.7, label=f"Wins ({len(wins)})")
        ax.axvline(0, color=MU, lw=0.8, ls="--")
        ax.axvline(float(np.mean(rets)), color=YL, lw=1.2, ls="--",
                   label=f"Mean {np.mean(rets):+.1f}%")
        ax.set_xlabel("Return on Risk Capital (%)")
        ax.legend(fontsize=7, facecolor="#1e2530", labelcolor="white",
                  edgecolor="#2d3748")

    # ── Drawdown ───────────────────────────────────────────────────────
    ax = axes["dd"]
    style(ax, f"Drawdown  |  Max DD {s['max_drawdown']:.1%}")
    if len(edf):
        vals  = edf["portfolio"].values
        dd    = (1 - vals / np.maximum.accumulate(vals)) * 100
        ax.fill_between(pd.to_datetime(edf["date"]), -dd, 0,
                        color=RD, alpha=0.4)
        ax.plot(pd.to_datetime(edf["date"]), -dd, color=RD, lw=0.8)
        ax.axhline(-s['max_drawdown'] * 100, color=YL, lw=0.8, ls="--",
                   label=f"Max DD {s['max_drawdown']:.1%}")
        ax.set_ylabel("Drawdown (%)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.legend(fontsize=7, facecolor="#1e2530", labelcolor="white",
                  edgecolor="#2d3748")

    # ── Exit breakdown ─────────────────────────────────────────────────
    ax = axes["exit"]
    style(ax, "Exit Reasons")
    exits = s["exits"]
    labels = list(exits.keys())
    counts = list(exits.values())
    colors = [GR if "GAMMA" in l or "CRUSH" in l else
              RD if "STOP" in l else MU for l in labels]
    bars = ax.bar(labels, counts, color=colors, alpha=0.85, edgecolor="#0d1117")
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=7)
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                str(cnt), ha="center", va="bottom", color="white", fontsize=7)

    # ── Tier breakdown ─────────────────────────────────────────────────
    ax = axes["tier"]
    style(ax, "Tier P&L")
    if len(tdf):
        t1 = tdf[tdf["tier"]==1]["net_pnl"].cumsum() / 1000
        t2 = tdf[tdf["tier"]==2]["net_pnl"].cumsum() / 1000
        ax.plot(range(len(t1)), t1.values, color=CY, lw=1.5,
                label=f"T1 3% (n={s['tier1']['n']}  WR={s['tier1']['win_rate']:.0%})")
        ax.plot(range(len(t2)), t2.values, color=YL, lw=1.5,
                label=f"T2 1% (n={s['tier2']['n']}  WR={s['tier2']['win_rate']:.0%})")
        ax.axhline(0, color=MU, lw=0.6, ls="--")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:.0f}K"))
        ax.legend(fontsize=7, facecolor="#1e2530", labelcolor="white",
                  edgecolor="#2d3748")

    # Stats box
    slippage_drag = s["total_slippage"] / pm.initial * 100
    comm_drag     = s["total_commission"] / pm.initial * 100
    stats_text = (
        f"Trades: {s['n_trades']}   Win Rate: {s['win_rate']:.1%}   "
        f"Profit Factor: {s['profit_factor']:.2f}\n"
        f"Avg Win: {s['avg_win_ret']:+.1%}   Avg Loss: {s['avg_loss_ret']:.1%}   "
        f"Slippage drag: {slippage_drag:.2f}%   Commission drag: {comm_drag:.2f}%\n"
        f"Recalibrations: {s['n_recals']}   Capped: {s['n_capped']}   "
        f"Max position: $500K"
    )
    fig.text(0.5, 0.01, stats_text, ha="center", va="bottom",
             color=MU, fontsize=8, style="italic")

    path = f"{out_prefix}_dashboard.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"Dashboard saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(
        prog="apex",
        description="APEX 0DTE QQQ Options Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Backtest with synthetic data
  python apex.py backtest --start 2018-01-01 --end 2026-01-01

  # Backtest with Tradier sandbox (real historical bars, synthetic options)
  python apex.py backtest --provider tradier --key YOUR_SANDBOX_KEY \\
         --start 2022-01-01 --end 2025-01-01 --sandbox

  # Backtest with CBOE daily CSV + real QQQ history
  python apex.py backtest --provider cboe --csv data/QQQ_daily.csv \\
         --start 2018-01-01 --end 2026-01-01

  # Paper trading (runs once; add to cron at 9:28am EST)
  python apex.py paper --provider tradier --key YOUR_SANDBOX_KEY

  # Live trading (requires broker integration — use with caution)
  python apex.py live --provider tradier --key YOUR_LIVE_KEY
        """
    )
    sub = p.add_subparsers(dest="mode", required=True)

    # ── backtest ──
    bt = sub.add_parser("backtest", help="Run historical backtest")
    bt.add_argument("--start",       default="2018-01-01", help="OOS start date")
    bt.add_argument("--end",         default="2026-01-01", help="End date")
    bt.add_argument("--capital",     type=float, default=100_000)
    bt.add_argument("--train-bars",  type=int,   default=500,
                    dest="train_bars", help="Training bars before OOS")
    bt.add_argument("--max-pos",     type=float, default=500_000,
                    dest="max_pos",    help="Max position size in USD")
    bt.add_argument("--slippage",    default="dynamic",
                    choices=["dynamic","fixed","zero"])
    bt.add_argument("--out",         default="apex_results",
                    help="Output file prefix")
    bt.add_argument("--provider",    default="synthetic",
                    choices=["synthetic","tradier","alpaca","cboe"])
    bt.add_argument("--symbol",      default="QQQ")
    bt.add_argument("--key",         default=None, help="API key")
    bt.add_argument("--secret",      default=None, help="API secret (Alpaca)")
    bt.add_argument("--sandbox",     action="store_true")
    bt.add_argument("--csv",         default=None, help="CBOE daily CSV path")
    bt.add_argument("--options-dir", default=None, dest="options_dir")

    # ── paper ──
    pp = sub.add_parser("paper", help="Paper trading (runs once per day)")
    pp.add_argument("--provider", default="tradier",
                    choices=["tradier","alpaca"])
    pp.add_argument("--symbol",   default="QQQ")
    pp.add_argument("--key",      required=True)
    pp.add_argument("--secret",   default=None)
    pp.add_argument("--sandbox",  action="store_true", default=True)
    pp.add_argument("--capital",  type=float, default=100_000)
    pp.add_argument("--max-pos",  type=float, default=500_000, dest="max_pos")
    pp.add_argument("--state",    default="apex_paper_state.pkl")

    # ── live ──
    lv = sub.add_parser("live", help="Live trading (use with caution)")
    lv.add_argument("--provider", default="tradier",
                    choices=["tradier","alpaca"])
    lv.add_argument("--symbol",   default="QQQ")
    lv.add_argument("--key",      required=True)
    lv.add_argument("--secret",   default=None)
    lv.add_argument("--capital",  type=float, default=100_000)
    lv.add_argument("--max-pos",  type=float, default=500_000, dest="max_pos")
    lv.add_argument("--state",    default="apex_live_state.pkl")

    return p


def main():
    parser = build_parser()
    args   = parser.parse_args()

    if args.mode == "backtest":
        run_backtest(args)
    elif args.mode == "paper":
        run_live(args, paper=True)
    elif args.mode == "live":
        confirm = input("⚠️  LIVE mode uses REAL money. Type 'confirm' to proceed: ")
        if confirm.strip() != "confirm":
            print("Aborted.")
            return
        run_live(args, paper=False)


if __name__ == "__main__":
    main()
