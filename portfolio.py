"""
APEX 0DTE — Portfolio Manager
===============================
Owns all state across trading days:
  - Current portfolio value
  - Trade log
  - Recalibration schedule
  - Performance metrics
  - Risk limits

Works identically in backtest and live modes.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import List, Optional

import numpy as np
import pandas as pd

from strategy import TradeResult, StrategyConfig, GapCalibrator

logger = logging.getLogger("apex.portfolio")


@dataclass
class RiskLimits:
    """Hard risk limits enforced before any trade is taken."""
    max_daily_loss_pct:    float = 0.05   # halt trading if down >5% in one day
    max_drawdown_pct:      float = 0.20   # halt if down >20% from peak ever
    max_open_positions:    int   = 1      # only 1 position at a time (0DTE rule)
    min_portfolio_usd:     float = 10_000 # stop trading if below this

    def check(self, portfolio: float, peak: float,
              daily_start: float, open_positions: int) -> tuple[bool, str]:
        """Returns (ok_to_trade, reason)."""
        if portfolio < self.min_portfolio_usd:
            return False, f"portfolio_below_minimum (${portfolio:,.0f})"
        dd = (peak - portfolio) / peak if peak > 0 else 0
        if dd > self.max_drawdown_pct:
            return False, f"max_drawdown_breach ({dd:.1%})"
        daily_loss = (daily_start - portfolio) / daily_start if daily_start > 0 else 0
        if daily_loss > self.max_daily_loss_pct:
            return False, f"daily_loss_limit ({daily_loss:.1%})"
        if open_positions >= self.max_open_positions:
            return False, "max_open_positions"
        return True, "ok"


class PortfolioManager:
    """
    Central coordinator for the APEX engine.
    Call process_day() for each trading day in sequence.
    """

    def __init__(self,
                 initial_capital: float = 100_000,
                 config: StrategyConfig = None,
                 risk_limits: RiskLimits = None):
        self.config        = config or StrategyConfig()
        self.limits        = risk_limits or RiskLimits()
        self.portfolio     = initial_capital
        self.initial       = initial_capital
        self.peak          = initial_capital
        self.daily_start   = initial_capital
        self.trade_log:    List[TradeResult] = []
        self.equity_curve: List[tuple]       = []   # (date, value)
        self.recal_log:    List[dict]        = []
        self.calibrator    = GapCalibrator(self.config)
        self._cal          = self.calibrator._defaults()
        self._last_recal_bar = 0
        self._last_rvol    = 0.20
        self._bar_count    = 0   # total OOS bars processed
        self._open_positions = 0
        self._halted       = False
        self._halt_reason  = ""

    # ─────────────────────────────────────────────────────────────────────
    # MAIN ENTRY POINT
    # ─────────────────────────────────────────────────────────────────────

    def process_day(self, trading_date: date,
                    daily_bars_history: pd.DataFrame,
                    intraday_bars: pd.DataFrame,
                    option_chain: pd.DataFrame,
                    gap_pct: float) -> Optional[TradeResult]:
        """
        Process a single trading day end-to-end.

        daily_bars_history: all past daily bars (NOT including today)
        intraday_bars: 5-min bars for today
        option_chain: option quotes at open for today's expiry
        gap_pct: today's opening gap fraction

        Returns TradeResult if a trade was taken, else None.
        """
        # Track equity at start of day
        self.daily_start = self.portfolio
        self.equity_curve.append((trading_date, self.portfolio))

        # ── Recalibration check ────────────────────────────────────────
        self._maybe_recalibrate(daily_bars_history)
        self._bar_count += 1

        # ── Risk gate ─────────────────────────────────────────────────
        if self._halted:
            logger.warning(f"{trading_date}: trading halted — {self._halt_reason}")
            return None

        ok, reason = self.limits.check(
            self.portfolio, self.peak, self.daily_start, self._open_positions)
        if not ok:
            logger.warning(f"{trading_date}: risk gate BLOCKED — {reason}")
            self._halted = True
            self._halt_reason = reason
            return None

        # ── Signal generation ──────────────────────────────────────────
        from strategy import signal_for_day, run_intraday_trade
        from data_providers import ExecutionModel

        signal = signal_for_day(gap_pct, daily_bars_history, self._cal, self.config)

        if not signal.trade:
            logger.debug(f"{trading_date}: no signal — {signal.reason}")
            return None

        # ── Execute trade ──────────────────────────────────────────────
        exec_model = ExecutionModel(
            spread_model=self.config.slippage_model,
            commission_per_contract=self.config.commission_per_contract,
        )

        self._open_positions += 1
        result = run_intraday_trade(
            signal       = signal,
            intraday_bars = intraday_bars,
            option_chain_open = option_chain,
            portfolio_value   = self.portfolio,
            config     = self.config,
            exec_model = exec_model,
            trading_date = trading_date,
        )
        self._open_positions -= 1

        if result is None:
            return None

        # ── Update portfolio ───────────────────────────────────────────
        self.portfolio += result.net_pnl
        result.portfolio_after = self.portfolio
        self.peak = max(self.peak, self.portfolio)

        # ── Store ──────────────────────────────────────────────────────
        self.trade_log.append(result)

        # ── Post-trade risk check ──────────────────────────────────────
        ok, reason = self.limits.check(
            self.portfolio, self.peak, self.daily_start, 0)
        if not ok:
            logger.warning(f"{trading_date}: POST-TRADE risk limit — {reason}")
            self._halted = True
            self._halt_reason = reason

        return result

    # ─────────────────────────────────────────────────────────────────────
    # RECALIBRATION
    # ─────────────────────────────────────────────────────────────────────

    def _maybe_recalibrate(self, daily_bars_history: pd.DataFrame):
        bars_since = self._bar_count - self._last_recal_bar
        if bars_since < self.config.recal_freq_bars:
            return

        old_cal  = dict(self._cal)
        self._cal = self.calibrator.calibrate(daily_bars_history)
        new_rvol  = self._cal["rvol"]

        vol_shift = abs(new_rvol - self._last_rvol) / (self._last_rvol + 1e-9)
        trigger   = "vol_shift" if vol_shift > self.config.vol_shift_trigger \
                    else "quarterly"

        self.recal_log.append({
            "bar":       self._bar_count,
            "trigger":   trigger,
            "vol_shift": vol_shift,
            "rvol":      new_rvol,
            **{k: self._cal[k] for k in ["c_lo","c_hi","p_lo","p_hi"]},
        })

        self._last_recal_bar = self._bar_count
        self._last_rvol      = new_rvol
        logger.info(f"Recalibrated (bar={self._bar_count}, {trigger}): "
                    f"rvol={new_rvol:.3f} vol_shift={vol_shift:.1%}")

    # ─────────────────────────────────────────────────────────────────────
    # REPORTING
    # ─────────────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        if not self.trade_log:
            return {"error": "no trades"}

        tl = self.trade_log
        pnls     = np.array([t.net_pnl   for t in tl])
        rets     = np.array([t.return_pct for t in tl])
        wins     = pnls > 0
        losses   = pnls <= 0

        # Equity curve stats
        eq_vals  = np.array([v for _, v in self.equity_curve])
        n_days   = len(eq_vals)
        yrs      = n_days / 252
        cagr     = (self.portfolio / self.initial) ** (1 / max(yrs, 0.01)) - 1
        daily_ret = np.diff(np.log(np.maximum(eq_vals, 1)))
        sharpe   = (np.mean(daily_ret) / (np.std(daily_ret) + 1e-9)) * 252 ** 0.5
        dd_curve = 1 - eq_vals / np.maximum.accumulate(eq_vals)
        max_dd   = float(dd_curve.max())

        # By tier
        t1 = [t for t in tl if t.tier == 1]
        t2 = [t for t in tl if t.tier == 2]

        # Exit breakdown
        exits = {}
        for t in tl:
            exits[t.exit_reason] = exits.get(t.exit_reason, 0) + 1

        # Slippage & commission
        total_slip = sum(t.slippage_cost for t in tl)
        total_comm = sum(t.commission for t in tl)

        win_trades  = [t for t in tl if t.net_pnl > 0]
        loss_trades = [t for t in tl if t.net_pnl <= 0]

        return {
            "portfolio_start": self.initial,
            "portfolio_end":   self.portfolio,
            "total_return":    (self.portfolio / self.initial) - 1,
            "cagr":            cagr,
            "max_drawdown":    max_dd,
            "sharpe":          sharpe,
            "n_trades":        len(tl),
            "n_days":          n_days,
            "win_rate":        float(wins.mean()),
            "profit_factor":   (pnls[wins].sum() / (-pnls[losses].sum() + 1e-9)),
            "avg_pnl":         float(pnls.mean()),
            "avg_win_pnl":     float(pnls[wins].mean()) if wins.any() else 0,
            "avg_loss_pnl":    float(pnls[losses].mean()) if losses.any() else 0,
            "avg_win_ret":     float(rets[wins].mean()) if wins.any() else 0,
            "avg_loss_ret":    float(rets[losses].mean()) if losses.any() else 0,
            "total_gross_pnl": float(pnls.sum()),
            "total_slippage":  total_slip,
            "total_commission": total_comm,
            "exits":           exits,
            "n_recals":        len(self.recal_log),
            "n_capped":        sum(1 for t in tl if t.capped),
            "halted":          self._halted,
            "halt_reason":     self._halt_reason,
            "tier1": {
                "n":        len(t1),
                "win_rate": np.mean([t.net_pnl > 0 for t in t1]) if t1 else 0,
                "avg_pnl":  np.mean([t.net_pnl for t in t1]) if t1 else 0,
                "total_pnl": sum(t.net_pnl for t in t1),
            },
            "tier2": {
                "n":        len(t2),
                "win_rate": np.mean([t.net_pnl > 0 for t in t2]) if t2 else 0,
                "avg_pnl":  np.mean([t.net_pnl for t in t2]) if t2 else 0,
                "total_pnl": sum(t.net_pnl for t in t2),
            },
        }

    def trade_dataframe(self) -> pd.DataFrame:
        if not self.trade_log:
            return pd.DataFrame()
        rows = []
        for t in self.trade_log:
            rows.append({
                "date":            t.date,
                "direction":       t.direction,
                "tier":            t.tier,
                "gap_pct":         t.gap_pct,
                "iv_daily":        t.iv_daily,
                "exit_reason":     t.exit_reason,
                "exit_bar":        t.exit_bar,
                "entry_fill":      t.entry_fill,
                "exit_fill":       t.exit_fill,
                "contracts":       t.contracts,
                "risk_cap":        t.risk_cap,
                "gross_pnl":       t.gross_pnl,
                "commission":      t.commission,
                "slippage":        t.slippage_cost,
                "net_pnl":         t.net_pnl,
                "return_pct":      t.return_pct,
                "portfolio_before": t.portfolio_before,
                "portfolio_after":  t.portfolio_after,
                "peak_gamma":       t.peak_gamma,
                "capped":           t.capped,
            })
        return pd.DataFrame(rows)

    def equity_dataframe(self) -> pd.DataFrame:
        dates = [d for d, _ in self.equity_curve]
        vals  = [v for _, v in self.equity_curve]
        return pd.DataFrame({"date": dates, "portfolio": vals})

    def print_summary(self):
        s = self.summary()
        print("\n" + "="*60)
        print("APEX 0DTE RESULTS")
        print("="*60)
        print(f"Portfolio: ${s['portfolio_start']:>12,.0f}  →  ${s['portfolio_end']:>14,.0f}")
        print(f"CAGR:      {s['cagr']:>+.1%}    Max DD: {s['max_drawdown']:.1%}    Sharpe: {s['sharpe']:.2f}")
        print(f"Trades:    {s['n_trades']}    Win rate: {s['win_rate']:.1%}    PF: {s['profit_factor']:.2f}")
        print(f"Avg win:   {s['avg_win_ret']:>+.1%}    Avg loss: {s['avg_loss_ret']:.1%}")
        print(f"Slippage:  ${s['total_slippage']:>+,.0f}    Commission: ${s['total_commission']:>+,.0f}")
        print(f"Exits:     {s['exits']}")
        print(f"\nTier-1 ({s['tier1']['n']} trades): WR={s['tier1']['win_rate']:.1%}  "
              f"AvgPnL=${s['tier1']['avg_pnl']:,.0f}  Total=${s['tier1']['total_pnl']:,.0f}")
        print(f"Tier-2 ({s['tier2']['n']} trades): WR={s['tier2']['win_rate']:.1%}  "
              f"AvgPnL=${s['tier2']['avg_pnl']:,.0f}  Total=${s['tier2']['total_pnl']:,.0f}")
        print(f"\nRecalibrations: {s['n_recals']}    Capped trades: {s['n_capped']}")
        if s['halted']:
            print(f"⚠️  Engine halted: {s['halt_reason']}")
        print("="*60)
