"""
APEX-6W: Rolling 6-Week Options on Top QQQ Names
==================================================
Strategy:
  - Every Monday, check regime (QQQ vs SMA50)
  - BULL: buy 6-week ATM calls on top 6 QQQ names, equal weight
  - BEAR: buy 6-week ATM puts on top 6 QQQ names, equal weight
  - MIXED signal: PLTR+NVDA composite filters direction per name
  - Roll when <= 7 days remain (buy new leg before closing old)
  - Position sizing: 2% of portfolio per name = 12% total exposure

Top 6 QQQ names (by weight, periodically updated):
  AAPL, MSFT, NVDA, AMZN, META, GOOGL

Run: python3 apex_6w_backtest.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import norm
from scipy import stats
from dataclasses import dataclass, field
from typing import Optional
import warnings, os, logging
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
TOP_6 = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL"]

# Historical QQQ top-6 by era (weights shift over time)
# We use fixed list but note NVDA replaced TSLA ~2023
TOP_6_BY_ERA = {
    "2018-01-01": ["AAPL", "MSFT", "AMZN", "GOOGL", "FB",   "INTC"],
    "2020-01-01": ["AAPL", "MSFT", "AMZN", "GOOGL", "FB",   "TSLA"],
    "2022-01-01": ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA"],
    "2023-06-01": ["AAPL", "MSFT", "NVDA", "AMZN",  "META", "GOOGL"],
}

RISK_PER_NAME  = 0.02   # 2% of portfolio per name
SMA_WINDOW     = 50
OPTION_WEEKS   = 6      # buy 6-week options
ROLL_DAYS_LEFT = 7      # roll when 7 days remain
OTM_CALL_PCT   = 0.02   # 2% OTM calls (slight OTM for leverage)
OTM_PUT_PCT    = 0.02   # 2% OTM puts
IV_BASE_MULT   = 1.20   # scale up HV for option pricing realism

TRADING_YEAR   = 252.0

# ── BSM PRICING ───────────────────────────────────────────────────────────────
def bsm_price(S, K, T, r, sigma, is_call):
    if T <= 0 or sigma <= 0:
        intrinsic = max(S - K, 0) if is_call else max(K - S, 0)
        return max(intrinsic, 0.01)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if is_call:
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def bsm_delta(S, K, T, r, sigma, is_call):
    if T <= 0 or sigma <= 0:
        return 1.0 if is_call else -1.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) if is_call else norm.cdf(d1) - 1

# ── DATA LOADING ──────────────────────────────────────────────────────────────
def load_data(tickers, start, end):
    """Load daily OHLCV for all tickers. Returns dict of DataFrames."""
    api_key    = os.environ.get("ALPACA_API_KEY", "")
    api_secret = os.environ.get("ALPACA_API_SECRET", "")

    if api_key and api_secret:
        try:
            import alpaca_trade_api as tradeapi
            api = tradeapi.REST(api_key, api_secret,
                                base_url="https://paper-api.alpaca.markets")
            data = {}
            for t in tickers:
                try:
                    raw = api.get_bars(t, "1Day", start=start, end=end,
                                       adjustment="all").df.reset_index()
                    raw.columns = [c.lower() for c in raw.columns]
                    raw["date"] = pd.to_datetime(raw.get("timestamp",
                                  raw.get("time", raw.index))).dt.date
                    data[t] = raw.sort_values("date").reset_index(drop=True)
                    log.info(f"  Loaded {t}: {len(raw)} days")
                except Exception as e:
                    log.warning(f"  {t} failed: {e}")
            if data:
                return data
        except Exception as e:
            log.warning(f"Alpaca failed: {e}")

    # ── SYNTHETIC fallback — calibrated to each stock's empirical params ──────
    log.info("Using SYNTHETIC data — set ALPACA keys for real data")
    np.random.seed(42)

    # Approximate params per ticker (annualised vol, beta to QQQ, drift)
    params = {
        "QQQ":  (0.22, 1.00, 0.15),
        "AAPL": (0.28, 1.10, 0.18),
        "MSFT": (0.26, 1.05, 0.20),
        "NVDA": (0.55, 1.80, 0.35),
        "AMZN": (0.32, 1.20, 0.22),
        "META": (0.38, 1.25, 0.25),
        "GOOGL":(0.28, 1.10, 0.18),
        "TSLA": (0.65, 1.50, 0.20),
        "FB":   (0.38, 1.25, 0.25),
        "INTC": (0.28, 0.90, 0.05),
    }

    n_days  = int((pd.to_datetime(end) - pd.to_datetime(start)).days * 252 / 365)
    dates   = pd.bdate_range(start=start, periods=n_days)

    # Generate correlated returns using QQQ as market factor
    qqq_vol   = 0.22 / np.sqrt(252)
    qqq_drift = 0.15 / 252
    qqq_rets  = np.random.normal(qqq_drift, qqq_vol, n_days)

    data = {}
    start_prices = {"QQQ": 400, "AAPL": 150, "MSFT": 250, "NVDA": 200,
                    "AMZN": 130, "META": 200, "GOOGL": 130,
                    "TSLA": 200, "FB": 200, "INTC": 50}

    for t in set(tickers + ["QQQ"]):
        vol, beta, drift = params.get(t, (0.30, 1.0, 0.15))
        idio_vol  = np.sqrt(max(vol**2/252 - (beta*qqq_vol)**2, 1e-8))
        idio      = np.random.normal(drift/252, idio_vol, n_days)
        rets      = beta * qqq_rets + idio

        # Add gap structure
        gap_days  = np.random.choice(n_days, size=int(n_days*0.05), replace=False)
        gaps      = np.random.normal(0, vol/np.sqrt(52), len(gap_days))
        rets[gap_days] += gaps

        prices    = start_prices.get(t, 100) * np.cumprod(1 + rets)
        opens     = prices * (1 + np.random.normal(0, 0.002, n_days))

        df = pd.DataFrame({
            "date":   [d.date() for d in dates],
            "open":   opens, "high": prices * 1.005,
            "low":    prices * 0.995, "close": prices,
            "volume": 20_000_000
        })
        data[t] = df

    return data

# ── GET ACTIVE TOP-6 FOR DATE ─────────────────────────────────────────────────
def get_top6(date):
    """Return the top-6 tickers for a given date based on historical weights."""
    active = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL"]
    for era_start, tickers in sorted(TOP_6_BY_ERA.items()):
        if str(date) >= era_start:
            active = tickers
    return active

# ── IV ESTIMATE ───────────────────────────────────────────────────────────────
def estimate_iv(hist_closes, window=21):
    if len(hist_closes) < window + 1:
        return 0.30
    rets = np.diff(np.log(hist_closes[-window-1:]))
    hv   = np.std(rets, ddof=1) * np.sqrt(252)
    return float(np.clip(hv * IV_BASE_MULT, 0.15, 2.00))

# ── REGIME ────────────────────────────────────────────────────────────────────
def get_regime(qqq_hist):
    """Returns 'bull', 'bear', or 'neutral'."""
    if len(qqq_hist) < SMA_WINDOW:
        return "neutral"
    closes = qqq_hist["close"].values
    sma50  = closes[-SMA_WINDOW:].mean()
    last   = closes[-1]
    if last > sma50 * 1.01:
        return "bull"
    elif last < sma50 * 0.99:
        return "bear"
    return "neutral"

# ── OPTION POSITION ───────────────────────────────────────────────────────────
@dataclass
class OptionPos:
    ticker:      str
    direction:   str        # 'call' or 'put'
    entry_date:  object
    expiry_date: object
    S_entry:     float
    K:           float
    iv:          float
    entry_px:    float
    contracts:   int
    cost:        float      # total premium paid
    portfolio_at_entry: float

# ── TRADE RESULT ──────────────────────────────────────────────────────────────
@dataclass
class TradeResult:
    ticker:      str
    direction:   str
    entry_date:  object
    exit_date:   object
    expiry_date: object
    S_entry:     float
    S_exit:      float
    K:           float
    iv:          float
    entry_px:    float
    exit_px:     float
    ret:         float
    pnl:         float
    exit_reason: str
    portfolio:   float
    regime:      str

# ── MAIN BACKTEST ─────────────────────────────────────────────────────────────
def run_backtest(data, capital, start, end):
    portfolio  = capital
    trades     = []
    open_pos   = []   # list of OptionPos
    peak       = capital

    qqq = data["QQQ"]
    all_dates = sorted(qqq["date"].tolist())
    all_dates = [d for d in all_dates
                 if str(d) >= start and str(d) <= end]

    # Track equity daily
    equity_curve = []

    r = 0.045  # risk-free rate

    for i, today in enumerate(all_dates):

        # ── Get QQQ history up to today ────────────────────────────────────
        qqq_hist = qqq[qqq["date"] <= today].tail(SMA_WINDOW + 30)
        if len(qqq_hist) < SMA_WINDOW:
            equity_curve.append((today, portfolio))
            continue

        regime = get_regime(qqq_hist)

        # ── Check open positions — close if expired or roll threshold ──────
        still_open = []
        for pos in open_pos:
            days_left = (pd.to_datetime(pos.expiry_date) -
                         pd.to_datetime(today)).days

            # Get current price
            stock_hist = data.get(pos.ticker,
                         data.get("AAPL"))  # fallback
            row = stock_hist[stock_hist["date"] == today]
            if row.empty:
                still_open.append(pos)
                continue

            S_now = float(row["close"].iloc[0])
            hist  = stock_hist[stock_hist["date"] <= today]
            iv_now = estimate_iv(hist["close"].values)
            T_now  = max(days_left / TRADING_YEAR, 0)
            is_call = pos.direction == "call"
            exit_px = bsm_price(S_now, pos.K, T_now, r, iv_now, is_call)
            exit_px = max(exit_px, 0.01)

            # Close if: expired, roll threshold, or stop (-80% loss)
            ret_pct = (exit_px - pos.entry_px) / pos.entry_px
            reason  = None

            if days_left <= 0:
                reason = "EXPIRED"
            elif days_left <= ROLL_DAYS_LEFT:
                reason = "ROLL"
            elif ret_pct <= -0.80:
                reason = "STOP"

            if reason:
                pnl = (exit_px - pos.entry_px) * 100 * pos.contracts
                portfolio += pnl
                trades.append(TradeResult(
                    ticker=pos.ticker, direction=pos.direction,
                    entry_date=pos.entry_date, exit_date=today,
                    expiry_date=pos.expiry_date,
                    S_entry=pos.S_entry, S_exit=S_now,
                    K=pos.K, iv=pos.iv,
                    entry_px=pos.entry_px, exit_px=exit_px,
                    ret=ret_pct, pnl=pnl,
                    exit_reason=reason,
                    portfolio=portfolio,
                    regime=regime,
                ))
            else:
                still_open.append(pos)

        open_pos = still_open

        # ── Open new positions every Monday (or first day of week) ─────────
        is_monday = (i == 0 or
                     pd.to_datetime(today).weekday() == 0 or
                     pd.to_datetime(today) > pd.to_datetime(all_dates[i-1]) + pd.Timedelta(days=3))

        if not is_monday:
            equity_curve.append((today, portfolio))
            continue

        if regime == "neutral":
            equity_curve.append((today, portfolio))
            continue

        # Drawdown check
        if portfolio > peak:
            peak = portfolio
        dd = (peak - portfolio) / peak
        if dd > 0.30:
            log.warning(f"{today}: halted — drawdown {dd:.1%}")
            break

        active_tickers = get_top6(today)
        direction = "call" if regime == "bull" else "put"
        is_call   = direction == "call"

        for ticker in active_tickers:
            if ticker not in data:
                continue

            # Skip if already have open position in this ticker+direction
            already_open = any(p.ticker == ticker and p.direction == direction
                               for p in open_pos)
            if already_open:
                continue

            stock_hist = data[ticker]
            row = stock_hist[stock_hist["date"] == today]
            if row.empty:
                continue

            S = float(row["close"].iloc[0])
            hist = stock_hist[stock_hist["date"] <= today]
            if len(hist) < 22:
                continue

            iv    = estimate_iv(hist["close"].values)
            K     = S * (1 + OTM_CALL_PCT) if is_call else S * (1 - OTM_PUT_PCT)
            T     = (OPTION_WEEKS * 5) / TRADING_YEAR  # ~6 weeks
            px    = bsm_price(S, K, T, r, iv, is_call)
            px    = max(px, 0.01)

            # Size: risk RISK_PER_NAME of portfolio
            risk_usd   = portfolio * RISK_PER_NAME
            cost_per_c = px * 100          # one contract = 100 shares
            contracts  = max(1, int(risk_usd / cost_per_c))
            total_cost = contracts * cost_per_c

            if total_cost > portfolio * 0.25:   # cap any single name at 25%
                contracts  = max(1, int(portfolio * 0.25 / cost_per_c))
                total_cost = contracts * cost_per_c

            # Deduct premium upfront
            portfolio -= total_cost

            expiry = pd.to_datetime(today) + pd.Timedelta(weeks=OPTION_WEEKS)

            open_pos.append(OptionPos(
                ticker=ticker, direction=direction,
                entry_date=today, expiry_date=expiry.date(),
                S_entry=S, K=K, iv=iv,
                entry_px=px, contracts=contracts,
                cost=total_cost,
                portfolio_at_entry=portfolio,
            ))

            log.info(f"  {today}  OPEN  {ticker:<5} {direction.upper():<4} "
                     f"S={S:.1f}  K={K:.1f}  T={OPTION_WEEKS}wk  "
                     f"IV={iv:.0%}  px={px:.2f}  "
                     f"#{contracts}  cost=${total_cost:,.0f}  "
                     f"regime={regime}")

        equity_curve.append((today, portfolio))

    # Close all remaining positions at last date
    last_date = all_dates[-1] if all_dates else today
    for pos in open_pos:
        stock_hist = data.get(pos.ticker, data.get("AAPL"))
        row = stock_hist[stock_hist["date"] == last_date]
        if row.empty:
            continue
        S_now  = float(row["close"].iloc[0])
        hist   = stock_hist[stock_hist["date"] <= last_date]
        iv_now = estimate_iv(hist["close"].values)
        days_left = (pd.to_datetime(pos.expiry_date) -
                     pd.to_datetime(last_date)).days
        T_now  = max(days_left / TRADING_YEAR, 0)
        is_call = pos.direction == "call"
        exit_px = bsm_price(S_now, pos.K, T_now, r, iv_now, is_call)
        exit_px = max(exit_px, 0.01)
        ret_pct = (exit_px - pos.entry_px) / pos.entry_px
        pnl     = (exit_px - pos.entry_px) * 100 * pos.contracts
        portfolio += pnl
        trades.append(TradeResult(
            ticker=pos.ticker, direction=pos.direction,
            entry_date=pos.entry_date, exit_date=last_date,
            expiry_date=pos.expiry_date,
            S_entry=pos.S_entry, S_exit=S_now,
            K=pos.K, iv=pos.iv,
            entry_px=pos.entry_px, exit_px=exit_px,
            ret=ret_pct, pnl=pnl,
            exit_reason="FINAL",
            portfolio=portfolio,
            regime="N/A",
        ))

    return trades, equity_curve

# ── REPORTING ─────────────────────────────────────────────────────────────────
def report(trades, equity_curve, capital, start, end):
    if not trades:
        print("No trades.")
        return {}

    n_yrs  = (pd.to_datetime(end) - pd.to_datetime(start)).days / 365.25
    final  = trades[-1].portfolio if trades else capital
    cagr   = (final / capital) ** (1 / n_yrs) - 1

    # Drawdown from equity curve
    ports  = [v for _, v in equity_curve]
    peaks  = np.maximum.accumulate(ports)
    dds    = (peaks - ports) / peaks
    max_dd = float(dds.max()) if len(dds) else 0

    rets   = np.array([t.ret for t in trades])
    wins   = rets[rets > 0]
    losses = rets[rets <= 0]
    wr     = float((rets > 0).mean())
    pf     = float(wins.sum() / abs(losses.sum())) if len(losses) > 0 else np.inf

    # Daily portfolio returns for Sharpe
    port_vals = [v for _, v in equity_curve if v > 0]
    if len(port_vals) > 2:
        daily_rets = np.diff(port_vals) / port_vals[:-1]
        sharpe = (daily_rets.mean() / daily_rets.std() * np.sqrt(252)
                  if daily_rets.std() > 0 else 0)
    else:
        sharpe = 0

    # Per-ticker breakdown
    by_ticker = {}
    for t in trades:
        by_ticker.setdefault(t.ticker, []).append(t.ret)

    # Per-direction breakdown
    by_dir = {}
    for t in trades:
        by_dir.setdefault(t.direction, []).append(t.ret)

    print()
    print("=" * 65)
    print("APEX-6W  —  ROLLING 6-WEEK OPTIONS ON TOP QQQ NAMES")
    print("=" * 65)
    print(f"Period:      {start} → {end}  ({n_yrs:.1f} yrs)")
    print(f"Capital:     ${capital:,.0f}  →  ${final:,.0f}")
    print(f"CAGR:        {cagr:+.1%}   Max DD: {max_dd:.1%}   Sharpe: {sharpe:.2f}")
    print(f"Trades:      {len(trades)}  ({len(trades)/n_yrs:.0f}/yr)   "
          f"WR: {wr:.1%}   PF: {pf:.2f}")
    print(f"Avg win:     {wins.mean():+.1%}" if len(wins) else "Avg win:  N/A")
    print(f"Avg loss:    {losses.mean():+.1%}" if len(losses) else "Avg loss: N/A")
    print(f"Exits:  " +
          "  ".join(f"{r}={sum(1 for t in trades if t.exit_reason==r)}"
                    for r in ["ROLL","EXPIRED","STOP","FINAL"]))
    print()

    print("Per-ticker breakdown:")
    print(f"  {'Ticker':<6}  {'N':>4}  {'WR':>6}  {'AvgRet':>8}  {'TotalPnL':>10}")
    print(f"  {'─'*6}  {'─'*4}  {'─'*6}  {'─'*8}  {'─'*10}")
    for ticker, tr in sorted(by_ticker.items()):
        arr = np.array(tr)
        pnl = sum(t.pnl for t in trades if t.ticker == ticker)
        print(f"  {ticker:<6}  {len(arr):>4}  {(arr>0).mean():>6.1%}  "
              f"{arr.mean():>+8.1%}  ${pnl:>+10,.0f}")
    print()

    print("Direction breakdown:")
    for d, dr in by_dir.items():
        arr = np.array(dr)
        pnl = sum(t.pnl for t in trades if t.direction == d)
        print(f"  {d.upper():<4}  N={len(arr)}  WR={( arr>0).mean():.1%}  "
              f"AvgRet={arr.mean():+.1%}  PnL=${pnl:+,.0f}")
    print("=" * 65)

    # Monte Carlo
    rng   = np.random.default_rng(42)
    n_sim = 100_000
    tr_per_yr = len(trades) / n_yrs
    n_draw    = max(int(round(tr_per_yr * n_yrs)), 1)
    avg_risk  = np.mean([abs(t.pnl) / max(t.portfolio, 1) for t in trades])
    samp      = rng.integers(0, len(rets), (n_sim, n_draw))
    path_rets = rets[samp]
    mults     = 1.0 + avg_risk * path_rets
    cum       = np.cumprod(mults, axis=1)
    paths     = capital * cum
    terminal  = paths[:, -1]
    cagrs_mc  = (terminal / capital) ** (1 / n_yrs) - 1
    peaks_mc  = np.maximum.accumulate(np.concatenate(
                    [np.full((n_sim, 1), capital), paths], axis=1), axis=1)
    dds_mc    = ((peaks_mc - np.concatenate(
                    [np.full((n_sim, 1), capital), paths], axis=1)) /
                  peaks_mc).max(axis=1)
    ruin      = (paths < capital * 0.5).any(axis=1).mean()

    pcts = [5, 25, 50, 75, 95]
    print()
    print("=" * 65)
    print(f"MONTE CARLO  ({n_sim:,} paths  ×  {n_draw} trades)")
    print("=" * 65)
    print(f"  {'Metric':<22}  {'P5':>8}  {'P25':>8}  {'P50':>8}  "
          f"{'P75':>8}  {'P95':>8}")
    print(f"  {'─'*22}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}")
    print(f"  {'CAGR':<22}  " +
          "  ".join(f"{v:>+7.1%}" for v in np.percentile(cagrs_mc, pcts)))
    print(f"  {'Max Drawdown':<22}  " +
          "  ".join(f"{v:>7.1%}" for v in np.percentile(dds_mc, pcts)))
    print(f"  {'Terminal':<22}  " +
          "  ".join(f"${v/1000:>6.0f}K" for v in np.percentile(terminal, pcts)))
    print(f"  {'Ruin P(port<50%)':<22}  {ruin:>8.2%}")
    print(f"  {'Profitable paths':<22}  {(cagrs_mc>0).mean():>8.2%}")
    print("=" * 65)

    return {"cagr": cagr, "max_dd": max_dd, "sharpe": sharpe,
            "n_trades": len(trades), "wr": wr,
            "equity_curve": equity_curve, "paths_mc": paths,
            "cagrs_mc": cagrs_mc}

# ── PLOT ──────────────────────────────────────────────────────────────────────
def plot(trades, equity_curve, stats_out, capital, out="results/apex_6w_dashboard.png"):
    try:
        BG=("#0d1117"); PAN=("#161b22"); GR=("#22c55e"); RD=("#ef4444")
        BL=("#3b82f6"); YL=("#eab308"); OR=("#f97316"); MU=("#94a3b8"); WH=("#f1f5f9")

        fig = plt.figure(figsize=(20, 12), facecolor=BG)
        fig.suptitle("APEX-6W — Rolling 6-Week Options on Top QQQ Names",
                     color=WH, fontsize=14, fontweight="bold", y=0.98)
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.32)

        def style(ax, title):
            ax.set_facecolor(PAN)
            for sp in ax.spines.values(): sp.set_color("#2d3748")
            ax.tick_params(colors=MU, labelsize=8)
            ax.set_title(title, color=WH, fontsize=9, fontweight="bold", pad=6)
            ax.grid(True, color="#2d3748", lw=0.4, alpha=0.5)

        # 1. Equity curve
        ax1 = fig.add_subplot(gs[0, :2])
        style(ax1, f"Equity Curve  CAGR={stats_out.get('cagr',0):+.1%}  "
                   f"Sharpe={stats_out.get('sharpe',0):.2f}")
        dates_eq = [pd.to_datetime(d) for d, _ in equity_curve]
        ports_eq = [v for _, v in equity_curve]
        ax1.plot(dates_eq, ports_eq, color=GR, lw=2)
        ax1.axhline(capital, color=MU, lw=0.8, ls="--", alpha=0.5)
        ax1.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
        ax1.fill_between(dates_eq, capital, ports_eq,
                         where=[p >= capital for p in ports_eq],
                         color=GR, alpha=0.10)
        ax1.fill_between(dates_eq, capital, ports_eq,
                         where=[p < capital for p in ports_eq],
                         color=RD, alpha=0.10)

        # 2. Per-ticker PnL
        ax2 = fig.add_subplot(gs[0, 2])
        style(ax2, "P&L by Ticker")
        by_t = {}
        for t in trades:
            by_t[t.ticker] = by_t.get(t.ticker, 0) + t.pnl
        tickers_sorted = sorted(by_t, key=lambda x: by_t[x])
        vals   = [by_t[t] for t in tickers_sorted]
        colors = [GR if v > 0 else RD for v in vals]
        ax2.barh(tickers_sorted, vals, color=colors, alpha=0.85, edgecolor=BG)
        ax2.axvline(0, color=MU, lw=0.8)
        ax2.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))

        # 3. Return distribution
        ax3 = fig.add_subplot(gs[1, 0])
        style(ax3, f"Return Distribution  WR={stats_out.get('wr',0):.1%}")
        rets = [t.ret * 100 for t in trades]
        wins_ = [r for r in rets if r > 0]
        loss_ = [r for r in rets if r <= 0]
        bins  = np.linspace(-100, 300, 50)
        if loss_: ax3.hist(loss_, bins=bins, color=RD, alpha=0.7, label=f"Loss ({len(loss_)})")
        if wins_: ax3.hist(wins_, bins=bins, color=GR, alpha=0.7, label=f"Win  ({len(wins_)})")
        ax3.axvline(0, color=MU, lw=0.8, ls="--")
        ax3.legend(fontsize=7, facecolor=PAN, labelcolor=WH, edgecolor="#2d3748")
        ax3.set_xlabel("Option Return %", color=MU, fontsize=8)

        # 4. Regime split
        ax4 = fig.add_subplot(gs[1, 1])
        style(ax4, "Win Rate by Regime & Direction")
        regimes = ["bull", "bear"]
        directions = ["call", "put"]
        x = np.arange(len(regimes))
        width = 0.35
        for j, (d, c) in enumerate(zip(directions, [BL, OR])):
            wrs = []
            for reg in regimes:
                sub = [t.ret for t in trades
                       if t.regime == reg and t.direction == d]
                wrs.append((np.mean([r > 0 for r in sub])
                            if sub else 0))
            ax4.bar(x + j*width, wrs, width, color=c,
                    alpha=0.85, label=d.upper(), edgecolor=BG)
        ax4.axhline(0.5, color=WH, lw=0.8, ls="--", alpha=0.5)
        ax4.set_xticks(x + width/2)
        ax4.set_xticklabels(regimes, color=MU)
        ax4.set_ylabel("Win Rate", color=MU, fontsize=8)
        ax4.legend(fontsize=7, facecolor=PAN, labelcolor=WH, edgecolor="#2d3748")
        ax4.set_ylim(0, 1)

        # 5. MC paths
        ax5 = fig.add_subplot(gs[1, 2])
        paths_mc = stats_out.get("paths_mc")
        cagrs_mc = stats_out.get("cagrs_mc", np.array([0]))
        if paths_mc is not None:
            style(ax5, f"Monte Carlo  P50 CAGR={np.percentile(cagrs_mc,50):+.1%}")
            n_plot = min(300, paths_mc.shape[0])
            x_mc   = np.arange(1, paths_mc.shape[1]+1)
            for i in range(n_plot):
                ax5.plot(x_mc, paths_mc[i], color=BL, alpha=0.015, lw=0.4)
            for p, c in zip([5,25,50,75,95], [RD,OR,YL,OR,RD]):
                pv = np.percentile(paths_mc, p, axis=0)
                lw = 2.0 if p == 50 else 1.0
                ax5.plot(x_mc, pv, color=c, lw=lw,
                         label=f"P{p}" if p in [5,50,95] else "")
            ax5.axhline(capital, color=MU, lw=0.8, ls="--", alpha=0.5)
            ax5.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda x_, _: f"${x_/1000:.0f}K"))
            ax5.set_xlabel("Trade #", color=MU, fontsize=8)
            ax5.legend(fontsize=7, facecolor=PAN,
                       labelcolor=WH, edgecolor="#2d3748")

        os.makedirs(os.path.dirname(out) if os.path.dirname(out) else ".", exist_ok=True)
        plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
        plt.close()
        log.info(f"Dashboard → {out}")
    except Exception as e:
        log.warning(f"Plot failed: {e}")

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--start",   default="2018-01-01")
    p.add_argument("--end",     default="2025-01-01")
    p.add_argument("--capital", type=float, default=100_000)
    p.add_argument("--out",     default="results/apex_6w")
    args = p.parse_args()

    os.makedirs("results", exist_ok=True)

    all_tickers = list(set(["QQQ"] + [t for tickers in TOP_6_BY_ERA.values()
                                       for t in tickers]))
    log.info(f"Loading data for: {all_tickers}")
    data = load_data(all_tickers, args.start, args.end)

    log.info(f"Running APEX-6W backtest: {args.start} → {args.end}  "
             f"capital=${args.capital:,.0f}")
    log.info(f"Strategy: 6-week {OTM_CALL_PCT:.0%} OTM calls (bull) / "
             f"puts (bear) on top-6 QQQ names")
    log.info(f"Roll threshold: {ROLL_DAYS_LEFT} days left  |  "
             f"Stop: -80% option return")

    trades, equity_curve = run_backtest(data, args.capital, args.start, args.end)
    stats_out = report(trades, equity_curve, args.capital, args.start, args.end)
    plot(trades, equity_curve, stats_out, args.capital,
         out=f"{args.out}_dashboard.png")

    # Save trades CSV
    rows = [{"date": t.entry_date, "exit": t.exit_date, "ticker": t.ticker,
             "dir": t.direction, "regime": t.regime,
             "S_entry": f"{t.S_entry:.2f}", "K": f"{t.K:.2f}",
             "entry_px": f"{t.entry_px:.2f}", "exit_px": f"{t.exit_px:.2f}",
             "ret_pct": f"{t.ret*100:.1f}", "pnl": f"{t.pnl:.0f}",
             "reason": t.exit_reason, "portfolio": f"{t.portfolio:.0f}"}
            for t in trades]
    pd.DataFrame(rows).to_csv(f"{args.out}_trades.csv", index=False)
    log.info(f"Trades → {args.out}_trades.csv")

if __name__ == "__main__":
    main()
