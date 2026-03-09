"""
APEX-SPY-AGG: Aggressive Risk Reversal on SPY
==============================================
BULL:  SELL ATM put (max gamma/theta)  + BUY 3% OTM call (cheap lottery)
BEAR:  SELL ATM call                   + BUY 3% OTM put

Aggressive vs standard:
  - SPY only (single liquid underlier, tightest spreads)
  - 3% OTM long leg (vs 5%) — more delta, costs more but wins bigger
  - 4% portfolio risk per position (vs 2%)
  - Roll at 14 days (vs 7) — capture more theta decay before rolling
  - Stop: short leg 2.5x (vs 2x) — let winners breathe
  - Add 2nd position if conviction strong (regime + VIX filter)
  - Weekly entry check (not just Monday)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import norm
from dataclasses import dataclass
import warnings, os, logging
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# ── CONFIG ────────────────────────────────────────────────────────────────────
TICKER         = "SPY"
OPTION_WEEKS   = 6
ROLL_DAYS      = 14        # roll at 2 weeks remaining (more theta captured)
OTM_PCT        = 0.03      # 3% OTM long leg (tighter = more delta)
SMA_FAST       = 20        # fast regime filter
SMA_SLOW       = 50        # slow regime filter
RISK_PCT       = 0.04      # 4% per position (aggressive)
MAX_POSITIONS  = 3         # up to 3 concurrent positions
SHORT_STOP_X   = 2.5       # stop if short leg 2.5x entry
R              = 0.045
TRADING_YEAR   = 252.0
IV_SKEW_PUT    = 1.10      # ATM put 10% richer than HV (real market skew)
IV_SKEW_WING   = 0.95      # OTM wings slightly cheaper

# ── BSM ───────────────────────────────────────────────────────────────────────
def bsm(S, K, T, iv, is_call):
    if T <= 1e-6 or iv <= 0:
        return max(S-K,0) if is_call else max(K-S,0)
    d1 = (np.log(S/K)+(R+.5*iv**2)*T)/(iv*np.sqrt(T))
    d2 = d1 - iv*np.sqrt(T)
    if is_call: return S*norm.cdf(d1) - K*np.exp(-R*T)*norm.cdf(d2)
    return K*np.exp(-R*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

def bsm_gamma(S, K, T, iv):
    if T<=1e-6 or iv<=0: return 0
    d1=(np.log(S/K)+(R+.5*iv**2)*T)/(iv*np.sqrt(T))
    return norm.pdf(d1)/(S*iv*np.sqrt(T))

# ── DATA ──────────────────────────────────────────────────────────────────────
def load_data(start, end):
    api_key    = os.environ.get("ALPACA_API_KEY","")
    api_secret = os.environ.get("ALPACA_API_SECRET","")
    if api_key and api_secret:
        try:
            import alpaca_trade_api as tradeapi
            api = tradeapi.REST(api_key, api_secret,
                                base_url="https://paper-api.alpaca.markets")
            raw = api.get_bars(TICKER,"1Day",start=start,end=end,
                               adjustment="all").df.reset_index()
            raw.columns = [c.lower() for c in raw.columns]
            tc = next((c for c in raw.columns if c in ("timestamp","time","t")),None)
            if tc: raw["date"] = pd.to_datetime(raw[tc]).dt.date
            raw = raw.sort_values("date").reset_index(drop=True)
            log.info(f"  SPY: {len(raw)} days (LIVE)")
            return raw, "live"
        except Exception as e:
            log.warning(f"Alpaca: {e}")

    # Synthetic SPY — calibrated to real SPY stats
    # SPY: vol~16%, drift~12%/yr, fat tails, vol clustering
    log.info("SYNTHETIC SPY — set ALPACA keys for real data")
    np.random.seed(42)
    n = int((pd.to_datetime(end)-pd.to_datetime(start)).days*252/365)
    dates = pd.bdate_range(start=start, periods=n)

    # GBM with vol clustering (GARCH-like)
    vol = 0.16/np.sqrt(252)
    drift = 0.12/252
    rets = []
    v = vol
    for _ in range(n):
        v = np.clip(0.94*v + 0.06*abs(np.random.normal(0,vol)), vol*0.3, vol*4)
        rets.append(np.random.normal(drift, v))

    rets = np.array(rets)
    # Add crash events (2018 Q4, 2020 Mar, 2022)
    crash_days = {int(n*0.12): -0.032, int(n*0.12)+1: -0.028,  # Q4 2018
                  int(n*0.27): -0.095, int(n*0.27)+1: -0.120,  # COVID crash
                  int(n*0.55): -0.025, int(n*0.55)+5: -0.030,  # 2022
                  int(n*0.60): -0.022}
    for day, ret in crash_days.items():
        if day < n: rets[day] += ret

    px = 280 * np.cumprod(1+rets)
    df = pd.DataFrame({
        "date": [d.date() for d in dates],
        "open": px*(1+np.random.normal(0,.001,n)),
        "high": px*(1+abs(np.random.normal(0,.005,n))),
        "low":  px*(1-abs(np.random.normal(0,.005,n))),
        "close": px, "volume": 80_000_000
    })
    return df, "synthetic"

# ── HELPERS ───────────────────────────────────────────────────────────────────
def get_regime(hist):
    """Dual SMA regime: strong bull, weak bull, neutral, weak bear, strong bear."""
    if len(hist) < SMA_SLOW: return "neutral", 0
    c = hist["close"].values
    sma20 = c[-SMA_FAST:].mean()
    sma50 = c[-SMA_SLOW:].mean()
    last  = c[-1]
    # Momentum score -2 to +2
    score = 0
    if last > sma20: score += 1
    if last > sma50: score += 1
    if sma20 > sma50: score += 0  # trend confirmation (no extra weight)
    if last < sma20: score -= 1
    if last < sma50: score -= 1
    if score >= 2:    return "strong_bull", score
    if score == 1:    return "weak_bull",   score
    if score == -1:   return "weak_bear",   score
    if score <= -2:   return "strong_bear", score
    return "neutral", score

def est_iv(closes, window=21):
    if len(closes) < window+1: return 0.16
    r = np.diff(np.log(closes[-window-1:]))
    hv = np.std(r,ddof=1)*np.sqrt(252)
    return float(np.clip(hv, .10, 1.50))

def n_positions_target(regime):
    """Aggressive: scale up in strong regimes."""
    return {"strong_bull":3,"weak_bull":2,"neutral":0,
            "weak_bear":2,"strong_bear":3}.get(regime,0)

# ── POSITION ──────────────────────────────────────────────────────────────────
@dataclass
class Pos:
    regime: str; entry_date: object; expiry_date: object
    S_entry: float
    K_short: float; short_is_call: bool; short_entry_px: float
    K_otm: float;   long_is_call: bool;  long_entry_px: float
    net_credit: float; contracts: int; iv: float; batch: int

@dataclass
class Trade:
    regime: str; entry_date: object; exit_date: object
    S_entry: float; S_exit: float
    K_short: float; K_otm: float
    iv: float; net_credit: float; exit_net: float
    ret: float; pnl: float; exit_reason: str; portfolio: float; batch: int

# ── BACKTEST ──────────────────────────────────────────────────────────────────
def run(df, capital, start, end):
    portfolio = capital; peak = capital
    trades = []; open_pos = []; equity = []
    batch_ctr = 0

    all_dates = sorted(d for d in df["date"].tolist() if start<=str(d)<=end)

    for i, today in enumerate(all_dates):
        hist = df[df["date"]<=today].tail(SMA_SLOW+10)

        # ── Manage open positions ──────────────────────────────────────────
        still_open = []
        for pos in open_pos:
            days_left = (pd.to_datetime(pos.expiry_date)-pd.to_datetime(today)).days
            row = df[df["date"]==today]
            if row.empty: still_open.append(pos); continue

            S     = float(row["close"].iloc[0])
            iv    = est_iv(df[df["date"]<=today]["close"].values)
            T     = max(days_left/TRADING_YEAR, 0)

            # Skew-adjusted IV per leg
            iv_short = iv*IV_SKEW_PUT if not pos.short_is_call else iv
            iv_long  = iv*IV_SKEW_WING

            short_px = bsm(S,pos.K_short,T,iv_short,pos.short_is_call)
            long_px  = bsm(S,pos.K_otm,  T,iv_long, pos.long_is_call)
            exit_net = short_px - long_px

            reason = None
            if days_left <= 0:              reason = "EXPIRED"
            elif days_left <= ROLL_DAYS:    reason = "ROLL"
            elif short_px >= pos.short_entry_px * SHORT_STOP_X: reason = "STOP"

            if reason:
                pnl_ps  = pos.net_credit - exit_net
                pnl     = pnl_ps * 100 * pos.contracts
                portfolio += pnl
                basis    = abs(pos.net_credit) if abs(pos.net_credit)>0.01 \
                           else pos.short_entry_px
                ret      = pnl_ps / basis
                trades.append(Trade(
                    regime=pos.regime, entry_date=pos.entry_date,
                    exit_date=today, S_entry=pos.S_entry, S_exit=S,
                    K_short=pos.K_short, K_otm=pos.K_otm,
                    iv=pos.iv, net_credit=pos.net_credit, exit_net=exit_net,
                    ret=ret, pnl=pnl, exit_reason=reason,
                    portfolio=portfolio, batch=pos.batch,
                ))
                log.info(f"  {today} CLOSE [{pos.regime:<11}] {reason:<8} "
                         f"S={S:.1f}  short={short_px:.2f}(x{short_px/pos.short_entry_px:.1f})  "
                         f"ret={ret:+.0%}  pnl=${pnl:+,.0f}  port=${portfolio:,.0f}")
            else:
                still_open.append(pos)
        open_pos = still_open

        if portfolio > peak: peak = portfolio
        dd = (peak-portfolio)/peak
        if dd > 0.40:
            log.warning(f"{today}: HALTED dd={dd:.1%}")
            break

        equity.append((today, portfolio))

        # ── Open new positions — every Monday or Thursday ──────────────────
        # Twice-weekly entry = more opportunities captured
        dow = pd.to_datetime(today).weekday()
        if dow not in (0, 3): continue   # Monday=0, Thursday=3

        regime, score = get_regime(hist)
        if regime == "neutral": continue

        target_n = n_positions_target(regime)
        current_n = len(open_pos)
        to_open   = max(0, target_n - current_n)
        if to_open == 0: continue

        row = df[df["date"]==today]
        if row.empty: continue
        S   = float(row["close"].iloc[0])
        iv  = est_iv(df[df["date"]<=today]["close"].values)

        is_bull = "bull" in regime
        T       = OPTION_WEEKS*5/TRADING_YEAR

        for _ in range(to_open):
            if "bull" in regime:
                # SELL ATM put + BUY 3% OTM call
                K_short     = S
                short_is_c  = False
                K_otm       = S*(1+OTM_PCT)
                long_is_c   = True
                short_px    = bsm(S,K_short,T,iv*IV_SKEW_PUT, False)
                long_px     = bsm(S,K_otm,  T,iv*IV_SKEW_WING,True)
            else:
                # SELL ATM call + BUY 3% OTM put
                K_short     = S
                short_is_c  = True
                K_otm       = S*(1-OTM_PCT)
                long_is_c   = False
                short_px    = bsm(S,K_short,T,iv,              True)
                long_px     = bsm(S,K_otm,  T,iv*IV_SKEW_WING, False)

            net_credit = short_px - long_px

            # Size: risk_pct × portfolio = approx max loss
            # max loss ≈ short_px × 2 (short blows to 2× before stop)
            max_loss_est = short_px * SHORT_STOP_X * 100
            contracts    = max(1, int((portfolio*RISK_PCT)/max_loss_est))
            contracts    = min(contracts, int(portfolio*0.20/(short_px*100)))

            cash_flow    = net_credit * 100 * contracts
            portfolio   += cash_flow

            batch_ctr   += 1
            expiry = (pd.to_datetime(today)+pd.Timedelta(weeks=OPTION_WEEKS)).date()
            open_pos.append(Pos(
                regime=regime, entry_date=today, expiry_date=expiry,
                S_entry=S, K_short=K_short, short_is_call=short_is_c,
                short_entry_px=short_px, K_otm=K_otm,
                long_is_call=long_is_c, long_entry_px=long_px,
                net_credit=net_credit, contracts=contracts,
                iv=iv, batch=batch_ctr,
            ))
            action = "SELL PUT +BUY CALL" if is_bull else "SELL CALL+BUY PUT"
            log.info(f"  {today} OPEN  [{regime:<11}] {action}  "
                     f"S={S:.1f} K_s={K_short:.1f}@{short_px:.2f} "
                     f"K_l={K_otm:.1f}@{long_px:.2f} "
                     f"CR={net_credit:.2f}  #{contracts}  "
                     f"cf=${cash_flow:+,.0f}  port=${portfolio:,.0f}")

    # Close all remaining
    if all_dates:
        last = all_dates[-1]
        for pos in open_pos:
            row = df[df["date"]==last]
            if row.empty: continue
            S   = float(row["close"].iloc[0])
            iv  = est_iv(df[df["date"]<=last]["close"].values)
            dl  = (pd.to_datetime(pos.expiry_date)-pd.to_datetime(last)).days
            T   = max(dl/TRADING_YEAR,0)
            sp  = bsm(S,pos.K_short,T,iv*IV_SKEW_PUT if not pos.short_is_call else iv,pos.short_is_call)
            lp  = bsm(S,pos.K_otm,  T,iv*IV_SKEW_WING,pos.long_is_call)
            en  = sp-lp; pps=pos.net_credit-en
            pnl = pps*100*pos.contracts; portfolio+=pnl
            basis=abs(pos.net_credit) if abs(pos.net_credit)>0.01 else pos.short_entry_px
            trades.append(Trade(
                regime=pos.regime,entry_date=pos.entry_date,exit_date=last,
                S_entry=pos.S_entry,S_exit=S,K_short=pos.K_short,K_otm=pos.K_otm,
                iv=pos.iv,net_credit=pos.net_credit,exit_net=en,
                ret=pps/basis,pnl=pnl,exit_reason="FINAL",
                portfolio=portfolio,batch=pos.batch,
            ))
        equity.append((last,portfolio))
    return trades, equity

# ── REPORT ────────────────────────────────────────────────────────────────────
def report(trades, equity, capital, start, end):
    if not trades: print("No trades."); return {}
    n_yrs = (pd.to_datetime(end)-pd.to_datetime(start)).days/365.25
    final = trades[-1].portfolio
    cagr  = (final/capital)**(1/n_yrs)-1
    ports = [v for _,v in equity]
    peaks = np.maximum.accumulate(ports)
    max_dd= float(((peaks-ports)/peaks).max()) if ports else 0
    dr    = np.diff(ports)/np.array(ports[:-1])
    sharpe= dr.mean()/dr.std()*np.sqrt(252) if dr.std()>0 else 0
    rets  = np.array([t.ret for t in trades])
    wins  = rets[rets>0]; losses=rets[rets<=0]
    wr    = (rets>0).mean()
    pf    = wins.sum()/abs(losses.sum()) if len(losses) else np.inf
    avg_credit = np.mean([t.net_credit for t in trades])

    print()
    print("="*70)
    print("APEX-SPY-AGG  —  AGGRESSIVE RISK REVERSALS ON SPY")
    print("SELL ATM put + BUY 3% OTM call  (bull)")
    print("SELL ATM call + BUY 3% OTM put  (bear)")
    print("="*70)
    print(f"Period:      {start} → {end}  ({n_yrs:.1f} yrs)")
    print(f"Capital:     ${capital:,.0f}  →  ${final:,.0f}")
    print(f"CAGR:        {cagr:+.1%}   Max DD: {max_dd:.1%}   Sharpe: {sharpe:.2f}")
    print(f"Trades:      {len(trades)}  ({len(trades)/n_yrs:.0f}/yr)   "
          f"WR: {wr:.1%}   PF: {pf:.2f}")
    if len(wins):   print(f"Avg win:     {wins.mean():+.1%}")
    if len(losses): print(f"Avg loss:    {losses.mean():+.1%}")
    print(f"Avg credit:  ${avg_credit:.2f}/share per contract")
    exits=["ROLL","EXPIRED","STOP","FINAL"]
    print("Exits:  "+"  ".join(f"{r}={sum(1 for t in trades if t.exit_reason==r)}"
                               for r in exits))
    print()

    for reg in ["strong_bull","weak_bull","weak_bear","strong_bear"]:
        sub=[t for t in trades if t.regime==reg]
        if not sub: continue
        arr=np.array([t.ret for t in sub])
        pnl=sum(t.pnl for t in sub)
        print(f"  {reg:<12}  N={len(sub):>3}  WR={(arr>0).mean():.1%}  "
              f"AvgRet={arr.mean():+.1%}  PnL=${pnl:+,.0f}")

    # Monthly P&L heatmap data
    print()
    monthly = {}
    for t in trades:
        ym = str(t.exit_date)[:7]
        monthly[ym] = monthly.get(ym,0) + t.pnl
    yrs = sorted(set(k[:4] for k in monthly))
    print("  Monthly P&L ($):")
    months = ["Jan","Feb","Mar","Apr","May","Jun",
              "Jul","Aug","Sep","Oct","Nov","Dec"]
    for yr in yrs:
        row_str = f"  {yr}  "
        for m in range(1,13):
            key=f"{yr}-{m:02d}"
            v=monthly.get(key,0)
            row_str += f"{v/1000:>+6.1f}K " if v!=0 else "       "
        print(row_str)

    # MC
    print()
    rng=np.random.default_rng(42); n_sim=100_000
    n_draw=max(len(trades),1)
    samp=rng.integers(0,len(rets),(n_sim,n_draw))
    risk=np.mean([abs(t.pnl)/max(t.portfolio,1) for t in trades])
    mults=1+risk*rets[samp]
    paths=capital*np.cumprod(mults,axis=1)
    term=paths[:,-1]; cagrs_=(term/capital)**(1/n_yrs)-1
    pk_=np.maximum.accumulate(np.hstack([np.full((n_sim,1),capital),paths]),axis=1)
    dd_=((pk_-np.hstack([np.full((n_sim,1),capital),paths]))/pk_).max(axis=1)
    ruin=(paths<capital*0.5).any(axis=1).mean()
    pcts=[5,25,50,75,95]
    print("="*70)
    print(f"MONTE CARLO  (100,000 paths × {n_draw} trades)")
    print("="*70)
    print(f"  {'Metric':<22}  {'P5':>8}  {'P25':>8}  {'P50':>8}  {'P75':>8}  {'P95':>8}")
    print("  "+"-"*68)
    print(f"  {'CAGR':<22}  "+"  ".join(f"{v:>+7.1%}" for v in np.percentile(cagrs_,pcts)))
    print(f"  {'Max Drawdown':<22}  "+"  ".join(f"{v:>7.1%}" for v in np.percentile(dd_,pcts)))
    print(f"  {'Terminal':<22}  "+"  ".join(f"${v/1000:>6.0f}K" for v in np.percentile(term,pcts)))
    print(f"  {'Ruin P(port<50%)':<22}  {ruin:>8.2%}")
    print(f"  {'Profitable paths':<22}  {(cagrs_>0).mean():>8.2%}")
    print("="*70)

    return {"cagr":cagr,"max_dd":max_dd,"sharpe":sharpe,
            "n_trades":len(trades),"wr":wr,"equity":equity,
            "paths":paths,"cagrs":cagrs_,"rets":rets,"monthly":monthly}

# ── PLOT ──────────────────────────────────────────────────────────────────────
def plot(trades, equity, stats, capital, out):
    try:
        BG="#0d1117";PAN="#161b22";GR="#22c55e";RD="#ef4444"
        BL="#3b82f6";YL="#eab308";OR="#f97316";MU="#94a3b8";WH="#f1f5f9"
        fig=plt.figure(figsize=(22,13),facecolor=BG)
        fig.suptitle("APEX-SPY-AGG  —  Aggressive Risk Reversals on SPY\n"
                     "SELL ATM put + BUY 3% OTM call (bull)  |  "
                     "SELL ATM call + BUY 3% OTM put (bear)",
                     color=WH,fontsize=13,fontweight="bold",y=0.99)
        gs=gridspec.GridSpec(2,3,figure=fig,hspace=0.44,wspace=0.32)

        def sty(ax,title):
            ax.set_facecolor(PAN)
            for sp in ax.spines.values(): sp.set_color("#2d3748")
            ax.tick_params(colors=MU,labelsize=8)
            ax.set_title(title,color=WH,fontsize=9,fontweight="bold",pad=6)
            ax.grid(True,color="#2d3748",lw=0.4,alpha=0.5)

        # 1. Equity curve
        ax1=fig.add_subplot(gs[0,:2])
        sty(ax1,f"Equity  CAGR={stats.get('cagr',0):+.1%}  "
               f"MaxDD={stats.get('max_dd',0):.1%}  "
               f"Sharpe={stats.get('sharpe',0):.2f}  "
               f"WR={stats.get('wr',0):.1%}")
        dates_=[pd.to_datetime(d) for d,_ in equity]
        ports_=[v for _,v in equity]
        ax1.plot(dates_,ports_,color=GR,lw=2,zorder=3)
        ax1.axhline(capital,color=MU,lw=0.8,ls="--",alpha=0.5)
        ax1.fill_between(dates_,capital,ports_,
                         where=[p>=capital for p in ports_],color=GR,alpha=0.08)
        ax1.fill_between(dates_,capital,ports_,
                         where=[p<capital for p in ports_],color=RD,alpha=0.15)
        # Color-coded trade entries
        for t in trades:
            c=GR if "bull" in t.regime else RD
            ax1.axvline(pd.to_datetime(t.entry_date),color=c,lw=0.25,alpha=0.25)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_:f"${x/1000:.0f}K"))

        # 2. Monthly P&L heatmap
        ax2=fig.add_subplot(gs[0,2])
        sty(ax2,"Monthly P&L Heatmap")
        monthly=stats.get("monthly",{})
        if monthly:
            yrs=sorted(set(k[:4] for k in monthly))
            mat=np.zeros((len(yrs),12))
            for k,v in monthly.items():
                yr,mo=k.split("-"); yi=yrs.index(yr); mat[yi,int(mo)-1]=v
            vmax=np.percentile(np.abs(mat[mat!=0]),90) if (mat!=0).any() else 1
            im=ax2.imshow(mat,aspect="auto",cmap="RdYlGn",
                          vmin=-vmax,vmax=vmax,interpolation="nearest")
            ax2.set_yticks(range(len(yrs))); ax2.set_yticklabels(yrs,fontsize=7)
            ax2.set_xticks(range(12))
            ax2.set_xticklabels(["J","F","M","A","M","J",
                                  "J","A","S","O","N","D"],fontsize=7)
            for yi in range(len(yrs)):
                for mi in range(12):
                    v=mat[yi,mi]
                    if v!=0:
                        ax2.text(mi,yi,f"{v/1000:+.0f}",ha="center",
                                 va="center",fontsize=5.5,
                                 color="white" if abs(v)>vmax*0.5 else "black")

        # 3. Return distribution
        ax3=fig.add_subplot(gs[1,0])
        rets_=stats.get("rets",np.array([]))
        sty(ax3,f"Return Distribution  WR={stats.get('wr',0):.1%}")
        if len(rets_):
            bins=np.linspace(-300,500,60)
            w_=[r*100 for r in rets_ if r>0]
            l_=[r*100 for r in rets_ if r<=0]
            if l_: ax3.hist(l_,bins=bins,color=RD,alpha=0.75,label=f"Loss ({len(l_)})")
            if w_: ax3.hist(w_,bins=bins,color=GR,alpha=0.75,label=f"Win  ({len(w_)})")
            ax3.axvline(0,color=MU,lw=0.8,ls="--")
            ax3.legend(fontsize=7,facecolor=PAN,labelcolor=WH,edgecolor="#2d3748")
        ax3.set_xlabel("Return %",color=MU,fontsize=8)

        # 4. WR by regime
        ax4=fig.add_subplot(gs[1,1])
        sty(ax4,"P&L by Regime")
        regs=["strong_bull","weak_bull","weak_bear","strong_bear"]
        reg_pnl =[sum(t.pnl for t in trades if t.regime==r) for r in regs]
        reg_cols=[GR,GR if True else GR,OR,RD]
        reg_cols=[GR,"#16a34a",OR,RD]
        bars=ax4.bar([r.replace("_","\n") for r in regs],
                     reg_pnl,
                     color=reg_cols,alpha=0.85,edgecolor=BG)
        ax4.axhline(0,color=MU,lw=0.8)
        ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_:f"${x/1000:.0f}K"))
        for bar,r in zip(bars,regs):
            n=sum(1 for t in trades if t.regime==r)
            sub=[t.ret for t in trades if t.regime==r]
            wr_=np.mean([x>0 for x in sub]) if sub else 0
            ax4.text(bar.get_x()+bar.get_width()/2,
                     bar.get_height()+(max(reg_pnl)-min(reg_pnl))*0.03,
                     f"WR={wr_:.0%}\nN={n}",ha="center",color=WH,fontsize=7)

        # 5. MC
        ax5=fig.add_subplot(gs[1,2])
        paths=stats.get("paths"); cagrs_=stats.get("cagrs",np.array([0]))
        if paths is not None:
            sty(ax5,f"Monte Carlo 100K  P50={np.percentile(cagrs_,50):+.1%} CAGR")
            x_mc=np.arange(1,paths.shape[1]+1)
            for i in range(min(500,paths.shape[0])):
                ax5.plot(x_mc,paths[i],color=BL,alpha=0.008,lw=0.4)
            for p,c in zip([5,25,50,75,95],[RD,OR,YL,OR,RD]):
                ax5.plot(x_mc,np.percentile(paths,p,axis=0),
                         color=c,lw=2 if p==50 else 1,
                         label=f"P{p}" if p in [5,50,95] else "")
            ax5.axhline(capital,color=MU,lw=0.8,ls="--",alpha=0.5)
            ax5.yaxis.set_major_formatter(plt.FuncFormatter(lambda x_,_:f"${x_/1000:.0f}K"))
            ax5.set_xlabel("Trade #",color=MU,fontsize=8)
            ax5.legend(fontsize=7,facecolor=PAN,labelcolor=WH,edgecolor="#2d3748")

        os.makedirs(os.path.dirname(out) if os.path.dirname(out) else ".",exist_ok=True)
        plt.savefig(out,dpi=150,bbox_inches="tight",facecolor=BG)
        plt.close(); log.info(f"Dashboard → {out}")
    except Exception as e:
        log.warning(f"Plot failed: {e}")

# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--start",  default="2018-01-01")
    p.add_argument("--end",    default="2025-01-01")
    p.add_argument("--capital",type=float,default=100_000)
    args=p.parse_args()
    os.makedirs("results",exist_ok=True)

    log.info(f"APEX-SPY-AGG  |  {args.start} → {args.end}  |  ${args.capital:,.0f}")
    log.info(f"SELL ATM put + BUY 3% OTM call  (bull regimes)")
    log.info(f"SELL ATM call + BUY 3% OTM put  (bear regimes)")
    log.info(f"Roll@{ROLL_DAYS}d  Stop@{SHORT_STOP_X}x  Risk={RISK_PCT:.0%}/pos  Max{MAX_POSITIONS} concurrent")

    df, src = load_data(args.start, args.end)
    log.info(f"Data: {src}  ({len(df)} days)")

    trades, equity = run(df, args.capital, args.start, args.end)
    stats = report(trades, equity, args.capital, args.start, args.end)
    plot(trades, equity, stats, args.capital, "results/apex_spy_agg_dashboard.png")

    if trades:
        rows=[{"date":t.entry_date,"exit":t.exit_date,"regime":t.regime,
               "S":f"{t.S_entry:.2f}","K_short":f"{t.K_short:.2f}",
               "K_otm":f"{t.K_otm:.2f}","iv":f"{t.iv:.2f}",
               "credit":f"{t.net_credit:.2f}","exit_net":f"{t.exit_net:.2f}",
               "ret":f"{t.ret*100:.1f}","pnl":f"{t.pnl:.0f}",
               "reason":t.exit_reason,"port":f"{t.portfolio:.0f}"}
              for t in trades]
        pd.DataFrame(rows).to_csv("results/apex_spy_agg_trades.csv",index=False)
        log.info("Trades → results/apex_spy_agg_trades.csv")
