"""
APEX-LADDER: Zero-Cost Bull Call Ladder on SPY
===============================================
Structure (5 legs, net ~zero cost or small credit):
  SHORT  ATM put         (K = S)        → collect ~$14 credit  (main engine)
  SHORT  5% OTM put      (K = S*0.95)   → collect ~$5  credit  (zero-cost offset)
  LONG   2% OTM call     (K = S*1.02)   → pay ~$11  high delta — catches most moves
  LONG   5% OTM call     (K = S*1.05)   → pay ~$5   mid delta  — bigger moves
  LONG   8% OTM call     (K = S*1.08)   → pay ~$2   low delta  — tail rip
  ──────────────────────────────────────────────────────
  NET:   ~$0.38 credit   (self-financing)

Three separate win conditions:
  Zone 1: SPY +2% to +5%   → 2% OTM call printing
  Zone 2: SPY +5% to +8%   → 2% + 5% OTM calls printing
  Zone 3: SPY +8%+         → All three calls printing (tail rip)

Risk:
  Max loss: SPY drops >5% below entry → both short puts ITM
  Put spread width = 5% of notional = defined, capped loss
  Vol-scaled sizing: go 4× in easy vol, 0.5× in crisis

Vol scaling: "go hard when market is easy, go easy when it's hard"
  EASY   HV<16%: 4× sizing (2017, 2024 low-vol bull)
  NORMAL HV<22%: 2× sizing
  HARD   HV≥22%: 0.5× sizing
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import norm
from dataclasses import dataclass
from typing import Optional
import warnings, os, logging
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# ── CONFIG ────────────────────────────────────────────────────────────────────
OPTION_WEEKS    = 6
ROLL_DAYS       = 7
R               = 0.045
TRADING_YEAR    = 252.0
SMA_WINDOW      = 50
REGIME_BAND     = 0.015      # ±1.5% from SMA50

# Strike distances
ATM_PUT_MONO    = 0.000      # short put 1: ATM
OTM_PUT_MONO    = 0.100      # short put 2: 10% OTM — only bleeds in real crashes
CALL_HI_MONO    = 0.020      # long call 1: 2% OTM  (high delta)
CALL_MID_MONO   = 0.050      # long call 2: 5% OTM  (mid delta)
CALL_LO_MONO    = 0.080      # long call 3: 8% OTM  (tail rip)

# Stop rules
STOP_PUT_MULT   = 2.5        # close if short put = 2.5× entry (hard stop)
STOP_SPREAD_PCT = 0.70       # close if spread loses 70% of max value — earlier exit

# Vol sizing
VOL_EASY        = 0.16       # HV threshold: easy (go hard)
VOL_NORMAL      = 0.22       # HV threshold: normal
VOL_HARD        = 0.30       # HV threshold: hard/crisis

MAX_CONCURRENT  = 2          # max 2 concurrent — double short puts need discipline
RISK_PCT        = 0.025      # base risk per position

# ── BSM ───────────────────────────────────────────────────────────────────────
def bsm(S, K, T, iv, is_call):
    if T <= 1e-6 or iv <= 0:
        return max(S-K,0) if is_call else max(K-S,0)
    d1=(np.log(S/K)+(R+.5*iv**2)*T)/(iv*np.sqrt(T)); d2=d1-iv*np.sqrt(T)
    if is_call: return S*norm.cdf(d1)-K*np.exp(-R*T)*norm.cdf(d2)
    return K*np.exp(-R*T)*norm.cdf(-d2)-S*norm.cdf(-d1)

def put_iv(iv, mono=0.0):
    """Skew: ATM put ~8% rich, deeper OTM more expensive."""
    return iv * (1.08 + 1.20 * abs(mono))

def vol_regime(closes, window=21):
    if len(closes) < window+1: return 0.18, "normal", 2.0
    hv = float(np.std(np.diff(np.log(closes[-window-1:])),ddof=1)*np.sqrt(252))
    if hv < VOL_EASY:   return hv, "easy",   3.0   # 3× — melt-up mode
    if hv < VOL_NORMAL: return hv, "normal", 1.5   # 1.5× — bread and butter
    if hv < VOL_HARD:   return hv, "hard",   0.5   # 0.5× — cautious
    return                     hv, "crisis", 0.25  # 0.25× — survive

def get_regime(hist):
    if len(hist) < SMA_WINDOW: return "neutral"
    c = hist["close"].values
    sma = c[-SMA_WINDOW:].mean(); last = c[-1]
    if last > sma*(1+REGIME_BAND): return "bull"
    return "neutral"

def est_iv(closes, window=21):
    if len(closes) < window+1: return 0.18
    r = np.diff(np.log(closes[-window-1:]))
    return float(np.clip(np.std(r,ddof=1)*np.sqrt(252)*1.10, .12, 1.5))

# ── POSITION ──────────────────────────────────────────────────────────────────
@dataclass
class LadderPos:
    entry_date:   object
    expiry_date:  object
    S_entry:      float
    iv:           float
    vol_env:      str
    contracts:    int
    # Strike prices
    K_sp1:   float   # short ATM put
    K_sp2:   float   # short 5% OTM put (offset)
    K_lc1:   float   # long 2% OTM call
    K_lc2:   float   # long 5% OTM call
    K_lc3:   float   # long 8% OTM call
    # Entry premiums
    px_sp1:  float   # received
    px_sp2:  float   # received
    px_lc1:  float   # paid
    px_lc2:  float   # paid
    px_lc3:  float   # paid
    net_credit: float   # total net per share (positive = credit received)

@dataclass
class Trade:
    entry_date:  object
    exit_date:   object
    S_entry:     float
    S_exit:      float
    iv:          float
    vol_env:     str
    contracts:   int
    net_credit:  float
    exit_cost:   float    # net cost to close all legs
    pnl:         float
    ret:         float
    exit_reason: str
    portfolio:   float

# ── DATA ──────────────────────────────────────────────────────────────────────
def load_data(start, end):
    api_key    = os.environ.get("ALPACA_API_KEY","")
    api_secret = os.environ.get("ALPACA_API_SECRET","")
    if api_key and api_secret:
        try:
            import alpaca_trade_api as tradeapi
            api = tradeapi.REST(api_key, api_secret,
                                base_url="https://paper-api.alpaca.markets")
            raw = api.get_bars("SPY","1Day",start=start,end=end,
                               adjustment="all").df.reset_index()
            raw.columns=[c.lower() for c in raw.columns]
            tc=next((c for c in raw.columns if c in ("timestamp","time","t")),None)
            if tc: raw["date"]=pd.to_datetime(raw[tc]).dt.date
            data=raw.sort_values("date").reset_index(drop=True)
            log.info(f"LIVE SPY: {len(data)} bars")
            return data, "live"
        except Exception as e:
            log.warning(f"Alpaca: {e}")
    log.info("SYNTHETIC — set ALPACA keys for real results")
    np.random.seed(42)
    n = int((pd.to_datetime(end)-pd.to_datetime(start)).days*252/365)
    dates = pd.bdate_range(start=start,periods=n)
    rets  = np.random.normal(.12/252,.18/np.sqrt(252),n)
    px    = 200*np.cumprod(1+rets)
    return pd.DataFrame({
        "date":[d.date() for d in dates],
        "open":px*(1+np.random.normal(0,.001,n)),
        "high":px*1.004,"low":px*.996,"close":px,"volume":80_000_000
    }), "synthetic"

# ── BACKTEST ──────────────────────────────────────────────────────────────────
def run(data, capital, start, end):
    portfolio = capital; peak = capital
    trades = []; open_pos = []; equity = []
    all_dates = sorted(d for d in data["date"].tolist() if start<=str(d)<=end)

    for today in all_dates:
        hist = data[data["date"]<=today].tail(SMA_WINDOW+25)

        # ── Manage open positions ─────────────────────────────────────────
        still_open = []
        for pos in open_pos:
            days_left = (pd.to_datetime(pos.expiry_date)-
                        pd.to_datetime(today)).days
            row = data[data["date"]==today]
            if row.empty: still_open.append(pos); continue

            S  = float(row["close"].iloc[0])
            h  = data[data["date"]<=today]
            iv = est_iv(h["close"].values)
            T  = max(days_left/TRADING_YEAR, 1e-6)
            reason = None

            # Current values of all 5 legs
            v_sp1 = bsm(S, pos.K_sp1, T, put_iv(iv,ATM_PUT_MONO), False)
            v_sp2 = bsm(S, pos.K_sp2, T, put_iv(iv,OTM_PUT_MONO), False)
            v_lc1 = bsm(S, pos.K_lc1, T, iv,                       True)
            v_lc2 = bsm(S, pos.K_lc2, T, iv,                       True)
            v_lc3 = bsm(S, pos.K_lc3, T, iv,                       True)

            # Cost to close: buy back shorts, sell longs
            exit_cost = (v_sp1 + v_sp2) - (v_lc1 + v_lc2 + v_lc3)
            # pnl per share = credit received - cost to close
            pnl_per   = pos.net_credit - exit_cost

            # Stop: either short put reached 2.5× entry (hard stop)
            if v_sp1 >= pos.px_sp1 * STOP_PUT_MULT: reason = "STOP_PUT"
            # Or put spread losing 85% of max width
            spread_loss = (pos.K_sp1 - S) if S < pos.K_sp1 else 0
            max_spread  = pos.K_sp1 - pos.K_sp2
            if spread_loss >= max_spread * STOP_SPREAD_PCT: reason = "STOP_SPREAD"

            if days_left <= 0:           reason = "EXPIRED"
            elif days_left <= ROLL_DAYS: reason = "ROLL"

            if reason:
                pnl = pnl_per * 100 * pos.contracts
                portfolio += pnl
                ret = pnl_per / max(abs(pos.net_credit), pos.px_sp1)
                trades.append(Trade(
                    entry_date=pos.entry_date, exit_date=today,
                    S_entry=pos.S_entry, S_exit=S, iv=pos.iv,
                    vol_env=pos.vol_env, contracts=pos.contracts,
                    net_credit=pos.net_credit, exit_cost=exit_cost,
                    pnl=pnl, ret=ret, exit_reason=reason, portfolio=portfolio))
                log.info(f"  {today} CLOSE [{pos.vol_env}] {reason:<12} "
                         f"S={S:.1f}({(S/pos.S_entry-1)*100:+.1f}%) "
                         f"ret={ret:+.0%} pnl=${pnl:+,.0f}")
            else:
                still_open.append(pos)

        open_pos = still_open

        if portfolio > peak: peak = portfolio
        if (peak-portfolio)/peak > 0.40:
            log.warning(f"{today}: HALTED drawdown {(peak-portfolio)/peak:.1%}")
            break

        equity.append((today, portfolio))

        # ── Open every Monday in bull regime ─────────────────────────────
        if pd.to_datetime(today).weekday() != 0: continue
        if len(open_pos) >= MAX_CONCURRENT: continue

        regime = get_regime(hist)
        if regime != "bull": continue

        h  = data[data["date"]<=today]
        if len(h) < 25: continue
        row = data[data["date"]==today]
        if row.empty: continue

        S   = float(row["close"].iloc[0])
        iv  = est_iv(h["close"].values)
        hv, vol_env, size_mult = vol_regime(h["close"].values)
        T   = OPTION_WEEKS*5/TRADING_YEAR
        expiry = (pd.to_datetime(today)+pd.Timedelta(weeks=OPTION_WEEKS)).date()

        # Price all 5 legs
        K_sp1 = S*(1 - ATM_PUT_MONO)   # = S
        K_sp2 = S*(1 - OTM_PUT_MONO)   # = S*0.95
        K_lc1 = S*(1 + CALL_HI_MONO)   # = S*1.02
        K_lc2 = S*(1 + CALL_MID_MONO)  # = S*1.05
        K_lc3 = S*(1 + CALL_LO_MONO)   # = S*1.08

        px_sp1 = bsm(S, K_sp1, T, put_iv(iv,ATM_PUT_MONO), False)
        px_sp2 = bsm(S, K_sp2, T, put_iv(iv,OTM_PUT_MONO), False)
        px_lc1 = bsm(S, K_lc1, T, iv,                       True)
        px_lc2 = bsm(S, K_lc2, T, iv,                       True)
        px_lc3 = bsm(S, K_lc3, T, iv,                       True)

        net_credit = (px_sp1 + px_sp2) - (px_lc1 + px_lc2 + px_lc3)

        # Price-invariant sizing: use spread WIDTH as % of S × notional
        # Works at SPY=$200 or SPY=$1400 without flooring to zero
        spread_pct            = OTM_PUT_MONO          # 10% of S
        max_loss_per_contract = S * spread_pct * 100  # e.g. $575*0.10*100 = $5,750
        risk_usd              = portfolio * RISK_PCT * size_mult
        contracts = max(1, int(risk_usd / max_loss_per_contract))
        contracts = min(contracts, max(1, int(portfolio * 0.05 / max_loss_per_contract)))

        # Cash flow at open: receive net_credit (could be small debit if negative)
        cf = net_credit * 100 * contracts
        portfolio += cf

        pos = LadderPos(
            entry_date=today, expiry_date=expiry,
            S_entry=S, iv=iv, vol_env=vol_env, contracts=contracts,
            K_sp1=K_sp1, K_sp2=K_sp2, K_lc1=K_lc1, K_lc2=K_lc2, K_lc3=K_lc3,
            px_sp1=px_sp1, px_sp2=px_sp2,
            px_lc1=px_lc1, px_lc2=px_lc2, px_lc3=px_lc3,
            net_credit=net_credit
        )
        open_pos.append(pos)

        theta_daily = (bsm(S,K_sp1,T-1/252,put_iv(iv),False) - px_sp1 +
                       bsm(S,K_sp2,T-1/252,put_iv(iv,OTM_PUT_MONO),False) - px_sp2) * \
                      100 * contracts * -1

        log.info(f"  {today} OPEN  [{vol_env} HV={hv:.0%} x{size_mult}] "
                 f"S={S:.1f} "
                 f"sp1={K_sp1:.0f}@{px_sp1:.2f} sp2={K_sp2:.0f}@{px_sp2:.2f} | "
                 f"lc1={K_lc1:.0f}@{px_lc1:.2f} lc2={K_lc2:.0f}@{px_lc2:.2f} "
                 f"lc3={K_lc3:.0f}@{px_lc3:.2f} | "
                 f"net={net_credit:+.2f} #{contracts} cf=${cf:+,.0f}")

    # Close remaining
    for pos in open_pos:
        row = data[data["date"]==all_dates[-1]]
        if row.empty: continue
        S  = float(row["close"].iloc[0])
        h  = data[data["date"]<=all_dates[-1]]
        iv = est_iv(h["close"].values)
        T  = 1/TRADING_YEAR
        v_sp1 = bsm(S,pos.K_sp1,T,put_iv(iv,ATM_PUT_MONO),False)
        v_sp2 = bsm(S,pos.K_sp2,T,put_iv(iv,OTM_PUT_MONO),False)
        v_lc1 = bsm(S,pos.K_lc1,T,iv,True)
        v_lc2 = bsm(S,pos.K_lc2,T,iv,True)
        v_lc3 = bsm(S,pos.K_lc3,T,iv,True)
        exit_cost = (v_sp1+v_sp2)-(v_lc1+v_lc2+v_lc3)
        pnl_per   = pos.net_credit - exit_cost
        pnl = pnl_per*100*pos.contracts
        portfolio += pnl
        ret = pnl_per/max(abs(pos.net_credit),pos.px_sp1)
        trades.append(Trade(
            entry_date=pos.entry_date, exit_date=all_dates[-1],
            S_entry=pos.S_entry, S_exit=S, iv=pos.iv,
            vol_env=pos.vol_env, contracts=pos.contracts,
            net_credit=pos.net_credit, exit_cost=exit_cost,
            pnl=pnl, ret=ret, exit_reason="FINAL", portfolio=portfolio))
    equity.append((all_dates[-1],portfolio))
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
    wins  = rets[rets>0]; losses = rets[rets<=0]
    wr    = (rets>0).mean()
    pf    = wins.sum()/abs(losses.sum()) if len(losses) else np.inf

    monthly={}
    for t in trades:
        k=str(t.exit_date)[:7]; monthly[k]=monthly.get(k,0)+t.pnl

    print()
    print("="*70)
    print("APEX-LADDER v2  —  Zero-Cost Bull Call Ladder on SPY")
    print(f"  SHORT ATM put + SHORT 10%OTM put  (tail-only offset)")
    print(f"  LONG  2% OTM call (high Δ)  +  5% OTM (mid Δ)  +  8% OTM (tail)")
    print(f"  Net: ~zero cost | 3 win zones | defined max loss = 5% put spread")
    print("="*70)
    print(f"Period:   {start} → {end}  ({n_yrs:.1f} yrs)")
    print(f"Capital:  ${capital:,.0f}  →  ${final:,.0f}")
    print(f"CAGR:     {cagr:+.1%}   Max DD: {max_dd:.1%}   Sharpe: {sharpe:.2f}")
    print(f"Trades:   {len(trades)}  ({len(trades)/n_yrs:.0f}/yr)   "
          f"WR: {wr:.1%}   PF: {pf:.2f}")
    if len(wins):   print(f"Avg win:  {wins.mean():+.1%}")
    if len(losses): print(f"Avg loss: {losses.mean():+.1%}")
    exits=["ROLL","EXPIRED","STOP_PUT","STOP_SPREAD","FINAL"]
    print("Exits: "+"  ".join(f"{r}={sum(1 for t in trades if t.exit_reason==r)}"
                              for r in exits if any(t.exit_reason==r for t in trades)))
    print()

    print("Vol regime breakdown:")
    for env in ["easy","normal","hard","crisis"]:
        sub=[t for t in trades if t.vol_env==env]
        if not sub: continue
        arr=np.array([t.ret for t in sub])
        pnl=sum(t.pnl for t in sub)
        print(f"  {env.upper():<7}  N={len(sub):>3}  WR={(arr>0).mean():.1%}  "
              f"AvgRet={arr.mean():+.1%}  PnL=${pnl:+,.0f}")
    print()

    print("Monthly P&L ($):")
    years=sorted(set(str(t.exit_date)[:4] for t in trades))
    mos=[f"{m:02d}" for m in range(1,13)]
    print(f"  {'Year':<5}",end="")
    for m in mos: print(f"  {m:>7}",end="")
    print()
    for yr in years:
        print(f"  {yr:<5}",end="")
        for mo in mos:
            v=monthly.get(f"{yr}-{mo}")
            print(f"  {f'{v/1000:+.1f}K' if v is not None else '':>7}",end="")
        print()
    print("="*70)

    # MC simulation
    rng=np.random.default_rng(42); n_sim=100_000; n_draw=max(len(trades),1)
    samp=rng.integers(0,len(rets),(n_sim,n_draw))
    risk=np.mean([abs(t.pnl)/max(t.portfolio,1) for t in trades])
    paths=capital*np.cumprod(1+risk*rets[samp],axis=1)
    term=paths[:,-1]; cagrs_=(term/capital)**(1/n_yrs)-1
    pk_=np.maximum.accumulate(np.hstack([np.full((n_sim,1),capital),paths]),axis=1)
    dd_=((pk_-np.hstack([np.full((n_sim,1),capital),paths]))/pk_).max(axis=1)
    ruin=(paths<capital*0.5).any(axis=1).mean()
    pcts=[5,25,50,75,95]
    print()
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

    return {"cagr":cagr,"max_dd":max_dd,"sharpe":sharpe,"wr":wr,
            "n_trades":len(trades),"equity":equity,
            "paths":paths,"cagrs":cagrs_,"rets":rets}

# ── PLOT ──────────────────────────────────────────────────────────────────────
def plot(trades, equity, stats, capital, out):
    try:
        BG="#0d1117";PAN="#161b22";GR="#22c55e";RD="#ef4444"
        BL="#3b82f6";YL="#eab308";OR="#f97316";MU="#94a3b8";WH="#f1f5f9"
        fig=plt.figure(figsize=(20,12),facecolor=BG)
        fig.suptitle(
            "APEX-LADDER v2  —  Short ATM+10%OTM Put  +  Long 2%/5%/8% OTM Call Ladder",
            color=WH,fontsize=12,fontweight="bold",y=0.98)
        gs=gridspec.GridSpec(2,3,figure=fig,hspace=0.42,wspace=0.32)

        def sty(ax,title):
            ax.set_facecolor(PAN)
            [sp.set_color("#2d3748") for sp in ax.spines.values()]
            ax.tick_params(colors=MU,labelsize=8)
            ax.set_title(title,color=WH,fontsize=9,fontweight="bold",pad=6)
            ax.grid(True,color="#2d3748",lw=0.4,alpha=0.5)

        ax1=fig.add_subplot(gs[0,:2])
        sty(ax1,f"Equity  CAGR={stats.get('cagr',0):+.1%}  "
               f"MaxDD={stats.get('max_dd',0):.1%}  Sharpe={stats.get('sharpe',0):.2f}")
        d_=[pd.to_datetime(d) for d,_ in equity]
        p_=[v for _,v in equity]
        ax1.plot(d_,p_,color=GR,lw=2)
        ax1.axhline(capital,color=MU,lw=0.8,ls="--",alpha=0.5)
        ax1.fill_between(d_,capital,p_,where=[v>=capital for v in p_],color=GR,alpha=0.10)
        ax1.fill_between(d_,capital,p_,where=[v<capital for v in p_],color=RD,alpha=0.15)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_:f"${x/1000:.0f}K"))
        col_map={"easy":YL,"normal":BL,"hard":OR,"crisis":RD}
        for t in trades:
            c=col_map.get(t.vol_env,BL)
            ax1.axvline(pd.to_datetime(t.exit_date),color=c,lw=0.4,alpha=0.35)

        ax2=fig.add_subplot(gs[0,2])
        sty(ax2,"P&L by Year  (bar=total, colors=vol regime)")
        years=sorted(set(str(t.exit_date)[:4] for t in trades))
        yr_pnl=[sum(t.pnl for t in trades if str(t.exit_date)[:4]==yr) for yr in years]
        ax2.bar(years,yr_pnl,color=[GR if v>0 else RD for v in yr_pnl],
                alpha=0.85,edgecolor=BG)
        ax2.axhline(0,color=MU,lw=0.8)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_:f"${x/1000:.0f}K"))
        ax2.tick_params(axis="x",rotation=45)

        ax3=fig.add_subplot(gs[1,0])
        rets_=stats.get("rets",np.array([]))
        sty(ax3,f"Return Distribution  WR={stats.get('wr',0):.1%}")
        if len(rets_):
            bins=np.linspace(-120,400,60)
            ax3.hist(rets_*100,bins=bins,color=BL,alpha=0.75,edgecolor=BG)
            ax3.axvline(0,color=MU,lw=0.8,ls="--")
            ax3.axvline(np.median(rets_)*100,color=YL,lw=1.5,ls="--",
                        label=f"Median {np.median(rets_)*100:.0f}%")
            ax3.legend(fontsize=7,facecolor=PAN,labelcolor=WH,edgecolor="#2d3748")
        ax3.set_xlabel("Return %",color=MU,fontsize=8)

        ax4=fig.add_subplot(gs[1,1])
        sty(ax4,"Vol Regime: WR & Avg Return")
        envs=[e for e in ["easy","normal","hard","crisis"]
              if any(t.vol_env==e for t in trades)]
        if envs:
            wrs_=[( np.array([t.ret for t in trades if t.vol_env==e])>0).mean() for e in envs]
            avgs=[  np.array([t.ret for t in trades if t.vol_env==e]).mean()     for e in envs]
            x=np.arange(len(envs)); w=0.35
            ax4.bar(x,    wrs_,w,color=GR,alpha=0.8,label="WR",edgecolor=BG)
            ax4.bar(x+w,  [a+1 for a in avgs],w,color=YL,alpha=0.8,
                    label="1+AvgRet",edgecolor=BG)
            ax4.set_xticks(x+w/2)
            ax4.set_xticklabels([e.upper() for e in envs],color=MU,fontsize=8)
            ax4.axhline(0.5,color=MU,lw=0.8,ls="--",alpha=0.4)
            ax4.legend(fontsize=7,facecolor=PAN,labelcolor=WH,edgecolor="#2d3748")

        ax5=fig.add_subplot(gs[1,2])
        paths=stats.get("paths"); cagrs_=stats.get("cagrs",np.array([0]))
        if paths is not None:
            sty(ax5,f"Monte Carlo  P50={np.percentile(cagrs_,50):+.1%} CAGR")
            x_mc=np.arange(1,paths.shape[1]+1)
            for i in range(min(300,paths.shape[0])):
                ax5.plot(x_mc,paths[i],color=BL,alpha=0.008,lw=0.5)
            for pct,c in zip([5,25,50,75,95],[RD,OR,YL,OR,RD]):
                pv=np.percentile(paths,pct,axis=0)
                ax5.plot(x_mc,pv,color=c,lw=2 if pct==50 else 1,
                         label=f"P{pct}" if pct in [5,50,95] else "")
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
    p.add_argument("--start",  default="2016-01-01")
    p.add_argument("--end",    default="2026-03-08")
    p.add_argument("--capital",type=float,default=100_000)
    args=p.parse_args()
    os.makedirs("results",exist_ok=True)

    data,src=load_data(args.start,args.end)
    log.info(f"Source: {src}")
    log.info(f"Structure: SHORT ATM put + SHORT {OTM_PUT_MONO:.0%}OTM put | "
             f"LONG {CALL_HI_MONO:.0%}/{CALL_MID_MONO:.0%}/{CALL_LO_MONO:.0%} OTM calls")
    log.info(f"Max concurrent: {MAX_CONCURRENT}  |  Vol sizing: "
             f"EASY x{4.0}  NORMAL x{2.0}  HARD x{0.75}")

    trades,equity=run(data,args.capital,args.start,args.end)
    stats=report(trades,equity,args.capital,args.start,args.end)
    plot(trades,equity,stats,args.capital,"results/apex_ladder_dashboard.png")

    if trades:
        rows=[{"entry":t.entry_date,"exit":t.exit_date,
               "S_entry":f"{t.S_entry:.2f}","S_exit":f"{t.S_exit:.2f}",
               "vol_env":t.vol_env,"contracts":t.contracts,
               "net_credit":f"{t.net_credit:.3f}",
               "ret":f"{t.ret*100:.1f}%","pnl":f"{t.pnl:.0f}",
               "reason":t.exit_reason,"portfolio":f"{t.portfolio:.0f}"}
              for t in trades]
        pd.DataFrame(rows).to_csv("results/apex_ladder_trades.csv",index=False)
        log.info("Trades → results/apex_ladder_trades.csv")
