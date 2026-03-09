"""
APEX-HYBRID: Asymmetric Two-Regime Options Strategy
====================================================
BULL (QQQ > SMA50 * 1.015):
  SELL ATM put + BUY 3% OTM call  → net credit, win if flat or up
  Put skew works FOR you: collect more on short put than cost of long call

BEAR (QQQ < SMA50 * 0.985):
  BUY ATM put SPREAD
    Long  ATM put  (K = S)
    Short OTM put  (K = S * 0.92, 8% lower)
  Net debit ~$3-6, max profit ~$8-12 on 8%+ drop
  Put skew now HELPS you: ATM put is expensive, buy it
  Defined risk, defined reward — no short gamma blowup

NEUTRAL: no position

Why this works:
  Bull RR:    collect put skew premium + own cheap OTM call lottery
  Bear spread: pay for ATM put with OTM put sale, skew makes ATM cheap relative
  Both use put skew — just from opposite sides depending on regime
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
TICKERS        = ["SPY"]       # trade on SPY only — deep liquid
SMA_WINDOW     = 50
REGIME_BAND    = 0.015         # ±1.5% from SMA to confirm regime
OPTION_WEEKS   = 6
ROLL_DAYS      = 7
R              = 0.045
TRADING_YEAR   = 252.0

# Bull RR params (from working strategy)
BULL_CALL_OTM  = 0.02          # buy 2% OTM call — tighter OTM = more delta = higher WR
BULL_STOP_X    = 2.0           # close if short put doubles

# Bear spread params
BEAR_SPREAD_W  = 0.08          # short put 8% below ATM long put
BEAR_STOP_LOSS = -0.80         # close spread if -80% value

# ── VOLATILITY-SCALED SIZING ─────────────────────────────────────────────────
# "Go hard when the market is easy, go easy when the market is hard"
# VIX proxy: use 21-day realized vol of SPY to classify environment
# Low vol  = HV < 12%  → EASY market  → scale UP
# Mid vol  = HV 12-20% → NORMAL       → base size
# High vol = HV 20-30% → HARD market  → scale DOWN
# Crisis   = HV > 30%  → DANGER       → minimum size or skip bear

VOL_LOW    = 0.12   # below this = easy, scale up
VOL_MID    = 0.20   # below this = normal
VOL_HIGH   = 0.30   # above this = danger zone

# Position size multipliers per vol regime
# Bull RR (selling puts) — scale hard in low vol, cautious in high vol
BULL_SIZE_EASY   = 5.0   # 5× base in low-vol bull (2017/2021 melt-up — go maximum)
BULL_SIZE_NORMAL = 2.5   # 2.5× base in normal vol
BULL_SIZE_HARD   = 0.75  # 0.75× base in high vol — cautious but still trade

# Bear spread (buying puts) — scale hard in high vol, skip in low vol
BEAR_SIZE_EASY   = 0.0   # skip bear spreads in low vol (puts too expensive, bounce likely)
BEAR_SIZE_NORMAL = 1.0   # base in normal vol
BEAR_SIZE_HARD   = 1.5   # 1.5× in high vol (measured increase)
BEAR_SIZE_CRISIS = 2.0   # 2.0× in crisis vol (disciplined — edge exists but vol is high)

# Sizing
RISK_PCT       = 0.02          # 2% portfolio per position (base)

# ── BSM ───────────────────────────────────────────────────────────────────────
def bsm(S, K, T, iv, is_call):
    if T <= 1e-6 or iv <= 0:
        return max(S-K,0) if is_call else max(K-S,0)
    d1 = (np.log(S/K)+(R+.5*iv**2)*T)/(iv*np.sqrt(T))
    d2 = d1 - iv*np.sqrt(T)
    if is_call: return S*norm.cdf(d1)-K*np.exp(-R*T)*norm.cdf(d2)
    return K*np.exp(-R*T)*norm.cdf(-d2)-S*norm.cdf(-d1)

def put_iv(iv):
    """ATM put trades ~8% richer than HV due to skew."""
    return iv * 1.08

def otm_put_iv(iv, moneyness):
    """OTM put IV increases with distance (volatility skew)."""
    return iv * (1.0 + 1.5 * moneyness)   # 1.5x skew slope

# ── DATA ──────────────────────────────────────────────────────────────────────
def load_data(tickers, start, end):
    api_key    = os.environ.get("ALPACA_API_KEY","")
    api_secret = os.environ.get("ALPACA_API_SECRET","")
    if api_key and api_secret:
        try:
            import alpaca_trade_api as tradeapi
            api  = tradeapi.REST(api_key, api_secret,
                                 base_url="https://paper-api.alpaca.markets")
            data = {}
            for t in tickers:
                raw = api.get_bars(t,"1Day",start=start,end=end,
                                   adjustment="all").df.reset_index()
                raw.columns=[c.lower() for c in raw.columns]
                tc=next((c for c in raw.columns if c in ("timestamp","time","t")),None)
                if tc: raw["date"]=pd.to_datetime(raw[tc]).dt.date
                data[t]=raw.sort_values("date").reset_index(drop=True)
                log.info(f"  {t}: {len(raw)} bars")
            if data: return data,"live"
        except Exception as e:
            log.warning(f"Alpaca: {e}")

    log.info("SYNTHETIC — set ALPACA keys for real results")
    np.random.seed(42)
    n=int((pd.to_datetime(end)-pd.to_datetime(start)).days*252/365)
    dates=pd.bdate_range(start=start,periods=n)
    rets=np.random.normal(.12/252,.18/np.sqrt(252),n)
    px=450*np.cumprod(1+rets)
    data={"SPY":pd.DataFrame({
        "date":[d.date() for d in dates],
        "open":px*(1+np.random.normal(0,.001,n)),
        "high":px*1.004,"low":px*.996,"close":px,"volume":80_000_000})}
    return data,"synthetic"

def get_regime(hist):
    if len(hist)<SMA_WINDOW+5: return "neutral"
    c   = hist["close"].values
    sma = c[-SMA_WINDOW:].mean()
    last = c[-1]

    if last > sma*(1+REGIME_BAND): return "bull"

    # Bear regime → neutral (bull-only mode)
    # Bear put spreads removed — bull RR edge is 4x stronger, no reason to hedge
    return "neutral"

def est_iv(closes,window=21):
    if len(closes)<window+1: return 0.18
    r=np.diff(np.log(closes[-window-1:]))
    return float(np.clip(np.std(r,ddof=1)*np.sqrt(252)*1.10,.12,1.5))

# ── POSITION ──────────────────────────────────────────────────────────────────
@dataclass
class Pos:
    regime:      str
    entry_date:  object
    expiry_date: object
    S_entry:     float
    iv:          float
    contracts:   int
    # Bull RR legs
    K_short_put: float = 0.0   # short put strike
    short_put_px: float = 0.0  # premium received
    K_long_call: float = 0.0   # long call strike
    long_call_px: float = 0.0  # premium paid
    net_credit:  float = 0.0   # net received (positive = credit)
    # Bear spread legs
    K_long_put:  float = 0.0   # long ATM put
    long_put_px: float = 0.0
    K_short_put2: float = 0.0  # short OTM put
    short_put_px2: float = 0.0
    net_debit:   float = 0.0   # net paid (positive = debit)

@dataclass
class Trade:
    regime:      str
    entry_date:  object
    exit_date:   object
    S_entry:     float
    S_exit:      float
    iv:          float
    entry_val:   float    # credit received (bull) or debit paid (bear)
    exit_val:    float    # cost to close (bull) or value received (bear)
    pnl:         float
    ret:         float
    exit_reason: str
    portfolio:   float
    vol_env:     str = ""

# ── BACKTEST ──────────────────────────────────────────────────────────────────

def vol_regime(closes, window=21):
    """
    Classify current realized volatility environment.
    Returns (hv_annualized, regime_label, bull_mult, bear_mult)
    """
    if len(closes) < window + 1:
        return 0.18, "normal", 1.5, 1.0
    r   = np.diff(np.log(closes[-window-1:]))
    hv  = float(np.std(r, ddof=1) * np.sqrt(252))

    if hv < VOL_LOW:
        return hv, "easy",   BULL_SIZE_EASY,   BEAR_SIZE_EASY
    elif hv < VOL_MID:
        return hv, "normal", BULL_SIZE_NORMAL, BEAR_SIZE_NORMAL
    elif hv < VOL_HIGH:
        return hv, "hard",   BULL_SIZE_HARD,   BEAR_SIZE_HARD
    else:
        return hv, "crisis", BULL_SIZE_HARD,   BEAR_SIZE_CRISIS

def run(data, capital, start, end):
    portfolio=capital; peak=capital
    trades=[]; open_pos=[]; equity=[]

    spy=data["SPY"]
    all_dates=sorted(d for d in spy["date"].tolist() if start<=str(d)<=end)

    for today in all_dates:
        hist=spy[spy["date"]<=today].tail(SMA_WINDOW+5)

        # ── Manage open positions ──────────────────────────────────────────
        still_open=[]
        for pos in open_pos:
            days_left=(pd.to_datetime(pos.expiry_date)-
                       pd.to_datetime(today)).days
            row=spy[spy["date"]==today]
            if row.empty: still_open.append(pos); continue

            S=float(row["close"].iloc[0])
            h=spy[spy["date"]<=today]
            iv=est_iv(h["close"].values)
            T=max(days_left/TRADING_YEAR,0)
            reason=None

            if pos.regime=="bull":
                # Bull RR: short put + long call
                sp_px=bsm(S,pos.K_short_put,T,put_iv(iv),False)
                lc_px=bsm(S,pos.K_long_call,T,iv,True)
                # Cost to close: buy back short put, sell long call
                exit_cost = sp_px - lc_px   # pay this to close
                pnl_per   = pos.net_credit - exit_cost

                if days_left<=0:            reason="EXPIRED"
                elif days_left<=ROLL_DAYS:  reason="ROLL"
                elif sp_px>=pos.short_put_px*BULL_STOP_X: reason="STOP"

                if reason:
                    pnl=pnl_per*100*pos.contracts
                    portfolio+=pnl
                    ret=pnl_per/max(abs(pos.net_credit),pos.short_put_px)*\
                        (1 if pos.net_credit>0 else -1)
                    trades.append(Trade(
                        regime=pos.regime,
                        entry_date=pos.entry_date,exit_date=today,
                        S_entry=pos.S_entry,S_exit=S,iv=pos.iv,
                        entry_val=pos.net_credit,exit_val=exit_cost,
                        pnl=pnl,ret=ret,
                        exit_reason=reason,portfolio=portfolio,
                        vol_env=getattr(pos,'vol_env','')))
                    log.info(f"  {today} CLOSE BULL-RR  {reason:<8} "
                             f"S={S:.1f} short_put={sp_px:.2f}(was {pos.short_put_px:.2f}) "
                             f"ret={ret:+.0%} pnl=${pnl:+,.0f}")
                else:
                    still_open.append(pos)

            else:  # bear spread
                # Bear put spread: long ATM put + short OTM put
                lp_iv  = put_iv(iv)
                sp2_iv = otm_put_iv(iv, BEAR_SPREAD_W)
                lp_px  = bsm(S,pos.K_long_put, T,lp_iv, False)
                sp2_px = bsm(S,pos.K_short_put2,T,sp2_iv,False)
                spread_val = lp_px - sp2_px   # current spread value
                ret_so_far = (spread_val-pos.net_debit)/pos.net_debit

                if days_left<=0:            reason="EXPIRED"
                elif days_left<=ROLL_DAYS:  reason="ROLL"
                elif ret_so_far<=BEAR_STOP_LOSS: reason="STOP"

                if reason:
                    # Sell spread to close: receive spread_val
                    exit_recv=spread_val*100*pos.contracts
                    portfolio+=exit_recv
                    pnl=(spread_val-pos.net_debit)*100*pos.contracts
                    ret_f=(spread_val-pos.net_debit)/pos.net_debit
                    trades.append(Trade(
                        regime=pos.regime,
                        entry_date=pos.entry_date,exit_date=today,
                        S_entry=pos.S_entry,S_exit=S,iv=pos.iv,
                        entry_val=pos.net_debit,exit_val=spread_val,
                        pnl=pnl,ret=ret_f,
                        exit_reason=reason,portfolio=portfolio))
                    log.info(f"  {today} CLOSE BEAR-SPD {reason:<8} "
                             f"S={S:.1f} spread={spread_val:.2f}(was {pos.net_debit:.2f}) "
                             f"ret={ret_f:+.0%} pnl=${pnl:+,.0f}")
                else:
                    still_open.append(pos)

        open_pos=still_open

        if portfolio>peak: peak=portfolio
        dd=(peak-portfolio)/peak
        if dd>0.40:
            log.warning(f"{today}: HALTED dd={dd:.1%}")
            break

        equity.append((today,portfolio))

        # ── Open every Monday ──────────────────────────────────────────────
        if pd.to_datetime(today).weekday()!=0: continue
        if any(p.regime is not None for p in open_pos): continue  # one at a time

        regime=get_regime(hist)
        if regime=="neutral": continue

        row=spy[spy["date"]==today]
        if row.empty: continue
        h=spy[spy["date"]<=today]
        if len(h)<22: continue

        S=float(row["close"].iloc[0])
        iv=est_iv(h["close"].values)
        T=OPTION_WEEKS*5/TRADING_YEAR
        risk=portfolio*RISK_PCT
        expiry=(pd.to_datetime(today)+pd.Timedelta(weeks=OPTION_WEEKS)).date()

        if regime=="bull":
            K_sp   = S                          # ATM short put
            K_lc   = S*(1+BULL_CALL_OTM)        # 3% OTM long call
            sp_px  = bsm(S,K_sp,T,put_iv(iv),False)
            lc_px  = bsm(S,K_lc,T,iv,True)
            credit = sp_px - lc_px

            # Vol-scaled sizing: go HARD when market is easy
            hv, vol_env, bull_mult, _ = vol_regime(h["close"].values)
            scaled_risk = risk * bull_mult
            contracts = max(1,int(scaled_risk/(sp_px*100*3)))
            contracts = min(contracts,int(portfolio*0.30/(sp_px*100)))
            cf        = credit*100*contracts
            portfolio += cf

            pos=Pos(regime="bull",entry_date=today,expiry_date=expiry,
                    S_entry=S,iv=iv,contracts=contracts,
                    K_short_put=K_sp,short_put_px=sp_px,
                    K_long_call=K_lc,long_call_px=lc_px,net_credit=credit)
            pos.vol_env = vol_env
            open_pos.append(pos)
            log.info(f"  {today} OPEN  BULL-RR  [{vol_env} vol={hv:.0%} x{bull_mult}] "
                     f"S={S:.1f} short_put={K_sp:.1f}@{sp_px:.2f} "
                     f"long_call={K_lc:.1f}@{lc_px:.2f} "
                     f"net={'CR' if credit>0 else 'DR'}{abs(credit):.2f} "
                     f"#{contracts} cf=${cf:+,.0f}")

        else:  # bear
            K_lp   = S                           # ATM long put
            K_sp2  = S*(1-BEAR_SPREAD_W)         # 8% OTM short put
            lp_iv_ = put_iv(iv)
            sp2_iv_= otm_put_iv(iv,BEAR_SPREAD_W)
            lp_px  = bsm(S,K_lp, T,lp_iv_, False)
            sp2_px = bsm(S,K_sp2,T,sp2_iv_,False)
            debit  = lp_px - sp2_px              # net cost

            # Vol-scaled sizing: go HARD when market is hard
            hv, vol_env, _, bear_mult = vol_regime(h["close"].values)
            # In bear regime, low-vol skips already filtered by regime confirmation
            # Override: if vol is low but we made it here, treat as normal
            if bear_mult == 0.0:
                bear_mult = 1.0
                vol_env   = "normal"
            scaled_risk = risk * bear_mult
            if debit <= 0.01: continue
            contracts = max(1,int(scaled_risk/(debit*100)))
            contracts = min(contracts,int(portfolio*0.25/(debit*100)))
            cf        = -debit*100*contracts     # pay debit
            portfolio += cf

            pos=Pos(regime="bear",entry_date=today,expiry_date=expiry,
                    S_entry=S,iv=iv,contracts=contracts,
                    K_long_put=K_lp,long_put_px=lp_px,
                    K_short_put2=K_sp2,short_put_px2=sp2_px,net_debit=debit)
            pos.vol_env = vol_env
            open_pos.append(pos)
            log.info(f"  {today} OPEN  BEAR-SPD [{vol_env} vol={hv:.0%} x{bear_mult}] "
                     f"S={S:.1f} long_put={K_lp:.1f}@{lp_px:.2f} "
                     f"short_put={K_sp2:.1f}@{sp2_px:.2f} "
                     f"debit={debit:.2f} max_prof={K_lp-K_sp2:.1f} "
                     f"#{contracts} cf=${cf:+,.0f}")

    # Close remaining
    for pos in open_pos:
        row=spy[spy["date"]==all_dates[-1]]
        if row.empty: continue
        S=float(row["close"].iloc[0])
        h=spy[spy["date"]<=all_dates[-1]]
        iv=est_iv(h["close"].values)
        T=1/TRADING_YEAR
        if pos.regime=="bull":
            sp_px=bsm(S,pos.K_short_put,T,put_iv(iv),False)
            lc_px=bsm(S,pos.K_long_call,T,iv,True)
            exit_cost=sp_px-lc_px
            pnl=(pos.net_credit-exit_cost)*100*pos.contracts
            ret=(pos.net_credit-exit_cost)/max(abs(pos.net_credit),pos.short_put_px)
            portfolio+=pnl
        else:
            lp_px=bsm(S,pos.K_long_put, T,put_iv(iv),False)
            sp2_px=bsm(S,pos.K_short_put2,T,otm_put_iv(iv,BEAR_SPREAD_W),False)
            sv=lp_px-sp2_px
            pnl=(sv-pos.net_debit)*100*pos.contracts
            ret=(sv-pos.net_debit)/pos.net_debit
            portfolio+=pnl*0
            portfolio+=sv*100*pos.contracts
            pnl=(sv-pos.net_debit)*100*pos.contracts
        trades.append(Trade(regime=pos.regime,
            entry_date=pos.entry_date,exit_date=all_dates[-1],
            S_entry=pos.S_entry,S_exit=S,iv=pos.iv,
            entry_val=pos.net_credit if pos.regime=="bull" else pos.net_debit,
            exit_val=0,pnl=pnl,ret=ret,
            exit_reason="FINAL",portfolio=portfolio))
    equity.append((all_dates[-1],portfolio))
    return trades,equity

# ── REPORT ────────────────────────────────────────────────────────────────────
def report(trades,equity,capital,start,end):
    if not trades: print("No trades."); return {}
    n_yrs=(pd.to_datetime(end)-pd.to_datetime(start)).days/365.25
    final=trades[-1].portfolio
    cagr=(final/capital)**(1/n_yrs)-1
    ports=[v for _,v in equity]
    peaks=np.maximum.accumulate(ports)
    max_dd=float(((peaks-ports)/peaks).max()) if ports else 0
    dr=np.diff(ports)/np.array(ports[:-1])
    sharpe=dr.mean()/dr.std()*np.sqrt(252) if dr.std()>0 else 0
    rets=np.array([t.ret for t in trades])
    wins=rets[rets>0]; losses=rets[rets<=0]
    wr=(rets>0).mean()
    pf=wins.sum()/abs(losses.sum()) if len(losses) else np.inf

    monthly={}
    for t in trades:
        k=str(t.exit_date)[:7]
        monthly[k]=monthly.get(k,0)+t.pnl

    print()
    print("="*70)
    print("APEX-BULL  —  Vol-Scaled Bull Risk Reversal (SPY, Bull Only)")
    print(f"  BULL: SELL ATM put + BUY {BULL_CALL_OTM:.0%} OTM call  → net credit")
    print(f"  BEAR: BUY ATM put + SELL {BEAR_SPREAD_W:.0%} OTM put   → defined risk")
    print(f"  NEUTRAL: no position")
    print("="*70)
    print(f"Period:   {start} → {end}  ({n_yrs:.1f} yrs)")
    print(f"Capital:  ${capital:,.0f}  →  ${final:,.0f}")
    print(f"CAGR:     {cagr:+.1%}   Max DD: {max_dd:.1%}   Sharpe: {sharpe:.2f}")
    print(f"Trades:   {len(trades)}  ({len(trades)/n_yrs:.0f}/yr)  "
          f"WR: {wr:.1%}   PF: {pf:.2f}")
    if len(wins):   print(f"Avg win:  {wins.mean():+.1%}")
    if len(losses): print(f"Avg loss: {losses.mean():+.1%}")
    exits=["ROLL","EXPIRED","STOP","FINAL"]
    print("Exits: "+"  ".join(f"{r}={sum(1 for t in trades if t.exit_reason==r)}"
                              for r in exits))
    print()

    for reg in ["bull","bear"]:
        sub=[t for t in trades if t.regime==reg]
        if not sub: continue
        arr=np.array([t.ret for t in sub])
        pnl=sum(t.pnl for t in sub)
        stops=sum(1 for t in sub if t.exit_reason=="STOP")
        print(f"  {reg.upper():<4}  N={len(sub):>3}  WR={(arr>0).mean():.1%}  "
              f"AvgRet={arr.mean():+.1%}  PnL=${pnl:+,.0f}  Stops={stops}")

    print()
    print("Vol regime sizing breakdown:")
    for env in ["easy","normal","hard","crisis"]:
        sub=[t for t in trades if getattr(t,'vol_env','')==env]
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

    # MC
    rng=np.random.default_rng(42); n_sim=100_000
    n_draw=max(len(trades),1)
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

    return {"cagr":cagr,"max_dd":max_dd,"sharpe":sharpe,
            "n_trades":len(trades),"wr":wr,"equity":equity,
            "paths":paths,"cagrs":cagrs_,"rets":rets}

# ── PLOT ──────────────────────────────────────────────────────────────────────
def plot(trades,equity,stats,capital,out):
    try:
        BG="#0d1117";PAN="#161b22";GR="#22c55e";RD="#ef4444"
        BL="#3b82f6";YL="#eab308";OR="#f97316";MU="#94a3b8";WH="#f1f5f9"
        fig=plt.figure(figsize=(20,12),facecolor=BG)
        fig.suptitle("APEX-BULL  —  Vol-Scaled Bull Risk Reversal (SPY, Bull Only)",
                     color=WH,fontsize=13,fontweight="bold",y=0.98)
        gs=gridspec.GridSpec(2,3,figure=fig,hspace=0.42,wspace=0.32)

        def sty(ax,title):
            ax.set_facecolor(PAN)
            [sp.set_color("#2d3748") for sp in ax.spines.values()]
            ax.tick_params(colors=MU,labelsize=8)
            ax.set_title(title,color=WH,fontsize=9,fontweight="bold",pad=6)
            ax.grid(True,color="#2d3748",lw=0.4,alpha=0.5)

        # Equity
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
        # Shade regimes
        for t in trades:
            c=BL if t.regime=="bull" else OR
            ax1.axvspan(pd.to_datetime(t.entry_date),
                        pd.to_datetime(t.exit_date),color=c,alpha=0.06)

        # Regime P&L bars
        ax2=fig.add_subplot(gs[0,2])
        sty(ax2,"P&L by Regime + Year")
        years=sorted(set(str(t.exit_date)[:4] for t in trades))
        x=np.arange(len(years)); w=0.35
        for j,(reg,c) in enumerate(zip(["bull","bear"],[BL,OR])):
            pnls=[sum(t.pnl for t in trades
                      if t.regime==reg and str(t.exit_date)[:4]==yr)
                  for yr in years]
            ax2.bar(x+j*w,pnls,w,color=c,alpha=0.85,
                    label=reg.upper(),edgecolor=BG)
        ax2.axhline(0,color=MU,lw=0.8)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_:f"${x/1000:.0f}K"))
        ax2.set_xticks(x+w/2); ax2.set_xticklabels(years,color=MU,fontsize=7,rotation=45)
        ax2.legend(fontsize=7,facecolor=PAN,labelcolor=WH,edgecolor="#2d3748")

        # Return dist by regime
        ax3=fig.add_subplot(gs[1,0])
        sty(ax3,f"Return Distribution  WR={stats.get('wr',0):.1%}")
        rets_=stats.get("rets",np.array([]))
        bull_r=[t.ret*100 for t in trades if t.regime=="bull"]
        bear_r=[t.ret*100 for t in trades if t.regime=="bear"]
        bins=np.linspace(-120,300,50)
        if bull_r: ax3.hist(bull_r,bins=bins,color=BL,alpha=0.70,label=f"Bull RR ({len(bull_r)})")
        if bear_r: ax3.hist(bear_r,bins=bins,color=OR,alpha=0.70,label=f"Bear Spd ({len(bear_r)})")
        ax3.axvline(0,color=MU,lw=0.8,ls="--")
        ax3.legend(fontsize=7,facecolor=PAN,labelcolor=WH,edgecolor="#2d3748")
        ax3.set_xlabel("Return %",color=MU,fontsize=8)

        # Cumulative P&L by regime
        ax4=fig.add_subplot(gs[1,1])
        sty(ax4,"Cumulative P&L: Bull RR vs Bear Spread")
        bull_trades=sorted([t for t in trades if t.regime=="bull"],key=lambda x:x.exit_date)
        bear_trades=sorted([t for t in trades if t.regime=="bear"],key=lambda x:x.exit_date)
        if bull_trades:
            cum_b=np.cumsum([t.pnl for t in bull_trades])
            ax4.plot([pd.to_datetime(t.exit_date) for t in bull_trades],
                     cum_b,color=BL,lw=2,label=f"Bull RR  ${cum_b[-1]:+,.0f}")
        if bear_trades:
            cum_be=np.cumsum([t.pnl for t in bear_trades])
            ax4.plot([pd.to_datetime(t.exit_date) for t in bear_trades],
                     cum_be,color=OR,lw=2,label=f"Bear Spd ${cum_be[-1]:+,.0f}")
        ax4.axhline(0,color=MU,lw=0.8,ls="--",alpha=0.5)
        ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_:f"${x/1000:.0f}K"))
        ax4.legend(fontsize=7,facecolor=PAN,labelcolor=WH,edgecolor="#2d3748")

        # MC
        ax5=fig.add_subplot(gs[1,2])
        paths=stats.get("paths"); cagrs_=stats.get("cagrs",np.array([0]))
        if paths is not None:
            sty(ax5,f"Monte Carlo  P50={np.percentile(cagrs_,50):+.1%} CAGR")
            x_mc=np.arange(1,paths.shape[1]+1)
            for i in range(min(400,paths.shape[0])):
                ax5.plot(x_mc,paths[i],color=BL,alpha=0.01,lw=0.4)
            for p,c in zip([5,25,50,75,95],[RD,OR,YL,OR,RD]):
                pv=np.percentile(paths,p,axis=0)
                ax5.plot(x_mc,pv,color=c,lw=2 if p==50 else 1,
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

    data,src=load_data(["SPY"],args.start,args.end)
    log.info(f"Source: {src}  |  "
             f"Bull: sell ATM put + buy {BULL_CALL_OTM:.0%} OTM call  |  "
             f"Bear: ATM/{BEAR_SPREAD_W:.0%}-OTM put spread")

    trades,equity=run(data,args.capital,args.start,args.end)
    stats=report(trades,equity,args.capital,args.start,args.end)
    plot(trades,equity,stats,args.capital,"results/apex_hybrid_dashboard.png")

    if trades:
        rows=[{"date":t.entry_date,"exit":t.exit_date,"regime":t.regime,
               "S":f"{t.S_entry:.2f}","entry_val":f"{t.entry_val:.2f}",
               "exit_val":f"{t.exit_val:.2f}","ret":f"{t.ret*100:.1f}",
               "pnl":f"{t.pnl:.0f}","reason":t.exit_reason,
               "port":f"{t.portfolio:.0f}"} for t in trades]
        pd.DataFrame(rows).to_csv("results/apex_hybrid_trades.csv",index=False)
        log.info("Trades → results/apex_hybrid_trades.csv")
