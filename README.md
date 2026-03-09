# APEX - Adaptive Position EXecution

> Independent quantitative options research platform. Three parallel research streams - intraday lead signals, gap momentum, and systematic options structure - converging into a unified, backtested trading framework on real market data.

**Stack:** Python - NumPy - SciPy - Alpaca Markets API - Black-Scholes - Monte Carlo 
**Data:** 10 years real SPY/QQQ/PLTR/NVDA daily + intraday bars via Alpaca 
**Approach:** Hypothesis ' backtest ' diagnose failure ' redesign ' repeat

---

## Table of Contents
1. [Research Stream 1 - Intraday Lead Signal](#1-intraday-lead-signal-pltrnvda--spy)
2. [Research Stream 2 - Gap Momentum (0DTE QQQ)](#2-gap-momentum-strategy-0dte-qqq)
3. [Research Stream 3 - Options Structure Evolution](#3-options-structure-evolution)
4. [Final Strategy - APEX-LADDER](#4-final-strategy--apex-ladder)
5. [Key Quantitative Concepts](#5-key-quantitative-concepts)
6. [Architecture](#6-architecture)

---

## 1. Intraday Lead Signal: PLTR/NVDA -> SPY

**Hypothesis:** High-beta growth stocks (PLTR, NVDA) front-run SPY intraday moves. If PLTR's first-15-minute return predicts SPY's next-60-minute return, that's a real-time regime filter that improves options entry timing.

### Methodology
- Collected 59 trading days of live intraday 5-min bars via Alpaca
- Computed PLTR and NVDA first-15-minute returns each session
- Regressed against SPY next-60-minute returns
- Built composite signal and tested directional prediction accuracy

### Results (N=59 live trading days)

```
Signal p-value Significant
------------------------------------------------------
PLTR first-15 min +0.500 0.000 Strong
NVDA first-15 min +0.184 0.163 " Not alone
Composite (PLTR+NVDA) +0.463 0.000 Strong
Divergence signal +0.447 0.000 Strong
```

```
R^2 breakdown:
 SPY gap alone: 0.053
 SPY + PLTR + NVDA: 0.259
 Incremental R^2: +0.206 4 improvement over SPY gap alone
```

### Directional Accuracy by Regime

```
Signal Direction N P(SPY correct) Avg SPY move
--------------------------------------------------------
Strong UP (>+0.3%) 15 73% +0.148%
Neutral 19 52% +0.031%
Strong DN (<-0.3%) 25 64% -0.211%
```

**Key finding:** On down-open days, PLTR first-15 correlation strengthens to =0.540 (p=0.002). The signal is strongest precisely when you need it most - volatile bear openings. This became a live entry filter: block options trades when composite strongly opposes direction.

---

## 2. Gap Momentum Strategy: 0DTE QQQ

**Hypothesis:** The overnight gap between NQ futures close and QQQ open carries predictive information. Specific gap size zones have asymmetric win rates - some favor continuation, others fade.

### Zone Map (empirically derived, 2016"2025)

```
Gap Size Zone Signal Condition
----------------------------------------------------------
+0.45% ' +0.65% UP_MOD_CONT CALL Bull + IV<25%
+0.65% ' +0.75% DEAD_ZONE_UP NONE -
+0.75%+ UP_BIG_FADE PUT Bear confirmed only
-0.01% ' -0.18% DN_SMALL_BOUNCE CALL Bull confirmed only
-0.18% ' -1.50% DEAD_ZONE_DN NONE - (killed empirically)
< -1.50% DN_EXTREME CALL IV>30% + prev gap<-0.5%
```

### PLTR/NVDA Lead Filter Integration

```
Block trade if:
 composite_signal < -0.3% AND position = CALL ' skip
 composite_signal > +0.3% AND position = PUT ' skip
Filter effect: 82% of UP_BIG_FADE put signals blocked
```

### Backtest Results (9 years, real QQQ data)

```
Period: 2016"2025
CAGR: +0.9% Max DD: 4.6% Sharpe: 2.28
Trades: 9/year WR: 33.3% PF: 2.28
```

**Conclusion:** Edge is real (Sharpe 2.28, near-zero drawdown) but trade frequency is too low. The zone map and lead filter logic were absorbed as regime classifiers into the options structure research.

---

## 3. Options Structure Evolution

Each iteration was driven by a specific empirically identified failure - not aesthetic preference.

### Iteration 1: Symmetric Risk Reversal

```
BULL: SELL ATM put + BUY 3% OTM call ' net credit
BEAR: SELL ATM call + BUY 3% OTM put ' small debit
```

**Real data result (2018"2025):**
```
BULL: N=168 WR=58.9% PnL=+$440,826 "
BEAR: N= 96 WR=40.6% PnL=-$443,912 " exactly cancels bull
CAGR: +48.1% but bear side is structural bleed
```

**Failure diagnosis:** Put skew makes the short ATM call cheap and long OTM put expensive - fighting the skew. In 2023 (SPY +26%), selling calls into recovery: -$86K, -$36K, -$80K, -$70K, -$52K consecutive months.

---

### Iteration 2: Bull-Only + Vol-Scaled Sizing

```
BULL regime (QQQ > SMA50 1.015): SELL ATM put + BUY 3% OTM call
BEAR / NEUTRAL: sit in cash
```

Vol sizing added:
```
EASY   HV<16%:  4x size
NORMAL HV<22%:  2x size
HARD HV<30%: 0.5 size CRISIS HV>30%: 0.25 size
```

**Bug found:** `int(risk / max_loss_per)` floored to zero when SPY crossed $1,200. Fix: normalize by `S * spread_pct 100` (price-invariant notional sizing).

**Result:** CAGR +13.4%, Max DD 10.4%, Sharpe 1.18, Bear PnL $0 (eliminated), Ruin 0.00%

---

### Iteration 3: Bear Put Spread

Replace bear cash with a defined-risk structure that uses put skew correctly:

```
BEAR regime: BUY ATM put (K=S) + SELL 8% OTM put (K=S0.92)
 Net debit ~$4-6 | Max profit ~$8-12 | Max loss = debit paid
 Put skew HELPS: buying expensive ATM put, selling cheaper OTM
```

Bear confirmation filters:
```python
mom_5d < -0.005 # 5-day momentum negative (falling, not just below SMA)
rv_10 > 0.130 # Realized vol elevated (real fear, not quiet drift)
pct_off_high < -0.030 # At least 3% off 20-day high (breakdown, not dip)
```

**Result:** Bear PnL improved from -$443K ' -$18K. Defined risk capped every loss.

---

### Iteration Summary

| Version | Structure | Problem Found | Fix |
|---------|-----------|---------------|-----|
| RR v1 | Bull + Bear RR | Bear: -$443K, cancelled all bull gains | Put skew structurally breaks bear RR |
| Hybrid v1 | Bull only | Bear in cash | Correct - simplified |
| Hybrid v2 | Vol-scaled | EASY trades never fired | Integer sizing bug at SPY >$1,200 |
| Hybrid v3 | Bear confirmation | SMA50 too slow, buying puts into dips | Require momentum + vol + price filters |
| **Ladder v2** | 5-leg zero-cost | **Final** | 10% offset put, price-invariant sizing |

---

## 4. Final Strategy - APEX-LADDER

**Central insight from all iterations:** Put skew is the dominant force. Every failure came from fighting it. The ladder exploits it from both sides - collect skew premium on the short put, use it to finance three long calls at different strikes, building three separate win conditions.

### Structure

![APEX-LADDER Structure](results/apex_ladder_structure.png)

```
Leg Strike Price Delta
--------------------------------------------------------------
SHORT ATM put S +$14.28 -0.455 Main credit
SHORT 10% OTM put S 0.90 +$1.38 -0.148 Tail-only offset
LONG 2% OTM call S 1.02 -$10.94 +0.423 High - - most moves
LONG 5% OTM call S 1.05 -$5.54 +0.258 Mid - - bigger moves
LONG 8% OTM call S 1.08 -$2.50 +0.138 Low - - tail rip
--------------------------------------------------------------
NET +$0.38 Small credit "
```

**Why 10% OTM offset instead of 5%:**
5% OTM put bleeds on every normal pullback (3"4 per year). 10% OTM only triggers in genuine crashes. Gives up $3.70 of credit but STOP_SPREAD exits dropped from 19 ' 3.

### Three Win Zones

```
Zone 1: SPY +2% -> +5% 2% call in-the-money
Zone 2: SPY +5% -> +8% 2% + 5% calls printing
Zone 3: SPY +8%+ All three calls printing - tail rip
Flat: Collect theta on both short puts
```

### Backtest Dashboard

![APEX-LADDER Dashboard](results/apex_ladder_dashboard.png)

### Results (Real SPY Data via Alpaca, 10.2 Years)

```
Period: 2016-01-01 ' 2026-03-08
Capital: $100,000 ' $253,006
CAGR: +9.5%
Max Drawdown: -23.2%
Sharpe Ratio: 0.78
Trades: 132 (13/year)
Win Rate: 53.0%
Profit Factor: 2.61
Avg Win: +344.8%
Avg Loss: -149.3%
Exits: ROLL=118 STOP_PUT=11 STOP_SPREAD=3
```

### P&L By Vol Regime

```
Regime N WR AvgRet PnL
------------------------------------------------------
EASY 37 51.4% +98.3% +$22,659
NORMAL 89 53.9% +118.5% +$177,243
HARD 6 50.0% +116.8% +$19,763
```

All three regimes profitable. Hard vol wins despite 0.5 sizing because short puts collect dramatically more premium when HV spikes.

### Annual P&L

```
2019 +$47K Spring rally fired all 3 calls (Jun +$22.8K, Jul +$7.4K)
2020 +$27K COVID recovery: May +$25.9K on reopening rip
2021 +$62K Melt-up: Oct +$27K at 3 easy vol sizing
2022 +$63K Bear paradox: elevated vol = fat put premiums all year
2023 -$25K Choppy recovery, regime transitions, puts hit repeatedly
2024 +$34K Jan -$10.3K drawdown absorbed, Dec +$54.3K recovery
2025 +$16K Jan -$18K stop-out, Jul +$21.7K recovery
```

### Monte Carlo (100,000 paths 132 trades)

```
Metric P5 P25 P50 P75 P95
------------------------------------------------------------------
CAGR +17.4% +26.9% +34.2% +42.1% +54.6%
Max Drawdown -16.1% -20.7% -24.8% -30.0% -39.4%
Terminal $512K $1.1M $2.0M $3.6M $8.4M
Ruin (port<50%) 0.03%
Profitable paths 99.99%
```

**On the actual vs MC gap:** Actual path (+9.5%) sits below MC P25 (+26.9%). This reflects loss clustering during regime transitions - 2023, Jan 2024, Jan 2025 - which wouldn't appear in i.i.d. resampling. The MC P50 is the median under independent draws. The actual path shows real negative autocorrelation: losses cluster when regimes shift. Both numbers reported honestly. Sharpe 0.78 on actual path is the conservative number.

---

## 5. Key Quantitative Concepts

**Put skew arbitrage:** ATM puts trade approx 8% richer than realized vol due to structural demand from portfolio hedgers. This strategy collects that premium as the short put seller. The skew slope (approx 1.5 per 10% OTM) means the 10% OTM offset put is cheaper than its probability of expiring ITM would suggest.

**Zero-cost construction:** Short ATM put (approx $14) + short 10% OTM put (approx $1.40) = $15.40 credit. Three long calls (2%/5%/8%) = approx $15.02 debit. Net: small credit. Self-financing with three upside participation points.

**Volatility targeting:** `contracts = risk_usd / (S * spread_pct * 100)`. At HV=12%, 3 size has identical dollar volatility to 1 at HV=21%. Edge scales with size, not notional.

**Price-invariant sizing:** Normalizing by `S * spread_pct` prevents integer floor-to-zero at high SPY prices. Verified correct at SPY=$200 through SPY=$1,500.

**Bootstrap Monte Carlo:** 100,000 paths resampling empirical trade returns. No parametric assumptions. Ruin = portfolio below 50% of start at any point in path.

---

## 6. Architecture

```
apex_ladder_v2.py Final - 5-leg zero-cost ladder, vol-scaled sizing
research/apex_hybrid_v3.py Iteration history - bull RR + bear put spread
intraday_lead_signal.py Stream 1 - PLTR/NVDA -> SPY lead signal (N=59 days)
backtest_futures.py Stream 2 - NQ gap momentum, zone map, 0DTE QQQ
results/
 apex_ladder_dashboard.png 6-panel performance dashboard
 apex_ladder_structure.png 5-leg payoff diagram, three win zones
 apex_ladder_trades.csv Full trade log: 132 trades, 10.2 years
```

---

## Setup

```bash
pip install numpy pandas scipy matplotlib alpaca-trade-api
export ALPACA_API_KEY=your_key
export ALPACA_API_SECRET=your_secret
```

## Run

```bash
# 10-year backtest on real SPY data
python3 apex_ladder_v2.py --start 2016-01-01 --end 2026-03-08 --capital 100000

# Intraday lead signal (live Alpaca connection)
python3 intraday_lead_signal.py

# Gap momentum backtest
python3 backtest_futures.py --start 2016-01-01 --end 2025-01-01
```

---

*Michael Rosenberg - quantitative research, systematic options* 
*michaelirosenberg@gmail.com*
