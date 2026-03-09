"""
PLTR + NVDA First-15min → SPY Next-60min Lead Signal
======================================================
Hypothesis: The combined move of PLTR and NVDA in the first 15 minutes
of the session predicts SPY's direction over the following 60 minutes.

Mechanism:
  - Institutional order flow hits single high-beta names first
  - Index rebalancing / ETF arbitrage lags 15-30 min
  - PLTR (AI/defense) + NVDA (AI/semis) = composite risk appetite signal

Run: python3 intraday_lead_signal.py
Requires: pip3 install yfinance pandas numpy scipy matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ── FETCH 60-DAY 5-MIN BARS ────────────────────────────────────────────────────
print("Fetching 5-min intraday data (60 days)...")
import yfinance as yf

def fetch(ticker):
    raw = yf.download(ticker, period="60d", interval="5m",
                      auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0].lower() for c in raw.columns]
    else:
        raw.columns = [c.lower() for c in raw.columns]
    raw.index = pd.to_datetime(raw.index)
    # Convert to Eastern time
    if raw.index.tzinfo is not None:
        raw.index = raw.index.tz_convert("America/New_York")
    else:
        raw.index = raw.index.tz_localize("UTC").tz_convert("America/New_York")
    return raw

pltr = fetch("PLTR")
nvda = fetch("NVDA")
spy  = fetch("SPY")
qqq  = fetch("QQQ")

print(f"  PLTR: {len(pltr)} bars")
print(f"  NVDA: {len(nvda)} bars")
print(f"  SPY:  {len(spy)} bars")

# ── BUILD DAILY SIGNAL WINDOWS ─────────────────────────────────────────────────
# Window A: 9:30 → 9:45  (first 15 min — 3 bars of 5min)
# Window B: 9:45 → 10:45 (next 60 min — 12 bars of 5min)  ← target

records = []

# Get all unique trading dates
dates = sorted(set(spy.index.date))

for d in dates:
    try:
        # Filter to this date, regular hours only (9:30-16:00)
        def day_bars(df, date):
            mask = (df.index.date == date) & \
                   (df.index.time >= pd.Timestamp("09:30").time()) & \
                   (df.index.time <= pd.Timestamp("15:55").time())
            return df[mask]

        sp = day_bars(spy,  d)
        pl = day_bars(pltr, d)
        nv = day_bars(nvda, d)

        if len(sp) < 15 or len(pl) < 3 or len(nv) < 3:
            continue

        # ── WINDOW A: first 15 min (9:30 open → 9:45 close) ──────────────────
        # open = first bar open, close_15 = bar ending at 9:45
        spy_open   = sp["open"].iloc[0]
        pltr_open  = pl["open"].iloc[0]
        nvda_open  = nv["open"].iloc[0]

        # 9:45 bar = index 2 (bars at 9:30, 9:35, 9:40, so close of 9:40 bar = 9:45 mark)
        spy_15     = sp["close"].iloc[2]
        pltr_15    = pl["close"].iloc[2]
        nvda_15    = nv["close"].iloc[2]

        spy_ret_15  = (spy_15  - spy_open)  / spy_open
        pltr_ret_15 = (pltr_15 - pltr_open) / pltr_open
        nvda_ret_15 = (nvda_15 - nvda_open) / nvda_open

        # ── WINDOW B: next 60 min (9:45 → 10:45) — SPY only ─────────────────
        # bars 3-14 (12 bars of 5min)
        if len(sp) < 15:
            continue
        spy_1045   = sp["close"].iloc[14]  # bar ending at ~10:45
        spy_ret_60 = (spy_1045 - spy_15) / spy_15

        # ── COMPOSITE SIGNAL ─────────────────────────────────────────────────
        # Equal weight PLTR + NVDA first-15 return
        composite       = (pltr_ret_15 + nvda_ret_15) / 2
        # Divergence: composite vs SPY first-15 (are they leading?)
        divergence      = composite - spy_ret_15
        # Normalized: composite / SPY (1.0 = in line, >1 = leading up, <1 = lagging)
        lead_ratio      = composite / spy_ret_15 if abs(spy_ret_15) > 0.0001 else np.nan

        records.append({
            "date":           d,
            "spy_ret_15":     spy_ret_15,
            "pltr_ret_15":    pltr_ret_15,
            "nvda_ret_15":    nvda_ret_15,
            "composite_15":   composite,
            "divergence_15":  divergence,
            "spy_ret_60":     spy_ret_60,   # ← TARGET
        })

    except Exception as e:
        continue

df = pd.DataFrame(records).dropna()
N  = len(df)
print(f"\n  Built {N} complete trading days")

# ── SIGNAL DEFINITIONS ─────────────────────────────────────────────────────────
# Signal 1: Raw composite direction
df["sig_composite_dir"]  = np.sign(df["composite_15"])

# Signal 2: Composite stronger than SPY (leading)
df["sig_lead"]           = np.sign(df["divergence_15"])

# Signal 3: Both PLTR and NVDA agree AND stronger than SPY
df["sig_both_lead"]      = ((df["pltr_ret_15"] > df["spy_ret_15"]) &
                             (df["nvda_ret_15"] > df["spy_ret_15"])).astype(int) - \
                            ((df["pltr_ret_15"] < df["spy_ret_15"]) &
                             (df["nvda_ret_15"] < df["spy_ret_15"])).astype(int)

# Signal 4: Large composite move (>0.3% in 15min = strong conviction)
threshold = 0.003
df["sig_strong_up"]   = (df["composite_15"] >  threshold).astype(int)
df["sig_strong_dn"]   = (df["composite_15"] < -threshold).astype(int)
df["sig_strong"]      = df["sig_strong_up"] - df["sig_strong_dn"]

# ── CORRELATION ANALYSIS ──────────────────────────────────────────────────────
print()
print("=" * 65)
print("INTRADAY LEAD SIGNAL — PLTR + NVDA → SPY")
print(f"Period: {df['date'].min()} → {df['date'].max()}  N={N} days")
print("=" * 65)

# Base correlations
r_pltr,  p_pltr  = stats.pearsonr(df["pltr_ret_15"],   df["spy_ret_60"])
r_nvda,  p_nvda  = stats.pearsonr(df["nvda_ret_15"],   df["spy_ret_60"])
r_comp,  p_comp  = stats.pearsonr(df["composite_15"],  df["spy_ret_60"])
r_spy15, p_spy15 = stats.pearsonr(df["spy_ret_15"],    df["spy_ret_60"])
r_div,   p_div   = stats.pearsonr(df["divergence_15"], df["spy_ret_60"])

print(f"\n{'CORRELATION → SPY next 60min':}")
print(f"  SPY  first-15  (baseline):    ρ={r_spy15:+.3f}  p={p_spy15:.3f}  "
      f"{'✅ sig' if p_spy15 < 0.05 else '❌ insig'}")
print(f"  PLTR first-15:                ρ={r_pltr:+.3f}  p={p_pltr:.3f}  "
      f"{'✅ sig' if p_pltr  < 0.05 else '❌ insig'}")
print(f"  NVDA first-15:                ρ={r_nvda:+.3f}  p={p_nvda:.3f}  "
      f"{'✅ sig' if p_nvda  < 0.05 else '❌ insig'}")
print(f"  COMPOSITE (PLTR+NVDA)/2:      ρ={r_comp:+.3f}  p={p_comp:.3f}  "
      f"{'✅ sig' if p_comp  < 0.05 else '❌ insig'}")
print(f"  DIVERGENCE (comp - SPY15):    ρ={r_div:+.3f}   p={p_div:.3f}  "
      f"{'✅ sig' if p_div   < 0.05 else '❌ insig'}")

# ── DIRECTIONAL SIGNAL ACCURACY ───────────────────────────────────────────────
print(f"\n{'DIRECTIONAL SIGNAL ACCURACY':}")

for sig_name, sig_col in [
    ("Composite direction",       "sig_composite_dir"),
    ("PLTR+NVDA lead SPY",        "sig_lead"),
    ("Both lead (agreement)",     "sig_both_lead"),
    ("Strong move >0.3%",         "sig_strong"),
]:
    mask = df[sig_col] != 0
    sub  = df[mask]
    if len(sub) < 5:
        continue
    # Correct = signal direction matches SPY next 60min direction
    correct = (sub[sig_col] * np.sign(sub["spy_ret_60"]) > 0)
    wr      = correct.mean()
    n       = len(sub)
    # Binomial test
    binom   = stats.binomtest(correct.sum(), n, p=0.5)
    avg_ret_when_correct = sub.loc[correct,  "spy_ret_60"].mean()
    avg_ret_when_wrong   = sub.loc[~correct, "spy_ret_60"].mean()

    print(f"\n  {sig_name}  (N={n})")
    print(f"    Win rate:   {wr:.1%}  (p={binom.pvalue:.3f} vs 50%  "
          f"{'✅ sig' if binom.pvalue < 0.10 else '❌ insig'})")
    print(f"    Avg SPY ret when correct: {avg_ret_when_correct:+.3%}")
    print(f"    Avg SPY ret when wrong:   {avg_ret_when_wrong:+.3%}")

# ── STRONG SIGNAL BREAKDOWN ───────────────────────────────────────────────────
print(f"\n{'STRONG COMPOSITE SIGNAL (|composite| > 0.3%) BREAKDOWN':}")
up_days  = df[df["sig_strong"] ==  1]
dn_days  = df[df["sig_strong"] == -1]
flat_days= df[df["sig_strong"] ==  0]

for label, sub in [("STRONG UP  (n=%d)" % len(up_days),   up_days),
                   ("STRONG DN  (n=%d)" % len(dn_days),   dn_days),
                   ("FLAT/MIXED (n=%d)" % len(flat_days), flat_days)]:
    if len(sub) < 3: continue
    wr  = (np.sign(sub["sig_strong"]) * np.sign(sub["spy_ret_60"]) > 0).mean() \
          if sub["sig_strong"].abs().sum() > 0 else np.nan
    avg = sub["spy_ret_60"].mean()
    std = sub["spy_ret_60"].std()
    print(f"  {label}")
    print(f"    SPY next-60 avg: {avg:+.3%}  std: {std:.3%}  "
          f"P(SPY>0): {(sub['spy_ret_60']>0).mean():.0%}")

# ── REGIME SPLIT ──────────────────────────────────────────────────────────────
print(f"\n{'SIGNAL QUALITY BY SPY REGIME (first-15 direction)':}")
spy_up_days = df[df["spy_ret_15"] > 0]
spy_dn_days = df[df["spy_ret_15"] < 0]

for label, sub in [("SPY opens UP (n=%d)"%len(spy_up_days), spy_up_days),
                   ("SPY opens DN (n=%d)"%len(spy_dn_days), spy_dn_days)]:
    r, p = stats.pearsonr(sub["composite_15"], sub["spy_ret_60"])
    print(f"  {label}:  composite→SPY60 ρ={r:+.3f}  p={p:.3f}  "
          f"{'✅' if p<0.10 else '❌'}")

# ── REGRESSION: INCREMENTAL VALUE ─────────────────────────────────────────────
print(f"\n{'INCREMENTAL PREDICTIVE VALUE (OLS)':}")
from numpy.linalg import lstsq

def ols(X, y):
    X_ = np.column_stack([np.ones(len(X)), X])
    coef, _, _, _ = lstsq(X_, y, rcond=None)
    yhat = X_ @ coef
    ss_res = ((y - yhat)**2).sum()
    ss_tot = ((y - y.mean())**2).sum()
    r2 = 1 - ss_res/ss_tot
    return coef, r2

y = df["spy_ret_60"].values

_, r2_spy  = ols(df["spy_ret_15"].values.reshape(-1,1), y)
_, r2_comp = ols(df["composite_15"].values.reshape(-1,1), y)
_, r2_both = ols(np.column_stack([df["spy_ret_15"].values,
                                   df["composite_15"].values]), y)
_, r2_all  = ols(np.column_stack([df["spy_ret_15"].values,
                                   df["pltr_ret_15"].values,
                                   df["nvda_ret_15"].values]), y)

print(f"  SPY first-15 alone:          R²={r2_spy:.3f}")
print(f"  Composite alone:             R²={r2_comp:.3f}")
print(f"  SPY + Composite:             R²={r2_both:.3f}  "
      f"(+{r2_both-r2_spy:.3f} incremental)")
print(f"  SPY + PLTR + NVDA separate:  R²={r2_all:.3f}  "
      f"(+{r2_all-r2_spy:.3f} incremental)")

print()
print("=" * 65)

# ── PLOT ──────────────────────────────────────────────────────────────────────
BG  = "#0d1117"; PAN = "#161b22"; GR = "#22c55e"; RD = "#ef4444"
BL  = "#3b82f6"; YL  = "#eab308"; OR = "#f97316"; MU = "#94a3b8"; WH = "#f1f5f9"

fig = plt.figure(figsize=(20, 12), facecolor=BG)
fig.suptitle("PLTR + NVDA First-15min → SPY Next-60min Lead Signal",
             color=WH, fontsize=14, fontweight="bold", y=0.98)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.30)

def style(ax, title):
    ax.set_facecolor(PAN)
    for sp in ax.spines.values(): sp.set_color("#2d3748")
    ax.tick_params(colors=MU, labelsize=8)
    ax.set_title(title, color=WH, fontsize=9, fontweight="bold", pad=6)
    ax.grid(True, color="#2d3748", lw=0.4, alpha=0.5)

# 1. Composite vs SPY next-60 scatter
ax1 = fig.add_subplot(gs[0, 0])
style(ax1, f"Composite 15min → SPY 60min  ρ={r_comp:+.3f}")
c_arr = np.where(df["spy_ret_60"] > 0, GR, RD)
ax1.scatter(df["composite_15"]*100, df["spy_ret_60"]*100,
            c=c_arr, alpha=0.5, s=20, edgecolors="none")
m, b_, *_ = stats.linregress(df["composite_15"], df["spy_ret_60"])
xs = np.linspace(df["composite_15"].min(), df["composite_15"].max(), 50)
ax1.plot(xs*100, (m*xs+b_)*100, color=YL, lw=1.8)
ax1.axhline(0, color=MU, lw=0.5); ax1.axvline(0, color=MU, lw=0.5)
ax1.set_xlabel("Composite 15min ret %", color=MU, fontsize=8)
ax1.set_ylabel("SPY next 60min ret %",  color=MU, fontsize=8)

# 2. Divergence vs SPY next-60 scatter
ax2 = fig.add_subplot(gs[0, 1])
style(ax2, f"Divergence (comp-SPY15) → SPY60  ρ={r_div:+.3f}")
ax2.scatter(df["divergence_15"]*100, df["spy_ret_60"]*100,
            c=c_arr, alpha=0.5, s=20, edgecolors="none")
m2, b2, *_ = stats.linregress(df["divergence_15"], df["spy_ret_60"])
xs2 = np.linspace(df["divergence_15"].min(), df["divergence_15"].max(), 50)
ax2.plot(xs2*100, (m2*xs2+b2)*100, color=OR, lw=1.8)
ax2.axhline(0, color=MU, lw=0.5); ax2.axvline(0, color=MU, lw=0.5)
ax2.set_xlabel("Divergence (comp - SPY15) %", color=MU, fontsize=8)
ax2.set_ylabel("SPY next 60min ret %",        color=MU, fontsize=8)

# 3. Win rate by composite quintile
ax3 = fig.add_subplot(gs[0, 2])
df["comp_q"] = pd.qcut(df["composite_15"], 5,
                        labels=["Q1\nStrong DN","Q2","Q3\nFlat","Q4","Q5\nStrong UP"])
q_wr  = df.groupby("comp_q", observed=True).apply(
            lambda x: (x["spy_ret_60"] > 0).mean())
q_n   = df.groupby("comp_q", observed=True).size()
colors_q = [RD, "#f97316", MU, "#84cc16", GR]
bars = ax3.bar(range(5), q_wr.values, color=colors_q, alpha=0.85, edgecolor=BG)
ax3.axhline(0.5, color=WH, lw=0.8, ls="--", alpha=0.6, label="50% baseline")
for i, (wr_v, n_v) in enumerate(zip(q_wr.values, q_n.values)):
    ax3.text(i, wr_v + 0.01, f"{wr_v:.0%}\nn={n_v}",
             ha="center", va="bottom", color=WH, fontsize=7)
ax3.set_xticks(range(5))
ax3.set_xticklabels(q_wr.index, color=MU, fontsize=7)
ax3.set_ylabel("P(SPY next-60 > 0)", color=MU, fontsize=8)
style(ax3, "SPY 60min Win Rate by Composite Quintile")
ax3.legend(fontsize=7, facecolor=PAN, labelcolor=WH, edgecolor="#2d3748")
ax3.set_ylim(0, 1)

# 4. Time series: composite signal vs SPY 60-min outcome
ax4 = fig.add_subplot(gs[1, :2])
style(ax4, "Composite Signal vs SPY 60min Outcome (daily)")
x = range(len(df))
ax4.bar(x, df["composite_15"]*100,
        color=[GR if v>0 else RD for v in df["composite_15"]],
        alpha=0.6, label="Composite 15min %", width=0.8)
ax4.plot(x, df["spy_ret_60"]*100, color=YL, lw=1.2, alpha=0.8,
         label="SPY next-60 %")
ax4.axhline(0, color=MU, lw=0.5)
ax4.set_xlabel("Trading days (recent 60d)", color=MU, fontsize=8)
ax4.set_ylabel("Return %", color=MU, fontsize=8)
ax4.legend(fontsize=8, facecolor=PAN, labelcolor=WH, edgecolor="#2d3748")

# 5. PLTR vs NVDA individual predictive power
ax5 = fig.add_subplot(gs[1, 2])
style(ax5, "Individual Signal Predictive Power")
labels  = ["SPY\nfirst-15", "PLTR\nfirst-15", "NVDA\nfirst-15",
           "Composite", "SPY+\nComposite"]
r2s     = [r2_spy, 
           stats.pearsonr(df["pltr_ret_15"],df["spy_ret_60"])[0]**2,
           stats.pearsonr(df["nvda_ret_15"],df["spy_ret_60"])[0]**2,
           r2_comp, r2_both]
cols    = [MU, BL, OR, YL, GR]
ax5.bar(range(len(labels)), [v*100 for v in r2s],
        color=cols, alpha=0.85, edgecolor=BG)
for i, v in enumerate(r2s):
    ax5.text(i, v*100 + 0.05, f"{v*100:.1f}%",
             ha="center", va="bottom", color=WH, fontsize=8)
ax5.set_xticks(range(len(labels)))
ax5.set_xticklabels(labels, color=MU, fontsize=8)
ax5.set_ylabel("R² (%)", color=MU, fontsize=8)

plt.savefig("intraday_lead_signal.png", dpi=150,
            bbox_inches="tight", facecolor=BG)
plt.close()
print("Chart saved → intraday_lead_signal.png")
