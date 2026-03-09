"""
PLTR vs SPY — OHLC Correlation & Divergence Analysis
=====================================================
Run: python3 pltr_spy_corr.py
Requires: pip install yfinance pandas numpy scipy matplotlib seaborn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.signal import correlate
import warnings
warnings.filterwarnings("ignore")

# ── FETCH ──────────────────────────────────────────────────────────────────────
print("Fetching data...")
import yfinance as yf

raw_pltr = yf.download("PLTR", period="2y", interval="1d", auto_adjust=True, progress=False)
raw_spy  = yf.download("SPY",  period="2y", interval="1d", auto_adjust=True, progress=False)

# yfinance 1.2+ returns MultiIndex columns — flatten to single level
if isinstance(raw_pltr.columns, pd.MultiIndex):
    raw_pltr.columns = [c[0].lower() for c in raw_pltr.columns]
else:
    raw_pltr.columns = [c.lower() for c in raw_pltr.columns]

if isinstance(raw_spy.columns, pd.MultiIndex):
    raw_spy.columns = [c[0].lower() for c in raw_spy.columns]
else:
    raw_spy.columns = [c.lower() for c in raw_spy.columns]

# Align dates
idx = raw_pltr.index.intersection(raw_spy.index)
pltr = raw_pltr.loc[idx].copy()
spy  = raw_spy.loc[idx].copy()
N    = len(idx)
print(f"  {N} trading days loaded  ({idx[0].date()} → {idx[-1].date()})")

# ── FEATURE ENGINEERING ────────────────────────────────────────────────────────
def features(df, name):
    f = pd.DataFrame(index=df.index)
    f[f"{name}_open"]       = df["open"]
    f[f"{name}_high"]       = df["high"]
    f[f"{name}_low"]        = df["low"]
    f[f"{name}_close"]      = df["close"]
    f[f"{name}_ret"]        = df["close"].pct_change()
    f[f"{name}_gap"]        = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)
    f[f"{name}_range"]      = (df["high"] - df["low"]) / df["open"]       # daily range %
    f[f"{name}_body"]       = (df["close"] - df["open"]) / df["open"]     # candle body %
    f[f"{name}_upper_wick"] = (df["high"] - np.maximum(df["open"], df["close"])) / df["open"]
    f[f"{name}_lower_wick"] = (np.minimum(df["open"], df["close"]) - df["low"]) / df["open"]
    f[f"{name}_close_pos"]  = (df["close"] - df["low"]) / (df["high"] - df["low"])  # 0=low, 1=high
    return f

fp = features(pltr, "pltr")
fs = features(spy,  "spy")
df = pd.concat([fp, fs], axis=1).dropna()

# ── ROLLING CORRELATIONS ───────────────────────────────────────────────────────
windows = [10, 21, 63]

roll_corr = {}
for w in windows:
    roll_corr[w] = df["pltr_ret"].rolling(w).corr(df["spy_ret"])

# ── PAIRWISE OHLC CORRELATIONS ────────────────────────────────────────────────
pltr_cols = ["pltr_ret", "pltr_gap", "pltr_range", "pltr_body",
             "pltr_upper_wick", "pltr_lower_wick", "pltr_close_pos"]
spy_cols  = ["spy_ret",  "spy_gap",  "spy_range",  "spy_body",
             "spy_upper_wick",  "spy_lower_wick",  "spy_close_pos"]

corr_matrix = df[pltr_cols + spy_cols].corr().loc[pltr_cols, spy_cols]

# ── DIVERGENCE SIGNAL ─────────────────────────────────────────────────────────
# Z-score spread between normalized prices
pltr_norm = (pltr["close"] - pltr["close"].rolling(63).mean()) / pltr["close"].rolling(63).std()
spy_norm  = (spy["close"]  - spy["close"].rolling(63).mean())  / spy["close"].rolling(63).std()
spread    = pltr_norm - spy_norm

spread_z  = (spread - spread.rolling(21).mean()) / spread.rolling(21).std()

# Divergence events: |z| > 2
div_events = spread_z[abs(spread_z) > 2.0]

# ── LEAD/LAG ANALYSIS ─────────────────────────────────────────────────────────
pltr_r = df["pltr_ret"].values
spy_r  = df["spy_ret"].values
lags   = range(-10, 11)
lag_corrs = {}
for lag in lags:
    if lag < 0:
        lag_corrs[lag] = np.corrcoef(pltr_r[:lag], spy_r[-lag:])[0,1]
    elif lag > 0:
        lag_corrs[lag] = np.corrcoef(pltr_r[lag:], spy_r[:-lag])[0,1]
    else:
        lag_corrs[lag] = np.corrcoef(pltr_r, spy_r)[0,1]

# ── REGIME ANALYSIS ───────────────────────────────────────────────────────────
# Split into bull/bear based on SPY 50-day SMA
spy_sma50 = spy["close"].rolling(50).mean()
bull_days  = spy["close"] > spy_sma50
bear_days  = spy["close"] <= spy_sma50

bull_corr  = df.loc[bull_days.loc[df.index], ["pltr_ret", "spy_ret"]].corr().iloc[0,1]
bear_corr  = df.loc[bear_days.loc[df.index], ["pltr_ret", "spy_ret"]].corr().iloc[0,1]

# ── CONDITIONAL CORRELATIONS ──────────────────────────────────────────────────
# When SPY makes large moves, how does PLTR respond?
big_up_spy   = df["spy_ret"] > df["spy_ret"].quantile(0.90)
big_dn_spy   = df["spy_ret"] < df["spy_ret"].quantile(0.10)
small_spy    = df["spy_ret"].abs() < df["spy_ret"].abs().quantile(0.33)

corr_big_up  = df.loc[big_up_spy,  ["pltr_ret","spy_ret"]].corr().iloc[0,1]
corr_big_dn  = df.loc[big_dn_spy,  ["pltr_ret","spy_ret"]].corr().iloc[0,1]
corr_small   = df.loc[small_spy,   ["pltr_ret","spy_ret"]].corr().iloc[0,1]

# Beta estimates
def beta(pltr_r, spy_r):
    cov = np.cov(pltr_r, spy_r)
    return cov[0,1] / cov[1,1]

beta_full    = beta(df["pltr_ret"].values, df["spy_ret"].values)
beta_bull    = beta(df.loc[big_up_spy,"pltr_ret"].values, df.loc[big_up_spy,"spy_ret"].values)
beta_bear    = beta(df.loc[big_dn_spy,"pltr_ret"].values, df.loc[big_dn_spy,"spy_ret"].values)
beta_small   = beta(df.loc[small_spy,"pltr_ret"].values,  df.loc[small_spy,"spy_ret"].values)

# ── PRINT REPORT ──────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("PLTR vs SPY  —  CORRELATION ANALYSIS")
print("=" * 65)

print(f"\n{'RETURN CORRELATION':}")
print(f"  Full period:           {df['pltr_ret'].corr(df['spy_ret']):+.3f}")
print(f"  Bull regime (SPY>SMA50): {bull_corr:+.3f}")
print(f"  Bear regime (SPY<SMA50): {bear_corr:+.3f}")
print(f"  Big UP days (SPY top 10%): {corr_big_up:+.3f}")
print(f"  Big DN days (SPY bot 10%): {corr_big_dn:+.3f}")
print(f"  Small moves (SPY bot 33%): {corr_small:+.3f}")

print(f"\n{'BETA':}")
print(f"  Full period beta:        {beta_full:+.2f}x")
print(f"  Up-day beta:             {beta_bull:+.2f}x")
print(f"  Down-day beta:           {beta_bear:+.2f}x")
print(f"  Small-move beta:         {beta_small:+.2f}x")
asymmetry = beta_bull - beta_bear
print(f"  Up/Down asymmetry:       {asymmetry:+.2f}  ({'PLTR amplifies UP more' if asymmetry>0 else 'PLTR amplifies DN more'})")

print(f"\n{'LEAD/LAG (negative = PLTR leads SPY)':}")
best_lag  = max(lag_corrs, key=lambda k: abs(lag_corrs[k]))
for lag in lags:
    bar = "█" * int(abs(lag_corrs[lag]) * 30)
    sign = "+" if lag_corrs[lag] > 0 else ""
    marker = " ◄ PEAK" if lag == best_lag else ""
    print(f"  lag {lag:+3d}d:  {sign}{lag_corrs[lag]:.3f}  {bar}{marker}")

print(f"\n{'OHLC FEATURE CORRELATIONS (PLTR rows vs SPY cols)':}")
print(f"  {'':20s}", end="")
short_labels = ["ret","gap","range","body","u_wick","l_wick","close%"]
for lbl in short_labels:
    print(f"  {lbl:>7s}", end="")
print()
print("  " + "-"*75)
for pr in pltr_cols:
    short_row = pr.replace("pltr_","")
    print(f"  {short_row:<20s}", end="")
    for sc in spy_cols:
        v = corr_matrix.loc[pr, sc]
        # highlight strong correlations
        if abs(v) > 0.5:
            marker = "**"
        elif abs(v) > 0.35:
            marker = "* "
        else:
            marker = "  "
        print(f"  {v:+.3f}{marker}", end="")
    print()

print(f"\n{'DIVERGENCE EVENTS (|z-score| > 2.0)':}")
print(f"  Total events:  {len(div_events)}")
if len(div_events) > 0:
    print(f"  Current z:     {spread_z.iloc[-1]:+.2f}  ", end="")
    if spread_z.iloc[-1] > 2:
        print("→ PLTR extended ABOVE SPY (potential mean revert down)")
    elif spread_z.iloc[-1] < -2:
        print("→ PLTR extended BELOW SPY (potential mean revert up)")
    else:
        print("→ within normal range")
    recent = div_events.tail(5)
    print(f"  Recent events:")
    for dt, z in recent.items():
        print(f"    {dt.date()}  z={z:+.2f}  {'PLTR>SPY' if z>0 else 'PLTR<SPY'}")

print("=" * 65)

# ── PLOT ──────────────────────────────────────────────────────────────────────
BG  = "#0d1117"
PAN = "#161b22"
GR  = "#22c55e"
RD  = "#ef4444"
BL  = "#3b82f6"
YL  = "#eab308"
OR  = "#f97316"
PU  = "#a855f7"
MU  = "#94a3b8"
WH  = "#f1f5f9"

fig = plt.figure(figsize=(20, 14), facecolor=BG)
fig.suptitle("PLTR vs SPY — OHLC Correlation & Divergence", color=WH,
             fontsize=15, fontweight="bold", y=0.98)

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.32)

def style(ax, title):
    ax.set_facecolor(PAN)
    for sp in ax.spines.values(): sp.set_color("#2d3748")
    ax.tick_params(colors=MU, labelsize=8)
    ax.set_title(title, color=WH, fontsize=9, fontweight="bold", pad=6)
    ax.grid(True, color="#2d3748", lw=0.4, alpha=0.5)

dates = df.index

# 1. Normalized price paths
ax1 = fig.add_subplot(gs[0, :2])
style(ax1, "Normalised Price (rebased to 100)")
pltr_idx = pltr["close"] / pltr["close"].iloc[0] * 100
spy_idx  = spy["close"]  / spy["close"].iloc[0]  * 100
ax1.plot(pltr["close"].index, pltr_idx.loc[pltr["close"].index], color=BL, lw=1.5, label="PLTR")
ax1.plot(spy["close"].index,  spy_idx.loc[spy["close"].index],   color=YL, lw=1.5, label="SPY")
ax1.legend(fontsize=8, facecolor=PAN, labelcolor=WH, edgecolor="#2d3748")
ax1.set_ylabel("Rebased (100)", color=MU, fontsize=8)

# 2. Rolling correlation
ax2 = fig.add_subplot(gs[0, 2])
style(ax2, "Rolling Return Correlation")
colors_rc = [GR, YL, OR]
for i, w in enumerate(windows):
    ax2.plot(dates, roll_corr[w].loc[dates], color=colors_rc[i], lw=1.2,
             alpha=0.85, label=f"{w}d")
ax2.axhline(0, color=MU, lw=0.8, ls="--", alpha=0.5)
ax2.axhline(df["pltr_ret"].corr(df["spy_ret"]), color=WH, lw=0.8,
            ls=":", alpha=0.6, label="Full")
ax2.set_ylim(-1, 1)
ax2.legend(fontsize=7, facecolor=PAN, labelcolor=WH, edgecolor="#2d3748")

# 3. Divergence spread with z-score
ax3 = fig.add_subplot(gs[1, :2])
style(ax3, "PLTR−SPY Normalised Spread (z-score, 21d rolling)")
spread_plot = spread_z.reindex(dates)
ax3.fill_between(dates, spread_plot, 0,
                 where=spread_plot > 0, color=BL, alpha=0.25)
ax3.fill_between(dates, spread_plot, 0,
                 where=spread_plot < 0, color=RD, alpha=0.25)
ax3.plot(dates, spread_plot, color=WH, lw=1.0, alpha=0.8)
ax3.axhline( 2, color=OR, lw=0.9, ls="--", alpha=0.7, label="+2σ divergence")
ax3.axhline(-2, color=PU, lw=0.9, ls="--", alpha=0.7, label="-2σ divergence")
ax3.axhline( 0, color=MU, lw=0.6, ls="-",  alpha=0.4)
# Mark current
last_z = spread_z.iloc[-1]
ax3.scatter([dates[-1]], [last_z], color=YL, s=60, zorder=5,
            label=f"Now: z={last_z:+.2f}")
ax3.legend(fontsize=7, facecolor=PAN, labelcolor=WH, edgecolor="#2d3748")
ax3.set_ylabel("Z-score", color=MU, fontsize=8)

# 4. Lead/lag cross-correlation
ax4 = fig.add_subplot(gs[1, 2])
style(ax4, "Lead/Lag Cross-Correlation\n(neg lag = PLTR leads SPY)")
lag_vals = [lag_corrs[l] for l in lags]
colors_lag = [GR if v > 0 else RD for v in lag_vals]
bars = ax4.bar(list(lags), lag_vals, color=colors_lag, alpha=0.8, edgecolor=BG, width=0.7)
ax4.axvline(0, color=MU, lw=0.8, ls="--", alpha=0.6)
ax4.axhline(0, color=MU, lw=0.6)
# Highlight peak
peak_val = lag_corrs[best_lag]
ax4.bar([best_lag], [peak_val], color=YL, alpha=1.0, edgecolor=BG, width=0.7)
ax4.set_xlabel("Lag (days)", color=MU, fontsize=8)
ax4.set_ylabel("Correlation", color=MU, fontsize=8)
ax4.text(best_lag, peak_val + 0.01*np.sign(peak_val),
         f"Peak\nlag={best_lag:+d}", ha="center", va="bottom" if peak_val>0 else "top",
         color=YL, fontsize=7)

# 5. OHLC feature heatmap
ax5 = fig.add_subplot(gs[2, :2])
style(ax5, "OHLC Feature Cross-Correlation Heatmap (PLTR rows × SPY cols)")
mat = corr_matrix.values
im  = ax5.imshow(mat, cmap="RdYlGn", vmin=-0.8, vmax=0.8, aspect="auto")
ax5.set_xticks(range(len(spy_cols)))
ax5.set_xticklabels([c.replace("spy_","") for c in spy_cols], color=MU, fontsize=8)
ax5.set_yticks(range(len(pltr_cols)))
ax5.set_yticklabels([c.replace("pltr_","") for c in pltr_cols], color=MU, fontsize=8)
for i in range(len(pltr_cols)):
    for j in range(len(spy_cols)):
        v = mat[i,j]
        ax5.text(j, i, f"{v:+.2f}", ha="center", va="center",
                 color=WH if abs(v) > 0.45 else MU, fontsize=7.5, fontweight="bold")
plt.colorbar(im, ax=ax5, shrink=0.8).ax.tick_params(colors=MU, labelsize=7)

# 6. Beta asymmetry scatter
ax6 = fig.add_subplot(gs[2, 2])
style(ax6, f"Beta Asymmetry\nUp β={beta_bull:+.2f}x  Dn β={beta_bear:+.2f}x")
spy_r_arr  = df["spy_ret"].values * 100
pltr_r_arr = df["pltr_ret"].values * 100
c_arr = np.where(spy_r_arr > 0, BL, RD)
ax6.scatter(spy_r_arr, pltr_r_arr, c=c_arr, alpha=0.25, s=8, edgecolors="none")
# Regression lines up/down days
for mask, color, label, b in [
    (spy_r_arr > 0, BL, f"Up days β={beta_bull:+.2f}x", beta_bull),
    (spy_r_arr < 0, RD, f"Dn days β={beta_bear:+.2f}x", beta_bear),
]:
    xm = spy_r_arr[mask]
    ym = pltr_r_arr[mask]
    if len(xm) > 2:
        m, c_, *_ = stats.linregress(xm, ym)
        xs = np.linspace(xm.min(), xm.max(), 50)
        ax6.plot(xs, m*xs + c_, color=color, lw=1.8, label=label)
ax6.axhline(0, color=MU, lw=0.5)
ax6.axvline(0, color=MU, lw=0.5)
ax6.set_xlabel("SPY daily ret %", color=MU, fontsize=8)
ax6.set_ylabel("PLTR daily ret %", color=MU, fontsize=8)
ax6.legend(fontsize=7, facecolor=PAN, labelcolor=WH, edgecolor="#2d3748")

plt.savefig("pltr_spy_correlation.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("\nChart saved → pltr_spy_correlation.png")
