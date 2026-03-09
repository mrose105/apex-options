"""
GAP PULLBACK ANALYSIS
======================
Tests the hypothesis: QQQ gaps at open, pulls back 20-40%, then continues.

Measures on real Alpaca intraday data:
  - How often does a gap pull back 20-40% intraday?
  - After that pullback, what % of time does the gap continue?
  - How much does it continue (avg move from pullback low to close)?
  - What does the option look like at the pullback entry vs open entry?
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import date, timedelta
from data_providers import AlpacaProvider

# ── Load data ──────────────────────────────────────────────────────────────
provider = AlpacaProvider(symbol="QQQ", paper=True)

print("Loading QQQ daily bars...")
daily = provider.get_daily_bars(date(2020, 1, 1), date(2026, 3, 7))
daily["date"] = pd.to_datetime(daily["date"]).dt.date
daily = daily.dropna(subset=["gap_pct"]).reset_index(drop=True)

# ── Parameters ────────────────────────────────────────────────────────────
GAP_MIN       = 0.002   # minimum gap to consider (0.2%)
PB_LO         = 0.20    # pullback low threshold (20% of gap retraced)
PB_HI         = 0.40    # pullback high threshold (40% of gap retraced)
CONT_BARS     = 48      # bars after pullback to measure continuation (~4hrs)
MAX_PB_BAR    = 24      # pullback must occur within first 2 hours (bar 0-24)

print(f"Analysing {len(daily)} trading days...")
print(f"Gap threshold: >{GAP_MIN*100:.1f}%")
print(f"Pullback band: {PB_LO*100:.0f}–{PB_HI*100:.0f}% of gap retraced")
print()

results = []
gap_days = daily[daily["gap_pct"].abs() >= GAP_MIN].copy()
print(f"Gap days (>{GAP_MIN*100:.1f}%): {len(gap_days)}")

for _, row in gap_days.iterrows():
    tdate     = row["date"]
    gap_pct   = float(row["gap_pct"])
    qqq_open  = float(row["open"])
    direction = "call" if gap_pct > 0 else "put"

    # Load intraday bars
    intra = provider.get_intraday_bars(tdate)
    if len(intra) < 10:
        continue

    intra["close"] = intra["close"].astype(float)
    intra["open"]  = intra["open"].astype(float)
    intra["high"]  = intra["high"].astype(float)
    intra["low"]   = intra["low"].astype(float)

    open_px   = float(intra.iloc[0]["open"])
    gap_pts   = open_px * abs(gap_pct)   # gap in $ terms

    # For calls: pullback = price drops below open
    # For puts:  pullback = price rises above open
    pb_found     = False
    pb_bar       = None
    pb_price     = None
    pb_retrace   = None

    for i, bar in intra.iterrows():
        if i > MAX_PB_BAR:
            break
        if i == 0:
            continue   # skip first bar (that's the open itself)

        bar_low  = float(bar["low"])
        bar_high = float(bar["high"])

        if direction == "call":
            # Pullback = low dips below open
            retrace = (open_px - bar_low) / gap_pts
        else:
            # Pullback = high rises above open
            retrace = (bar_high - open_px) / gap_pts

        if PB_LO <= retrace <= PB_HI:
            pb_found   = True
            pb_bar     = i
            pb_retrace = retrace
            pb_price   = bar_low if direction == "call" else bar_high
            break
        elif retrace > PB_HI:
            # Gap already faded more than 40% — count as "deep fade" not pullback
            pb_found = False
            break

    # Measure continuation after pullback
    if pb_found and pb_bar is not None:
        remaining = intra.iloc[pb_bar:]
        if len(remaining) < 5:
            continue

        eod_price   = float(intra.iloc[-1]["close"])
        bar36_price = float(intra.iloc[min(36, len(intra)-1)]["close"])

        if direction == "call":
            cont_to_eod  = (eod_price  - pb_price) / pb_price
            cont_to_bar36 = (bar36_price - pb_price) / pb_price
            # Did gap continue? = price at EOD > open
            continued_eod  = eod_price  > open_px
            continued_bar36 = bar36_price > open_px
        else:
            cont_to_eod  = (pb_price - eod_price)  / pb_price
            cont_to_bar36 = (pb_price - bar36_price) / pb_price
            continued_eod  = eod_price  < open_px
            continued_bar36 = bar36_price < open_px

        # Option pricing: how much cheaper is the option at pullback vs open?
        # Simple proxy: (open_px - pb_price) / open_px for calls
        option_discount = abs(open_px - pb_price) / open_px

        results.append({
            "date":            tdate,
            "direction":       direction,
            "gap_pct":         gap_pct,
            "open_px":         open_px,
            "pb_bar":          pb_bar,
            "pb_price":        pb_price,
            "pb_retrace_pct":  pb_retrace,
            "cont_to_eod":     cont_to_eod,
            "cont_to_bar36":   cont_to_bar36,
            "continued_eod":   continued_eod,
            "continued_bar36": continued_bar36,
            "option_discount": option_discount,
            "eod_price":       eod_price,
        })

    else:
        # No pullback in band — record as non-pullback day
        eod_price = float(intra.iloc[-1]["close"])
        if direction == "call":
            continued = eod_price > open_px
        else:
            continued = eod_price < open_px

        results.append({
            "date":            tdate,
            "direction":       direction,
            "gap_pct":         gap_pct,
            "open_px":         open_px,
            "pb_bar":          None,
            "pb_price":        None,
            "pb_retrace_pct":  None,
            "cont_to_eod":     None,
            "cont_to_bar36":   None,
            "continued_eod":   continued,
            "continued_bar36": continued,
            "option_discount": 0,
            "eod_price":       eod_price,
        })

# ── Results ────────────────────────────────────────────────────────────────
df = pd.DataFrame(results)
pb_days  = df[df["pb_bar"].notna()]
no_pb    = df[df["pb_bar"].isna()]

print("=" * 60)
print("GAP PULLBACK PATTERN ANALYSIS — REAL QQQ DATA")
print("=" * 60)
print(f"\nTotal gap days (>{GAP_MIN*100:.1f}%): {len(df)}")
print(f"  Pullback {PB_LO*100:.0f}–{PB_HI*100:.0f}% days:   {len(pb_days)} ({len(pb_days)/len(df)*100:.1f}%)")
print(f"  No pullback days:     {len(no_pb)}  ({len(no_pb)/len(df)*100:.1f}%)")

if len(pb_days):
    print(f"\n── PULLBACK DAYS (n={len(pb_days)}) ──────────────────────────────")
    print(f"  Continuation to EOD:   {pb_days['continued_eod'].mean()*100:.1f}% of days")
    print(f"  Continuation to bar36: {pb_days['continued_bar36'].mean()*100:.1f}% of days")
    print(f"  Avg cont move (EOD):   {pb_days['cont_to_eod'].mean()*100:+.3f}%")
    print(f"  Avg cont move (bar36): {pb_days['cont_to_bar36'].mean()*100:+.3f}%")
    print(f"  Avg pullback depth:    {pb_days['pb_retrace_pct'].mean()*100:.1f}% of gap retraced")
    print(f"  Avg pullback bar:      {pb_days['pb_bar'].mean():.1f} (bar×5min = {pb_days['pb_bar'].mean()*5:.0f}min)")
    print(f"  Avg option discount:   {pb_days['option_discount'].mean()*100:.3f}% underlying move")

    # By direction
    for d in ["call", "put"]:
        sub = pb_days[pb_days["direction"]==d]
        if len(sub):
            print(f"\n  {d.upper()}S (n={len(sub)}):")
            print(f"    Continuation EOD:   {sub['continued_eod'].mean()*100:.1f}%")
            print(f"    Continuation bar36: {sub['continued_bar36'].mean()*100:.1f}%")
            print(f"    Avg gap:            {sub['gap_pct'].mean()*100:+.3f}%")
            print(f"    Avg cont (EOD):     {sub['cont_to_eod'].mean()*100:+.3f}%")

    # Gap size buckets
    print(f"\n── BY GAP SIZE ───────────────────────────────────────────")
    pb_days["gap_abs"] = pb_days["gap_pct"].abs()
    bins = [0.002, 0.004, 0.007, 0.012, 0.999]
    labels = ["0.2-0.4%","0.4-0.7%","0.7-1.2%",">1.2%"]
    pb_days["gap_bucket"] = pd.cut(pb_days["gap_abs"], bins=bins, labels=labels)
    for lbl in labels:
        sub = pb_days[pb_days["gap_bucket"]==lbl]
        if len(sub):
            print(f"  Gap {lbl:>10}: n={len(sub):>3}  "
                  f"cont%={sub['continued_eod'].mean()*100:.1f}%  "
                  f"avg_move={sub['cont_to_eod'].mean()*100:+.3f}%")

print(f"\n── NO-PULLBACK DAYS (n={len(no_pb)}) ────────────────────────")
print(f"  (Gap went straight — no 20-40% retrace)")
print(f"  Continuation to EOD: {no_pb['continued_eod'].mean()*100:.1f}%")

# ── Plot ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 9), facecolor="#0d1117")
fig.suptitle("QQQ Gap Pullback Pattern Analysis (Real Data 2020–2026)",
             color="white", fontsize=13, fontweight="bold")

RD="#ef4444"; GR="#22c55e"; BL="#3b82f6"; MU="#94a3b8"; YL="#eab308"

def style(ax, title):
    ax.set_facecolor("#161b22")
    for sp in ax.spines.values(): sp.set_color("#2d3748")
    ax.tick_params(colors=MU, labelsize=8)
    ax.set_title(title, color="white", fontsize=9, pad=6)
    ax.grid(True, color="#2d3748", lw=0.4, alpha=0.7)

# 1. Continuation rate: pullback vs no-pullback
ax = axes[0,0]
style(ax, "Continuation Rate: Pullback vs No-Pullback")
cats  = ["Pullback\n20-40%\n(n={})".format(len(pb_days)),
          "No Pullback\n(n={})".format(len(no_pb))]
rates = [pb_days["continued_eod"].mean()*100 if len(pb_days) else 0,
          no_pb["continued_eod"].mean()*100   if len(no_pb)  else 0]
colors = [GR if r > 50 else RD for r in rates]
bars = ax.bar(cats, rates, color=colors, alpha=0.85, edgecolor="#0d1117", width=0.5)
for bar, r in zip(bars, rates):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
            f"{r:.1f}%", ha="center", color="white", fontsize=11, fontweight="bold")
ax.axhline(50, color=MU, lw=0.8, ls="--", label="50% (coin flip)")
ax.set_ylim(0, 100); ax.set_ylabel("% Days Gap Continued to EOD", color=MU)
ax.legend(fontsize=7, facecolor="#1e2530", labelcolor="white", edgecolor="#2d3748")

# 2. Continuation move distribution
ax = axes[0,1]
style(ax, f"Cont. Move from Pullback (EOD)  n={len(pb_days)}")
if len(pb_days):
    wins   = pb_days[pb_days["cont_to_eod"] > 0]["cont_to_eod"] * 100
    losses = pb_days[pb_days["cont_to_eod"] <= 0]["cont_to_eod"] * 100
    bins   = np.linspace(-3, 3, 40)
    ax.hist(losses.clip(-3,3), bins=bins, color=RD, alpha=0.7, label=f"Faded ({len(losses)})")
    ax.hist(wins.clip(-3,3),   bins=bins, color=GR, alpha=0.7, label=f"Continued ({len(wins)})")
    ax.axvline(0, color=MU, lw=0.8, ls="--")
    mu = pb_days["cont_to_eod"].mean()*100
    ax.axvline(mu, color=YL, lw=1.5, ls="--", label=f"μ={mu:+.3f}%")
ax.set_xlabel("QQQ % move from pullback to EOD", color=MU)
ax.legend(fontsize=7, facecolor="#1e2530", labelcolor="white", edgecolor="#2d3748")

# 3. Pullback depth vs continuation
ax = axes[0,2]
style(ax, "Pullback Depth vs Continuation")
if len(pb_days):
    sc = ax.scatter(pb_days["pb_retrace_pct"]*100,
                    pb_days["cont_to_eod"]*100,
                    c=pb_days["continued_eod"].astype(int),
                    cmap="RdYlGn", alpha=0.6, s=20, vmin=0, vmax=1)
    ax.axhline(0, color=MU, lw=0.8, ls="--")
    ax.set_xlabel("Pullback depth (% of gap retraced)", color=MU)
    ax.set_ylabel("Continuation move to EOD (%)", color=MU)

# 4. Pullback bar distribution (when does pullback happen?)
ax = axes[1,0]
style(ax, "When Does Pullback Occur? (bar × 5min)")
if len(pb_days):
    ax.hist(pb_days["pb_bar"], bins=range(0, MAX_PB_BAR+2),
            color=BL, alpha=0.85, edgecolor="#0d1117")
    ax.set_xlabel("Bar number (bar 1 = 9:35am, bar 12 = 10:30am)", color=MU)
    ax.set_ylabel("Count", color=MU)
    med_bar = pb_days["pb_bar"].median()
    ax.axvline(med_bar, color=YL, lw=1.5, ls="--",
               label=f"Median: bar {med_bar:.0f} ({med_bar*5:.0f}min)")
    ax.legend(fontsize=7, facecolor="#1e2530", labelcolor="white", edgecolor="#2d3748")

# 5. Continuation rate by gap size
ax = axes[1,1]
style(ax, "Continuation Rate by Gap Size")
if len(pb_days) and "gap_bucket" in pb_days.columns:
    valid = pb_days.dropna(subset=["gap_bucket"])
    bucket_stats = valid.groupby("gap_bucket", observed=True).agg(
        cont_rate=("continued_eod","mean"),
        n=("continued_eod","count")
    ).reset_index()
    colors_b = [GR if r > 0.5 else RD for r in bucket_stats["cont_rate"]]
    bars2 = ax.bar(bucket_stats["gap_bucket"].astype(str),
                   bucket_stats["cont_rate"]*100,
                   color=colors_b, alpha=0.85, edgecolor="#0d1117")
    for bar, (_, row) in zip(bars2, bucket_stats.iterrows()):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                f"{row['cont_rate']*100:.0f}%\nn={row['n']:.0f}",
                ha="center", color="white", fontsize=8)
    ax.axhline(50, color=MU, lw=0.8, ls="--")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Continuation rate to EOD (%)", color=MU)

# 6. Avg P&L profile: entry at open vs entry at pullback
ax = axes[1,2]
style(ax, "Entry at Open vs Pullback (avg underlying move)")
if len(pb_days):
    open_move = (pb_days["eod_price"] - pb_days["open_px"]) / pb_days["open_px"] * 100
    open_move = open_move * pb_days["direction"].map({"call":1,"put":-1})
    pb_move   = pb_days["cont_to_eod"] * 100
    bins = np.linspace(-3, 3, 35)
    ax.hist(open_move.clip(-3,3), bins=bins, color=MU, alpha=0.5, label=f"Enter at open  μ={open_move.mean():+.3f}%")
    ax.hist(pb_move.clip(-3,3),   bins=bins, color=GR, alpha=0.7, label=f"Enter at pullback  μ={pb_move.mean():+.3f}%")
    ax.axvline(0, color="white", lw=0.8, ls="--")
    ax.set_xlabel("Underlying move in trade direction (%)", color=MU)
    ax.legend(fontsize=7, facecolor="#1e2530", labelcolor="white", edgecolor="#2d3748")

plt.tight_layout(rect=[0,0,1,0.96])
fig.savefig("results/gap_pullback_analysis.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print(f"\nChart saved → results/gap_pullback_analysis.png")
print("open results/gap_pullback_analysis.png")
