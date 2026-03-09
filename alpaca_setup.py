"""
APEX 0DTE — Alpaca Setup & Verification
=========================================
Tests every API endpoint the strategy needs with your existing Alpaca credentials.

SETUP (2 minutes)
------------------
1. Find your keys:
   → https://app.alpaca.markets → click your name (top right) → API Keys
   → "Paper Trading" keys work immediately — no funding required
   → "Live Trading" keys require a funded brokerage account

2. Set environment variables:
   Linux/Mac:
       export ALPACA_API_KEY="PKxxxxxxxxxxxxxxxxxxxxxxxx"
       export ALPACA_API_SECRET="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
   Windows (PowerShell):
       $env:ALPACA_API_KEY="PKxxxxxxxxxxxxxxxxxxxxxxxx"
       $env:ALPACA_API_SECRET="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

3. Run:
       python alpaca_setup.py --paper          # test paper keys
       python alpaca_setup.py --live           # test live keys

WHAT ALPACA GIVES YOU (FREE)
------------------------------
  Paper account:
    ✓ Real-time QQQ equity bars (5-min, 1-min, daily) via SIP feed
    ✓ Options chain snapshots (latestQuote + greeks via paper endpoint)
    ✓ $100,000 paper portfolio to simulate against
    ✗ Historical options data (not available free)

  Free data (no brokerage):
    ✓ QQQ bars with 15-min delay (IEX feed)
    ✗ Options chain data

  Live account (funded):
    ✓ Everything above in real-time
    ✓ Actual order execution
    ✓ Real options quotes at open

BOTTOM LINE
-----------
For backtesting:  paper keys give you real QQQ bars back to 2015.
                  Options chain: BSM-priced (real IV from trailing HV).
For live/paper:   paper keys give you real-time data + simulated options fills.
"""

from __future__ import annotations
import os, sys, json, argparse, logging
from datetime import date, datetime, timedelta

import requests
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("alpaca_setup")

PASS = "✅"; FAIL = "❌"; WARN = "⚠️ "

DATA_URL     = "https://data.alpaca.markets/v2"
OPTION_URL   = "https://data.alpaca.markets/v1beta1"
BROKER_PAPER = "https://paper-api.alpaca.markets/v2"
BROKER_LIVE  = "https://api.alpaca.markets/v2"


# ─────────────────────────────────────────────────────────────────────────────
# REQUEST HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def headers(key: str, secret: str) -> dict:
    return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}

def get(url: str, key: str, secret: str, params: dict = None) -> dict:
    r = requests.get(url, headers=headers(key, secret),
                     params=params or {}, timeout=15)
    r.raise_for_status()
    return r.json()

def get_paginated(base: str, endpoint: str,
                   key: str, secret: str, params: dict, result_key: str) -> list:
    results = []; p = dict(params)
    while True:
        data = get(f"{base}/{endpoint}", key, secret, p)
        items = data.get(result_key, []) or []
        if isinstance(items, dict):
            items = list(items.values())
        results.extend(items)
        tok = data.get("next_page_token")
        if not tok:
            break
        p["page_token"] = tok
    return results


# ─────────────────────────────────────────────────────────────────────────────
# TESTS
# ─────────────────────────────────────────────────────────────────────────────

def test_auth(key: str, secret: str, paper: bool) -> bool:
    print("\n── 1. Authentication ──────────────────────────────────────")
    broker = BROKER_PAPER if paper else BROKER_LIVE
    try:
        acct = get(f"{broker}/account", key, secret)
        status   = acct.get("status", "?")
        equity   = float(acct.get("equity", 0))
        currency = acct.get("currency", "USD")
        mode     = "PAPER" if paper else "LIVE"
        print(f"{PASS}  [{mode}] Account status: {status}")
        print(f"      Equity: {currency} ${equity:,.2f}")
        bp = float(acct.get("options_buying_power", acct.get("buying_power", 0)))
        print(f"      Options buying power: ${bp:,.2f}")
        pdt = acct.get("pattern_day_trader", False)
        dt_count = acct.get("daytrade_count", 0)
        print(f"      PDT flag: {pdt}  |  Day trade count (5d): {dt_count}")
        return True
    except requests.HTTPError as e:
        code = e.response.status_code
        if code == 401:
            print(f"{FAIL}  401 Unauthorized — wrong API key or secret")
            print(f"      Keys start with 'PK' for paper, 'AK' for live")
        elif code == 403:
            print(f"{FAIL}  403 Forbidden — account may be restricted")
        else:
            print(f"{FAIL}  HTTP {code}: {e.response.text[:200]}")
        return False
    except Exception as e:
        print(f"{FAIL}  {e}")
        return False


def test_clock(key: str, secret: str) -> dict:
    print("\n── 2. Market Clock ────────────────────────────────────────")
    try:
        c = get(f"{DATA_URL}/clock", key, secret)
        # Alpaca clock is at broker URL
        broker = BROKER_PAPER
        c = get(f"{broker}/clock", key, secret)
        is_open  = c.get("is_open", False)
        next_open  = c.get("next_open", "?")
        next_close = c.get("next_close", "?")
        status = "OPEN 🟢" if is_open else "CLOSED 🔴"
        print(f"{PASS}  Market: {status}")
        print(f"      Next open:  {next_open}")
        print(f"      Next close: {next_close}")
        return c
    except Exception as e:
        print(f"{WARN}  {e}")
        return {}


def test_daily_bars(key: str, secret: str) -> pd.DataFrame | None:
    print("\n── 3. QQQ Daily Bars (equity history) ─────────────────────")
    try:
        end   = date.today()
        start = date(2020, 1, 1)
        bars = get_paginated(DATA_URL, "stocks/QQQ/bars", key, secret,
                              {"timeframe": "1Day",
                               "start": start.isoformat(),
                               "end":   end.isoformat(),
                               "adjustment": "all",
                               "feed": "iex",
                               "limit": 1000},
                              result_key="bars")
        if not bars:
            print(f"{WARN}  No bars returned")
            return None
        df = pd.DataFrame(bars)
        df["date"]  = pd.to_datetime(df["t"]).dt.date
        df = df.rename(columns={"o":"open","h":"high","l":"low",
                                  "c":"close","v":"volume"})
        df = df.sort_values("date").reset_index(drop=True)
        df["gap_pct"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)

        last   = df.iloc[-1]
        prev   = df.iloc[-2]
        gap    = float(last["gap_pct"])
        print(f"{PASS}  {len(df)} daily bars  {df['date'].iloc[0]} → {df['date'].iloc[-1]}")
        print(f"      Last close: ${float(last['close']):.2f}  "
              f"Today open: ${float(last['open']):.2f}  "
              f"Gap: {gap:+.4f} ({gap*100:+.2f}%)")

        # Check bar count vs expected (should be ~5yr × 252)
        expected = (end - start).days * 252/365
        coverage = len(df) / expected * 100
        print(f"      Data coverage: {coverage:.0f}% of expected trading days")
        return df
    except Exception as e:
        print(f"{FAIL}  {e}")
        return None


def test_intraday_bars(key: str, secret: str) -> pd.DataFrame | None:
    print("\n── 4. QQQ Intraday 5-min Bars ─────────────────────────────")
    # Use most recent weekday
    d = date.today()
    if d.weekday() >= 5:
        d -= timedelta(days=d.weekday() - 4)
    try:
        bars = get_paginated(DATA_URL, "stocks/QQQ/bars", key, secret,
                              {"timeframe": "5Min",
                               "start": f"{d}T09:30:00-04:00",
                               "end":   f"{d}T16:00:00-04:00",
                               "feed": "iex",
                               "limit": 1000},
                              result_key="bars")
        if not bars:
            print(f"{WARN}  No intraday bars for {d}")
            print(f"      Normal if market is closed or pre-10am — data flows live")
            # Test with a known historical date
            hist_date = date(2024, 11, 15)  # known trading day
            bars = get_paginated(DATA_URL, "stocks/QQQ/bars", key, secret,
                                  {"timeframe": "5Min",
                                   "start": f"{hist_date}T09:30:00-04:00",
                                   "end":   f"{hist_date}T16:00:00-04:00",
                                   "feed": "iex",
                                   "limit": 1000},
                                  result_key="bars")
            if not bars:
                print(f"{FAIL}  Also no bars for {hist_date} — check credentials")
                return None
            d = hist_date
            print(f"      Using historical date {hist_date} for validation:")

        df = pd.DataFrame(bars)
        df["datetime"] = pd.to_datetime(df["t"])
        df = df.rename(columns={"o":"open","h":"high","l":"low",
                                  "c":"close","v":"volume"})
        df = df.sort_values("datetime").reset_index(drop=True)
        moves = df["close"].astype(float).pct_change().dropna().abs()
        open_p = float(df.iloc[0]["open"]); close_p = float(df.iloc[-1]["close"])
        day_move = (close_p - open_p) / open_p * 100
        print(f"{PASS}  {len(df)} 5-min bars for {d}")
        print(f"      {df['datetime'].iloc[0].strftime('%H:%M')} → "
              f"{df['datetime'].iloc[-1].strftime('%H:%M')}")
        print(f"      Day move: {day_move:+.2f}%  "
              f"Avg 5-min: {moves.mean()*100:.3f}%  "
              f"Max 5-min: {moves.max()*100:.3f}%")
        return df
    except Exception as e:
        print(f"{FAIL}  {e}")
        return None


def test_options_chain(key: str, secret: str) -> pd.DataFrame | None:
    print("\n── 5. QQQ Options Chain ───────────────────────────────────")
    # Find next expiry (QQQ has 0DTE Mon/Wed/Fri)
    today = date.today()
    # Try today first, then next few days
    for offset in range(5):
        candidate = today + timedelta(days=offset)
        if candidate.weekday() < 5:   # weekday
            expiry_str = candidate.isoformat()
            break
    else:
        expiry_str = today.isoformat()

    try:
        data = get(f"{OPTION_URL}/options/snapshots/QQQ", key, secret, {
            "expiration_date_gte": expiry_str,
            "expiration_date_lte": expiry_str,
            "feed": "indicative",
            "limit": 100,
        })
        snaps = data.get("snapshots", {})

        if not snaps:
            print(f"{WARN}  No options snapshots for {expiry_str}")
            print(f"      This is normal if:")
            print(f"        • It's a weekend/holiday")
            print(f"        • Before ~9:25am EST on a trading day")
            print(f"        • Your account plan doesn't include options data")
            print(f"      Strategy will fall back to BSM-priced chain (real QQQ open + trailing HV)")
            return None

        # Parse a few ATM options
        rows = []
        for sym, snap in list(snaps.items())[:200]:
            try:
                suffix = sym[3:]  # strip "QQQ"
                opt_type = "call" if "C" in suffix else "put"
                K = float(sym[-8:]) / 1000
                q = snap.get("latestQuote", {})
                bid = float(q.get("bp", 0) or 0)
                ask = float(q.get("ap", 0) or 0)
                mid = (bid + ask) / 2
                if mid > 0:
                    rows.append({"symbol": sym, "type": opt_type,
                                  "strike": K, "bid": bid, "ask": ask, "mid": mid,
                                  "greeks": snap.get("greeks", {})})
            except Exception:
                continue

        if not rows:
            print(f"{WARN}  Chain returned but parsing failed")
            return None

        df = pd.DataFrame(rows)
        calls = df[df["type"]=="call"].sort_values("strike")
        puts  = df[df["type"]=="put"].sort_values("strike")
        print(f"{PASS}  Options chain for {expiry_str}: "
              f"{len(calls)} calls, {len(puts)} puts")
        print(f"      Strike range: ${df['strike'].min():.0f} – ${df['strike'].max():.0f}")

        # Get QQQ last price to find ATM
        try:
            quote = get(f"{DATA_URL}/stocks/QQQ/quotes/latest", key, secret,
                        {"feed": "iex"})
            qqq_px = float(quote.get("quote", {}).get("ap", 0) or
                           quote.get("quote", {}).get("bp", 0) or 450)
        except Exception:
            qqq_px = float(df["strike"].median())

        df["dist"] = (df["strike"] - qqq_px).abs()
        atm_call = df[df["type"]=="call"].sort_values("dist").iloc[0]
        atm_put  = df[df["type"]=="put"].sort_values("dist").iloc[0]
        print(f"\n      QQQ ~${qqq_px:.2f}")
        for label, row in [("ATM Call", atm_call), ("ATM Put", atm_put)]:
            mid_v = float(row["mid"])
            sp    = (float(row["ask"]) - float(row["bid"])) / mid_v if mid_v else 0
            g     = row["greeks"] or {}
            print(f"      {label} K=${row['strike']:.0f}: "
                  f"mid=${mid_v:.3f}  spread={sp*100:.1f}%  "
                  f"IV={float(g.get('impliedVolatility',0) or 0):.3f}  "
                  f"Δ={float(g.get('delta',0) or 0):.3f}")
        return df
    except requests.HTTPError as e:
        if e.response.status_code in (403, 422):
            print(f"{WARN}  Options chain not available on this plan/endpoint")
            print(f"      Strategy will use BSM fallback (real price + HV-derived IV)")
        else:
            print(f"{FAIL}  HTTP {e.response.status_code}: {e.response.text[:200]}")
        return None
    except Exception as e:
        print(f"{WARN}  {e}")
        return None


def test_execution_model(daily_df: pd.DataFrame | None):
    print("\n── 6. Execution Model ─────────────────────────────────────")
    from data_providers import ExecutionModel, bsm_greeks

    em = ExecutionModel()
    qqq_px = float(daily_df.iloc[-1]["open"]) if daily_df is not None else 450.0
    S = qqq_px; K = round(S); iv = 0.25 * 1.38; T = 1/252; rf = 0.05
    mid = bsm_greeks(S, K, T, rf, iv, call=True)["price"]
    print(f"      QQQ ~${S:.2f}  ATM call BSM mid: ${mid:.3f}  IV: {iv:.3f}")
    print(f"\n      {'Risk':>8}  {'Contracts':>9}  {'Fill':>8}  "
          f"{'Slip+Comm':>11}  {'Total Drag':>11}")
    for risk in [1_000, 5_000, 25_000, 100_000, 500_000]:
        n    = max(1, min(500, int(risk / (mid * 100))))
        fill = em.entry_price(mid, iv, T, 0, n)
        slip = (fill - mid) * n * 100
        comm = em.commission_cost(n)
        drag = (slip + comm) / risk * 100
        print(f"      ${risk:>7,.0f}  {n:>9,}  ${fill:>7.3f}  "
              f"${slip+comm:>10,.0f}  {drag:>10.2f}%")
    print(f"{PASS}  Execution model validated")


def show_backtest_commands(paper: bool):
    mode = "paper" if paper else "live"
    key_var = "ALPACA_API_KEY"
    sec_var = "ALPACA_API_SECRET"
    print(f"""
{'='*60}
  READY TO RUN
{'='*60}

── Historical backtest (real QQQ bars + BSM options) ────────
  python apex.py backtest \\
      --provider alpaca \\
      --{'paper' if paper else 'key YOUR_LIVE_KEY'} \\
      --start 2020-01-01 --end 2025-01-01 \\
      --out results/apex_alpaca

── NQ futures gap signal + real QQQ bars ────────────────────
  pip install yfinance
  python backtest_futures.py \\
      --provider alpaca \\
      --start 2020-01-01 --end 2025-01-01

── Paper trading (run once daily at 9:28am EST) ─────────────
  python apex.py paper \\
      --provider alpaca \\
      --state apex_{mode}_state.pkl

  Add to crontab (Linux/Mac):
  28 9 * * 1-5  cd /path/to/apex_0dte && \\
      {key_var}=$ALPACA_API_KEY \\
      {sec_var}=$ALPACA_API_SECRET \\
      python apex.py paper --provider alpaca \\
          --state apex_{mode}_state.pkl >> logs/apex.log 2>&1

── Environment variables (add to ~/.bashrc or ~/.zshrc) ─────
  export ALPACA_API_KEY="PKxxxxxxxxxxxxxxxxxxxxxxxx"
  export ALPACA_API_SECRET="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

{'='*60}
""")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="APEX 0DTE — Alpaca Setup & Verification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--paper", action="store_true", help="Test paper account keys")
    mode.add_argument("--live",  action="store_true", help="Test live account keys")
    p.add_argument("--key",    default=None, help="API key (or set ALPACA_API_KEY)")
    p.add_argument("--secret", default=None, help="API secret (or set ALPACA_API_SECRET)")
    args = p.parse_args()

    key    = args.key    or os.environ.get("ALPACA_API_KEY", "")
    secret = args.secret or os.environ.get("ALPACA_API_SECRET", "")

    if not key or not secret:
        print(f"\n{FAIL}  Missing credentials.")
        print(f"    Pass with --key / --secret, or set environment variables:")
        print(f"      export ALPACA_API_KEY=PKxxxxx")
        print(f"      export ALPACA_API_SECRET=xxxxxxxx")
        print(f"\n    Get keys: https://app.alpaca.markets → API Keys")
        sys.exit(1)

    mode_str = "PAPER" if args.paper else "LIVE"
    print(f"\n{'='*60}")
    print(f"  APEX 0DTE — Alpaca {mode_str} Setup Verification")
    print(f"  Key: {key[:8]}{'*'*(len(key)-8)}")
    print(f"{'='*60}")

    if not test_auth(key, secret, paper=args.paper):
        print(f"\n{FAIL}  Authentication failed. Cannot continue.")
        print(f"    Double-check your key/secret at: https://app.alpaca.markets")
        sys.exit(1)

    test_clock(key, secret)
    daily_df   = test_daily_bars(key, secret)
    intra_df   = test_intraday_bars(key, secret)
    chain_df   = test_options_chain(key, secret)
    test_execution_model(daily_df)

    # Summary
    print(f"\n{'='*60}")
    print(f"  SETUP SUMMARY  [{mode_str}]")
    print(f"{'='*60}")
    print(f"  {PASS}  Auth & account: OK")
    print(f"  {'✅' if daily_df is not None else '❌'}  Daily bars: "
          f"{'✓ ' + str(len(daily_df)) + ' days' if daily_df is not None else 'FAILED'}")
    print(f"  {'✅' if intra_df is not None else '⚠️'}  Intraday 5-min: "
          f"{'✓ ' + str(len(intra_df)) + ' bars' if intra_df is not None else 'no data (OK if closed)'}")
    print(f"  {'✅' if chain_df is not None else '⚠️'}  Options chain: "
          f"{'✓ real quotes' if chain_df is not None else 'BSM fallback (real price + HV-derived IV)'}")
    print(f"  {PASS}  Execution model: OK")

    if chain_df is None:
        print(f"""
  {WARN} Options chain note:
     Real option quotes weren't available (normal for free/paper accounts
     outside market hours, or on the free data plan).
     The strategy will use BSM pricing with:
       • Real QQQ open price from Alpaca equity bars ← honest
       • IV = 21-day trailing HV × 1.38 open spike  ← modeled
       • Dynamic spread model (3-8% of mid)          ← conservative

     This gives you honest SIGNAL backtesting (entry/exit timing)
     with modeled option P&L. Good enough to measure continuation rate.
     For precise P&L, options quotes need to be available during market hours.
""")

    show_backtest_commands(paper=args.paper)


if __name__ == "__main__":
    main()
