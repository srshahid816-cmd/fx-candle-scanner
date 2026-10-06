#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FX CANDLE SCANNER
=================
Price action + candle reaction + buyer/seller positions, combined in one scanner.

Har pair ke liye ye tool:
  1. data download karta hai (Yahoo Finance, real spot forex) ya aapki TradingView CSV padhta hai
  2. 3 modules chalata hai (Momentum, Positions, Rejection) + filters
  3. last CLOSED candle par faisla deta hai: NEXT candle ke liye BUY / SELL / HOLD
  4. usi rule ko us pair ke purane data par backtest karke "historical win %" batata hai
     (train/test split ke saath, taake out-of-sample result bhi dikhe)

IMPORTANT
  * Win % koi prediction nahi hai. Ye us pair par is rule ka PURANA hit-rate hai.
  * Binary payout ke hisaab se breakeven win rate bhi dikhaya jata hai (1 / (1 + payout)).
  * Sirf real forex data par chalta hai. Broker ke OTC charts ka data yahan available nahi.

Usage examples:
  python fx_scanner.py                              # menu: pair select karo
  python fx_scanner.py --pairs EURUSD GBPJPY        # specific pairs
  python fx_scanner.py --all --interval 5m          # 28 major/minor pairs
  python fx_scanner.py --pairs EURJPY --detail      # module-wise stats ke saath
  python fx_scanner.py --pairs EURJPY --tune        # parameters train data par tune karo
  python fx_scanner.py --pairs EURJPY --watch       # har candle close par dobara scan
  python fx_scanner.py --csv tv_export.csv --name EURJPY --interval 5m   # TradingView CSV
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from itertools import product

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None

try:
    from rich.console import Console
    from rich.table import Table

    CONSOLE = Console()
    HAVE_RICH = True
except Exception:  # pragma: no cover
    CONSOLE = None
    HAVE_RICH = False

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------
PAIRS = [
    "EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCHF", "USDCAD",
    "EURGBP", "EURJPY", "EURCHF", "EURAUD", "EURNZD", "EURCAD",
    "GBPJPY", "GBPCHF", "GBPAUD", "GBPNZD", "GBPCAD",
    "AUDJPY", "AUDCHF", "AUDNZD", "AUDCAD",
    "NZDJPY", "NZDCHF", "NZDCAD",
    "CADJPY", "CADCHF", "CHFJPY",
]

INTERVAL_SEC = {"1m": 60, "2m": 120, "5m": 300, "15m": 900, "30m": 1800, "60m": 3600, "1h": 3600}
# Yahoo ki intraday limits (din)
MAX_DAYS = {"1m": 29, "2m": 59, "5m": 59, "15m": 59, "30m": 59, "60m": 729, "1h": 729}
ROUND_STEP = {"1m": 10, "2m": 10, "5m": 25, "15m": 50, "30m": 50, "60m": 100, "1h": 100}  # points (1 pip = 10 points)
DEFAULT_DAYS = {"1m": 29, "2m": 30, "5m": 59, "15m": 59, "30m": 59, "60m": 365, "1h": 365}


@dataclass(frozen=True)
class Params:
    atr_n: int = 14              # ATR period
    pivot_k: int = 3             # swing pivot: k candles dono taraf (confirm k candle baad -> no look-ahead)
    strong_body: float = 0.55    # strong candle: body / range >= ye
    strong_atr: float = 0.60     # strong candle: body >= ye * ATR
    weak_body: float = 0.35      # weak candle: body / range < ye
    wick_ratio: float = 0.40     # rejection wick / range >= ye
    round_step_pts: int = 50     # round-number level har itne "points" par (1 pip = 10 points)
    tol_atr: float = 0.15        # "touch" tolerance (ATR ke hisaab se)
    near_target_atr: float = 1.0 # target itna paas ho to reversal nahi
    min_score: float = 1.0       # BUY/SELL ke liye minimum score (1.0 = koi bhi ek trend-aligned module)
    train_frac: float = 0.70     # train / test split
    min_n_tune: int = 30         # tuning mein minimum trades
    low_vol_ratio: float = 0.5   # ATR is se kam (median ka) to market "dead"


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------
def point_size(sym: str) -> float:
    """1 point = 1/10 pip."""
    return 0.001 if "JPY" in sym else 0.00001


def normalize_symbol(s: str) -> str:
    s = s.upper().strip().replace("/", "").replace("-", "").replace("_", "").replace(" ", "")
    if s.endswith("=X"):
        s = s[:-2]
    return s


def wilson(w: int, n: int, z: float = 1.96):
    if n <= 0:
        return (float("nan"), float("nan"))
    p = w / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def pct(x) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.1f}%"


# --------------------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------------------
def _clean_yf(raw):
    if raw is None or len(raw) == 0:
        return None
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.rename(columns=lambda x: str(x).lower())
    need = ["open", "high", "low", "close"]
    if not all(x in raw.columns for x in need):
        return None
    out = raw[need].astype(float).dropna()
    idx = pd.DatetimeIndex(pd.to_datetime(out.index))
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    out.index = idx
    return out


def fetch_yf(base: str, interval: str, days: int):
    if yf is None:
        raise RuntimeError("yfinance install nahi hai -> pip install -r requirements.txt")
    days = int(min(days, MAX_DAYS.get(interval, 59)))
    ticker = base + "=X"
    end = datetime.now(timezone.utc)
    start_all = end - timedelta(days=days)
    chunk = timedelta(days=7) if interval == "1m" else timedelta(days=days)
    frames = []
    cur_end = end
    while cur_end > start_all:
        cur_start = max(start_all, cur_end - chunk)
        try:
            raw = yf.download(
                ticker,
                start=cur_start,
                end=cur_end + timedelta(minutes=5),
                interval=interval,
                progress=False,
                auto_adjust=False,
                threads=False,
            )
            cl = _clean_yf(raw)
            if cl is not None and len(cl):
                frames.append(cl)
        except Exception as e:  # network / rate-limit
            print(f"  [warn] {base}: chunk fail ({e})", file=sys.stderr)
        cur_end = cur_start
    if not frames:
        return None
    out = pd.concat(frames).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out


def drop_incomplete(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    sec = INTERVAL_SEC.get(interval, 60)
    now = pd.Timestamp.now(tz="UTC")
    if len(df) and df.index[-1] + pd.Timedelta(seconds=sec) > now:
        df = df.iloc[:-1]
    return df


def load_csv(path: str) -> pd.DataFrame:
    """TradingView 'Export chart data' CSV (time, open, high, low, close, ...)."""
    raw = pd.read_csv(path)
    cols = {str(c).lower().strip(): c for c in raw.columns}
    tcol = next((cols[k] for k in ("time", "datetime", "date", "timestamp") if k in cols), None)
    if tcol is None or not all(k in cols for k in ("open", "high", "low", "close")):
        raise ValueError("CSV mein time/open/high/low/close columns chahiye")
    t = raw[tcol]
    if pd.api.types.is_numeric_dtype(t):
        idx = pd.to_datetime(t, unit="s", utc=True)
    else:
        idx = pd.to_datetime(t, utc=True, errors="coerce")
    out = pd.DataFrame({k: raw[cols[k]].astype(float).to_numpy() for k in ("open", "high", "low", "close")})
    out.index = pd.DatetimeIndex(idx)
    out = out[out.index.notna()].sort_index().dropna()
    return out


# --------------------------------------------------------------------------------------
# Feature engineering  (sab kuch sirf closed candles se, koi look-ahead nahi)
# --------------------------------------------------------------------------------------
def compute_positions(o, h, l, c):
    """
    Video 5 ka "first support / first resistance" (meri tarjumani, heuristic):
      * har external candle (jo pichli candle ka high/low tode) -> S = uska low, R = uska high
      * agar close pichli R ke upar: resistance toot gayi -> support shift ho kar us candle ke open par
      * agar close pichli S ke neeche: support toot gayi -> resistance shift ho kar us candle ke open par
    """
    n = len(o)
    S = np.full(n, np.nan)
    R = np.full(n, np.nan)
    s = r = np.nan
    for i in range(1, n):
        broke = False
        if not np.isnan(r) and c[i] > r:
            r = np.nan
            s = o[i]
            broke = True
        elif not np.isnan(s) and c[i] < s:
            s = np.nan
            r = o[i]
            broke = True
        inside = h[i] <= h[i - 1] and l[i] >= l[i - 1]
        if not broke and not inside:
            s = l[i]
            r = h[i]
        S[i] = s
        R[i] = r
    return S, R


def trend_adjust(sig, trend, near):
    """Trend ke saath: full weight. Range: 0.75. Trend ke khilaf: 0.5, aur target paas ho to 0."""
    sig = np.asarray(sig, dtype=float)
    same = (np.sign(sig) == trend) & (trend != 0)
    flat = trend == 0
    counter = (~same) & (~flat) & (sig != 0)
    w = np.where(same, 1.0, np.where(flat, 0.75, 0.5))
    w = np.where(counter & near, 0.0, w)
    return sig * w


def build(df: pd.DataFrame, sym: str, P: Params):
    d = df.copy()
    o = d["open"]
    h = d["high"]
    l = d["low"]
    c = d["close"]
    rng = (h - l).replace(0, np.nan)
    body = (c - o).abs()
    d["dir"] = np.sign(c - o).astype(int)
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    d["atr"] = tr.rolling(P.atr_n, min_periods=P.atr_n).mean()
    d["body_ratio"] = (body / rng).fillna(0.0)
    d["upper"] = ((h - np.maximum(o, c)) / rng).fillna(0.0)
    d["lower"] = ((np.minimum(o, c) - l) / rng).fillna(0.0)
    d["strong"] = (d["body_ratio"] >= P.strong_body) & (body >= P.strong_atr * d["atr"])
    d["weak"] = (d["body_ratio"] < P.weak_body) | (body < 0.30 * d["atr"])
    d["inside"] = (h <= h.shift(1)) & (l >= l.shift(1))
    ext = ~d["inside"]
    atr = d["atr"]
    tol = P.tol_atr * atr

    # ---- swing pivots (k candle baad confirm) -> structure / trend / target
    k = P.pivot_k
    ph_flag = h == h.rolling(2 * k + 1, center=True).max()
    pl_flag = l == l.rolling(2 * k + 1, center=True).min()
    ph_val = h.shift(k).where(ph_flag.shift(k, fill_value=False).astype(bool))
    pl_val = l.shift(k).where(pl_flag.shift(k, fill_value=False).astype(bool))
    ph_s = ph_val.dropna()
    pl_s = pl_val.dropna()
    d["last_ph"] = ph_s.reindex(d.index).ffill()
    d["prev_ph"] = ph_s.shift(1).reindex(d.index).ffill()
    d["last_pl"] = pl_s.reindex(d.index).ffill()
    d["prev_pl"] = pl_s.shift(1).reindex(d.index).ffill()
    up = (d["last_ph"] > d["prev_ph"]) & (d["last_pl"] > d["prev_pl"])
    dn = (d["last_ph"] < d["prev_ph"]) & (d["last_pl"] < d["prev_pl"])
    trend = np.where(up, 1, np.where(dn, -1, 0))
    d["trend"] = trend
    near = (
        ((trend == 1) & (d["last_ph"] > c) & ((d["last_ph"] - c) < P.near_target_atr * atr))
        | ((trend == -1) & (c > d["last_pl"]) & ((c - d["last_pl"]) < P.near_target_atr * atr))
    ).to_numpy()
    d["near_target"] = near
    d["lowvol"] = atr < P.low_vol_ratio * atr.rolling(200, min_periods=50).median()

    dr = d["dir"]
    strong = d["strong"]

    # ================= MODULE A : Momentum / retracement / continuation (Videos 1-4) =================
    A = np.zeros(len(d))
    for r_len in (1, 2):
        for dm in (1, -1):
            retr = pd.Series(True, index=d.index)
            for j in range(1, r_len + 1):
                retr = retr & (dr.shift(j) == -dm)
            mom = (dr.shift(r_len + 1) == dm) & (dr.shift(r_len + 2) == dm)
            base = (mom & retr).to_numpy()
            retr_first_open = o.shift(r_len)
            mom_last_open = o.shift(r_len + 1)
            if dm == 1:
                cont = (dr == 1) & strong & (c > retr_first_open)       # buyer ne strong continuation di
                fail = (dr == -1) & strong & (c < mom_last_open)       # buyer dead -> seller strong
            else:
                cont = (dr == -1) & strong & (c < retr_first_open)
                fail = (dr == 1) & strong & (c > mom_last_open)
            A += np.where(base & cont.to_numpy(), dm, 0)
            A += np.where(base & fail.to_numpy(), -dm, 0)
    A = np.clip(A, -1, 1)

    # ================= MODULE B : Buyer / Seller positions (Video 5) =================
    S, R = compute_positions(o.to_numpy(), h.to_numpy(), l.to_numpy(), c.to_numpy())
    S_prev = pd.Series(S, index=d.index).shift(1)
    R_prev = pd.Series(R, index=d.index).shift(1)
    d["S1"] = S_prev
    d["R1"] = R_prev
    bull_cnt = (dr == 1).rolling(5).sum()
    bear_cnt = (dr == -1).rolling(5).sum()
    no_res = R_prev.isna() | (h < R_prev - tol)       # candle ne resistance touch nahi ki
    no_sup = S_prev.isna() | (l > S_prev + tol)       # candle ne support touch nahi ki
    tap_sup = l <= S_prev + tol
    tap_res = h >= R_prev - tol
    trend_s = pd.Series(trend, index=d.index)
    b_up = (trend_s == 1) & ext & (dr == 1) & tap_sup & no_res & (bull_cnt >= 3)
    b_dn = (trend_s == -1) & ext & (dr == -1) & tap_res & no_sup & (bear_cnt >= 3)
    b_up_rev = (trend_s == 1) & ext & (dr == -1) & tap_res & (c < R_prev) & no_sup
    b_dn_rev = (trend_s == -1) & ext & (dr == 1) & tap_sup & (c > S_prev) & no_res
    B = (b_up.astype(float) + b_dn_rev.astype(float) - b_dn.astype(float) - b_up_rev.astype(float)).to_numpy()

    # ================= MODULE C : Rejection / fake breakout at levels (Videos 1-4) =================
    step = P.round_step_pts * point_size(sym)
    Lr = np.round(l / step) * step
    Hr = np.round(h / step) * step
    bull_r = (l <= Lr + tol) & (l >= Lr - 0.6 * atr) & (c > Lr) & (d["lower"] >= P.wick_ratio) & (dr == 1)
    bear_r = (h >= Hr - tol) & (h <= Hr + 0.6 * atr) & (c < Hr) & (d["upper"] >= P.wick_ratio) & (dr == -1)
    bull_s = (l <= d["last_pl"] + tol) & (l >= d["last_pl"] - 0.6 * atr) & (c > d["last_pl"]) & (d["lower"] >= P.wick_ratio) & (dr == 1)
    bear_s = (h >= d["last_ph"] - tol) & (h <= d["last_ph"] + 0.6 * atr) & (c < d["last_ph"]) & (d["upper"] >= P.wick_ratio) & (dr == -1)
    bull = (bull_r | bull_s) & ext
    bear = (bear_r | bear_s) & ext
    C = (bull.astype(float) - bear.astype(float)).to_numpy()

    # ---- trend filter + global filters
    A_adj = trend_adjust(A, trend, near)
    B_adj = trend_adjust(B, trend, near)
    C_adj = trend_adjust(C, trend, near)
    score = A_adj + B_adj + C_adj
    bad = d["inside"].to_numpy() | d["lowvol"].to_numpy() | d["atr"].isna().to_numpy()
    bad[:60] = True  # warm-up
    score = np.where(bad, 0.0, score)
    mods = {
        "Momentum": np.where(bad, 0.0, A_adj),
        "Positions": np.where(bad, 0.0, B_adj),
        "Rejection": np.where(bad, 0.0, C_adj),
    }
    return d, mods, score


def decide(score, min_score):
    score = np.asarray(score)
    return np.where(score >= min_score, 1, np.where(score <= -min_score, -1, 0))


# --------------------------------------------------------------------------------------
# Backtest statistics  (signal candle i -> trade next candle i+1: open -> close)
# --------------------------------------------------------------------------------------
def stats(sig, d, sl=None, payout=0.85):
    nxt = (d["close"] - d["open"]).shift(-1).to_numpy()
    s = np.asarray(sig)
    m = (s != 0) & ~np.isnan(nxt) & (nxt != 0)
    if sl is not None:
        mm = np.zeros(len(s), dtype=bool)
        mm[sl] = True
        m &= mm
    n = int(m.sum())
    w = int(((np.sign(nxt) == np.sign(s)) & m).sum())
    wr = w / n if n else float("nan")
    lb, ub = wilson(w, n)
    ev = wr * payout - (1 - wr) if n else float("nan")
    return {"n": n, "w": w, "wr": wr, "lb": lb, "ub": ub, "ev": ev}


def tune_params(df, sym, P, payout):
    split = int(len(df) * P.train_frac)
    best_key, best_P = -1.0, P
    for sb, rs, ms in product((0.45, 0.55, 0.65), (5, 10, 25, 50), (0.5, 1.0, 1.5)):
        Pt = replace(P, strong_body=sb, round_step_pts=rs, min_score=ms)
        d, _, score = build(df, sym, Pt)
        st = stats(decide(score, ms), d, slice(0, split), payout)
        if st["n"] >= P.min_n_tune and st["lb"] > best_key:
            best_key, best_P = st["lb"], Pt
    return best_P


# --------------------------------------------------------------------------------------
# Analysis of one pair
# --------------------------------------------------------------------------------------
def analyze(name, df, P, payout, gate, tune=False):
    if df is None or len(df) < 400:
        return {"pair": name, "error": f"data kam hai ({0 if df is None else len(df)} candles, kam az kam 400 chahiye)"}
    if tune:
        P = tune_params(df, name, P, payout)
    d, mods, score = build(df, name, P)
    split = int(len(d) * P.train_frac)
    sig = decide(score, P.min_score)
    be = 1.0 / (1.0 + payout)

    all_st = stats(sig, d, None, payout)
    test_st = stats(sig, d, slice(split, None), payout)
    train_st = stats(sig, d, slice(0, split), payout)
    buy_all = stats(np.where(sig == 1, 1, 0), d, None, payout)
    sell_all = stats(np.where(sig == -1, -1, 0), d, None, payout)
    buy_oos = stats(np.where(sig == 1, 1, 0), d, slice(split, None), payout)
    sell_oos = stats(np.where(sig == -1, -1, 0), d, slice(split, None), payout)

    cur_score = float(score[-1])
    cur = int(sig[-1])
    side = "BUY" if cur == 1 else "SELL" if cur == -1 else "HOLD"
    if cur == 1:
        h_all, h_oos = buy_all, buy_oos
    elif cur == -1:
        h_all, h_oos = sell_all, sell_oos
    else:
        h_all, h_oos = all_st, test_st

    # verdict (sirf out-of-sample par)
    if cur == 0:
        verdict = "-"
    elif h_oos["n"] < 30:
        verdict = "DATA KAM (OOS<30)"
    elif h_oos["lb"] > be:
        verdict = "EDGE (CI > breakeven)"
    elif h_oos["wr"] > be:
        verdict = "UNPROVEN (point > BE, CI nahi)"
    else:
        verdict = "NO EDGE"

    ok_point = cur != 0 and h_oos["n"] >= 30 and h_oos["wr"] > be
    ok_ci = cur != 0 and h_oos["n"] >= 30 and h_oos["lb"] > be
    action = side
    if cur != 0 and ((gate == "point" and not ok_point) or (gate == "ci" and not ok_ci)):
        action = "HOLD"

    reasons = []
    for k_, arr in mods.items():
        v = arr[-1]
        if v != 0:
            reasons.append(f"{k_}{'↑' if v > 0 else '↓'}")
    tr = int(d["trend"].iloc[-1])
    trend_txt = {1: "UP", -1: "DOWN", 0: "RANGE"}[tr]

    mod_rows = []
    for k_, arr in mods.items():
        msig = np.where(arr >= 0.5, 1, np.where(arr <= -0.5, -1, 0))
        mod_rows.append((k_, stats(msig, d, None, payout), stats(msig, d, slice(split, None), payout)))

    return {
        "pair": name, "time": d.index[-1], "close": float(d["close"].iloc[-1]), "trend": trend_txt,
        "score": cur_score, "signal": side, "action": action, "reasons": ", ".join(reasons) or "-",
        "hist_all": h_all, "hist_oos": h_oos, "all": all_st, "train": train_st, "test": test_st,
        "buy_all": buy_all, "sell_all": sell_all, "buy_oos": buy_oos, "sell_oos": sell_oos,
        "verdict": verdict, "be": be, "mods": mod_rows, "params": P, "bars": len(d),
    }


# --------------------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------------------

def pool_stats(stat_list, payout):
    n = sum(x["n"] for x in stat_list)
    w = sum(x["w"] for x in stat_list)
    wr = w / n if n else float("nan")
    lb, ub = wilson(w, n)
    ev = wr * payout - (1 - wr) if n else float("nan")
    return {"n": n, "w": w, "wr": wr, "lb": lb, "ub": ub, "ev": ev}


def print_pooled(ok, payout):
    if len(ok) < 2:
        return
    be = 1.0 / (1.0 + payout)
    print(f"\n=== POOLED: {len(ok)} pairs ek saath (breakeven {pct(be)}) ===")
    rows = [("COMBINED all", pool_stats([r["all"] for r in ok], payout)),
            ("COMBINED OOS", pool_stats([r["test"] for r in ok], payout))]
    names = [m[0] for m in ok[0]["mods"]]
    for i, nm in enumerate(names):
        rows.append((f"{nm} all", pool_stats([r["mods"][i][1] for r in ok], payout)))
        rows.append((f"{nm} OOS", pool_stats([r["mods"][i][2] for r in ok], payout)))
    for lbl, s_ in rows:
        flag = ""
        if s_["n"] >= 300:
            flag = "  <-- EDGE" if s_["lb"] > be else ("  (point>BE, CI nahi)" if s_["wr"] > be else "")
        print(f"{lbl:18} n={s_['n']:6d} win={pct(s_['wr']):>6} CI=[{pct(s_['lb'])}, {pct(s_['ub'])}] EV/1unit={s_['ev']:+.3f}{flag}")


def color_of(a):
    return {"BUY": "green", "SELL": "red", "HOLD": "yellow"}.get(a, "white")


def print_results(results, args, interval):
    be = 1.0 / (1.0 + args.payout)
    ok = [r for r in results if "error" not in r]
    bad = [r for r in results if "error" in r]
    sec = INTERVAL_SEC.get(interval, 60)
    title = (f"FX SCAN | tf={interval} | trade: agli candle open par, expiry = 1 candle ({sec // 60} min) | "
             f"payout={args.payout:.0%} -> breakeven {pct(be)}")
    if HAVE_RICH:
        t = Table(title=title, show_lines=False)
        for col in ("Pair", "Candle (UTC)", "Trend", "Score", "SIGNAL", "ACTION", "Hist win% (all)", "OOS win% [n]", "95% CI (OOS)", "Verdict", "Kyun"):
            t.add_column(col)
        for r in ok:
            ho, hh = r["hist_oos"], r["hist_all"]
            ci = f"{pct(ho['lb'])}-{pct(ho['ub'])}" if ho["n"] else "n/a"
            t.add_row(
                r["pair"], r["time"].strftime("%m-%d %H:%M"), r["trend"], f"{r['score']:+.2f}",
                f"[{color_of(r['signal'])}]{r['signal']}[/]", f"[bold {color_of(r['action'])}]{r['action']}[/]",
                f"{pct(hh['wr'])} [{hh['n']}]", f"{pct(ho['wr'])} [{ho['n']}]", ci, r["verdict"], r["reasons"],
            )
        CONSOLE.print(t)
    else:
        print(title)
        print(f"{'Pair':8} {'Candle':12} {'Trend':6} {'Score':>6} {'SIGNAL':6} {'ACTION':6} {'All win% [n]':16} {'OOS win% [n]':16} {'Verdict':28} Kyun")
        for r in ok:
            ho, hh = r["hist_oos"], r["hist_all"]
            print(f"{r['pair']:8} {r['time'].strftime('%m-%d %H:%M'):12} {r['trend']:6} {r['score']:+6.2f} {r['signal']:6} {r['action']:6} "
                  f"{pct(hh['wr']) + ' [' + str(hh['n']) + ']':16} {pct(ho['wr']) + ' [' + str(ho['n']) + ']':16} {r['verdict']:28} {r['reasons']}")
    for r in bad:
        print(f"[!] {r['pair']}: {r['error']}")

    if args.detail:
        for r in ok:
            print(f"\n--- {r['pair']} detail ({r['bars']} candles, params: strong_body={r['params'].strong_body}, "
                  f"round_step={r['params'].round_step_pts}pts, min_score={r['params'].min_score}) ---")
            print(f"{'Module':10} {'n(all)':>7} {'win%(all)':>10} {'n(OOS)':>7} {'win%(OOS)':>10}")
            for k_, sa, so in r["mods"]:
                print(f"{k_:10} {sa['n']:7d} {pct(sa['wr']):>10} {so['n']:7d} {pct(so['wr']):>10}")
            for lbl, key in (("COMBINED all", "all"), ("COMBINED train", "train"), ("COMBINED test(OOS)", "test"),
                             ("BUY only (all)", "buy_all"), ("SELL only (all)", "sell_all"),
                             ("BUY only (OOS)", "buy_oos"), ("SELL only (OOS)", "sell_oos")):
                s = r[key]
                print(f"{lbl:20} n={s['n']:4d} win={pct(s['wr']):>6} CI=[{pct(s['lb'])}, {pct(s['ub'])}] EV/1unit={s['ev']:+.3f}")

    print_pooled(ok, args.payout)
    print("\nNOTE: Win% = is rule ka PURANA hit-rate (agli candle ki direction) is pair ke data par. Ye guarantee ya prediction nahi.")
    print(f"      Breakeven {pct(be)}; usse neeche = paisa jata hai. ACTION = SIGNAL jab out-of-sample win% breakeven se upar ho (gate={args.gate}), warna HOLD.")


# --------------------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------------------
def pick_pairs_interactive():
    print("\nPairs (numbers ya naam comma se alag karke likho; 'all' = sab):")
    for i, p in enumerate(PAIRS, 1):
        print(f"{i:2d}. {p}", end="   " if i % 4 else "\n")
    print()
    raw = input("Select: ").strip()
    if raw.lower() == "all":
        return list(PAIRS)
    out = []
    for tok in raw.replace(" ", ",").split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok.isdigit() and 1 <= int(tok) <= len(PAIRS):
            out.append(PAIRS[int(tok) - 1])
        else:
            out.append(normalize_symbol(tok))
    return out


def run_scan(pairs, args, interval, days, P):
    results = []
    for p in pairs:
        print(f"... {p} data le raha hoon", flush=True)
        try:
            if args.csv:
                df = load_csv(args.csv)
            else:
                df = fetch_yf(p, interval, days)
                if df is not None:
                    df = drop_incomplete(df, interval)
            results.append(analyze(p, df, P, args.payout, args.gate, args.tune))
        except Exception as e:
            results.append({"pair": p, "error": str(e)})
    print_results(results, args, interval)
    if args.out:
        rows = []
        for r in results:
            if "error" in r:
                continue
            rows.append({
                "scan_time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "pair": r["pair"], "candle_utc": r["time"], "interval": interval, "trend": r["trend"],
                "score": round(r["score"], 3), "signal": r["signal"], "action": r["action"],
                "hist_win_all": r["hist_all"]["wr"], "n_all": r["hist_all"]["n"],
                "oos_win": r["hist_oos"]["wr"], "n_oos": r["hist_oos"]["n"],
                "oos_ci_low": r["hist_oos"]["lb"], "oos_ci_high": r["hist_oos"]["ub"],
                "breakeven": r["be"], "verdict": r["verdict"], "reasons": r["reasons"],
            })
        if rows:
            out = pd.DataFrame(rows)
            try:
                old = pd.read_csv(args.out)
                out = pd.concat([old, out], ignore_index=True)
            except Exception:
                pass
            out.to_csv(args.out, index=False)
            print(f"(results save: {args.out})")


def main():
    ap = argparse.ArgumentParser(description="FX candle scanner: BUY / SELL / HOLD + historical win%")
    ap.add_argument("--pairs", nargs="+", help="e.g. EURUSD GBPJPY")
    ap.add_argument("--all", action="store_true", help="sab 28 major/minor pairs")
    ap.add_argument("--interval", default="1m", choices=list(INTERVAL_SEC.keys()))
    ap.add_argument("--days", type=int, default=None, help="kitne din ka data (default timeframe ke hisaab se)")
    ap.add_argument("--payout", type=float, default=0.85, help="binary payout, 0.85 = 85%%")
    ap.add_argument("--min-score", type=float, default=Params.min_score)
    ap.add_argument("--gate", choices=["none", "point", "ci"], default="point",
                    help="ACTION kab BUY/SELL ho: none = hamesha signal, point = OOS win%% > breakeven, ci = OOS CI low > breakeven")
    ap.add_argument("--round-step", type=int, default=None, help="round-number level gap, points mein (default timeframe ke hisaab se)")
    ap.add_argument("--tune", action="store_true", help="parameters train data par tune karo (overfit ka khatra, OOS dekho)")
    ap.add_argument("--detail", action="store_true", help="module-wise stats")
    ap.add_argument("--csv", help="TradingView export CSV (is mode mein --name zaroor do)")
    ap.add_argument("--name", help="CSV wale pair ka naam, e.g. EURJPY")
    ap.add_argument("--watch", action="store_true", help="har candle close par dobara scan (Ctrl+C se band)")
    ap.add_argument("--out", default="scan_results.csv", help="results CSV (khali chhodo to save nahi hoga: --out '')")
    args = ap.parse_args()

    if not (0 < args.payout <= 1.5):
        ap.error("--payout 0.0-1.5 ke beech")
    interval = args.interval
    days = args.days or DEFAULT_DAYS.get(interval, 30)
    P = replace(Params(), min_score=args.min_score, round_step_pts=args.round_step or ROUND_STEP.get(interval, 50))

    if args.csv:
        pairs = [normalize_symbol(args.name or "CSVPAIR")]
    elif args.all:
        pairs = list(PAIRS)
    elif args.pairs:
        pairs = [normalize_symbol(x) for x in args.pairs]
    else:
        pairs = pick_pairs_interactive()
        tf = input(f"Timeframe [{interval}] (1m/5m/15m/30m/60m): ").strip()
        if tf in INTERVAL_SEC:
            interval = tf
            days = args.days or DEFAULT_DAYS[interval]
    if not pairs:
        print("Koi pair select nahi hua.")
        return
    P = replace(P, round_step_pts=args.round_step or ROUND_STEP.get(interval, 50))

    if not args.watch:
        run_scan(pairs, args, interval, days, P)
        return
    sec = INTERVAL_SEC[interval]
    try:
        while True:
            run_scan(pairs, args, interval, days, P)
            now = time.time()
            nxt = (math.floor(now / sec) + 1) * sec + 8   # candle close + 8s buffer
            wait = max(1, int(nxt - now))
            print(f"\nAgla scan {wait}s baad (Ctrl+C = band)\n")
            time.sleep(wait)
    except KeyboardInterrupt:
        print("\nBand.")


if __name__ == "__main__":
    main()
