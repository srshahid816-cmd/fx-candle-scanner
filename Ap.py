#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FX Candle Scanner — Streamlit version (self-contained)"""
from __future__ import annotations

import math
import tempfile
import os
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from itertools import product

import numpy as np
import pandas as pd
import streamlit as st

try:
    import yfinance as yf
except Exception:
    yf = None

# ==================== CONSTANTS ====================
PAIRS = [
    "EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCHF", "USDCAD",
    "EURGBP", "EURJPY", "EURCHF", "EURAUD", "EURNZD", "EURCAD",
    "GBPJPY", "GBPCHF", "GBPAUD", "GBPNZD", "GBPCAD",
    "AUDJPY", "AUDCHF", "AUDNZD", "AUDCAD",
    "NZDJPY", "NZDCHF", "NZDCAD",
    "CADJPY", "CADCHF", "CHFJPY",
]
INTERVAL_SEC = {"1m": 60, "2m": 120, "5m": 300, "15m": 900, "30m": 1800, "60m": 3600, "1h": 3600}
MAX_DAYS = {"1m": 29, "2m": 59, "5m": 59, "15m": 59, "30m": 59, "60m": 729, "1h": 729}
DEFAULT_DAYS = {"1m": 29, "2m": 30, "5m": 59, "15m": 59, "30m": 59, "60m": 365, "1h": 365}

@dataclass(frozen=True)
class Params:
    atr_n: int = 14
    pivot_k: int = 3
    strong_body: float = 0.55
    strong_atr: float = 0.60
    weak_body: float = 0.35
    wick_ratio: float = 0.40
    round_step_pts: int = 50
    tol_atr: float = 0.15
    near_target_atr: float = 1.0
    min_score: float = 1.0
    train_frac: float = 0.70
    min_n_tune: int = 30
    low_vol_ratio: float = 0.5

# ==================== HELPERS ====================
def point_size(sym: str) -> float:
    return 0.001 if "JPY" in sym else 0.00001

def wilson(w: int, n: int, z: float = 1.96):
    if n <= 0:
        return (float("nan"), float("nan"))
    p = w / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))

# ==================== DATA LOADING ====================
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

@st.cache_data(ttl=600, show_spinner=False)
def fetch_yf(base: str, interval: str, days: int):
    if yf is None:
        raise RuntimeError("yfinance install nahi hai")
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
            raw = yf.download(ticker, start=cur_start, end=cur_end + timedelta(minutes=5),
                              interval=interval, progress=False, auto_adjust=False, threads=False)
            cl = _clean_yf(raw)
            if cl is not None and len(cl):
                frames.append(cl)
        except Exception:
            pass
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

# ==================== FEATURE ENGINEERING ====================
def compute_positions(o, h, l, c):
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
    sig = np.asarray(sig, dtype=float)
    same = (np.sign(sig) == trend) & (trend != 0)
    flat = trend == 0
    counter = (~same) & (~flat) & (sig != 0)
    w = np.where(same, 1.0, np.where(flat, 0.75, 0.5))
    w = np.where(counter & near, 0.0, w)
    return sig * w

def build(df: pd.DataFrame, sym: str, P: Params):
    d = df.copy()
    o, h, l, c = d["open"], d["high"], d["low"], d["close"]
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
                cont = (dr == 1) & strong & (c > retr_first_open)
                fail = (dr == -1) & strong & (c < mom_last_open)
            else:
                cont = (dr == -1) & strong & (c < retr_first_open)
                fail = (dr == 1) & strong & (c > mom_last_open)
            A += np.where(base & cont.to_numpy(), dm, 0)
            A += np.where(base & fail.to_numpy(), -dm, 0)
    A = np.clip(A, -1, 1)

    S, R = compute_positions(o.to_numpy(), h.to_numpy(), l.to_numpy(), c.to_numpy())
    S_prev = pd.Series(S, index=d.index).shift(1)
    R_prev = pd.Series(R, index=d.index).shift(1)
    d["S1"] = S_prev
    d["R1"] = R_prev
    bull_cnt = (dr == 1).rolling(5).sum()
    bear_cnt = (dr == -1).rolling(5).sum()
    no_res = R_prev.isna() | (h < R_prev - tol)
    no_sup = S_prev.isna() | (l > S_prev + tol)
    tap_sup = l <= S_prev + tol
    tap_res = h >= R_prev - tol
    trend_s = pd.Series(trend, index=d.index)
    b_up = (trend_s == 1) & ext & (dr == 1) & tap_sup & no_res & (bull_cnt >= 3)
    b_dn = (trend_s == -1) & ext & (dr == -1) & tap_res & no_sup & (bear_cnt >= 3)
    b_up_rev = (trend_s == 1) & ext & (dr == -1) & tap_res & (c < R_prev) & no_sup
    b_dn_rev = (trend_s == -1) & ext & (dr == 1) & tap_sup & (c > S_prev) & no_res
    B = (b_up.astype(float) + b_dn_rev.astype(float) - b_dn.astype(float) - b_up_rev.astype(float)).to_numpy()

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

    A_adj = trend_adjust(A, trend, near)
    B_adj = trend_adjust(B, trend, near)
    C_adj = trend_adjust(C, trend, near)
    score = A_adj + B_adj + C_adj
    bad = d["inside"].to_numpy() | d["lowvol"].to_numpy() | d["atr"].isna().to_numpy()
    bad[:60] = True
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

# ==================== STATS ====================
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

def pool_stats(stat_list, payout):
    n = sum(x["n"] for x in stat_list)
    w = sum(x["w"] for x in stat_list)
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
        stt = stats(decide(score, ms), d, slice(0, split), payout)
        if stt["n"] >= P.min_n_tune and stt["lb"] > best_key:
            best_key, best_P = stt["lb"], Pt
    return best_P

# ==================== ANALYZE ====================
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

# ==================== STREAMLIT UI ====================
st.set_page_config(page_title="FX Candle Scanner", layout="wide")
st.title("💹 FX Candle Scanner")
st.caption("Price action + Candle reaction + Buyer/Seller positions")

st.sidebar.header("⚙️ Settings")
interval = st.sidebar.selectbox("Interval", ["5m", "15m", "30m", "60m", "1m"], index=1)
max_days = int(min(DEFAULT_DAYS.get(interval, 30), 59))
days = st.sidebar.slider("Days of Data", 5, max_days, max_days)
min_score = st.sidebar.slider("Min Score", 0.5, 2.0, 1.0, 0.1)
payout = st.sidebar.slider("Binary Payout", 0.5, 1.0, 0.85, 0.01)
gate = st.sidebar.selectbox("Gate", ["point", "ci"], index=0)
tune = st.sidebar.checkbox("Tune Parameters (slow)", value=False)

st.header("🔍 Pair Selection")
mode = st.radio("Mode", ["Single Pair", "Multiple Pairs", "All 28 Pairs"], horizontal=True)
if mode == "Single Pair":
    selected_pairs = [st.selectbox("Pair", PAIRS)]
elif mode == "Multiple Pairs":
    selected_pairs = st.multiselect("Pairs", PAIRS, default=["EURUSD", "GBPJPY"])
else:
    selected_pairs = PAIRS

st.header("📁 Optional: TradingView CSV Upload")
uploaded = st.file_uploader("CSV file", type=["csv"])
csv_pair_name = st.text_input("CSV Pair Name", value="EURJPY")

if st.button("🚀 Run Scan", type="primary"):
    if uploaded is None and not selected_pairs:
        st.warning("Kam az kam ek pair select karo ya CSV upload karo.")
        st.stop()

    P = Params(min_score=min_score)
    all_results = []

    if uploaded is not None:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name
            df = load_csv(tmp_path)
            df = drop_incomplete(df, interval)
            os.unlink(tmp_path)
            with st.spinner(f"Analyzing {csv_pair_name}..."):
                result = analyze(csv_pair_name, df, P, payout, gate, tune=tune)
            all_results.append(result)
        except Exception as e:
            st.error(f"CSV error: {e}")
    else:
        progress = st.progress(0)
        status = st.empty()
        for i, pair in enumerate(selected_pairs):
            status.text(f"Fetching {pair}... ({i+1}/{len(selected_pairs)})")
            try:
                df = fetch_yf(pair, interval, days)
                if df is not None:
                    df = drop_incomplete(df, interval)
                result = analyze(pair, df, P, payout, gate, tune=tune)
                all_results.append(result)
            except Exception as e:
                all_results.append({"pair": pair, "error": str(e)})
            progress.progress((i + 1) / len(selected_pairs))
        status.text("Done ✅")

    st.header("📊 Summary")
    valid = [r for r in all_results if "error" not in r]
    errors = [r for r in all_results if "error" in r]

    if valid:
        summary_rows = []
        for r in valid:
            summary_rows.append({
                "Pair": r["pair"], "Signal": r["signal"], "Action": r["action"],
                "Trend": r["trend"], "Score": round(r["score"], 2), "Verdict": r["verdict"],
                "OOS Win%": f"{100 * r['hist_oos']['wr']:.1f}%" if r["hist_oos"]["n"] > 0 else "n/a",
                "OOS Trades": r["hist_oos"]["n"], "Reasons": r["reasons"],
            })
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    if errors:
        st.subheader("⚠️ Errors")
        for e in errors:
            st.warning(f"{e['pair']}: {e['error']}")

    if valid:
        st.header("🔎 Detail Analysis")
        for r in valid:
            with st.expander(f"**{r['pair']}** — {r['signal']} ({r['action']})"):
                c1, c2, c3 = st.columns(3)
                c1.metric("Close", f"{r['close']:.5f}")
                c2.metric("Trend", r["trend"])
                c3.metric("Score", f"{r['score']:.2f}")
                st.markdown(f"**Verdict:** {r['verdict']}")
                st.markdown(f"**Reasons:** {r['reasons']}")
                st.markdown(f"**Breakeven Win%:** {100 * r['be']:.1f}%")
                st.markdown(f"**Bars:** {r['bars']}")
                st.markdown("---")
                h = r["hist_all"]
                if h["n"] > 0:
                    st.write(f"**All:** Trades {h['n']} | Win% {100*h['wr']:.1f}% | CI [{100*h['lb']:.1f}%, {100*h['ub']:.1f}%] | EV {h['ev']:.3f}")
                h = r["hist_oos"]
                if h["n"] > 0:
                    st.write(f"**OOS:** Trades {h['n']} | Win% {100*h['wr']:.1f}% | CI [{100*h['lb']:.1f}%, {100*h['ub']:.1f}%] | EV {h['ev']:.3f}")
                st.markdown("---")
                mod_rows = []
                for name, all_s, oos_s in r["mods"]:
                    mod_rows.append({
                        "Module": name, "All Trades": all_s["n"],
                        "All Win%": f"{100*all_s['wr']:.1f}%" if all_s["n"] > 0 else "n/a",
                        "OOS Trades": oos_s["n"],
                        "OOS Win%": f"{100*oos_s['wr']:.1f}%" if oos_s["n"] > 0 else "n/a",
                    })
          
