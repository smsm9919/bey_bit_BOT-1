# -*- coding: utf-8 -*-
"""
BYBIT — SUI/USDT Perp — RF (Pine-exact CLOSED) + Council Smart Consensus
• Exchange: Bybit USDT Perps via CCXT
• Entry: Range Filter (TradingView-like), CLOSED-candle flip only (Pine-exact B&S logic)
• Priority: Bottom/Top hunting (Demand/Supply + swing + strong rejection) — Council consensus
• No partial TPs/ATR trail — strict close at council-confirmed max only (one-shot)
• Protective rails: spread guard, cooldown between orders, wait-for-opposite-RF after close
• Cumulative PnL tracking + reduceOnly strict close + tiny residual guard
• Flask /metrics /health + rotated logs
"""

import os, time, math, random, signal, sys, traceback, logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
import pandas as pd
import numpy as np
import ccxt
from flask import Flask, jsonify
from decimal import Decimal, ROUND_DOWN, InvalidOperation

try:
    from termcolor import colored
except Exception:
    def colored(t,*a,**k): return t

# =================== ENV / MODE ===================
API_KEY     = os.getenv("BYBIT_API_KEY", "")
API_SECRET  = os.getenv("BYBIT_API_SECRET", "")
MODE_LIVE   = bool(API_KEY and API_SECRET)

SELF_URL    = os.getenv("SELF_URL", "")
PORT        = int(os.getenv("PORT", 5000))

# =================== FIXED CONFIG (SUI) ===================
SYMBOL      = os.getenv("SYMBOL", "SUI/USDT:USDT")
INTERVAL    = os.getenv("INTERVAL", "15m")

LEVERAGE    = int(os.getenv("LEVERAGE", 10))
RISK_ALLOC  = float(os.getenv("RISK_ALLOC", 0.60))
POSITION_MODE = os.getenv("POSITION_MODE", "oneway")  # oneway / hedge

# Range Filter (TV-like, Pine-exact on CLOSED)
RF_SOURCE        = os.getenv("RF_SOURCE", "close")
RF_PERIOD        = int(os.getenv("RF_PERIOD", 20))
RF_MULT          = float(os.getenv("RF_MULT", 3.5))
RF_CLOSED_ONLY   = True  # Pine-exact CLOSED فقط

# Indicators (RSI/ADX/ATR — Wilder/TV-compat)
RSI_LEN = int(os.getenv("RSI_LEN", 14))
ADX_LEN = int(os.getenv("ADX_LEN", 14))
ATR_LEN = int(os.getenv("ATR_LEN", 14))

# Guards
MAX_SPREAD_BPS = float(os.getenv("MAX_SPREAD_BPS", 8.0))
FINAL_CHUNK_QTY = float(os.getenv("FINAL_CHUNK_QTY", 0.20))

# Council (consensus)
COUNCIL_MIN_VOTES_FOR_ENTRY  = 3
COUNCIL_MIN_VOTES_FOR_EXIT   = 3
LEVEL_NEAR_BPS               = 12.0
RETEST_MAX_BARS              = 6
ADX_COOL_DROP                = 2.0
RSI_NEUTRAL_MIN, RSI_NEUTRAL_MAX = 45.0, 55.0

# Bottom/Top hunting weights
WEIGHT_DEMAND_TOUCH_REJECT = 1.5
WEIGHT_SUPPLY_TOUCH_REJECT = 1.5
WEIGHT_SWING_CONFIRM       = 1.0
WEIGHT_CANDLE_REJECT       = 1.0

# Anti-flap
MIN_SECONDS_BETWEEN_ORDERS = 45
MIN_BARS_BETWEEN_ENTRIES   = 1

# Pacing
BASE_SLEEP   = 5
NEAR_CLOSE_S = 1

# =================== LOGGING ===================
def setup_file_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not any(isinstance(h, RotatingFileHandler) and getattr(h, "baseFilename", "").endswith("bot.log")
               for h in logger.handlers):
        fh = RotatingFileHandler("bot.log", maxBytes=5_000_000, backupCount=7, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
        logger.addHandler(fh)
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    print(colored("🗂️ log rotation ready", "cyan"))

setup_file_logging()

# =================== EXCHANGE (Bybit) ===================
def make_ex():
    return ccxt.bybit({
        "apiKey": API_KEY,
        "secret": API_SECRET,
        "enableRateLimit": True,
        "timeout": 20000,
        "options": {"defaultType": "swap"}
    })

ex = make_ex()
MARKET = {}
AMT_PREC = 0
LOT_STEP = None
LOT_MIN  = None

def load_market_specs():
    global MARKET, AMT_PREC, LOT_STEP, LOT_MIN
    try:
        ex.load_markets()
        MARKET = ex.markets.get(SYMBOL, {})
        AMT_PREC = int((MARKET.get("precision", {}) or {}).get("amount", 0) or 0)
        LOT_STEP = (MARKET.get("limits", {}) or {}).get("amount", {}).get("step", None)
        LOT_MIN  = (MARKET.get("limits", {}) or {}).get("amount", {}).get("min",  None)
        print(colored(f"🔧 precision={AMT_PREC}, step={LOT_STEP}, min={LOT_MIN}", "cyan"))
    except Exception as e:
        print(colored(f"⚠️ load_market_specs: {e}", "yellow"))

def ensure_leverage_mode():
    try:
        try:
            ex.set_leverage(LEVERAGE, SYMBOL, params={"side": "BOTH"})
        except Exception as e:
            print(colored(f"⚠️ set_leverage warn: {e}", "yellow"))
        print(colored(f"📌 position mode: {POSITION_MODE}", "cyan"))
    except Exception as e:
        print(colored(f"⚠️ ensure_leverage_mode: {e}", "yellow"))

try:
    load_market_specs()
    ensure_leverage_mode()
except Exception as e:
    print(colored(f"⚠️ exchange init: {e}", "yellow"))

# =================== HELPERS / STATE ===================
compound_pnl = 0.0
wait_for_next_signal_side = None    # بعد أي إغلاق: لازم RF عكسي قبل دخول جديد
last_order_ts = 0.0
last_entry_bar_time = None
last_adx_peak = None
cond_ini = None                     # Pine CondIni bootstrap

STATE = {
    "open": False, "side": None, "entry": None, "qty": 0.0,
    "pnl": 0.0, "bars": 0,
    "highest_profit_pct": 0.0,
    "breakeven": None,
    "council": {"votes": 0, "reasons": []}
}

def _round_amt(q):
    if q is None: return 0.0
    try:
        d = Decimal(str(q))
        if LOT_STEP and isinstance(LOT_STEP,(int,float)) and LOT_STEP>0:
            step = Decimal(str(LOT_STEP))
            d = (d/step).to_integral_value(rounding=ROUND_DOWN)*step
        prec = int(AMT_PREC) if AMT_PREC and AMT_PREC>=0 else 0
        d = d.quantize(Decimal(1).scaleb(-prec), rounding=ROUND_DOWN)
        if LOT_MIN and isinstance(LOT_MIN,(int,float)) and LOT_MIN>0 and d < Decimal(str(LOT_MIN)): return 0.0
        return float(d)
    except (InvalidOperation, ValueError, TypeError):
        return max(0.0, float(q))

def safe_qty(q):
    q = _round_amt(q)
    if q<=0: print(colored(f"⚠️ qty invalid after normalize → {q}", "yellow"))
    return q

def fmt(v, d=6, na="—"):
    try:
        if v is None or (isinstance(v,float) and (math.isnan(v) or math.isinf(v))): return na
        return f"{float(v):.{d}f}"
    except Exception:
        return na

def with_retry(fn, tries=3, base_wait=0.4):
    for i in range(tries):
        try:
            return fn()
        except Exception:
            if i == tries-1: raise
            time.sleep(base_wait*(2**i) + random.random()*0.25)

def fetch_ohlcv(limit=600):
    rows = with_retry(lambda: ex.fetch_ohlcv(SYMBOL, timeframe=INTERVAL, limit=limit, params={"type":"swap"}))
    return pd.DataFrame(rows, columns=["time","open","high","low","close","volume"])

def price_now():
    try:
        t = with_retry(lambda: ex.fetch_ticker(SYMBOL))
        return t.get("last") or t.get("close")
    except Exception: return None

def balance_usdt():
    if not MODE_LIVE: return 100.0
    try:
        b = with_retry(lambda: ex.fetch_balance(params={"type":"swap"}))
        return b.get("total",{}).get("USDT") or b.get("free",{}).get("USDT")
    except Exception: return None

def orderbook_spread_bps():
    try:
        ob = with_retry(lambda: ex.fetch_order_book(SYMBOL, limit=5))
        bid = ob["bids"][0][0] if ob["bids"] else None
        ask = ob["asks"][0][0] if ob["asks"] else None
        if not (bid and ask): return None
        mid = (bid+ask)/2.0
        return ((ask-bid)/mid)*10000.0
    except Exception:
        return None

def _interval_seconds(iv: str) -> int:
    iv=(iv or "").lower().strip()
    if iv.endswith("m"): return int(float(iv[:-1]))*60
    if iv.endswith("h"): return int(float(iv[:-1]))*3600
    if iv.endswith("d"): return int(float(iv[:-1]))*86400
    return 15*60

def time_to_candle_close(df: pd.DataFrame) -> int:
    tf = _interval_seconds(INTERVAL)
    if len(df) == 0: return tf
    cur_start_ms = int(df["time"].iloc[-1])
    now_ms = int(time.time()*1000)
    next_close_ms = cur_start_ms + tf*1000
    while next_close_ms <= now_ms:
        next_close_ms += tf*1000
    left = max(0, next_close_ms - now_ms)
    return int(left/1000)

# =================== INDICATORS (TV-Compat) ===================
def _rma(s: pd.Series, length: int):
    alpha = 1.0/float(length)
    return s.ewm(alpha=alpha, adjust=False).mean()

def _true_range(h, l, c):
    prev_c = c.shift(1)
    tr = pd.concat([(h-l).abs(), (h-prev_c).abs(), (l-prev_c).abs()], axis=1).max(axis=1)
    return tr

def compute_indicators(df: pd.DataFrame):
    global last_adx_peak
    if len(df) < max(ATR_LEN, RSI_LEN, ADX_LEN) + 2:
        return {"rsi":50.0,"plus_di":0.0,"minus_di":0.0,"dx":0.0,"adx":0.0,"atr":0.0,"atr_pctl":None}
    c = df["close"].astype(float); h=df["high"].astype(float); l=df["low"].astype(float)

    # RSI
    delta=c.diff(); up=delta.clip(lower=0.0); dn=(-delta).clip(lower=0.0)
    rma_up=_rma(up, RSI_LEN); rma_dn=_rma(dn, RSI_LEN).replace(0,1e-12)
    rs=rma_up/rma_dn; rsi=100-(100/(1+rs))

    # ADX
    up_move=h.diff(); down_move=l.shift(1)-l
    plus_dm=up_move.where((up_move>down_move)&(up_move>0),0.0)
    minus_dm=down_move.where((down_move>up_move)&(down_move>0),0.0)
    tr=_true_range(h,l,c); atr=_rma(tr, ATR_LEN).replace(0,1e-12)
    plus_di=100*_rma(plus_dm, ADX_LEN)/atr
    minus_di=100*_rma(minus_dm, ADX_LEN)/atr
    dx=(100*(plus_di-minus_di).abs()/(plus_di+minus_di).replace(0,1e-12)).fillna(0.0)
    adx=_rma(dx, ADX_LEN)

    # Keep last ADX peak for cooling detection
    try:
        cur_adx = float(adx.iloc[-1])
        if last_adx_peak is None or cur_adx > last_adx_peak:
            last_adx_peak = cur_adx
    except Exception:
        pass

    # ATR percentile (chop hint)
    atr_hist = _rma(tr, ATR_LEN)
    atr_pctl = None
    try:
        last_atr = float(atr.iloc[-1])
        window = atr_hist.iloc[-200:].dropna().astype(float).values
        if len(window)>=20:
            atr_pctl = float((np.sum(window <= last_atr)/len(window)))
    except Exception:
        atr_pctl = None

    i = len(df)-1
    return {
        "rsi": float(rsi.iloc[i]), "plus_di": float(plus_di.iloc[i]),
        "minus_di": float(minus_di.iloc[i]), "dx": float(dx.iloc[i]),
        "adx": float(adx.iloc[i]), "atr": float(atr.iloc[i]),
        "atr_pctl": atr_pctl
    }

# =================== RANGE FILTER — Pine-exact (CLOSED candle) ===================
def rf_signal_closed_pine(df: pd.DataFrame):
    """
    Pine-exact of 'Range Filter - B&S Signals' (DonovanWall) with CondIni logic:
    - نشتغل على آخر شمعة مغلقة فقط (k = -2)
    - longCondition/shortCondition مثل Pine حرفيًا
    """
    global cond_ini
    need = RF_PERIOD + 3
    n = len(df)
    if n < need:
        i = -1
        price = float(df["close"].iloc[i]) if n else 0.0
        return {
            "time": int(df["time"].iloc[i]) if n else int(time.time()*1000),
            "price": price, "long": False, "short": False,
            "filter": price, "hi": price, "lo": price
        }

    src = df[RF_SOURCE].astype(float)

    def _ema(s, span): return s.ewm(span=span, adjust=False).mean()
    def _rng_size(x, qty, per):
        wper = (per * 2) - 1
        avrng = _ema((x - x.shift(1)).abs(), per)
        return _ema(avrng, wper) * qty

    def _rng_filter(x, r):
        rf = [float(x.iloc[0])]
        for i in range(1, len(x)):
            prev = rf[-1]
            xi = float(x.iloc[i]); ri = float(r.iloc[i])
            cur = prev
            if xi - ri > prev: cur = xi - ri
            if xi + ri < prev: cur = xi + ri
            rf.append(cur)
        filt = pd.Series(rf, index=x.index, dtype="float64")
        return (filt + r), (filt - r), filt

    hi, lo, filt = _rng_filter(src, _rng_size(src, RF_MULT, RF_PERIOD))

    # آخر شمعة مغلقة
    k, km1 = -2, -3
    p_k   = float(src.iloc[k])
    f_k   = float(filt.iloc[k])
    f_km1 = float(filt.iloc[km1])

    upward   = 1 if f_k > f_km1 else 0
    downward = 1 if f_k < f_km1 else 0

    # شروط Pine
    longCond  = (p_k > f_k) and (upward > 0)
    shortCond = (p_k < f_k) and (downward > 0)

    prev_cond = cond_ini if cond_ini is not None else 0
    new_cond  = 1 if longCond else (-1 if shortCond else prev_cond)

    longSignal  = bool(longCond  and (prev_cond == -1))
    shortSignal = bool(shortCond and (prev_cond ==  1))

    cond_ini = new_cond

    return {
        "time": int(df["time"].iloc[k]),
        "price": float(src.iloc[-1]),
        "long": longSignal,
        "short": shortSignal,
        "filter": f_k,
        "hi": float(hi.iloc[k]),
        "lo": float(lo.iloc[k]),
    }

def bootstrap_cond_ini_from_history(df: pd.DataFrame):
    """تهيئة CondIni من التاريخ (شموع مُغلقة فقط) ليتطابق مع Pine."""
    global cond_ini
    try:
        if len(df) < RF_PERIOD + 3:
            cond_ini = 0 if cond_ini is None else cond_ini
            return
        src = df[RF_SOURCE].astype(float)

        def _ema(s, span): return s.ewm(span=span, adjust=False).mean()
        def _rng_size(x, qty, per):
            wper = (per * 2) - 1
            avrng = _ema((x - x.shift(1)).abs(), per)
            return _ema(avrng, wper) * qty
        def _rng_filter(x, r):
            rf = [float(x.iloc[0])]
            for i in range(1, len(x)):
                prev = rf[-1]
                xi = float(x.iloc[i]); ri = float(r.iloc[i])
                cur = prev
                if xi - ri > prev: cur = xi - ri
                if xi + ri < prev: cur = xi + ri
                rf.append(cur)
            return pd.Series(rf, index=x.index, dtype="float64")

        r = _rng_size(src, RF_MULT, RF_PERIOD)
        filt = _rng_filter(src, r)

        ci = 0
        for i in range(RF_PERIOD + 2, len(df) - 0):
            if i-1 < 0: continue
            p_k   = float(src.iloc[i])
            f_k   = float(filt.iloc[i])
            f_km1 = float(filt.iloc[i-1])
            upward   = 1 if f_k > f_km1 else 0
            downward = 1 if f_k < f_km1 else 0
            longCond  = (p_k > f_k) and (upward > 0)
            shortCond = (p_k < f_k) and (downward > 0)
            ci = 1 if longCond else (-1 if shortCond else ci)

        cond_ini = ci
        print(colored(f"🔧 CondIni bootstrapped → {cond_ini}", "cyan"))
    except Exception as e:
        print(colored(f"⚠️ bootstrap CondIni error: {e}", "yellow"))
        if cond_ini is None: cond_ini = 0

# =================== STRUCTURE / S&D BOXES ===================
def _find_swings(df: pd.DataFrame, left:int=2, right:int=2):
    if len(df) < left+right+3:
        return None, None
    h = df["high"].astype(float).values
    l = df["low"].astype(float).values
    ph = [None]*len(df); pl = [None]*len(df)
    for i in range(left, len(df)-right):
        if all(h[i] >= h[j] for j in range(i-left, i+right+1)): ph[i] = h[i]
        if all(l[i] <= l[j] for j in range(i-left, i+right+1)): pl[i] = l[i]
    return ph, pl

def _nearest_level(px, levels, bps=LEVEL_NEAR_BPS):
    try:
        levels = [lv for lv in levels if lv is not None]
        for lv in levels:
            if abs((px-lv)/lv)*10000.0 <= bps: return lv
    except Exception:
        pass
    return None

def detect_sdz(df: pd.DataFrame):
    """
    استنباط صناديق Supply/Demand مبسّط:
    - آخر swing قوي (شمعة جسم كبير + ذيول قصيرة) ⇒ صندوق.
    - demand: شمعة صاعدة قوية → box [قاع, قمة الجسم]، supply: هابطة قوية بالعكس.
    """
    try:
        d = df.copy()
        ob = None
        for i in range(len(d)-3, max(len(d)-60, 3), -1):
            o=float(d["open"].iloc[i]); c=float(d["close"].iloc[i])
            h=float(d["high"].iloc[i]); l=float(d["low"].iloc[i])
            rng=max(h-l,1e-12); body=abs(c-o)
            upper=h-max(o,c); lower=min(o,c)-l
            if body>=0.6*rng and (upper/rng)<=0.2 and (lower/rng)<=0.2:
                side="demand" if c>o else "supply"
                bot=min(o,c); top=max(o,c)
                ob={"side":side, "bot":bot, "top":top, "time": int(d["time"].iloc[i])}
                break
        return ob
    except Exception:
        return None

def detect_bottom_top_votes(df: pd.DataFrame, price: float):
    """
    تصويت لاكتشاف قاع/قمة مؤكدة:
    - لمس صندوق demand/supply + رفض قوي
    - تأكيد swing (قاع/قمة محلية)
    - شمعة رفض (Hammer / Shooting / Marubozu مع اتجاه معاكس)
    """
    votes = []
    # سلوك شمعة
    if len(df)<3:
        return votes
    o=float(df["open"].iloc[-2]); h=float(df["high"].iloc[-2])
    l=float(df["low"].iloc[-2]);  c=float(df["close"].iloc[-2])
    rng=max(h-l,1e-12); upper=h-max(o,c); lower=min(o,c)-l
    body=abs(c-o)
    candle = "NONE"
    if body>=0.85*rng and max(upper,lower)<=0.07*rng: candle="MARUBOZU"
    elif lower>=0.60*rng and body<=0.30*rng and c>o: candle="HAMMER"
    elif upper>=0.60*rng and body<=0.30*rng and c<o: candle="SHOOTING"

    sdz = detect_sdz(df.iloc[:-1])  # على الشموع المغلقة
    if sdz:
        if sdz["side"]=="demand" and abs((l - sdz["top"])/sdz["top"])*10000 <= LEVEL_NEAR_BPS:
            votes += ["demand_touch"]
            if candle in ("HAMMER","MARUBOZU"): votes += ["demand_reject"]
        if sdz["side"]=="supply" and abs((h - sdz["bot"])/sdz["bot"])*10000 <= LEVEL_NEAR_BPS:
            votes += ["supply_touch"]
            if candle in ("SHOOTING","MARUBOZU"): votes += ["supply_reject"]

    # swing تأكيد
    ph, pl = _find_swings(df, 2, 2)
    if pl and pl[-2] is not None and price>c: votes += ["swing_low_confirm"]
    if ph and ph[-2] is not None and price<c: votes += ["swing_high_confirm"]

    # تصويتات موزونة
    score = 0.0
    for v in votes:
        if v=="demand_touch" or v=="supply_touch": score += 1.0
        if v=="demand_reject": score += WEIGHT_DEMAND_TOUCH_REJECT
        if v=="supply_reject": score += WEIGHT_SUPPLY_TOUCH_REJECT
        if v=="swing_low_confirm" or v=="swing_high_confirm": score += WEIGHT_SWING_CONFIRM
        if v in ("HAMMER","SHOOTING"): score += WEIGHT_CANDLE_REJECT
    return votes, score

# =================== COUNCIL ===================
def council_entry(df, ind, info):
    """يجمع أصوات (قرب مستوى/صندوق + شموع + swing) لقرار الدخول."""
    price = info.get("price")
    votes = []
    # صناديق
    sdz = detect_sdz(df.iloc[:-1])
    if sdz:
        near = (abs((price - (sdz["bot"] if sdz["side"]=="supply" else sdz["top"]))/(sdz["top"])) * 10000.0) if sdz["side"]=="demand" else (abs((price - (sdz["bot"]))/(sdz["bot"])) * 10000.0)
        if near <= LEVEL_NEAR_BPS:
            votes.append(f"near_{sdz['side']}_box")
    # شموع
    _, btm_score = detect_bottom_top_votes(df, price)
    if btm_score >= 2.0: votes.append("swing_confirmed")
    # زخم
    if (RSI_NEUTRAL_MIN <= ind.get("rsi",50) <= RSI_NEUTRAL_MAX) and (ind.get("adx",0)>=18):
        votes.append("rsi_neutral_adx_ok")
    return {"votes": votes, "ok": len(votes) >= COUNCIL_MIN_VOTES_FOR_ENTRY, "sdz": sdz}

def council_exit(df, ind, info):
    """قرار إغلاق صارم عندما تتراكم الأسباب (رفض قوي/لمس صندوق معاكس/تبريد زخم...)."""
    votes=[]
    price=info["price"]
    sdz = detect_sdz(df.iloc[:-1])
    # رفض معاكس عند صندوق أو swing
    vt,_ = detect_bottom_top_votes(df, price)
    votes += [v for v in vt if "reject" in v]
    # تبريد ADX
    global last_adx_peak
    try:
        adx=float(ind.get("adx") or 0.0)
        if last_adx_peak and (last_adx_peak - adx) >= ADX_COOL_DROP:
            votes.append("adx_cool_off")
    except Exception:
        pass
    # لمس صندوق معاكس
    if sdz:
        if STATE.get("side")=="long" and sdz["side"]=="supply": votes.append("touch_supply_box")
        if STATE.get("side")=="short" and sdz["side"]=="demand": votes.append("touch_demand_box")
    return {"votes": votes, "ok": len(votes) >= COUNCIL_MIN_VOTES_FOR_EXIT, "sdz": sdz}

# =================== ORDERS ===================
def _params_open(side):
    if POSITION_MODE == "hedge":
        return {"positionSide": "LONG" if side=="buy" else "SHORT", "reduceOnly": False}
    return {"positionSide": "BOTH", "reduceOnly": False}

def _params_close():
    if POSITION_MODE == "hedge":
        return {"positionSide": "LONG" if STATE.get("side")=="long" else "SHORT", "reduceOnly": True}
    return {"positionSide": "BOTH", "reduceOnly": True}

def _read_position():
    try:
        poss = ex.fetch_positions(params={"type":"swap"})
        for p in poss:
            sym = (p.get("symbol") or p.get("info",{}).get("symbol") or "")
            if SYMBOL.split(":")[0] not in sym: continue
            qty = abs(float(p.get("contracts") or p.get("info",{}).get("positionAmt") or 0))
            if qty <= 0: return 0.0, None, None
            entry = float(p.get("entryPrice") or p.get("info",{}).get("avgEntryPrice") or 0)
            side_raw = (p.get("side") or p.get("info",{}).get("positionSide") or "").lower()
            side = "long" if ("long" in side_raw or float(p.get("cost",0))>0) else "short"
            return qty, side, entry
    except Exception as e:
        logging.error(f"_read_position error: {e}")
    return 0.0, None, None

def compute_size(balance, price):
    effective = (balance or 0.0)*RISK_ALLOC*LEVERAGE
    raw = max(0.0, effective / max(float(price or 0.0),1e-9))
    return safe_qty(raw)

def _cooldown_ok(df):
    global last_order_ts, last_entry_bar_time
    now=time.time()
    if (now - last_order_ts) < MIN_SECONDS_BETWEEN_ORDERS:
        return False, f"cooldown {int(MIN_SECONDS_BETWEEN_ORDERS-(now-last_order_ts))}s"
    # bars
    cur_bar_time = int(df["time"].iloc[-1])
    if last_entry_bar_time is not None and cur_bar_time == last_entry_bar_time:
        return False, "same bar"
    return True, None

def open_market(side, qty, price, df):
    global last_order_ts, last_entry_bar_time
    ok,msg = _cooldown_ok(df)
    if not ok:
        print(colored(f"⏸️ skip open — {msg}", "yellow"))
        return False
    if qty<=0:
        print(colored("❌ skip open (qty<=0)", "red")); return False
    if MODE_LIVE:
        try:
            try: ex.set_leverage(LEVERAGE, SYMBOL, params={"side":"BOTH"})
            except Exception: pass
            ex.create_order(SYMBOL, "market", side, qty, None, _params_open(side))
        except Exception as e:
            print(colored(f"❌ open: {e}", "red")); logging.error(f"open_market error: {e}")
            return False
    STATE.update({
        "open": True, "side": "long" if side=="buy" else "short", "entry": price,
        "qty": qty, "pnl": 0.0, "bars": 0,
        "highest_profit_pct": 0.0, "breakeven": None,
        "council": {"votes": 0, "reasons": []}
    })
    last_order_ts = time.time()
    last_entry_bar_time = int(df["time"].iloc[-1])
    print(colored(f"🚀 OPEN {('🟩 LONG' if side=='buy' else '🟥 SHORT')} qty={fmt(qty,4)} @ {fmt(price)}", "green" if side=='buy' else "red"))
    logging.info(f"OPEN {side} qty={qty} price={price}")
    return True

def close_market_strict(reason="STRICT"):
    global compound_pnl, wait_for_next_signal_side, last_order_ts
    exch_qty, exch_side, exch_entry = _read_position()
    if exch_qty <= 0:
        if STATE.get("open"): _reset_after_close(reason)
        return
    side_to_close = "sell" if (exch_side=="long") else "buy"
    qty_to_close  = safe_qty(exch_qty)
    attempts=0; last_error=None
    while attempts < 6:
        try:
            if MODE_LIVE:
                params=_params_close(); params["reduceOnly"]=True
                ex.create_order(SYMBOL,"market",side_to_close,qty_to_close,None,params)
            time.sleep(1.5)
            left_qty, _, _ = _read_position()
            if left_qty <= 0:
                px = price_now() or STATE.get("entry")
                entry_px = STATE.get("entry") or exch_entry or px
                side = STATE.get("side") or exch_side or ("long" if side_to_close=="sell" else "short")
                qty  = exch_qty
                pnl  = (px - entry_px) * qty * (1 if side=="long" else -1)
                compound_pnl += pnl
                print(colored(f"🔚 STRICT CLOSE {side} reason={reason} pnl={fmt(pnl)} total={fmt(compound_pnl)}","magenta"))
                logging.info(f"STRICT_CLOSE {side} pnl={pnl} total={compound_pnl}")
                _reset_after_close(reason, prev_side=side)
                last_order_ts = time.time()
                return
            qty_to_close = safe_qty(left_qty)
            attempts += 1
            print(colored(f"⚠️ strict close retry {attempts}/6 — residual={fmt(left_qty,4)}","yellow"))
            time.sleep(1.5)
        except Exception as e:
            last_error = e; logging.error(f"close_market_strict attempt {attempts+1}: {e}"); attempts += 1; time.sleep(1.5)
    print(colored(f"❌ STRICT CLOSE FAILED — last error: {last_error}", "red"))

def _reset_after_close(reason, prev_side=None):
    global wait_for_next_signal_side
    prev_side = prev_side or STATE.get("side")
    STATE.update({
        "open": False, "side": None, "entry": None, "qty": 0.0,
        "pnl": 0.0, "bars": 0, "highest_profit_pct": 0.0,
        "breakeven": None, "council": {"votes":0,"reasons":[]}
    })
    if prev_side == "long":  wait_for_next_signal_side = "sell"
    elif prev_side == "short": wait_for_next_signal_side = "buy"
    else: wait_for_next_signal_side = None
    logging.info(f"AFTER_CLOSE reason={reason} wait_for={wait_for_next_signal_side}")

# =================== AFTER-ENTRY MANAGEMENT ===================
def manage_after_entry(df, ind, info):
    if not STATE["open"] or STATE["qty"]<=0: return
    px = info["price"]; entry=STATE["entry"]; side=STATE["side"]
    rr = (px - entry)/entry*100*(1 if side=="long" else -1)

    # تتبع أعلى ربح
    if rr > STATE["highest_profit_pct"]:
        STATE["highest_profit_pct"] = rr

    # مجلس الإدارة يتابع الإغلاق الصارم عند تكدّس الأسباب
    decision = council_exit(df, ind, info)
    if decision["ok"]:
        STATE["council"] = {"votes": len(decision["votes"]), "reasons": decision["votes"]}
        close_market_strict("COUNCIL_STRICT_EXIT")
        return

# =================== SNAPSHOT ===================
def pretty_snapshot(bal, info, ind, spread_bps, reason=None, df=None):
    left_s = time_to_candle_close(df) if df is not None else 0
    print(colored("─"*110,"cyan"))
    print(colored(f"📊 {SYMBOL} {INTERVAL} • {'LIVE' if MODE_LIVE else 'PAPER'} • {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC","cyan"))
    print(colored("─"*110,"cyan"))
    print("📈 RF (CLOSED Pine-exact) & INDICATORS")
    print(f"   💲 Price {fmt(info.get('price'))} | RF filt={fmt(info.get('filter'))}  hi={fmt(info.get('hi'))} lo={fmt(info.get('lo'))} | spread={fmt(spread_bps,2)} bps")
    print(f"   🧮 RSI={fmt(ind.get('rsi'))}  +DI={fmt(ind.get('plus_di'))}  -DI={fmt(ind.get('minus_di'))}  ADX={fmt(ind.get('adx'))}  ATR={fmt(ind.get('atr'))}  ATRpctl={fmt(ind.get('atr_pctl'),3)}")
    print(f"   ⏱️ closes_in ≈ {left_s}s")

    print("\n🧭 POSITION")
    bal_line = f"Balance={fmt(bal,2)}  Risk={int(RISK_ALLOC*100)}%×{LEVERAGE}x  CompoundPnL={fmt(compound_pnl)}  Eq~{fmt((bal or 0)+compound_pnl,2)}"
    print(colored(f"   {bal_line}", "yellow"))
    if STATE["open"]:
        lamp='🟩 LONG' if STATE['side']=='long' else '🟥 SHORT'
        print(f"   {lamp}  Entry={fmt(STATE['entry'])}  Qty={fmt(STATE['qty'],4)}  Bars={STATE['bars']}  HP={fmt(STATE['highest_profit_pct'],2)}%")
        print(f"   🧠 Council: votes={STATE['council']['votes']} reasons={STATE['council']['reasons']}")
    else:
        print("   ⚪ FLAT")
        if wait_for_next_signal_side:
            print(colored(f"   ⏳ Waiting opposite RF: {wait_for_next_signal_side.upper()}", "cyan"))
    if reason: print(colored(f"   ℹ️ reason: {reason}", "white"))
    print(colored("─"*110,"cyan"))

# =================== LOOP ===================
def trade_loop():
    global cond_ini
    loop_i=0
    while True:
        try:
            bal = balance_usdt()
            px  = price_now()
            df  = fetch_ohlcv()

            # Bootstrap CondIni أول تشغيل
            if cond_ini is None:
                df_closed_init = df.iloc[:-1] if len(df) >= 2 else df.copy()
                bootstrap_cond_ini_from_history(df_closed_init)

            info = rf_signal_closed_pine(df)
            ind  = compute_indicators(df)
            spread_bps = orderbook_spread_bps()

            # تحديث PnL
            if STATE["open"] and px:
                STATE["pnl"] = (px-STATE["entry"])*STATE["qty"] if STATE["side"]=="long" else (STATE["entry"]-px)*STATE["qty"]

            # إدارة بعد الدخول
            manage_after_entry(df, ind, {"price": px or info["price"], **info})

            # حارس السبريد
            reason=None
            if spread_bps is not None and spread_bps > MAX_SPREAD_BPS:
                reason=f"spread too high ({fmt(spread_bps,2)}bps > {MAX_SPREAD_BPS})"

            # مجلس الإدارة للدخول (اصطياد قاع/قمة له أولوية)
            entry_council = council_entry(df, ind, {"price": px or info["price"], **info})

            # ENTRY: أولاً لو مجلس الإدارة يرى قاع/قمة مؤكدة ندخل، وإلا RF Pine-exact
            sig = None
            if not STATE["open"] and reason is None:
                # أولوية الصناديق/السوينغ
                if entry_council["ok"]:
                    sdz = entry_council["sdz"]
                    if sdz and sdz["side"]=="demand":
                        sig = "buy"
                    elif sdz and sdz["side"]=="supply":
                        sig = "sell"
                # وإلا التزم RF Pine-exact
                if sig is None:
                    sig = "buy" if info["long"] else ("sell" if info["short"] else None)

            # انتظار RF العكسي بعد أي إغلاق
            if not STATE["open"] and sig and reason is None:
                if wait_for_next_signal_side and sig != wait_for_next_signal_side:
                    reason=f"waiting opposite RF: need {wait_for_next_signal_side.upper()}"
                else:
                    qty = compute_size(bal, px or info["price"])
                    if qty>0:
                        ok = open_market(sig, qty, px or info["price"], df)
                        if ok:
                            wait_for_next_signal_side = None
                    else:
                        reason="qty<=0"

            pretty_snapshot(bal, {"price": px or info["price"], **info}, ind, spread_bps, reason, df)

            # عداد البارات
            if len(df)>=2 and int(df["time"].iloc[-1])!=int(df["time"].iloc[-2]) and STATE["open"]:
                STATE["bars"] += 1

            loop_i += 1
            sleep_s = NEAR_CLOSE_S if time_to_candle_close(df)<=10 else BASE_SLEEP
            time.sleep(sleep_s)
        except Exception as e:
            print(colored(f"❌ loop error: {e}\n{traceback.format_exc()}", "red"))
            logging.error(f"trade_loop error: {e}\n{traceback.format_exc()}")
            time.sleep(BASE_SLEEP)

# =================== API / KEEPALIVE ===================
app = Flask(__name__)
@app.route("/")
def home():
    mode='LIVE' if MODE_LIVE else 'PAPER'
    return f"✅ RF-CLOSED (Pine-exact) — SUI/USDT — {INTERVAL} — {mode} — Council smart consensus (bottom/top priority) — strict exit — FinalChunk={FINAL_CHUNK_QTY}"

@app.route("/metrics")
def metrics():
    return jsonify({
        "symbol": SYMBOL, "interval": INTERVAL, "mode": "live" if MODE_LIVE else "paper",
        "leverage": LEVERAGE, "risk_alloc": RISK_ALLOC, "price": price_now(),
        "state": STATE, "compound_pnl": compound_pnl,
        "entry_mode": "RF_CLOSED_ONLY_PINE_with_COUNCIL",
        "waiting_for": wait_for_next_signal_side
    })

@app.route("/health")
def health():
    return jsonify({
        "ok": True, "mode": "live" if MODE_LIVE else "paper",
        "open": STATE["open"], "side": STATE["side"], "qty": STATE["qty"],
        "compound_pnl": compound_pnl, "timestamp": datetime.utcnow().isoformat(),
        "entry_mode": "RF_CLOSED_ONLY_PINE_with_COUNCIL",
        "council_votes": STATE.get("council",{}).get("votes",0)
    }), 200

def keepalive_loop():
    url=(SELF_URL or "").strip().rstrip("/")
    if not url:
        print(colored("⛔ keepalive disabled (SELF_URL not set)", "yellow"))
        return
    import requests
    sess=requests.Session(); sess.headers.update({"User-Agent":"rf-closed-council/keepalive"})
    print(colored(f"KEEPALIVE every 50s → {url}", "cyan"))
    while True:
        try: sess.get(url, timeout=8)
        except Exception: pass
        time.sleep(50)

# =================== BOOT ===================
if __name__ == "__main__":
    print(colored(f"MODE: {'LIVE' if MODE_LIVE else 'PAPER'}  •  {SYMBOL}  •  {INTERVAL}", "yellow"))
    print(colored(f"RISK: {int(RISK_ALLOC*100)}% × {LEVERAGE}x  •  ENTRY=RF_CLOSED_ONLY (Pine-exact) + Council priority bottom/top", "yellow"))
    print(colored(f"NO partials/trailing • STRICT EXIT by COUNCIL • FINAL_CHUNK_QTY={FINAL_CHUNK_QTY}", "yellow"))
    logging.info("service starting…")
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    signal.signal(signal.SIGINT,  lambda *_: sys.exit(0))
    import threading
    threading.Thread(target=trade_loop, daemon=True).start()
    threading.Thread(target=keepalive_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)
