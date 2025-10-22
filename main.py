# -*- coding: utf-8 -*-
"""
RF Futures Bot — RF-LIVE FUSION PRO (BingX Perp via CCXT)
• Entry: Range Filter (TradingView-like) — LIVE CANDLE ONLY (no closed-candle entries)
• Post-entry: Fusion Orchestrator (SMC + Liquidity/Traps + Candles + RSI/ADX + ATR/Volume Explosions)
• Dynamic TP ladder + Breakeven + ATR-trailing (with ratchet lock)
• "Apex Confirmed" single-shot full take when last swing likely holds (save fees)
• Strict close with exchange verification + Dust/Final-chunk guard (≤ 50 DOGE)
• Opposite-signal WAIT policy after a close (enter only from RF again)
• Trap/Fakeout guard on opposite RF while in position (defensive partial + tighten trail + votes)
• Flask /metrics + /health + rotated logging + keepalive
• + EVX GUARD: انفجار/انهيار كفلتر حماية قبل الدخول وأثناء الصفقة وبعدها
"""

import os, time, math, random, signal, sys, traceback, logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
import pandas as pd
import ccxt
from flask import Flask, jsonify
from decimal import Decimal, ROUND_DOWN, InvalidOperation

try:
    from termcolor import colored
except Exception:
    def colored(t,*a,**k): return t

# =================== ENV / MODE ===================
API_KEY = os.getenv("BINGX_API_KEY", "")
API_SECRET = os.getenv("BINGX_API_SECRET", "")
MODE_LIVE = bool(API_KEY and API_SECRET)

SELF_URL = os.getenv("SELF_URL", "") or os.getenv("RENDER_EXTERNAL_URL", "")
PORT = int(os.getenv("PORT", 5000))

# =================== SETTINGS ===================
SYMBOL     = os.getenv("SYMBOL", "DOGE/USDT:USDT")
INTERVAL   = os.getenv("INTERVAL", "15m")
LEVERAGE   = int(os.getenv("LEVERAGE", 10))
RISK_ALLOC = float(os.getenv("RISK_ALLOC", 0.60))   # 60% of equity
POSITION_MODE = os.getenv("BINGX_POSITION_MODE", "oneway")  # oneway/hedge

# RF (TradingView-like) — live candle only
RF_SOURCE = "close"
RF_PERIOD = int(os.getenv("RF_PERIOD", 20))
RF_MULT   = float(os.getenv("RF_MULT", 3.5))
RF_LIVE_ONLY = True
RF_HYST_BPS  = 6.0  # hysteresis to reduce live-candle flicker

# Indicators
RSI_LEN = 14
ADX_LEN = 14
ATR_LEN = 14

# Entry only from RF
ENTRY_RF_ONLY = True

# Spread guard
MAX_SPREAD_BPS = float(os.getenv("MAX_SPREAD_BPS", 6.0))

# Dynamic TP / trail
TP1_PCT_BASE       = 0.40
TP1_CLOSE_FRAC     = 0.50
BREAKEVEN_AFTER    = 0.30
TRAIL_ACTIVATE_PCT = 1.20
ATR_TRAIL_MULT     = 1.6

# Trend defaults
TREND_TPS       = [0.50, 1.00, 1.80]
TREND_TP_FRACS  = [0.30, 0.30, 0.20]

# Dust / final-chunk guard
FINAL_CHUNK_QTY = float(os.getenv("FINAL_CHUNK_QTY", 50.0))  # ≤ 50 DOGE → strict close
RESIDUAL_MIN_QTY = float(os.getenv("RESIDUAL_MIN_QTY", 9.0)) # min practical lot

# Strict close
CLOSE_RETRY_ATTEMPTS = 6
CLOSE_VERIFY_WAIT_S  = 2.0

# Patience / Opposite RF votes before full close
OPP_RF_VOTES_NEEDED = 2
OPP_RF_MIN_ADX      = 22.0
OPP_RF_MIN_HYST_BPS = 8.0

# Ratchet lock
RATCHET_LOCK_FALLBACK = 0.60

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

# =================== EXCHANGE ===================
def make_ex():
    return ccxt.bingx({
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
            print(colored(f"✅ leverage set: {LEVERAGE}x", "green"))
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

# =================== HELPERS ===================
_consec_err = 0
last_loop_ts = time.time()

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
    global _consec_err
    for i in range(tries):
        try:
            r = fn()
            _consec_err = 0
            return r
        except Exception:
            _consec_err += 1
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

# =================== INDICATORS ===================
def wilder_ema(s: pd.Series, n: int): 
    return s.ewm(alpha=1/n, adjust=False).mean()

def compute_indicators(df: pd.DataFrame):
    if len(df) < max(ATR_LEN, RSI_LEN, ADX_LEN) + 2:
        return {"rsi":50.0,"plus_di":0.0,"minus_di":0.0,"dx":0.0,"adx":0.0,"atr":0.0}
    c,h,l = df["close"].astype(float), df["high"].astype(float), df["low"].astype(float)
    tr = pd.concat([(h-l).abs(), (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    atr = wilder_ema(tr, ATR_LEN)

    delta=c.diff(); up=delta.clip(lower=0.0); dn=(-delta).clip(lower=0.0)
    rs = wilder_ema(up, RSI_LEN) / wilder_ema(dn, RSI_LEN).replace(0,1e-12)
    rsi = 100 - (100/(1+rs))

    up_move=h.diff(); down_move=l.shift(1)-l
    plus_dm=up_move.where((up_move>down_move)&(up_move>0),0.0)
    minus_dm=down_move.where((down_move>up_move)&(down_move>0),0.0)
    plus_di=100*(wilder_ema(plus_dm, ADX_LEN)/atr.replace(0,1e-12))
    minus_di=100*(wilder_ema(minus_dm, ADX_LEN)/atr.replace(0,1e-12))
    dx=(100*(plus_di-minus_di).abs()/(plus_di+minus_di).replace(0,1e-12)).fillna(0.0)
    adx=wilder_ema(dx, ADX_LEN)

    i=len(df)-1
    return {
        "rsi": float(rsi.iloc[i]), "plus_di": float(plus_di.iloc[i]),
        "minus_di": float(minus_di.iloc[i]), "dx": float(dx.iloc[i]),
        "adx": float(adx.iloc[i]), "atr": float(atr.iloc[i])
    }

# =================== RANGE FILTER (TV-like) ===================
def _ema(s: pd.Series, n: int): return s.ewm(span=n, adjust=False).mean()
def _rng_size(src: pd.Series, qty: float, n: int) -> pd.Series:
    avrng = _ema((src - src.shift(1)).abs(), n); wper = (n*2)-1
    return _ema(avrng, wper) * qty

def _rng_filter(src: pd.Series, rsize: pd.Series):
    rf=[float(src.iloc[0])]
    for i in range(1,len(src)):
        prev=rf[-1]; x=float(src.iloc[i]); r=float(rsize.iloc[i]); cur=prev
        if x - r > prev: cur = x - r
        if x + r < prev: cur = x + r
        rf.append(cur)
    filt=pd.Series(rf, index=src.index, dtype="float64")
    return filt + rsize, filt - rsize, filt

def rf_signal_live(df: pd.DataFrame):
    """RF signal on LIVE candle (no close needed) with simple hysteresis."""
    if len(df) < RF_PERIOD + 3:
        i = -1
        price = float(df["close"].iloc[i]) if len(df) else None
        return {"time": int(df["time"].iloc[i]) if len(df) else int(time.time()*1000),
                "price": price or 0.0, "long": False, "short": False,
                "filter": price or 0.0, "hi": price or 0.0, "lo": price or 0.0}
    src = df[RF_SOURCE].astype(float)
    hi, lo, filt = _rng_filter(src, _rng_size(src, RF_MULT, RF_PERIOD))

    def _bps(a,b):
        try: return abs((a-b)/b)*10000.0
        except Exception: return 0.0

    p_now = float(src.iloc[-1]); p_prev = float(src.iloc[-2])
    f_now = float(filt.iloc[-1]); f_prev = float(filt.iloc[-2])

    long_flip  = (p_prev <= f_prev and p_now > f_now and _bps(p_now, f_now) >= RF_HYST_BPS)
    short_flip = (p_prev >= f_prev and p_now < f_now and _bps(p_now, f_now) >= RF_HYST_BPS)

    return {
        "time": int(df["time"].iloc[-1]), "price": p_now,
        "long": bool(long_flip), "short": bool(short_flip),
        "filter": f_now, "hi": float(hi.iloc[-1]), "lo": float(lo.iloc[-1])
    }

# =================== SMC / STRUCTURE / LIQUIDITY ===================
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

def detect_smc_levels(df: pd.DataFrame):
    """Equal High/Low + a recent OB + a recent simple FVG window"""
    try:
        d = df.copy()
        ph, pl = _find_swings(d, 2, 2)
        # Equal High/Low (tolerance 0.05%)
        def _eq_levels(vals, is_high=True):
            res = []
            tol_pct = 0.05
            for i, price in enumerate(vals):
                if price is None: continue
                tol = price * tol_pct / 100.0
                neighbors = [vals[j] for j in range(max(0,i-10), min(len(vals),i+10))
                             if vals[j] is not None and abs(vals[j] - price) <= tol]
                if len(neighbors) >= 2:
                    res.append(max(neighbors) if is_high else min(neighbors))
            if not res: return None
            return max(res) if is_high else min(res)
        eqh = _eq_levels(ph, True)
        eql = _eq_levels(pl, False)

        # Simple OB: last strong candle with small wicks (body dominates)
        ob = None
        for i in range(len(d)-2, max(len(d)-40, 1), -1):
            o=float(d["open"].iloc[i]); c=float(d["close"].iloc[i])
            h=float(d["high"].iloc[i]); l=float(d["low"].iloc[i])
            rng=max(h-l,1e-12); body=abs(c-o)
            upper=h-max(o,c); lower=min(o,c)-l
            if body>=0.6*rng and (upper/rng)<=0.2 and (lower/rng)<=0.2:
                side="bull" if c>o else "bear"
                ob={"side":side,"bot":min(o,c),"top":max(o,c),"time":int(d["time"].iloc[i])}
                break

        # Simple FVG (within last 20 bars)
        fvg=None
        for i in range(len(d)-3, max(len(d)-20, 2), -1):
            prev_high = float(d["high"].iloc[i-1]); prev_low = float(d["low"].iloc[i-1])
            curr_low  = float(d["low"].iloc[i]);   curr_high = float(d["high"].iloc[i])
            if curr_low > prev_high:
                fvg={"type":"BULL_FVG","bottom":prev_high,"top":curr_low}; break
            if curr_high < prev_low:
                fvg={"type":"BEAR_FVG","bottom":curr_high,"top":prev_low}; break

        return {"eqh":eqh, "eql":eql, "ob":ob, "fvg":fvg}
    except Exception:
        return {"eqh":None,"eql":None,"ob":None,"fvg":None}

def _near_level(px, lvl, bps):
    try: return abs((px-lvl)/lvl)*10000.0 <= bps
    except Exception: return False

def detect_stop_hunt(df: pd.DataFrame, smc: dict):
    """Long wick near EQH/EQL/OB/FVG + volume spike ⇒ potential trap/stop-hunt"""
    if len(df) < 3: return None
    try:
        o=float(df["open"].iloc[-1]); h=float(df["high"].iloc[-1])
        l=float(df["low"].iloc[-1]);  c=float(df["close"].iloc[-1])
        rng=max(h-l,1e-12); body=abs(c-o)
        upper=h-max(o,c); lower=min(o,c)-l
        upper_pct=upper/rng*100.0; lower_pct=lower/rng*100.0; body_pct=body/rng*100.0
        v=float(df["volume"].iloc[-1]); vma=df["volume"].iloc[-21:-1].astype(float).mean() if len(df)>=21 else 0.0
        vol_ok=(vma>0 and (v/vma)>=1.3)

        eqh=smc.get("eqh"); eql=smc.get("eql"); ob=smc.get("ob"); fvg=smc.get("fvg")
        near_eqh = (eqh and _near_level(h, eqh, 12.0))
        near_eql = (eql and _near_level(l, eql, 12.0))
        near_ob_res = (ob and ob.get("side")=="bear" and _near_level(h, ob["bot"], 12.0))
        near_ob_sup = (ob and ob.get("side")=="bull" and _near_level(l, ob["top"], 12.0))
        near_fvg_res = (fvg and fvg.get("type")=="BEAR_FVG" and _near_level(h, fvg.get("bottom", h), 12.0))
        near_fvg_sup = (fvg and fvg.get("type")=="BULL_FVG" and _near_level(l, fvg.get("top", l), 12.0))

        bull_trap = (lower_pct>=60 and body_pct<=25 and (near_eql or near_ob_sup or near_fvg_sup))
        bear_trap = (upper_pct>=60 and body_pct<=25 and (near_eqh or near_ob_res or near_fvg_res))

        if (bull_trap or bear_trap) and vol_ok:
            return {"trap": "bull" if bull_trap else "bear"}
    except Exception:
        pass
    return None

# =================== CANDLES & EXPLOSIONS ===================
def detect_candle(df: pd.DataFrame):
    if len(df)<3:
        return {"pattern":"NONE","strength":0,"dir":0}
    o=float(df["open"].iloc[-1]); h=float(df["high"].iloc[-1])
    l=float(df["low"].iloc[-1]);  c=float(df["close"].iloc[-1])
    rng=max(h-l,1e-12); body=abs(c-o)
    upper=h-max(o,c); lower=min(o,c)-l
    upper_pct=upper/rng*100.0; lower_pct=lower/rng*100.0; body_pct=body/rng*100.0
    if body_pct<=10: return {"pattern":"DOJI","strength":1,"dir":0}
    if body_pct>=85 and upper_pct<=7 and lower_pct<=7:
        return {"pattern":"MARUBOZU","strength":3,"dir":(1 if c>o else -1)}
    if lower_pct>=60 and body_pct<=30 and c>o: return {"pattern":"HAMMER","strength":2,"dir":1}
    if upper_pct>=60 and body_pct<=30 and c<o: return {"pattern":"SHOOTING","strength":2,"dir":-1}
    return {"pattern":"NONE","strength":0,"dir":(1 if c>o else -1)}

def explosion_signal(df: pd.DataFrame, ind: dict):
    """ATR & Volume spike → potential explosion direction"""
    if len(df)<21: return {"explosion":False,"dir":0,"ratio":0.0}
    try:
        v=float(df["volume"].iloc[-1]); vma=df["volume"].iloc[-21:-1].astype(float).mean() or 1e-9
        atr=float(ind.get("atr") or 0.0); 
        o=float(df["open"].iloc[-1]); c=float(df["close"].iloc[-1]); body=abs(c-o)
        react=(body/max(atr,1e-9))
        ratio=v/vma
        strong = (ratio>=1.8 and react>=1.2)
        return {"explosion":bool(strong),"dir":(1 if c>o else -1),"ratio":float(ratio)}
    except Exception:
        return {"explosion":False,"dir":0,"ratio":0.0}

# >>> EVX GUARD — SIGNAL — START
def evx_signal(df: pd.DataFrame, atr_now: float):
    """
    فلتر الانفجار/الانهيار (EVX):
      v_ratio = Volume / MA20
      react   = |close-open| / ATR(14)
      z       = Z-Score لمدى الشمعة مقابل نافذة 20
    """
    if len(df) < 25 or (atr_now or 0) <= 0:
        return {"evx": False, "dir": 0, "v_ratio": 1.0, "react": 0.0, "z": 0.0}

    v = float(df["volume"].iloc[-1])
    vma = df["volume"].astype(float).rolling(20).mean().iloc[-2] or 1e-9
    v_ratio = v / vma

    o = float(df["open"].iloc[-1]); c = float(df["close"].iloc[-1])
    body = abs(c - o)
    react = body / max(float(atr_now), 1e-9)

    hi = float(df["high"].iloc[-1]); lo = float(df["low"].iloc[-1])
    rng = hi - lo
    rng20 = (df["high"].astype(float).rolling(20).max()
             - df["low"].astype(float).rolling(20).min())
    mu = float(rng20.iloc[-20:-1].mean())
    sd = float(rng20.iloc[-20:-1].std() or 1e-9)
    z = (rng - mu) / sd

    EVX_V_SPIKE   = float(os.getenv("EVX_V_SPIKE", "2.0"))
    EVX_ATR_REACT = float(os.getenv("EVX_ATR_REACT", "1.4"))
    EVX_ZSCORE    = float(os.getenv("EVX_ZSCORE", "1.8"))

    evx = (v_ratio >= EVX_V_SPIKE) and (react >= EVX_ATR_REACT) and (z >= EVX_ZSCORE)
    dirn = 1 if c > o else -1 if c < o else 0

    return {"evx": bool(evx), "dir": dirn, "v_ratio": float(v_ratio), "react": float(react), "z": float(z)}
# >>> EVX GUARD — SIGNAL — END

# =================== FUSION ORCHESTRATOR ===================
def fusion_orchestrator(df: pd.DataFrame, ind: dict, info: dict, smc: dict):
    """Builds a joint view: momentum/trend/structure/trap/liq/explosion ⇒ scores & flags."""
    try:
        # momentum / trend
        adx=float(ind.get("adx") or 0.0)
        rsi=float(ind.get("rsi") or 50.0)
        pdi=float(ind.get("plus_di") or 0.0); mdi=float(ind.get("minus_di") or 0.0)
        side_bias = 1 if pdi>mdi else (-1 if mdi>pdi else 0)
        mom = (1.0 if adx>=28 else 0.5 if adx>=20 else 0.2) \
              + (0.5 if rsi>=55 else 0.5 if rsi<=45 else 0.0)

        # structure / liquidity targets
        structure=0.0
        price=info.get("price")
        if smc.get("eqh") and price and price>smc["eqh"]: structure+=0.5
        if smc.get("eql") and price and price<smc["eql"]: structure+=0.5
        if smc.get("ob"):
            ob=smc["ob"]; 
            structure += 0.3

        # trap risk
        trap = detect_stop_hunt(df, smc)
        trap_risk = 0.6 if trap else 0.0

        # classic explosion
        expl = explosion_signal(df, ind)

        # >>> EVX GUARD — FUSION HOOK — START
        evx = evx_signal(df, ind.get("atr", 0.0))
        boom = 0.6 if evx["evx"] else (0.6 if expl["explosion"] else 0.0)
        # >>> EVX GUARD — FUSION HOOK — END

        # candle context
        cnd = detect_candle(df)
        cscore = 0.2 if cnd["pattern"] in ("MARUBOZU","HAMMER","SHOOTING") else 0.0

        fusion = min(1.0, max(0.0, 0.25*mom + 0.35*structure + 0.25*boom + 0.15*cscore))
        return {
            "fusion_score": float(fusion),
            "trap_risk": float(trap_risk),
            "explosion": expl,
            "candle": cnd,
            "structure_bias": side_bias,
            "smc": smc,
            "trap": trap,
            # >>> EVX GUARD — RETURN — START
            "evx": evx
            # >>> EVX GUARD — RETURN — END
        }
    except Exception:
        return {"fusion_score":0.0,"trap_risk":0.0,"explosion":{"explosion":False,"dir":0,"ratio":0.0},
                "candle":{"pattern":"NONE","strength":0,"dir":0},"structure_bias":0,"smc":smc,"trap":None,
                "evx":{"evx":False,"dir":0,"v_ratio":1.0,"react":0.0,"z":0.0}}

# =================== APEX (FINAL SWING) CONFIRM ===================
def apex_confirmed(side: str, df: pd.DataFrame, ind: dict, smc: dict):
    """
    Decide last swing likely holds → single-shot full take:
    • near EQH/EQL or OB edge
    • long wick rejection + ADX cooling or RSI neutral/divergence
    """
    try:
        price=float(df["close"].iloc[-1]); adx=float(ind.get("adx") or 0.0)
        rsi=float(ind.get("rsi") or 50.0); atr=float(ind.get("atr") or 0.0)
        o=float(df["open"].iloc[-1]); h=float(df["high"].iloc[-1])
        l=float(df["low"].iloc[-1]);  c=float(df["close"].iloc[-1])
        rng=max(h-l,1e-12); upper=h-max(o,c); lower=min(o,c)-l
        near_top = (smc.get("eqh") and _near_level(h, smc["eqh"], 10.0)) or (smc.get("ob") and smc["ob"].get("side")=="bear" and _near_level(h, smc["ob"]["bot"], 10.0))
        near_bot = (smc.get("eql") and _near_level(l, smc["eql"], 10.0)) or (smc.get("ob") and smc["ob"].get("side")=="bull" and _near_level(l, smc["ob"]["top"], 10.0))

        reject_top = (upper/rng>=0.55 and (adx<20 or 45<=rsi<=55))
        reject_bot = (lower/rng>=0.55 and (adx<20 or 45<=rsi<=55))

        if side=="long" and near_top and reject_top and atr>0:
            return True
        if side=="short" and near_bot and reject_bot and atr>0:
            return True
    except Exception:
        pass
    return False

# =================== STATE ===================
STATE = {
    "open": False, "side": None, "entry": None, "qty": 0.0,
    "pnl": 0.0, "bars": 0, "trail": None, "breakeven": None,
    "tp1_done": False, "highest_profit_pct": 0.0,
    "profit_targets_achieved": 0,
    # fusion diagnostics
    "fusion_score": 0.0, "trap_risk": 0.0, "opp_votes": 0
}
compound_pnl = 0.0
wait_for_next_signal_side = None  # "buy" or "sell"

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
    effective = balance or 0.0
    capital = effective * RISK_ALLOC * LEVERAGE
    raw = max(0.0, capital / max(float(price or 0.0), 1e-9))
    return safe_qty(raw)

def open_market(side, qty, price):
    if qty<=0: 
        print(colored("❌ skip open (qty<=0)", "red"))
        return False
    if MODE_LIVE:
        try:
            try: ex.set_leverage(LEVERAGE, SYMBOL, params={"side":"BOTH"})
            except Exception: pass
            ex.create_order(SYMBOL, "market", side, qty, None, _params_open(side))
        except Exception as e:
            print(colored(f"❌ open: {e}", "red"))
            logging.error(f"open_market error: {e}")
            return False
    STATE.update({
        "open": True, "side": "long" if side=="buy" else "short", "entry": price,
        "qty": qty, "pnl": 0.0, "bars": 0, "trail": None, "breakeven": None,
        "tp1_done": False, "highest_profit_pct": 0.0, "profit_targets_achieved": 0,
        "opp_votes": 0
    })
    print(colored(f"🚀 OPEN {('🟩 LONG' if side=='buy' else '🟥 SHORT')} qty={fmt(qty,4)} @ {fmt(price)}", "green" if side=="buy" else "red"))
    logging.info(f"OPEN {side} qty={qty} price={price}")
    return True

def close_market_strict(reason="STRICT"):
    global compound_pnl, wait_for_next_signal_side
    exch_qty, exch_side, exch_entry = _read_position()
    if exch_qty <= 0:
        if STATE.get("open"):
            _reset_after_close(reason)
        return
    side_to_close = "sell" if (exch_side=="long") else "buy"
    qty_to_close  = safe_qty(exch_qty)
    attempts=0; last_error=None
    while attempts < CLOSE_RETRY_ATTEMPTS:
        try:
            if MODE_LIVE:
                params = _params_close(); params["reduceOnly"]=True
                ex.create_order(SYMBOL,"market",side_to_close,qty_to_close,None,params)
            time.sleep(CLOSE_VERIFY_WAIT_S)
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
                return
            qty_to_close = safe_qty(left_qty)
            attempts += 1
            print(colored(f"⚠️ strict close retry {attempts}/{CLOSE_RETRY_ATTEMPTS} — residual={fmt(left_qty,4)}","yellow"))
            time.sleep(CLOSE_VERIFY_WAIT_S)
        except Exception as e:
            last_error = e; logging.error(f"close_market_strict attempt {attempts+1}: {e}"); attempts += 1; time.sleep(CLOSE_VERIFY_WAIT_S)
    print(colored(f"❌ STRICT CLOSE FAILED after {CLOSE_RETRY_ATTEMPTS} attempts — last error: {last_error}", "red"))
    logging.critical(f"STRICT CLOSE FAILED — last_error={last_error}")

def _reset_after_close(reason, prev_side=None):
    global wait_for_next_signal_side
    prev_side = prev_side or STATE.get("side")
    STATE.update({
        "open": False, "side": None, "entry": None, "qty": 0.0,
        "pnl": 0.0, "bars": 0, "trail": None, "breakeven": None,
        "tp1_done": False, "highest_profit_pct": 0.0, "profit_targets_achieved": 0,
        "opp_votes": 0
    })
    if prev_side == "long":  wait_for_next_signal_side = "sell"
    elif prev_side == "short": wait_for_next_signal_side = "buy"
    else: wait_for_next_signal_side = None
    logging.info(f"AFTER_CLOSE waiting_for={wait_for_next_signal_side}")

def close_partial(frac, reason):
    """Partial close + residual guard + auto strict close when remaining ≤ FINAL_CHUNK_QTY."""
    if not STATE["open"] or STATE["qty"]<=0: return
    qty_close = safe_qty(max(0.0, STATE["qty"] * min(max(frac,0.0),1.0)))
    px = price_now() or STATE["entry"]
    min_unit = max(RESIDUAL_MIN_QTY, LOT_MIN or RESIDUAL_MIN_QTY)
    if qty_close < min_unit:
        print(colored(f"⏸️ skip partial (amount={fmt(qty_close,4)} < min_unit={fmt(min_unit,4)})", "yellow"))
        return
    side = "sell" if STATE["side"]=="long" else "buy"
    if MODE_LIVE:
        try: ex.create_order(SYMBOL,"market",side,qty_close,None,_params_close())
        except Exception as e: print(colored(f"❌ partial close: {e}", "red")); return
    pnl = (px - STATE["entry"]) * qty_close * (1 if STATE["side"]=="long" else -1)
    STATE["qty"] = safe_qty(STATE["qty"] - qty_close)
    logging.info(f"PARTIAL_CLOSE {reason} qty={qty_close} pnl={pnl} rem={STATE['qty']}")
    print(colored(f"🔻 PARTIAL {reason} closed={fmt(qty_close,4)} pnl={fmt(pnl)} rem={fmt(STATE['qty'],4)}","magenta"))
    if STATE["qty"] <= FINAL_CHUNK_QTY and STATE["qty"]>0:
        print(colored(f"🧹 Final chunk ≤ {FINAL_CHUNK_QTY} DOGE → strict close", "yellow"))
        close_market_strict("FINAL_CHUNK_RULE")

# =================== DEFENSIVE ON OPPOSITE RF WHILE IN POSITION ===================
def defensive_on_opposite_rf(ind: dict, info: dict):
    """Do NOT reverse. Defensive partial + tighten trail + collect votes. Full close only after votes+confirm."""
    if not STATE["open"] or STATE["qty"]<=0: return
    px = info.get("price") or price_now() or STATE["entry"]
    rf = info.get("filter")
    adx=float(ind.get("adx") or 0.0)
    base_frac = 0.25 if not STATE.get("tp1_done") else 0.20
    close_partial(base_frac, "Opposite RF — defensive")
    if STATE.get("breakeven") is None: STATE["breakeven"]=STATE["entry"]
    atr=float(ind.get("atr") or 0.0)
    if atr>0 and px is not None:
        gap = atr * max(ATR_TRAIL_MULT, 1.2)
        if STATE["side"]=="long":
            STATE["trail"]=max(STATE["trail"] or (px-gap), px-gap)
        else:
            STATE["trail"]=min(STATE["trail"] or (px+gap), px+gap)
    STATE["opp_votes"]=int(STATE.get("opp_votes",0))+1

    hyst=0.0
    try:
        if px and rf: hyst = abs((px-rf)/rf)*10000.0
    except Exception: pass
    votes_ok = STATE["opp_votes"]>=OPP_RF_VOTES_NEEDED
    confirmed = (adx>=OPP_RF_MIN_ADX) and (hyst>=OPP_RF_MIN_HYST_BPS)
    if votes_ok and confirmed:
        close_market_strict("OPPOSITE_RF_CONFIRMED")

# =================== DYNAMIC TP ===================
def _consensus(ind, info, side) -> float:
    score=0.0
    try:
        adx=float(ind.get("adx") or 0.0)
        rsi=float(ind.get("rsi") or 50.0)
        if (side=="long" and rsi>=55) or (side=="short" and rsi<=45): score += 1.0
        if adx>=28: score += 1.0
        elif adx>=20: score += 0.5
        if abs(info["price"]-info["filter"])/max(info["filter"],1e-9) >= (RF_HYST_BPS/10000.0): score += 0.5
    except Exception: pass
    return float(score)

def _tp_ladder(info, ind, side):
    px = info["price"]; atr = float(ind.get("atr") or 0.0)
    atr_pct = (atr / max(px,1e-9))*100.0 if px else 0.5
    score = _consensus(ind, info, side)
    if score >= 2.5: mults = [1.8, 3.2, 5.0]
    elif score >= 1.5: mults = [1.6, 2.8, 4.5]
    else: mults = [1.2, 2.4, 4.0]
    tps = [round(m*atr_pct, 2) for m in mults]
    frs = [0.25, 0.30, 0.45]
    return tps, frs

def manage_after_entry(df, ind, info, fusion):
    """Breakeven + Dynamic TP + ATR trail + Apex single-shot + Ratchet lock."""
    if not STATE["open"] or STATE["qty"]<=0: return
    px = info["price"]; entry=STATE["entry"]; side=STATE["side"]
    rr = (px - entry)/entry*100*(1 if side=="long" else -1)

    # dyn ladder
    dyn_tps, dyn_fracs = _tp_ladder(info, ind, side)
    STATE["_tp_cache"]=dyn_tps; STATE["_tp_fracs"]=dyn_fracs
    k = int(STATE.get("profit_targets_achieved", 0))

    # TP1 baseline (adaptive by ADX)
    tp1_now = TP1_PCT_BASE*(2.2 if ind.get("adx",0)>=35 else 1.8 if ind.get("adx",0)>=28 else 1.0)
    if (not STATE["tp1_done"]) and rr >= tp1_now:
        close_partial(TP1_CLOSE_FRAC, f"TP1@{tp1_now:.2f}%")
        STATE["tp1_done"]=True
        if rr >= BREAKEVEN_AFTER: STATE["breakeven"]=entry

    # APEX single-shot
    smc = fusion.get("smc", {})
    if rr >= max(0.4, TP1_PCT_BASE*0.8) and apex_confirmed(side, df, ind, smc):
        close_market_strict("APEX_CONFIRMED_FULL_TAKE")
        return

    # explosion/hold bias: delay 1st dyn TP if explosion strong & ADX strong
    hold_explosion = fusion["explosion"]["explosion"] and float(ind.get("adx",0))>=28
    if k < len(dyn_tps) and rr >= dyn_tps[k] and not hold_explosion:
        frac = dyn_fracs[k] if k < len(dyn_fracs) else 0.25
        close_partial(frac, f"TP_dyn@{dyn_tps[k]:.2f}%")
        STATE["profit_targets_achieved"] = k + 1

    # highest profit tracking + ratchet lock
    if rr > STATE["highest_profit_pct"]: STATE["highest_profit_pct"]=rr
    if STATE["highest_profit_pct"]>=TRAIL_ACTIVATE_PCT and rr < STATE["highest_profit_pct"]*RATCHET_LOCK_FALLBACK:
        close_partial(0.50, f"RatchetLock {STATE['highest_profit_pct']:.2f}%→{rr:.2f}%")

    # ATR Trailing after activation
    atr=float(ind.get("atr") or 0.0)
    if rr >= TRAIL_ACTIVATE_PCT and atr>0:
        gap = atr * ATR_TRAIL_MULT
        if side=="long":
            new_trail = px - gap
            STATE["trail"] = max(STATE["trail"] or new_trail, new_trail)
            if STATE["breakeven"] is not None: STATE["trail"] = max(STATE["trail"], STATE["breakeven"])
            if px < STATE["trail"]: close_market_strict(f"TRAIL_ATR({ATR_TRAIL_MULT}x)")
        else:
            new_trail = px + gap
            STATE["trail"] = min(STATE["trail"] or new_trail, new_trail)
            if STATE["breakeven"] is not None: STATE["trail"] = min(STATE["trail"], STATE["breakeven"])
            if px > STATE["trail"]: close_market_strict(f"TRAIL_ATR({ATR_TRAIL_MULT}x)")

    # >>> EVX GUARD — PM (IN-TRADE) — START
    evx = fusion.get("evx", {}) if isinstance(fusion, dict) else {}
    EVX_COOL_K       = float(os.getenv("EVX_COOL_K", "0.55"))
    EVX_SCALE_IN_CAP = float(os.getenv("EVX_SCALE_IN_CAP", "0.50"))

    try:
        if STATE.get("open") and evx.get("evx"):
            cap_qty = STATE["qty"] * EVX_SCALE_IN_CAP
            add_qty = safe_qty(min(cap_qty, STATE["qty"] * 0.25))
            if add_qty > 0:
                if STATE["side"] == "long" and evx.get("dir", 0) > 0:
                    if MODE_LIVE:
                        ex.create_order(SYMBOL, "market", "buy", add_qty, None, _params_open("buy"))
                    STATE["qty"] = safe_qty(STATE["qty"] + add_qty)
                    print(colored(f"➕ SCALE-IN(EVX) add={fmt(add_qty,4)} long", "green"))
                elif STATE["side"] == "short" and evx.get("dir", 0) < 0:
                    if MODE_LIVE:
                        ex.create_order(SYMBOL, "market", "sell", add_qty, None, _params_open("sell"))
                    STATE["qty"] = safe_qty(STATE["qty"] + add_qty)
                    print(colored(f"➕ SCALE-IN(EVX) add={fmt(add_qty,4)} short", "red"))
    except Exception:
        pass

    STATE.setdefault("_evx_peak", 0.0)
    cur_react = float(evx.get("react") or 0.0)
    if cur_react > STATE["_evx_peak"]:
        STATE["_evx_peak"] = cur_react

    cooled = (STATE["_evx_peak"] > 0 and cur_react <= STATE["_evx_peak"] * EVX_COOL_K)
    if STATE.get("open") and evx.get("evx") and cooled:
        close_market_strict("EVX_COOLDOWN_FULL_TAKE")
        return

    try:
        if STATE.get("open") and evx.get("evx"):
            opposite = ((STATE["side"] == "long" and evx.get("dir", 0) < 0) or
                        (STATE["side"] == "short" and evx.get("dir", 0) > 0))
            if opposite:
                close_partial(0.40, "EVX opposite burst")
                if atr > 0 and px:
                    gap2 = atr * max(ATR_TRAIL_MULT, 1.6)
                    if STATE["side"] == "long":
                        STATE["trail"] = max(STATE.get("trail") or (px - gap2), px - gap2)
                    else:
                        STATE["trail"] = min(STATE.get("trail") or (px + gap2), px + gap2)
    except Exception:
        pass
    # >>> EVX GUARD — PM (IN-TRADE) — END

# =================== LOG / SNAPSHOT ===================
def pretty_snapshot(bal, info, ind, spread_bps, fusion, reason=None, df=None):
    left_s = time_to_candle_close(df) if df is not None else 0
    trap_flag = "🪤" if fusion.get("trap") else "—"
    boom_flag = "💥" if fusion.get("explosion",{}).get("explosion") else "—"
    cpat = fusion.get("candle",{}).get("pattern","NONE")
    smc = fusion.get("smc",{})

    print(colored("─"*110,"cyan"))
    print(colored(f"📊 {SYMBOL} {INTERVAL} • {'LIVE' if MODE_LIVE else 'PAPER'} • {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC","cyan"))
    print(colored("─"*110,"cyan"))
    print("📈 RF & INDICATORS")
    print(f"   💲 Price {fmt(info.get('price'))} | RF filt={fmt(info.get('filter'))}  hi={fmt(info.get('hi'))} lo={fmt(info.get('lo'))} | spread={fmt(spread_bps,2)} bps")
    print(f"   🧮 RSI={fmt(ind.get('rsi'))}  +DI={fmt(ind.get('plus_di'))}  -DI={fmt(ind.get('minus_di'))}  ADX={fmt(ind.get('adx'))}  ATR={fmt(ind.get('atr'))}")
    print(f"   🧠 Fusion: score={fusion.get('fusion_score',0):.2f}  trap_risk={fusion.get('trap_risk',0):.2f} {trap_flag}  boom={boom_flag} ratio={fmt(fusion.get('explosion',{}).get('ratio'),2)}  candle={cpat}")
    print(f"   🏗️ SMC: EQH={fmt(smc.get('eqh'))}  EQL={fmt(smc.get('eql'))}  OB={smc.get('ob')}  FVG={smc.get('fvg')}")

    # >>> EVX GUARD — SNAPSHOT — START
    ev = fusion.get("evx", {}) if isinstance(fusion, dict) else {}
    print(f"   💣 EVX evx={ev.get('evx')} dir={'UP' if ev.get('dir',0)>0 else 'DOWN' if ev.get('dir',0)<0 else '—'} "
          f"v×={fmt(ev.get('v_ratio'),2)} react={fmt(ev.get('react'),2)} z={fmt(ev.get('z'),2)}")
    # >>> EVX GUARD — SNAPSHOT — END

    print(f"   ⏱️ closes_in ≈ {left_s}s")

    print("\n🧭 POSITION")
    bal_line = f"Balance={fmt(bal,2)}  Risk={int(RISK_ALLOC*100)}%×{LEVERAGE}x  CompoundPnL={fmt(compound_pnl)}  Eq~{fmt((bal or 0)+compound_pnl,2)}"
    print(colored(f"   {bal_line}", "yellow"))
    if STATE["open"]:
        lamp='🟩 LONG' if STATE['side']=='long' else '🟥 SHORT'
        print(f"   {lamp}  Entry={fmt(STATE['entry'])}  Qty={fmt(STATE['qty'],4)}  Bars={STATE['bars']}  Trail={fmt(STATE['trail'])}  BE={fmt(STATE['breakeven'])}")
        print(f"   🎯 TP_done={STATE['profit_targets_achieved']}  HP={fmt(STATE['highest_profit_pct'],2)}%  OppVotes={STATE.get('opp_votes',0)}")
    else:
        print("   ⚪ FLAT")
        if wait_for_next_signal_side:
            print(colored(f"   ⏳ Waiting RF opposite: {wait_for_next_signal_side.upper()}", "cyan"))
    if reason: print(colored(f"   ℹ️ reason: {reason}", "white"))
    print(colored("─"*110,"cyan"))

# =================== LOOP ===================
def trade_loop():
    global wait_for_next_signal_side
    loop_i=0
    while True:
        try:
            bal = balance_usdt()
            px  = price_now()
            df  = fetch_ohlcv()

            info = rf_signal_live(df)             # ⚡ RF LIVE ONLY
            ind  = compute_indicators(df)
            spread_bps = orderbook_spread_bps()

            # SMC levels
            df_closed = df.iloc[:-1] if len(df)>=2 else df.copy()
            smc = detect_smc_levels(df_closed)

            # Fusion orchestration
            fusion = fusion_orchestrator(df, ind, {"price": px or info["price"], **info}, smc)
            STATE["fusion_score"]=fusion["fusion_score"]
            STATE["trap_risk"]=fusion["trap_risk"]

            # PnL snapshot
            if STATE["open"] and px:
                STATE["pnl"] = (px-STATE["entry"])*STATE["qty"] if STATE["side"]=="long" else (STATE["entry"]-px)*STATE["qty"]

            # Manage after entry
            manage_after_entry(df, ind, {"price": px or info["price"], **info}, fusion)

            # Opposite RF defense
            if STATE["open"]:
                if STATE["side"]=="long" and info["short"]:
                    defensive_on_opposite_rf(ind, {"price": px or info["price"], **info})
                elif STATE["side"]=="short" and info["long"]:
                    defensive_on_opposite_rf(ind, {"price": px or info["price"], **info})

            reason=None
            if spread_bps is not None and spread_bps > MAX_SPREAD_BPS:
                reason=f"spread too high ({fmt(spread_bps,2)}bps > {MAX_SPREAD_BPS})"

            # ENTRY: RF LIVE ONLY
            sig = "buy" if (ENTRY_RF_ONLY and info["long"]) else ("sell" if (ENTRY_RF_ONLY and info["short"]) else None)

            # >>> EVX GUARD — ENTRY FILTER — START
            EVX_BLOCK_CHOP = str(os.getenv("EVX_BLOCK_CHOP", "true")).lower() == "true"
            EVX_MIN_ADX    = float(os.getenv("EVX_MIN_ADX", "22"))
            evx = fusion.get("evx", {}) if isinstance(fusion, dict) else {}
            adx = float(ind.get("adx") or 0.0)

            if sig and EVX_BLOCK_CHOP and (not evx.get("evx")) and (adx < EVX_MIN_ADX):
                reason = (reason or "") + " | EVX: no explosion & ADX weak"
                sig = None

            if sig == "buy" and evx.get("evx") and evx.get("dir") < 0:
                reason = (reason or "") + " | EVX: bearish burst vs long"
                sig = None
            elif sig == "sell" and evx.get("evx") and evx.get("dir") > 0:
                reason = (reason or "") + " | EVX: bullish burst vs short"
                sig = None
            # >>> EVX GUARD — ENTRY FILTER — END

            # After a close: wait for opposite RF side
            if not STATE["open"] and sig and reason is None:
                if wait_for_next_signal_side and sig != wait_for_next_signal_side:
                    reason=f"waiting opposite RF: need {wait_for_next_signal_side.upper()}"
                else:
                    qty = compute_size(bal, px or info["price"])
                    if qty>0:
                        ok = open_market(sig, qty, px or info["price"])
                        if ok:
                            wait_for_next_signal_side = None
                    else:
                        reason="qty<=0"

            pretty_snapshot(bal, {"price": px or info["price"], **info}, ind, spread_bps, fusion, reason, df)

            # bar counter
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
    return f"✅ RF-LIVE FUSION PRO — {SYMBOL} {INTERVAL} — {mode} — Entry: RF LIVE only — Fusion Orchestrator — Dynamic TP — Strict Close — FinalChunk={FINAL_CHUNK_QTY}DOGE — EVX Guard"

@app.route("/metrics")
def metrics():
    return jsonify({
        "symbol": SYMBOL, "interval": INTERVAL, "mode": "live" if MODE_LIVE else "paper",
        "leverage": LEVERAGE, "risk_alloc": RISK_ALLOC, "price": price_now(),
        "state": STATE, "compound_pnl": compound_pnl,
        "entry_mode": "RF_LIVE_ONLY", "wait_for_next_signal": wait_for_next_signal_side,
        "guards": {"max_spread_bps": MAX_SPREAD_BPS, "final_chunk_qty": FINAL_CHUNK_QTY},
        "fusion": {"score": STATE.get("fusion_score"), "trap_risk": STATE.get("trap_risk")},
        "evx": True
    })

@app.route("/health")
def health():
    return jsonify({
        "ok": True, "mode": "live" if MODE_LIVE else "paper",
        "open": STATE["open"], "side": STATE["side"], "qty": STATE["qty"],
        "compound_pnl": compound_pnl, "timestamp": datetime.utcnow().isoformat(),
        "entry_mode": "RF_LIVE_ONLY", "wait_for_next_signal": wait_for_next_signal_side,
        "fusion": {"score": STATE.get("fusion_score"), "trap_risk": STATE.get("trap_risk")},
        "tp_done": STATE.get("profit_targets_achieved", 0), "opp_votes": STATE.get("opp_votes",0)
    }), 200

def keepalive_loop():
    url=(SELF_URL or "").strip().rstrip("/")
    if not url:
        print(colored("⛔ keepalive disabled (SELF_URL not set)", "yellow"))
        return
    import requests
    sess=requests.Session(); sess.headers.update({"User-Agent":"rf-live-fusion/keepalive"})
    print(colored(f"KEEPALIVE every 50s → {url}", "cyan"))
    while True:
        try: sess.get(url, timeout=8)
        except Exception: pass
        time.sleep(50)

# =================== BOOT ===================
if __name__ == "__main__":
    print(colored(f"MODE: {'LIVE' if MODE_LIVE else 'PAPER'}  •  {SYMBOL}  •  {INTERVAL}", "yellow"))
    print(colored(f"RISK: {int(RISK_ALLOC*100)}% × {LEVERAGE}x  •  RF_LIVE={RF_LIVE_ONLY}", "yellow"))
    print(colored(f"ENTRY: RF ONLY  •  FINAL_CHUNK_QTY={FINAL_CHUNK_QTY}", "yellow"))
    logging.info("service starting…")
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    signal.signal(signal.SIGINT,  lambda *_: sys.exit(0))
    import threading
    threading.Thread(target=trade_loop, daemon=True).start()
    threading.Thread(target=keepalive_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)
