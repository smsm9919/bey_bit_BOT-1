# -*- coding: utf-8 -*-
"""
BYBIT SOL Bot — RF+Council+EVX — Compounding ON
Author: you
Notes:
- API keys ONLY from ENV (BYBIT_API_KEY, BYBIT_API_SECRET)
- All strategy params are hard-coded here (as requested).
- Compounding (use balance + compound_pnl) is ENABLED.
"""

import os, time, math, random, signal, sys, traceback, logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from decimal import Decimal, ROUND_DOWN, InvalidOperation

import pandas as pd
from flask import Flask, jsonify

try:
    import ccxt
except Exception:
    raise RuntimeError("ccxt is required. pip install ccxt flask pandas")

try:
    from termcolor import colored
except Exception:
    def colored(t,*a,**k): return t

# =================== EXCHANGE / ENV ===================
BYBIT_API_KEY    = os.getenv("BYBIT_API_KEY", "")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET", "")
MODE_LIVE = bool(BYBIT_API_KEY and BYBIT_API_SECRET)

SELF_URL = os.getenv("SELF_URL", "") or os.getenv("RENDER_EXTERNAL_URL", "")
PORT     = int(os.getenv("PORT", 5000))

# =================== STATIC CONFIG (as requested) ===================
EXCHANGE_NAME = "bybit"
SYMBOL   = "SOL/USDT:USDT"
INTERVAL = "15m"

# Risk & leverage
LEVERAGE   = 10
RISK_ALLOC = 0.60   # 60% of equity used for position notionals (× leverage)

# Range Filter (TradingView-like) — LIVE candle only (no close requirement)
RF_SOURCE     = "close"
RF_PERIOD     = 20
RF_MULT       = 3.5
RF_HYST_BPS   = 6.0          # min hysteresis to accept live cross
TV_CLOSED_ONLY = False        # keep LIVE by default

# Indicators
RSI_LEN = 14
ADX_LEN = 14
ATR_LEN = 14

# EVX (Explosion/Implosion Filter)
EVX_VOL_WIN     = 21
EVX_VOL_RATIO   = 1.8   # volume spike vs SMA
EVX_BODY_ATR    = 1.2   # real body vs ATR
EVX_COOL_WIN    = 5     # if market cools (no more EVX) we take “one-shot” profit

# Spread guard
SPREAD_GUARD_BPS = 6.0

# Smart exits + patience
WAIT_CONFIRM_AFTER_DOJI   = True
CONFIRM_BAR_MAX_SEC       = 9*60   # wait up to 9min (for 15m TF) to see post-doji intent (live logic)
RETEST_WAIT_BARS_MAX      = 4      # allow waiting retest after breakout

# Profit management — “one-shot” after cool-off
TP1_PCT_BASE     = 0.40
TP1_CLOSE_FRAC   = 0.50
BREAKEVEN_AFTER  = 0.30
TRAIL_ACTIVATE_PCT = 1.20
ATR_TRAIL_MULT     = 1.6

# Dust/final-chunk strict close (Bybit lot min varies; we guard here)
FINAL_CHUNK_QTY = 0.2

# Council voting weights (0..1)
W_RF      = 0.40  # entry authority
W_CANDLE  = 0.15
W_SMC     = 0.20
W_EVX     = 0.15
W_MOM     = 0.10  # RSI/ADX/DI
W_TRAP    = 0.10  # stop-hunt/trap flag reduces aggression

VOTE_THRESHOLD_OPEN = 0.65
VOTE_THRESHOLD_HOLD = 0.55

# Pacing
BASE_SLEEP   = 5
NEAR_CLOSE_S = 1

# Position mode
POSITION_MODE = "oneway"

# =================== LOGGING ===================
def setup_file_logging():
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not any(isinstance(h, RotatingFileHandler) and getattr(h, "baseFilename", "").endswith("bot.log")
               for h in root.handlers):
        fh = RotatingFileHandler("bot.log", maxBytes=5_000_000, backupCount=7, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
        root.addHandler(fh)
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    print(colored("🗂️ log rotation ready", "cyan"))
setup_file_logging()

# =================== EXCHANGE INIT ===================
def make_ex():
    opts = {
        "apiKey": BYBIT_API_KEY,
        "secret": BYBIT_API_SECRET,
        "enableRateLimit": True,
        "timeout": 20000,
        "options": {"defaultType":"swap"}
    }
    return ccxt.bybit(opts)

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
            ex.set_leverage(LEVERAGE, SYMBOL, params={"side":"BOTH"})
        except Exception as e:
            print(colored(f"set_leverage warn (bybit): {e}", "yellow"))
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
    except Exception:
        return None

def balance_usdt():
    if not MODE_LIVE: return 100.0
    try:
        b = with_retry(lambda: ex.fetch_balance(params={"type":"swap"}))
        return b.get("total",{}).get("USDT") or b.get("free",{}).get("USDT")
    except Exception:
        return None

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

def fmt(v, d=6, na="—"):
    try:
        if v is None or (isinstance(v,float) and (math.isnan(v) or math.isinf(v))): return na
        return f"{float(v):.{d}f}"
    except Exception:
        return na

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
    if q<=0: print(colored(f"⚠️ qty invalid→ {q}", "yellow"))
    return q

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
    if len(df) < max(ATR_LEN, RSI_LEN, ADX_LEN) + 3:
        return {"rsi":None,"plus_di":None,"minus_di":None,"dx":None,"adx":None,"atr":None}
    d = df.copy()
    c,h,l = d["close"].astype(float), d["high"].astype(float), d["low"].astype(float)
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

    i=len(d)-1
    return {
        "rsi": float(rsi.iloc[i]), "plus_di": float(plus_di.iloc[i]),
        "minus_di": float(minus_di.iloc[i]), "dx": float(dx.iloc[i]),
        "adx": float(adx.iloc[i]), "atr": float(atr.iloc[i])
    }

# =================== RANGE FILTER ===================
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
    if len(df) < RF_PERIOD + 3:
        i=-1
        price = float(df["close"].iloc[i]) if len(df) else 0.0
        return {"time": int(df["time"].iloc[i]) if len(df) else int(time.time()*1000),
                "price": price, "long": False, "short": False,
                "filter": price, "hi": price, "lo": price}
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

# =================== CANDLE SYSTEM ===================
def detect_candle(df: pd.DataFrame):
    if len(df)<3:
        return {"pattern":"NONE","strength":0,"dir":0,"doji":False}
    o=float(df["open"].iloc[-1]); h=float(df["high"].iloc[-1])
    l=float(df["low"].iloc[-1]);  c=float(df["close"].iloc[-1])
    rng=max(h-l,1e-12); body=abs(c-o)
    upper=h-max(o,c); lower=min(o,c)-l
    upper_pct=upper/rng*100.0; lower_pct=lower/rng*100.0; body_pct=body/rng*100.0

    doji = body_pct<=10
    if doji: return {"pattern":"DOJI","strength":1,"dir":0,"doji":True}
    if body_pct>=85 and upper_pct<=7 and lower_pct<=7:
        return {"pattern":"MARUBOZU","strength":3,"dir":(1 if c>o else -1),"doji":False}
    if lower_pct>=60 and body_pct<=30 and c>o:
        return {"pattern":"HAMMER","strength":2,"dir":1,"doji":False}
    if upper_pct>=60 and body_pct<=30 and c<o:
        return {"pattern":"SHOOTING","strength":2,"dir":-1,"doji":False}
    return {"pattern":"NORMAL","strength":1,"dir":(1 if c>o else -1),"doji":False}

# =================== SMC: swings/EQ/OB/FVG/CHoCH ===================
def _find_swings(df: pd.DataFrame, left:int=2, right:int=2):
    if len(df) < left+right+3: return None, None
    h = df["high"].astype(float).values
    l = df["low"].astype(float).values
    ph = [None]*len(df); pl = [None]*len(df)
    for i in range(left, len(df)-right):
        if all(h[i] >= h[j] for j in range(i-left, i+right+1)): ph[i] = h[i]
        if all(l[i] <= l[j] for j in range(i-left, i+right+1)): pl[i] = l[i]
    return ph, pl

def detect_smc_levels(df: pd.DataFrame):
    try:
        d = df.copy()
        ph, pl = _find_swings(d, 2, 2)

        def _eq_levels(vals, is_high=True):
            res = []
            tol_pct = 0.05
            for i, price in enumerate(vals):
                if price is None: continue
                tol = price * tol_pct / 100.0
                neighbors = [vals[j] for j in range(max(0,i-10), min(len(vals),i+10))
                             if vals[j] is not None and abs(vals[j]-price)<=tol]
                if len(neighbors) >= 2:
                    res.append(max(neighbors) if is_high else min(neighbors))
            if not res: return None
            return max(res) if is_high else min(res)
        eqh = _eq_levels(ph, True)
        eql = _eq_levels(pl, False)

        # Order block (simple)
        ob=None
        for i in range(len(d)-2, max(len(d)-40, 1), -1):
            o=float(d["open"].iloc[i]); c=float(d["close"].iloc[i])
            h=float(d["high"].iloc[i]); l=float(d["low"].iloc[i])
            rng=max(h-l,1e-12); body=abs(c-o)
            upper=h-max(o,c); lower=min(o,c)-l
            if body>=0.6*rng and (upper/rng)<=0.2 and (lower/rng)<=0.2:
                side="bull" if c>o else "bear"
                ob={"side":side,"bot":min(o,c),"top":max(o,c),"time":int(d["time"].iloc[i])}
                break

        # FVG (last 20 bars)
        fvg=None
        for i in range(len(d)-3, max(len(d)-20,2), -1):
            prev_high=float(d["high"].iloc[i-1]); prev_low=float(d["low"].iloc[i-1])
            curr_low=float(d["low"].iloc[i]);   curr_high=float(d["high"].iloc[i])
            if curr_low > prev_high:
                fvg={"type":"BULL_FVG","bottom":prev_high,"top":curr_low}; break
            if curr_high < prev_low:
                fvg={"type":"BEAR_FVG","bottom":curr_high,"top":prev_low}; break

        # Simple CHoCH/BoS
        choch=None
        if ph and pl:
            # last LL->HH or LH->HL toggle
            try:
                last_h = [x for x in ph if x is not None][-3:]
                last_l = [x for x in pl if x is not None][-3:]
                if len(last_h)>=2 and len(last_l)>=2:
                    if last_l[-1] > last_l[-2] and last_h[-1] > last_h[-2]:
                        choch="BULL"
                    elif last_l[-1] < last_l[-2] and last_h[-1] < last_h[-2]:
                        choch="BEAR"
            except Exception:
                pass

        return {"eqh":eqh,"eql":eql,"ob":ob,"fvg":fvg,"choch":choch}
    except Exception:
        return {"eqh":None,"eql":None,"ob":None,"fvg":None,"choch":None}

def _near_level(px, lvl, bps):
    try: return abs((px-lvl)/lvl)*10000.0 <= bps
    except Exception: return False

def detect_stop_hunt(df: pd.DataFrame, smc: dict):
    if len(df) < 3: return None
    try:
        o=float(df["open"].iloc[-1]); h=float(df["high"].iloc[-1])
        l=float(df["low"].iloc[-1]);  c=float(df["close"].iloc[-1])
        rng=max(h-l,1e-12); body=abs(c-o)
        upper=h-max(o,c); lower=min(o,c)-l
        upper_pct=upper/rng*100.0; lower_pct=lower/rng*100.0; body_pct=body/rng*100.0
        v=float(df["volume"].iloc[-1]); vma=df["volume"].iloc[-EVX_VOL_WIN:-1].astype(float).mean() if len(df)>=EVX_VOL_WIN+1 else 0.0
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

# =================== EVX (Explosion/Implosion) ===================
def evx_signal(df: pd.DataFrame, ind: dict):
    if len(df) < EVX_VOL_WIN+3: 
        return {"ok":False,"dir":0,"ratio":0.0}
    try:
        v=float(df["volume"].iloc[-1]); vma=df["volume"].iloc[-EVX_VOL_WIN:-1].astype(float).mean() or 1e-9
        o=float(df["open"].iloc[-1]); c=float(df["close"].iloc[-1]); body=abs(c-o)
        atr=float(ind.get("atr") or 0.0)
        ratio=v/vma
        strong = (ratio>=EVX_VOL_RATIO and atr>0 and (body/atr)>=EVX_BODY_ATR)
        return {"ok":bool(strong), "dir": (1 if c>o else -1), "ratio": float(ratio)}
    except Exception:
        return {"ok":False,"dir":0,"ratio":0.0}

# =================== COUNCIL (Voting) ===================
def council_vote(df, info, ind, smc, evx, candle):
    px = info.get("price"); rf_dir = 1 if info.get("long") else (-1 if info.get("short") else 0)
    rsi=float(ind.get("rsi") or 50.0); adx=float(ind.get("adx") or 20.0)
    pdi=float(ind.get("plus_di") or 0.0); mdi=float(ind.get("minus_di") or 0.0)
    mom_dir = 1 if pdi>mdi else (-1 if mdi>pdi else 0)

    smc_bias=0.0
    if smc.get("choch")=="BULL": smc_bias += 0.5
    if smc.get("choch")=="BEAR": smc_bias -= 0.5
    if smc.get("ob"):
        smc_bias += 0.2 if smc["ob"]["side"]=="bull" else -0.2

    trap = detect_stop_hunt(df, smc)
    trap_penalty = 0.25 if trap else 0.0

    evx_bias = 0.25 if evx["ok"] else 0.0
    evx_dir  = evx["dir"]

    candle_bias = 0.15 if candle["pattern"] in ("MARUBOZU","HAMMER","SHOOTING") else 0.05
    candle_dir  = candle["dir"]

    # normalize to [-1..1] then weighted to [0..1]
    score = 0.0
    if rf_dir != 0:
        score += W_RF * (0.5 + 0.5*rf_dir)
    if candle_dir != 0:
        score += W_CANDLE * (0.5 + 0.5*candle_dir)
    score += W_SMC * (0.5 + 0.5*(1 if smc_bias>0 else (-1 if smc_bias<0 else 0)))
    if evx_dir != 0:
        score += W_EVX * (0.5 + 0.5*evx_dir)
    score += W_MOM * (0.5 + 0.5*mom_dir)
    if trap_penalty>0: score -= W_TRAP * 0.5

    score = max(0.0, min(1.0, score))
    return {"score":score, "trap":bool(trap), "evx_hold":evx["ok"], "dir_hint": rf_dir or evx_dir or mom_dir}

# =================== APEX CHECK ===================
def apex_confirmed(side: str, df: pd.DataFrame, ind: dict, smc: dict):
    try:
        adx=float(ind.get("adx") or 0.0); rsi=float(ind.get("rsi") or 50.0)
        o=float(df["open"].iloc[-1]); h=float(df["high"].iloc[-1])
        l=float(df["low"].iloc[-1]);  c=float(df["close"].iloc[-1])
        rng=max(h-l,1e-12); upper=h-max(o,c); lower=min(o,c)-l
        near_top = (smc.get("eqh") and _near_level(h, smc["eqh"], 10.0)) or (smc.get("ob") and smc["ob"].get("side")=="bear" and _near_level(h, smc["ob"]["bot"], 10.0))
        near_bot = (smc.get("eql") and _near_level(l, smc["eql"], 10.0)) or (smc.get("ob") and smc["ob"].get("side")=="bull" and _near_level(l, smc["ob"]["top"], 10.0))
        reject_top = (upper/rng>=0.55 and (adx<20 or 45<=rsi<=55))
        reject_bot = (lower/rng>=0.55 and (adx<20 or 45<=rsi<=55))
        if side=="long"  and near_top and reject_top: return True
        if side=="short" and near_bot and reject_bot: return True
    except Exception:
        pass
    return False

# =================== STATE ===================
STATE = {
    "open": False, "side": None, "entry": None, "qty": 0.0,
    "pnl": 0.0, "bars": 0, "trail": None, "breakeven": None,
    "tp1_done": False, "highest_profit_pct": 0.0, "profit_targets_achieved": 0,
    "opp_votes": 0, "evx_cool": 0
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
            qty = abs(float(p.get("contracts") or p.get("info",{}).get("size") or 0))
            if qty <= 0: return 0.0, None, None
            entry = float(p.get("entryPrice") or p.get("info",{}).get("avgPrice") or 0)
            side_raw = (p.get("side") or p.get("info",{}).get("positionSide") or "").lower()
            side = "long" if ("long" in side_raw or float(p.get("cost",0))>0) else "short"
            return qty, side, entry
    except Exception as e:
        logging.error(f"_read_position error: {e}")
    return 0.0, None, None

def compute_size(balance, price):
    # Compounding: balance + compound_pnl
    effective = (balance or 0.0) + float(compound_pnl or 0.0)
    capital = effective * RISK_ALLOC * LEVERAGE
    raw = max(0.0, capital / max(float(price or 0.0), 1e-9))
    return safe_qty(raw)

def open_market(side, qty, price):
    if qty<=0:
        print(colored("❌ skip open (qty<=0)", "red")); return False
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
        "opp_votes": 0, "evx_cool": 0
    })
    print(colored(f"🚀 OPEN {('🟩 LONG' if side=='buy' else '🟥 SHORT')} qty={fmt(qty,4)} @ {fmt(price)}", "green" if side=="buy" else "red"))
    logging.info(f"OPEN {side} qty={qty} price={price}")
    return True

def _finalize_close(pnl_reason):
    global compound_pnl, wait_for_next_signal_side
    px = price_now() or STATE.get("entry")
    entry_px = STATE.get("entry") or px
    side = STATE.get("side") or "long"
    qty  = STATE.get("qty") or 0.0
    pnl  = (px - entry_px) * qty * (1 if side=="long" else -1)
    compound_pnl += pnl
    print(colored(f"🔚 STRICT CLOSE {side} reason={pnl_reason} pnl={fmt(pnl)} total={fmt(compound_pnl)}","magenta"))
    logging.info(f"STRICT_CLOSE {side} pnl={pnl} total={compound_pnl}")
    prev_side = side
    STATE.update({
        "open": False, "side": None, "entry": None, "qty": 0.0,
        "pnl": 0.0, "bars": 0, "trail": None, "breakeven": None,
        "tp1_done": False, "highest_profit_pct": 0.0, "profit_targets_achieved": 0,
        "opp_votes": 0, "evx_cool": 0
    })
    wait_for_next_signal_side = "sell" if prev_side=="long" else "buy"
    logging.info(f"AFTER_CLOSE waiting_for={wait_for_next_signal_side}")

def close_market_strict(reason="STRICT"):
    exch_qty, exch_side, _ = _read_position()
    if exch_qty <= 0:
        if STATE.get("open"): _finalize_close(reason)
        return
    side_to_close = "sell" if (exch_side=="long") else "buy"
    qty_to_close  = safe_qty(exch_qty)
    try:
        if MODE_LIVE:
            params = _params_close(); params["reduceOnly"]=True
            ex.create_order(SYMBOL,"market",side_to_close,qty_to_close,None,params)
        time.sleep(1.5)
    except Exception as e:
        logging.error(f"strict close error: {e}")
    _finalize_close(reason)

def close_partial(frac, reason):
    if not STATE["open"] or STATE["qty"]<=0: return
    qty_close = safe_qty(max(0.0, STATE["qty"] * min(max(frac,0.0),1.0)))
    px = price_now() or STATE["entry"]
    min_unit = max(FINAL_CHUNK_QTY, LOT_MIN or FINAL_CHUNK_QTY)
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
        print(colored(f"🧹 Final chunk ≤ {FINAL_CHUNK_QTY} → strict close", "yellow"))
        close_market_strict("FINAL_CHUNK_RULE")

# =================== MANAGEMENT ===================
def manage_after_entry(df, ind, info, council, smc):
    if not STATE["open"] or STATE["qty"]<=0: return
    px = info["price"]; entry=STATE["entry"]; side=STATE["side"]
    rr = (px - entry)/entry*100*(1 if side=="long" else -1)

    # patience after doji/hammer/shooting: don't exit in panic
    # (we just avoid taking partials if doji seen, wait for confirm)
    # EVX cool-off: once EVX true then cools for N bars => single-shot
    if council["evx_hold"]:
        STATE["evx_cool"]=0
    else:
        STATE["evx_cool"]=min(EVX_COOL_WIN, STATE.get("evx_cool",0)+1)
        if STATE["evx_cool"]>=EVX_COOL_WIN and rr>0.2:
            close_market_strict("EVX_COOLOFF_ONE_SHOT"); return

    # TP1 baseline (adaptive by ADX)
    adx=float(ind.get("adx") or 0.0)
    tp1_now = TP1_PCT_BASE*(2.2 if adx>=35 else 1.8 if adx>=28 else 1.0)
    if (not STATE["tp1_done"]) and rr >= tp1_now:
        close_partial(TP1_CLOSE_FRAC, f"TP1@{tp1_now:.2f}%")
        STATE["tp1_done"]=True
        if rr >= BREAKEVEN_AFTER: STATE["breakeven"]=entry

    # Apex single-shot (save fees)
    if rr>=max(0.4, TP1_PCT_BASE*0.8) and apex_confirmed(side, df, ind, smc):
        close_market_strict("APEX_CONFIRMED"); return

    # Trail after activation
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

# =================== LOG ===================
def pretty_snapshot(bal, info, ind, spread_bps, council, smc, evx, candle, reason=None, df=None):
    left_s = time_to_candle_close(df) if df is not None else 0
    trap_flag = "🪤" if council.get("trap") else "—"
    boom_flag = "💥" if evx.get("ok") else "—"
    cpat = candle.get("pattern","NONE")
    eq = (bal or 0.0) + float(compound_pnl or 0.0)

    print(colored("─"*112,"cyan"))
    print(colored(f"📊 {SYMBOL} {INTERVAL} • {'LIVE' if MODE_LIVE else 'PAPER'} • {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC","cyan"))
    print(colored("─"*112,"cyan"))
    print("📈 RF & INDICATORS")
    print(f"   💲 Price {fmt(info.get('price'))} | RF filt={fmt(info.get('filter'))}  hi={fmt(info.get('hi'))} lo={fmt(info.get('lo'))} | spread={fmt(spread_bps,2)} bps")
    print(f"   🧮 RSI={fmt(ind.get('rsi'))}  +DI={fmt(ind.get('plus_di'))}  -DI={fmt(ind.get('minus_di'))}  ADX={fmt(ind.get('adx'))}  ATR={fmt(ind.get('atr'))}")
    print(f"   🧨 EVX ok={evx.get('ok')} dir={evx.get('dir')} ratio={fmt(evx.get('ratio'),2)}  candle={cpat}")
    print(f"   🧠 Council score={council.get('score',0):.2f} trap={trap_flag} boom={boom_flag}  ⏱ closes_in ≈ {left_s}s")
    smc_str = f"EQH={fmt(smc.get('eqh'))} EQL={fmt(smc.get('eql'))} OB={smc.get('ob')} FVG={smc.get('fvg')} CHoCH={smc.get('choch')}"
    print(f"   🏗️ SMC: {smc_str}")

    print("\n🧭 POSITION")
    bal_line = f"Balance={fmt(bal,2)}  CompoundPnL={fmt(compound_pnl,2)}  Eq~{fmt(eq,2)}  Risk={int(RISK_ALLOC*100)}%×{LEVERAGE}x"
    print(colored(f"   {bal_line}", "yellow"))
    if STATE["open"]:
        lamp='🟩 LONG' if STATE['side']=='long' else '🟥 SHORT'
        print(f"   {lamp}  Entry={fmt(STATE['entry'])}  Qty={fmt(STATE['qty'],4)}  Bars={STATE['bars']}  Trail={fmt(STATE['trail'])}  BE={fmt(STATE['breakeven'])}")
        print(f"   🎯 TP_done={STATE['profit_targets_achieved']}  HP={fmt(STATE['highest_profit_pct'],2)}%  OppVotes={STATE.get('opp_votes',0)} EVXCool={STATE.get('evx_cool',0)}")
    else:
        print("   ⚪ FLAT")
        if wait_for_next_signal_side:
            print(colored(f"   ⏳ Waiting RF opposite: {wait_for_next_signal_side.upper()}", "cyan"))
    if reason: print(colored(f"   ℹ️ reason: {reason}", "white"))
    print(colored("─"*112,"cyan"))

# =================== LOOP ===================
def trade_loop():
    global wait_for_next_signal_side
    while True:
        try:
            bal = balance_usdt()
            px  = price_now()
            df  = fetch_ohlcv()

            info = rf_signal_live(df)
            ind  = compute_indicators(df)
            spread_bps = orderbook_spread_bps()

            df_closed = df.iloc[:-1] if len(df)>=2 else df.copy()
            smc = detect_smc_levels(df_closed)
            evx = evx_signal(df, ind)
            candle = detect_candle(df)

            council = council_vote(df, {"price": px or info["price"], **info}, ind, smc, evx, candle)

            # PnL snapshot
            if STATE["open"] and px:
                STATE["pnl"] = (px-STATE["entry"])*STATE["qty"] if STATE["side"]=="long" else (STATE["entry"]-px)*STATE["qty"]

            # manage
            manage_after_entry(df, ind, {"price": px or info["price"], **info}, council, smc)

            # ENTRY logic — RF LIVE ONLY + Council
            reason=None
            if spread_bps is not None and spread_bps > SPREAD_GUARD_BPS:
                reason=f"spread too high ({fmt(spread_bps,2)}bps > {SPREAD_GUARD_BPS})"

            sig = "buy" if info["long"] else ("sell" if info["short"] else None)

            if not STATE["open"] and sig and not reason:
                # wait for opposite signal after last close
                if wait_for_next_signal_side and sig != wait_for_next_signal_side:
                    reason=f"waiting opposite RF: need {wait_for_next_signal_side.upper()}"
                else:
                    # patience: if last candle is doji/hammer/shooting, require confirm
                    if WAIT_CONFIRM_AFTER_DOJI and candle.get("doji"):
                        reason="doji wait confirm"
                    else:
                        # council check
                        if council["score"]>=VOTE_THRESHOLD_OPEN:
                            qty = compute_size(bal, px or info["price"])
                            if qty>0 and sig in ("buy","sell"):
                                ok = open_market(sig, qty, px or info["price"])
                                if ok: wait_for_next_signal_side=None
                        else:
                            reason=f"council score low {council['score']:.2f}<{VOTE_THRESHOLD_OPEN}"

            # log
            pretty_snapshot(bal, {"price": px or info["price"], **info}, ind, spread_bps, council, smc, evx, candle, reason, df)

            # bar counter
            if len(df)>=2 and int(df["time"].iloc[-1])!=int(df["time"].iloc[-2]) and STATE["open"]:
                STATE["bars"] += 1
                # track highest profit%
                if px and STATE["entry"]:
                    rr = (px-STATE["entry"])/STATE["entry"]*100*(1 if STATE["side"]=="long" else -1)
                    if rr > STATE["highest_profit_pct"]: STATE["highest_profit_pct"]=rr

            # sleep
            sleep_s = NEAR_CLOSE_S if time_to_candle_close(df)<=10 else BASE_SLEEP
            time.sleep(sleep_s)
        except Exception as e:
            print(colored(f"❌ loop error: {e}\n{traceback.format_exc()}", "red"))
            logging.error(f"trade_loop error: {e}\n{traceback.format_exc()}")
            time.sleep(BASE_SLEEP)

# =================== API ===================
app = Flask(__name__)

@app.route("/")
def home():
    mode='LIVE' if MODE_LIVE else 'PAPER'
    return f"✅ BYBIT RF+EVX Council — {SYMBOL} {INTERVAL} — {mode} — Compounding=ON — StrictClose+OppRFWait"

@app.route("/metrics")
def metrics():
    bal = balance_usdt()
    eq  = (bal or 0.0) + float(compound_pnl or 0.0)
    return jsonify({
        "exchange": EXCHANGE_NAME, "symbol": SYMBOL, "interval": INTERVAL,
        "mode": "live" if MODE_LIVE else "paper",
        "leverage": LEVERAGE, "risk_alloc": RISK_ALLOC, "price": price_now(),
        "state": STATE, "compound_pnl": compound_pnl, "equity_virtual": eq,
        "guards": {"spread_bps": SPREAD_GUARD_BPS, "final_chunk_qty": FINAL_CHUNK_QTY},
        "council": {"vote_threshold_open": VOTE_THRESHOLD_OPEN, "vote_threshold_hold": VOTE_THRESHOLD_HOLD}
    })

@app.route("/health")
def health():
    return jsonify({
        "ok": True, "mode": "live" if MODE_LIVE else "paper",
        "open": STATE["open"], "side": STATE["side"], "qty": STATE["qty"],
        "compound_pnl": compound_pnl, "timestamp": datetime.utcnow().isoformat(),
        "tp_done": STATE.get("profit_targets_achieved", 0), "opp_votes": STATE.get("opp_votes",0)
    }), 200

def keepalive_loop():
    url=(SELF_URL or "").strip().rstrip("/")
    if not url:
        print(colored("⛔ keepalive disabled (SELF_URL not set)", "yellow"))
        return
    import requests
    sess=requests.Session(); sess.headers.update({"User-Agent":"bybit-rf-evx/keepalive"})
    print(colored(f"KEEPALIVE every 50s → {url}", "cyan"))
    while True:
        try: sess.get(url, timeout=8)
        except Exception: pass
        time.sleep(50)

# =================== BOOT ===================
if __name__ == "__main__":
    print(colored(f"MODE: {'LIVE' if MODE_LIVE else 'PAPER'}  •  {SYMBOL}  •  {INTERVAL}", "yellow"))
    print(colored(f"RISK: {int(RISK_ALLOC*100)}% × {LEVERAGE}x  •  RF_LIVE=True", "yellow"))
    print(colored(f"ENTRY: RF ONLY + EVX Council  •  FINAL_CHUNK_QTY={FINAL_CHUNK_QTY}", "yellow"))
    logging.info("service starting…")
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    signal.signal(signal.SIGINT,  lambda *_: sys.exit(0))
    # Threads
    import threading
    threading.Thread(target=trade_loop, daemon=True).start()
    threading.Thread(target=keepalive_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)
