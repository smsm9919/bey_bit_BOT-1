# -*- coding: utf-8 -*-
"""
BYBIT Futures Bot — RF + EVX + Smart Mgmt (TV-matched)
- Exchange: Bybit linear USDT Perp via CCXT
- TV-matching indicators: RMA(Wilder) for RSI/ADX/ATR, closed-bar option, source=close|hlc3
- Range Filter entries (live or closed)
- EVX (Explosion/Collapse) guard as entry filter + exit accelerator
- Smart patience: confirmation-on-edges, doji/chop tolerance, strict-close + residual guard
- Flask / /metrics /health
"""

import os, time, math, random, signal, sys, traceback, logging
from logging.handlers import RotatingFileHandler
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN, InvalidOperation
import pandas as pd
import ccxt
from flask import Flask, jsonify

try:
    from termcolor import colored
except Exception:
    def colored(t,*a,**k): return t

# =================== ENV ===================
EXCHANGE = os.getenv("EXCHANGE", "bybit").lower()
SYMBOL   = os.getenv("SYMBOL", "SOL/USDT:USDT")     # Bybit linear USDT perp
INTERVAL = os.getenv("INTERVAL", "15m")

# Risk & leverage
LEVERAGE   = int(os.getenv("LEVERAGE", 10))
RISK_ALLOC = float(os.getenv("RISK_ALLOC", 0.60))
POSITION_MODE = os.getenv("POSITION_MODE", "oneway")

# Range Filter (TV-like)
RF_SOURCE = os.getenv("RF_SOURCE", "close").lower()  # kept for backward compatibility
RF_PERIOD = int(os.getenv("RF_PERIOD", 20))
RF_MULT   = float(os.getenv("RF_MULT", 3.5))
RF_LIVE_ONLY = str(os.getenv("RF_LIVE_ONLY","true")).lower()=="true"
RF_HYST_BPS  = float(os.getenv("RF_HYST_BPS", 6.0))

# TV matching
TV_MATCH_MODE  = str(os.getenv("TV_MATCH_MODE","true")).lower()=="true"
USE_CLOSED_ONLY= str(os.getenv("USE_CLOSED_ONLY","false")).lower()=="true"
TV_SOURCE      = os.getenv("TV_SOURCE","close").lower()  # close|hlc3
PRICE_FEED     = os.getenv("PRICE_FEED","last").lower()  # last|mark
TZ             = os.getenv("TZ","UTC").upper()

# Indicators
RSI_LEN = int(os.getenv("RSI_LEN", 14))
ADX_LEN = int(os.getenv("ADX_LEN", 14))
ATR_LEN = int(os.getenv("ATR_LEN", 14))

# Guards
SPREAD_GUARD_BPS = float(os.getenv("SPREAD_GUARD_BPS", 6.0))
COOLDOWN_AFTER_CLOSE_BARS = int(os.getenv("COOLDOWN_AFTER_CLOSE_BARS", 0))

# Smart profit / trail
TP1_PCT           = float(os.getenv("TP1_PCT", 0.40))
TP1_CLOSE_FRAC    = float(os.getenv("TP1_CLOSE_FRAC", 0.50))
BREAKEVEN_AFTER   = float(os.getenv("BREAKEVEN_AFTER_PCT", 0.30))
TRAIL_ACTIVATE_PCT= float(os.getenv("TRAIL_ACTIVATE_PCT", 0.60))
ATR_MULT_TRAIL    = float(os.getenv("ATR_MULT_TRAIL", 1.6))

# EVX (explosion/implosion) filter
EVX_ARM            = str(os.getenv("EVX_ARM","true")).lower()=="true"
EVX_MIN_VOL_RATIO  = float(os.getenv("EVX_MIN_VOL_RATIO", 1.8))
EVX_MIN_ATR_REACT  = float(os.getenv("EVX_MIN_ATR_REACT", 1.2))
EVX_COOLDOWN_BARS  = int(os.getenv("EVX_COOLDOWN_BARS", 2))

# Residual/final-chunk strict close qty
FINAL_CHUNK_QTY = float(os.getenv("FINAL_CHUNK_QTY", 0.2))

# Pacing / Web
BASE_SLEEP   = int(os.getenv("DECISION_EVERY_S", 30))
SELF_URL     = os.getenv("SELF_URL", "") or os.getenv("RENDER_EXTERNAL_URL","")
PORT         = int(os.getenv("PORT", 5000))

# =================== LOGGING ===================
def setup_file_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not any(isinstance(h, RotatingFileHandler) and getattr(h, "baseFilename","").endswith("bot.log")
               for h in logger.handlers):
        fh = RotatingFileHandler("bot.log", maxBytes=5_000_000, backupCount=7, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
        logger.addHandler(fh)
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    print(colored("🗂️ log rotation ready","cyan"))

setup_file_logging()

# =================== EXCHANGE ===================
def make_ex():
    api_key = os.getenv(f"{EXCHANGE.upper()}_API_KEY","")
    api_secret = os.getenv(f"{EXCHANGE.upper()}_API_SECRET","")
    cls = getattr(ccxt, EXCHANGE)
    return cls({
        "apiKey": api_key,
        "secret": api_secret,
        "enableRateLimit": True,
        "timeout": 20000,
        "options": {"defaultType":"swap", "adjustForTimeDifference": True}
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
        AMT_PREC = int((MARKET.get("precision",{}) or {}).get("amount", 0) or 0)
        LOT_STEP = (MARKET.get("limits",{}) or {}).get("amount",{}).get("step", None)
        LOT_MIN  = (MARKET.get("limits",{}) or {}).get("amount",{}).get("min",  None)
        print(colored(f"🔧 precision={AMT_PREC}, step={LOT_STEP}, min={LOT_MIN}","cyan"))
    except Exception as e:
        print(colored(f"⚠️ load_market_specs: {e}","yellow"))

def ensure_leverage_mode():
    try:
        try:
            ex.set_leverage(LEVERAGE, SYMBOL, params={"side":"BOTH"})
            print(colored(f"✅ leverage set: {LEVERAGE}x","green"))
        except Exception as e:
            print(colored(f"⚠️ set_leverage warn: {e}","yellow"))
        print(colored(f"📌 position mode: {POSITION_MODE}","cyan"))
    except Exception as e:
        print(colored(f"⚠️ ensure_leverage_mode: {e}","yellow"))

try:
    load_market_specs()
    ensure_leverage_mode()
except Exception as e:
    print(colored(f"⚠️ exchange init: {e}","yellow"))

# =================== HELPERS ===================
_consec_err = 0

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
    if q<=0: print(colored(f"⚠️ qty invalid after normalize → {q}","yellow"))
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

def _interval_seconds(iv: str) -> int:
    iv=(iv or "").lower().strip()
    if iv.endswith("m"): return int(float(iv[:-1]))*60
    if iv.endswith("h"): return int(float(iv[:-1]))*3600
    if iv.endswith("d"): return int(float(iv[:-1]))*86400
    return 15*60

def _aligned_since(tf_s:int, bars:int=600):
    now = int(time.time())
    return (now // tf_s - bars) * tf_s * 1000

def fetch_ohlcv(limit=600):
    tf_s = _interval_seconds(INTERVAL)
    since = _aligned_since(tf_s, limit)
    rows = with_retry(lambda: ex.fetch_ohlcv(SYMBOL, timeframe=INTERVAL, limit=limit, since=since, params={"type":"swap"}))
    return pd.DataFrame(rows, columns=["time","open","high","low","close","volume"])

def price_now():
    try:
        t = with_retry(lambda: ex.fetch_ticker(SYMBOL))
        if PRICE_FEED == "mark":
            return float(t.get("info",{}).get("markPrice") or t.get("mark") or t.get("last") or t.get("close"))
        return float(t.get("last") or t.get("close"))
    except Exception:
        return None

def balance_usdt():
    try:
        b = with_retry(lambda: ex.fetch_balance(params={"type":"swap"}))
        # Bybit USDT linear:
        total = b.get("USDT",{}).get("total")
        free  = b.get("USDT",{}).get("free")
        return float(total if total is not None else free)
    except Exception:
        return 100.0  # paper fallback

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

# =================== TV-LIKE INDICATORS ===================
def rma(s: pd.Series, n: int):
    s = s.astype(float)
    if n <= 1: return s
    alpha = 1.0/float(n)
    r = pd.Series(index=s.index, dtype="float64")
    if len(s)==0: return s*0.0
    r.iloc[0] = s.iloc[0]
    for i in range(1,len(s)):
        r.iloc[i] = r.iloc[i-1] + alpha*(s.iloc[i]-r.iloc[i-1])
    return r

def tv_source_series(df: pd.DataFrame) -> pd.Series:
    if TV_SOURCE == "hlc3":
        return (df["high"].astype(float)+df["low"].astype(float)+df["close"].astype(float))/3.0
    return df["close"].astype(float)

def compute_indicators(df: pd.DataFrame):
    d = df.copy()
    if USE_CLOSED_ONLY and len(d)>=2:
        d = d.iloc[:-1]
    if len(d) < max(ATR_LEN,RSI_LEN,ADX_LEN)+2:
        return {"rsi":50.0,"plus_di":0.0,"minus_di":0.0,"dx":0.0,"adx":0.0,"atr":0.0}
    c = d["close"].astype(float); h=d["high"].astype(float); l=d["low"].astype(float)

    tr = pd.concat([(h-l).abs(), (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    atr = rma(tr, ATR_LEN)

    delta = c.diff(); up=delta.clip(lower=0.0); dn=(-delta).clip(lower=0.0)
    rs = rma(up, RSI_LEN) / rma(dn, RSI_LEN).replace(0,1e-12)
    rsi = 100 - (100/(1+rs))

    up_move=h.diff(); down_move=l.shift(1)-l
    plus_dm=up_move.where((up_move>down_move)&(up_move>0),0.0)
    minus_dm=down_move.where((down_move>up_move)&(down_move>0),0.0)
    plus_di=100*(rma(plus_dm, ADX_LEN)/atr.replace(0,1e-12))
    minus_di=100*(rma(minus_dm, ADX_LEN)/atr.replace(0,1e-12))
    dx=(100*(plus_di-minus_di).abs()/(plus_di+minus_di).replace(0,1e-12)).fillna(0.0)
    adx=rma(dx, ADX_LEN)

    i=len(d)-1
    return {
        "rsi": float(rsi.iloc[i]),
        "plus_di": float(plus_di.iloc[i]),
        "minus_di": float(minus_di.iloc[i]),
        "dx": float(dx.iloc[i]),
        "adx": float(adx.iloc[i]),
        "atr": float(atr.iloc[i])
    }

# =================== RANGE FILTER (TV-like) ===================
def _ema(s: pd.Series, n:int): return s.ewm(span=n, adjust=False).mean()
def _rng_size(src: pd.Series, qty: float, n: int) -> pd.Series:
    avrng = _ema((src - src.shift(1)).abs(), n)
    wper = (n*2)-1
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

def rf_signal(df: pd.DataFrame):
    d = df.copy()
    if USE_CLOSED_ONLY or not RF_LIVE_ONLY:
        if len(d)>=2: d = d.iloc[:-1]
    if len(d) < RF_PERIOD + 3:
        px = float(d["close"].iloc[-1]) if len(d) else 0.0
        t  = int(d["time"].iloc[-1]) if len(d) else int(time.time()*1000)
        return {"time":t,"price":px,"long":False,"short":False,"filter":px,"hi":px,"lo":px}

    src = tv_source_series(d)
    hi, lo, filt = _rng_filter(src, _rng_size(src, RF_MULT, RF_PERIOD))

    def _bps(a,b):
        try: return abs((a-b)/b)*10000.0
        except Exception: return 0.0

    p_now=float(src.iloc[-1]); p_prev=float(src.iloc[-2])
    f_now=float(filt.iloc[-1]); f_prev=float(filt.iloc[-2])

    long_flip  = (p_prev <= f_prev and p_now > f_now and _bps(p_now,f_now) >= RF_HYST_BPS)
    short_flip = (p_prev >= f_prev and p_now < f_now and _bps(p_now,f_now) >= RF_HYST_BPS)

    return {
        "time": int(d["time"].iloc[-1]),
        "price": p_now, "long": bool(long_flip), "short": bool(short_flip),
        "filter": f_now, "hi": float(hi.iloc[-1]), "lo": float(lo.iloc[-1])
    }

# =================== EVX FILTER ===================
def evx_signal(df: pd.DataFrame, ind: dict):
    d = df.copy()
    if USE_CLOSED_ONLY and len(d)>=2:
        d = d.iloc[:-1]
    if len(d) < 21: return {"ok": False, "dir":0, "ratio":0.0}
    v=float(d["volume"].iloc[-1]); vma=d["volume"].iloc[-21:-1].astype(float).mean() or 1e-9
    atr=float(ind.get("atr") or 0.0)
    o=float(d["open"].iloc[-1]); c=float(d["close"].iloc[-1]); body=abs(c-o)
    react=(body/max(atr,1e-9))
    ratio=v/vma
    strong = (ratio>=EVX_MIN_VOL_RATIO and react>=EVX_MIN_ATR_REACT)
    return {"ok": bool(strong), "dir": (1 if c>o else -1), "ratio": float(ratio)}

# =================== STATE ===================
STATE = {
    "open": False, "side": None, "entry": None, "qty": 0.0,
    "bars": 0, "trail": None, "breakeven": None,
    "tp1_done": False, "highest_profit_pct": 0.0, "profit_targets_achieved": 0,
    "cooldown": 0, "evx_cool": 0
}
compound_pnl = 0.0

# =================== ORDERS ===================
def _params_open(side):
    if POSITION_MODE=="hedge":
        return {"positionSide":"LONG" if side=="buy" else "SHORT", "reduceOnly": False}
    return {"positionSide":"BOTH", "reduceOnly": False}

def _params_close():
    if POSITION_MODE=="hedge":
        return {"positionSide":"LONG" if STATE.get("side")=="long" else "SHORT", "reduceOnly": True}
    return {"positionSide":"BOTH", "reduceOnly": True}

def _read_position():
    try:
        poss = ex.fetch_positions(params={"type":"swap"})
        for p in poss:
            sym = (p.get("symbol") or p.get("info",{}).get("symbol") or "")
            if SYMBOL.split(":")[0] not in sym: continue
            qty = abs(float(p.get("contracts") or p.get("info",{}).get("size") or 0))
            if qty <= 0: return 0.0, None, None
            entry = float(p.get("entryPrice") or p.get("info",{}).get("avgPrice") or 0)
            side_raw = (p.get("side") or p.get("info",{}).get("side") or "").lower()
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
        print(colored("❌ skip open (qty<=0)","red"))
        return False
    try:
        try: ex.set_leverage(LEVERAGE, SYMBOL, params={"side":"BOTH"})
        except Exception: pass
        ex.create_order(SYMBOL, "market", side, qty, None, _params_open(side))
    except Exception as e:
        print(colored(f"❌ open: {e}","red"))
        logging.error(f"open_market error: {e}")
        return False
    STATE.update({
        "open": True, "side": "long" if side=="buy" else "short", "entry": price,
        "qty": qty, "bars": 0, "trail": None, "breakeven": None,
        "tp1_done": False, "highest_profit_pct": 0.0, "profit_targets_achieved": 0
    })
    print(colored(f"🚀 OPEN {('🟩 LONG' if side=='buy' else '🟥 SHORT')} qty={fmt(qty,4)} @ {fmt(price)}",
                  "green" if side=="buy" else "red"))
    logging.info(f"OPEN {side} qty={qty} price={price}")
    return True

def _reset_after_close():
    STATE.update({
        "open": False, "side": None, "entry": None, "qty": 0.0,
        "bars": 0, "trail": None, "breakeven": None,
        "tp1_done": False, "highest_profit_pct": 0.0, "profit_targets_achieved": 0
    })
    # cooldown
    STATE["cooldown"] = max(COOLDOWN_AFTER_CLOSE_BARS, STATE.get("cooldown",0))

def close_market_strict(reason="STRICT"):
    global compound_pnl
    exch_qty, exch_side, exch_entry = _read_position()
    if exch_qty <= 0:
        if STATE.get("open"):
            _reset_after_close()
        return
    side_to_close = "sell" if (exch_side=="long") else "buy"
    qty_to_close  = safe_qty(exch_qty)
    try:
        ex.create_order(SYMBOL,"market",side_to_close,qty_to_close,None,_params_close())
        time.sleep(0.8)
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
            _reset_after_close()
            return
        # residual → try again
        ex.create_order(SYMBOL,"market",side_to_close,safe_qty(left_qty),None,_params_close())
        _reset_after_close()
    except Exception as e:
        print(colored(f"❌ strict close: {e}","red"))
        logging.error(f"close_market_strict error: {e}")

def close_partial(frac, reason):
    if not STATE["open"] or STATE["qty"]<=0: return
    qty_close = safe_qty(max(0.0, STATE["qty"] * min(max(frac,0.0),1.0)))
    px = price_now() or STATE["entry"]
    min_unit = max(FINAL_CHUNK_QTY, LOT_MIN or FINAL_CHUNK_QTY)
    if qty_close < min_unit:
        print(colored(f"⏸️ skip partial (amount={fmt(qty_close,4)} < min_unit={fmt(min_unit,4)})","yellow"))
        return
    side = "sell" if STATE["side"]=="long" else "buy"
    try:
        ex.create_order(SYMBOL,"market",side,qty_close,None,_params_close())
    except Exception as e:
        print(colored(f"❌ partial close: {e}","red")); return
    STATE["qty"] = safe_qty(STATE["qty"] - qty_close)
    print(colored(f"🔻 PARTIAL {reason} closed={fmt(qty_close,4)} rem={fmt(STATE['qty'],4)}","magenta"))
    if 0 < STATE["qty"] <= FINAL_CHUNK_QTY:
        print(colored(f"🧹 Final chunk ≤ {FINAL_CHUNK_QTY} → strict close","yellow"))
        close_market_strict("FINAL_CHUNK_RULE")

# =================== MANAGEMENT ===================
def manage_after_entry(df, ind, rf_info, evx):
    if not STATE["open"] or STATE["qty"]<=0: return
    px = rf_info.get("price") or price_now() or STATE["entry"]
    entry=STATE["entry"]; side=STATE["side"]
    rr = (px - entry)/entry*100*(1 if side=="long" else -1)

    # TP1
    if (not STATE["tp1_done"]) and rr >= TP1_PCT:
        close_partial(TP1_CLOSE_FRAC, f"TP1@{TP1_PCT:.2f}%")
        STATE["tp1_done"]=True
        if rr >= BREAKEVEN_AFTER: STATE["breakeven"]=entry

    # EVX exit accelerator: لو كان انفجار قوي وعكس
    if EVX_ARM and evx["ok"]:
        # لو انفجار عكسي ضد اتجاهنا نحمي الربح/نقلل تعرض
        if (side=="long" and evx["dir"]<0 and rr>0) or (side=="short" and evx["dir"]>0 and rr>0):
            close_partial(0.40, "EVX-Accelerator")
            STATE["breakeven"]=entry

    # Trail ATR بعد التفعيل
    atr=float(ind.get("atr") or 0.0)
    if rr >= TRAIL_ACTIVATE_PCT and atr>0 and px is not None:
        gap = atr * ATR_MULT_TRAIL
        if side=="long":
            new_trail = px - gap
            STATE["trail"] = max(STATE["trail"] or new_trail, new_trail, STATE.get("breakeven") or -1e9)
            if px < STATE["trail"]:
                close_market_strict(f"TRAIL_ATR({ATR_MULT_TRAIL}x)")
        else:
            new_trail = px + gap
            STATE["trail"] = min(STATE["trail"] or new_trail, new_trail, STATE.get("breakeven") or 1e9)
            if px > STATE["trail"]:
                close_market_strict(f"TRAIL_ATR({ATR_MULT_TRAIL}x)")

# =================== SNAPSHOT ===================
def time_to_candle_close(df: pd.DataFrame) -> int:
    tf = _interval_seconds(INTERVAL)
    if len(df)==0: return tf
    cur_start_ms = int(df["time"].iloc[-1])
    now_ms = int(time.time()*1000)
    next_close_ms = cur_start_ms + tf*1000
    while next_close_ms <= now_ms:
        next_close_ms += tf*1000
    return max(0, int((next_close_ms - now_ms)/1000))

def pretty_snapshot(bal, info, ind, spread_bps, evx, reason=None, df=None):
    left_s = time_to_candle_close(df) if df is not None else 0
    print(colored("─"*110,"cyan"))
    print(colored(f"📊 {SYMBOL} {INTERVAL} • {'UTC' if TZ=='UTC' else TZ} • {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC","cyan"))
    print(colored("─"*110,"cyan"))
    print("📈 RF & INDICATORS")
    print(f"   💲 Price {fmt(info.get('price'))} | RF filt={fmt(info.get('filter'))}  hi={fmt(info.get('hi'))} lo={fmt(info.get('lo'))} | spread={fmt(spread_bps,2)} bps")
    print(f"   🧮 RSI={fmt(ind.get('rsi'))}  +DI={fmt(ind.get('plus_di'))}  -DI={fmt(ind.get('minus_di'))}  ADX={fmt(ind.get('adx'))}  ATR={fmt(ind.get('atr'))}")
    print(f"   💥 EVX ok={evx['ok']} dir={evx['dir']} ratio={fmt(evx['ratio'],2)}")
    print(f"   ⏱️ closes_in ≈ {left_s}s")
    print("\n🧭 POSITION")
    bal_line = f"Balance={fmt(bal,2)}  Risk={int(RISK_ALLOC*100)}%×{LEVERAGE}x"
    print(colored(f"   {bal_line}","yellow"))
    if STATE["open"]:
        lamp='🟩 LONG' if STATE['side']=='long' else '🟥 SHORT'
        print(f"   {lamp}  Entry={fmt(STATE['entry'])}  Qty={fmt(STATE['qty'],4)}  Bars={STATE['bars']}  Trail={fmt(STATE['trail'])}  BE={fmt(STATE['breakeven'])}")
        print(f"   🎯 TP_done={STATE['profit_targets_achieved']}  HP={fmt(STATE['highest_profit_pct'],2)}%")
    else:
        print("   ⚪ FLAT")
    if reason: print(colored(f"   ℹ️ reason: {reason}","white"))
    print(colored("─"*110,"cyan"))

# =================== LOOP ===================
def trade_loop():
    while True:
        try:
            bal = balance_usdt()
            df  = fetch_ohlcv()
            px  = price_now()

            rf = rf_signal(df)
            ind = compute_indicators(df)
            evx = evx_signal(df, ind)
            spread_bps = orderbook_spread_bps()

            # update pnl/highest
            if STATE["open"] and px and STATE["entry"]:
                rr = (px-STATE["entry"])/STATE["entry"]*100*(1 if STATE["side"]=="long" else -1)
                if rr>STATE["highest_profit_pct"]: STATE["highest_profit_pct"]=rr

            # manage after entry
            manage_after_entry(df, ind, {"price": px or rf["price"], **rf}, evx)

            # entry logic
            reason=None
            if spread_bps is not None and spread_bps > SPREAD_GUARD_BPS:
                reason=f"spread too high ({fmt(spread_bps,2)}bps > {SPREAD_GUARD_BPS})"

            if not STATE["open"] and reason is None:
                sig = "buy" if rf["long"] else ("sell" if rf["short"] else None)

                # cooldown after close
                if STATE.get("cooldown",0)>0:
                    STATE["cooldown"] -= 1 if len(df)>=2 and int(df["time"].iloc[-1])!=int(df["time"].iloc[-2]) else 0
                    sig=None; reason="cooldown"

                # EVX as entry guard
                if EVX_ARM and sig and not evx["ok"]:
                    sig=None; reason="EVX guard"

                if sig:
                    qty = compute_size(bal, px or rf["price"])
                    if qty>0:
                        open_market(sig, qty, px or rf["price"])
                    else:
                        reason="qty<=0"

            # bar counter
            if len(df)>=2 and int(df["time"].iloc[-1])!=int(df["time"].iloc[-2]) and STATE["open"]:
                STATE["bars"] += 1

            pretty_snapshot(bal, {"price": px or rf["price"], **rf}, ind, spread_bps, evx, reason, df)

            time.sleep(BASE_SLEEP)
        except Exception as e:
            print(colored(f"❌ loop error: {e}\n{traceback.format_exc()}","red"))
            logging.error(f"trade_loop error: {e}\n{traceback.format_exc()}")
            time.sleep(BASE_SLEEP)

# =================== API ===================
app = Flask(__name__)
@app.route("/")
def home():
    mode = "LIVE" if (os.getenv(f"{EXCHANGE.upper()}_API_KEY","") and os.getenv(f"{EXCHANGE.upper()}_API_SECRET","")) else "PAPER"
    return f"✅ BYBIT RF+EVX — {SYMBOL} {INTERVAL} — {mode} — TV_MATCH={TV_MATCH_MODE} — ClosedOnly={USE_CLOSED_ONLY} — EVX={EVX_ARM}"

@app.route("/metrics")
def metrics():
    return jsonify({
        "exchange": EXCHANGE, "symbol": SYMBOL, "interval": INTERVAL,
        "leverage": LEVERAGE, "risk_alloc": RISK_ALLOC, "price": price_now(),
        "state": STATE, "compound_pnl": compound_pnl,
        "guards": {"spread_bps": SPREAD_GUARD_BPS, "final_chunk_qty": FINAL_CHUNK_QTY},
        "tv": {"match": TV_MATCH_MODE, "source": TV_SOURCE, "use_closed_only": USE_CLOSED_ONLY, "price_feed": PRICE_FEED}
    })

@app.route("/health")
def health():
    return jsonify({
        "ok": True, "open": STATE["open"], "side": STATE["side"], "qty": STATE["qty"],
        "compound_pnl": compound_pnl, "timestamp": datetime.utcnow().isoformat()
    }), 200

def keepalive_loop():
    url=(SELF_URL or "").strip().rstrip("/")
    if not url:
        print(colored("⛔ keepalive disabled (SELF_URL not set)","yellow"))
        return
    import requests
    sess=requests.Session(); sess.headers.update({"User-Agent":"bybit-rf-evx/keepalive"})
    print(colored(f"KEEPALIVE every 50s → {url}","cyan"))
    while True:
        try: sess.get(url, timeout=8)
        except Exception: pass
        time.sleep(50)

# =================== BOOT ===================
if __name__ == "__main__":
    print(colored(f"MODE: {'LIVE' if (os.getenv(f'{EXCHANGE.upper()}_API_KEY','') and os.getenv(f'{EXCHANGE.upper()}_API_SECRET','')) else 'PAPER'}  •  {SYMBOL}  •  {INTERVAL}","yellow"))
    print(colored(f"RISK: {int(RISK_ALLOC*100)}% × {LEVERAGE}x  •  RF_LIVE={RF_LIVE_ONLY}  •  TV_MATCH={TV_MATCH_MODE}  •  USE_CLOSED_ONLY={USE_CLOSED_ONLY}","yellow"))
    print(colored(f"ENTRY: RF{' (EVX guarded)' if EVX_ARM else ''}  •  FINAL_CHUNK_QTY={FINAL_CHUNK_QTY}","yellow"))
    logging.info("service starting…")
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    signal.signal(signal.SIGINT,  lambda *_: sys.exit(0))
    import threading
    threading.Thread(target=trade_loop, daemon=True).start()
    threading.Thread(target=keepalive_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)
