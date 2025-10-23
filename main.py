# -*- coding: utf-8 -*-
"""
BYBIT — RF-CLOSE FUSION (Entry by RF only, CLOSED candle)
• Exchange: Bybit USDT Perps via CCXT
• Entry: Range Filter (TradingView-like), CLOSED-candle flip only
• Council (after entry only): SMC (EQH/EQL + OB + FVG + SDZ) + Candles + EVX + Momentum + Fake/Real Break + Retest + Liquidity traps
• NO partial take-profits, NO ATR trailing. One-shot strict close at 'max-logic' profit (council-confirmed)
• Cumulative PnL tracking + strict close (reduceOnly) + final-chunk guard (tiny residual) + Flask /metrics /health
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
API_KEY = os.getenv("BYBIT_API_KEY", "") or os.getenv("BINGX_API_KEY","")
API_SECRET = os.getenv("BYBIT_API_SECRET", "") or os.getenv("BINGX_API_SECRET","")
MODE_LIVE = bool(API_KEY and API_SECRET)

SELF_URL = os.getenv("SELF_URL", "")
PORT = int(os.getenv("PORT", 5000))

# =================== FIXED CONFIG ===================
SYMBOL     = "SOL/USDT:USDT"
INTERVAL   = "15m"

LEVERAGE   = 10
RISK_ALLOC = 0.60
POSITION_MODE = "oneway"

# Range Filter (TV-like)
RF_SOURCE   = "close"
RF_PERIOD   = 20
RF_MULT     = 3.5
RF_HYST_BPS = 6.0           # hysteresis bps for flip
RF_CLOSED_ONLY = True       # <-- دخول على الشمعة المغلقة فقط

# Indicators (RMA/Wilder - TV compat)
RSI_LEN = 14
ADX_LEN = 14
ATR_LEN = 14

# Final chunk strict close threshold (contracts)
FINAL_CHUNK_QTY = 0.2

# Council thresholds (يمكن تعديلها بحذر)
COUNCIL_MIN_VOTES_FOR_STRICT = 3         # عدد إشارات من المجلس مطلوبة لتأكيد الخروج
LEVEL_NEAR_BPS               = 10.0      # قرب من مستوى SMC/SDZ بالـbps
EVX_STRONG_RATIO             = 1.8       # انفجار حجم
EVX_BODY_ATR_MIN             = 1.2       # جسم/ATR
EVX_COOL_OFF_RATIO           = 1.1       # تبريد الانفجار
ADX_COOL_OFF_DROP            = 2.0       # هبوط ADX ≥ 2 نقاط من قمة قريبة
RSI_NEUTRAL_MIN, RSI_NEUTRAL_MAX = 45.0, 55.0
RETEST_MAX_BARS              = 6         # أعظم عدد شموع لمراقبة إعادة الاختبار
CHOP_ATR_PCTL                = 0.25      # لو ATR ضمن رُبع التوزيع الأخير => سوق نايم

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
wait_for_next_signal_side = None   # بعد الإغلاق: انتظر إشارة RF المعاكسة
last_adx_peak = None               # تتبع قمة ADX الأخيرة لأغراض التبريد

STATE = {
    "open": False, "side": None, "entry": None, "qty": 0.0,
    "pnl": 0.0, "bars": 0,
    "highest_profit_pct": 0.0,
    "breakeven": None,  # غير مستخدمة الآن ولكن نعرضها في اللوج
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
        return {"rsi":None,"plus_di":None,"minus_di":None,"dx":None,"adx":None,"atr":None,"atr_pctl":None}
    c = df["close"].astype(float); h=df["high"].astype(float); l=df["low"].astype(float)

    # RSI (RMA/Wilder)
    delta=c.diff(); up=delta.clip(lower=0.0); dn=(-delta).clip(lower=0.0)
    rma_up=_rma(up, RSI_LEN); rma_dn=_rma(dn, RSI_LEN).replace(0,1e-12)
    rs=rma_up/rma_dn; rsi=100-(100/(1+rs))

    # ADX (RMA/Wilder)
    up_move=h.diff(); down_move=l.shift(1)-l
    plus_dm=up_move.where((up_move>down_move)&(up_move>0),0.0)
    minus_dm=down_move.where((down_move>up_move)&(down_move>0),0.0)
    tr=_true_range(h,l,c); atr=_rma(tr, ATR_LEN).replace(0,1e-12)
    plus_di=100*_rma(plus_dm, ADX_LEN)/atr
    minus_di=100*_rma(minus_dm, ADX_LEN)/atr
    dx=(100*(plus_di-minus_di).abs()/(plus_di+minus_di).replace(0,1e-12)).fillna(0.0)
    adx=_rma(dx, ADX_LEN)

    # ATR percentile (chop detector)
    atr_hist = _rma(tr, ATR_LEN)
    atr_pctl = None
    try:
        last_atr = float(atr.iloc[-1])
        window = atr_hist.iloc[-200:].dropna().astype(float).values
        if len(window)>=20:
            atr_pctl = float((np.sum(window <= last_atr)/len(window)))
    except Exception:
        atr_pctl = None

    # track last adx peak
    try:
        cur_adx = float(adx.iloc[-1])
        if last_adx_peak is None or cur_adx > last_adx_peak:
            last_adx_peak = cur_adx
    except Exception:
        pass

    i = len(df)-1
    return {
        "rsi": float(rsi.iloc[i]), "plus_di": float(plus_di.iloc[i]),
        "minus_di": float(minus_di.iloc[i]), "dx": float(dx.iloc[i]),
        "adx": float(adx.iloc[i]), "atr": float(atr.iloc[i]),
        "atr_pctl": atr_pctl
    }

# =================== RANGE FILTER (TV-like, CLOSED CANDLE) ===================
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

def rf_signal_closed(df: pd.DataFrame):
    """Flip detected on LAST CLOSED candle only."""
    if len(df) < RF_PERIOD + 3:
        i=-1; price=float(df["close"].iloc[i]) if len(df) else None
        return {"time": int(df["time"].iloc[i]) if len(df) else int(time.time()*1000),
                "price": price or 0.0, "long": False, "short": False,
                "filter": price or 0.0, "hi": price or 0.0, "lo": price or 0.0}
    src = df[RF_SOURCE].astype(float)
    hi, lo, filt = _rng_filter(src, _rng_size(src, RF_MULT, RF_PERIOD))

    def _bps(a,b):
        try: return abs((a-b)/b)*10000.0
        except Exception: return 0.0

    # نستخدم آخر شمعة مُغلقة: idx=-2 مقارنة بـ -3
    p_now=float(src.iloc[-2]); p_prev=float(src.iloc[-3])
    f_now=float(filt.iloc[-2]); f_prev=float(filt.iloc[-3])

    long_flip  = (p_prev <= f_prev and p_now > f_now and _bps(p_now, f_now) >= RF_HYST_BPS)
    short_flip = (p_prev >= f_prev and p_now < f_now and _bps(p_now, f_now) >= RF_HYST_BPS)

    return {
        "time": int(df["time"].iloc[-2]), "price": float(src.iloc[-1]),  # price الحالي للعرض فقط
        "long": bool(long_flip), "short": bool(short_flip),
        "filter": f_now, "hi": float(hi.iloc[-2]), "lo": float(lo.iloc[-2])
    }

# =================== PATTERNS / SMC / SDZ / EVX ===================
def detect_candle(df: pd.DataFrame):
    if len(df)<3: return {"pattern":"NONE","strength":0,"dir":0}
    o=float(df["open"].iloc[-2]); h=float(df["high"].iloc[-2])
    l=float(df["low"].iloc[-2]);  c=float(df["close"].iloc[-2])  # آخر شمعة مُغلقة
    rng=max(h-l,1e-12); body=abs(c-o)
    upper=h-max(o,c); lower=min(o,c)-l
    upper_pct=upper/rng*100.0; lower_pct=lower/rng*100.0; body_pct=body/rng*100.0
    if body_pct<=10: return {"pattern":"DOJI","strength":1,"dir":0}
    if body_pct>=85 and upper_pct<=7 and lower_pct<=7:
        return {"pattern":"MARUBOZU","strength":3,"dir":(1 if c>o else -1)}
    if lower_pct>=60 and body_pct<=30 and c>o: return {"pattern":"HAMMER","strength":2,"dir":1}
    if upper_pct>=60 and body_pct<=30 and c<o: return {"pattern":"SHOOTING","strength":2,"dir":-1}
    return {"pattern":"NONE","strength":0,"dir":(1 if c>o else -1)}

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
            if abs((px-lv)/lv)*10000.0 <= bps:
                return lv
    except Exception:
        pass
    return None

def detect_smc_levels(df: pd.DataFrame):
    """EQH/EQL + simple OB + simple FVG + SDZ (supply/demand zones from swing bodies)."""
    try:
        d = df.copy()
        ph, pl = _find_swings(d, 2, 2)

        def _eq(vals, is_high=True):
            res = []; tol_pct = 0.05
            for i, price in enumerate(vals):
                if price is None: continue
                tol = price * tol_pct / 100.0
                neighbors = [vals[j] for j in range(max(0,i-10), min(len(vals),i+10))
                             if vals[j] is not None and abs(vals[j] - price) <= tol]
                if len(neighbors) >= 2:
                    res.append(max(neighbors) if is_high else min(neighbors))
            if not res: return None
            return max(res) if is_high else min(res)
        eqh = _eq(ph, True)
        eql = _eq(pl, False)

        # OB (آخر شمعة قوية)
        ob = None
        for i in range(len(d)-3, max(len(d)-50, 1), -1):
            o=float(d["open"].iloc[i]); c=float(d["close"].iloc[i])
            h=float(d["high"].iloc[i]); l=float(d["low"].iloc[i])
            rng=max(h-l,1e-12); body=abs(c-o)
            upper=h-max(o,c); lower=min(o,c)-l
            if body>=0.6*rng and (upper/rng)<=0.2 and (lower/rng)<=0.2:
                side="bull" if c>o else "bear"
                ob={"side":side,"bot":min(o,c),"top":max(o,c),"time":int(d["time"].iloc[i])}
                break

        # FVG (بسيط)
        fvg=None
        for i in range(len(d)-3, max(len(d)-30, 2), -1):
            prev_high = float(d["high"].iloc[i-1]); prev_low = float(d["low"].iloc[i-1])
            curr_low  = float(d["low"].iloc[i]);   curr_high = float(d["high"].iloc[i])
            if curr_low > prev_high:
                fvg={"type":"BULL_FVG","bottom":prev_high,"top":curr_low}; break
            if curr_high < prev_low:
                fvg={"type":"BEAR_FVG","bottom":curr_high,"top":prev_low}; break

        # SDZ: مناطق عرض/طلب مبسطة من أجسام السوينجز السابقة
        sdz = None
        try:
            # خذ آخر swing قوي للجسم
            idxs = [i for i,v in enumerate(ph) if v is not None] + [i for i,v in enumerate(pl) if v is not None]
            if idxs:
                focus = max(idxs)
                o=float(d["open"].iloc[focus]); c=float(d["close"].iloc[focus])
                bot=min(o,c); top=max(o,c)
                side = "demand" if c>o else "supply"
                sdz = {"side":side, "bot":bot, "top":top, "time": int(d["time"].iloc[focus])}
        except Exception:
            sdz = None

        return {"eqh":eqh, "eql":eql, "ob":ob, "fvg":fvg, "sdz": sdz}
    except Exception:
        return {"eqh":None,"eql":None,"ob":None,"fvg":None,"sdz":None}

def explosion_signal(df: pd.DataFrame, ind: dict):
    if len(df)<21: return {"explosion":False,"dir":0,"ratio":0.0}
    try:
        # استخدم آخر شمعة مُغلقة
        o=float(df["open"].iloc[-2]); c=float(df["close"].iloc[-2])
        v=float(df["volume"].iloc[-2]); vma=df["volume"].iloc[-22:-2].astype(float).mean() or 1e-9
        atr=float(ind.get("atr") or 0.0)
        body=abs(c-o); react=(body/max(atr,1e-9)); ratio=v/max(vma,1e-9)
        strong = (ratio>=EVX_STRONG_RATIO and react>=EVX_BODY_ATR_MIN)
        return {"explosion":bool(strong),"dir":(1 if c>o else -1),"ratio":float(ratio), "react": float(react)}
    except Exception:
        return {"explosion":False,"dir":0,"ratio":0.0,"react":0.0}

# =============== FAKE/REAL BREAK + RETEST + TRAPS =================
def near_level(px, lvl, bps=LEVEL_NEAR_BPS):
    try: return abs((px-lvl)/lvl)*10000.0 <= bps
    except Exception: return False

def detect_fake_break(df: pd.DataFrame, smc: dict):
    """كسر وهمي: اختراق هاي/لو ثم إغلاق داخل/رفض واضح مع فتيل طويل وحجم أعلى من المعتاد."""
    if len(df)<4: return {"fake_break": False, "side": None}
    try:
        # ننظر للشمعة المغلقة الأخيرة
        o=float(df["open"].iloc[-2]); h=float(df["high"].iloc[-2])
        l=float(df["low"].iloc[-2]);  c=float(df["close"].iloc[-2])
        prev_h=float(df["high"].iloc[-3]); prev_l=float(df["low"].iloc[-3])
        rng=max(h-l,1e-12); upper=h-max(o,c); lower=min(o,c)-l
        vol=float(df["volume"].iloc[-2]); vma=df["volume"].iloc[-22:-2].astype(float).mean() or 1e-9

        eqh, eql = smc.get("eqh"), smc.get("eql")
        fake_up=False; fake_dn=False
        if eqh and h>eqh and c<eqh and (upper/rng)>=0.55 and vol/vma>=1.2:
            fake_up=True
        if eql and l<eql and c>eql and (lower/rng)>=0.55 and vol/vma>=1.2:
            fake_dn=True
        if fake_up: return {"fake_break": True, "side": "up"}
        if fake_dn: return {"fake_break": True, "side": "down"}
    except Exception:
        pass
    return {"fake_break": False, "side": None}

def detect_retest(df: pd.DataFrame, level: float, side: str):
    """إعادة اختبار بسيطة خلال RE-TEST window."""
    try:
        if level is None: return False
        # راقب آخر RE-TEST window من الشموع المغلقة
        closes = df["close"].astype(float).iloc[-(RETEST_MAX_BARS+1):-1].values
        if side=="long":
            # بعد كسر مستوى مقاومة، نريد هبوط طفيف يلمس قريب المستوى ثم ارتداد
            return any(near_level(px, level) for px in closes)
        else:
            # بعد كسر دعم، نريد صعود طفيف يلمس قريب المستوى ثم رفض
            return any(near_level(px, level) for px in closes)
    except Exception:
        return False

def detect_trap(df: pd.DataFrame, smc: dict):
    """Stop-hunt/Trap: فتائل طويلة عند EQH/EQL/OB/SDZ + حجم أعلى."""
    if len(df)<3: return None
    try:
        o=float(df["open"].iloc[-2]); h=float(df["high"].iloc[-2])
        l=float(df["low"].iloc[-2]);  c=float(df["close"].iloc[-2])
        rng=max(h-l,1e-12); upper=h-max(o,c); lower=min(o,c)-l
        v=float(df["volume"].iloc[-2]); vma=df["volume"].iloc[-22:-2].astype(float).mean() or 1e-9
        near_eqh = smc.get("eqh") and near_level(h, smc["eqh"], 12.0)
        near_eql = smc.get("eql") and near_level(l, smc["eql"], 12.0)
        ob=smc.get("ob") or {}
        sdz=smc.get("sdz") or {}
        near_ob = ob and (near_level(h, ob.get("bot",h), 12.0) or near_level(l, ob.get("top",l), 12.0))
        near_sdz = sdz and (near_level(h, sdz.get("bot",h), 12.0) or near_level(l, sdz.get("top",l), 12.0))

        bull_trap = (lower/rng>=0.60 and (near_eql or near_ob or near_sdz) and v/vma>=1.2)
        bear_trap = (upper/rng>=0.60 and (near_eqh or near_ob or near_sdz) and v/vma>=1.2)
        if bull_trap: return {"trap":"bull"}
        if bear_trap: return {"trap":"bear"}
    except Exception:
        pass
    return None

# =================== COUNCIL DECISION ===================
def council_assess(df, ind, info, smc, cache):
    """يرجع قرار المجلس: hold / exit_strict + الأسباب (votes)"""
    votes = []
    price = info.get("price")

    # 1) قرب مستويات سيولة/هيكلة (EQH/EQL/OB/SDZ/FVG)
    near_any = False
    try:
        lvls = []
        for k in ["eqh","eql"]:
            if smc.get(k): lvls.append(smc[k])
        for k in ["ob","sdz"]:
            if smc.get(k):
                lvls += [smc[k].get("bot"), smc[k].get("top")]
        if smc.get("fvg"):
            lvls += [smc["fvg"].get("bottom"), smc["fvg"].get("top")]
        lvls = [x for x in lvls if x]
        hit = _nearest_level(price, lvls, LEVEL_NEAR_BPS)
        if hit is not None:
            near_any = True
            votes.append(f"near_structure:{hit:.4f}")
    except Exception:
        pass

    # 2) شمعة رفض/تعب (منذ الشمعة المغلقة الأخيرة)
    cndl = detect_candle(df)
    if cndl["pattern"] in ("SHOOTING","HAMMER") or (cndl["pattern"]=="DOJI" and near_any):
        votes.append(f"candle_reject:{cndl['pattern']}")

    # 3) انفجار ثم تبريد (EVX → cool-off)
    evx = explosion_signal(df, ind)
    if evx["explosion"]:
        votes.append("evx_strong")
    else:
        # تبريد بعد انفجار سابق (بسيط: حجم قريب من المتوسط + جسم/ATR هادئ)
        if evx["ratio"] <= EVX_COOL_OFF_RATIO:
            votes.append("evx_cool")

    # 4) تبريد زخم: ADX هابط من قمة قريبة أو RSI محايد
    try:
        adx = float(ind.get("adx") or 0.0); rsi=float(ind.get("rsi") or 50.0)
        if last_adx_peak is not None and (last_adx_peak - adx) >= ADX_COOL_OFF_DROP:
            votes.append("adx_cool")
        if RSI_NEUTRAL_MIN <= rsi <= RSI_NEUTRAL_MAX:
            votes.append("rsi_neutral")
    except Exception:
        pass

    # 5) كسر وهمي / إعادة اختبار فاشلة
    fk = detect_fake_break(df, smc)
    if fk["fake_break"]:
        votes.append(f"fake_break:{fk['side']}")
    # re-test: إذا كانت الصفقة long وتجاوزت EQH/OB، ابحث عن retest؛ والعكس للـ short
    side = STATE.get("side")
    l_for_retest = smc.get("eqh") if side=="long" else smc.get("eql")
    if detect_retest(df, l_for_retest, "long" if side=="long" else "short"):
        votes.append("retest_touched")

    # 6) سوق نايم (chop) + تذبذب وفتائل عند المستويات ⇒ خروج على ربح منطقي بدل الانتظار
    atr_pctl = ind.get("atr_pctl")
    if atr_pctl is not None and atr_pctl <= CHOP_ATR_PCTL and near_any and cndl["pattern"] in ("DOJI","SHOOTING","HAMMER"):
        votes.append("chop_near_level")

    # 7) فخاخ سيولة
    trp = detect_trap(df, smc)
    if trp: votes.append(f"trap:{trp['trap']}")

    # احتساب القرار
    decision = "hold"
    if len(votes) >= COUNCIL_MIN_VOTES_FOR_STRICT:
        decision = "exit_strict"

    # حفظ تفاصيل المجلس في الحالة
    STATE["council"] = {"votes": len(votes), "reasons": votes}
    return {"decision": decision, "votes": votes}

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

def open_market(side, qty, price):
    if qty<=0: print(colored("❌ skip open (qty<=0)", "red")); return False
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
    print(colored(f"🚀 OPEN {('🟩 LONG' if side=='buy' else '🟥 SHORT')} qty={fmt(qty,4)} @ {fmt(price)}", "green" if side=='buy' else "red"))
    logging.info(f"OPEN {side} qty={qty} price={price}")
    return True

def close_market_strict(reason="STRICT"):
    global compound_pnl, wait_for_next_signal_side
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
    # انتظر إشارة RF المعاكسة قبل دخول جديد
    if prev_side == "long":  wait_for_next_signal_side = "sell"
    elif prev_side == "short": wait_for_next_signal_side = "buy"
    else: wait_for_next_signal_side = None
    logging.info(f"AFTER_CLOSE reason={reason} wait_for={wait_for_next_signal_side}")

# =================== COUNCIL-DRIVEN MANAGEMENT ===================
def manage_after_entry(df, ind, info, smc):
    """بدون TPs وبدون Trailing. المجلس فقط يقرر الخروج الصارم عند أقوى سبب منطقي."""
    if not STATE["open"] or STATE["qty"]<=0: return
    px = info["price"]; entry=STATE["entry"]; side=STATE["side"]
    rr = (px - entry)/entry*100*(1 if side=="long" else -1)

    # أعلى ربح مُسجّل
    if rr > STATE["highest_profit_pct"]:
        STATE["highest_profit_pct"] = rr

    # تقييم المجلس
    decision = council_assess(df, ind, info, smc, STATE.get("council", {}))

    # لو المجلس قرر خروج صارم → أغلق بالكامل
    if decision["decision"] == "exit_strict":
        close_market_strict("COUNCIL_MAX_PROFIT_CONFIRMED")
        return

# =================== SNAPSHOT ===================
def pretty_snapshot(bal, info, ind, smc, reason=None, df=None):
    left_s = time_to_candle_close(df) if df is not None else 0
    cndl = detect_candle(df)
    evx  = explosion_signal(df, ind)

    print(colored("─"*110,"cyan"))
    print(colored(f"📊 {SYMBOL} {INTERVAL} • {'LIVE' if MODE_LIVE else 'PAPER'} • {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC","cyan"))
    print(colored("─"*110,"cyan"))
    print("📈 RF (CLOSED) & INDICATORS (TV-Compat)")
    print(f"   💲 Price {fmt(info.get('price'))} | RF filt={fmt(info.get('filter'))}  hi={fmt(info.get('hi'))} lo={fmt(info.get('lo'))}")
    print(f"   🧮 RSI={fmt(ind.get('rsi'))}  +DI={fmt(ind.get('plus_di'))}  -DI={fmt(ind.get('minus_di'))}  ADX={fmt(ind.get('adx'))}  ATR={fmt(ind.get('atr'))}  ATRpctl={fmt(ind.get('atr_pctl'),3)}")
    print(f"   💥 EVX: strong={'Yes' if evx.get('explosion') else 'No'}  ratio={fmt(evx.get('ratio'),2)} react={fmt(evx.get('react'),2)}  candle={cndl.get('pattern')}")
    print(f"   🏗️ SMC: EQH={fmt(smc.get('eqh'))}  EQL={fmt(smc.get('eql'))}  OB={smc.get('ob')}  FVG={smc.get('fvg')}  SDZ={smc.get('sdz')}")
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
    loop_i=0
    last_side = None
    while True:
        try:
            bal = balance_usdt()
            px  = price_now()
            df  = fetch_ohlcv()

            info = rf_signal_closed(df)            # RF على الشمعة المُغلقة فقط
            ind  = compute_indicators(df)

            # SMC على تاريخ مغلق فقط (استبعاد الشمعة الحالية)
            df_closed = df.iloc[:-1] if len(df)>=2 else df.copy()
            smc = detect_smc_levels(df_closed)

            # تحديث PnL الحالي
            if STATE["open"] and px:
                STATE["pnl"] = (px-STATE["entry"])*STATE["qty"] if STATE["side"]=="long" else (STATE["entry"]-px)*STATE["qty"]

            # إدارة الصفقة بالمجلس
            manage_after_entry(df, ind, {"price": px or info["price"], **info}, smc)

            # ENTRY: RF CLOSED ONLY — مع انتظار الإشارة المعاكسة بعد الإغلاق
            reason=None
            sig = "buy" if info["long"] else ("sell" if info["short"] else None)
            if (not STATE["open"]) and sig:
                if wait_for_next_signal_side and sig != wait_for_next_signal_side:
                    reason=f"waiting opposite RF: need {wait_for_next_signal_side.upper()}"
                else:
                    qty = compute_size(bal, px or info["price"])
                    raw_qty = ((bal or 0.0)*RISK_ALLOC*LEVERAGE)/max(px or info["price"] or 1e-9,1e-9)
                    logging.info(f"QTY_DEBUG bal={fmt(bal,4)} price={fmt(px or info['price'])} raw={fmt(raw_qty,8)} -> qty={fmt(qty,8)} min={LOT_MIN} step={LOT_STEP} prec={AMT_PREC}")
                    if qty>0:
                        open_market(sig, qty, px or info["price"])
                        last_side = sig
                        wait_for_next_signal_side = None
                    else:
                        reason="qty<=0"

            pretty_snapshot(bal, {"price": px or info["price"], **info}, ind, smc, reason, df)

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
    return f"✅ RF-CLOSE FUSION — {SYMBOL} {INTERVAL} — {mode} — Entry: RF CLOSED only — Council strict-exit — FinalChunk={FINAL_CHUNK_QTY}"

@app.route("/metrics")
def metrics():
    return jsonify({
        "symbol": SYMBOL, "interval": INTERVAL, "mode": "live" if MODE_LIVE else "paper",
        "leverage": LEVERAGE, "risk_alloc": RISK_ALLOC, "price": price_now(),
        "state": STATE, "compound_pnl": compound_pnl,
        "entry_mode": "RF_CLOSED_ONLY", "waiting_for": wait_for_next_signal_side
    })

@app.route("/health")
def health():
    return jsonify({
        "ok": True, "mode": "live" if MODE_LIVE else "paper",
        "open": STATE["open"], "side": STATE["side"], "qty": STATE["qty"],
        "compound_pnl": compound_pnl, "timestamp": datetime.utcnow().isoformat(),
        "entry_mode": "RF_CLOSED_ONLY", "council_votes": STATE.get("council",{}).get("votes",0)
    }), 200

def keepalive_loop():
    url=(SELF_URL or "").strip().rstrip("/")
    if not url:
        print(colored("⛔ keepalive disabled (SELF_URL not set)", "yellow"))
        return
    import requests
    sess=requests.Session(); sess.headers.update({"User-Agent":"rf-close/keepalive"})
    print(colored(f"KEEPALIVE every 50s → {url}", "cyan"))
    while True:
        try: sess.get(url, timeout=8)
        except Exception: pass
        time.sleep(50)

# =================== BOOT ===================
if __name__ == "__main__":
    print(colored(f"MODE: {'LIVE' if MODE_LIVE else 'PAPER'}  •  {SYMBOL}  •  {INTERVAL}", "yellow"))
    print(colored(f"RISK: {int(RISK_ALLOC*100)}% × {LEVERAGE}x  •  ENTRY=RF_CLOSED_ONLY", "yellow"))
    print(colored(f"NO TPs/NO TRAIL • STRICT EXIT by COUNCIL • FINAL_CHUNK_QTY={FINAL_CHUNK_QTY}", "yellow"))
    logging.info("service starting…")
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    signal.signal(signal.SIGINT,  lambda *_: sys.exit(0))
    import threading
    threading.Thread(target=trade_loop, daemon=True).start()
    threading.Thread(target=keepalive_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)
