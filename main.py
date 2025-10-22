# -*- coding: utf-8 -*-
"""
RF Futures Bot — Trend Pro++ (BingX Perp, CCXT) — LIVE-RF ENTRY — FINAL (TRAPS+WICKS)
• دخول: Range Filter على الشمعة الحيّة + حارس السبريد
• تأكيد القمم/القيعان + فشل الاختراق → جني/تشديد/إغلاق
• قراءة شموع: body/upper/lower wick + range/ATR + انفجارات
• FVG/OB/EQH/EQL (SMC مبسّط) + فوبوناتشي امتدادات + Golden Pocket
• Wick Harvest + EMA Harvest + ATR Trail + Ratchet Lock
• لوج احترافي + /metrics + /health
"""

import os, time, math, threading, requests, traceback, random, signal, sys, logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from decimal import Decimal, ROUND_DOWN, InvalidOperation
import pandas as pd

try: import ccxt
except Exception: ccxt=None

def colored(t,*a,**k):
    try:
        from termcolor import colored as _c
        return _c(t,*a,**k)
    except Exception:
        return t

# =================== ENV / MODE ===================
API_KEY    = os.getenv("BINGX_API_KEY", "")
API_SECRET = os.getenv("BINGX_API_SECRET", "")
MODE_LIVE  = bool(API_KEY and API_SECRET and ccxt is not None)
SELF_URL   = os.getenv("SELF_URL", "") or os.getenv("RENDER_EXTERNAL_URL", "")
PORT       = int(os.getenv("PORT", 5000))

# =================== SETTINGS ===================
SYMBOL   = "DOGE/USDT:USDT"
INTERVAL = "15m"

LEVERAGE   = 10          # 10x
RISK_ALLOC = 0.60        # 60% من الرصيد
BINGX_POSITION_MODE = "oneway"

# RF live entry
RF_SOURCE   = "close"
RF_PERIOD   = 20
RF_MULT     = 3.5
USE_RF_LIVE = True

# مؤشرات
RSI_LEN=14; ADX_LEN=14; ATR_LEN=14

# إدارة الربح/الخروج
TP1_PCT         = 0.40
TP1_CLOSE_FRAC  = 0.50
BREAKEVEN_AFTER = 0.30
TRAIL_ACTIVATE  = 1.20
ATR_MULT_TRAIL  = 1.6
TREND_TARGETS     = [0.50, 1.00, 1.80]
TREND_CLOSE_FRACS = [0.30, 0.30, 0.20]

# حرس السوق
MAX_SPREAD_BPS = 6.0
MIN_TRADE_USDT = 5.0

# EMA harvesting
EMA9_PARTIAL=0.20; EMA20_PARTIAL=0.35
IMPULSE_HARVEST_ATR=1.6

# فشل اختراق HH/LL
MIN_TREND_ADX_HOLD=25
FAIL_PROX_BPS=10.0
FAIL_CONFIRM_BARS=2
FAIL_DEFEND_PARTIAL=0.40
FAIL_CLOSE_AFTER_VOTES=2

# SMC
SMC_EQHL_LOOKBACK=30; SMC_OB_LOOKBACK=40; SMC_FVG_LOOKBACK=20
SMC_FVG_MAX_ATR=2.0; SMC_VOL_SPIKE=1.30

# فوبوناتشي
FIB_EXTS=[1.0,1.272,1.618]; FIB_GP_MIN=0.618; FIB_GP_MAX=0.65

# صبر تداول
PATIENT_TRADER_MODE=True
PATIENT_HOLD_BARS=8
PATIENT_HOLD_SECONDS=3600
NEVER_FULL_CLOSE_BEFORE_TP1=True

# طبقة طوارئ + Breakout
EMERGENCY_ENABLED=True
EMERGENCY_ADX_MIN=40; EMERGENCY_RSI_HOT=72; EMERGENCY_RSI_COLD=28
EMERGENCY_ATR_SPIKE=1.6; EMERGENCY_HARVEST=0.60
BREAKOUT_ENABLED=True
BREAKOUT_ADX=25; BREAKOUT_ATR_SPIKE=1.8; BREAKOUT_LOOKBACK=20

# شموع/فخاخ/ذيول
WICK_LONG_RATIO=0.6          # طول الذيل/المدى
BIG_RANGE_ATR =1.8           # مدى الشمعة/ATR
EXPLOSION_ADX_DELTA=5.0      # تسارع ADX
WICK_HARVEST_FRAC=0.45       # جني من ذيل طويل معنا
TRAP_NEAR_BPS=12.0           # قرب من EQH/EQL لاعتبار الفخ
TRUE_BREAK_MIN_CLOSE_BPS=6.0 # إقفال فوق/تحت المستوى ليُعتبر حقيقي

# Pacing
ADAPTIVE_PACING=True; BASE_SLEEP=10; NEAR_CLOSE_SLEEP=1; JUST_CLOSED_WIN=8

# Logging setup
def setup_file_logging():
    lg=logging.getLogger(); lg.setLevel(logging.INFO)
    if not any(isinstance(h, RotatingFileHandler) and getattr(h,"baseFilename","").endswith("bot.log") for h in lg.handlers):
        fh=RotatingFileHandler("bot.log", maxBytes=5_000_000, backupCount=7, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
        lg.addHandler(fh)
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    print(colored("🗂️ log rotation ready","cyan"))
setup_file_logging()

# =================== EXCHANGE ===================
def make_exchange():
    if not MODE_LIVE:
        class _Paper:
            def load_markets(self): self.markets={}
            def fetch_ohlcv(self, *a, **k): raise RuntimeError("Paper mode يحتاج مصدر بيانات حقيقي.")
            def fetch_ticker(self, *a, **k): raise RuntimeError("ticker unavailable in Paper stub")
            def fetch_order_book(self, *a, **k): return {"bids":[[0,0]], "asks":[[0,0]]}
            def fetch_balance(self, *a, **k): return {"total":{"USDT":100.0}, "free":{"USDT":100.0}}
            def set_leverage(self, *a, **k): pass
            def create_order(self, *a, **k): pass
            def fetch_positions(self, *a, **k): return []
        return _Paper()
    return ccxt.bingx({"apiKey":API_KEY,"secret":API_SECRET,"enableRateLimit":True,"timeout":20000,"options":{"defaultType":"swap"}})

ex=make_exchange()
MARKET={}; AMT_PREC=0; LOT_STEP=None; LOT_MIN=None

def load_market_specs():
    global MARKET, AMT_PREC, LOT_STEP, LOT_MIN
    try:
        ex.load_markets()
        MARKET = getattr(ex,"markets",{}).get(SYMBOL,{})
        AMT_PREC=int((MARKET.get("precision",{}) or {}).get("amount",0) or 0)
        lims=MARKET.get("limits",{}) or {}
        LOT_STEP=lims.get("amount",{}).get("step")
        LOT_MIN =lims.get("amount",{}).get("min")
        print(colored(f"📊 precision={AMT_PREC}, step={LOT_STEP}, min={LOT_MIN}","cyan"))
    except Exception as e:
        print(colored(f"⚠️ load_market_specs: {e}","yellow"))
def ensure_leverage_and_mode():
    try:
        if MODE_LIVE:
            try: ex.set_leverage(LEVERAGE, SYMBOL, params={"side":"BOTH"})
            except Exception as e: print(colored(f"⚠️ set_leverage: {e}","yellow"))
        print(colored(f"ℹ️ position mode: {BINGX_POSITION_MODE}","cyan"))
    except Exception as e:
        print(colored(f"⚠️ ensure mode: {e}","yellow"))
try:
    load_market_specs(); ensure_leverage_and_mode()
except Exception as e:
    print(colored(f"⚠️ exchange init: {e}","yellow"))

# =================== STATE ===================
STATE_FILE="bot_state.json"
compound_pnl=0.0
state={
    "open":False,"side":None,"entry":None,"qty":0.0,"pnl":0.0,"bars":0,
    "trail":None,"breakeven":None,"scale_ins":0,"scale_outs":0,
    "last_action":None,"action_reason":None,"highest_profit_pct":0.0,
    "tp1_done":False,"opened_at":None,"_fail_votes":0,"_smc":{},
    "diagnosed_after_open":False,
    # مراقبة فورية:
    "trap_prob":0.0,"explosion_flag":False,"wick_profile":{}
}
_state_lock=threading.Lock()

def get_state():
    with _state_lock: return dict(state)
def set_state(upd):
    with _state_lock: state.update(upd)
def save_state():
    import json, os
    try:
        with open(STATE_FILE+".tmp","w",encoding="utf-8") as f:
            json.dump({"state":state,"compound_pnl":compound_pnl,"ts":time.time()},f,ensure_ascii=False)
            f.flush(); os.fsync(f.fileno())
        os.replace(STATE_FILE+".tmp",STATE_FILE)
    except Exception as e: logging.error(f"save_state: {e}")
def load_state():
    import json, os
    global compound_pnl
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE,"r",encoding="utf-8") as f: d=json.load(f)
            set_state(d.get("state",{})); compound_pnl=float(d.get("compound_pnl",0.0))
            print(colored("✅ state restored","green"))
    except Exception as e: logging.error(f"load_state: {e}")

def _graceful_exit(*_): save_state(); sys.exit(0)
signal.signal(signal.SIGTERM,_graceful_exit); signal.signal(signal.SIGINT,_graceful_exit)

# =================== HELPERS ===================
def fmt(v,d=6,na="N/A"):
    try:
        if v is None or (isinstance(v,float) and (math.isnan(v) or math.isinf(v))): return na
        return f"{float(v):.{d}f}"
    except Exception: return na
def _interval_seconds(iv:str)->int:
    iv=iv.lower(); 
    return int(float(iv[:-1]))*60 if iv.endswith("m") else int(float(iv[:-1]))*3600
def time_to_candle_close(df): 
    tf=_interval_seconds(INTERVAL)
    if len(df)==0: return tf
    cur=int(df["time"].iloc[-1]); now=int(time.time()*1000)
    nxt=cur+tf*1000
    while nxt<=now: nxt+=tf*1000
    return int(max(0,nxt-now)/1000)
def compute_next_sleep(df):
    if not ADAPTIVE_PACING: return BASE_SLEEP
    left=time_to_candle_close(df); tf=_interval_seconds(INTERVAL)
    if left<=10 or (tf-left)<=JUST_CLOSED_WIN: return NEAR_CLOSE_SLEEP
    return BASE_SLEEP

# =================== DATA / QUANTS ===================
def fetch_ohlcv(limit=600):
    rows=ex.fetch_ohlcv(SYMBOL,timeframe=INTERVAL,limit=limit,params={"type":"swap"})
    return pd.DataFrame(rows,columns=["time","open","high","low","close","volume"])
def price_now():
    try:
        t=ex.fetch_ticker(SYMBOL)
        return t.get("last") or t.get("close")
    except Exception: return None
def balance_usdt():
    if not MODE_LIVE: return 100.0
    try:
        b=ex.fetch_balance(params={"type":"swap"})
        return b.get("total",{}).get("USDT") or b.get("free",{}).get("USDT")
    except Exception: return None
def orderbook_spread_bps():
    try:
        ob=ex.fetch_order_book(SYMBOL,limit=5); bid=ob["bids"][0][0] if ob["bids"] else None; ask=ob["asks"][0][0] if ob["asks"] else None
        if not (bid and ask): return None
        mid=(bid+ask)/2.0; return ((ask-bid)/mid)*10000.0
    except Exception: return None

# كَمّيّة
AMT_PREC=int((MARKET.get("precision",{}) or {}).get("amount") or 0)
LOT_STEP=(MARKET.get("limits",{}).get("amount",{}) or {}).get("step",None)
LOT_MIN=(MARKET.get("limits",{}).get("amount",{}) or {}).get("min",None)

from decimal import Decimal
def _round_amt(q):
    if q is None: return 0.0
    try:
        d=Decimal(str(q))
        if LOT_STEP and LOT_STEP>0:
            step=Decimal(str(LOT_STEP)); d=(d/step).to_integral_value(rounding=ROUND_DOWN)*step
        prec=int(AMT_PREC) if AMT_PREC and AMT_PREC>0 else 0
        d=d.quantize(Decimal(1).scaleb(-prec), rounding=ROUND_DOWN)
        if LOT_MIN and LOT_MIN>0 and d<Decimal(str(LOT_MIN)): return 0.0
        return float(d)
    except Exception: return max(0.0,float(q))
def safe_qty(q):
    q=_round_amt(q)
    if q<=0: logging.warning(f"qty invalid after normalize → {q}")
    return q
def compute_size(balance, price):
    eff=(balance or 0.0)+compound_pnl
    notional=eff*RISK_ALLOC*LEVERAGE
    if (price or 0)<=0: return 0.0
    q=notional/float(price); q=safe_qty(q)
    if float(price)*q<MIN_TRADE_USDT: return 0.0
    return q

# =================== INDICATORS ===================
def wilder_ema(s,n): return s.ewm(alpha=1/n,adjust=False).mean()
def compute_indicators(df: pd.DataFrame):
    if len(df)<max(ATR_LEN,RSI_LEN,ADX_LEN)+2:
        return {"rsi":50.0,"plus_di":0.0,"minus_di":0.0,"dx":0.0,"adx":0.0,"atr":0.0,
                "st":None,"st_dir":0,"ema9":None,"ema20":None,"ema9_slope":0.0}
    c,h,l=df["close"].astype(float), df["high"].astype(float), df["low"].astype(float)
    tr=pd.concat([(h-l).abs(), (h-c.shift(1)).abs(), (l-c.shift(1)).abs()],axis=1).max(axis=1)
    atr=wilder_ema(tr,ATR_LEN)

    delta=c.diff(); up=delta.clip(lower=0.0); dn=(-delta).clip(lower=0.0)
    rs=wilder_ema(up,RSI_LEN)/wilder_ema(dn,RSI_LEN).replace(0,1e-12)
    rsi=100-(100/(1+rs))

    up_move=h.diff(); down_move=l.shift(1)-l
    plus_dm=up_move.where((up_move>down_move)&(up_move>0),0.0)
    minus_dm=down_move.where((down_move>up_move)&(down_move>0),0.0)
    plus_di=100*(wilder_ema(plus_dm,ADX_LEN)/atr.replace(0,1e-12))
    minus_di=100*(wilder_ema(minus_dm,ADX_LEN)/atr.replace(0,1e-12))
    dx=(100*(plus_di-minus_di).abs()/(plus_di+minus_di).replace(0,1e-12)).fillna(0.0)
    adx=wilder_ema(dx,ADX_LEN)

    # Supertrend خفيف
    st_period=10; st_mult=3.0
    hl2=(h+l)/2.0; atr_st=wilder_ema(tr,st_period)
    upper=hl2+st_mult*atr_st; lower=hl2-st_mult*atr_st
    st=[float('nan')]; dirv=[0]
    for i in range(1,len(df)):
        prev_st=st[-1]; prev_dir=dirv[-1]; cur=float(c.iloc[i]); ub=float(upper.iloc[i]); lb=float(lower.iloc[i])
        if math.isnan(prev_st): st.append(lb if cur>lb else ub); dirv.append(1 if cur>lb else -1); continue
        if prev_dir==1:
            ub=min(ub,prev_st); st_now=lb if cur>ub else ub; dir_now=1 if cur>ub else -1
        else:
            lb=max(lb,prev_st); st_now=ub if cur<lb else lb; dir_now=-1 if cur<lb else 1
        st.append(st_now); dirv.append(dir_now)
    st_val=float(pd.Series(st[1:],index=df.index[1:]).reindex(df.index,method='pad').iloc[-1])
    st_dir=int(pd.Series(dirv[1:],index=df.index[1:]).reindex(df.index,method='pad').fillna(0).iloc[-1])

    ema9=c.ewm(span=9,adjust=False).mean(); ema20=c.ewm(span=20,adjust=False).mean()
    slope=0.0
    if len(ema9)>6:
        e_old=float(ema9.iloc[-6]); e_new=float(ema9.iloc[-1]); base=max(abs(e_old),1e-9); slope=(e_new-e_old)/base

    return {"rsi":float(rsi.iloc[-1]),"plus_di":float(plus_di.iloc[-1]),"minus_di":float(minus_di.iloc[-1]),
            "dx":float(dx.iloc[-1]),"adx":float(adx.iloc[-1]),"atr":float(atr.iloc[-1]),
            "st":st_val,"st_dir":st_dir,"ema9":float(ema9.iloc[-1]),
            "ema20":float(ema20.iloc[-1]),"ema9_slope":float(slope)}

# ============= RF =============
def _ema(s,n): return s.ewm(span=n,adjust=False).mean()
def _rng_size(src, qty, n):
    avrng=_ema((src-src.shift(1)).abs(), n); wper=(n*2)-1
    return _ema(avrng, wper) * qty
def _rng_filter(src, rsize):
    rf=[float(src.iloc[0])]
    for i in range(1,len(src)):
        prev=rf[-1]; x=float(src.iloc[i]); r=float(rsize.iloc[i]); cur=prev
        if x-r>prev: cur=x-r
        if x+r<prev: cur=x+r
        rf.append(cur)
    filt=pd.Series(rf,index=src.index,dtype="float64")
    return filt+rsize, filt-rsize, filt
def compute_rf_closed(df):
    if len(df) < RF_PERIOD+3:
        return {"time":int(df["time"].iloc[-1]),"price":float(df["close"].iloc[-1]),
                "long":False,"short":False,"filter":float(df["close"].iloc[-1]),
                "hi":float(df["close"].iloc[-1]),"lo":float(df["close"].iloc[-1]),"fdir":0}
    src=df[RF_SOURCE].astype(float)
    hi,lo,filt=_rng_filter(src,_rng_size(src,RF_MULT,RF_PERIOD))
    dfilt=filt-filt.shift(1); fdir=dfilt.apply(lambda x:1 if x>0 else (-1 if x<0 else 0)).ffill().fillna(0)
    upward=(fdir==1).astype(int); downward=(fdir==-1).astype(int)
    src_gt_f=(src>filt); src_lt_f=(src<filt)
    cond_prev=pd.Series(0,index=src.index)
    for i in range(1,len(src)):
        if bool(src_gt_f.iloc[i] and upward.iloc[i]>0): cond_prev.iloc[i]=1
        elif bool(src_lt_f.iloc[i] and downward.iloc[i]>0): cond_prev.iloc[i]=-1
        else: cond_prev.iloc[i]=cond_prev.iloc[i-1]
    longSignal=(src_gt_f & (cond_prev.shift(1)==-1))
    shortSignal=(src_lt_f & (cond_prev.shift(1)==1))
    i=len(src)-1
    return {"time":int(df["time"].iloc[i]),"price":float(df["close"].iloc[i]),
            "long":bool(longSignal.iloc[i]),"short":bool(shortSignal.iloc[i]),
            "filter":float(filt.iloc[i]),"hi":float(hi.iloc[i]),"lo":float(lo.iloc[i]),"fdir":float(fdir.iloc[i])}
def compute_rf_live(df):
    if len(df)<RF_PERIOD+3:
        return {"time":int(df["time"].iloc[-1]),"price":float(df["close"].iloc[-1]),
                "buy":False,"sell":False,"filter":float(df["close"].iloc[-1])}
    src=df[RF_SOURCE].astype(float)
    _,_,filt=_rng_filter(src,_rng_size(src,RF_MULT,RF_PERIOD))
    fdir=(filt-filt.shift(1)).apply(lambda x:1 if x>0 else (-1 if x<0 else 0)).ffill().fillna(0)
    cur=int(fdir.iloc[-1]); prev=int(fdir.iloc[-2]); px=float(src.iloc[-1]); f=float(filt.iloc[-1])
    buy  = (cur==1  and prev!=-1 and px>f)
    sell = (cur==-1 and prev!= 1 and px<f)
    return {"time":int(df["time"].iloc[-1]),"price":px,"buy":bool(buy),"sell":bool(sell),"filter":f}

# =================== SMC ===================
def _find_swings(df,left=2,right=2):
    if len(df)<left+right+3: return None,None
    h=df["high"].astype(float).values; l=df["low"].astype(float).values
    ph=[None]*len(df); pl=[None]*len(df)
    for i in range(left,len(df)-right):
        if all(h[i]>=h[j] for j in range(i-left, i+right+1)): ph[i]=h[i]
        if all(l[i]<=l[j] for j in range(i-left, i+right+1)): pl[i]=l[i]
    return ph,pl
def smc_snapshot(df, atr_now):
    try:
        d=df.iloc[:-1] if len(df)>=2 else df.copy()
        ph,pl=_find_swings(d,2,2)
        eqh=max([v for v in ph if v is not None], default=None)
        eql=min([v for v in pl if v is not None], default=None)
        ob=None
        for i in range(len(d)-2, max(len(d)-SMC_OB_LOOKBACK,1), -1):
            o=float(d["open"].iloc[i]); c=float(d["close"].iloc[i])
            h=float(d["high"].iloc[i]); l=float(d["low"].iloc[i])
            if abs(c-o) >= 1.2*max(atr_now,1e-12):
                side="bull" if c>o else "bear"; ob={"side":side,"bot":min(o,c),"top":max(o,c)}; break
        fvg=None
        for i in range(len(d)-3, max(len(d)-SMC_FVG_LOOKBACK,2), -1):
            phh=float(d["high"].iloc[i-1]); pll=float(d["low"].iloc[i-1])
            ch=float(d["high"].iloc[i]);  cl=float(d["low"].iloc[i])
            if cl>phh and (cl-phh)<=SMC_FVG_MAX_ATR*max(atr_now,1e-12):
                fvg={"type":"BULL_FVG","bottom":phh,"top":cl}; break
            if ch<pll and (pll-ch)<=SMC_FVG_MAX_ATR*max(atr_now,1e-12):
                fvg={"type":"BEAR_FVG","bottom":ch,"top":pll}; break
        return {"eqh":eqh,"eql":eql,"ob":ob,"fvg":fvg}
    except Exception:
        return {"eqh":None,"eql":None,"ob":None,"fvg":None}

# =================== Fibonacci ===================
def fib_targets_from_swings(entry_px, swing_low, swing_high, side):
    if not (entry_px and swing_low and swing_high) or swing_high<=swing_low: return []
    targets=[]
    if side=="long":
        base=swing_high-swing_low
        for ext in FIB_EXTS:
            t=swing_low+base*ext; pct=(t-entry_px)/entry_px*100.0
            if pct>0: targets.append(round(pct,2))
    else:
        base=swing_high-swing_low
        for ext in FIB_EXTS:
            t=swing_high-base*ext; pct=(entry_px-t)/entry_px*100.0
            if pct>0: targets.append(round(pct,2))
    return sorted(set(targets))
def in_golden_pocket(px,a,b):
    if a is None or b is None: return False
    lo,hi=(a,b) if a<b else (b,a)
    gp_min=lo+(hi-lo)*FIB_GP_MIN; gp_max=lo+(hi-lo)*FIB_GP_MAX
    return gp_min<=px<=gp_max

# =================== Candle Analytics / Traps ===================
def candle_stats(df, atr_now):
    """مقاييس الشمعة الحالية: الجسم/الذيول/المدى بالنسبة لـ ATR."""
    if len(df)==0: return {}
    o=float(df["open"].iloc[-1]); c=float(df["close"].iloc[-1])
    h=float(df["high"].iloc[-1]); l=float(df["low"].iloc[-1])
    rng=max(h-l,1e-12); body=abs(c-o); up_wick=h-max(o,c); lo_wick=min(o,c)-l
    return {
        "body":body, "range":rng, "up_wick":up_wick, "lo_wick":lo_wick,
        "body_ratio": body/rng, "up_ratio": up_wick/rng, "lo_ratio": lo_wick/rng,
        "range_atr": (rng/max(atr_now,1e-12))
    }

def near_bps(px,lvl,bps):
    try: return abs((px-lvl)/lvl)*10000.0 <= bps
    except Exception: return False

def trap_detector(df, smc_snap, ind):
    """يرصد فخاخ Liquidity: اختراق كاذب EQH/EQL بذيل طويل وإقفال داخلي + زخم غير داعم."""
    if not smc_snap: return {"bull_trap":0.0,"bear_trap":0.0}
    eqh, eql = smc_snap.get("eqh"), smc_snap.get("eql")
    st=candle_stats(df, ind.get("atr") or 0.0); px=float(df["close"].iloc[-1])
    bull=0.0; bear=0.0

    # Bear trap: كسر كاذب أسفل EQL بذيل سفلي طويل ثم إقفال فوقه مع ADX ضعيف/انعكاس RSI
    if eql:
        broke_down = df["low"].iloc[-1] < eql and px > eql and st.get("lo_ratio",0)>WICK_LONG_RATIO
        if broke_down and (ind.get("adx",0.0) < MIN_TREND_ADX_HOLD or ind.get("rsi",50.0) > 45):
            bear = min(1.0, 0.5 + 0.5*st["lo_ratio"])

    # Bull trap: اختراق كاذب أعلى EQH بذيل علوي طويل ثم إقفال تحته مع ADX ضعيف/انعكاس RSI
    if eqh:
        broke_up = df["high"].iloc[-1] > eqh and px < eqh and st.get("up_ratio",0)>WICK_LONG_RATIO
        if broke_up and (ind.get("adx",0.0) < MIN_TREND_ADX_HOLD or ind.get("rsi",50.0) < 55):
            bull = min(1.0, 0.5 + 0.5*st["up_ratio"])

    # قيمة احتمالية عامة (0..1)
    return {"bull_trap": round(bull,2), "bear_trap": round(bear,2)}

def true_break_confirmed(df, level, above=True):
    """يعدّ الكسر حقيقيًا فقط إذا الإقفال تجاوز المستوى بهامش منطقي."""
    if level is None or len(df)==0: return False
    close=float(df["close"].iloc[-1])
    if above: return ((close-level)/level)*10000.0 >= TRUE_BREAK_MIN_CLOSE_BPS
    else:     return ((level-close)/level)*10000.0 >= TRUE_BREAK_MIN_CLOSE_BPS

def explosion_detector(df, ind, prev_ind):
    """انفجار/انهيار محتمل حسب مدى الشمعة/ATR + تسارع ADX + جسم الشمعة."""
    st=candle_stats(df, ind.get("atr") or 0.0)
    adx_now=float(ind.get("adx") or 0.0); adx_prev=float(prev_ind.get("adx") or 0.0)
    adx_acc = adx_now - adx_prev
    big_range = st.get("range_atr",0.0) >= BIG_RANGE_ATR
    big_body  = st.get("body_ratio",0.0) >= 0.55
    accel     = adx_acc >= EXPLOSION_ADX_DELTA
    flag = bool(big_range and big_body and accel)
    return {"flag":flag, "range_atr":round(st.get("range_atr",0.0),2), "adx_delta":round(adx_acc,2), "body_ratio":round(st.get("body_ratio",0.0),2),
            "wick_up":round(st.get("up_ratio",0.0),2), "wick_lo":round(st.get("lo_ratio",0.0),2)}

# =================== ORDERS / MANAGEMENT ===================
def _position_params_for_open(side):
    return {"positionSide": "BOTH", "reduceOnly": False} if BINGX_POSITION_MODE=="oneway" else \
           {"positionSide": ("LONG" if side=="buy" else "SHORT"), "reduceOnly": False}
def _position_params_for_close():
    return {"positionSide": "BOTH", "reduceOnly": True} if BINGX_POSITION_MODE=="oneway" else \
           {"positionSide": ("LONG" if state.get("side")=="long" else "SHORT"), "reduceOnly": True}
def _read_exchange_position():
    try:
        poss=ex.fetch_positions(params={"type":"swap"})
        for p in poss:
            sym=(p.get("symbol") or p.get("info",{}).get("symbol") or "")
            if SYMBOL.split(":")[0] not in sym: continue
            qty=abs(float(p.get("contracts") or p.get("info",{}).get("positionAmt") or 0))
            if qty<=0: return 0.0,None,None
            entry=float(p.get("entryPrice") or p.get("info",{}).get("avgEntryPrice") or 0)
            side_raw=(p.get("side") or p.get("info",{}).get("positionSide") or "").lower()
            side="long" if ("long" in side_raw or float(p.get("cost",0))>0) else "short"
            return qty, side, entry
    except Exception as e:
        logging.error(f"_read_exchange_position: {e}")
    return 0.0,None,None

def open_market(side, qty, price, reason="OPEN"):
    if qty<=0: print(colored("❌ qty<=0 skip open","red")); return
    params=_position_params_for_open(side)
    if MODE_LIVE:
        try:
            try: ex.set_leverage(LEVERAGE, SYMBOL, params={"side":"BOTH"})
            except Exception as e: print(colored(f"⚠️ set_leverage: {e}","yellow"))
            ex.create_order(SYMBOL,"market",side,qty,None,params)
        except Exception as e:
            print(colored(f"❌ open: {e}","red")); logging.error(f"open_market: {e}"); return
    set_state({"open":True,"side":"long" if side=="buy" else "short","entry":price,"qty":qty,
               "pnl":0.0,"bars":0,"trail":None,"breakeven":None,"scale_ins":0,"scale_outs":0,
               "last_action":reason,"action_reason":reason,"highest_profit_pct":0.0,
               "tp1_done":False,"opened_at":time.time(),"_fail_votes":0,"diagnosed_after_open":False})
    print(colored(f"✅ OPEN {side.upper()} qty={fmt(qty,4)} @ {fmt(price)} • reason={reason}",
                  "green" if side=='buy' else "red"))
    save_state()

def _min_unit_qty():
    if LOT_MIN and LOT_MIN>0: return float(LOT_MIN)
    if LOT_STEP and LOT_STEP>0: return float(LOT_STEP)
    return 1.0
def close_partial(frac, reason):
    global compound_pnl
    if not state["open"] or state["qty"]<=0: return
    frac=max(0.0,min(1.0,float(frac)))
    qty_close=safe_qty(state["qty"]*frac)
    min_unit=_min_unit_qty()
    if state["qty"]-qty_close<min_unit: qty_close=safe_qty(max(0.0,state["qty"]-min_unit))
    if qty_close<min_unit:
        print(colored(f"⏸️ skip partial (amount={fmt(qty_close,4)} < min_unit={fmt(min_unit,4)})","yellow")); return
    side="sell" if state["side"]=="long" else "buy"
    if MODE_LIVE:
        try: ex.create_order(SYMBOL,"market",side,qty_close,None,_position_params_for_close())
        except Exception as e: print(colored(f"❌ partial close: {e}","red")); logging.error(f"close_partial: {e}"); return
    px=price_now() or state["entry"]
    pnl=(px-state["entry"])*qty_close*(1 if state["side"]=="long" else -1)
    compound_pnl+=pnl
    set_state({"qty":state["qty"]-qty_close,"scale_outs":state["scale_outs"]+1,"last_action":"SCALE_OUT","action_reason":reason})
    print(colored(f"🔻 PARTIAL {reason} closed={fmt(qty_close,4)} pnl={fmt(pnl)} rem={fmt(state['qty'],4)}","magenta"))
    save_state()

def reset_after_full_close(reason):
    set_state({"open":False,"side":None,"entry":None,"qty":0.0,"pnl":0.0,"bars":0,"trail":None,
               "breakeven":None,"scale_ins":0,"scale_outs":0,"last_action":"CLOSE",
               "action_reason":reason,"highest_profit_pct":0.0,"tp1_done":False,"opened_at":None,
               "_fail_votes":0,"_smc":{},"diagnosed_after_open":False,"trap_prob":0.0,"explosion_flag":False,"wick_profile":{}})
    save_state()
def close_market_strict(reason):
    global compound_pnl
    exch_qty, exch_side, exch_entry=_read_exchange_position()
    if exch_qty<=0:
        if state.get("open"): reset_after_full_close(f"{reason} (flat)")
        return
    side_to_close="sell" if (exch_side=="long") else "buy"
    qty_to_close=safe_qty(exch_qty)
    try:
        if MODE_LIVE:
            params=_position_params_for_close(); params["reduceOnly"]=True
            ex.create_order(SYMBOL,"market",side_to_close,qty_to_close,None,params)
        px=price_now() or state.get("entry")
        entry_px=state.get("entry") or exch_entry or px
        side=state.get("side") or exch_side or ("long" if side_to_close=="sell" else "short")
        pnl=(px-entry_px)*qty_to_close*(1 if side=="long" else -1)
        compound_pnl+=pnl
        print(colored(f"🔚 STRICT CLOSE {side} reason={reason} pnl={fmt(pnl)} total={fmt(compound_pnl)}","magenta"))
    except Exception as e: logging.error(f"strict close: {e}")
    reset_after_full_close(reason)

# ===== إدارة بعد الفتح =====
def trend_rr_pct(px):
    if not state["open"]: return 0.0
    e=state["entry"]; s=state["side"]
    if not (px and e and s): return 0.0
    return (px-e)/e*100.0*(1 if s=="long" else -1)

def ema_harvest(ind, px):
    if not state["open"] or state["qty"]<=0: return None
    side=state["side"]; ema9=ind.get("ema9"); ema20=ind.get("ema20")
    acted=None
    if ema9 is not None and ((side=="long" and px<ema9) or (side=="short" and px>ema9)):
        close_partial(EMA9_PARTIAL,"EMA9 touch"); acted="EMA9"
    if ema20 is not None and ((side=="long" and px<ema20) or (side=="short" and px>ema20)):
        close_partial(EMA20_PARTIAL,"EMA20 break"); acted=(acted or "")+"+EMA20"; state["breakeven"]=state.get("breakeven") or state.get("entry")
    return acted

def atr_trail(ind, px):
    if not state["open"] or state["qty"]<=0: return
    rr=trend_rr_pct(px); atr=ind.get("atr") or 0.0
    if rr<TRAIL_ACTIVATE or atr<=0: return
    if state["side"]=="long":
        new=px-atr*ATR_MULT_TRAIL; state["trail"]=max(state["trail"] or new, new)
        if state["breakeven"] is not None: state["trail"]=max(state["trail"], state["breakeven"])
        if px<state["trail"]: close_market_strict(f"TRAIL_ATR({ATR_MULT_TRAIL}x)")
    else:
        new=px+atr*ATR_MULT_TRAIL; state["trail"]=min(state["trail"] or new, new)
        if state["breakeven"] is not None: state["trail"]=min(state["trail"], state["breakeven"])
        if px>state["trail"]: close_market_strict(f"TRAIL_ATR({ATR_MULT_TRAIL}x)")

def ratchet_lock(px):
    if not state["open"] or state["qty"]<=0: return
    rr=trend_rr_pct(px)
    if rr>state["highest_profit_pct"]: state["highest_profit_pct"]=rr
    if state["highest_profit_pct"]>=20 and rr < state["highest_profit_pct"]*0.60:
        close_partial(0.5, f"Ratchet {state['highest_profit_pct']:.1f}%→{rr:.1f}%")
        state["highest_profit_pct"]=rr

def dynamic_tp_ladder(ind, smc_snap):
    px=price_now() or state.get("entry"); atr=ind.get("atr") or 0.0
    if not (state["open"] and px): return TREND_TARGETS, TREND_CLOSE_FRACS
    atr_pct=(atr/max(px,1e-9))*100.0 if atr>0 else 0.5
    base=[round(x,2) for x in [1.6*atr_pct, 2.6*atr_pct, 4.0*atr_pct]]
    fibs=[]
    if smc_snap.get("eql") and smc_snap.get("eqh"):
        fibs=fib_targets_from_swings(state["entry"], smc_snap["eql"], smc_snap["eqh"], state["side"])
    all_tps=sorted(set(base+fibs+TREND_TARGETS))
    fracs=[0.25,0.30,0.45][:len(all_tps)]
    return all_tps[:4], fracs[:len(all_tps[:4])]

def breakout_failure_guard(df, px, ind, smc_snap):
    if not (state["open"] and px and smc_snap): return
    side=state["side"]; hh=smc_snap.get("eqh"); ll=smc_snap.get("eql")
    if side=="long" and hh and near_bps(px,hh,FAIL_PROX_BPS):
        if ind.get("adx",0.0)<MIN_TREND_ADX_HOLD or not true_break_confirmed(df, hh, above=True):
            state["_fail_votes"]+=1
            if state["_fail_votes"]==1:
                close_partial(FAIL_DEFEND_PARTIAL,"HH failure defend"); state["breakeven"]=state.get("breakeven") or state.get("entry")
            if state["_fail_votes"]>=FAIL_CLOSE_AFTER_VOTES:
                allow=True
                if PATIENT_TRADER_MODE:
                    elapsed=time.time()-(state.get("opened_at") or time.time())
                    allow=(state["bars"]>=PATIENT_HOLD_BARS) or (elapsed>=PATIENT_HOLD_SECONDS)
                if NEVER_FULL_CLOSE_BEFORE_TP1 and not state.get("tp1_done"): allow=False
                if allow: close_market_strict("HH_BREAK_FAILURE_CONFIRMED")
    if side=="short" and ll and near_bps(px,ll,FAIL_PROX_BPS):
        if ind.get("adx",0.0)<MIN_TREND_ADX_HOLD or not true_break_confirmed(df, ll, above=False):
            state["_fail_votes"]+=1
            if state["_fail_votes"]==1:
                close_partial(FAIL_DEFEND_PARTIAL,"LL failure defend"); state["breakeven"]=state.get("breakeven") or state.get("entry")
            if state["_fail_votes"]>=FAIL_CLOSE_AFTER_VOTES:
                allow=True
                if PATIENT_TRADER_MODE:
                    elapsed=time.time()-(state.get("opened_at") or time.time())
                    allow=(state["bars"]>=PATIENT_HOLD_BARS) or (elapsed>=PATIENT_HOLD_SECONDS)
                if NEVER_FULL_CLOSE_BEFORE_TP1 and not state.get("tp1_done"): allow=False
                if allow: close_market_strict("LL_BREAK_FAILURE_CONFIRMED")

def wick_harvest(df, ind):
    """جني أرباح من ذيل طويل في صالحنا (يظهر فخ سيولة أو امتصاص)."""
    if not state["open"] or state["qty"]<=0: return
    cs=candle_stats(df, ind.get("atr") or 0.0); side=state["side"]
    if side=="long" and cs.get("up_ratio",0)>=WICK_LONG_RATIO and cs.get("range_atr",0)>=1.2:
        close_partial(WICK_HARVEST_FRAC, "Wick Harvest (upper)")
    if side=="short" and cs.get("lo_ratio",0)>=WICK_LONG_RATIO and cs.get("range_atr",0)>=1.2:
        close_partial(WICK_HARVEST_FRAC, "Wick Harvest (lower)")

def emergency_layer(ind, prev_ind, px):
    if not (EMERGENCY_ENABLED and state["open"]): return
    adx=float(ind.get("adx") or 0.0); rsi=float(ind.get("rsi") or 50.0)
    atr_now=float(ind.get("atr") or 0.0); atr_prev=float(prev_ind.get("atr") or atr_now)
    if atr_prev>0 and atr_now>=atr_prev*EMERGENCY_ATR_SPIKE and adx>=EMERGENCY_ADX_MIN:
        if (state["side"]=="long" and rsi<=EMERGENCY_RSI_COLD) or (state["side"]=="short" and rsi>=EMERGENCY_RSI_HOT):
            close_market_strict("EMERGENCY opposite thrust")
        else:
            close_partial(EMERGENCY_HARVEST,"EMERGENCY harvest"); state["breakeven"]=state.get("breakeven") or state.get("entry")

def breakout_signal(df, ind, prev_ind):
    if not BREAKOUT_ENABLED or len(df)<BREAKOUT_LOOKBACK+2: return None
    price=float(df["close"].iloc[-1]); adx=float(ind.get("adx") or 0.0)
    atr_now=float(ind.get("atr") or 0.0); atr_prev=float(prev_ind.get("atr") or atr_now)
    if not (adx>=BREAKOUT_ADX and atr_prev>0 and atr_now>=atr_prev*BREAKOUT_ATR_SPIKE): return None
    highs=df["high"].iloc[-BREAKOUT_LOOKBACK:-1].astype(float); lows=df["low"].iloc[-BREAKOUT_LOOKBACK:-1].astype(float)
    if len(highs) and price>highs.max(): return "BUY"
    if len(lows)  and price<lows.min():  return "SELL"
    return None

# =================== HUD ===================
def snapshot(bal, ind, spread_bps, df, smc, traps, expl):
    left=time_to_candle_close(df)
    cs=candle_stats(df, ind.get("atr") or 0.0)
    print(colored("—"*108,"cyan"))
    print(colored(f"📊 {SYMBOL} {INTERVAL} • {'LIVE' if MODE_LIVE else 'PAPER'} • {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC","cyan"))
    print(f"💲 Price {fmt(df['close'].iloc[-1])} | ATR={fmt(ind['atr'])} | spread_bps={fmt(spread_bps,2)} | close_in≈{left}s")
    print(f"🧮 RSI={fmt(ind['rsi'])}  ADX={fmt(ind['adx'])}  +DI={fmt(ind['plus_di'])}  -DI={fmt(ind['minus_di'])}  ST_dir={ind['st_dir']}")
    print(f"🕯️ body={fmt(cs.get('body_ratio',0.0),2)}  upW={fmt(cs.get('up_ratio',0.0),2)}  loW={fmt(cs.get('lo_ratio',0.0),2)}  range/ATR={fmt(cs.get('range_atr',0.0),2)}")
    print(f"🎭 traps → bull={traps['bull_trap']}  bear={traps['bear_trap']}   💥 explosion={expl['flag']} (ΔADX={expl['adx_delta']}, r/ATR={expl['range_atr']})")
    eff=(bal or 0.0)+compound_pnl
    print(f"💰 Balance={fmt(bal,2)}  Risk={int(RISK_ALLOC*100)}%×{LEVERAGE}x  CompoundPnL={fmt(compound_pnl)}  Eq={fmt(eff)}")
    if state["open"]:
        lamp='🟩 LONG' if state['side']=='long' else '🟥 SHORT'
        print(f"{lamp} Entry={fmt(state['entry'])} Qty={fmt(state['qty'],4)} Bars={state['bars']} Trail={fmt(state['trail'])} PnL≈{fmt(state['pnl'])}")
        print(f"TP1={state['tp1_done']}  BE={fmt(state['breakeven'])}  HighRR={fmt(state['highest_profit_pct'],2)}%  FailVotes={state['_fail_votes']}")
    else:
        print("⚪ FLAT")
    if smc:
        print(f"SMC: EQH={fmt(smc.get('eqh'))}  EQL={fmt(smc.get('eql'))}  OB={smc.get('ob')}  FVG={smc.get('fvg')}")
    print(colored("—"*108,"cyan"))

# =================== LOOP ===================
_last_bar_ts=None
def update_bar_counters(df):
    global _last_bar_ts
    if len(df)==0: return False
    ts=int(df["time"].iloc[-1])
    if _last_bar_ts is None: _last_bar_ts=ts; return False
    if ts!=_last_bar_ts:
        _last_bar_ts=ts
        if state["open"]: state["bars"]+=1
        return True
    return False

def post_open_diagnostics(df, ind, smc, traps, expl):
    """تشغيل نظام الفحوص بعد الفتح مباشرة لتفادي الفخاخ القريبة."""
    if state["diagnosed_after_open"]: return
    msgs=[]
    if smc.get("fvg"): msgs.append("FVG near — راقب ملء الفجوة")
    if smc.get("ob"):  msgs.append(f"OB {smc['ob']['side']} zone — احتمال ارتداد")
    if smc.get("eqh"): msgs.append("EQH overhead — احذر فخ قمة")
    if smc.get("eql"): msgs.append("EQL below — احذر فخ قاع")
    if traps["bull_trap"]>0.5 or traps["bear_trap"]>0.5: msgs.append("Trap probability HIGH — tighten")
    if expl["flag"]: msgs.append("Explosion momentum — harvest بسرعة لو انعكس")
    if msgs:
        print(colored("🧪 POST-OPEN CHECKS: " + " | ".join(msgs), "yellow"))
    set_state({"diagnosed_after_open":True})

def trade_loop():
    loop=0
    while True:
        try:
            bal=balance_usdt(); px=price_now(); df=fetch_ohlcv()
            new_bar=update_bar_counters(df)

            ind=compute_indicators(df); prev_ind=compute_indicators(df.iloc[:-1]) if len(df)>=2 else ind
            rf_live=compute_rf_live(df); rf_closed=compute_rf_closed(df.iloc[:-1] if len(df)>=2 else df)
            smc=smc_snapshot(df, ind.get("atr") or 0.0)

            # تحليلات الشموع/الفخاخ/الانفجار
            traps=trap_detector(df, smc, ind)
            expl =explosion_detector(df, ind, prev_ind)

            # Spread guard
            spread_bps=orderbook_spread_bps()
            if spread_bps is not None and spread_bps>MAX_SPREAD_BPS:
                snapshot(bal, ind, spread_bps, df, smc, traps, expl)
                time.sleep(compute_next_sleep(df)); continue

            # تحديث PnL
            if state["open"] and px:
                state["pnl"]=(px-state["entry"])*state["qty"] if state["side"]=="long" else (state["entry"]-px)*state["qty"]

            # طوارئ
            if state["open"]: emergency_layer(ind, prev_ind, px or rf_live["price"])

            # انفجار/انهيار: لو فاضي، استخدم الإشارة للدخول مبكّرًا
            if not state["open"] and expl["flag"]:
                side="buy" if ind.get("st_dir",0)==1 or ind.get("plus_di",0)>ind.get("minus_di",0) else "sell"
                qty=compute_size(bal, px or rf_live["price"])
                if qty>0: open_market(side, qty, px or rf_live["price"], reason="EXPLOSION")

            # Breakout (اختياري)
            if not state["open"]:
                br=breakout_signal(df, ind, prev_ind)
                if br:
                    qty=compute_size(bal, px or rf_live["price"])
                    if qty>0: open_market("buy" if br=="BUY" else "sell", qty, px or rf_live["price"], reason="BREAKOUT")

            # RF live entry
            if not state["open"] and USE_RF_LIVE:
                sig="buy" if rf_live["buy"] else ("sell" if rf_live["sell"] else None)
                if sig:
                    qty=compute_size(bal, px or rf_live["price"])
                    if qty>0: open_market(sig, qty, px or rf_live["price"], reason="RF_LIVE")

            # بعد الفتح: شغّل المنظومة كلها
            if state["open"]:
                post_open_diagnostics(df, ind, smc, traps, expl)

                rr=trend_rr_pct(px or rf_live["price"])
                if not state.get("tp1_done") and rr>=TP1_PCT:
                    close_partial(TP1_CLOSE_FRAC, f"TP1@{TP1_PCT:.2f}%"); state["tp1_done"]=True
                    if rr>=BREAKEVEN_AFTER: state["breakeven"]=state.get("entry")

                # سلم ديناميكي + فوبو
                tps, fracs=dynamic_tp_ladder(ind, smc)
                for i, tgt in enumerate(tps):
                    if rr>=tgt and state["qty"]>0:
                        close_partial(fracs[i], f"TP_dyn@{tgt:.2f}%"); tps[i]=10_000.0

                # Golden Pocket حماية
                if smc.get("eql") and smc.get("eqh"):
                    a,b=(smc["eql"], smc["eqh"]) if state["side"]=="long" else (smc["eqh"], smc["eql"])
                    if px and in_golden_pocket(px, a, b):
                        close_partial(0.25, "Golden Pocket protect"); state["breakeven"]=state.get("breakeven") or state.get("entry")

                # Wick / EMA / ATR / Ratchet
                wick_harvest(df, ind)
                ema_harvest(ind, px or rf_live["price"])
                atr_trail(ind, px or rf_live["price"])
                ratchet_lock(px or rf_live["price"])

                # فشل اختراق القمم/القيعان
                breakout_failure_guard(df, px or rf_live["price"], ind, smc)

            snapshot(bal, ind, spread_bps, df, smc, traps, expl)
            if loop % 5 == 0: save_state()
            loop+=1
            time.sleep(compute_next_sleep(df))

        except Exception as e:
            print(colored(f"❌ loop error: {e}\n{traceback.format_exc()}","red"))
            logging.error(f"trade_loop error: {e}\n{traceback.format_exc()}")
            time.sleep(BASE_SLEEP)

# =================== KEEPALIVE & API ===================
def keepalive_loop():
    url=(SELF_URL or "").strip().rstrip("/")
    if not url:
        print(colored("⛔ keepalive disabled (no SELF_URL)","yellow")); return
    sess=requests.Session(); sess.headers.update({"User-Agent":"rf-pro-bot/keepalive"})
    print(colored(f"KEEPALIVE → {url} every 50s","cyan"))
    while True:
        try: sess.get(url, timeout=8)
        except Exception: pass
        time.sleep(50)

from flask import Flask, jsonify
app=Flask(__name__)
import logging as flask_logging
flask_logging.getLogger('werkzeug').setLevel(flask_logging.ERROR)

@app.route("/")
def home():
    mode='LIVE' if MODE_LIVE else 'PAPER'
    return f"✅ RF Pro++ Bot — {SYMBOL} {INTERVAL} — {mode} — TRAPS+WICKS — STRICT CLOSE"

@app.route("/metrics")
def metrics():
    s=get_state()
    return jsonify({
        "symbol":SYMBOL, "interval":INTERVAL, "mode":"live" if MODE_LIVE else "paper",
        "leverage":LEVERAGE, "risk_alloc":RISK_ALLOC, "price":price_now(),
        "position":s, "compound_pnl":compound_pnl, "time":datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "smc":s.get("_smc", {}), "trap_prob":s.get("trap_prob",0.0)
    })

@app.route("/health")
def health():
    s=get_state()
    return jsonify({"ok":True, "mode":"live" if MODE_LIVE else "paper", "open":s["open"], "side":s["side"],
                    "qty":s["qty"], "compound_pnl":compound_pnl, "tp1_done":s.get("tp1_done",False),
                    "trail":s.get("trail"), "timestamp":datetime.utcnow().isoformat()}), 200

# =================== BOOT ===================
if __name__=="__main__":
    print(colored(f"MODE: {'LIVE' if MODE_LIVE else 'PAPER'} • {SYMBOL} • {INTERVAL}","yellow"))
    print(colored(f"RISK: {int(RISK_ALLOC*100)}% × {LEVERAGE}x • RF_LIVE={USE_RF_LIVE}","yellow"))
    load_state()
    threading.Thread(target=trade_loop, daemon=True).start()
    threading.Thread(target=keepalive_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)
