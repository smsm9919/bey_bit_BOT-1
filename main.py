# -*- coding: utf-8 -*-
# BYBIT — SOL/USDT:USDT Perp — RF ENTRY + SMART COUNCIL (Patience/SMC/EVX2)
import os, time, math, random, signal, sys, logging, traceback
from datetime import datetime
from logging.handlers import RotatingFileHandler
from decimal import Decimal, ROUND_DOWN, InvalidOperation
import pandas as pd
import ccxt
from flask import Flask, jsonify

# ========== ENV ==========
API_KEY   = os.getenv("BYBIT_API_KEY","")
API_SECRET= os.getenv("BYBIT_API_SECRET","")
MODE_LIVE = bool(API_KEY and API_SECRET)

SELF_URL = os.getenv("SELF_URL","")
PORT = int(os.getenv("PORT", 5000))

SYMBOL     = os.getenv("SYMBOL","SOL/USDT:USDT")   # Bybit linear perp
INTERVAL   = os.getenv("INTERVAL","15m")
LEVERAGE   = int(os.getenv("LEVERAGE", 10))
RISK_ALLOC = float(os.getenv("RISK_ALLOC", 0.60))
POSITION_MODE = os.getenv("POSITION_MODE","oneway") # oneway/hedge

# RF settings
RF_SOURCE="close"; RF_PERIOD=int(os.getenv("RF_PERIOD",20))
RF_MULT=float(os.getenv("RF_MULT",3.5))
RF_LIVE_ONLY = os.getenv("RF_LIVE_ONLY","True").lower()=="true"
RF_HYST_BPS  = float(os.getenv("RF_HYST_BPS",6.0))

# Indicators
RSI_LEN=14; ADX_LEN=14; ATR_LEN=14
MAX_SPREAD_BPS=float(os.getenv("MAX_SPREAD_BPS", 8.0))

# Dynamic TP / trail
TP1_PCT_BASE=0.40; TP1_CLOSE_FRAC=0.50
BREAKEVEN_AFTER=0.30; TRAIL_ACTIVATE_PCT=1.20; ATR_TRAIL_MULT=1.6

# Patience / Council thresholds
MIN_HOLD_BARS=int(os.getenv("MIN_HOLD_BARS",2))
BRK_CONFIRM_CLOSES=int(os.getenv("BRK_CONFIRM_CLOSES",1))
BRK_MIN_VRATIO=float(os.getenv("BRK_MIN_VRATIO",1.3))
BRK_MIN_ADX=float(os.getenv("BRK_MIN_ADX",22))
RETEST_TIMEOUT_BARS=int(os.getenv("RETEST_TIMEOUT_BARS",6))
DOJI_IGNORE_IN_DEFENSE=os.getenv("DOJI_IGNORE_IN_DEFENSE","1")=="1"
DOJI_CONFIRM_BARS=int(os.getenv("DOJI_CONFIRM_BARS",1))
TRAP_RSI_OVERBOUGHT=float(os.getenv("TRAP_RSI_OVERBOUGHT",68))
TRAP_RSI_OVERSOLD=float(os.getenv("TRAP_RSI_OVERSOLD",32))
TH_TIGHT=float(os.getenv("TH_TIGHT",1.2)); TH_PART=float(os.getenv("TH_PARTIAL",1.9)); TH_FULL=float(os.getenv("TH_FULL",2.8))
RCS_SMC_BPS=float(os.getenv("RCS_SMC_BPS",12.0))

# EVX2
EVX_V_SPIKE=float(os.getenv("EVX_V_SPIKE",2.0))
EVX_ATR_REACT=float(os.getenv("EVX_ATR_REACT",1.4))
EVX_ZSCORE=float(os.getenv("EVX_ZSCORE",1.8))

# Opposite RF defense
OPP_RF_VOTES_NEEDED=int(os.getenv("OPP_RF_VOTES_NEEDED",3))
OPP_RF_MIN_ADX=float(os.getenv("OPP_RF_MIN_ADX",25))
OPP_RF_MIN_HYST_BPS=float(os.getenv("OPP_RF_MIN_HYST_BPS",8.0))

FINAL_CHUNK_QTY=float(os.getenv("FINAL_CHUNK_QTY", 0.2))  # SOL
RESIDUAL_MIN_QTY=float(os.getenv("RESIDUAL_MIN_QTY", 0.05))
CLOSE_RETRY_ATTEMPTS=6; CLOSE_VERIFY_WAIT_S=2.0
BASE_SLEEP=5; NEAR_CLOSE_S=1

# ========== LOG ==========
root = logging.getLogger(); root.setLevel(logging.INFO)
if not any(isinstance(h,RotatingFileHandler) for h in root.handlers):
    fh=RotatingFileHandler("bot.log",maxBytes=5_000_000,backupCount=7,encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s")); root.addHandler(fh)
logging.getLogger("werkzeug").setLevel(logging.ERROR)

def colored(t,*a,**k): return t

# ========== EXCHANGE ==========
def make_ex():
    return ccxt.bybit({
        "apiKey": API_KEY, "secret": API_SECRET,
        "enableRateLimit": True, "timeout": 20000,
        "options": {
            "defaultType":"swap",     # linear USDT perp
            "hedgeMode": (POSITION_MODE=="hedge"),
        }
    })
ex = make_ex()
MARKET={}; AMT_PREC=0; LOT_STEP=None; LOT_MIN=None

def load_market_specs():
    global MARKET, AMT_PREC, LOT_STEP, LOT_MIN
    try:
        ex.load_markets()
        MARKET = ex.markets.get(SYMBOL,{})
        AMT_PREC = int((MARKET.get("precision",{}) or {}).get("amount",0) or 0)
        LOT_STEP = (MARKET.get("limits",{}) or {}).get("amount",{}).get("step",None)
        LOT_MIN  = (MARKET.get("limits",{}) or {}).get("amount",{}).get("min",None)
        print(colored(f"🔧 {SYMBOL} prec={AMT_PREC} step={LOT_STEP} min={LOT_MIN}","cyan"))
    except Exception as e:
        print(colored(f"⚠️ load_market_specs: {e}","yellow"))

def ensure_leverage_mode():
    try:
        try:
            ex.set_leverage(LEVERAGE, SYMBOL, params={"buyLeverage":LEVERAGE,"sellLeverage":LEVERAGE})
        except Exception as e:
            print(colored(f"⚠️ set_leverage warn: {e}","yellow"))
        try:
            ex.set_position_mode(POSITION_MODE=="hedge", SYMBOL)  # True=hedge
        except Exception as e:
            print(colored(f"⚠️ set_position_mode warn: {e}","yellow"))
        print(colored(f"📌 mode={POSITION_MODE} lev={LEVERAGE}x","cyan"))
    except Exception as e:
        print(colored(f"⚠️ ensure_leverage_mode: {e}","yellow"))

try:
    load_market_specs(); ensure_leverage_mode()
except Exception as e:
    print(colored(f"⚠️ ex init: {e}","yellow"))

# ========== HELPERS ==========
STATE = {
    "open": False, "side": None, "entry": None, "qty": 0.0, "pnl": 0.0,
    "bars": 0, "trail": None, "breakeven": None, "tp1_done": False,
    "highest_profit_pct": 0.0, "profit_targets_achieved": 0, "opp_votes": 0,
    "fusion_score": 0.0, "trap_risk": 0.0
}
compound_pnl = 0.0; wait_for_next_signal_side=None

def fmt(v,d=6,na="—"):
    try:
        if v is None or (isinstance(v,float) and (math.isnan(v) or math.isinf(v))): return na
        return f"{float(v):.{d}f}"
    except Exception: return na

def _round_amt(q):
    if q is None: return 0.0
    try:
        d=Decimal(str(q))
        if LOT_STEP and isinstance(LOT_STEP,(int,float)) and LOT_STEP>0:
            step=Decimal(str(LOT_STEP))
            d=(d/step).to_integral_value(rounding=ROUND_DOWN)*step
        prec=int(AMT_PREC) if AMT_PREC and AMT_PREC>=0 else 0
        d=d.quantize(Decimal(1).scaleb(-prec), rounding=ROUND_DOWN)
        if LOT_MIN and isinstance(LOT_MIN,(int,float)) and LOT_MIN>0 and d<Decimal(str(LOT_MIN)): return 0.0
        return float(d)
    except (InvalidOperation, ValueError, TypeError):
        return max(0.0, float(q))

def safe_qty(q): 
    q=_round_amt(q)
    if q<=0: print(colored(f"⚠️ qty invalid → {q}","yellow"))
    return q

def with_retry(fn, tries=3, base_wait=0.4):
    for i in range(tries):
        try: return fn()
        except Exception:
            if i==tries-1: raise
            time.sleep(base_wait*(2**i) + random.random()*0.25)

def fetch_ohlcv(limit=600):
    rows=with_retry(lambda: ex.fetch_ohlcv(SYMBOL, timeframe=INTERVAL, limit=limit, params={"category":"linear"}))
    return pd.DataFrame(rows, columns=["time","open","high","low","close","volume"])

def price_now():
    try: t=with_retry(lambda: ex.fetch_ticker(SYMBOL)); return t.get("last") or t.get("close")
    except Exception: return None

def balance_usdt():
    if not MODE_LIVE: return 100.0
    try:
        b=with_retry(lambda: ex.fetch_balance(params={"type":"swap"}))
        return b.get("total",{}).get("USDT") or b.get("free",{}).get("USDT")
    except Exception: return None

def orderbook_spread_bps():
    try:
        ob=with_retry(lambda: ex.fetch_order_book(SYMBOL, limit=5))
        bid=ob["bids"][0][0] if ob["bids"] else None
        ask=ob["asks"][0][0] if ob["asks"] else None
        if not (bid and ask): return None
        mid=(bid+ask)/2.0; return ((ask-bid)/mid)*10000.0
    except Exception: return None

def _interval_seconds(iv:str)->int:
    iv=(iv or "").lower().strip()
    if iv.endswith("m"): return int(float(iv[:-1]))*60
    if iv.endswith("h"): return int(float(iv[:-1]))*3600
    if iv.endswith("d"): return int(float(iv[:-1]))*86400
    return 15*60

def time_to_candle_close(df: pd.DataFrame)->int:
    tf=_interval_seconds(INTERVAL)
    if len(df)==0: return tf
    cur=int(df["time"].iloc[-1]); now=int(time.time()*1000); nxt=cur+tf*1000
    while nxt<=now: nxt+=tf*1000
    return int(max(0,nxt-now)/1000)

# ========== INDICATORS ==========
def wilder_ema(s, n): return s.ewm(alpha=1/n, adjust=False).mean()
def compute_indicators(df: pd.DataFrame):
    if len(df)<max(ATR_LEN,RSI_LEN,ADX_LEN)+2:
        return {"rsi":50.0,"plus_di":0.0,"minus_di":0.0,"dx":0.0,"adx":0.0,"atr":0.0}
    c=df["close"].astype(float); h=df["high"].astype(float); l=df["low"].astype(float)
    tr=pd.concat([(h-l).abs(), (h-c.shift(1)).abs(), (l-c.shift(1)).abs()],axis=1).max(axis=1)
    atr=wilder_ema(tr, ATR_LEN)
    delta=c.diff(); up=delta.clip(lower=0.0); dn=(-delta).clip(lower=0.0)
    rs=wilder_ema(up,RSI_LEN)/wilder_ema(dn,RSI_LEN).replace(0,1e-12)
    rsi=100-(100/(1+rs))
    up_move=h.diff(); down_move=l.shift(1)-l
    plus_dm=up_move.where((up_move>down_move)&(up_move>0),0.0)
    minus_dm=down_move.where((down_move>up_move)&(down_move>0),0.0)
    plus_di=100*(wilder_ema(plus_dm,ADX_LEN)/atr.replace(0,1e-12))
    minus_di=100*(wilder_ema(minus_dm,ADX_LEN)/atr.replace(0,1e-12))
    dx=(100*(plus_di-minus_di).abs()/(plus_di+minus_di).replace(0,1e-12)).fillna(0.0)
    adx=wilder_ema(dx, ADX_LEN)
    i=len(df)-1
    return {"rsi":float(rsi.iloc[i]),"plus_di":float(plus_di.iloc[i]),"minus_di":float(minus_di.iloc[i]),"dx":float(dx.iloc[i]),"adx":float(adx.iloc[i]),"atr":float(atr.iloc[i])}

# ========== RANGE FILTER ==========
def _ema(s,n): return s.ewm(span=n,adjust=False).mean()
def _rng_size(src, qty, n):
    avrng=_ema((src-src.shift(1)).abs(), n); wper=(n*2)-1
    return _ema(avrng, wper)*qty
def _rng_filter(src, rsize):
    rf=[float(src.iloc[0])]
    for i in range(1,len(src)):
        prev=rf[-1]; x=float(src.iloc[i]); r=float(rsize.iloc[i]); cur=prev
        if x-r>prev: cur=x-r
        if x+r<prev: cur=x+r
        rf.append(cur)
    filt=pd.Series(rf, index=src.index, dtype="float64")
    return filt+rsize, filt-rsize, filt
def rf_signal_live(df):
    if len(df)<RF_PERIOD+3:
        price=float(df["close"].iloc[-1]) if len(df) else 0.0
        return {"time":int(df["time"].iloc[-1]) if len(df) else int(time.time()*1000),"price":price,"long":False,"short":False,"filter":price,"hi":price,"lo":price}
    src=df[RF_SOURCE].astype(float)
    hi,lo,filt=_rng_filter(src,_rng_size(src,RF_MULT,RF_PERIOD))
    def _bps(a,b): 
        try: return abs((a-b)/b)*10000.0
        except Exception: return 0.0
    p_now=float(src.iloc[-1]); p_prev=float(src.iloc[-2])
    f_now=float(filt.iloc[-1]); f_prev=float(filt.iloc[-2])
    long_flip=(p_prev<=f_prev and p_now>f_now and _bps(p_now,f_now)>=RF_HYST_BPS)
    short_flip=(p_prev>=f_prev and p_now<f_now and _bps(p_now,f_now)>=RF_HYST_BPS)
    return {"time":int(df["time"].iloc[-1]),"price":p_now,"long":bool(long_flip),"short":bool(short_flip),"filter":f_now,"hi":float(hi.iloc[-1]),"lo":float(lo.iloc[-1])}

# ========== SIMPLE SMC ==========
def _find_swings(df,left=2,right=2):
    if len(df)<left+right+3: return None,None
    h=df["high"].astype(float).values; l=df["low"].astype(float).values
    ph=[None]*len(df); pl=[None]*len(df)
    for i in range(left,len(df)-right):
        if all(h[i]>=h[j] for j in range(i-left,i+right+1)): ph[i]=h[i]
        if all(l[i]<=l[j] for j in range(i-left,i+right+1)): pl[i]=l[i]
    return ph,pl
def detect_smc_levels(df):
    try:
        ph,pl=_find_swings(df,2,2)
        def _eq(vals,is_high=True):
            res=[]; tol_pct=0.05
            for i,p in enumerate(vals):
                if p is None: continue
                tol=p*tol_pct/100.0
                neighbors=[vals[j] for j in range(max(0,i-10),min(len(vals),i+10)) if vals[j] is not None and abs(vals[j]-p)<=tol]
                if len(neighbors)>=2: res.append(max(neighbors) if is_high else min(neighbors))
            return (max(res) if is_high else min(res)) if res else None
        eqh=_eq(ph,True); eql=_eq(pl,False)
        ob=None
        for i in range(len(df)-2,max(len(df)-40,1),-1):
            o=float(df["open"].iloc[i]); c=float(df["close"].iloc[i])
            h=float(df["high"].iloc[i]); l=float(df["low"].iloc[i])
            rng=max(h-l,1e-12); body=abs(c-o); up=h-max(o,c); dn=min(o,c)-l
            if body>=0.6*rng and (up/rng)<=0.2 and (dn/rng)<=0.2:
                ob={"side":"bull" if c>o else "bear","bot":min(o,c),"top":max(o,c),"time":int(df["time"].iloc[i])}; break
        fvg=None
        for i in range(len(df)-3,max(len(df)-20,2),-1):
            ph=float(df["high"].iloc[i-1]); pl=float(df["low"].iloc[i-1])
            ch=float(df["high"].iloc[i]);  cl=float(df["low"].iloc[i])
            if cl>ph: fvg={"type":"BULL_FVG","bottom":ph,"top":cl}; break
            if ch<pl: fvg={"type":"BEAR_FVG","bottom":ch,"top":pl}; break
        return {"eqh":eqh,"eql":eql,"ob":ob,"fvg":fvg}
    except Exception:
        return {"eqh":None,"eql":None,"ob":None,"fvg":None}

# ========== EVX2 ==========
def evx2(df, ind):
    if len(df)<30: return {"trigger":False,"cooldown":False,"metrics":{"v_ratio":1.0,"react":0.0,"z":0.0}}
    v=df["volume"].astype(float); vma=v.rolling(20).mean(); vsd=v.rolling(20).std().replace(0,1e-9)
    z=float(((v.iloc[-1]-vma.iloc[-1])/vsd.iloc[-1]) if not math.isnan(vsd.iloc[-1]) else 0.0)
    v_ratio=float(v.iloc[-1]/max(vma.iloc[-1],1e-9))
    o=float(df["open"].iloc[-1]); c=float(df["close"].iloc[-1]); atr=float(ind.get("atr") or 0.0)
    react=abs(c-o)/max(atr,1e-9)
    trig=(v_ratio>=EVX_V_SPIKE) and (react>=EVX_ATR_REACT or z>=EVX_ZSCORE)
    cool=(not trig) and v_ratio>=1.2 and react<1.0
    return {"trigger":bool(trig),"cooldown":bool(cool),"metrics":{"v_ratio":v_ratio,"react":react,"z":z}}

# ========== CANDLES ==========
def detect_candle(df: pd.DataFrame):
    if df is None or len(df)<3: return {"pattern":"NONE","strength":0,"dir":0}
    o=float(df["open"].iloc[-1]); h=float(df["high"].iloc[-1]); l=float(df["low"].iloc[-1]); c=float(df["close"].iloc[-1])
    rng=max(h-l,1e-12); body=abs(c-o); up=h-max(o,c); dn=min(o,c)-l
    up_pct=up/rng*100.0; dn_pct=dn/rng*100.0; body_pct=body/rng*100.0
    if body_pct<=10: return {"pattern":"DOJI","strength":1,"dir":0}
    if body_pct>=85 and up_pct<=7 and dn_pct<=7: return {"pattern":"MARUBOZU","strength":3,"dir":(1 if c>o else -1)}
    if dn_pct>=60 and body_pct<=30 and c>o: return {"pattern":"HAMMER","strength":2,"dir":1}
    if up_pct>=60 and body_pct<=30 and c<o: return {"pattern":"SHOOTING","strength":2,"dir":-1}
    return {"pattern":"NONE","strength":0,"dir":(1 if c>o else -1)}

def candle_psych(c):
    pat=(c or {}).get("pattern","NONE")
    if pat=="MARUBOZU": return 1.2,"CSTRONG"
    if pat in ("BULLISH_ENGULFING","BEARISH_ENGULFING"): return 1.0,"CLEAN"
    if pat in ("SHOOTING","SHOOTING_STAR","HANGING_MAN"): return 1.4,"TOP_REJECT"
    if pat in ("HAMMER","INVERTED_HAMMER"): return 1.4,"BOT_REJECT"
    if pat in ("DOJI","DOJI_LIKE","NONE"): return 0.3,"DOJI"
    return 0.5,"GENERIC"

# ========== FUSION SUMMARY ==========
def fusion_orchestrator(df, ind, info, smc):
    adx=float(ind.get("adx") or 0.0); rsi=float(ind.get("rsi") or 50.0)
    pdi=float(ind.get("plus_di") or 0.0); mdi=float(ind.get("minus_di") or 0.0)
    side_bias=1 if pdi>mdi else (-1 if mdi>pdi else 0)
    mom=(1.0 if adx>=28 else 0.5 if adx>=20 else 0.2)+(0.5 if rsi>=55 else 0.5 if rsi<=45 else 0.0)
    structure=0.3 if smc.get("ob") else 0.0
    ev=evx2(df, ind); boom=0.6 if ev["trigger"] else 0.0
    cnd=detect_candle(df); cscore=0.2 if cnd["pattern"] in ("MARUBOZU","HAMMER","SHOOTING") else 0.0
    fusion=min(1.0,max(0.0,0.25*mom+0.35*structure+0.25*boom+0.15*cscore))
    return {"fusion_score":float(fusion),"explosion":ev,"candle":cnd,"structure_bias":side_bias,"smc":smc}

# ========== PATIENCE / SMC COUNCIL ==========
def _last_real_swings(df,left=2,right=2):
    if len(df)<left+right+3: return None,None,False,False
    h=df["high"].astype(float).values; l=df["low"].astype(float).values
    ph=[]; pl=[]
    for i in range(left,len(df)-right):
        if all(h[i]>=h[j] for j in range(i-left,i+right+1)): ph.append((i,h[i]))
        if all(l[i]<=l[j] for j in range(i-left,i+right+1)): pl.append((i,l[i]))
    hh=len(ph)>=2 and ph[-1][1]>ph[-2][1]; ll=len(pl)>=2 and pl[-1][1]<pl[-2][1]
    return (ph[-1] if ph else (None,None)), (pl[-1] if pl else (None,None)), hh, ll

def classify_breakout(df, ind, key, side_hint):
    if key is None: return {"real":False,"fake":False,"needs_retest":False}
    closes=df["close"].astype(float); highs=df["high"].astype(float); lows=df["low"].astype(float)
    v=df["volume"].astype(float); vma=(v.rolling(20).mean().iloc[-1] or 1e-9)
    v_ratio=float(v.iloc[-1]/vma); adx=float(ind.get("adx") or 0.0)
    need=BRK_CONFIRM_CLOSES
    ok=False
    if side_hint=="long":  ok=all(closes.iloc[-i]>key for i in range(1,need+1))
    if side_hint=="short": ok=all(closes.iloc[-i]<key for i in range(1,need+1))
    real=bool(ok and v_ratio>=BRK_MIN_VRATIO and adx>=BRK_MIN_ADX)
    o=float(df["open"].iloc[-1]); c=float(df["close"].iloc[-1]); rng=float(highs.iloc[-1]-lows.iloc[-1] or 1e-9)
    up=float(highs.iloc[-1]-max(o,c)); dn=float(min(o,c)-lows.iloc[-1])
    fake=False
    if side_hint=="long":  fake=(c<key and (up/rng)>=0.55 and v_ratio>=1.2)
    if side_hint=="short": fake=(c>key and (dn/rng)>=0.55 and v_ratio>=1.2)
    return {"real":real,"fake":fake,"needs_retest":real,"v_ratio":v_ratio,"adx":adx}

def patience_and_smc_logic(df, ind, fusion, info):
    # update grace
    if len(df)>=2 and int(df["time"].iloc[-1])!=int(df["time"].iloc[-2]):
        if STATE.get("grace_bars_left",0)>0: STATE["grace_bars_left"]-=1
        if STATE.get("doji_wait_left",0)>0: STATE["doji_wait_left"]-=1

    smc=fusion.get("smc",{}) or {}
    px=float(info.get("price") or 0.0)
    ob=smc.get("ob")
    up = smc.get("eqh") or (ob.get("bot") if (ob and ob.get("side")=="bear") else None)
    dn = smc.get("eql") or (ob.get("top") if (ob and ob.get("side")=="bull") else None)
    key = up if STATE["side"]=="long" else dn

    brk=classify_breakout(df, ind, key, ("long" if STATE["side"]=="long" else "short"))
    if not STATE.get("brk_confirmed",False) and brk["real"]:
        STATE["brk_confirmed"]=True; STATE["retest_await"]=True
        STATE["retest_deadline"]=STATE["bars"]+RETEST_TIMEOUT_BARS

    if STATE.get("retest_await",False) and key is not None:
        if (STATE["side"]=="long" and px<=key) or (STATE["side"]=="short" and px>=key):
            STATE["retest_await"]=False
        elif STATE["bars"]>=STATE.get("retest_deadline",0):
            STATE["retest_await"]=False

    cnd = detect_candle(df); cs, tag = candle_psych(cnd)
    tol=RCS_SMC_BPS; near=0.0
    try:
        if smc.get("eqh") and abs((px-smc["eqh"])/smc["eqh"])*10000.0<=tol: near+=1.0
        if smc.get("eql") and abs((px-smc["eql"])/smc["eql"])*10000.0<=tol: near+=1.0
        if ob:
            if ob.get("side")=="bear" and abs((px-ob["bot"])/ob["bot"])*10000.0<=tol: near+=1.0
            if ob.get("side")=="bull" and abs((px-ob["top"])/ob["top"])*10000.0<=tol: near+=1.0
    except Exception: pass
    ev=fusion.get("explosion",{})
    adx=float(ind.get("adx") or 0.0)
    score=cs+(1.0*near)+(1.0 if ev.get("trigger") else 0.0)-(0.5 if adx<20 else 0.0)

    action="NONE"
    if tag=="DOJI" and DOJI_IGNORE_IN_DEFENSE:
        if score>=TH_FULL: action="FULL_CLOSE"
        elif score>=TH_PART: action="PARTIAL_CLOSE"
        elif score>=TH_TIGHT: action="TIGHTEN_TRAIL"
        else: STATE["doji_wait_left"]=DOJI_CONFIRM_BARS; action="WAIT_CONFIRM"
    else:
        if (tag=="TOP_REJECT" and STATE["side"]=="long") or (tag=="BOT_REJECT" and STATE["side"]=="short"):
            if score>=TH_PART: action="PARTIAL_CLOSE"
            elif score>=TH_TIGHT: action="TIGHTEN_TRAIL"
    return action

# ========== ORDERS ==========
def _params_open(side):
    if POSITION_MODE=="hedge":
        return {"positionSide":"LONG" if side=="buy" else "SHORT","reduceOnly":False}
    return {"reduceOnly":False}

def _params_close():
    if POSITION_MODE=="hedge":
        return {"positionSide":"LONG" if STATE.get("side")=="long" else "SHORT","reduceOnly":True}
    return {"reduceOnly":True}

def _read_position():
    try:
        poss=ex.fetch_positions(params={"category":"linear"})
        for p in poss:
            sym=(p.get("symbol") or p.get("info",{}).get("symbol") or "")
            if SYMBOL.split(":")[0].replace("/","") not in sym: continue
            qty=abs(float(p.get("contracts") or p.get("info",{}).get("size") or 0))
            if qty<=0: return 0.0,None,None
            entry=float(p.get("entryPrice") or p.get("info",{}).get("avgPrice") or 0)
            side_raw=(p.get("side") or p.get("info",{}).get("side") or "").lower()
            side="long" if "long" in side_raw or float(p.get("contracts") or 0)>0 else "short"
            return qty, side, entry
    except Exception as e:
        logging.error(f"_read_position error: {e}")
    return 0.0,None,None

def compute_size(balance, price):
    eff=balance or 0.0; capital=eff*RISK_ALLOC*LEVERAGE
    raw=max(0.0, capital/max(float(price or 0.0),1e-9))
    return safe_qty(raw)

def open_market(side, qty, price):
    if qty<=0: print(colored("❌ skip open (qty<=0)","red")); return False
    if MODE_LIVE:
        try:
            try: ex.set_leverage(LEVERAGE, SYMBOL, params={"buyLeverage":LEVERAGE,"sellLeverage":LEVERAGE})
            except Exception: pass
            ex.create_order(SYMBOL,"market",side,qty,None,_params_open(side))
        except Exception as e:
            print(colored(f"❌ open: {e}","red")); logging.error(f"open_market: {e}"); return False
    STATE.update({"open":True,"side":"long" if side=="buy" else "short","entry":price,"qty":qty,
                  "pnl":0.0,"bars":0,"trail":None,"breakeven":None,"tp1_done":False,"highest_profit_pct":0.0,
                  "profit_targets_achieved":0,"opp_votes":0,
                  "grace_bars_left":MIN_HOLD_BARS,"brk_confirmed":False,"retest_await":False,
                  "retest_deadline":0,"doji_wait_left":0})
    print(colored(f"🚀 OPEN {('🟩 LONG' if side=='buy' else '🟥 SHORT')} qty={fmt(qty,4)} @ {fmt(price)}","green" if side=="buy" else "red"))
    return True

def _reset_after_close(prev_side=None):
    global wait_for_next_signal_side
    prev_side=prev_side or STATE.get("side")
    STATE.update({"open":False,"side":None,"entry":None,"qty":0.0,"pnl":0.0,"bars":0,"trail":None,"breakeven":None,
                  "tp1_done":False,"highest_profit_pct":0.0,"profit_targets_achieved":0,"opp_votes":0})
    wait_for_next_signal_side = "sell" if prev_side=="long" else ("buy" if prev_side=="short" else None)

def close_market_strict(reason="STRICT"):
    global compound_pnl
    exch_qty, exch_side, exch_entry = _read_position()
    if exch_qty<=0:
        if STATE.get("open"): _reset_after_close(reason)
        return
    side_to_close="sell" if (exch_side=="long") else "buy"
    qty_to_close=safe_qty(exch_qty)
    attempts=0; last_error=None
    while attempts<CLOSE_RETRY_ATTEMPTS:
        try:
            if MODE_LIVE: ex.create_order(SYMBOL,"market",side_to_close,qty_to_close,None,_params_close())
            time.sleep(CLOSE_VERIFY_WAIT_S)
            left_qty,_,_= _read_position()
            if left_qty<=0:
                px=price_now() or STATE.get("entry"); entry_px=STATE.get("entry") or exch_entry or px
                side=STATE.get("side") or exch_side or ("long" if side_to_close=="sell" else "short")
                qty=exch_qty; pnl=(px-entry_px)*qty*(1 if side=="long" else -1)
                compound_pnl+=pnl; print(colored(f"🔚 STRICT CLOSE {side} reason={reason} pnl={fmt(pnl)} total={fmt(compound_pnl)}","magenta"))
                _reset_after_close(side); return
            qty_to_close=safe_qty(left_qty); attempts+=1
            print(colored(f"⚠️ strict close retry {attempts}/{CLOSE_RETRY_ATTEMPTS} residual={fmt(left_qty,4)}","yellow"))
            time.sleep(CLOSE_VERIFY_WAIT_S)
        except Exception as e:
            last_error=e; logging.error(f"strict close attempt {attempts+1}: {e}"); attempts+=1; time.sleep(CLOSE_VERIFY_WAIT_S)
    print(colored(f"❌ STRICT CLOSE FAILED: {last_error}","red"))

def close_partial(frac, reason):
    if not STATE["open"] or STATE["qty"]<=0: return
    qty_close=safe_qty(max(0.0, STATE["qty"]*min(max(frac,0.0),1.0)))
    px=price_now() or STATE["entry"]; min_unit=max(RESIDUAL_MIN_QTY, LOT_MIN or RESIDUAL_MIN_QTY)
    if qty_close<min_unit: return
    side="sell" if STATE["side"]=="long" else "buy"
    if MODE_LIVE:
        try: ex.create_order(SYMBOL,"market",side,qty_close,None,_params_close())
        except Exception as e: print(colored(f"❌ partial close: {e}","red")); return
    pnl=(px-STATE["entry"])*qty_close*(1 if STATE["side"]=="long" else -1)
    STATE["qty"]=safe_qty(STATE["qty"]-qty_close)
    print(colored(f"🔻 PARTIAL {reason} closed={fmt(qty_close,4)} pnl={fmt(pnl)} rem={fmt(STATE['qty'],4)}","magenta"))
    if STATE["qty"]<=FINAL_CHUNK_QTY and STATE["qty"]>0:
        print(colored(f"🧹 Final chunk ≤ {FINAL_CHUNK_QTY} → strict close","yellow"))
        close_market_strict("FINAL_CHUNK_RULE")

# ========== DEFENSE ON OPP RF ==========
def defensive_on_opposite_rf(ind, info):
    if not STATE["open"] or STATE["qty"]<=0: return
    # صبر/انتظار
    if STATE.get("grace_bars_left",0)>0 or STATE.get("retest_await",False) or STATE.get("doji_wait_left",0)>0:
        return
    # defensive partial + trail
    close_partial(0.20 if not STATE.get("tp1_done") else 0.15, "Opposite RF — defensive")
    atr=float(ind.get("atr") or 0.0); px=info.get("price")
    if atr>0 and px:
        gap=atr*max(ATR_TRAIL_MULT,1.2)
        if STATE["side"]=="long": STATE["trail"]=max(STATE["trail"] or (px-gap), px-gap)
        else: STATE["trail"]=min(STATE["trail"] or (px+gap), px+gap)
    # votes
    STATE["opp_votes"]=int(STATE.get("opp_votes",0))+1
    rf=info.get("filter"); hyst=abs((px-rf)/rf)*10000.0 if px and rf else 0.0
    adx=float(ind.get("adx") or 0.0)
    if STATE["opp_votes"]>=OPP_RF_VOTES_NEEDED and adx>=OPP_RF_MIN_ADX and hyst>=OPP_RF_MIN_HYST_BPS:
        close_market_strict("OPPOSITE_RF_CONFIRMED")

# ========== DYNAMIC TP/TRAIL ==========
def _tp_ladder(info, ind, side):
    px=info["price"]; atr=float(ind.get("atr") or 0.0)
    atr_pct=(atr/max(px,1e-9))*100.0 if px else 0.5
    adx=float(ind.get("adx") or 0.0)
    if adx>=35: mults=[1.8,3.2,5.0]
    elif adx>=28: mults=[1.6,2.8,4.5]
    else: mults=[1.2,2.4,4.0]
    tps=[round(m*atr_pct,2) for m in mults]; frs=[0.25,0.30,0.45]
    return tps, frs

def manage_after_entry(df, ind, info, fusion):
    if not STATE["open"] or STATE["qty"]<=0: return
    px=info["price"]; entry=STATE["entry"]; side=STATE["side"]
    rr=(px-entry)/entry*100*(1 if side=="long" else -1)
    dyn_tps, dyn_fracs=_tp_ladder(info, ind, side)
    k=int(STATE.get("profit_targets_achieved",0))
    tp1_now=TP1_PCT_BASE*(2.2 if ind.get("adx",0)>=35 else 1.8 if ind.get("adx",0)>=28 else 1.0)
    if (not STATE["tp1_done"]) and rr>=tp1_now:
        close_partial(TP1_CLOSE_FRAC, f"TP1@{tp1_now:.2f}%"); STATE["tp1_done"]=True
        if rr>=BREAKEVEN_AFTER: STATE["breakeven"]=entry
    if k<len(dyn_tps) and rr>=dyn_tps[k] and not fusion["explosion"]["trigger"]:
        frac=dyn_fracs[k] if k<len(dyn_fracs) else 0.25
        close_partial(frac, f"TP_dyn@{dyn_tps[k]:.2f}%"); STATE["profit_targets_achieved"]=k+1
    if rr>STATE["highest_profit_pct"]: STATE["highest_profit_pct"]=rr
    if STATE["highest_profit_pct"]>=TRAIL_ACTIVATE_PCT and rr<STATE["highest_profit_pct"]*0.60:
        close_partial(0.50, f"RatchetLock {STATE['highest_profit_pct']:.2f}%→{rr:.2f}%")
    atr=float(ind.get("atr") or 0.0)
    if rr>=TRAIL_ACTIVATE_PCT and atr>0:
        gap=atr*ATR_TRAIL_MULT
        if side=="long":
            new_trail=px-gap; STATE["trail"]=max(STATE["trail"] or new_trail, new_trail)
            if STATE["breakeven"] is not None: STATE["trail"]=max(STATE["trail"], STATE["breakeven"])
            if px<STATE["trail"]: close_market_strict(f"TRAIL_ATR({ATR_TRAIL_MULT}x)")
        else:
            new_trail=px+gap; STATE["trail"]=min(STATE["trail"] or new_trail, new_trail)
            if STATE["breakeven"] is not None: STATE["trail"]=min(STATE["trail"], STATE["breakeven"])
            if px>STATE["trail"]: close_market_strict(f"TRAIL_ATR({ATR_TRAIL_MULT}x)")
    # مجلس الإدارة (الصبر/SMC/الشموع/EVX2)
    action = patience_and_smc_logic(df, ind, fusion, info)
    if action=="TIGHTEN_TRAIL":
        atr=float(ind.get("atr") or 0.0); 
        if atr>0 and px:
            gap=atr*max(ATR_TRAIL_MULT,1.2)
            if side=="long": STATE["trail"]=max(STATE["trail"] or (px-gap), px-gap)
            else: STATE["trail"]=min(STATE["trail"] or (px+gap), px+gap)
    elif action=="PARTIAL_CLOSE":
        close_partial(0.30, "COUNCIL_PARTIAL")
    elif action=="FULL_CLOSE":
        close_market_strict("COUNCIL_FULL")

# ========== SNAPSHOT ==========
def pretty_snapshot(bal, info, ind, spread_bps, fusion, reason=None, df=None):
    left=time_to_candle_close(df) if df is not None else 0
    print("-"*110)
    print(f"📊 {SYMBOL} {INTERVAL} • {'LIVE' if MODE_LIVE else 'PAPER'} • {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("📈 RF & INDICATORS")
    print(f"   💲 Price {fmt(info.get('price'))} | RF filt={fmt(info.get('filter'))} hi={fmt(info.get('hi'))} lo={fmt(info.get('lo'))} | spread={fmt(spread_bps,2)} bps")
    print(f"   🧮 RSI={fmt(ind.get('rsi'))} +DI={fmt(ind.get('plus_di'))} -DI={fmt(ind.get('minus_di'))} ADX={fmt(ind.get('adx'))} ATR={fmt(ind.get('atr'))}")
    ev=fusion.get("explosion",{}); c=fusion.get("candle",{})
    print(f"   💥 EVX2 trig={ev.get('trigger')} cool={ev.get('cooldown')} v={fmt(ev.get('metrics',{}).get('v_ratio'))} react={fmt(ev.get('metrics',{}).get('react'))}")
    print(f"   🕯️ candle={c.get('pattern')}  fusionScore={fmt(fusion.get('fusion_score'),2)}  closes_in≈{left}s")
    smc=fusion.get("smc",{}); print(f"   🏗️ SMC: EQH={fmt(smc.get('eqh'))} EQL={fmt(smc.get('eql'))} OB={smc.get('ob')} FVG={smc.get('fvg')}")
    print("\n🧭 POSITION")
    print(f"   Balance={fmt(bal,2)}  Risk={int(RISK_ALLOC*100)}%×{LEVERAGE}x  CompoundPnL={fmt(compound_pnl)}  Eq~{fmt((bal or 0)+compound_pnl,2)}")
    if STATE["open"]:
        lamp='🟩 LONG' if STATE['side']=='long' else '🟥 SHORT'
        print(f"   {lamp} Entry={fmt(STATE['entry'])} Qty={fmt(STATE['qty'],4)} Bars={STATE['bars']} Trail={fmt(STATE['trail'])} BE={fmt(STATE['breakeven'])}")
        print(f"   🎯 TP_done={STATE['profit_targets_achieved']} HP={fmt(STATE['highest_profit_pct'],2)}% OppVotes={STATE.get('opp_votes',0)}")
    else:
        print("   ⚪ FLAT")
        if wait_for_next_signal_side: print(f"   ⏳ waiting RF opposite: {wait_for_next_signal_side.upper()}")
    if reason: print(f"   ℹ️ reason: {reason}")
    print("-"*110)

# ========== LOOP ==========
def trade_loop():
    global wait_for_next_signal_side
    loop_i=0
    while True:
        try:
            bal=balance_usdt(); px=price_now(); df=fetch_ohlcv()
            info=rf_signal_live(df); ind=compute_indicators(df); spread_bps=orderbook_spread_bps()
            df_closed=df.iloc[:-1] if len(df)>=2 else df.copy(); smc=detect_smc_levels(df_closed)
            fusion=fusion_orchestrator(df, ind, {"price":px or info["price"], **info}, smc)
            STATE["fusion_score"]=fusion["fusion_score"]
            if STATE["open"] and px:
                STATE["pnl"] = (px-STATE["entry"])*STATE["qty"] if STATE["side"]=="long" else (STATE["entry"]-px)*STATE["qty"]

            manage_after_entry(df, ind, {"price":px or info["price"], **info}, fusion)

            if STATE["open"]:
                if STATE["side"]=="long" and info["short"]: defensive_on_opposite_rf(ind, {"price":px or info["price"], **info})
                elif STATE["side"]=="short" and info["long"]: defensive_on_opposite_rf(ind, {"price":px or info["price"], **info})

            reason=None
            if spread_bps is not None and spread_bps>MAX_SPREAD_BPS:
                reason=f"spread too high ({fmt(spread_bps,2)}bps > {MAX_SPREAD_BPS})"

            sig = "buy" if (RF_LIVE_ONLY and info["long"]) else ("sell" if (RF_LIVE_ONLY and info["short"]) else None)
            if not RF_LIVE_ONLY:
                # دخول عند إغلاق الشمعة فقط
                if len(df)>=2 and int(df["time"].iloc[-1])!=int(df["time"].iloc[-2]):
                    sig = "buy" if info["long"] else ("sell" if info["short"] else None)

            if not STATE["open"] and sig and reason is None:
                if wait_for_next_signal_side and sig!=wait_for_next_signal_side:
                    reason=f"waiting opposite RF: need {wait_for_next_signal_side.upper()}"
                else:
                    qty=compute_size(bal, px or info["price"])
                    if qty>0:
                        ok=open_market(sig, qty, px or info["price"])
                        if ok: wait_for_next_signal_side=None
                    else: reason="qty<=0"

            pretty_snapshot(bal, {"price":px or info["price"], **info}, ind, spread_bps, fusion, reason, df)

            if len(df)>=2 and int(df["time"].iloc[-1])!=int(df["time"].iloc[-2]) and STATE["open"]:
                STATE["bars"] += 1

            loop_i+=1; sleep_s=NEAR_CLOSE_S if time_to_candle_close(df)<=10 else BASE_SLEEP
            time.sleep(sleep_s)
        except Exception as e:
            print(f"❌ loop error: {e}\n{traceback.format_exc()}"); time.sleep(BASE_SLEEP)

# ========== API ==========
app=Flask(__name__)
@app.route("/")
def home():
    mode='LIVE' if MODE_LIVE else 'PAPER'
    return f"✅ BYBIT SMART RF — {SYMBOL} {INTERVAL} — {mode} — Council+Patience+EVX2"

@app.route("/metrics")
def metrics():
    return jsonify({
        "symbol":SYMBOL,"interval":INTERVAL,"mode":"live" if MODE_LIVE else "paper",
        "leverage":LEVERAGE,"risk_alloc":RISK_ALLOC,"price":price_now(),
        "state":STATE,"compound_pnl":compound_pnl,"wait_for_next_signal":wait_for_next_signal_side
    })

@app.route("/health")
def health():
    return jsonify({"ok":True,"open":STATE["open"],"side":STATE["side"],"qty":STATE["qty"],
                    "pnl":STATE["pnl"],"timestamp":datetime.utcnow().isoformat()}), 200

def keepalive_loop():
    url=(SELF_URL or "").strip().rstrip("/")
    if not url: 
        print("⛔ keepalive disabled"); return
    import requests; s=requests.Session(); s.headers.update({"User-Agent":"bybit-smart/keepalive"})
    while True:
        try: s.get(url,timeout=8)
        except Exception: pass
        time.sleep(50)

if __name__=="__main__":
    print(f"MODE: {'LIVE' if MODE_LIVE else 'PAPER'} • {SYMBOL} • {INTERVAL}")
    print(f"RISK: {int(RISK_ALLOC*100)}%×{LEVERAGE}x • RF_LIVE={RF_LIVE_ONLY}")
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    signal.signal(signal.SIGINT,  lambda *_: sys.exit(0))
    import threading
    threading.Thread(target=trade_loop, daemon=True).start()
    threading.Thread(target=keepalive_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)
