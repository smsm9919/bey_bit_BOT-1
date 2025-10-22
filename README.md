# DOGEE_BOT_Z — Smart DOGE/USDT Bot (BingX Perp)

بوت تداول ذكي لـ **DOGE/USDT** مبني على Flask + CCXT يعمل على Render.
يدخل الصفقات **بنفس إشارة TradingView** (عبر Range Filter/TV بار) ثم يفعل منطق الذكاء:
- قراءة الشموع (Doji/Pin/Engulfing + Fake Break)
- مؤشرات: **RSI/ADX/+DI/-DI/DX/ATR**
- TP1 + Breakeven + **ATR Trailing**
- وضع **Smart Exit** للسوق العرضي (حماية ربح سريعة)
- **Scale‑In** تدريجي مع قوة الترند
- HUD لوجات واضحة + `/metrics` + keepalive

> **تحذير:** التداول الحقيقي على مسؤوليتك. فعّل `LIVE_TRADING=true` فقط إذا كنت متأكدًا.

## التشغيل المحلي
```bash
pip install -r requirements.txt
export FLASK_ENV=production
python main.py
```

## متغيرات البيئة (Render أو `.env`)
انظر ملف **env.example** ثم ضعه في إعدادات Render Environment أو محليًا.

## Render
- ضع رابط الخدمة `RENDER_EXTERNAL_URL` بعد أول تشغيل.
- ملف النشر: `render.yaml`
