"""Umumiy yordamchi funksiyalar."""
import re
import pandas as pd

RARE = ["FEA", "DIS", "ANG", "SUR"]          # faqat shular tarjima qilinadi
NEVER = ["TRU", "MIX"]                        # tarjima qilinmaydi
LABEL_CANDIDATES = ["label", "Label", "emotion", "Emotion"]
OUT_COLS = ["id", "text", "label", "Intensity", "Sarcasm", "source", "src_text", "src_dataset", "src_label"]


def label_col(df):
    for c in LABEL_CANDIDATES:
        if c in df.columns:
            return c
    raise SystemExit(f"Toifa ustuni topilmadi. Mavjud ustunlar: {list(df.columns)}")


def norm(s):
    """Solishtirish uchun matnni normallashtirish."""
    s = str(s).lower()
    s = re.sub(r"[ʻ’‘`´']", "'", s)
    s = re.sub(r"[^\w\s']", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def read_csv(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)
