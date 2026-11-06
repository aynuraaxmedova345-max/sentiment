#!/usr/bin/env python
"""UzEDSA xlsx -> tozalangan data/train.csv, dev.csv, test.csv (stratified, matn bo'yicha kesishmasiz).

Tozalash faqat Sifat_nazorati varag'ida hujjatlangan muammolar bo'yicha:
  - takroriy sarlavha qatorlari va belgisiz qatorlar tashlanadi
  - sinonim teglar: NEUTRAL->NEU, ANGER->ANG, SURPRISE->SUR, DISGUST->DIS, FEAR->FEA
  - tagsetdan tashqari teglar (HOPE, INSIGHT, ...) tashlanadi (10 tegga avtomatik keltirib bo'lmaydi)
  - Domain bo'sh joylari kesiladi
  - bir xil matnlar (normallashtirilgandan keyin) bitta qoldiriladi -> train/dev/test o'rtasida sizish bo'lmaydi
Intensity=3 qiymatlari O'ZGARTIRILMAYDI (hisobotda ko'rsatiladi).
"""
import argparse, os
import pandas as pd
from sklearn.model_selection import train_test_split
from common import norm

TAGS = ["JOY", "SAD", "ANG", "FEA", "SUR", "TRU", "DIS", "ANT", "MIX", "NEU"]
SYN = {"NEUTRAL": "NEU", "ANGER": "ANG", "SURPRISE": "SUR", "DISGUST": "DIS", "FEAR": "FEA"}

ap = argparse.ArgumentParser()
ap.add_argument("--xlsx", required=True); ap.add_argument("--outdir", default="data")
ap.add_argument("--dev", type=float, default=0.1); ap.add_argument("--test", type=float, default=0.1)
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()

df = pd.read_excel(a.xlsx, sheet_name="data", dtype=str).fillna("")
src = pd.read_excel(a.xlsx, sheet_name="Manba_ID", dtype=str)
df["row"] = range(2, len(df) + 2)                              # Excel qator raqami
m = dict(zip(src["Data_qator"].astype(int), src["Manba_ID"]))
df["id"] = df["row"].map(m)
n0 = len(df); rep = {}

bad = df["Emotion_primary"].str.strip().eq("Emotion_primary") | df["ID"].str.strip().isin(["№", "ID"])
rep["takroriy sarlavha"] = int(bad.sum()); df = df[~bad]
empty = df["Emotion_primary"].str.strip().eq("") | df["Text_uz (izoh)"].str.strip().eq("")
rep["belgisiz/bo'sh"] = int(empty.sum()); df = df[~empty]
df["label"] = df["Emotion_primary"].str.strip().str.upper().replace(SYN)
off = ~df["label"].isin(TAGS)
rep["tagsetdan tashqari"] = int(off.sum()); rep["tagsetdan tashqari tegler"] = df[off]["label"].value_counts().head(10).to_dict()
df = df[~off]
df["Domain"] = df["Domain"].str.strip()
df["text"] = df["Text_uz (izoh)"].str.strip()
df["_n"] = df.text.map(norm)
before = len(df); df = df.drop_duplicates("_n", keep="first"); rep["takroriy matn"] = before - len(df)
df = df[df._n != ""]

cols = ["id", "text", "label", "Intensity", "Sarcasm", "Emotion_multi", "Polarity", "Domain"]
df = df[cols]
tr, rest = train_test_split(df, test_size=a.dev + a.test, stratify=df.label, random_state=a.seed)
dv, te = train_test_split(rest, test_size=a.test / (a.dev + a.test), stratify=rest.label, random_state=a.seed)
os.makedirs(a.outdir, exist_ok=True)
for name, d in (("train", tr), ("dev", dv), ("test", te)):
    d.to_csv(f"{a.outdir}/{name}.csv", index=False)

print("Boshlang'ich qatorlar:", n0, "| qoldi:", len(df)); print("Tashlangan:", rep)
print("Intensity=3 (o'zgartirilmagan):", int((df.Intensity == "3").sum()))
for name, d in (("train", tr), ("dev", dv), ("test", te)):
    print(name, len(d), d.label.value_counts().to_dict())
assert not (set(tr.text.map(norm)) & set(dv.text.map(norm))) and not (set(tr.text.map(norm)) & set(te.text.map(norm))) and not (set(dv.text.map(norm)) & set(te.text.map(norm)))
print("train/dev/test o'rtasida matn kesishmasi: 0")
