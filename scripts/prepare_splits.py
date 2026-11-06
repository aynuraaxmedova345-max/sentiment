#!/usr/bin/env python
"""UzEDSA xlsx -> tozalangan data/train.csv, dev.csv, test.csv (group-aware: union-find bilan o'xshash matnlar bitta to'plamda).

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
# --- group-aware bo'linish: o'xshash matnlar union-find bilan bitta guruhga, guruh butunligicha bitta to'plamga ---
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold

ap2_thr = 0.80   # kosinus o'xshashlik chegarasi (char n-gram TF-IDF)
df = df.reset_index(drop=True)
X = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), min_df=2, sublinear_tf=True, dtype=np.float32).fit_transform(df.text.map(norm))
parent = list(range(len(df)))
def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]; x = parent[x]
    return x
def union(a_, b_):
    ra, rb = find(a_), find(b_)
    if ra != rb: parent[rb] = ra
XT = X.T.tocsr(); pairs = 0
for i in range(0, X.shape[0], 1500):
    S = (X[i:i + 1500] @ XT).tocoo()
    m = (S.data >= ap2_thr) & (S.col > S.row + i)
    for r, c in zip(S.row[m] + i, S.col[m]):
        union(int(r), int(c)); pairs += 1
df["group"] = [find(i) for i in range(len(df))]
gs = df.group.value_counts()
rep["o'xshash juftliklar (>=%.2f)" % ap2_thr] = pairs
rep["bir nechta matnli guruhlar"] = int((gs > 1).sum())
rep["eng katta guruh"] = int(gs.max())

k = round(1 / a.test)  # 10 ta fold -> 1 fold test, 1 fold dev
folds = np.zeros(len(df), dtype=int)
for f, (_, idx) in enumerate(StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=a.seed).split(df, df.label, df.group)):
    folds[idx] = f
te = df[folds == 0]; dv = df[folds == 1]; tr = df[folds >= 2]
for d in (tr, dv, te): d.drop(columns="group", inplace=True, errors="ignore")
os.makedirs(a.outdir, exist_ok=True)
for name, d in (("train", tr), ("dev", dv), ("test", te)):
    d.to_csv(f"{a.outdir}/{name}.csv", index=False)

print("Boshlang'ich qatorlar:", n0, "| qoldi:", len(df)); print("Tashlangan:", rep)
print("Intensity=3 (o'zgartirilmagan):", int((df.Intensity == "3").sum()))
for name, d in (("train", tr), ("dev", dv), ("test", te)):
    print(name, len(d), d.label.value_counts().to_dict())
assert not (set(tr.text.map(norm)) & set(dv.text.map(norm))) and not (set(tr.text.map(norm)) & set(te.text.map(norm))) and not (set(dv.text.map(norm)) & set(te.text.map(norm)))
print("train/dev/test o'rtasida aniq matn kesishmasi: 0; guruh kesishmasi: 0 (konstruksiya bo'yicha)")
