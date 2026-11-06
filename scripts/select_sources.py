#!/usr/bin/env python
"""Qo'lda tarjima uchun toza manba tanlash: inglizcha emotsiya klassifikatori ishonchli tasdiqlagan izohlar."""
import argparse, pandas as pd, numpy as np
from transformers import pipeline
from common import norm
from augment_translate import load_go, load_dair, positive_filter

CLS = "j-hartmann/emotion-english-distilroberta-base"
MAP = {"FEA": "fear", "DIS": "disgust", "ANG": "anger", "SUR": "surprise"}
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
ap.add_argument("--per", type=int, default=300); ap.add_argument("--anger", type=int, default=60)
ap.add_argument("--min-conf", type=float, default=0.85)
a = ap.parse_args()

df = pd.DataFrame(load_go() + load_dair(), columns=["src_text", "label", "src_dataset", "src_label"])
df["_n"] = df.src_text.map(norm); df = df.drop_duplicates("_n")
df = df[df.src_text.str.split().str.len().between(4, 35)]
clf = pipeline("text-classification", model=CLS, top_k=None, truncation=True, max_length=128)
probs = []
for i in range(0, len(df), 64):
    for r in clf(df.src_text.iloc[i:i + 64].tolist()):
        probs.append({x["label"]: x["score"] for x in r})
P = pd.DataFrame(probs, index=df.index)
df["conf"] = [P.loc[i, MAP[l]] for i, l in zip(df.index, df.label)]
df = df[df.conf >= a.min_conf]
NEG = r"\b(not|no longer|never|didn t|didnt|don t|dont|isn t|wasn t|without|stop feeling|avoid|wanted to not|hesitant|apprehensive)\b|n't"
df = df[~df.src_text.str.lower().str.contains(NEG, regex=True)]
print("yuqori ishonchli:", df.label.value_counts().to_dict())
out = []
for l, n in (("FEA", a.per), ("DIS", a.per), ("ANG", a.anger), ("SUR", a.per)):
    sub = df[df.label == l]
    go, dr = sub[sub.src_dataset == "go_emotions"].sort_values("conf", ascending=False), sub[sub.src_dataset != "go_emotions"].sample(frac=1, random_state=1)
    sub = pd.concat([go, dr.head(int(n * 0.35))]) if l != "SUR" else pd.concat([go, dr])
    if l == "SUR":
        keep = positive_filter(sub.src_text.tolist()); sub = sub[pd.Series(keep, index=sub.index)]
        print("SUR ijobiy+yuqori ishonchli:", len(sub))
    out.append(sub.head(n))
res = pd.concat(out).reset_index(drop=True)
res.insert(0, "id", [f"m_{i:04d}" for i in range(len(res))])
res[["id", "label", "src_text", "src_dataset", "src_label", "conf"]].to_csv(a.out, index=False)
print(res.label.value_counts().to_dict())
