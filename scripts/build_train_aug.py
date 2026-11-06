#!/usr/bin/env python
"""train + tarjima qatorlarini birlashtiradi. data/*.csv O'ZGARMAYDI. dev/test'ga hech narsa qo'shilmaydi.

QC'dan keyin: --qc tarjima_QC.xlsx berilsa, 'Qoldirish'=yo'q qatorlar tashlanadi va 'Tuzatilgan matn' qo'llanadi.
"""
import argparse, sys
import pandas as pd
from common import RARE, NEVER, OUT_COLS, label_col, norm, read_csv

ap = argparse.ArgumentParser()
ap.add_argument("--train", required=True); ap.add_argument("--dev", required=True); ap.add_argument("--test", required=True)
ap.add_argument("--translated", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--qc", help="to'ldirilgan QC Excel (ixtiyoriy)")
a = ap.parse_args()

tr, dv, te, tx = (read_csv(p) for p in (a.train, a.dev, a.test, a.translated))
lc = label_col(tr)
if lc != "label":
    for d in (tr, dv, te):
        d.rename(columns={lc: "label"}, inplace=True)

if a.qc:
    qc = pd.read_excel(a.qc, dtype=str).fillna("")
    qc.columns = ["id", "label", "src", "text", "q", "err", "fixed", "keep"]
    drop = set(qc[qc.keep.str.strip().str.lower().isin(["yo'q", "yoq", "no"])].id)
    fix = dict(zip(qc[qc.fixed.str.strip() != ""].id, qc[qc.fixed.str.strip() != ""].fixed))
    tx = tx[~tx.id.isin(drop)].copy()
    tx["text"] = [fix.get(i, t) for i, t in zip(tx.id, tx.text)]
    print(f"QC: {len(drop)} qator tashlandi, {len(fix)} qator tuzatildi")

assert set(tx.label) <= set(RARE), f"Tarjimada ruxsat etilmagan toifa: {set(tx.label) - set(RARE)}"
assert not (set(tx.label) & set(NEVER))

# dev/test bilan kesishgan yoki train'da bor matnlarni tashlash
held = set(dv.text.map(norm)) | set(te.text.map(norm))
before = len(tx)
tx = tx[~tx.text.map(norm).isin(held | set(tr.text.map(norm)))]
tx = tx[~tx.src_text.map(norm).isin(held)] if "src_text" in tx else tx
tx = tx.drop_duplicates("text")
print(f"Kesishish/dublikat sababli tashlandi: {before - len(tx)}")

# har toifada tarjima soni real sondan oshmasin
real = tr.label.value_counts().to_dict()
keep = []
for c, g in tx.groupby("label"):
    keep.append(g.head(real.get(c, 0)))
tx = pd.concat(keep) if keep else tx.iloc[0:0]
tx["Intensity"] = ""; tx["Sarcasm"] = ""

tr = tr.copy(); tr["source"] = "real"
for c in OUT_COLS:
    for d in (tr, tx):
        if c not in d.columns:
            d[c] = ""
aug = pd.concat([tr, tx], ignore_index=True)
cols = OUT_COLS + [c for c in aug.columns if c not in OUT_COLS]
aug[cols].to_csv(a.out, index=False)

inter_dev = len(set(aug.text.map(norm)) & set(dv.text.map(norm)))
inter_test = len(set(aug.text.map(norm)) & set(te.text.map(norm)))
print("Qo'shilgan tarjima:", tx.label.value_counts().to_dict())
print("train_aug hajmi:", len(aug), "(real", len(tr), "+ tarjima", len(tx), ")")
print("Toifalar taqsimoti:", aug.label.value_counts().to_dict())
print(f"dev bilan kesishma: {inter_dev}, test bilan kesishma: {inter_test}")
if inter_dev or inter_test:
    sys.exit("XATO: dev/test bilan kesishma bor")
