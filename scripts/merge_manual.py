#!/usr/bin/env python
"""tr/*.txt (id ::: tarjima) + candidates.csv -> translated_en.csv. Apostroflar: oʻ gʻ (ʻ), qolganlari ʼ."""
import argparse, glob, re
import pandas as pd
from common import OUT_COLS

ap = argparse.ArgumentParser(); ap.add_argument("--cand", required=True); ap.add_argument("--dir", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
c = pd.read_csv(a.cand, dtype=str, keep_default_na=False).set_index("id")
rows = []
for f in sorted(glob.glob(f"{a.dir}/*.txt")):
    for ln in open(f, encoding="utf-8"):
        if " ::: " not in ln: continue
        i, t = ln.rstrip("\n").split(" ::: ", 1); i = i.strip(); t = t.strip()
        t = re.sub(r"([OoGg])['‘’`]", lambda m: m.group(1) + "ʻ", t)
        t = t.replace("'", "ʼ")
        r = c.loc[i]
        rows.append(dict(id=i, text=t, label=r.label, Intensity="", Sarcasm="", source="translated", src_text=r.src_text, src_dataset=r.src_dataset, src_label=r.src_label))
d = pd.DataFrame(rows)
assert d.id.is_unique, "takroriy id"
d[OUT_COLS].to_csv(a.out, index=False)
print(len(d), d.label.value_counts().to_dict())
