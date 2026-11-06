#!/usr/bin/env python
"""translated_en.csv uchun avtomatik tuzatmalar (4-qadam): halusinatsiya, HTML qoldig'i, bo'shliqlar."""
import argparse, re
import pandas as pd
from common import read_csv

ap = argparse.ArgumentParser(); ap.add_argument("--inp", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
d = read_csv(a.inp); n0 = len(d)
rep = re.compile(r"\b(\w+)\b(?:\W+\1\b){3,}", re.I)
html = re.compile(r"href|https?|www\b|&\w+;|<\w+", re.I)
bad = d.text.map(lambda t: bool(rep.search(t))) | d.src_text.str.contains(html) | d.text.str.contains(html)
print("tashlandi (takror/HTML):", int(bad.sum()), d[bad].label.value_counts().to_dict())
d = d[~bad].copy()
d["text"] = (d.text.str.replace(r"\s+([,.!?;:])", r"\1", regex=True).str.replace(r"\s+", " ", regex=True).str.strip())
d = d[d.text.str.split().str.len() >= 2]
d.to_csv(a.out, index=False); print(n0, "->", len(d), d.label.value_counts().to_dict())
