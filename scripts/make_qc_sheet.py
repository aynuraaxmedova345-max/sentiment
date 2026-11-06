#!/usr/bin/env python
"""Tarjima sifatini qo'lda tekshirish uchun Excel varag'i (QC ustunlari BO'SH qoldiriladi)."""
import argparse
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation
from common import read_csv

ap = argparse.ArgumentParser()
ap.add_argument("--translated", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

df = read_csv(a.translated).sort_values(["label", "id"])
wb = Workbook(); ws = wb.active; ws.title = "QC"
head = ["id", "label", "src_text (EN)", "text (UZ)", "Sifat (1-5)", "Xato turi", "Tuzatilgan matn", "Qoldirish (ha/yo'q)"]
ws.append(head)
for _, r in df.iterrows():
    ws.append([r["id"], r["label"], r["src_text"], r["text"], None, None, None, None])
for c in ws[1]:
    c.font = Font(bold=True, color="FFFFFF"); c.fill = PatternFill("solid", fgColor="305496")
for col, w in zip("ABCDEFGH", [10, 8, 55, 55, 12, 18, 55, 18]):
    ws.column_dimensions[col].width = w
for row in ws.iter_rows(min_row=2):
    for c in row:
        c.alignment = Alignment(wrap_text=True, vertical="top")
n = ws.max_row
dv1 = DataValidation(type="list", formula1='"1,2,3,4,5"', allow_blank=True)
dv2 = DataValidation(type="list", formula1='"ma\'no xatosi,grammatika,emotsiya o\'zgargan,tarjima qilinmagan,boshqa"', allow_blank=True)
dv3 = DataValidation(type="list", formula1='"ha,yo\'q"', allow_blank=True)
for dv, col in ((dv1, "E"), (dv2, "F"), (dv3, "H")):
    ws.add_data_validation(dv); dv.add(f"{col}2:{col}{n}")
ws.freeze_panes = "C2"; ws.auto_filter.ref = f"A1:H{n}"
wb.save(a.out)
print(f"{a.out}: {n - 1} qator")
