"""SOXTA ma'lumot: faqat skriptlar ishlashini sinash uchun. Haqiqiy UzEDSA emas."""
import os, pandas as pd
os.makedirs("tests/fake", exist_ok=True)
dist = {"TRU": 300, "MIX": 250, "ANG": 60, "FEA": 20, "DIS": 15, "SUR": 30}
def mk(split, k):
    rows = [(f"{split}{i}", f"fake {split} gap {c} {i}", c, "", "") for c, n in dist.items() for i in range(int(n * k))]
    return pd.DataFrame(rows, columns=["id", "text", "label", "Intensity", "Sarcasm"])
for s, k in (("train", 1), ("dev", .1), ("test", .1)):
    mk(s, k).to_csv(f"tests/fake/{s}.csv", index=False)
