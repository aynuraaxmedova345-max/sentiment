#!/usr/bin/env python
"""Kam toifalarni (FEA, DIS, ANG, SUR) inglizcha HF datasetlaridan olib, o'zbekchaga (lotin) tarjima qiladi.

Manbalar:
  google-research-datasets/go_emotions (config "simplified")  : fear, disgust, anger, surprise
  dair-ai/emotion (faqat --use-dair bilan)                    : fear, anger, surprise (disgust yo'q)
SUR uchun sentiment filtri: cardiffnlp/twitter-roberta-base-sentiment-latest, faqat 'positive' qoladi.
Tarjima: facebook/nllb-200-distilled-600M, eng_Latn -> uzn_Latn.

Qoidalar: har toifada tarjima soni train'dagi real sondan oshmaydi; Intensity/Sarcasm BO'SH.
"""
import argparse, re, sys, time
import pandas as pd
from common import RARE, OUT_COLS, label_col, norm, read_csv

GO_MAP = {"fear": "FEA", "disgust": "DIS", "anger": "ANG", "surprise": "SUR"}
DAIR_MAP = {"fear": "FEA", "anger": "ANG", "surprise": "SUR"}   # dair label ids: 0 sad,1 joy,2 love,3 anger,4 fear,5 surprise
DAIR_NAMES = ["sadness", "joy", "love", "anger", "fear", "surprise"]
SENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
MT_MODEL = "facebook/nllb-200-distilled-600M"
BAD = re.compile(r"\[(NAME|RELIGION)\]|https?://|www\.|@\w+|#\w+", re.I)


def ok_text(t):
    n = len(t.split())
    return 3 <= n <= 50 and not BAD.search(t)


def load_go():
    from datasets import load_dataset
    ds = load_dataset("google-research-datasets/go_emotions", "simplified")
    names = ds["train"].features["labels"].feature.names
    rows = []
    for split in ds:
        for ex in ds[split]:
            if len(ex["labels"]) != 1:           # faqat bitta toifali izohlar
                continue
            name = names[ex["labels"][0]]
            if name in GO_MAP and ok_text(ex["text"]):
                rows.append((ex["text"], GO_MAP[name], "go_emotions", name))
    return rows


def load_dair():
    from datasets import load_dataset
    ds = load_dataset("dair-ai/emotion")           # split config
    rows = []
    for split in ds:
        for ex in ds[split]:
            name = DAIR_NAMES[ex["label"]]
            if name in DAIR_MAP and ok_text(ex["text"]):
                rows.append((ex["text"], DAIR_MAP[name], "dair_emotion", name))
    return rows


def positive_filter(texts, bs=32):
    from transformers import pipeline
    clf = pipeline("text-classification", model=SENT_MODEL, device=_device(), truncation=True, max_length=128)
    keep = []
    for i in range(0, len(texts), bs):
        for r in clf(texts[i:i + bs]):
            keep.append(r["label"].lower() == "positive")
    return keep


def _device():
    import torch
    return 0 if torch.cuda.is_available() else -1


class Translator:
    def __init__(self, bs):
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(MT_MODEL, src_lang="eng_Latn")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(MT_MODEL).to(self.dev).eval()
        if self.dev == "cuda":
            self.model.half()
        self.bs = bs
        self.tgt = self.tok.convert_tokens_to_ids("uzn_Latn")

    def __call__(self, texts):
        out = []
        for i in range(0, len(texts), self.bs):
            b = texts[i:i + self.bs]
            enc = self.tok(b, return_tensors="pt", padding=True, truncation=True, max_length=128).to(self.dev)
            with self.torch.no_grad():
                gen = self.model.generate(**enc, forced_bos_token_id=self.tgt, num_beams=4, max_length=128)
            out += self.tok.batch_decode(gen, skip_special_tokens=True)
            print(f"  tarjima: {min(i + self.bs, len(texts))}/{len(texts)}", flush=True)
        return out


class MockTranslator:
    def __call__(self, texts):
        return [f"[uz] {t}" for t in texts]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--use-dair", action="store_true", help="dair-ai/emotion'ni ham qo'shish")
    ap.add_argument("--target", type=int, default=None, help="har kam toifa uchun maqsad (default: train toifalari medianasi)")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--bench", type=int, default=0, help="faqat N ta namunada tarjima tezligini o'lchab chiq")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mock", action="store_true", help="sinov: tarmoqsiz soxta manba va tarjima (tests/ uchun)")
    a = ap.parse_args()

    tr = read_csv(a.train)
    lc = label_col(tr)
    counts = tr[lc].value_counts().to_dict()
    target = a.target or int(pd.Series(counts).median())
    need = {}
    for c in RARE:
        real = counts.get(c, 0)
        need[c] = max(0, min(target - real, real))     # train'dagi real sondan oshmaydi
    print("Train taqsimoti:", counts, "| maqsad:", target)
    print("Kerak (tarjima):", need)

    if a.mock:
        import random
        random.seed(a.seed)
        pool = [(f"this is mock {c} sentence number {i} about feeling", c, "mock", c) for c in RARE for i in range(400)]
        translate = MockTranslator()
    else:
        pool = load_go() + (load_dair() if a.use_dair else [])
        translate = None

    df = pd.DataFrame(pool, columns=["src_text", "label", "src_dataset", "src_label"])
    df["_n"] = df.src_text.map(norm)
    df = df.drop_duplicates("_n").sample(frac=1, random_state=a.seed).reset_index(drop=True)
    print("Manba qatorlari:", df.label.value_counts().to_dict())

    picked = []
    for c in RARE:
        sub = df[df.label == c]
        if need[c] == 0:
            continue
        if c == "SUR" and not a.mock:
            cand = sub.head(need[c] * 3)                 # filtrdan keyin yetishi uchun zaxira bilan
            keep = positive_filter(cand.src_text.tolist())
            sub = cand[pd.Series(keep, index=cand.index)]
            print(f"SUR: {len(cand)} nomzoddan {len(sub)} tasi ijobiy")
        picked.append(sub.head(need[c]))
    sel = pd.concat(picked, ignore_index=True) if picked else pd.DataFrame(columns=df.columns)

    if a.bench:
        t = Translator(a.bs)
        s = sel.sample(min(a.bench, len(sel)), random_state=a.seed).src_text.tolist()
        t0 = time.time(); t(s); dt = time.time() - t0
        per = dt / len(s)
        print(f"Tezlik: {per:.2f} s/izoh. Jami {len(sel)} izoh uchun taxminan {per * len(sel) / 60:.0f} daqiqa.")
        return

    translate = translate or Translator(a.bs)
    sel["text"] = translate(sel.src_text.tolist())
    sel = sel[sel.text.str.strip().str.len() > 0]
    sel = sel[sel.text.map(norm) != sel.src_text.map(norm)]   # tarjima bo'lmagan (o'zgarmagan) qatorlar
    sel = sel.drop_duplicates("text")
    sel["id"] = [f"tr_{i:05d}" for i in range(len(sel))]
    sel["Intensity"] = ""; sel["Sarcasm"] = ""; sel["source"] = "translated"
    sel[OUT_COLS].to_csv(a.out, index=False)

    got = sel.label.value_counts().to_dict()
    print("\nTarjima qo'shildi:", got)
    for c in RARE:
        real, g = counts.get(c, 0), got.get(c, 0)
        exp = need[c]
        if g < exp:
            print(f"  maqsaddan kam: {c}: {g}/{exp} (real={real})" + ("  -> --use-dair bilan qayta ishga tushiring" if not a.use_dair else ""))
    print("Yozildi:", a.out)


if __name__ == "__main__":
    main()
