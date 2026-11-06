# UzEDSA: kam toifalarni tarjima orqali to'ldirish

Skriptlar (`scripts/`):
- `prepare_splits.py` — UzEDSA xlsx -> tozalangan `data/train|dev|test.csv`
- `augment_translate.py` — FEA/DIS/ANG/SUR uchun inglizcha HF datasetlaridan o'zbekchaga (lotin) tarjima
- `make_qc_sheet.py` — qo'lda tekshirish uchun Excel
- `build_train_aug.py` — `train_aug.csv` yig'ish (dev/test'ga qo'shilmaydi, kesishma tekshiriladi)

`data/` ochiq repoga yuklanmaydi (`.gitignore`). `tests/` ichidagi ma'lumot soxta, faqat sinov uchun.
