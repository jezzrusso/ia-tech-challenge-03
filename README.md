tech-challenge-amazon-titles/
├─ data/
│  ├─ raw/            # trn.json (original)
│  ├─ processed/      # train/val/test jsonl
│  └─ interim/        # amostras, logs, preds
├─ src/
│  ├─ data_prep.py    # subamostra + jsonl p/ SFT
│  ├─ train.py        # FT FLAN-T5 + LoRA
│  ├─ infer.py        # CLI de inferência
│  └─ eval.py         # ROUGE-L + amostra qualitativa
├─ scripts/
│  ├─ prepare.sh
│  ├─ train.sh
│  └─ demo.sh
├─ models/            # adapters LoRA
├─ notebooks/         # opcional
├─ requirements.txt
└─ README.md


Comandos utilizados:

python -m src.data_prep --in_path data/raw/trn.json --out_dir data/processed --subset 5000
python -m src.eval baseline
python -m src.train --epochs 1
python -m src.eval sft
python -m src.infer --title "Apple iPhone 14 Pro Max 128GB" --question "Descreva o produto"
