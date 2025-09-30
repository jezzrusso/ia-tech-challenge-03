@echo off
python -m src.data_prep --in_path data/raw/trn.json --out_dir data/processed --subset 50000
pause
