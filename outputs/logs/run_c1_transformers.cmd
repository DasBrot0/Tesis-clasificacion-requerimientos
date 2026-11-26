@echo off
cd /d C:\Users\Franco\Projects\ULima\SeminarioI-binario
set PYTHONIOENCODING=utf-8
set MPLBACKEND=Agg
py run_all_training.py --tasks subcategories --conditions c1_augmented --models distilbeto beto roberta_bne >> outputs\logs\run_c1_transformers.log 2>&1
py scripts\build_report.py >> outputs\logs\run_c1_transformers.log 2>&1
