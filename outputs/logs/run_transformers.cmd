@echo off
cd /d C:\Users\Franco\Projects\ULima\SeminarioI-binario
set PYTHONIOENCODING=utf-8
set MPLBACKEND=Agg
py run_all_training.py --tasks binary --models distilbeto >> outputs\logs\run_transformers.log 2>&1
py run_all_training.py --tasks subcategories --conditions c0_real --models distilbeto >> outputs\logs\run_transformers.log 2>&1
py run_all_training.py --tasks subcategories --conditions c1_augmented --models distilbeto >> outputs\logs\run_transformers.log 2>&1
py scripts\build_report.py >> outputs\logs\run_transformers.log 2>&1
