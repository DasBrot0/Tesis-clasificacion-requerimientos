# Cola de la laptop: espera a que termine el C1 de los Transformers y sigue sin la binaria de
# DOSSPRE de BETO y RoBERTa-BNE, que se corren en el laboratorio. Relanzarla retoma lo que falte.
$root = "C:\Users\Franco\Projects\ULima\SeminarioI-binario"
$log = "$root\outputs\logs\run_rest.log"
Set-Location $root
$env:PYTHONIOENCODING = "utf-8"; $env:MPLBACKEND = "Agg"
Wait-Process -Id 18208 -ErrorAction SilentlyContinue
Add-Content $log "[$(Get-Date -Format s)] empieza la cola"
$steps = @(
    "run_all_training.py --tasks subcategories --conditions c0_real --models beto roberta_bne",
    "scripts\build_report.py",
    "run_all_training.py --tasks binary --models beto roberta_bne --datasets promise secreq",
    "run_all_training.py --models cnn",
    "run_all_training.py --models bilstm",
    "scripts\build_report.py"
)
foreach ($s in $steps) { cmd /c "py $s >> `"$log`" 2>&1" }
Add-Content $log "[$(Get-Date -Format s)] cola terminada"
