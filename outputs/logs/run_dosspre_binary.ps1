# Binaria de BETO y RoBERTa-BNE en DOSSPRE (se descartó el laboratorio). Relanzarla retoma lo que falte.
$root = "C:\Users\Franco\Projects\ULima\SeminarioI-binario"
$log = "$root\outputs\logs\run_dosspre_binary.log"
Set-Location $root
$env:PYTHONIOENCODING = "utf-8"; $env:MPLBACKEND = "Agg"
Add-Content $log "[$(Get-Date -Format s)] empieza la cola"
foreach ($s in @("run_all_training.py --tasks binary --models beto roberta_bne --datasets dosspre", "scripts\build_report.py")) {
    cmd /c "py $s >> `"$log`" 2>&1"
}
Add-Content $log "[$(Get-Date -Format s)] cola terminada"
