# Detiene el entrenamiento a las 07:40 si todavía sigue (las corridas se retoman después).
$limit = Get-Date -Hour 7 -Minute 40 -Second 0
while ((Get-Date) -lt $limit) { Start-Sleep -Seconds 60 }
$procs = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match "run_transformers.cmd|run_all_training.py|train.py" }
foreach ($p in $procs) { Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue }
Add-Content "C:\Users\Franco\Projects\ULima\SeminarioI-binario\outputs\logs\run_transformers.log" "[$(Get-Date -Format s)] DETENIDO a las 07:40 ($($procs.Count) procesos)"
