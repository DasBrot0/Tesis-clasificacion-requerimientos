# Detiene el entrenamiento el viernes 27-11 a las 07:50 (las corridas se retoman en la misma laptop).
$limit = Get-Date -Year 2026 -Month 11 -Day 27 -Hour 7 -Minute 50 -Second 0
while ((Get-Date) -lt $limit) { Start-Sleep -Seconds 60 }
$procs = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match "run_rest.ps1|run_c1_transformers.cmd|run_all_training.py|train.py" }
foreach ($p in $procs) { Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue }
Add-Content "C:\Users\Franco\Projects\ULima\SeminarioI-binario\outputs\logs\run_rest.log" "[$(Get-Date -Format s)] DETENIDO a las 07:50 ($($procs.Count) procesos)"
