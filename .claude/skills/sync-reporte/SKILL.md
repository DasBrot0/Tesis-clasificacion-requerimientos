---
name: sync-reporte
description: Sincroniza el reporte técnico de Google Docs con el espejo markdown de ../Tesis-clasificacion-requerimientos-docs/reporte_tecnico/gdoc/. Úsala al iniciar la sesión (pull automático), cuando el usuario pida ver el estado del reporte o cuando pida subir cambios al Google Doc (push).
---

# Sincronización del reporte técnico con Google Docs

El reporte técnico vive en el repo hermano `../Tesis-clasificacion-requerimientos-docs/`. Las
instrucciones completas están en una sola copia, la de ese repo:

`../Tesis-clasificacion-requerimientos-docs/.claude/skills/sync-reporte/SKILL.md`

Léela y síguela tal cual. Sus rutas son relativas a la raíz del repo de docs, así que desde aquí
antepón `../Tesis-clasificacion-requerimientos-docs/`, por ejemplo:

```powershell
py -I ../Tesis-clasificacion-requerimientos-docs/reporte_tecnico/_sync/gdoc_sync.py pull "<ruta del JSON>"
```

Recordatorio: el pull es automático al iniciar la sesión; el push solo cuando el usuario lo pide
explícitamente.
