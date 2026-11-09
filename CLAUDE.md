# Reglas del proyecto

Tesis de clasificación multietiqueta de requisitos de seguridad en español. La estructura y el uso
del código están en `README.md`.

## Repositorios hermanos

La tesis se reparte en dos repositorios que están en la misma carpeta:

```
Projects/
  Tesis-clasificacion-requerimientos/        este repositorio: código, datos y resultados
  Tesis-clasificacion-requerimientos-docs/   planes, notas, reporte técnico, figuras, papers, APA
```

- La documentación ya no está en `docs/` de este repo, sino en
  `../Tesis-clasificacion-requerimientos-docs/` (su contenido está en la raíz: `planes/`,
  `contexto_sesion/`, `reporte_tecnico/`, `figuras/`, `papers/`, `normas_apa_ulima/`).
  Cualquier mención antigua a `docs/X` se refiere a `../Tesis-clasificacion-requerimientos-docs/X`.
- Las reglas de documentación, del Google Doc y de redacción del paper están en el `CLAUDE.md`
  del repo de docs, que se importa aquí:
  @../Tesis-clasificacion-requerimientos-docs/CLAUDE.md
- Cada repositorio tiene su propio git y su propio remoto (`git@github-ulima:DasBrot0/...`, alias
  SSH `github-ulima`). Los commits de un repo nunca incluyen archivos del otro; un cambio que toque
  código y notas se commitea por separado en cada uno.
- El código no escribe en el repo de docs. Si un script necesita leer un archivo de docs, lo
  resuelve como `ROOT.parent / "Tesis-clasificacion-requerimientos-docs" / ...`.

Los papers de referencia están en `../Tesis-clasificacion-requerimientos-docs/papers/`. Lee
primero los markdowns, que están en `papers/markdowns/papers/`.

## Decisiones respaldadas en literatura

- Toda decisión del proyecto debe apoyarse en literatura. Esto incluye los modelos, la
  arquitectura, los hiperparámetros y sus rangos, el etiquetado, la traducción, el aumento de datos,
  las métricas y las pruebas estadísticas.
- Si ningún paper la respalda, márcala como **[Decisión propia]**, justifícala y avísale al usuario.
  Nunca la presentes como si viniera de un paper.
- Las fases del flujo general con muchas decisiones (entrenamiento, validación, presentación e
  interpretación de resultados, discusión) tienen su propio plan en
  `../Tesis-clasificacion-requerimientos-docs/planes/`. Ejemplos: `PLAN_AUMENTO_DATOS.md`,
  `PLAN_TRADUCCION_GLOSARIO.md`.

## Código

- Los scripts para leer o actualizar el Google Doc van en el repo de docs
  (`reporte_tecnico/_sync/`), nunca junto al código de deep learning.
- Los cambios de código, resultados o datos no implican push al Google Doc: eso solo se hace
  cuando el usuario lo pide explícitamente (skill `sync-reporte`).
