# Guía de anotación de subcategorías de seguridad, versión 1 (PROVISIONAL)

> **Estado:** borrador redactado por Claude Code (CC) el 18-10-2026 a partir de las definiciones
> y los casos de frontera del paso E1 de `PLAN_ETIQUETADO_SUBCATEGORIAS.md`. **No ha sido revisado
> por A1, A2 ni por ningún experto.** Se usa solo para la preanotación provisional de Sonnet
> (paso E2). Las reglas de frontera de la sección 3 son propuestas de CC y deben ser reemplazadas
> o confirmadas por EXP1 en el piloto (E3).

## 1. Tarea

Cada requisito de software (en inglés) ya fue clasificado como de **seguridad**. Hay que indicar
cuáles de las seis subcaracterísticas de seguridad de ISO/IEC 25010:2023 expresa. Es una tarea
**multietiqueta**: un requisito puede expresar varias o ninguna. Se etiqueta lo que el texto dice
explícitamente o lo que implica de forma directa; no se infiere intención ni detalles de diseño que
el texto no menciona.

## 2. Definiciones (ISO/IEC 25010:2023)

| Etiqueta | Definición | Se marca cuando el requisito... | No se marca cuando... |
|---|---|---|---|
| **Confidentiality** | Los datos solo son accesibles para quienes están autorizados. | restringe el acceso o la lectura de información a usuarios, roles o dispositivos autorizados; exige cifrado, ocultación o protección de datos frente a divulgación; exige privacidad. | solo pide identificar a alguien sin hablar de qué puede ver o leer (eso es Authenticity). |
| **Integrity** | Se impide el acceso o la modificación no autorizados de datos o programas. | exige que los datos no sean alterados, corrompidos ni manipulados; pide MAC, hash o firma para detectar alteración; pide validar la exactitud o coherencia de los datos; restringe quién puede modificar. | solo restringe la lectura (Confidentiality). |
| **Non_repudiation** | Se puede demostrar que acciones o eventos ocurrieron, de modo que no puedan negarse después. | exige pruebas, comprobantes, firmas o evidencia con valor probatorio frente a una disputa; menciona explícitamente repudio o no repudio. | solo registra eventos para auditoría interna sin fin probatorio frente a terceros (Accountability). |
| **Accountability** | Las acciones de una entidad se pueden rastrear de forma única hasta ella. | exige registros de auditoría, bitácoras, historial o trazabilidad de quién hizo qué y cuándo. | el registro es solo funcional (p. ej. guardar un historial de compras para el usuario) sin objetivo de rastrear responsables. |
| **Authenticity** | Se puede demostrar que un sujeto o recurso es quien dice ser. | exige autenticar o identificar usuarios, dispositivos, tarjetas o componentes (contraseña, PIN, certificado, credencial, verificación de identidad u origen). | habla de permisos tras la autenticación sin pedir verificar la identidad (Confidentiality o Integrity, según el caso). |
| **Resistance** | El sistema sigue operando mientras está bajo ataque de un actor malicioso. | pide resistir, detectar, bloquear o recuperarse de ataques, abuso intencional, malware, intrusiones, repetición de mensajes (replay) o intentos reiterados de acceso; pide estados seguros ante fallo malicioso. | pide solo disponibilidad o tolerancia a fallas no maliciosas (fiabilidad, fuera de alcance). |

**Ninguna aplica:** si el requisito es de seguridad pero no encaja con claridad en ninguna
subcaracterística (por ejemplo, un requisito genérico de "el sistema debe ser seguro"), devuelve
las seis etiquetas en 0. No fuerces una etiqueta.

**No se puede decidir (UNCLEAR):** es distinto de "ninguna aplica". Si el texto no permite saber
si una subcaracterística aplica (por ejemplo, "the card shall check the data" sin decir para qué),
incluye esa subcaracterística en `unclear` en vez de adivinar. Esos requisitos los revisa un
experto sin la preanotación. (Agregado el 18-10-2026, a partir de la plantilla "Tag Requirements"
de IREB. No se usó en la corrida provisional.)

## 3. Reglas para casos de frontera (propuestas de CC, pendientes de EXP1)

1. **Bloqueo tras varios intentos fallidos de acceso**: marcar **Resistance** (contiene el ataque
   de fuerza bruta). Marcar también **Authenticity** solo si el texto habla de verificar
   credenciales o identidad, no por el mero hecho de que haya un login.
2. **Registro de transacciones o eventos**: marcar **Accountability** si el fin es rastrear quién
   y cuándo. Marcar **Non_repudiation** solo si el texto habla de pruebas, disputas, comprobantes
   o de impedir que una parte niegue haber actuado. Pueden ir ambas.
3. **Control de acceso por roles o permisos**: marcar **Confidentiality** si limita quién puede
   ver o leer; marcar **Integrity** si limita quién puede crear, modificar o borrar. Pueden ir
   ambas ("solo el administrador puede ver y editar").
4. **Firmas digitales, MAC y hash**: marcar **Integrity** (detectan alteración). Añadir
   **Authenticity** si el texto indica que verifican el origen o la identidad del emisor, y
   **Non_repudiation** solo si se menciona que el firmante no podrá negar su acción.
5. **Cifrado y canales seguros**: **Confidentiality**; añadir **Integrity** solo si el texto
   menciona detección de alteración.
6. **Autenticación (contraseña, PIN, certificado, identificación)**: **Authenticity**. La mera
   exigencia de "autenticarse antes de acceder a X" no implica Confidentiality salvo que el texto
   hable de proteger la información.
7. **Expiración de sesión, cierre por inactividad**: **Resistance** si busca limitar el abuso o la
   exposición ante un atacante; no marcar Confidentiality salvo mención explícita de proteger datos.
8. **Validación de entradas** (SQL injection, XSS, entradas malformadas): **Resistance** (ataque)
   e **Integrity** si menciona proteger los datos de modificación.
9. **Disponibilidad y respaldos**: no es una subcaracterística de seguridad en ISO/IEC 25010;
   no marcar nada por disponibilidad o mantenibilidad. Si el requisito solo trata de eso, devuelve
   todo en 0.
10. **Privacidad y datos personales**: **Confidentiality**.
11. **Dos objetivos en un mismo requisito**: marca ambas etiquetas; no elijas "la principal".
12. **Palabras gatillo**: una palabra como *secure*, *securely*, *protected*, *safe*, *password* o
    *log* no basta para marcar una subcategoría. Marca lo que el requisito exige. Por ejemplo,
    "the system shall securely display the menu" no expresa ninguna subcaracterística por decir
    *securely*; "the system shall log the user out after 10 minutes" no es Accountability, porque
    *log out* no es un registro. (Agregada el 18-10-2026 a partir de la advertencia de la guía de
    prompts de IREB; no se usó en la corrida provisional.)

## 4. Ejemplos resueltos (redactados por CC, ilustrativos)

| Requisito | Etiquetas | Motivo |
|---|---|---|
| "The system shall encrypt all stored passwords." | Confidentiality | Cifrado de datos sensibles. |
| "The card shall verify the MAC of each received message and discard it if invalid." | Integrity, Authenticity | MAC detecta alteración y verifica el origen; descartar el mensaje también protege contra ataques, pero el texto no menciona un ataque explícito. |
| "After three failed login attempts the account shall be locked." | Resistance | Contención de fuerza bruta (regla 1). |
| "Every transaction shall be logged with the user id and timestamp." | Accountability | Rastreo de quién y cuándo. |
| "The signed receipt shall serve as proof in case of dispute." | Non_repudiation, Integrity | Prueba ante disputa; la firma protege la integridad. |
| "Only administrators shall be able to view and modify user records." | Confidentiality, Integrity | Regla 3. |
| "The system shall be secure." | (ninguna) | Genérico. |

## 5. Salida

Para cada requisito: `id`, las seis etiquetas en 0/1, una confianza de 0 a 1 por etiqueta, la
lista `unclear` (vacía si todo está claro) y una justificación de una oración. Si el requisito parece no ser de seguridad, anótalo igual según
las reglas anteriores (probablemente todo en 0) y menciónalo en la justificación; no cambies la
etiqueta `Security`.
