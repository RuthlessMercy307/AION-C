# TRAINING_SPEC.md — Especificación del dataset de entrenamiento de AION-C

> **Propósito de este documento:** enumerar EXACTAMENTE qué debe contener el dataset
> de entrenamiento del 1.1B y qué formato debe tener cada ejemplo, para que el
> modelo aprenda TODOS los comportamientos necesarios desde el primer step y
> no descubramos agujeros de cobertura a mes 1 de training.
>
> **Lección que motivó este documento:** el training del H200 produjo un modelo
> con F1 alto en empathy pero con diversidad de datos colapsada. El training
> del tiny canonical producía tokens garbage cuando el prompt contenía
> `[SKILL: identity]` porque ese skill casi nunca aparecía en el dataset
> (bug OOD). Cada fallo previo fue un bug de COBERTURA DE DATOS, no de
> arquitectura. Este spec cierra esa clase de bug.

---

## 1. Formato canónico — invariantes duras

Todo ejemplo es una única cadena de texto con esta gramática:

```
EXAMPLE := [SKILL: ...] [SKILL: ...] [MEM: ...] [MEM: ...] TURN+ [EOS]
TURN    := [USER: ...] (TOOL_CALL)? [AION: ...]
TOOL_CALL := [TOOL: json] [RESULT: ...]
```

**Invariantes que deben validarse en CI ANTES de entrenar:**

| # | Invariante | Verificación |
|---|---|---|
| I1 | Todo ejemplo termina con `[EOS]` exactamente una vez | `text.count("[EOS]") == 1 and text.rstrip().endswith("[EOS]")` |
| I2 | Los `[SKILL:]` y `[MEM:]` siempre preceden al primer `[USER:]` | parse order check |
| I3 | Todo `[TOOL:]` es seguido por exactamente un `[RESULT:]` | parser |
| I4 | Todo `[RESULT:]` está precedido por un `[TOOL:]` | parser |
| I5 | Multi-turn: `[USER:]` y `[AION:]` alternan, no hay `[USER:][USER:]` | parser |
| I6 | El contenido de `[TOOL:]` es JSON válido parseable | `json.loads` en cada uno |
| I7 | Ningún ejemplo supera `max_tokens=1024` tras BPE | tokenizer check |
| I8 | Ningún ejemplo está vacío o contiene solo tags (contenido real ≥ 5 chars por turno) | longitud mínima |
| I9 | Cada ejemplo tiene `language ∈ {en, es}` y `domain ∈ {cora, forge_c, muse, axiom, empathy, general, metacognitive}` | metadata |
| I10 | `[SKILL:]` y `[MEM:]` si existen contienen contenido que APARECE referenciado en la respuesta `[AION:]` | relevance check (al menos keyword) |

Script de validación obligatorio en `synth/validate_dataset.py`: correr sobre el
dataset final y FALLAR el training pipeline si alguna invariante rompe.

---

## 2. Categorías obligatorias y cobertura mínima

El modelo debe ver CADA categoría durante el training con un volumen mínimo.
Estas son las clases que ya existen + las nuevas que introduce este spec.

### 2.1 — Clases ya existentes (70K actuales)

| Categoría | Generador | Actual | Mínimo | Notas |
|---|---|---|---|---|
| Legacy direct QA | (pre-canonical) | 57,500 | ≥50,000 | Base de conocimiento |
| Multi-turn conversational | `conversational_gen.py` | 5,000 | ≥5,000 | 2-6 turnos |
| Tool use | `tool_gen.py` | 3,000 | ≥3,000 | Los 8 tools cubiertos |
| Skill-injected | `skill_injected_gen.py` | 2,000 | ≥2,000 | Los 11 skills referenciados |
| MEM-injected | `mem_injected_gen.py` | 2,000 | ≥2,000 | Recuperación + cita |
| Identity | `identity_gen.py` | 500 | ≥1,500 ⚠️ | **Subir a 1,500** — el bug OOD del tiny fue aquí |

### 2.2 — Clases nuevas que DEBE añadir este spec

| Categoría | Subcategoría | Vol mínimo | Justificación |
|---|---|---|---|
| **Metacognitive** | out_of_knowledge | 500 | El modelo debe saber cuándo buscar en la web |
| | propose_learning | 500 | Debe saber pedir permiso para aprender |
| | context_overflow | 500 | Debe resumir mentalmente sin perder coherencia |
| | out_of_capability | 500 | Debe ser honesto sobre límites duros |
| | low_confidence_disclaimer | 500 | Debe graduar certeza verbal |
| **SOMA** | primitive_commands | 300 | MOVE/ROTATE/GRIP + feedback sensorial |
| | high_level_goals | 300 | "explora la habitación", "busca X" |
| | safety_abort | 100 | Abortar ante riesgo físico |
| **Multi-turn largo** | 8-15 turnos | 500 | Coherencia a largo plazo |
| **Reasoning trace** | axiom paso a paso | 500 | Mostrar el razonamiento explícito |
| | forge_c debugging | 500 | Traza mental del debugging |
| **Graph explicit** | respuesta con graph | 1,000 | Casos donde [MEM: graph] viene y se usa |
| **Refusal / red-team** | jailbreak-like | 300 | Negarse con cortesía + explicación |
| | legal/medical disclaimer | 200 | Derivar a profesional sin cortar respuesta |

**Total objetivo:** 70,000 existentes + 2,500 metacognitive + 5,600 resto nuevo =
**~78,000 ejemplos** (72,500 si sólo el metacognitive aterriza antes de Fase E;
el resto puede llegar en una iteración de refuerzo posterior).

---

## 3. Matriz de cobertura (categoría × modalidad)

Para cada **categoría** debe haber ejemplos con CADA combinación de modalidades.
Un dataset completo se verifica con una matriz así (los valores son volumen mínimo):

| Categoría | en | es | single-turn | multi-turn | con skill | con mem | con tool | con graph |
|---|---|---|---|---|---|---|---|---|
| Legacy QA | ≥30k | ≥20k | ≥40k | ≥0 | 0 | ≥20k | 0 | ≥20k |
| Multi-turn conv | ≥2.5k | ≥2.5k | 0 | ≥5k | ≥500 | ≥500 | ≥500 | 0 |
| Tool use | ≥1.5k | ≥1.5k | ≥2k | ≥1k | ≥500 | ≥500 | ≥3k | 0 |
| Skill-injected | ≥1k | ≥1k | ≥1.5k | ≥500 | ≥2k | ≥500 | 0 | 0 |
| MEM-injected | ≥1k | ≥1k | ≥1.5k | ≥500 | 0 | ≥2k | 0 | ≥500 |
| Identity | ≥750 | ≥750 | ≥1k | ≥500 | ≥1.5k | 0 | 0 | 0 |
| Metacognitive | ≥1.25k | ≥1.25k | ≥2k | ≥500 | ≥500 | ≥500 | **≥500 (out_of_knowledge)** | 0 |
| SOMA | ≥350 | ≥350 | ≥400 | ≥300 | ≥100 | ≥100 | ≥200 | 0 |

**Regla dura:** si una celda está en 0, debe ser intencional — 0 = la combinación
no tiene sentido semántico (ej. "Legacy QA con tool" no aplica porque legacy
son Q&A puros). Todas las demás deben tener ≥10% del volumen de la categoría
o fallar el CI.

---

## 4. Las 5 categorías metacognitivas — ejemplos canónicos

### 4.1 `out_of_knowledge` (500 ejemplos)

**Disparadores:** APIs recientes (post entrenamiento), eventos actuales, datos
privados del usuario que el modelo no tiene, versiones específicas de libs.

**Output esperado:** emitir un `[TOOL: search_web]`, esperar el `[RESULT:]`,
citar explícitamente en la respuesta con la fuente.

```
[USER: ¿cuál es la versión estable de Rust este mes?]
[TOOL: {"tool":"search_web","query":"latest stable Rust release"}]
[RESULT: Rust 1.85.0 released 2025-02-20 per blog.rust-lang.org]
[AION: Según blog.rust-lang.org, la versión estable más reciente es Rust 1.85.0 (lanzada el 2025-02-20). Incluye [...].]
[EOS]
```

Variaciones obligatorias:
- con `[SKILL: web_search_etiquette]` preinyectado (100 ej)
- multi-turn donde el usuario corrige la búsqueda (50 ej)
- en español e inglés (50/50 split)

### 4.2 `propose_learning` (500 ejemplos)

**Disparadores:** preguntas que requieren conocimiento profundo no integrado
(ej. un dominio muy específico que ningún motor cubre hoy).

**Output esperado:** negarse a inventar, proponer el ciclo de auto-learn.

```
[USER: explícame cómo usar la librería interna "zkframework-v2" que usamos en la empresa]
[AION: No tengo ese conocimiento integrado — nunca vi "zkframework-v2" en mi training ni en la memoria de esta sesión. ¿Quieres que lo aprenda? Si me das un documento o un repo, puedo iniciar un ciclo de auto-learn y en ~5 minutos tendré un adapter de FORGE-C especializado. Mientras tanto, podés decirme brevemente qué hace y te ayudo con lo que pueda inferir.]
[EOS]
```

Variaciones:
- respuesta negativa del usuario ("no, sólo respóndeme lo que sabes") → el
  modelo responde con un resumen honestamente limitado + disclaimer. (200 ej)
- respuesta positiva → el modelo emite `[TOOL: start_auto_learn]` simulado (150 ej)
- en los dos idiomas (50/50)

### 4.3 `context_overflow` (500 ejemplos)

**Disparadores:** historiales con 10+ turnos donde el primer turno es relevante
para el último.

**Output esperado:** usar `[SUMMARY: ...]` al inicio del turno actual para
compactar lo relevante, luego responder breve sin perder coherencia.

```
[USER: ok, entonces le decís que sí cierra]
[SUMMARY: usuario pregunta por el bug del viernes; el cliente reporta que la API cierra conexión a los 30s; se identificó que es el idle timeout del ALB; se acordó bajarlo a 25s en staging; ahora confirma si implementar la fix]
[AION: Sí, bajemos `idle_timeout_seconds` de 30 a 25 en el target group de staging. Como hablamos, eso le da 5s de margen al client antes del ALB drop. ¿Lo deployo ya o querés ver un diff primero?]
[EOS]
```

Variaciones:
- cortar el historial después del `[SUMMARY:]` → el modelo debe responder
  correctamente sin haber visto el historial crudo (300 ej) — esto entrena al
  modelo a confiar en el summary
- summary ausente → el modelo lo genera él mismo antes de responder (200 ej)

### 4.4 `out_of_capability` (500 ejemplos)

**Disparadores:** tareas físicamente imposibles (OS kernels completos, proofs
matemáticos abiertos, simulaciones cuánticas exactas), tareas que excederían
mucho el context window o el tiempo razonable.

**Output esperado:** honestidad sobre el límite + alternativa útil más simple.

```
[USER: escribí un kernel Linux completo desde cero con scheduler y VM manager]
[AION: Esto excede lo que puedo hacer en una respuesta útil. Un kernel Linux real son cientos de miles de líneas, con decisiones de diseño que dependen del hardware objetivo y del modelo de driver — no es algo que genere bien de un tirón, y ofrecerte código así te daría algo no-compilable que parece real. Propuestas: (1) te armo un esqueleto mínimo estilo xv6 en C (~1000 líneas); (2) te explico la arquitectura alta de Linux para orientarte; (3) si querés un subsistema concreto (scheduler round-robin, por ejemplo), ese sí te lo puedo implementar bien. ¿Cuál preferís?]
[EOS]
```

Variaciones:
- en los dos idiomas
- con el usuario eligiendo la alternativa (multi-turn, 200 ej)
- tareas absurdas ("demuéstrame Riemann") → cortesía + explicación del estado
  del problema (100 ej)

### 4.5 `low_confidence_disclaimer` (500 ejemplos)

**Disparadores:** queries donde el modelo tiene información PARCIAL — ni fresca
ni totalmente ignorada. Es la zona gris entre "lo sé" y "búscalo tú".

**Output esperado:** respuesta completa pero con marcadores verbales de
confianza ("creo que", "si no recuerdo mal", "diría que" + cifra
aproximada).

```
[USER: ¿cuánta energía consume entrenar un modelo 1B durante 10 horas en una 4090?]
[AION: No estoy 100% seguro porque depende del batch size y del utilization real, pero diría que en el orden de 4-5 kWh: una 4090 tiene TDP 450W, a 80% utilización ≈ 360W sostenidos, y 360W × 10h = 3.6 kWh, más ~10% de overhead del resto de la máquina. Si querés el número exacto, pasame el nvidia-smi log del run y lo calculo.]
[EOS]
```

Variaciones:
- con `[SKILL: uncertainty_calibration]` preinyectado (100 ej)
- casos donde el usuario insiste en un número definitivo → el modelo lo da
  pero marca explícito el rango (150 ej)
- casos donde la mejor respuesta es "no sé con certeza — buscá en X" (150 ej)

---

## 5. Balance por dominio (motor target)

Después de sumar todas las categorías, el dataset debe mantener balance:

```
cora:          ≥15%  (razonamiento causal)
forge_c:       ≥15%  (código)
axiom:         ≥12%  (matemática)
muse:          ≥9%   (creativo)
empathy:       ≥9%   (emocional)
general:       ≥8%   (routing mixto)
metacognitive: ≥4%   (clases nuevas — debe sobre-muestrearse si baja)
```

El dataset actual tiene esto casi ok. El nuevo metacognitive agrega la categoría
7 y debe quedar cerca del 4% del total.

---

## 6. Distribución por idioma y modalidad

Reglas invariantes:

- **en / es:** 55% / 45% (el current es 57/43, aceptable)
- **single-turn / multi-turn:** 70% / 30% (current es 93/7 — **el dataset actual
  está muy poco entrenado en multi-turn**; este es un gap real que la Parte 2.2
  del spec debe atender eventualmente)
- **con tool / sin tool:** 8% / 92%
- **con skill / sin skill:** 10% / 90%
- **con mem / sin mem:** 80% / 20% (MEM es el default en el dataset actual
  porque los legacy tenían grafos → canonicalizados como `[MEM:]`)

**Gap crítico a cerrar:** el ratio multi-turn actual (7%) es bajo. El modelo
tiene poco incentivo para mantener coherencia a largo plazo. **Regenerar los
`conversational_gen.py` para subir el ratio a 20-30%** es un item prioritario
si hay tiempo antes de Fase E.

---

## 7. Señales de entrenamiento que deben venir en cada ejemplo

Además del texto, el dataset canónico carga metadata que el training loop usa:

| Campo | Tipo | Usado por | Obligatorio |
|---|---|---|---|
| `text` | str | LM loss | ✅ |
| `domain` | str | routing supervised loss (Parte 22) | ✅ |
| `language` | str | balance sampling | ✅ |
| `type` | str | balance sampling (WeightedRandomSampler) | ✅ |
| `has_skill` / `has_mem` / `has_tool` | bool | balance sampling | ✅ |
| `is_multi_turn` / `turn_count` | bool/int | balance sampling | ✅ |
| `target_sparsity` | float | (Parte 27) sparsity_loss target | ⚪ opcional |
| `expected_motor_sequence` | List[str] | (Parte 22.5) trajectory supervised | ⚪ opcional |

Los campos opcionales SÓLO se llenan en subsets específicos (los metacognitive
tienen `expected_motor_sequence=["cora"]` por default). No requerirlos
universalmente para no romper backward-compat con el 70K existente.

---

## 8. Cosas que el modelo debe APRENDER a NO hacer

Ejemplos adversariales (mínimo 500 en total):

- **No alucinar fuentes.** Si dice "según X", X debe estar en `[RESULT:]`
  real. Entrenar con ejemplos donde la respuesta correcta es "no tengo una
  fuente confiable, aquí va mi mejor intento: ..."
- **No romper el formato canónico.** Entrenar explícitamente ejemplos donde
  un usuario pide "responde sin etiquetas" y el modelo responde con el
  contenido pero mantiene el `[AION: ...][EOS]` wrapper.
- **No continuar más allá del `[EOS]`.** Loss mask debe forzar 0 gradient
  después del `[EOS]` token. Verificar en el dataloader.
- **No ejecutar tools sin confirmación cuando son side-effectful.** Todos
  los ejemplos donde el tool es destructivo (`WriteFile`, `RunCode` con
  side-effects) deben mostrar al modelo pidiendo confirmación explícita
  antes de la llamada.
- **No inventar skills.** Si un `[SKILL:]` no estaba en el contexto, el
  modelo no debe referirse a él.

---

## 9. Pipeline de validación del dataset (obligatorio antes de Fase E)

Correr EN ESTE ORDEN:

```bash
python -m synth.validate_dataset \
    --path datasets/dataset_canonical_72_5k.jsonl \
    --check-eos \
    --check-tag-order \
    --check-tool-json \
    --check-length \
    --check-metadata \
    --check-coverage-matrix \
    --fail-on-gap
```

El script debe producir `datasets/validation_report.json` con:

```json
{
  "total_examples": 72500,
  "invariants": {"I1": "pass", "I2": "pass", ...},
  "coverage_matrix": {...},
  "gaps": [],           // vacío = listo para training
  "warnings": [],
  "ready_for_training": true
}
```

**Si `ready_for_training == false`, el training pipeline no arranca.** Esta
es la barrera que evita el bug de "me di cuenta a mes 1 que el formato
estaba mal".

---

## 10. Checklist pre-Fase E

- [ ] Generar 2,500 ejemplos metacognitive (`synth/metacognitive_gen.py`)
- [ ] Subir identity a 1,500 (`synth/identity_gen.py --count 1500`)
- [ ] Correr `synth/validate_dataset.py` sobre el 72.5K
- [ ] Verificar `ready_for_training == true`
- [ ] `train_1b_canonical.py --config tiny --steps 50` dry-run con Fase F activa
- [ ] `sparsity_loss` converge
- [ ] Routing accuracy tiny ≥ 95%
- [ ] `exam_pass_rate` del tiny con adapters attached y detached = 1.0 (garantía Parte 22.1)
- [ ] Repackage `aion_c_vast.zip` incluyendo **todos** los nuevos paquetes de Fase F
- [ ] Subir a Vast.ai y arrancar el 1.1B

---

## 11. Notas para el siguiente release (post-Fase E)

Cosas que NO llegan a este training run pero deben estar en el próximo:

1. **Multi-turn ratio 20-30%**. Regenerar `conversational_gen.py` con historias
   más largas.
2. **SOMA samples reales**. Hoy las 700 líneas de SOMA son placeholders — con
   backend SOMA real generar ejemplos desde traces.
3. **Graph-conditioned responses**. Ejemplos donde el `[MEM:]` contiene un
   grafo serializado y la respuesta cita nodos específicos.
4. **Refusal/red-team real**. Usar un adversarial harness en lugar de los 300
   ejemplos handcrafted.
5. **Reward-labeled finetuning**. Usar el `RewardLedger` persistido del backend
   como fuente de preferencias RLHF-style.

---

**Autor:** generado durante la preparación de Fase F→E. Este archivo vive en
`AION-C-H200/AION-C/TRAINING_SPEC.md` y debe actualizarse cada vez que se
descubra un hueco de cobertura durante un training.
