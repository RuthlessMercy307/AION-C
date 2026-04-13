# AION-C — Resultados de Benchmarks

Todos los benchmarks usan: `hidden_dim=64`, `vocab word-level del dataset`, `nivel=1`,
`batch=1`, `lr=3e-4`, `torch.set_num_threads(4)`, CPU (sin GPU).

---

## 1. benchmark_5motors_tiny — Los 5 motores aprenden?

**Script:** `experiments/benchmark_5motors_tiny.py`
**Fecha:** 2026-04-07
**Config:** hidden_dim=64, 500 ejemplos/motor, 2000 steps/motor, nivel=1
**Metric:** Word F1 (token overlap pred vs ref)

| Motor   | F1 train | F1 nuevos | Tiempo | Estado    |
|---------|----------|-----------|--------|-----------|
| cora    | 0.217    | 0.163     | 222s   | FUNCIONA  |
| axiom   | 0.239    | 0.150     | 280s   | FUNCIONA  |
| empathy | 0.209    | 0.115     | 256s   | FUNCIONA  |
| muse    | 0.035    | 0.042     | 279s   | PARCIAL   |
| forge_c | 0.013    | 0.027     | 250s   | BAJO      |
| **MEDIA** | **0.143** | **0.099** | 1287s | — |

**Conclusiones:**
- 3/5 motores aprenden claramente (F1 > 0.2 en train)
- forge_c bajo: nombres de función únicos no generalizan con 500 ejemplos
- muse bajo: respuestas muy variables, alta varianza
- El mismatch train/inference previo (batched vs pipeline directo) era el bug crítico; resuelto usando `pipeline()` directo en ambos

**Params modelo:** 1,466,996 (hidden_dim=64, pipeline MoSE completo con 5 motores)

---

## 2. benchmark_motors_vs_transformer — Comparación INJUSTA (referencia)

**Script:** `experiments/benchmark_motors_vs_transformer.py` (primera versión)
**Fecha:** 2026-04-07
**Config:** MoSE 5 motores (1.49M params) vs Transformer (287K params) — ratio 5.2×
**Problema:** Comparación injusta — MoSE incluía orquestador + 5 motores + unifier

| Motor   | F1 Motor | F1 TF | Delta  | Ganador |
|---------|----------|-------|--------|---------|
| cora    | 0.193    | 0.279 | -0.086 | TF      |
| forge_c | 0.051    | 0.483 | -0.432 | TF      |
| muse    | 0.082    | 0.487 | -0.405 | TF      |
| axiom   | 0.030    | 0.143 | -0.113 | TF      |
| empathy | 0.210    | 0.353 | -0.143 | TF      |
| **MEDIA** | **0.113** | **0.349** | **-0.236** | TF 5/5 |

**Nota:** Resultados no válidos para comparación arquitectural (params muy desiguales).

---

## 3. benchmark_motors_vs_transformer — Comparación JUSTA, sin fix EOS

**Script:** `experiments/benchmark_motors_vs_transformer.py` (v2)
**Fecha:** 2026-04-07
**Config:** Motor = CORAPipeline (661K params), TF = Decoder-only (670K params), ratio 0.99×
**Arquitectura motor:** Encoder → Crystallizer → CRE → Decoder (1 motor, sin orch/unif)
**Arquitectura TF:** Decoder-only causal, hidden=64, n_layers=13, n_heads=2, ffn_mult=4
**Problema:** EOS prematuro — motor genera respuestas vacías a pesar de loss más bajo

| Motor   | F1 Motor | F1 TF | Delta  | Loss Motor | Loss TF | Tiempo Motor | Tiempo TF |
|---------|----------|-------|--------|------------|---------|--------------|-----------|
| cora    | 0.164    | 0.434 | -0.270 | —          | —       | 269s         | 58s       |
| forge_c | 0.000    | 0.543 | -0.543 | —          | —       | 260s         | 56s       |
| muse    | 0.074    | 0.596 | -0.522 | —          | —       | 295s         | 56s       |
| axiom   | 0.000    | 0.135 | -0.135 | —          | —       | 246s         | 50s       |
| empathy | 0.053    | 0.355 | -0.302 | —          | —       | 305s         | 52s       |
| **MEDIA** | **0.058** | **0.413** | **-0.354** | — | — | 275s avg | 54s avg |

**Bug identificado:** El motor generaba EOS como primer token en la mayoría de inferencias
a pesar de haber aprendido la distribución correctamente (loss 0.17 vs 0.35 del TF).
El TF no sufría del mismo problema porque sus outputs naturalmente no empiezan con EOS.

---

## 4. benchmark_motors_vs_transformer — Comparación JUSTA + Fix EOS fijo=3

**Script:** `experiments/benchmark_motors_vs_transformer.py` (v3, con `MIN_NEW_BEFORE_EOS=3`)
**Fecha:** 2026-04-07
**Fix:** `logits[EOS] = -inf` para los primeros 3 tokens generados en ambos modelos
**Config:** Misma que #3 (661K vs 670K params)

### CORA (run rápido de validación)

| Modelo | F1    | Loss final | Tiempo |
|--------|-------|------------|--------|
| Motor  | 0.281 | 0.1700     | 272s   |
| TF     | 0.334 | 0.3533     | 54s    |
| Delta  | -0.053 (TF gana) | Motor 2.1× mejor loss | — |

**Conclusión del fix:** F1 motor subió de 0.164 → 0.281 (+71%). Brecha con TF reducida de 0.270 → 0.053.
El motor tiene loss 2× mejor que el TF pero convierte peor en F1 porque sus respuestas son
más largas que las referencias cortas de nivel=1.

### 5 motores (run completo — ver sección 5)

---

## 5. benchmark_motors_vs_transformer — Comparación JUSTA + Fix EOS=3, 5 motores

**Script:** `experiments/benchmark_motors_vs_transformer.py` (v3)
**Fecha:** 2026-04-07
**Config:** Idéntica a #4, todos los 5 motores. Tiempo total: 1909s (~32 min)

| Motor   | F1 Motor | F1 TF | Delta  | Loss Motor | Loss TF | Ganador |
|---------|----------|-------|--------|------------|---------|---------|
| cora    | 0.332    | 0.401 | -0.069 | 0.113      | 0.299   | TF      |
| forge_c | 0.239    | 0.573 | -0.334 | 0.238      | 0.560   | TF      |
| muse    | 0.381    | 0.572 | -0.191 | 0.106      | 0.539   | TF      |
| axiom   | 0.189    | 0.283 | -0.094 | 0.559      | 1.568   | TF      |
| empathy | 0.479    | 0.583 | -0.104 | 0.404      | 0.721   | TF      |
| **MEDIA** | **0.324** | **0.482** | **-0.158** | **0.284** | **0.737** | TF 5/5 |

**Key findings:**
- El motor tiene **loss consistentemente 2-3× menor** que el TF en todos los dominios
- La brecha de F1 se redujo de 0.354 (sin fix) a **0.158** (con fix EOS=3)
- **empathy** es el dominio más cercano: 0.479 vs 0.583 (delta solo -0.104)
- **muse** muestra F1=0.381 — el motor captura patrones narrativos complejos
- **forge_c** sigue siendo el más difícil para el motor (nombres únicos de función)

---

## 6. Experimento EOS dinámico (descartado)

**Fecha:** 2026-04-07
**Fix probado:** `min_new = max(3, len(input_tokens) // 3)`
**Resultado CORA:** Motor F1=0.222 (vs 0.281 con fijo=3) — **peor**

**Por qué falla:** Las respuestas CORA nivel=1 son cortas (6-8 tokens: "Sí, X lleva a Y").
Con prompts de ~15 tokens, el dinámico suprime EOS en los primeros 5 tokens, generando
relleno que baja el overlap con la referencia. El fijo=3 es mejor para respuestas cortas.
**Decisión:** Revertido al fijo=3.

---

## Resumen de cambios de arquitectura activados

| Cambio | Archivo | Estado |
|--------|---------|--------|
| `use_convergence_gate=True` default | `cre/config.py:48` | Activado |
| `cre_use_convergence_gate=True` default | `router/pipeline.py:130` | Activado |
| `use_budget_manager=True` default | `router/pipeline.py:137` | Activado |
| Tests actualizados para nuevos defaults | `tests/test_*.py` | Verde (1578 passed) |
| Fix EOS prematuro en greedy_decode | `experiments/benchmark_motors_vs_transformer.py` | Aplicado |
| Benchmarks Motor vs TF (justa) | `experiments/benchmark_motors_vs_transformer.py` | Nuevo |

## Observaciones transversales

**Motor vs Transformer con params iguales (~661K):**
- El motor aprende mejor la distribución de entrenamiento (**loss 2× menor**)
- El motor genera respuestas vacías sin fix EOS (bug de inferencia, no de aprendizaje)
- Con fix EOS, la brecha de F1 en CORA es pequeña (0.281 vs 0.334)
- El motor entrena **~5× más lento por step** (cristalización + CRE iterativo)

**Implicaciones para escalar:**
- Con datasets grandes (50k+ ejemplos), el inductive bias del grafo debería compensar la menor velocidad por step
- Para dominios con razonamiento multi-hop, el CRE debería mostrar ventaja clara sobre el TF
- Los resultados actuales (nivel=1, 500 ejemplos) son demasiado simples para demostrar la ventaja del grafo causal
