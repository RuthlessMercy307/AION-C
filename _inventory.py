import sys
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

SEP = "=" * 72

print(SEP)
print("INVENTARIO AION-C — 2026-04-04")
print(SEP)

print("""
╔══════════════════════════════════════════════════════════════════════╗
║                     MÓDULOS IMPLEMENTADOS                            ║
╚══════════════════════════════════════════════════════════════════════╝

─── core/ ──────────────────────────────────────────────────────────────
  graph.py
    Clases : NodeType, CausalRelation, CausalNode, CausalEdge, CausalGraph
    Tests  : tests/test_graph.py (11 clases de test)
    Pipeline: SÍ — estructura de datos base usada por todos los módulos

─── encoder/ ────────────────────────────────────────────────────────────
  mamba_layer.py
    Clases : StreamEncoderConfig, RMSNorm, GatedFFN, SelectiveSSM, MambaLayer
  model.py
    Clases : StreamEncoder
    Tests  : tests/test_encoder.py (12 clases)
    Pipeline: SÍ — CORAPipeline.encoder (Mamba SSM, O(L))

─── crystallizer/ ───────────────────────────────────────────────────────
  config.py          → CrystallizerConfig
  node_detector.py   → NodeDetector
  pooler.py          → CrossAttentionPooler
  relation_scorer.py → AsymmetricRelationScorer
  model.py           → CrystallizerOutput, GraphCrystallizer
    Tests  : tests/test_crystallizer.py (12 clases)
    Pipeline: SÍ — CORAPipeline.crystallizer (encoder → grafo causal)

─── cre/ ────────────────────────────────────────────────────────────────
  config.py         → CREConfig
  aggregator.py     → AttentiveAggregator
  message_passing.py → _EdgeUpdater, CausalMessagePassingLayer
  engine.py         → CREOutput, CausalReasoningEngine
  scratch_pad.py    → ScratchPadConfig, DifferentiableScratchPad
  weakness.py       → Weakness, WeaknessReport, WeaknessDetector  [nuevo]
  convergence.py    → ConvergenceDecision, ConvergenceGate         [nuevo]
    Tests  : tests/test_cre.py (11) + test_scratch_pad.py (10)
             + tests/test_weakness.py (10) + tests/test_convergence.py (10)
    Pipeline: SÍ — CORAPipeline.cre (WeaknessDetector + ConvergenceGate
              opcionales via CREConfig.use_convergence_gate=True)

─── decoder/ ────────────────────────────────────────────────────────────
  config.py      → StreamDecoderConfig
  hybrid_layer.py → HybridDecoderLayer (Mamba + 2x cross-attn + FFN)
  meta_head.py   → MetaOutput, OutputMetaHead
  model.py       → DecoderOutput, StreamDecoder
    Tests  : tests/test_decoder.py (12 clases)
    Pipeline: SÍ — CORAPipeline.decoder
    Nota   : grounding fix activo (encoder_concepts opcional para compat.)

─── router/ ─────────────────────────────────────────────────────────────
  pipeline.py → CORAConfig, PipelineOutput, CORAPipeline
    Tests  : tests/test_pipeline.py (9 clases, 62 tests)
    Pipeline: ES el pipeline — integra los 4 módulos anteriores

─── synth/ ──────────────────────────────────────────────────────────────
  causal_graph_gen.py → CausalGraphGenerator + generadores L1-L5
    Tests  : tests/test_causal_graph_gen.py (12 clases)
    Pipeline: NO (datos sintéticos de entrenamiento, no inferencia)
""")

print("ESTADO DE TESTS: 717 passed | 4 skipped | 0 failed")
print()

print("""
╔══════════════════════════════════════════════════════════════════════╗
║              MÓDULOS PENDIENTES DEL PLAN v2                          ║
╚══════════════════════════════════════════════════════════════════════╝

1. SparseMoE / Mixture-of-Experts
   Estado : NO implementado — ni aparece como módulo propio en el plan v2.
            El CRE cubre su rol: weight sharing × N iters = profundidad
            sin multiplicar parámetros.
   Crítico para benchmark 500M: NO
   Veredicto: PUEDE ESPERAR. No está en el plan base de AION-C.
              Posible extensión futura en Fase F (mes 9+, scaling 3B→8B).

2. BudgetManager  (budget/manager.py, classifier.py, flop_counter.py)
   Estado : NO implementado. El plan lo ubica en budget/ (carpeta no existe).
            Aparece en pseudo-código de EnergyFunction (ComputeBudget).
   Crítico para benchmark 500M: PARCIALMENTE
   Veredicto: PUEDE ESPERAR para el benchmark de 2000 ejemplos.
              Necesario antes del entrenamiento real largo (GPU) para
              controlar FLOPs por iteración del CRE. Prioridad post-bench.

3. VAL — Verificador
   Estado : NO implementado. Plan: FASE 7 del entrenamiento + FASE 3
            post-CORA-500M. Evalúa calidad de respuestas y errores en
            el grafo causal generado.
   Crítico para benchmark 500M: NO
   Veredicto: PUEDE ESPERAR. El plan lo sitúa explícitamente post-500M.
              No bloquea ningún componente del pipeline actual.

4. DatasetQualityAnalyzer
   Estado : NO implementado. El plan (sección 18, FASE 1 inmediata) lo
            recomienda ANTES de cada entrenamiento para predecir calidad
            del modelo resultante. Solo herramienta de diagnóstico.
   Crítico para benchmark 500M: ÚTIL (no bloqueante)
   Veredicto: PRIORIDAD MEDIA. ~200 LOC, coste bajo, valor alto.
              Recomendado antes del primer entrenamiento real con GPU
              para detectar problemas en el dataset sintético.

5. CuriosityModule
   Estado : NO implementado. Plan: FASE 2 post-benchmark.
            Funcionalidad: búsqueda web → parseo → escritura en MEM.
            PREREQUISITO: MEM (MemoryStore) tampoco implementado.
   Crítico para benchmark 500M: NO
   Veredicto: PUEDE ESPERAR. Bloquedado por MEM. Feature de nivel 2
              (auto-mejora), no de arquitectura base.

6. ReflectionModule
   Estado : NO implementado. Plan: FASE 3 post-CORA-500M (Nivel 3).
            Analiza por qué el modelo erró y actualiza meta-reglas en MEM.
            PREREQUISITOS: VAL + MEM (ninguno existe aún).
   Crítico para benchmark 500M: NO
   Veredicto: PUEDE ESPERAR. Timeline: mes 7+ (Fase F).
              Depende de VAL y MEM como prerequisitos encadenados.

─────────────────────────────────────────────────────────────────────────
PRIORIDADES CONCRETAS
─────────────────────────────────────────────────────────────────────────

  AHORA (pipeline completo, 717 tests verdes):
    ✓ Pipeline funcional encoder→crystallizer→CRE→decoder
    ✓ WeaknessDetector + ConvergenceGate implementados
    → Ejecutar benchmark CORA vs Transformer (notebook listo)

  ANTES DEL ENTRENAMIENTO REAL:
    → DatasetQualityAnalyzer: analizar dataset synth antes de GPU
    → BudgetManager ligero: FLOPs/iter para run de 500M

  POST-BENCHMARK (modelo entrenado):
    → VAL básico
    → MEM (MemoryStore mínimo)
    → CuriosityModule (depende de MEM)
    → ReflectionModule (depende de VAL + MEM)

  POST-PRODUCCIÓN 3B:
    → SparseMoE (si se necesita scaling, no está en el plan base)
    → AdaptiveLoRAGrowth
─────────────────────────────────────────────────────────────────────────
""")
