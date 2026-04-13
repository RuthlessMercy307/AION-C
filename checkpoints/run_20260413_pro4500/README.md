# AION-C 1.1B — Sequential Training Run (2026-04-13)

Checkpoint de un entrenamiento POC del modelo AION-C 1.1B usando el pipeline
Motor-Sequential (4 fases). Esta corrida se hizo en una Vast.ai RTX PRO 4500
Blackwell 32 GB como validación antes de escalar al 13B.

## Archivos

| Archivo | Qué es |
|---|---|
| `aion_1p1b_sequential.pt` | Checkpoint final (1.1B params, 4.2 GB). Incluye backbone + 5 motores + orchestrator + 5 LoRA adapters. Clave `phase` = `phase_4_adapters`. |
| `aion_1b_sequential.summary.json` | Métricas por fase del último `train_1b_sequential.py` ejecutado sobre este checkpoint (Phase 3 + Phase 4 post-fix). |
| `eval_final_report.json` | Reporte del `eval_final.py` con los 50 canonical prompts (experimentos skippeados por bug de rebuild, ver "Known issues"). |
| `training_logs/` | Logs crudos de cada ejecución (`phase1.log`, `phase234.log`, `phase34.log`, `phase3_refix.log`, `install.log`). |

## Config del training

- Modelo: 1,100,935,897 params (1.1B)
- Tokenizer: BPE 32K (`tokenizer/aion_32k.model`)
- Dataset: `datasets/dataset_canonical_86k.jsonl` (85,600 ejemplos, 7 dominios, bilingüe en/es)
- max_len: 128 tokens
- Optimizer: AdamW, lr=3e-4 (Phase 1), default config para Phase 2-4
- Hardware: RTX PRO 4500 Blackwell 32 GB, PyTorch 2.12 nightly cu128, compute 12.0 (sm_120)

## Fases ejecutadas

| Fase | Trainable | Steps | Final loss | Tiempo |
|---|---|---|---|---|
| 1 — backbone | 952.8M (86.5%) | 1500 | 1.68 | 20.7 min |
| 2 — cora | 36.4M (3.3%) | 1500 | 1.28 | 11.8 min |
| 2 — forge_c | 36.4M | 1500 | 3.12 | 10.5 min |
| 2 — axiom | 36.4M | 1500 | **0.71** 🏆 | 9.5 min |
| 2 — muse | 36.4M | 1000 | 1.52 | 6.9 min |
| 2 — empathy | 36.4M | 1000 | 1.50 | 6.7 min |
| 3 — orchestrator (re-run con balanced sampling) | 658K (0.06%) | 1500 | 3.76 | 3.7 min |
| 4 — LoRA adapters (re-run) | 328K (0.03%) | 1500 | 1.48 | 10.3 min |

## Resultados del eval

```
exact_match: 0.020
bleu:        0.010
routing_acc: 0.200
COMBINED:    0.0901
```

**Routing per-domain** (después del fix de balanced sampling):

| Dominio | Routing accuracy |
|---|---|
| cora | 0.0 |
| forge_c | **0.6** |
| axiom | 0.0 |
| muse | **0.4** |
| empathy | 0.0 |

El orchestrator YA no colapsa a cora (ese era el bug pre-fix), ahora distribuye
entre varios motores pero aún no es preciso para los 5 dominios. Conclusión: el
balanced sampling fix funciona en principio pero el classifier arrancó desde
pesos ya sesgados, limitando su capacidad de recuperar la calibración correcta.

## Known issues / lecciones para el 13B

1. **Phase 1 demasiado corto.** 1500 steps da BLEU ~0.01. Para generación
   coherente el 13B necesita ≥10000-20000 steps en Phase 1.

2. **Phase 3 necesita fresh init del `orchestrator.classifier`.** Hacer resume
   desde un classifier ya sesgado mitiga pero no elimina el colapso. La próxima
   corrida debe resetear a random `pipeline.orchestrator.classifier.*.weight`
   antes de empezar Phase 3.

3. **Bug en `eval_final.py`:** al reconstruir el pipeline con `build_pipeline("1b")`
   no se attachean los LoRA adapters de Phase 4 antes de `load_state_dict`, por
   lo que hay `unexpected=60` keys. Los experimentos de Parte 21 también crashean
   por `missing=20` keys en el `crystallizer.pooler` (estructura divergente entre
   el pipeline del training y el del eval).

4. **Bug en el resume logic de `train_1b_sequential.py`:** con
   `--resume-checkpoint + --only <phase>`, si la fase solicitada viene ANTES en
   el plan que la fase guardada en el checkpoint, se saltea (el check de resume
   corre antes que el filtro de `--only`). Workaround: reescribir `ck["phase"]`
   en el checkpoint antes del relaunch.

5. **Class imbalance en Phase 3:** `DOMAIN_TO_MOTOR_IDX` mapea `general` y
   `metacognitive` a motor 0 (cora), haciendo que el 39% de los targets de
   Phase 3 sean cora. Fix aplicado en este run: `balanced_motor_data_fn` hace
   round-robin sobre los 5 motor-domains reales, excluyendo general/metacognitive.

## Cómo cargar el checkpoint

```python
import torch
from experiments.benchmark_local import build_pipeline

pipeline, cfg = build_pipeline("1b", vocab_size=32000)
ck = torch.load("checkpoints/run_20260413_pro4500/aion_1p1b_sequential.pt",
                map_location="cpu", weights_only=False)
pipeline.load_state_dict(ck["model_state"], strict=False)
pipeline.eval()
```

Nota: `strict=False` es necesario por el bug #3. Los adapters LoRA (60 keys
"unexpected") no se cargan automáticamente a menos que se attacheen antes.
