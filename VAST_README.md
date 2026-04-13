# AION-C 1.1B Training en Vast.ai (RTX 4090)

## Setup inicial (5 min)

```bash
# 1) Subir el zip
scp -P <PORT> aion_c_vast.zip root@<HOST>:/root/

# 2) SSH al instance
ssh -p <PORT> root@<HOST>

# 3) Descomprimir
cd /root
unzip aion_c_vast.zip
cd AION-C

# 4) Instalar deps mínimas (la imagen de Vast suele traer torch ya)
pip install -r requirements_vast.txt

# 5) Verificar GPU
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

## Pre-flight: correr tests críticos (2 min)

```bash
python -m pytest tests/test_evaluation.py tests/test_canonical_dataloader.py tests/test_brain_version.py -q
```

Esperado: ~110 tests pass, 0 fail.

## Sanity check con tiny (5 min)

Antes de tirar las 17 horas de GPU al 1.1B, verifica que el script corre OK:

```bash
python train_tiny_canonical.py --steps 100
python verify_tiny_e2e.py
```

Esperado: tiny entrenado, 7/7 E2E checks PASS.

## Training del 1.1B (estimado 15-17h con early stopping)

```bash
python train_1b_canonical.py --config 1b --steps 15000 2>&1 | tee train_1b.log
```

Parámetros activos:
  - fp16 mixed precision (autocast + GradScaler)
  - cosine LR con warmup=300, lr base 1e-4
  - grad_accum=16 (effective batch 16)
  - routing_w=1.0, balance_w=0.5
  - eval cada 200 steps con 50 prompts canónicos
  - save best por gen_quality.combined (no val_loss)
  - early stopping patience=500
  - BrainVersionManager → brain/v1/
  - context 1024 (dec_max_seq_len)

Logs cada 50 steps: lm/route/balance loss, routing acc%, lr, sps, ETA, VRAM.
Eval cada 200 steps: exact_match, BLEU, routing_accuracy, combined.

## Checkpoint final + descarga

El checkpoint best-by-combined queda en:
  - `checkpoints/aion_1b_canonical.pt`  (state_dict + history JSON)
  - `brain/v1/weights.pt`               (BrainVersionManager)
  - `brain/v1/metadata.json`            (metrics + parent)
  - `checkpoints/aion_1b_canonical.metrics.json`  (history completo)

Para descargar a tu PC:
```bash
scp -P <PORT> root@<HOST>:/root/AION-C/checkpoints/aion_1b_canonical.pt ./
scp -P <PORT> -r root@<HOST>:/root/AION-C/brain/ ./
```

## Recovery: si Vast se cae a mitad

```bash
# Resume desde el último best
python train_1b_canonical.py --config 1b --steps 15000 \
    --resume checkpoints/aion_1b_canonical.pt
```

## Budget esperado

- $0.229/hr × 17h ≈ $3.90
- 15K steps × ~3.5 sps ≈ 75 min training puro
- Eval cada 200 steps × 75 evals × ~3 min ≈ 225 min eval
- **Total ~5 horas** si converge rápido, hasta 17h si llega a max steps

## Si el modelo sale mediocre

1. Mira `checkpoints/aion_1b_canonical.metrics.json` — ¿el combined estaba subiendo o flat?
2. Si está flat → la loss del LM no baja → puede ser dataset o config problem
3. Si combined sube pero exact_match no → el modelo aprende formato pero no contenido
4. Considera más steps o diferente recipe (subir lr, bajar warmup, distinct dataset)
