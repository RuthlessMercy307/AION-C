"""
_gen_notebook.py — Genera colab_train_cora_50m.ipynb

Lee todos los archivos fuente, los codifica en base64 y construye
el notebook JSON para Google Colab (T4 GPU).
"""

import base64
import json
import pathlib
import sys

BASE = pathlib.Path('C:/Users/USER/Desktop/ias/AION-C')
OUT  = BASE / 'experiments' / 'colab_train_cora_50m.ipynb'

# ─── Lista de archivos a embeder ─────────────────────────────────────────────
FILES = [
    ('core/__init__.py',               BASE / 'core/__init__.py'),
    ('core/graph.py',                  BASE / 'core/graph.py'),
    ('synth/__init__.py',              BASE / 'synth/__init__.py'),
    ('synth/causal_graph_gen.py',      BASE / 'synth/causal_graph_gen.py'),
    ('encoder/__init__.py',            BASE / 'encoder/__init__.py'),
    ('encoder/mamba_layer.py',         BASE / 'encoder/mamba_layer.py'),
    ('encoder/model.py',               BASE / 'encoder/model.py'),
    ('crystallizer/__init__.py',       BASE / 'crystallizer/__init__.py'),
    ('crystallizer/config.py',         BASE / 'crystallizer/config.py'),
    ('crystallizer/node_detector.py',  BASE / 'crystallizer/node_detector.py'),
    ('crystallizer/relation_scorer.py',BASE / 'crystallizer/relation_scorer.py'),
    ('crystallizer/pooler.py',         BASE / 'crystallizer/pooler.py'),
    ('crystallizer/model.py',          BASE / 'crystallizer/model.py'),
    ('cre/__init__.py',                BASE / 'cre/__init__.py'),
    ('cre/config.py',                  BASE / 'cre/config.py'),
    ('cre/aggregator.py',              BASE / 'cre/aggregator.py'),
    ('cre/message_passing.py',         BASE / 'cre/message_passing.py'),
    ('cre/scratch_pad.py',             BASE / 'cre/scratch_pad.py'),
    ('cre/engine.py',                  BASE / 'cre/engine.py'),
    ('decoder/__init__.py',            BASE / 'decoder/__init__.py'),
    ('decoder/config.py',              BASE / 'decoder/config.py'),
    ('decoder/meta_head.py',           BASE / 'decoder/meta_head.py'),
    ('decoder/hybrid_layer.py',        BASE / 'decoder/hybrid_layer.py'),
    ('decoder/model.py',               BASE / 'decoder/model.py'),
    ('router/__init__.py',             BASE / 'router/__init__.py'),
    ('router/pipeline.py',             BASE / 'router/pipeline.py'),
    ('experiments/__init__.py',        None),   # will create empty
    ('experiments/train_cora_50m.py',  BASE / 'experiments/train_cora_50m.py'),
]

EXPERIMENTS_INIT = '# experiments package\n'

# ─── Leer y codificar en base64 ───────────────────────────────────────────────
files_b64 = {}
for rel_path, abs_path in FILES:
    if abs_path is None:
        content = EXPERIMENTS_INIT
    else:
        content = abs_path.read_text(encoding='utf-8')
    b64 = base64.b64encode(content.encode('utf-8')).decode('ascii')
    files_b64[rel_path] = b64
    print(f'  encoded {rel_path:50s}  ({len(content):6d} chars)')


# ─── Helpers para construir source de celdas ──────────────────────────────────

def lines(src: str):
    """Convierte un bloque de texto en lista de líneas para nbformat."""
    raw = src.split('\n')
    result = []
    for i, line in enumerate(raw):
        if i < len(raw) - 1:
            result.append(line + '\n')
        else:
            if line:   # última línea no vacía
                result.append(line)
    return result


def code_cell(cid: str, src: str):
    return {
        'cell_type': 'code',
        'id': cid,
        'metadata': {},
        'execution_count': None,
        'outputs': [],
        'source': lines(src),
    }


def md_cell(cid: str, src: str):
    return {
        'cell_type': 'markdown',
        'id': cid,
        'metadata': {},
        'source': lines(src),
    }


# ─── Celda 3: código Python que decodifica y escribe los archivos ─────────────

def build_cell3():
    cell3_lines = [
        'import os, pathlib, base64, sys\n',
        '\n',
        "ROOT = pathlib.Path('/content/aion_c')\n",
        "for pkg in ['core', 'synth', 'encoder', 'crystallizer', 'cre', 'decoder', 'router', 'experiments']:\n",
        '    (ROOT / pkg).mkdir(parents=True, exist_ok=True)\n',
        '\n',
        'files_b64 = {\n',
    ]

    for rel_path, b64_str in files_b64.items():
        # Dividir la cadena base64 en líneas de 76 caracteres para legibilidad
        chunks = [b64_str[i:i+76] for i in range(0, len(b64_str), 76)]
        # Formato multi-línea con paréntesis
        cell3_lines.append(f'    {repr(rel_path)}: (\n')
        for chunk in chunks:
            cell3_lines.append(f'        {repr(chunk)}\n')
        cell3_lines.append('    ),\n')

    cell3_lines += [
        '}\n',
        '\n',
        'for rel_path, b64_content in files_b64.items():\n',
        '    if isinstance(b64_content, tuple):\n',
        "        b64_content = ''.join(b64_content)\n",
        '    content = base64.b64decode(b64_content).decode(\'utf-8\')\n',
        '    (ROOT / rel_path).write_text(content, encoding=\'utf-8\')\n',
        "    print(f'  \u2713 {rel_path}')\n",
        '\n',
        'sys.path.insert(0, str(ROOT))\n',
        "print(f'\\n\u2713 {len(files_b64)} archivos escritos en {ROOT}')\n",
        "print('\u2713 sys.path configurado')\n",
    ]
    return cell3_lines


# ─── Contenido de cada celda ──────────────────────────────────────────────────

CELL1_SRC = """\
# CORA 50M — Entrenamiento en T4
Pipeline completo: Encoder(Mamba,8L) → Crystallizer → CRE(10iter) → Decoder(8L)
~37M parámetros | vocab_size=8000 | 5000 steps | RX6600/T4 GPU\
"""

CELL2_SRC = """\
from google.colab import drive
drive.mount('/content/drive')
DRIVE_DIR = '/content/drive/MyDrive/cora_50m'
import os
os.makedirs(DRIVE_DIR, exist_ok=True)
os.makedirs(f'{DRIVE_DIR}/checkpoints', exist_ok=True)
print(f'Drive montado. Checkpoints en: {DRIVE_DIR}/checkpoints')\
"""

CELL4_SRC = """\
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA disponible: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    free, total = torch.cuda.mem_get_info()
    print(f'VRAM: {free/1e9:.1f}GB libre / {total/1e9:.1f}GB total')
    # Estimate model VRAM
    N_PARAMS = 37_053_274
    model_mb = N_PARAMS * 4 / 1e6
    optim_mb = N_PARAMS * 4 * 2 / 1e6
    activ_mb = 400
    total_est = model_mb + optim_mb + activ_mb
    print(f'\\nEstimacion VRAM del modelo:')
    print(f'  Modelo (f32):      {model_mb:.0f} MB')
    print(f'  Optimizer (AdamW): {optim_mb:.0f} MB')
    print(f'  Activaciones:      {activ_mb} MB')
    print(f'  TOTAL estimado:    {total_est:.0f} MB ({total_est/1024:.2f} GB)')
    if total_est/1024 < free/1e9:
        print(f'  \\u2713 Cabe en VRAM disponible')
    else:
        print(f'  \\u26a0 ADVERTENCIA: puede no caber')
else:
    print('\\u26a0 Sin GPU. Ve a Runtime \\u2192 Change runtime type \\u2192 T4 GPU')\
"""

CELL5_SRC = """\
import sys
sys.path.insert(0, '/content/aion_c')

# Import and run training
import experiments.train_cora_50m as trainer

# Override constants for T4
trainer.N_DATASET    = 5000
trainer.N_STEPS      = 5000
trainer.ACCUM_STEPS  = 4
trainer.EVAL_EVERY   = 500
trainer.CKPT_EVERY   = 1000
trainer.PRINT_EVERY  = 100
trainer.CRE_ITERS    = 10
trainer.TRAIN_FRAC   = 0.80

# Override checkpoint dir to use Google Drive
import os
DRIVE_DIR = '/content/drive/MyDrive/cora_50m'
original_save = trainer.save_checkpoint

def drive_checkpoint(*args, **kwargs):
    kwargs['ckpt_dir'] = f'{DRIVE_DIR}/checkpoints'
    return original_save(*args, **kwargs)

trainer.save_checkpoint = drive_checkpoint

# Run training
trainer.train()\
"""

CELL6_SRC = """\
import json, glob, matplotlib
import matplotlib.pyplot as plt

# Find latest results file
results_files = sorted(glob.glob('/content/aion_c/experiments/results/train_cora_50m_*.json'))
if not results_files:
    results_files = sorted(glob.glob(f'{DRIVE_DIR}/results/train_cora_50m_*.json'))

if results_files:
    with open(results_files[-1]) as f:
        data = json.load(f)

    # Plot loss curve
    steps = [r['step'] for r in data['loss_curve']]
    total_loss = [r['total'] for r in data['loss_curve']]
    lm_loss = [r['lm'] for r in data['loss_curve'] if r.get('lm')]
    lm_steps = [r['step'] for r in data['loss_curve'] if r.get('lm')]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(steps, total_loss, alpha=0.4, color='gray', linewidth=0.8, label='raw')
    # Smooth with rolling average
    window = 50
    smoothed = [sum(total_loss[max(0,i-window):i+1])/len(total_loss[max(0,i-window):i+1])
                for i in range(len(total_loss))]
    axes[0].plot(steps, smoothed, color='steelblue', linewidth=2, label=f'MA-{window}')
    axes[0].set_title('Loss Total')
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    if lm_loss:
        lm_smooth = [sum(lm_loss[max(0,i-window):i+1])/len(lm_loss[max(0,i-window):i+1])
                     for i in range(len(lm_loss))]
        axes[1].plot(lm_steps, lm_loss, alpha=0.4, color='gray', linewidth=0.8)
        axes[1].plot(lm_steps, lm_smooth, color='coral', linewidth=2)
        axes[1].set_title('LM Loss (generación de respuestas)')
        axes[1].set_xlabel('Step')
        axes[1].grid(alpha=0.3)

    plt.suptitle('CORA 50M — Curva de entrenamiento', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{DRIVE_DIR}/loss_curve.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Show eval snapshots
    print('\\n' + '='*70)
    print('EVALUACIONES DURANTE ENTRENAMIENTO')
    print('='*70)

    for snap in data.get('eval_snapshots', []):
        print(f"\\n--- Step {snap['step']} ---")
        print(f"node_acc={snap['avg_node_acc']:.1%}  rel_recall={snap['avg_rel_recall']:.1%}  word_F1={snap['avg_word_f1']:.1%}")

    # Show last 5 eval examples from last snapshot
    snaps = data.get('eval_snapshots', [])
    if snaps:
        last = snaps[-1]
        print(f'\\n{"="*70}')
        print(f'ULTIMOS 5 EJEMPLOS EVAL (step {last["step"]})')
        print(f'{"="*70}')
        for i, ex in enumerate(last.get('examples', [])[:5], 1):
            print(f'\\n[{i}] Nivel {ex["level"]}')
            print(f'  Input   : {ex["q_preview"]}...')
            print(f'  GT nodos: {ex["gt_n"]}  Pred nodos: {ex["pred_n"]}  NodeAcc: {ex["node_acc"]:.0%}')
            print(f'  GT      : {ex["gt_answer"][:80]}')
            print(f'  Generado: {ex["gen_answer"][:80] or "(vacío)"}')
            print(f'  Word F1 : {ex["word_overlap"]:.0%}')

else:
    print('No se encontraron archivos de resultados.')
    print('Asegúrate de haber ejecutado la celda de entrenamiento.')\
"""


# ─── Construir el notebook ────────────────────────────────────────────────────

cell3_source = build_cell3()

cells = [
    md_cell('cell-001', CELL1_SRC),
    code_cell('cell-002', CELL2_SRC),
    {
        'cell_type': 'code',
        'id': 'cell-003',
        'metadata': {},
        'execution_count': None,
        'outputs': [],
        'source': cell3_source,
    },
    code_cell('cell-004', CELL4_SRC),
    code_cell('cell-005', CELL5_SRC),
    code_cell('cell-006', CELL6_SRC),
]

notebook = {
    'nbformat': 4,
    'nbformat_minor': 5,
    'metadata': {
        'kernelspec': {
            'display_name': 'Python 3',
            'language': 'python',
            'name': 'python3',
        },
        'language_info': {
            'name': 'python',
            'version': '3.10.0',
        },
        'accelerator': 'GPU',
        'colab': {
            'name': 'colab_train_cora_50m.ipynb',
            'provenance': [],
        },
    },
    'cells': cells,
}

# ─── Escribir el archivo ──────────────────────────────────────────────────────

out_str = json.dumps(notebook, ensure_ascii=False, indent=1)
OUT.write_text(out_str, encoding='utf-8')

size_bytes = OUT.stat().st_size
size_mb    = size_bytes / 1_000_000

print(f'\n[OK] Notebook escrito en: {OUT}')
print(f'     Tamaño: {size_bytes:,} bytes  ({size_mb:.2f} MB)')
print(f'     Celdas: {len(cells)}')
print(f'     Archivos embebidos: {len(files_b64)}')
