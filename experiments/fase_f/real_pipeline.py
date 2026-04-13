"""
real_pipeline.py — Helpers para correr los experimentos de Fase F contra el
tiny_canonical.pt real (no FakeMotor).

Carga el MoSEPipeline con los pesos reales del tiny, expone sus 5 motores y
provee funciones de exam compatibles con `common.py`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parent.parent.parent
CHECKPOINT = REPO / "checkpoints" / "tiny_canonical.pt"


def load_real_pipeline():
    """Carga el MoSEPipeline con los pesos de tiny_canonical.pt.

    Returns:
        pipeline en modo eval. Lanza FileNotFoundError si no existe el checkpoint.
    """
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"No tiny checkpoint at {CHECKPOINT}")

    from router.pipeline import MoSEPipeline, MoSEConfig
    from experiments.train_production import build_tokenizer

    tok = build_tokenizer(32_000)
    cfg = MoSEConfig(
        hidden_dim=64, vocab_size=tok.vocab_size,
        enc_n_layers=2, enc_state_dim=4, enc_expand=2, enc_d_conv=4, enc_ffn_mult=2,
        orch_mlp_hidden=32, orch_max_motors=3, orch_min_confidence=0.3,
        motor_max_nodes=8, motor_n_heads=4, motor_threshold=0.01, unif_n_heads=4,
        dec_n_layers=2, dec_n_heads=4, dec_max_seq_len=128,
        dec_state_dim=4, dec_expand=2, dec_d_conv=4, dec_ffn_mult=2,
    )
    pipeline = MoSEPipeline(cfg)
    ck = torch.load(str(CHECKPOINT), map_location="cpu", weights_only=False)
    pipeline.load_state_dict(ck["model_state"], strict=False)
    pipeline.eval()
    return pipeline


def real_motor_targets(motor: nn.Module, max_targets: int = 6) -> List[str]:
    """Selecciona paths de Linear significativos en un motor real del tiny.

    Usa auto_target_paths de growth.adapters, que prioriza proyecciones
    q/k/v/out y fallbacks de nombre.
    """
    from growth.adapters import auto_target_paths
    return auto_target_paths(motor, max_targets=max_targets)


def real_motor_exam(
    motor: nn.Module, n: int = 10, seed: int = 7, d_hidden: int = 64
) -> List[torch.Tensor]:
    """Crea un exam determinista de N inputs compatibles con un motor real.

    El motor real espera concepts [B, L, D] donde D = hidden_dim. Para un
    exam unitario, usamos B=1, L=4, D=d_hidden.
    """
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(1, 4, d_hidden, generator=g) for _ in range(n)]


def real_motor_outputs(motor: nn.Module, exam: List[torch.Tensor]) -> List[torch.Tensor]:
    """Corre el crystallizer.project del motor sobre cada input del exam.

    Nota: no corremos build_graph completo porque eso atraviesa el CRE con
    message passing tipado y no es determinista en esta configuración.
    Medimos sobre la proyección del crystallizer que SÍ es determinista y
    sensible a adapters en proj/out/input_proj/message.
    """
    outs: List[torch.Tensor] = []
    with torch.no_grad():
        for x in exam:
            # Atravesamos las proyecciones accesibles del crystallizer.
            # Forma segura: si tiene pooler, pasamos por q/k/v/out; si no,
            # usamos node_detector.confidence_head o equivalente. Elegimos
            # el camino más determinista: el pooler q_proj existe en forge_c
            # y en varios motores del MoSE.
            h = x  # [1, 4, 64]
            if hasattr(motor.crystallizer, "pooler"):
                # Atravesamos el pooler completo: suma lineal de q_proj
                # sobre el promedio del input como proxy de activación.
                mean = h.mean(dim=1)  # [1, 64]
                y = motor.crystallizer.pooler.q_proj(mean)
                y = y + motor.crystallizer.pooler.out_proj(y)
            else:
                # Fallback: usar el primer Linear que encontremos
                for mod in motor.modules():
                    if isinstance(mod, nn.Linear):
                        y = mod(h.mean(dim=1) if h.dim() == 3 else h)
                        break
            outs.append(y.detach().clone())
    return outs


def real_exam_pass_rate(
    motor: nn.Module,
    exam: List[torch.Tensor],
    reference: List[torch.Tensor],
    tolerance: float = 1e-6,
) -> float:
    cur = real_motor_outputs(motor, exam)
    hits = 0
    for a, b in zip(cur, reference):
        if a.shape == b.shape and torch.allclose(a, b, atol=tolerance):
            hits += 1
    return hits / len(exam) if exam else 0.0
