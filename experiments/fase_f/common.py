"""
experiments/fase_f/common.py — utilidades compartidas para los 5 experimentos.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn


RESULTS_DIR = Path(__file__).parent / "results"


# ════════════════════════════════════════════════════════════════════════════
# Fake motor — evita cargar un pipeline real en los experimentos
# ════════════════════════════════════════════════════════════════════════════

class FakeMotor(nn.Module):
    """Motor mínimo que imita la estructura .crystallizer/.cre de los reales.

    Determinista via seed, utilizado para probar adapters, pruning, etc.
    """

    def __init__(self, d_in: int = 8, d_hid: int = 16, seed: int = 0) -> None:
        super().__init__()
        self.crystallizer = nn.Sequential()
        self.crystallizer.project = nn.Linear(d_in, d_hid)
        self.crystallizer.out = nn.Linear(d_hid, d_hid)
        self.cre = nn.Sequential()
        self.cre.input_proj = nn.Linear(d_hid, d_hid)
        self.cre.message = nn.Linear(d_hid, d_hid)
        g = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for p in self.parameters():
                p.copy_(torch.empty_like(p).uniform_(-0.5, 0.5, generator=g))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.crystallizer.project(x))
        h = torch.relu(self.crystallizer.out(h))
        h = torch.relu(self.cre.input_proj(h))
        return self.cre.message(h)


MOTOR_TARGET_PATHS = [
    "crystallizer.project",
    "crystallizer.out",
    "cre.input_proj",
    "cre.message",
]


# ════════════════════════════════════════════════════════════════════════════
# Exam
# ════════════════════════════════════════════════════════════════════════════

def make_exam(n: int = 10, d: int = 8, seed: int = 7) -> List[torch.Tensor]:
    """Crea un exam determinista de N inputs."""
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(1, d, generator=g) for _ in range(n)]


def exam_outputs(motor: nn.Module, exam: List[torch.Tensor]) -> List[torch.Tensor]:
    out = []
    with torch.no_grad():
        for x in exam:
            out.append(motor(x).detach().clone())
    return out


def exam_pass_rate(
    motor: nn.Module,
    exam: List[torch.Tensor],
    reference: List[torch.Tensor],
    tolerance: float = 1e-6,
) -> float:
    """% de exámenes que siguen dando EXACTAMENTE la misma salida."""
    hits = 0
    with torch.no_grad():
        for x, y_ref in zip(exam, reference):
            y = motor(x)
            if torch.allclose(y, y_ref, atol=tolerance):
                hits += 1
    return hits / len(exam) if exam else 0.0


# ════════════════════════════════════════════════════════════════════════════
# Report
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class ExperimentReport:
    experiment_id: str
    name: str
    started_at: float
    ended_at: float
    passed: bool
    summary: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        return (self.ended_at - self.started_at) * 1000.0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["duration_ms"] = self.duration_ms
        return d


def write_report(report: ExperimentReport, results_dir: Path = RESULTS_DIR) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{report.experiment_id}.json"
    out_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return out_path
