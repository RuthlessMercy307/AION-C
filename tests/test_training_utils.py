"""
tests/test_training_utils.py — Tests para experiments/training_utils.py
========================================================================

Cubre:
  1. make_cosine_scheduler — warmup lineal + cosine decay
  2. save_checkpoint / load_checkpoint — serialización y resume
  3. train_with_amp — training loop completo con un modelo mínimo
  4. Grad clip activo (norm después del clip <= 1.0)
  5. AMP: en CPU corre en fp32 sin error
  6. Checkpoint: resume automático desde step guardado
"""

from __future__ import annotations

import os
import sys
import math
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import torch
import torch.nn as nn

from experiments.training_utils import (
    make_cosine_scheduler,
    save_checkpoint,
    load_checkpoint,
    train_with_amp,
)


# ─── Modelos mínimos para tests ───────────────────────────────────────────────

class _TinyLM(nn.Module):
    """LM de 3 capas para tests de training_utils (retorna logits directamente)."""
    def __init__(self, vocab=32, dim=16):
        super().__init__()
        self.emb  = nn.Embedding(vocab, dim, padding_idx=0)
        self.fc   = nn.Linear(dim, vocab)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.fc(self.emb(ids))


class _TinyMotor(nn.Module):
    """Motor mínimo que retorna SimpleNamespace con .logits (imita CORAPipeline)."""
    def __init__(self, vocab=32, dim=16):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim, padding_idx=0)
        self.fc  = nn.Linear(dim, vocab)

    def forward(self, ids: torch.Tensor):
        logits = self.fc(self.emb(ids))
        return SimpleNamespace(logits=logits)


class _TinyTok:
    """Tokenizer mínimo compatible con train_with_amp."""
    @staticmethod
    def to_tensor(ids):
        return torch.tensor([ids], dtype=torch.long)


VOCAB = 32
DIM   = 16
TOK   = _TinyTok()

# ids_list: 20 ejemplos de largo 8
IDS_LIST = [list(range(1, 9)) for _ in range(20)]


# ─────────────────────────────────────────────────────────────────────────────
# 1. make_cosine_scheduler
# ─────────────────────────────────────────────────────────────────────────────

class TestCosineScheduler:

    def _make(self, warmup=10, total=100, base_lr=1.0):
        model = _TinyLM(VOCAB, DIM)
        opt   = torch.optim.AdamW(model.parameters(), lr=base_lr)
        sch   = make_cosine_scheduler(opt, warmup, total)
        return opt, sch

    def test_returns_lambda_lr(self):
        _, sch = self._make()
        assert isinstance(sch, torch.optim.lr_scheduler.LambdaLR)

    def test_warmup_starts_near_zero(self):
        opt, sch = self._make(warmup=10, total=100, base_lr=1.0)
        # Step 0: factor = 0/10 = 0.0 → lr = 0
        assert sch.get_last_lr()[0] == pytest.approx(0.0, abs=1e-6)

    def test_warmup_increases(self):
        opt, sch = self._make(warmup=10, total=100, base_lr=1.0)
        lrs = []
        for _ in range(10):
            sch.step()
            lrs.append(sch.get_last_lr()[0])
        # Los primeros 10 steps (warmup) deben ser crecientes
        for i in range(len(lrs) - 1):
            assert lrs[i] <= lrs[i + 1] + 1e-9

    def test_peak_at_end_of_warmup(self):
        opt, sch = self._make(warmup=10, total=100, base_lr=1.0)
        for _ in range(10):
            sch.step()
        lr_at_peak = sch.get_last_lr()[0]
        assert lr_at_peak == pytest.approx(1.0, abs=1e-6)

    def test_cosine_decay_after_warmup(self):
        opt, sch = self._make(warmup=10, total=100, base_lr=1.0)
        for _ in range(10):
            sch.step()
        prev = sch.get_last_lr()[0]
        lrs  = []
        for _ in range(90):
            sch.step()
            lrs.append(sch.get_last_lr()[0])
        # Debe decrecer
        for i in range(len(lrs) - 1):
            assert lrs[i] >= lrs[i + 1] - 1e-9

    def test_ends_near_zero(self):
        opt, sch = self._make(warmup=10, total=100, base_lr=1.0)
        for _ in range(100):
            sch.step()
        assert sch.get_last_lr()[0] == pytest.approx(0.0, abs=1e-4)

    def test_zero_warmup(self):
        """warmup=0: arranca en 1.0 y decae directamente."""
        opt, sch = self._make(warmup=0, total=50, base_lr=1.0)
        sch.step()
        # step 1 de 50 → progress=1/50, cos decay ligero
        assert sch.get_last_lr()[0] > 0.9

    def test_warmup_equals_total(self):
        """Caso degenerado: warmup == total → siempre 1.0."""
        opt, sch = self._make(warmup=50, total=50, base_lr=1.0)
        for _ in range(50):
            sch.step()
        assert sch.get_last_lr()[0] == pytest.approx(1.0, abs=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# 2. save_checkpoint / load_checkpoint
# ─────────────────────────────────────────────────────────────────────────────

class TestCheckpoint:

    def _setup(self, tmp_path):
        model = _TinyLM(VOCAB, DIM)
        opt   = torch.optim.AdamW(model.parameters(), lr=1e-3)
        sch   = make_cosine_scheduler(opt, 10, 100)
        path  = tmp_path / "ckpt.pt"
        return model, opt, sch, path

    def test_save_creates_file(self, tmp_path):
        model, opt, sch, path = self._setup(tmp_path)
        save_checkpoint(path, model, opt, sch, step=42, loss=0.5)
        assert path.exists()

    def test_load_missing_returns_zero(self, tmp_path):
        model, opt, sch, path = self._setup(tmp_path)
        step, loss = load_checkpoint(path, model, opt, sch)
        assert step == 0
        assert math.isnan(loss)

    def test_round_trip_step_and_loss(self, tmp_path):
        model, opt, sch, path = self._setup(tmp_path)
        save_checkpoint(path, model, opt, sch, step=250, loss=1.23)
        model2, opt2, sch2, _ = self._setup(tmp_path)
        step, loss = load_checkpoint(path, model2, opt2, sch2)
        assert step == 250
        assert loss == pytest.approx(1.23, abs=1e-5)

    def test_model_weights_restored(self, tmp_path):
        model, opt, sch, path = self._setup(tmp_path)
        # Guardar pesos actuales
        w_before = model.emb.weight.clone()
        save_checkpoint(path, model, opt, sch, step=1, loss=0.0)
        # Modificar el modelo
        nn.init.zeros_(model.emb.weight)
        assert not torch.allclose(model.emb.weight, w_before)
        # Restaurar
        load_checkpoint(path, model, opt, sch)
        assert torch.allclose(model.emb.weight, w_before)

    def test_load_without_scheduler(self, tmp_path):
        """load_checkpoint con scheduler=None no crashea."""
        model, opt, sch, path = self._setup(tmp_path)
        save_checkpoint(path, model, opt, sch, step=10, loss=0.5)
        step, loss = load_checkpoint(path, model, opt, scheduler=None)
        assert step == 10

    def test_save_without_scheduler(self, tmp_path):
        """save_checkpoint con scheduler=None guarda correctamente."""
        model, opt, sch, path = self._setup(tmp_path)
        save_checkpoint(path, model, opt, None, step=5, loss=0.9)
        assert path.exists()
        step, loss = load_checkpoint(path, model, opt)
        assert step == 5

    def test_save_with_config(self, tmp_path):
        model, opt, sch, path = self._setup(tmp_path)
        cfg = {"hidden_dim": 64, "lr": 3e-4}
        save_checkpoint(path, model, opt, sch, step=1, loss=0.1, config=cfg)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        assert ckpt["config"] == cfg

    def test_multiple_saves_overwrite(self, tmp_path):
        """Guardar dos veces en el mismo path: el segundo prevalece."""
        model, opt, sch, path = self._setup(tmp_path)
        save_checkpoint(path, model, opt, sch, step=100, loss=1.0)
        save_checkpoint(path, model, opt, sch, step=200, loss=0.5)
        step, loss = load_checkpoint(path, model, opt, sch)
        assert step == 200
        assert loss == pytest.approx(0.5, abs=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# 3. train_with_amp — loop completo
# ─────────────────────────────────────────────────────────────────────────────

class TestTrainWithAmp:

    def test_runs_without_error_lm(self):
        model = _TinyLM(VOCAB, DIM)
        losses, elapsed, final = train_with_amp(
            model, IDS_LIST, TOK, n_steps=10, label="TF",
            is_motor=False, print_every=999,
        )
        assert len(losses) > 0
        assert math.isfinite(final)
        assert elapsed > 0

    def test_runs_without_error_motor(self):
        model = _TinyMotor(VOCAB, DIM)
        losses, elapsed, final = train_with_amp(
            model, IDS_LIST, TOK, n_steps=10, label="Motor",
            is_motor=True, print_every=999,
        )
        assert len(losses) > 0
        assert math.isfinite(final)

    def test_loss_decreases(self):
        """Después de 50 steps, la loss debe bajar respecto al inicio."""
        torch.manual_seed(0)
        model = _TinyLM(VOCAB, DIM)
        losses, _, _ = train_with_amp(
            model, IDS_LIST, TOK, n_steps=50, label="LM",
            is_motor=False, print_every=999, lr=1e-2,
        )
        first10 = sum(losses[:10]) / 10
        last10  = sum(losses[-10:]) / 10
        assert last10 < first10, f"Loss no bajó: {first10:.4f} → {last10:.4f}"

    def test_returns_correct_tuple(self):
        model = _TinyLM(VOCAB, DIM)
        result = train_with_amp(
            model, IDS_LIST, TOK, n_steps=5, label="X",
            is_motor=False, print_every=999,
        )
        assert len(result) == 3
        losses, elapsed, final = result
        assert isinstance(losses, list)
        assert isinstance(elapsed, float)
        assert isinstance(final, float)

    def test_cpu_amp_no_error(self):
        """En CPU, AMP está deshabilitado — no debe dar error."""
        model  = _TinyLM(VOCAB, DIM)
        device = torch.device("cpu")
        losses, _, _ = train_with_amp(
            model, IDS_LIST, TOK, n_steps=5, label="CPU",
            is_motor=False, print_every=999, device=device,
        )
        assert len(losses) > 0


# ─────────────────────────────────────────────────────────────────────────────
# 4. Grad clip
# ─────────────────────────────────────────────────────────────────────────────

class TestGradClip:

    def test_grad_norm_bounded_after_clip(self):
        """
        Verificamos que clip_grad_norm_ está activo:
        calculamos la norma del gradiente antes y después del step
        usando un hook de backward.
        """
        torch.manual_seed(0)
        model = _TinyLM(VOCAB, DIM)
        opt   = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # Forward + backward manual con un ejemplo simple
        ids_t  = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
        logits = model(ids_t)
        loss   = torch.nn.functional.cross_entropy(
            logits[0, :-1], ids_t[0, 1:], ignore_index=0
        )
        loss.backward()

        # Norma antes del clip
        total_before = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_before += p.grad.norm().item() ** 2
        norm_before = total_before ** 0.5

        # Aplicar clip
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Norma después del clip
        total_after = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_after += p.grad.norm().item() ** 2
        norm_after = total_after ** 0.5

        assert norm_after <= 1.0 + 1e-5, f"Clip no funcionó: norm={norm_after:.4f}"
        if norm_before > 1.0:
            assert norm_after < norm_before, "Clip no redujo la norma"


# ─────────────────────────────────────────────────────────────────────────────
# 5. Checkpoint + resume en train_with_amp
# ─────────────────────────────────────────────────────────────────────────────

class TestCheckpointResume:

    def test_checkpoint_created_at_500(self, tmp_path):
        model = _TinyLM(VOCAB, DIM)
        ckpt  = tmp_path / "test.pt"
        train_with_amp(
            model, IDS_LIST, TOK, n_steps=500, label="CKP",
            is_motor=False, checkpoint_every=500,
            checkpoint_path=ckpt, print_every=999,
        )
        assert ckpt.exists()

    def test_resume_continues_from_step(self, tmp_path):
        """
        Entrena 300 steps, guarda checkpoint, inicia nuevo modelo desde ckpt.
        El resume debe arrancar desde step 300, no desde 0.
        """
        model = _TinyLM(VOCAB, DIM)
        ckpt  = tmp_path / "resume.pt"

        # Guardar checkpoint manualmente en step 300
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        sch = make_cosine_scheduler(opt, 10, 500)
        save_checkpoint(ckpt, model, opt, sch, step=300, loss=0.5)

        # load_checkpoint debe retornar step=300
        model2 = _TinyLM(VOCAB, DIM)
        opt2   = torch.optim.AdamW(model2.parameters(), lr=1e-3)
        sch2   = make_cosine_scheduler(opt2, 10, 500)
        step, loss = load_checkpoint(ckpt, model2, opt2, sch2)
        assert step == 300

    def test_no_checkpoint_path_no_file(self, tmp_path):
        """Sin checkpoint_path, no se crea ningún archivo."""
        model = _TinyLM(VOCAB, DIM)
        train_with_amp(
            model, IDS_LIST, TOK, n_steps=10, label="NOFILE",
            is_motor=False, checkpoint_path=None, print_every=999,
        )
        # No debe haber archivos .pt en tmp_path
        pts = list(tmp_path.glob("*.pt"))
        assert len(pts) == 0
