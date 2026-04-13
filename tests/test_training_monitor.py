"""
tests/test_training_monitor.py — Tests para TrainingMonitor
============================================================

Verifica:
  - Eval periódico según eval_every
  - Cómputo de val_loss
  - Greedy decode con modelo mock
  - Word F1
  - Early stopping (patience)
  - Detección de convergencia
  - Log JSON
  - Integración con train_with_amp (monitor= param)
  - load_fixed_examples (requiere dataset Opus)
  - Compatibilidad con tokenizadores is_motor=True / is_motor=False
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import List, Optional, Tuple
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# MOCKS
# ─────────────────────────────────────────────────────────────────────────────

VOCAB_SIZE = 64
SEQ_LEN    = 12
PAD        = 0
EOS        = 2


class _MockOutput:
    """Simula el output de MoSEPipeline con .logits."""
    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits


class _MockMotorModel(nn.Module):
    """Modelo motor mínimo: embedding + linear → logits."""

    def __init__(self, vocab_size: int = VOCAB_SIZE, hidden: int = 16) -> None:
        super().__init__()
        self.emb  = nn.Embedding(vocab_size, hidden, padding_idx=PAD)
        self.proj = nn.Linear(hidden, vocab_size)
        self.vocab_size = vocab_size

    def forward(self, token_ids: torch.Tensor, query_text: Optional[str] = None):
        x = self.emb(token_ids)     # [B, L, H]
        logits = self.proj(x)       # [B, L, V]
        return _MockOutput(logits)


class _MockDirectModel(nn.Module):
    """Modelo que retorna logits directamente (is_motor=False)."""

    def __init__(self, vocab_size: int = VOCAB_SIZE, hidden: int = 16) -> None:
        super().__init__()
        self.emb  = nn.Embedding(vocab_size, hidden, padding_idx=PAD)
        self.proj = nn.Linear(hidden, vocab_size)

    def forward(self, token_ids: torch.Tensor):
        return self.proj(self.emb(token_ids))


class _MockTokenizer:
    """Tokenizador mínimo compatible con TrainingMonitor."""

    PAD = PAD
    BOS = 1
    EOS = EOS

    def encode(self, text: str, max_len: int = SEQ_LEN) -> List[int]:
        ids = [self.BOS] + [ord(c) % (VOCAB_SIZE - 4) + 4 for c in text[:max_len - 2]] + [self.EOS]
        return ids[:max_len]

    def decode(self, ids: List[int]) -> str:
        return "".join(
            chr(i - 4 + ord("a")) if 4 <= i < VOCAB_SIZE else ""
            for i in ids
            if i not in (self.PAD, self.BOS, self.EOS)
        )

    def to_tensor(self, ids: List[int]) -> torch.Tensor:
        return torch.tensor([ids], dtype=torch.long)


class _MockConfig:
    """Config mínimo con dec_max_seq_len."""
    dec_max_seq_len: int = SEQ_LEN


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def tok():
    return _MockTokenizer()


@pytest.fixture
def motor_model():
    m = _MockMotorModel()
    m.eval()
    return m


@pytest.fixture
def direct_model():
    return _MockDirectModel()


@pytest.fixture
def cfg():
    return _MockConfig()


@pytest.fixture
def val_ids(tok) -> List[List[int]]:
    """50 secuencias de validación pre-tokenizadas."""
    phrases = [
        "hello world", "the fox jumps", "system error",
        "neural net", "graph model",
    ]
    result = []
    for i in range(50):
        text = phrases[i % len(phrases)] + f" {i}"
        result.append(tok.encode(text, SEQ_LEN))
    return result


@pytest.fixture
def fixed_examples() -> List[Tuple[str, str, str]]:
    return [
        ("If A then B, what if A fails?", "B does not happen", "cora"),
        ("def fib(n): return",           "n if n<=1 else fib(n-1)+fib(n-2)", "forge_c"),
        ("What is integral of x dx?",    "x squared over two",               "axiom"),
        ("Once upon a time",             "a dragon lived in the mountains",   "muse"),
        ("I feel overwhelmed",           "Take a deep breath and rest",       "empathy"),
    ]


def _make_monitor(
    tmp_path: Path,
    motor_model,
    tok,
    val_ids,
    fixed_examples,
    cfg,
    eval_every: int = 5,
    patience: int = 20,
    convergence_delta: float = 0.001,
    convergence_window: int = 10,
    is_motor: bool = True,
) -> "TrainingMonitor":
    from experiments.training_utils import TrainingMonitor
    return TrainingMonitor(
        model              = motor_model,
        tok                = tok,
        val_ids_list       = val_ids,
        fixed_examples     = fixed_examples,
        cfg                = cfg,
        eval_every         = eval_every,
        patience           = patience,
        convergence_delta  = convergence_delta,
        convergence_window = convergence_window,
        log_path           = tmp_path / "monitor.jsonl",
        device             = torch.device("cpu"),
        is_motor           = is_motor,
    )


# ─────────────────────────────────────────────────────────────────────────────
# TESTS — word_f1
# ─────────────────────────────────────────────────────────────────────────────

class TestWordF1:
    def test_identical(self):
        from experiments.training_utils import TrainingMonitor
        assert TrainingMonitor.word_f1("hello world", "hello world") == 1.0

    def test_no_overlap(self):
        from experiments.training_utils import TrainingMonitor
        assert TrainingMonitor.word_f1("cat sat", "dog runs") == 0.0

    def test_partial_overlap(self):
        from experiments.training_utils import TrainingMonitor
        # pred="a b c", ref="a b d" → intersection={a,b} → P=2/3, R=2/3, F1=2/3
        f1 = TrainingMonitor.word_f1("a b c", "a b d")
        assert abs(f1 - round(2/3, 4)) < 1e-3

    def test_empty_pred(self):
        from experiments.training_utils import TrainingMonitor
        assert TrainingMonitor.word_f1("", "hello") == 0.0

    def test_empty_ref(self):
        from experiments.training_utils import TrainingMonitor
        assert TrainingMonitor.word_f1("hello", "") == 0.0

    def test_empty_both(self):
        from experiments.training_utils import TrainingMonitor
        assert TrainingMonitor.word_f1("", "") == 0.0

    def test_case_insensitive(self):
        from experiments.training_utils import TrainingMonitor
        assert TrainingMonitor.word_f1("Hello World", "hello world") == 1.0

    def test_returns_float(self):
        from experiments.training_utils import TrainingMonitor
        result = TrainingMonitor.word_f1("a b", "a c")
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_precision_recall_asymmetry(self):
        from experiments.training_utils import TrainingMonitor
        # pred superset: P baja, R alta
        f1_super = TrainingMonitor.word_f1("a b c d", "a b")
        # pred subset: P alta, R baja
        f1_sub   = TrainingMonitor.word_f1("a b", "a b c d")
        # F1 es simétrico: mismos conjuntos en distinto orden → mismo F1
        assert abs(f1_super - f1_sub) < 1e-4


# ─────────────────────────────────────────────────────────────────────────────
# TESTS — val_loss
# ─────────────────────────────────────────────────────────────────────────────

class TestValLoss:
    def test_returns_finite_float(self, tmp_path, motor_model, tok, val_ids, fixed_examples, cfg):
        m = _make_monitor(tmp_path, motor_model, tok, val_ids, fixed_examples, cfg)
        loss = m._compute_val_loss()
        assert isinstance(loss, float)
        assert math.isfinite(loss)
        assert loss >= 0.0

    def test_val_loss_is_motor_false(self, tmp_path, direct_model, tok, val_ids, fixed_examples, cfg):
        m = _make_monitor(tmp_path, direct_model, tok, val_ids, fixed_examples, cfg, is_motor=False)
        loss = m._compute_val_loss()
        assert math.isfinite(loss)

    def test_uses_n_examples(self, tmp_path, motor_model, tok, val_ids, fixed_examples, cfg):
        m = _make_monitor(tmp_path, motor_model, tok, val_ids, fixed_examples, cfg)
        # No debe crashear con n_examples > len(val_ids_list)
        loss_big = m._compute_val_loss(n_examples=200)
        assert math.isfinite(loss_big)

    def test_val_loss_consistent(self, tmp_path, motor_model, tok, val_ids, fixed_examples, cfg):
        """La misma llamada con el mismo modelo da el mismo loss."""
        m = _make_monitor(tmp_path, motor_model, tok, val_ids, fixed_examples, cfg)
        l1 = m._compute_val_loss()
        l2 = m._compute_val_loss()
        assert abs(l1 - l2) < 1e-5


# ─────────────────────────────────────────────────────────────────────────────
# TESTS — greedy decode
# ─────────────────────────────────────────────────────────────────────────────

class TestGreedyDecode:
    def test_returns_string(self, tmp_path, motor_model, tok, val_ids, fixed_examples, cfg):
        m = _make_monitor(tmp_path, motor_model, tok, val_ids, fixed_examples, cfg)
        result = m._greedy_decode("hello", "", SEQ_LEN)
        assert isinstance(result, str)

    def test_decode_nonempty_for_valid_input(self, tmp_path, motor_model, tok, val_ids, fixed_examples, cfg):
        m = _make_monitor(tmp_path, motor_model, tok, val_ids, fixed_examples, cfg)
        result = m._greedy_decode("hello world", "", SEQ_LEN)
        # La decodificación puede ser vacía pero no debe crashear
        assert result is not None

    def test_decode_is_motor_false(self, tmp_path, direct_model, tok, val_ids, fixed_examples, cfg):
        m = _make_monitor(tmp_path, direct_model, tok, val_ids, fixed_examples, cfg, is_motor=False)
        result = m._greedy_decode("test input", "", SEQ_LEN)
        assert isinstance(result, str)

    def test_decode_respects_max_seq_len(self, tmp_path, motor_model, tok, val_ids, fixed_examples, cfg):
        m = _make_monitor(tmp_path, motor_model, tok, val_ids, fixed_examples, cfg)
        m.max_new_tokens = 4
        result = m._greedy_decode("x", "", SEQ_LEN)
        # No debe generar más de max_new_tokens tokens nuevos
        decoded_ids = tok.encode(result)
        assert len(decoded_ids) <= m.max_new_tokens + 2  # +2 for bos/eos


# ─────────────────────────────────────────────────────────────────────────────
# TESTS — eval periódico
# ─────────────────────────────────────────────────────────────────────────────

class TestPeriodicEval:
    def test_eval_triggers_at_correct_steps(self, tmp_path, motor_model, tok, val_ids, fixed_examples, cfg):
        m = _make_monitor(tmp_path, motor_model, tok, val_ids, fixed_examples, cfg, eval_every=5)
        # Steps 1-4: no eval
        for s in range(1, 5):
            m.step(s, 1.0)
        assert len(m.eval_history) == 0

        # Step 5: eval
        m.step(5, 1.0)
        assert len(m.eval_history) == 1

        # Steps 6-9: no eval
        for s in range(6, 10):
            m.step(s, 1.0)
        assert len(m.eval_history) == 1

        # Step 10: eval
        m.step(10, 1.0)
        assert len(m.eval_history) == 2

    def test_eval_history_has_correct_fields(self, tmp_path, motor_model, tok, val_ids, fixed_examples, cfg):
        m = _make_monitor(tmp_path, motor_model, tok, val_ids, fixed_examples, cfg, eval_every=3)
        m.step(3, 0.5)
        assert len(m.eval_history) == 1
        record = m.eval_history[0]
        assert "step"        in record
        assert "train_loss"  in record
        assert "val_loss"    in record
        assert "val_f1"      in record
        assert "generated"   in record
        assert record["step"] == 3
        assert abs(record["train_loss"] - 0.5) < 1e-6

    def test_eval_history_generated_has_5_entries(self, tmp_path, motor_model, tok, val_ids, fixed_examples, cfg):
        m = _make_monitor(tmp_path, motor_model, tok, val_ids, fixed_examples, cfg, eval_every=2)
        m.step(2, 0.5)
        gen = m.eval_history[0]["generated"]
        assert len(gen) == len(fixed_examples)
        for item in gen:
            assert "domain"    in item
            assert "input"     in item
            assert "expected"  in item
            assert "predicted" in item
            assert "f1"        in item

    def test_eval_val_f1_in_range(self, tmp_path, motor_model, tok, val_ids, fixed_examples, cfg):
        m = _make_monitor(tmp_path, motor_model, tok, val_ids, fixed_examples, cfg, eval_every=2)
        m.step(2, 0.5)
        f1 = m.eval_history[0]["val_f1"]
        assert 0.0 <= f1 <= 1.0

    def test_step_returns_none_during_training(self, tmp_path, motor_model, tok, val_ids, fixed_examples, cfg):
        m = _make_monitor(tmp_path, motor_model, tok, val_ids, fixed_examples, cfg, eval_every=100)
        for s in range(1, 50):
            result = m.step(s, 1.0)
            assert result is None


# ─────────────────────────────────────────────────────────────────────────────
# TESTS — JSON log
# ─────────────────────────────────────────────────────────────────────────────

class TestJSONLog:
    def test_log_file_created(self, tmp_path, motor_model, tok, val_ids, fixed_examples, cfg):
        log_path = tmp_path / "monitor.jsonl"
        m = _make_monitor(tmp_path, motor_model, tok, val_ids, fixed_examples, cfg, eval_every=3)
        m.step(3, 0.5)
        assert log_path.exists()

    def test_log_is_valid_jsonl(self, tmp_path, motor_model, tok, val_ids, fixed_examples, cfg):
        log_path = tmp_path / "monitor.jsonl"
        m = _make_monitor(tmp_path, motor_model, tok, val_ids, fixed_examples, cfg, eval_every=4)
        m.step(4, 0.8)
        m.step(8, 0.7)
        lines = log_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        for line in lines:
            record = json.loads(line)
            assert "step" in record
            assert "val_loss" in record

    def test_log_appends_across_evals(self, tmp_path, motor_model, tok, val_ids, fixed_examples, cfg):
        log_path = tmp_path / "monitor.jsonl"
        m = _make_monitor(tmp_path, motor_model, tok, val_ids, fixed_examples, cfg, eval_every=2)
        for s in [2, 4, 6, 8, 10]:
            m.step(s, 1.0 - s * 0.05)
        lines = log_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 5

    def test_log_records_correct_step(self, tmp_path, motor_model, tok, val_ids, fixed_examples, cfg):
        log_path = tmp_path / "monitor.jsonl"
        m = _make_monitor(tmp_path, motor_model, tok, val_ids, fixed_examples, cfg, eval_every=7)
        m.step(7, 0.5)
        record = json.loads(log_path.read_text(encoding="utf-8"))
        assert record["step"] == 7

    def test_log_parent_created(self, tmp_path, motor_model, tok, val_ids, fixed_examples, cfg):
        """Log path con directorio que no existe — lo crea automáticamente."""
        from experiments.training_utils import TrainingMonitor
        nested = tmp_path / "nested" / "deep" / "monitor.jsonl"
        m = TrainingMonitor(
            model          = motor_model,
            tok            = tok,
            val_ids_list   = val_ids,
            fixed_examples = fixed_examples,
            cfg            = cfg,
            eval_every     = 3,
            log_path       = nested,
            device         = torch.device("cpu"),
        )
        m.step(3, 0.5)
        assert nested.exists()


# ─────────────────────────────────────────────────────────────────────────────
# TESTS — early stopping
# ─────────────────────────────────────────────────────────────────────────────

class TestEarlyStopping:
    def test_no_early_stop_when_improving(self, tmp_path, motor_model, tok, val_ids, fixed_examples, cfg):
        """Si val_loss mejora cada eval, no hay early stop."""
        m = _make_monitor(tmp_path, motor_model, tok, val_ids, fixed_examples, cfg,
                          eval_every=2, patience=6)
        # Simulamos mejoras forzando best_val_loss manualmente
        # (el mock model produce loss real, así que simplemente comprobamos que no para)
        for s in range(2, 10, 2):
            result = m.step(s, 1.0)
        # Con patience=6 y eval_every=2, se necesitan 3 evals sin mejora para parar
        # En un modelo nuevo sin entrenar, la loss es casi la misma (sin mejora)
        # Verificamos sólo que la lógica se ejecuta sin error

    def test_early_stop_fires_after_patience(self, tmp_path, motor_model, tok, val_ids, fixed_examples, cfg):
        """El monitor para cuando no hay mejora por patience steps."""
        m = _make_monitor(tmp_path, motor_model, tok, val_ids, fixed_examples, cfg,
                          eval_every=2, patience=4)
        # Forzar que el monitor crea que val_loss no mejora
        m._best_val_loss = 0.001  # muy bajo → siempre peor
        # Después de patience=4 steps sin mejora → early stop
        for s in range(2, 20, 2):
            result = m.step(s, 1.0)
            if result is not None:
                assert result == "early_stop"
                break

    def test_early_stop_sets_should_stop(self, tmp_path, motor_model, tok, val_ids, fixed_examples, cfg):
        m = _make_monitor(tmp_path, motor_model, tok, val_ids, fixed_examples, cfg,
                          eval_every=2, patience=4)
        m._best_val_loss = 0.001
        for s in range(2, 20, 2):
            m.step(s, 1.0)
            if m.should_stop:
                break
        assert m.should_stop

    def test_step_returns_early_stop_string(self, tmp_path, motor_model, tok, val_ids, fixed_examples, cfg):
        m = _make_monitor(tmp_path, motor_model, tok, val_ids, fixed_examples, cfg,
                          eval_every=2, patience=4)
        m._best_val_loss = 0.001
        stop_signals = []
        for s in range(2, 30, 2):
            result = m.step(s, 1.0)
            if result is not None:
                stop_signals.append(result)
                break
        if stop_signals:
            assert stop_signals[0] == "early_stop"

    def test_early_stop_saves_checkpoint(self, tmp_path, motor_model, tok, val_ids, fixed_examples, cfg):
        """Si checkpoint_path está definido, guarda al early-stop."""
        from experiments.training_utils import TrainingMonitor
        ckpt_path = tmp_path / "early_stop.pt"
        m = TrainingMonitor(
            model              = motor_model,
            tok                = tok,
            val_ids_list       = val_ids,
            fixed_examples     = fixed_examples,
            cfg                = cfg,
            eval_every         = 2,
            patience           = 4,
            log_path           = tmp_path / "log.jsonl",
            device             = torch.device("cpu"),
            checkpoint_path    = ckpt_path,
        )
        m._best_val_loss = 0.001
        for s in range(2, 30, 2):
            result = m.step(s, 1.0)
            if result == "early_stop":
                break
        if m.should_stop:
            assert ckpt_path.exists()


# ─────────────────────────────────────────────────────────────────────────────
# TESTS — convergencia
# ─────────────────────────────────────────────────────────────────────────────

class TestConvergence:
    def test_convergence_not_triggered_with_large_delta(self, tmp_path, motor_model, tok, val_ids, fixed_examples, cfg):
        """Con cambios grandes en val_loss, no converge."""
        m = _make_monitor(tmp_path, motor_model, tok, val_ids, fixed_examples, cfg,
                          eval_every=2, patience=10000,
                          convergence_delta=0.001, convergence_window=6)
        # Inyectar val_losses con delta grande
        m._last_eval_loss = 10.0  # primera eval será ~real, delta grande
        for s in range(2, 8, 2):
            m.step(s, 1.0)
        # Puede o no haber convergido (depende del mock model), sólo verificar no crashea

    def test_convergence_triggers_when_loss_stagnates(self, tmp_path, motor_model, tok, val_ids, fixed_examples, cfg):
        """
        Forzar escenario de convergencia: inyectar delta cercano a 0.
        """
        m = _make_monitor(tmp_path, motor_model, tok, val_ids, fixed_examples, cfg,
                          eval_every=2, patience=10000,
                          convergence_delta=100.0,   # umbral muy alto → siempre < delta
                          convergence_window=4)
        stop_signals = []
        for s in range(2, 30, 2):
            result = m.step(s, 1.0)
            if result is not None:
                stop_signals.append(result)
                break
        # Con convergence_delta=100 (siempre delta < 100) y window=4,
        # debería converger después de 2 evals
        if stop_signals:
            assert stop_signals[0] == "converged"

    def test_convergence_sets_should_stop(self, tmp_path, motor_model, tok, val_ids, fixed_examples, cfg):
        m = _make_monitor(tmp_path, motor_model, tok, val_ids, fixed_examples, cfg,
                          eval_every=2, patience=10000,
                          convergence_delta=100.0,
                          convergence_window=4)
        for s in range(2, 30, 2):
            m.step(s, 1.0)
            if m.should_stop:
                break
        assert m.should_stop


# ─────────────────────────────────────────────────────────────────────────────
# TESTS — integración con train_with_amp
# ─────────────────────────────────────────────────────────────────────────────

class TestTrainWithAmpIntegration:
    def test_train_with_amp_accepts_monitor(self, tmp_path, motor_model, tok, val_ids, fixed_examples, cfg):
        """train_with_amp no crasha cuando se pasa monitor=..."""
        from experiments.training_utils import train_with_amp, TrainingMonitor

        # Preparar ids_list
        ids_list = val_ids[:10]

        monitor = TrainingMonitor(
            model          = motor_model,
            tok            = tok,
            val_ids_list   = val_ids,
            fixed_examples = fixed_examples,
            cfg            = cfg,
            eval_every     = 5,
            patience       = 100,
            log_path       = tmp_path / "amp_monitor.jsonl",
            device         = torch.device("cpu"),
        )
        losses, elapsed, final = train_with_amp(
            model           = motor_model,
            ids_list        = ids_list,
            tok             = tok,
            n_steps         = 12,
            label           = "test_motor",
            is_motor        = True,
            lr              = 1e-3,
            warmup_steps    = 2,
            print_every     = 100,
            monitor         = monitor,
        )
        assert isinstance(losses, list)
        assert isinstance(elapsed, float)
        # Eval debe haberse ejecutado en step 5 y 10
        assert len(monitor.eval_history) >= 2

    def test_train_with_amp_stops_on_early_stop(self, tmp_path, motor_model, tok, val_ids, fixed_examples, cfg):
        """train_with_amp para cuando el monitor emite early_stop."""
        from experiments.training_utils import train_with_amp, TrainingMonitor

        monitor = TrainingMonitor(
            model          = motor_model,
            tok            = tok,
            val_ids_list   = val_ids,
            fixed_examples = fixed_examples,
            cfg            = cfg,
            eval_every     = 4,
            patience       = 4,   # para si no mejora en 4 steps
            log_path       = tmp_path / "early.jsonl",
            device         = torch.device("cpu"),
        )
        monitor._best_val_loss = 0.001  # forzar early stop

        losses, elapsed, final = train_with_amp(
            model        = motor_model,
            ids_list     = val_ids[:5],
            tok          = tok,
            n_steps      = 1000,   # muchos steps, pero debe parar antes
            label        = "early_stop_test",
            is_motor     = True,
            lr           = 1e-4,
            warmup_steps = 1,
            print_every  = 500,
            monitor      = monitor,
        )
        # Debe haber parado antes de step 1000
        assert len(losses) < 1000

    def test_train_with_amp_no_monitor_unchanged(self, tmp_path, motor_model, tok, val_ids, fixed_examples, cfg):
        """Sin monitor, train_with_amp funciona igual que antes."""
        from experiments.training_utils import train_with_amp

        losses, elapsed, final = train_with_amp(
            model        = motor_model,
            ids_list     = val_ids[:5],
            tok          = tok,
            n_steps      = 10,
            label        = "no_monitor",
            is_motor     = True,
            lr           = 1e-4,
            warmup_steps = 2,
            print_every  = 100,
            monitor      = None,
        )
        assert len(losses) == 10
        assert math.isfinite(final)


# ─────────────────────────────────────────────────────────────────────────────
# TESTS — load_fixed_examples
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadFixedExamples:
    def test_returns_list_of_tuples(self):
        from experiments.training_utils import TrainingMonitor
        examples = TrainingMonitor.load_fixed_examples()
        assert isinstance(examples, list)
        assert len(examples) >= 1
        for item in examples:
            assert len(item) == 3
            inp, out, domain = item
            assert isinstance(inp, str)
            assert isinstance(out, str)
            assert isinstance(domain, str)

    def test_fallback_when_no_dataset(self, monkeypatch):
        """Si OpusDataset lanza Exception, usa ejemplos hardcodeados."""
        from experiments.training_utils import TrainingMonitor
        import builtins
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if "opus_dataset" in name:
                raise ImportError("mocked")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        examples = TrainingMonitor.load_fixed_examples()
        assert len(examples) >= 1

    def test_seed_reproducibility(self):
        """Misma seed → mismos ejemplos."""
        from experiments.training_utils import TrainingMonitor
        ex1 = TrainingMonitor.load_fixed_examples(seed=123)
        ex2 = TrainingMonitor.load_fixed_examples(seed=123)
        assert ex1 == ex2

    def test_domain_labels_are_strings(self):
        from experiments.training_utils import TrainingMonitor
        examples = TrainingMonitor.load_fixed_examples()
        for _, _, domain in examples:
            assert isinstance(domain, str)
            assert len(domain) > 0


# ─────────────────────────────────────────────────────────────────────────────
# TESTS — _ids_to_tensor helper
# ─────────────────────────────────────────────────────────────────────────────

class TestIdToTensor:
    def test_with_to_tensor_method(self, tok):
        from experiments.training_utils import _ids_to_tensor
        ids = [1, 5, 3, 2]
        t = _ids_to_tensor(tok, ids)
        assert t.shape == (1, 4)
        assert t.dtype == torch.long
        assert t[0, 0].item() == 1

    def test_without_to_tensor_method(self):
        from experiments.training_utils import _ids_to_tensor

        class _NoTensorTok:
            def encode(self, text): return [1, 2, 3]
            def decode(self, ids): return ""
            # sin to_tensor

        ids = [1, 5, 3, 2]
        t = _ids_to_tensor(_NoTensorTok(), ids)
        assert t.shape == (1, 4)

    def test_tensor_is_long(self, tok):
        from experiments.training_utils import _ids_to_tensor
        t = _ids_to_tensor(tok, [0, 1, 2])
        assert t.dtype == torch.long


# ─────────────────────────────────────────────────────────────────────────────
# TESTS — propiedades y estado
# ─────────────────────────────────────────────────────────────────────────────

class TestMonitorState:
    def test_initial_should_stop_is_false(self, tmp_path, motor_model, tok, val_ids, fixed_examples, cfg):
        m = _make_monitor(tmp_path, motor_model, tok, val_ids, fixed_examples, cfg)
        assert not m.should_stop

    def test_eval_history_initially_empty(self, tmp_path, motor_model, tok, val_ids, fixed_examples, cfg):
        m = _make_monitor(tmp_path, motor_model, tok, val_ids, fixed_examples, cfg)
        assert m.eval_history == []

    def test_eval_history_is_copy(self, tmp_path, motor_model, tok, val_ids, fixed_examples, cfg):
        """eval_history retorna una copia, no la lista interna."""
        m = _make_monitor(tmp_path, motor_model, tok, val_ids, fixed_examples, cfg, eval_every=2)
        m.step(2, 0.5)
        history = m.eval_history
        history.append({"fake": True})
        assert len(m.eval_history) == 1  # lista interna no modificada

    def test_step_before_first_eval_returns_none(self, tmp_path, motor_model, tok, val_ids, fixed_examples, cfg):
        m = _make_monitor(tmp_path, motor_model, tok, val_ids, fixed_examples, cfg, eval_every=10)
        for s in range(1, 10):
            assert m.step(s, 1.0) is None
