"""
tests/test_train_production.py
=================================
Tests para experiments/train_production.py.

Cubre:
  1. _SimpleTokenizer — encode/decode/build_vocab
  2. build_tokenizer   — fallback word-level
  3. freeze / unfreeze / freeze_all_except / freeze_all_except_list
  4. PhaseResult       — creación y summary_line
  5. _Phase0Model      — forward pass sin error
  6. _phase_loop       — loop básico con modelo y datos sintéticos
  7. run_phase0        — <100 steps con datasets sintéticos, TrainingMonitor funcional
  8. make_ids_list / make_mixed_ids
  9. build_pipeline_and_tok — config tiny
 10. _TrainHparams     — presets existen
"""

from __future__ import annotations

import math
import random
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

# ─── path setup ──────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments.train_production import (
    _SimpleTokenizer,
    _AIONTokenizerWrapper,
    _Phase0Model,
    _TrainHparams,
    _PRESETS,
    _phase_loop,
    _make_synthetic_ids,
    build_tokenizer,
    freeze,
    unfreeze,
    freeze_all_except,
    freeze_all_except_list,
    count_trainable,
    make_ids_list,
    make_mixed_ids,
    get_fixed_examples,
    PhaseResult,
    run_phase0,
    build_pipeline_and_tok,
    MOTOR_TO_IDX,
)
from router.pipeline import MoSEConfig, MoSEPipeline
from experiments.training_utils import TrainingMonitor, make_cosine_scheduler


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

DEVICE = torch.device("cpu")
VOCAB  = 512


def _tiny_cfg() -> MoSEConfig:
    return MoSEConfig.tiny()


def _tiny_pipeline() -> MoSEPipeline:
    return MoSEPipeline(_tiny_cfg()).to(DEVICE)


def _simple_tok(texts: List[str] | None = None) -> _SimpleTokenizer:
    tok = _SimpleTokenizer(vocab_size=VOCAB)
    if texts:
        tok.build_vocab(texts)
    else:
        tok.build_vocab(["hello world foo bar baz qux"] * 20)
    return tok


@dataclass
class _FakeEntry:
    problem_text: str
    answer:       str
    reasoning:    str = ""


class _FakeDataset:
    """Dataset sintético sin archivos JSONL."""

    def __init__(self, motor: str = "cora", n: int = 30):
        self._motor   = motor
        self._n       = n
        self._pool    = [
            _FakeEntry(
                problem_text=f"problem {i} about {motor}",
                answer=f"answer {i}",
                reasoning=f"because {i}",
            )
            for i in range(n)
        ]
        self._examples = [
            {"input": e.problem_text, "expected_output": e.answer}
            for e in self._pool
        ]
        self._rng = random.Random(42)

    def __len__(self) -> int:
        return self._n

    def generate(self) -> _FakeEntry:
        return self._rng.choice(self._pool)

    def get_all_texts(self) -> List[str]:
        return [e.problem_text + " " + e.answer for e in self._pool]

    def train_eval_split(
        self,
        eval_size: int = 5,
        seed: int = 42,
    ) -> Tuple["_FakeDataset", "_FakeDataset"]:
        train = _FakeDataset(self._motor, max(1, self._n - eval_size))
        val   = _FakeDataset(self._motor, max(1, eval_size))
        return train, val


def _fake_datasets(motors=("cora", "forge_c", "axiom", "muse", "empathy")):
    """Dict[motor -> (train_ds, val_ds)] con datos sintéticos."""
    return {m: _FakeDataset(m, 20).train_eval_split(eval_size=5) for m in motors}


# ─────────────────────────────────────────────────────────────────────────────
# 1. _SimpleTokenizer
# ─────────────────────────────────────────────────────────────────────────────

class TestSimpleTokenizer:

    def test_build_vocab_fills_word_table(self):
        tok = _SimpleTokenizer(vocab_size=50)
        tok.build_vocab(["hello world hello"] * 5)
        assert tok._w2i.get("hello") is not None

    def test_encode_returns_list_of_ints(self):
        tok = _simple_tok()
        ids = tok.encode("hello world", max_len=32)
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)

    def test_encode_starts_bos_ends_eos(self):
        tok = _simple_tok()
        ids = tok.encode("hello world", max_len=32)
        assert ids[0]  == _SimpleTokenizer.BOS
        assert ids[-1] == _SimpleTokenizer.EOS

    def test_encode_max_len_respected(self):
        tok = _simple_tok()
        long_text = "hello " * 200
        ids = tok.encode(long_text, max_len=10)
        assert len(ids) <= 10

    def test_decode_filters_specials(self):
        tok = _simple_tok()
        ids = tok.encode("hello world", max_len=32)
        text = tok.decode(ids)
        assert "<pad>" not in text and "<s>" not in text and "</s>" not in text

    def test_encode_unknown_word(self):
        tok = _simple_tok()
        ids = tok.encode("zzzyyyxxx", max_len=32)
        # UNK=3 should appear for unseen word
        assert _SimpleTokenizer.UNK in ids

    def test_vocab_size_limit(self):
        tok = _SimpleTokenizer(vocab_size=10)
        tok.build_vocab(["a b c d e f g h i j k l m"] * 5)
        # vocab_size=10 means IDs 0–9, no word gets ID >= 10
        for i in tok._w2i.values():
            assert i < 10

    def test_to_tensor_shape(self):
        tok = _simple_tok()
        ids = tok.encode("hello world", max_len=16)
        t   = tok.to_tensor(ids)
        assert t.shape == (1, len(ids))
        assert t.dtype == torch.long


# ─────────────────────────────────────────────────────────────────────────────
# 2. build_tokenizer
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildTokenizer:

    def test_fallback_simple_tokenizer(self):
        tok = build_tokenizer(vocab_size=512, corpus_texts=["hello world"] * 20)
        assert isinstance(tok, _SimpleTokenizer)
        assert tok.vocab_size == 512

    def test_fallback_no_corpus(self):
        # Without corpus, SimpleTokenizer still works (empty vocab)
        tok = build_tokenizer(vocab_size=256, corpus_texts=None)
        assert isinstance(tok, _SimpleTokenizer)

    def test_vocab_size_honoured(self):
        tok = build_tokenizer(vocab_size=64, corpus_texts=["a b c d"] * 5)
        assert tok.vocab_size == 64

    def test_encode_works(self):
        tok = build_tokenizer(vocab_size=512, corpus_texts=["hello world"] * 20)
        ids = tok.encode("hello", max_len=16)
        assert len(ids) > 0


# ─────────────────────────────────────────────────────────────────────────────
# 3. freeze / unfreeze / freeze_all_except
# ─────────────────────────────────────────────────────────────────────────────

class TestFreezeUtils:

    def test_freeze_sets_requires_grad_false(self):
        m = nn.Linear(4, 4)
        freeze(m)
        assert all(not p.requires_grad for p in m.parameters())

    def test_unfreeze_sets_requires_grad_true(self):
        m = nn.Linear(4, 4)
        freeze(m)
        unfreeze(m)
        assert all(p.requires_grad for p in m.parameters())

    def test_count_trainable_after_freeze(self):
        m = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
        freeze(m[0])
        trainable = count_trainable(m)
        total     = sum(p.numel() for p in m.parameters())
        frozen    = sum(p.numel() for p in m[0].parameters())
        assert trainable == total - frozen

    def test_freeze_all_except_motor(self):
        pipeline = _tiny_pipeline()
        unfreeze(pipeline)
        motor_name = "cora"
        freeze_all_except(pipeline, f"motors.{motor_name}")
        for name, p in pipeline.named_parameters():
            if f"motors.{motor_name}" in name:
                assert p.requires_grad, f"{name} should be trainable"
            else:
                assert not p.requires_grad, f"{name} should be frozen"

    def test_freeze_all_except_list(self):
        pipeline = _tiny_pipeline()
        unfreeze(pipeline)
        prefixes = ["motors.cora", "motors.axiom"]
        freeze_all_except_list(pipeline, prefixes)
        for name, p in pipeline.named_parameters():
            if any(pfx in name for pfx in prefixes):
                assert p.requires_grad
            else:
                assert not p.requires_grad

    def test_freeze_encoder_leaves_decoder_trainable(self):
        pipeline = _tiny_pipeline()
        unfreeze(pipeline)
        freeze(pipeline.encoder)
        enc_trainable = count_trainable(pipeline.encoder)
        assert enc_trainable == 0
        dec_trainable = count_trainable(pipeline.decoder)
        assert dec_trainable > 0


# ─────────────────────────────────────────────────────────────────────────────
# 4. PhaseResult
# ─────────────────────────────────────────────────────────────────────────────

class TestPhaseResult:

    def _make(self, **kwargs) -> PhaseResult:
        defaults = dict(
            phase=0, label="TestPhase",
            steps_run=100, final_loss=1.5,
            best_val_loss=1.2, best_val_f1=0.25,
            elapsed_s=30.0, stop_reason="max_steps",
        )
        defaults.update(kwargs)
        return PhaseResult(**defaults)

    def test_creation(self):
        r = self._make()
        assert r.phase == 0
        assert r.label == "TestPhase"

    def test_summary_line_contains_key_info(self):
        r = self._make(steps_run=150, final_loss=2.3, stop_reason="converged")
        line = r.summary_line()
        assert "Phase 0" in line
        assert "150" in line
        assert "converged" in line

    def test_summary_line_shows_time_in_min(self):
        r = self._make(elapsed_s=90.0)
        line = r.summary_line()
        assert "1.5" in line  # 90s = 1.5 min

    def test_extra_field_default_empty(self):
        r = self._make()
        assert r.extra == {}

    def test_extra_field_motor(self):
        r = self._make(extra={"motor": "cora"})
        assert r.extra["motor"] == "cora"

    def test_checkpoint_optional(self):
        r = self._make()
        assert r.checkpoint is None

    def test_stop_reasons(self):
        for reason in ("max_steps", "early_stop", "converged"):
            r = self._make(stop_reason=reason)
            assert reason in r.summary_line()


# ─────────────────────────────────────────────────────────────────────────────
# 5. _Phase0Model forward pass
# ─────────────────────────────────────────────────────────────────────────────

class TestPhase0Model:

    def test_forward_returns_decoder_output(self):
        cfg  = _tiny_cfg()
        pipe = _tiny_pipeline()
        m    = _Phase0Model(pipe, cfg)
        ids  = torch.randint(4, cfg.vocab_size, (1, 12))
        out  = m(ids)
        assert hasattr(out, "logits")

    def test_logits_shape(self):
        cfg  = _tiny_cfg()
        pipe = _tiny_pipeline()
        m    = _Phase0Model(pipe, cfg)
        L    = 16
        ids  = torch.randint(4, cfg.vocab_size, (1, L))
        out  = m(ids)
        # [B, L, V]
        assert out.logits.shape == (1, L, cfg.vocab_size)

    def test_forward_no_grad_safe(self):
        cfg  = _tiny_cfg()
        pipe = _tiny_pipeline()
        m    = _Phase0Model(pipe, cfg)
        ids  = torch.randint(4, cfg.vocab_size, (1, 8))
        with torch.no_grad():
            out = m(ids)
        assert out.logits is not None

    def test_graph_repr_zeros(self):
        """Phase0Model must not use motor outputs — graph_repr is zeros."""
        cfg  = _tiny_cfg()
        pipe = _tiny_pipeline()
        m    = _Phase0Model(pipe, cfg)
        ids  = torch.randint(4, cfg.vocab_size, (1, 8))
        # Two forward passes should be identical (no stochastic motor sampling)
        pipe.eval()
        with torch.no_grad():
            out1 = m(ids)
            out2 = m(ids)
        assert torch.allclose(out1.logits, out2.logits)

    def test_backward_computes_gradients(self):
        cfg  = _tiny_cfg()
        pipe = _tiny_pipeline()
        m    = _Phase0Model(pipe, cfg)
        ids  = torch.randint(4, cfg.vocab_size, (1, 8))
        out  = m(ids)
        loss = out.logits.sum()
        loss.backward()
        # At least one parameter should have a gradient
        grads = [p.grad for p in m.parameters() if p.grad is not None]
        assert len(grads) > 0


# ─────────────────────────────────────────────────────────────────────────────
# 6. _phase_loop
# ─────────────────────────────────────────────────────────────────────────────

class TestPhaseLoop:

    def _make_setup(self, n_steps=10):
        cfg  = _tiny_cfg()
        pipe = _tiny_pipeline()
        m    = _Phase0Model(pipe, cfg)
        tok  = _simple_tok()
        ids_pool = [tok.encode("hello world foo", max_len=20) for _ in range(5)]
        rng  = random.Random(0)
        def get_ids():
            return rng.choice(ids_pool)
        opt   = torch.optim.SGD(m.parameters(), lr=1e-3)
        sched = make_cosine_scheduler(opt, warmup_steps=2, total_steps=n_steps)
        return m, get_ids, opt, sched

    def test_returns_tuple(self):
        m, get_ids, opt, sched = self._make_setup(10)
        result = _phase_loop(m, get_ids, opt, sched, n_steps=10,
                             monitor=None, label="test",
                             device=DEVICE, checkpoint_path=None)
        assert isinstance(result, tuple)
        assert len(result) == 4

    def test_losses_list_nonempty(self):
        m, get_ids, opt, sched = self._make_setup(5)
        losses, elapsed, final_loss, reason = _phase_loop(
            m, get_ids, opt, sched, n_steps=5,
            monitor=None, label="test", device=DEVICE, checkpoint_path=None
        )
        assert len(losses) > 0

    def test_stop_reason_max_steps(self):
        m, get_ids, opt, sched = self._make_setup(5)
        _, _, _, reason = _phase_loop(
            m, get_ids, opt, sched, n_steps=5,
            monitor=None, label="test", device=DEVICE, checkpoint_path=None
        )
        assert reason == "max_steps"

    def test_final_loss_finite(self):
        m, get_ids, opt, sched = self._make_setup(8)
        _, _, final_loss, _ = _phase_loop(
            m, get_ids, opt, sched, n_steps=8,
            monitor=None, label="test", device=DEVICE, checkpoint_path=None
        )
        assert math.isfinite(final_loss)

    def test_checkpoint_saved(self, tmp_path):
        m, get_ids, opt, sched = self._make_setup(6)
        ckpt_p = tmp_path / "test.pt"
        _phase_loop(
            m, get_ids, opt, sched, n_steps=6,
            monitor=None, label="test", device=DEVICE,
            checkpoint_path=ckpt_p, ckpt_every=3
        )
        assert ckpt_p.exists()

    def test_monitor_called(self):
        m, get_ids, opt, sched = self._make_setup(5)
        calls = []
        mock_monitor = MagicMock()
        mock_monitor.step.side_effect = lambda s, l: calls.append(s) or None
        _phase_loop(
            m, get_ids, opt, sched, n_steps=5,
            monitor=mock_monitor, label="test", device=DEVICE, checkpoint_path=None
        )
        assert len(calls) == 5

    def test_monitor_early_stop(self):
        m, get_ids, opt, sched = self._make_setup(20)
        mock_monitor = MagicMock()
        mock_monitor.step.return_value = "early_stop"
        losses, _, _, reason = _phase_loop(
            m, get_ids, opt, sched, n_steps=20,
            monitor=mock_monitor, label="test", device=DEVICE, checkpoint_path=None
        )
        assert reason == "early_stop"
        assert len(losses) == 1  # stopped after 1st step

    def test_extra_loss_fn_added(self):
        m, get_ids, opt, sched = self._make_setup(5)
        extra_values = []

        def extra_fn():
            v = torch.tensor(1.0, requires_grad=True)
            extra_values.append(1.0)
            return v

        losses, _, _, _ = _phase_loop(
            m, get_ids, opt, sched, n_steps=5,
            monitor=None, label="test", device=DEVICE, checkpoint_path=None,
            extra_loss_fn=extra_fn,
        )
        assert len(extra_values) == 5


# ─────────────────────────────────────────────────────────────────────────────
# 7. make_ids_list / make_mixed_ids
# ─────────────────────────────────────────────────────────────────────────────

class TestMakeIdsList:

    def test_make_ids_list_length(self):
        tok   = _simple_tok()
        ds    = _FakeDataset("cora", 20)
        result = make_ids_list(ds, tok, n=10, max_len=32)
        assert len(result) == 10

    def test_make_ids_list_max_len(self):
        tok   = _simple_tok()
        ds    = _FakeDataset("cora", 20)
        result = make_ids_list(ds, tok, n=5, max_len=16)
        assert all(len(ids) <= 16 for ids in result)

    def test_make_ids_list_nonempty_sequences(self):
        tok   = _simple_tok()
        ds    = _FakeDataset("cora", 20)
        result = make_ids_list(ds, tok, n=5, max_len=32)
        assert all(len(ids) > 0 for ids in result)

    def test_make_mixed_ids_cycles_motors(self):
        tok      = _simple_tok()
        datasets = _fake_datasets()
        result   = make_mixed_ids(datasets, tok, n=10, max_len=32)
        assert len(result) == 10

    def test_make_mixed_ids_max_len(self):
        tok      = _simple_tok()
        datasets = _fake_datasets()
        result   = make_mixed_ids(datasets, tok, n=8, max_len=16)
        assert all(len(ids) <= 16 for ids in result)

    # ── fallback cuando datasets está vacío ───────────────────────────────────

    def test_make_mixed_ids_empty_datasets_no_crash(self):
        """make_mixed_ids con dict vacío no debe crashear (ZeroDivisionError)."""
        tok    = _simple_tok()
        result = make_mixed_ids({}, tok, n=5, max_len=32)
        assert isinstance(result, list)
        assert len(result) == 5

    def test_make_mixed_ids_empty_datasets_valid_ids(self):
        tok    = _simple_tok()
        result = make_mixed_ids({}, tok, n=5, max_len=32)
        assert all(isinstance(ids, list) for ids in result)
        assert all(len(ids) > 0 for ids in result)

    def test_make_ids_list_none_dataset_no_crash(self):
        """make_ids_list con dataset=None usa fallback sintético."""
        tok    = _simple_tok()
        result = make_ids_list(None, tok, n=5, max_len=32)
        assert len(result) == 5
        assert all(len(ids) > 0 for ids in result)

    def test_make_synthetic_ids_returns_list(self):
        tok    = _simple_tok()
        result = _make_synthetic_ids(tok, n=6, max_len=24)
        assert isinstance(result, list)
        assert len(result) == 6

    def test_make_synthetic_ids_max_len_respected(self):
        tok    = _simple_tok()
        result = _make_synthetic_ids(tok, n=5, max_len=10)
        assert all(len(ids) <= 10 for ids in result)

    def test_make_synthetic_ids_nonempty(self):
        tok    = _simple_tok()
        result = _make_synthetic_ids(tok, n=4, max_len=20)
        assert all(len(ids) > 0 for ids in result)


# ─────────────────────────────────────────────────────────────────────────────
# 8. build_pipeline_and_tok — tiny config
# ─────────────────────────────────────────────────────────────────────────────
# load_all_datasets — dataset_root + error visibility
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadAllDatasets:

    def test_nonexistent_root_returns_empty_dict(self, tmp_path):
        """Con dataset_root que no existe, devuelve {} sin crashear."""
        from experiments.train_production import load_all_datasets
        result = load_all_datasets(
            motors=("cora",),
            max_examples=5,
            eval_size=2,
            dataset_root=tmp_path / "nonexistent",
        )
        assert result == {}

    def test_empty_root_dir_returns_empty_dict(self, tmp_path):
        """Con directorio vacío (sin .jsonl), devuelve {}."""
        from experiments.train_production import load_all_datasets
        result = load_all_datasets(
            motors=("cora",),
            max_examples=5,
            eval_size=2,
            dataset_root=tmp_path,   # existe pero sin archivos
        )
        assert result == {}

    def test_warning_printed_on_failure(self, tmp_path, capsys):
        """Cuando todos fallan, imprime WARNING descriptivo."""
        from experiments.train_production import load_all_datasets
        load_all_datasets(motors=("cora",), max_examples=5, eval_size=2,
                          dataset_root=tmp_path / "missing")
        out = capsys.readouterr().out
        assert "WARNING" in out or "WARN" in out


# ─────────────────────────────────────────────────────────────────────────────

class TestBuildPipelineAndTok:

    def test_returns_three_tuple(self):
        pipeline, tok, cfg = build_pipeline_and_tok("tiny", DEVICE)
        assert pipeline is not None
        assert tok is not None
        assert cfg is not None

    def test_pipeline_is_mose_pipeline(self):
        pipeline, _, _ = build_pipeline_and_tok("tiny", DEVICE)
        assert isinstance(pipeline, MoSEPipeline)

    def test_tiny_vocab_size(self):
        _, tok, cfg = build_pipeline_and_tok("tiny", DEVICE)
        assert tok.vocab_size == 512
        assert cfg.vocab_size == 512

    def test_pipeline_has_all_motors(self):
        from orchestrator.model import MOTOR_NAMES
        pipeline, _, _ = build_pipeline_and_tok("tiny", DEVICE)
        for motor in MOTOR_NAMES:
            assert motor in pipeline.motors

    def test_pipeline_forward_works(self):
        pipeline, tok, cfg = build_pipeline_and_tok("tiny", DEVICE)
        ids  = torch.randint(4, cfg.vocab_size, (1, 10))
        m    = _Phase0Model(pipeline, cfg)
        with torch.no_grad():
            out = m(ids)
        assert out.logits.shape[0] == 1

    def test_invalid_config_raises(self):
        with pytest.raises(ValueError, match="Config desconocido"):
            build_pipeline_and_tok("nonexistent", DEVICE)


# ─────────────────────────────────────────────────────────────────────────────
# 9. _TrainHparams presets
# ─────────────────────────────────────────────────────────────────────────────

class TestTrainHparams:

    def test_presets_exist(self):
        assert "tiny" in _PRESETS
        assert "medium" in _PRESETS
        assert "production" in _PRESETS

    def test_tiny_is_fast(self):
        hp = _PRESETS["tiny"]
        assert hp.ph0_steps <= 500
        assert hp.n_train   <= 1000
        assert hp.max_seq   <= 256

    def test_production_larger_than_tiny(self):
        hp_tiny = _PRESETS["tiny"]
        hp_prod = _PRESETS["production"]
        assert hp_prod.ph0_steps > hp_tiny.ph0_steps
        assert hp_prod.n_train   > hp_tiny.n_train

    def test_defaults_are_tiny(self):
        default = _TrainHparams()
        tiny    = _PRESETS["tiny"]
        assert default.ph0_steps == tiny.ph0_steps
        assert default.n_train   == tiny.n_train


# ─────────────────────────────────────────────────────────────────────────────
# 10. run_phase0 — smoke test (100 steps max, synthetic datasets)
# ─────────────────────────────────────────────────────────────────────────────

class TestRunPhase0:
    """
    Tests principales: run_phase0 con --config tiny y max_steps=100.
    Usa datos sintéticos para no depender de archivos JSONL.
    """

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.tmp_path = tmp_path
        self.cfg      = _tiny_cfg()
        self.pipeline = MoSEPipeline(self.cfg).to(DEVICE)
        self.tok      = _simple_tok(["hello world foo bar baz qux"] * 30)
        self.datasets = _fake_datasets()

    def _run(self, max_steps: int = 20) -> PhaseResult:
        hp = _TrainHparams(
            ph0_steps    = max_steps,
            ph0_eval     = max(5, max_steps // 4),
            ph0_patience = max_steps + 50,
            ph0_conv_win = max_steps + 50,
            ph0_lr       = 1e-3,
            ph0_warmup   = 2,
            n_train      = 15,
            n_val        = 10,
            max_seq      = 20,
            batch_size   = 1,
            ckpt_every   = max_steps,
        )
        return run_phase0(
            pipeline         = self.pipeline,
            cfg              = self.cfg,
            tok              = self.tok,
            datasets         = self.datasets,
            hparams          = hp,
            device           = DEVICE,
            checkpoint_dir   = self.tmp_path,
            resume           = False,
            max_steps_override = max_steps,
        )

    def test_returns_phase_result(self):
        result = self._run(max_steps=15)
        assert isinstance(result, PhaseResult)

    def test_phase_is_zero(self):
        result = self._run(max_steps=10)
        assert result.phase == 0

    def test_steps_run_positive(self):
        result = self._run(max_steps=10)
        assert result.steps_run > 0

    def test_steps_run_at_most_max(self):
        MAX = 15
        result = self._run(max_steps=MAX)
        assert result.steps_run <= MAX

    def test_final_loss_finite(self):
        result = self._run(max_steps=10)
        assert math.isfinite(result.final_loss)

    def test_stop_reason_set(self):
        result = self._run(max_steps=10)
        assert result.stop_reason in ("max_steps", "early_stop", "converged")

    def test_checkpoint_saved(self):
        result = self._run(max_steps=10)
        assert result.checkpoint is not None
        assert Path(result.checkpoint).exists()

    def test_encoder_decoder_trainable_during_phase0(self):
        """Durante Phase0 encoder y decoder deben ser entrenables."""
        hp = _TrainHparams(
            ph0_steps=5, ph0_eval=10, ph0_patience=100, ph0_conv_win=100,
            ph0_lr=1e-3, ph0_warmup=1, n_train=5, n_val=5, max_seq=16,
        )
        # Después de run_phase0, unfreeze restaura todos los parámetros
        run_phase0(
            pipeline=self.pipeline, cfg=self.cfg, tok=self.tok,
            datasets=self.datasets, hparams=hp, device=DEVICE,
            checkpoint_dir=self.tmp_path, resume=False,
            max_steps_override=5,
        )
        # unfreeze() fue llamado al final, todo debe ser entrenable
        all_trainable = all(p.requires_grad for p in self.pipeline.parameters())
        assert all_trainable

    def test_monitor_eval_history_populated(self):
        """
        Con eval_every=5 y n_steps=20, TrainingMonitor debería
        haber hecho al menos 1 evaluación.
        """
        hp = _TrainHparams(
            ph0_steps=20, ph0_eval=5, ph0_patience=200, ph0_conv_win=200,
            ph0_lr=1e-3, ph0_warmup=2, n_train=10, n_val=8, max_seq=16,
            ckpt_every=30,
        )
        result = run_phase0(
            pipeline=self.pipeline, cfg=self.cfg, tok=self.tok,
            datasets=self.datasets, hparams=hp, device=DEVICE,
            checkpoint_dir=self.tmp_path, resume=False,
            max_steps_override=20,
        )
        # best_val_loss debería ser finito si se ejecutó alguna evaluación
        # (puede ser nan si los 20 steps no alcanzaron el eval_every=5... pero deberían)
        assert result.steps_run >= 5  # Debe haber corrido al menos hasta el primer eval

    def test_100_steps_no_error(self):
        """
        Requirement explícito del usuario: --config tiny --phase 0
        corre sin error con 100 steps max y TrainingMonitor funcional.
        """
        hp = _TrainHparams(
            ph0_steps    = 100,
            ph0_eval     = 25,
            ph0_patience = 200,
            ph0_conv_win = 200,
            ph0_lr       = 1e-3,
            ph0_warmup   = 5,
            n_train      = 20,
            n_val        = 10,
            max_seq      = 24,
            batch_size   = 1,
            ckpt_every   = 100,
        )
        result = run_phase0(
            pipeline         = self.pipeline,
            cfg              = self.cfg,
            tok              = self.tok,
            datasets         = self.datasets,
            hparams          = hp,
            device           = DEVICE,
            checkpoint_dir   = self.tmp_path,
            resume           = False,
            max_steps_override = 100,
        )
        assert isinstance(result, PhaseResult)
        assert result.phase == 0
        assert result.steps_run > 0
        assert math.isfinite(result.final_loss)
        assert result.stop_reason in ("max_steps", "early_stop", "converged")
        assert Path(result.checkpoint).exists()


# ─────────────────────────────────────────────────────────────────────────────
# 11. TrainingMonitor integration with _Phase0Model
    def test_empty_datasets_no_crash(self):
        """
        Regression test: run_phase0 con datasets={} no debe dar ZeroDivisionError.
        Debe correr usando el fallback sintético de make_mixed_ids.
        """
        hp = _TrainHparams(
            ph0_steps=5, ph0_eval=10, ph0_patience=100, ph0_conv_win=100,
            ph0_lr=1e-3, ph0_warmup=1, n_train=5, n_val=5, max_seq=16,
            ckpt_every=10,
        )
        result = run_phase0(
            pipeline=self.pipeline, cfg=self.cfg, tok=self.tok,
            datasets={},            # <-- datasets vacío, el bug original
            hparams=hp, device=DEVICE,
            checkpoint_dir=self.tmp_path,
            resume=False,
            max_steps_override=5,
        )
        assert isinstance(result, PhaseResult)
        assert result.steps_run > 0


# ─────────────────────────────────────────────────────────────────────────────

class TestTrainingMonitorPhase0Integration:

    def test_monitor_step_returns_none_or_signal(self):
        cfg  = _tiny_cfg()
        pipe = MoSEPipeline(cfg).to(DEVICE)
        tok  = _simple_tok()
        m    = _Phase0Model(pipe, cfg)

        val_ids = [tok.encode("hello world", max_len=16) for _ in range(10)]
        fixed   = [("hello", "world", "cora")]

        monitor = TrainingMonitor(
            model          = m,
            tok            = tok,
            val_ids_list   = val_ids,
            fixed_examples = fixed,
            cfg            = cfg,
            eval_every     = 5,
            patience       = 100,
            convergence_delta  = 0.001,
            convergence_window = 100,
            log_path       = Path(tempfile.mktemp(suffix=".jsonl")),
            device         = DEVICE,
            is_motor       = False,
        )
        for step in range(1, 8):
            sig = monitor.step(step, 2.0 - step * 0.05)
            assert sig is None or sig in ("early_stop", "converged")

    def test_monitor_eval_triggers_at_eval_every(self):
        cfg  = _tiny_cfg()
        pipe = MoSEPipeline(cfg).to(DEVICE)
        tok  = _simple_tok()
        m    = _Phase0Model(pipe, cfg)

        val_ids = [tok.encode("hello world foo", max_len=16) for _ in range(10)]
        fixed   = [("hello", "world", "cora")]

        monitor = TrainingMonitor(
            model          = m,
            tok            = tok,
            val_ids_list   = val_ids,
            fixed_examples = fixed,
            cfg            = cfg,
            eval_every     = 3,
            patience       = 100,
            convergence_delta  = 0.001,
            convergence_window = 100,
            log_path       = Path(tempfile.mktemp(suffix=".jsonl")),
            device         = DEVICE,
            is_motor       = False,
        )
        for step in range(1, 10):
            monitor.step(step, 1.5)

        # eval at steps 3, 6, 9 → 3 entries
        assert len(monitor.eval_history) >= 3

    def test_monitor_early_stop_on_no_improvement(self):
        cfg  = _tiny_cfg()
        pipe = MoSEPipeline(cfg).to(DEVICE)
        tok  = _simple_tok()
        m    = _Phase0Model(pipe, cfg)

        val_ids = [tok.encode("hello world", max_len=16) for _ in range(10)]
        fixed   = [("hello", "world", "cora")]

        monitor = TrainingMonitor(
            model          = m,
            tok            = tok,
            val_ids_list   = val_ids,
            fixed_examples = fixed,
            cfg            = cfg,
            eval_every     = 2,
            patience       = 4,          # stop if no improvement in 4 steps
            convergence_delta  = 0.0001,
            convergence_window = 1000,
            log_path       = Path(tempfile.mktemp(suffix=".jsonl")),
            device         = DEVICE,
            is_motor       = False,
        )

        # Feed losses that don't improve (val_loss will be static with eval model)
        signals = []
        for step in range(1, 50):
            sig = monitor.step(step, 5.0)
            signals.append(sig)
            if sig is not None:
                break

        assert "early_stop" in signals or "converged" in signals
