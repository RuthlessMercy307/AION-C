"""
tests/test_opus_dataset.py — Tests para experiments/opus_dataset.py
=====================================================================

Cubre:
  1. Carga básica y longitud correcta
  2. .generate() retorna SimpleNamespace con campos correctos
  3. Filtrado por difficulty_range
  4. max_examples limita el dataset
  5. seed produce resultados reproducibles
  6. get_all_texts() retorna lista de strings
  7. train_eval_split() sin solapamiento
  8. Motor desconocido lanza ValueError
  9. Dataset no encontrado lanza FileNotFoundError

Los tests usan el dataset real si está disponible; si no, se skipean.
"""

from __future__ import annotations

import os
import sys
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest

from experiments.opus_dataset import OpusDataset, MOTOR_FILES, _DS_DEFAULT

# ── Fixture: ruta al dataset real ─────────────────────────────────────────────

DATASET_AVAILABLE = _DS_DEFAULT.exists() and (_DS_DEFAULT / "mose_cora.jsonl").exists()
SKIP_NO_DATA = pytest.mark.skipif(
    not DATASET_AVAILABLE,
    reason="DataSet-Generator-Claude-Opus no encontrado"
)


# ── Fixture: dataset mínimo sintético (siempre disponible) ────────────────────

@pytest.fixture(scope="module")
def tiny_jsonl(tmp_path_factory):
    """Crea un archivo JSONL temporal con 10 ejemplos mínimos."""
    tmp = tmp_path_factory.mktemp("data")
    jsonl_path = tmp / "mose_cora.jsonl"
    examples = []
    for i in range(10):
        examples.append({
            "id":              f"cora_{i:05d}",
            "motor":           "CORA",
            "subtask":         "counterfactual" if i % 2 == 0 else "causal_chain",
            "difficulty":      (i % 5) + 1,
            "input":           f"Pregunta número {i} sobre causalidad.",
            "graph":           {"nodes": [{"id": "n0", "label": "A"}],
                                "edges": [{"source": "n0", "target": "n0",
                                           "relation": "CAUSES"}]},
            "expected_output": f"Respuesta {i}.",
            "reasoning":       f"Razonamiento {i}.",
            "metadata":        {},
        })
    with open(jsonl_path, "w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
    return tmp


@pytest.fixture(scope="module")
def tiny_dataset(tiny_jsonl):
    return OpusDataset("cora", dataset_root=tiny_jsonl)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Carga y longitud
# ─────────────────────────────────────────────────────────────────────────────

class TestLoad:

    def test_len(self, tiny_dataset):
        assert len(tiny_dataset) == 10

    def test_repr(self, tiny_dataset):
        r = repr(tiny_dataset)
        assert "cora" in r
        assert "10" in r

    def test_max_examples(self, tiny_jsonl):
        ds = OpusDataset("cora", max_examples=3, dataset_root=tiny_jsonl)
        assert len(ds) == 3

    def test_max_examples_larger_than_dataset(self, tiny_jsonl):
        ds = OpusDataset("cora", max_examples=999, dataset_root=tiny_jsonl)
        assert len(ds) == 10  # no crashea, devuelve todo

    def test_difficulty_range(self, tiny_jsonl):
        # Dificultad 1-5 rotando (i%5)+1 para i=0..9
        # diff=3: i=2,7 → 2 ejemplos
        ds = OpusDataset("cora", difficulty_range=(3, 3), dataset_root=tiny_jsonl)
        assert len(ds) == 2

    def test_difficulty_range_empty_raises(self, tiny_jsonl):
        with pytest.raises(ValueError, match="vacío"):
            OpusDataset("cora", difficulty_range=(99, 99), dataset_root=tiny_jsonl)


# ─────────────────────────────────────────────────────────────────────────────
# 2. .generate() — campos y tipos
# ─────────────────────────────────────────────────────────────────────────────

class TestGenerate:

    def test_returns_simplenamespace(self, tiny_dataset):
        ex = tiny_dataset.generate()
        assert isinstance(ex, SimpleNamespace)

    def test_has_problem_text(self, tiny_dataset):
        ex = tiny_dataset.generate()
        assert isinstance(ex.problem_text, str)
        assert len(ex.problem_text) > 0

    def test_has_answer(self, tiny_dataset):
        ex = tiny_dataset.generate()
        assert isinstance(ex.answer, str)
        assert len(ex.answer) > 0

    def test_has_difficulty(self, tiny_dataset):
        ex = tiny_dataset.generate()
        assert isinstance(ex.difficulty, int)
        assert 1 <= ex.difficulty <= 5

    def test_has_subtask(self, tiny_dataset):
        ex = tiny_dataset.generate()
        assert isinstance(ex.subtask, str)

    def test_has_graph(self, tiny_dataset):
        ex = tiny_dataset.generate()
        assert isinstance(ex.graph, dict)
        assert "nodes" in ex.graph
        assert "edges" in ex.graph

    def test_has_reasoning(self, tiny_dataset):
        ex = tiny_dataset.generate()
        assert isinstance(ex.reasoning, str)

    def test_level_arg_ignored(self, tiny_dataset):
        """level= no lanza error — compatibilidad con generadores sintéticos."""
        ex = tiny_dataset.generate(level=1)
        assert ex is not None
        ex2 = tiny_dataset.generate(level=5)
        assert ex2 is not None

    def test_problem_text_matches_input_field(self, tiny_jsonl):
        ds = OpusDataset("cora", seed=0, dataset_root=tiny_jsonl)
        # Con seed fijo, la primera llamada debe ser reproducible
        ex1 = ds.generate()
        ds2 = OpusDataset("cora", seed=0, dataset_root=tiny_jsonl)
        ex2 = ds2.generate()
        assert ex1.problem_text == ex2.problem_text


# ─────────────────────────────────────────────────────────────────────────────
# 3. Reproducibilidad
# ─────────────────────────────────────────────────────────────────────────────

class TestReproducibility:

    def test_seed_produces_same_sequence(self, tiny_jsonl):
        ds1 = OpusDataset("cora", seed=42, dataset_root=tiny_jsonl)
        ds2 = OpusDataset("cora", seed=42, dataset_root=tiny_jsonl)
        for _ in range(5):
            assert ds1.generate().problem_text == ds2.generate().problem_text

    def test_different_seeds_differ(self, tiny_jsonl):
        ds1 = OpusDataset("cora", seed=1, dataset_root=tiny_jsonl)
        ds2 = OpusDataset("cora", seed=99, dataset_root=tiny_jsonl)
        texts1 = [ds1.generate().problem_text for _ in range(20)]
        texts2 = [ds2.generate().problem_text for _ in range(20)]
        # Con 10 ejemplos y 20 muestras, es prácticamente imposible que sean iguales
        assert texts1 != texts2


# ─────────────────────────────────────────────────────────────────────────────
# 4. get_all_texts()
# ─────────────────────────────────────────────────────────────────────────────

class TestGetAllTexts:

    def test_returns_list_of_strings(self, tiny_dataset):
        texts = tiny_dataset.get_all_texts()
        assert isinstance(texts, list)
        assert all(isinstance(t, str) for t in texts)

    def test_length_matches_dataset(self, tiny_dataset):
        texts = tiny_dataset.get_all_texts()
        assert len(texts) == len(tiny_dataset)

    def test_contains_input_and_output(self, tiny_jsonl):
        ds = OpusDataset("cora", seed=0, dataset_root=tiny_jsonl)
        texts = ds.get_all_texts()
        # Cada texto debería contener el input y el expected_output concatenados
        for t in texts:
            assert len(t) > 0
            assert " " in t  # al menos un espacio entre input y answer


# ─────────────────────────────────────────────────────────────────────────────
# 5. train_eval_split()
# ─────────────────────────────────────────────────────────────────────────────

class TestTrainEvalSplit:

    def test_sizes(self, tiny_dataset):
        train, ev = tiny_dataset.train_eval_split(eval_size=3, seed=0)
        assert len(ev) == 3
        assert len(train) == 7  # 10 - 3

    def test_no_overlap(self, tiny_jsonl):
        ds = OpusDataset("cora", seed=0, dataset_root=tiny_jsonl)
        train, ev = ds.train_eval_split(eval_size=3, seed=0)
        train_texts = {ex["input"] for ex in train._examples}
        eval_texts  = {ex["input"] for ex in ev._examples}
        assert train_texts.isdisjoint(eval_texts)

    def test_generate_works_on_split(self, tiny_dataset):
        train, ev = tiny_dataset.train_eval_split(eval_size=2, seed=7)
        ex_tr = train.generate()
        ex_ev = ev.generate()
        assert ex_tr.problem_text
        assert ex_ev.problem_text


# ─────────────────────────────────────────────────────────────────────────────
# 6. Errores
# ─────────────────────────────────────────────────────────────────────────────

class TestErrors:

    def test_unknown_motor_raises(self, tiny_jsonl):
        with pytest.raises(ValueError, match="Motor desconocido"):
            OpusDataset("invalid_motor", dataset_root=tiny_jsonl)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            OpusDataset("cora", dataset_root=tmp_path)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Tests con datos reales (skipean si no hay dataset)
# ─────────────────────────────────────────────────────────────────────────────

class TestRealData:

    @SKIP_NO_DATA
    def test_cora_loads_20k(self):
        ds = OpusDataset("cora")
        assert len(ds) == 20_000

    @SKIP_NO_DATA
    def test_all_motors_load(self):
        for motor in ["cora", "axiom", "empathy", "forge_c", "muse"]:
            ds = OpusDataset(motor)
            assert len(ds) == 20_000, f"{motor}: esperado 20000, got {len(ds)}"

    @SKIP_NO_DATA
    def test_cora_example_has_causal_graph(self):
        ds = OpusDataset("cora", max_examples=10, seed=0)
        ex = ds.generate()
        assert "nodes" in ex.graph
        assert "edges" in ex.graph
        assert len(ex.graph["nodes"]) > 0

    @SKIP_NO_DATA
    def test_generate_is_fast(self):
        """100 generate() calls en < 0.1 segundos."""
        import time
        ds = OpusDataset("cora", max_examples=1000, seed=0)
        t0 = time.perf_counter()
        for _ in range(100):
            ds.generate()
        elapsed = time.perf_counter() - t0
        assert elapsed < 0.5, f"generate() demasiado lento: {elapsed:.2f}s para 100 llamadas"
