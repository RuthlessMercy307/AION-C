"""
tests/test_brain_version.py — Tests para Parte 9.5 del MEGA-PROMPT
====================================================================

Cubre:
  BrainVersion: serialization roundtrip
  BrainVersionManager: save_version, load_version, list_versions, latest,
                       compare, rollback, delete_version, parent linking
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import torch
import torch.nn as nn

from brain.version_manager import (
    BrainVersion, BrainVersionManager,
    WEIGHTS_FILENAME, METADATA_FILENAME,
)


# ─────────────────────────────────────────────────────────────────────────────
# BrainVersion
# ─────────────────────────────────────────────────────────────────────────────


class TestBrainVersion:
    def test_roundtrip(self):
        v = BrainVersion(
            id="v3",
            parent_id="v2",
            timestamp=12345.6,
            notes="learned javascript",
            metrics={"score": 0.92, "exam": 0.88},
            metadata={"trigger": "auto-learn"},
        )
        d = v.to_dict()
        v2 = BrainVersion.from_dict(d)
        assert v2.id == "v3"
        assert v2.parent_id == "v2"
        assert v2.notes == "learned javascript"
        assert v2.metrics == {"score": 0.92, "exam": 0.88}
        assert v2.metadata["trigger"] == "auto-learn"


# ─────────────────────────────────────────────────────────────────────────────
# BrainVersionManager
# ─────────────────────────────────────────────────────────────────────────────


def fake_state():
    return {"weight": torch.tensor([1.0, 2.0, 3.0]), "bias": torch.tensor([0.5])}


class TestBrainVersionManager:
    def test_save_creates_v1(self, tmp_path):
        bvm = BrainVersionManager(tmp_path)
        v = bvm.save_version(fake_state(), notes="initial")
        assert v.id == "v1"
        assert (tmp_path / "v1" / WEIGHTS_FILENAME).exists()
        assert (tmp_path / "v1" / METADATA_FILENAME).exists()

    def test_subsequent_versions_increment(self, tmp_path):
        bvm = BrainVersionManager(tmp_path)
        v1 = bvm.save_version(fake_state())
        v2 = bvm.save_version(fake_state())
        v3 = bvm.save_version(fake_state())
        assert [v.id for v in (v1, v2, v3)] == ["v1", "v2", "v3"]

    def test_parent_link_auto(self, tmp_path):
        bvm = BrainVersionManager(tmp_path)
        v1 = bvm.save_version(fake_state())
        v2 = bvm.save_version(fake_state())
        assert v1.parent_id is None
        assert v2.parent_id == "v1"

    def test_parent_link_manual(self, tmp_path):
        bvm = BrainVersionManager(tmp_path)
        bvm.save_version(fake_state())
        bvm.save_version(fake_state())
        v3 = bvm.save_version(fake_state(), parent_id="v1")
        assert v3.parent_id == "v1"

    def test_load_version_returns_state(self, tmp_path):
        bvm = BrainVersionManager(tmp_path)
        original = fake_state()
        v = bvm.save_version(original)
        loaded = bvm.load_version(v.id)
        assert torch.equal(loaded["weight"], original["weight"])
        assert torch.equal(loaded["bias"], original["bias"])

    def test_list_versions_ordered(self, tmp_path):
        bvm = BrainVersionManager(tmp_path)
        for _ in range(5):
            bvm.save_version(fake_state())
        ids = [v.id for v in bvm.list_versions()]
        assert ids == ["v1", "v2", "v3", "v4", "v5"]

    def test_latest_returns_last(self, tmp_path):
        bvm = BrainVersionManager(tmp_path)
        bvm.save_version(fake_state())
        bvm.save_version(fake_state(), notes="last one")
        v = bvm.latest()
        assert v.id == "v2"
        assert v.notes == "last one"

    def test_latest_empty(self, tmp_path):
        bvm = BrainVersionManager(tmp_path)
        assert bvm.latest() is None

    def test_metrics_and_notes_persist(self, tmp_path):
        bvm = BrainVersionManager(tmp_path)
        v = bvm.save_version(
            fake_state(),
            notes="learned X",
            metrics={"score": 0.95, "exam": 0.91},
            metadata={"learned_topic": "javascript"},
        )
        loaded_meta = bvm.get_metadata(v.id)
        assert loaded_meta.notes == "learned X"
        assert loaded_meta.metrics == {"score": 0.95, "exam": 0.91}
        assert loaded_meta.metadata["learned_topic"] == "javascript"

    def test_compare_diff_metrics(self, tmp_path):
        bvm = BrainVersionManager(tmp_path)
        bvm.save_version(fake_state(), metrics={"score": 0.90})
        bvm.save_version(fake_state(), metrics={"score": 0.95})
        diff = bvm.compare("v1", "v2")
        assert diff["metric_diff"]["score"]["a"] == 0.90
        assert diff["metric_diff"]["score"]["b"] == 0.95
        assert diff["metric_diff"]["score"]["delta"] == pytest.approx(0.05)

    def test_compare_handles_missing_metric(self, tmp_path):
        bvm = BrainVersionManager(tmp_path)
        bvm.save_version(fake_state(), metrics={"a": 0.5})
        bvm.save_version(fake_state(), metrics={"b": 0.7})
        diff = bvm.compare("v1", "v2")
        assert diff["metric_diff"]["a"]["b"] is None
        assert diff["metric_diff"]["a"]["delta"] is None

    def test_rollback_returns_old_state(self, tmp_path):
        bvm = BrainVersionManager(tmp_path)
        s1 = {"w": torch.tensor([1.0])}
        s2 = {"w": torch.tensor([2.0])}
        bvm.save_version(s1)
        bvm.save_version(s2)
        rolled = bvm.rollback("v1")
        assert torch.equal(rolled["w"], torch.tensor([1.0]))

    def test_rollback_missing_version_raises(self, tmp_path):
        bvm = BrainVersionManager(tmp_path)
        with pytest.raises(FileNotFoundError):
            bvm.rollback("v99")

    def test_exists(self, tmp_path):
        bvm = BrainVersionManager(tmp_path)
        bvm.save_version(fake_state())
        assert bvm.exists("v1")
        assert not bvm.exists("v99")

    def test_delete_version(self, tmp_path):
        bvm = BrainVersionManager(tmp_path)
        bvm.save_version(fake_state())
        bvm.save_version(fake_state())
        assert bvm.delete_version("v1")
        assert not bvm.exists("v1")
        assert bvm.exists("v2")

    def test_delete_nonexistent_returns_false(self, tmp_path):
        bvm = BrainVersionManager(tmp_path)
        assert not bvm.delete_version("v999")

    def test_explicit_version_id(self, tmp_path):
        bvm = BrainVersionManager(tmp_path)
        v = bvm.save_version(fake_state(), version_id="experiment_42")
        assert v.id == "experiment_42"
        assert (tmp_path / "experiment_42").exists()

    def test_works_with_real_module_state(self, tmp_path):
        bvm = BrainVersionManager(tmp_path)
        m = nn.Linear(4, 4)
        v = bvm.save_version(m.state_dict(), notes="trained")
        loaded = bvm.load_version(v.id)
        m2 = nn.Linear(4, 4)
        m2.load_state_dict(loaded)
        assert torch.allclose(m.weight, m2.weight)
