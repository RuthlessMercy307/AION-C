"""
tests/test_compression.py — Parte 26 (Compresión jerárquica 3 niveles).

Cubre:
    - MemoryLevel enum
    - StoredItem / HierarchicalStore: add, dedupe, list_by_level, usage
    - Clusterer: greedy clustering con Jaccard, threshold, min_size
    - HierarchicalCompressor: episodic→stable, anchors preservados,
      stable→nuclear por usage threshold, ingesta desde episodios
    - sleep_compress_hook: integración con SleepCycle
"""

from __future__ import annotations

import pytest

from compression import (
    MemoryLevel,
    StoredItem,
    Cluster,
    HierarchicalStore,
    Clusterer,
    HierarchicalCompressor,
    jaccard_similarity,
    sleep_compress_hook,
)
from sleep import Episode, EpisodicBuffer, SleepCycle


# ════════════════════════════════════════════════════════════════════════════
# jaccard_similarity
# ════════════════════════════════════════════════════════════════════════════

class TestJaccard:
    def test_identical(self):
        assert jaccard_similarity("a b c", "a b c") == 1.0

    def test_disjoint(self):
        assert jaccard_similarity("a b", "c d") == 0.0

    def test_partial(self):
        # {a,b,c} vs {a,b,d} → 2/4 = 0.5
        assert jaccard_similarity("a b c", "a b d") == 0.5

    def test_empty(self):
        assert jaccard_similarity("", "a") == 0.0
        assert jaccard_similarity("a", "") == 0.0


# ════════════════════════════════════════════════════════════════════════════
# HierarchicalStore
# ════════════════════════════════════════════════════════════════════════════

class TestHierarchicalStore:
    def test_add_and_get(self):
        s = HierarchicalStore()
        s.add(StoredItem(id="a", text="hello", level=MemoryLevel.EPISODIC))
        assert s.get("a").text == "hello"

    def test_duplicate_id_rejected(self):
        s = HierarchicalStore()
        s.add(StoredItem(id="a", text="x", level=MemoryLevel.EPISODIC))
        with pytest.raises(KeyError):
            s.add(StoredItem(id="a", text="y", level=MemoryLevel.EPISODIC))

    def test_list_by_level(self):
        s = HierarchicalStore()
        s.add(StoredItem("a", "1", MemoryLevel.EPISODIC))
        s.add(StoredItem("b", "2", MemoryLevel.STABLE))
        s.add(StoredItem("c", "3", MemoryLevel.EPISODIC))
        assert len(s.list_by_level(MemoryLevel.EPISODIC)) == 2
        assert len(s.list_by_level(MemoryLevel.STABLE)) == 1
        assert len(s.list_by_level(MemoryLevel.NUCLEAR)) == 0

    def test_mark_used_increments(self):
        s = HierarchicalStore()
        s.add(StoredItem("a", "x", MemoryLevel.STABLE))
        s.mark_used("a")
        s.mark_used("a")
        assert s.get("a").usage_count == 2

    def test_new_id_unique(self):
        s = HierarchicalStore()
        ids = {s.new_id("ep") for _ in range(10)}
        assert len(ids) == 10

    def test_demote_to_episodic(self):
        s = HierarchicalStore()
        s.add(StoredItem("a", "x", MemoryLevel.NUCLEAR))
        s.demote_to_episodic("a")
        assert s.get("a").level == MemoryLevel.EPISODIC


# ════════════════════════════════════════════════════════════════════════════
# Clusterer
# ════════════════════════════════════════════════════════════════════════════

class TestClusterer:
    def test_clusters_similar_items(self):
        items = [
            StoredItem("1", "python ownership python types", MemoryLevel.EPISODIC),
            StoredItem("2", "python types hints mypy", MemoryLevel.EPISODIC),
            StoredItem("3", "completely unrelated topic here", MemoryLevel.EPISODIC),
            StoredItem("4", "rust lifetimes borrow rust", MemoryLevel.EPISODIC),
            StoredItem("5", "rust borrow ownership lifetimes", MemoryLevel.EPISODIC),
        ]
        clusters = Clusterer(threshold=0.3, min_size=2).cluster(items)
        # Debería formar python_cluster y rust_cluster; el item 3 queda solo.
        assert len(clusters) >= 1
        ids_in_clusters = set()
        for c in clusters:
            ids_in_clusters.update(c.member_ids)
        assert "1" in ids_in_clusters
        assert "2" in ids_in_clusters
        # Ningún cluster contiene el item 3 aislado
        for c in clusters:
            assert "3" not in c.member_ids

    def test_empty_input(self):
        assert Clusterer().cluster([]) == []

    def test_below_min_size_empty(self):
        items = [StoredItem("1", "x", MemoryLevel.EPISODIC)]
        assert Clusterer(min_size=2).cluster(items) == []

    def test_anchors_in_members(self):
        items = [
            StoredItem("1", "python python types", MemoryLevel.EPISODIC),
            StoredItem("2", "python types strict", MemoryLevel.EPISODIC),
            StoredItem("3", "python strict types", MemoryLevel.EPISODIC),
        ]
        clusters = Clusterer(threshold=0.3).cluster(items)
        assert len(clusters) == 1
        c = clusters[0]
        assert set(c.anchor_ids).issubset(set(c.member_ids))
        assert 1 <= len(c.anchor_ids) <= 2

    def test_invalid_threshold_rejected(self):
        with pytest.raises(ValueError):
            Clusterer(threshold=1.5)
        with pytest.raises(ValueError):
            Clusterer(threshold=-0.1)

    def test_invalid_min_size_rejected(self):
        with pytest.raises(ValueError):
            Clusterer(min_size=1)

    def test_summary_is_top_tokens(self):
        items = [
            StoredItem("1", "python python fast", MemoryLevel.EPISODIC),
            StoredItem("2", "python fast python", MemoryLevel.EPISODIC),
        ]
        c = Clusterer(threshold=0.3).cluster(items)[0]
        assert "python" in c.summary


# ════════════════════════════════════════════════════════════════════════════
# HierarchicalCompressor
# ════════════════════════════════════════════════════════════════════════════

class TestHierarchicalCompressor:
    def test_compress_episodic_to_stable(self):
        store = HierarchicalStore()
        store.add(StoredItem("1", "python types hints", MemoryLevel.EPISODIC))
        store.add(StoredItem("2", "python types strict", MemoryLevel.EPISODIC))
        store.add(StoredItem("3", "python strict types", MemoryLevel.EPISODIC))
        comp = HierarchicalCompressor(store, Clusterer(threshold=0.3))
        created = comp.compress_episodic_to_stable()
        assert len(created) == 1
        assert created[0].level == MemoryLevel.STABLE
        # Los episodios originales siguen ahí
        assert len(store.list_by_level(MemoryLevel.EPISODIC)) == 3
        assert len(store.list_by_level(MemoryLevel.STABLE)) == 1

    def test_stable_preserves_anchors(self):
        store = HierarchicalStore()
        store.add(StoredItem("1", "rust borrow", MemoryLevel.EPISODIC))
        store.add(StoredItem("2", "rust borrow lifetimes", MemoryLevel.EPISODIC))
        comp = HierarchicalCompressor(store, Clusterer(threshold=0.3))
        stable = comp.compress_episodic_to_stable()[0]
        # Los anchors deben apuntar a ids reales del nivel EPISODIC
        for anchor in stable.anchor_ids:
            assert store.has(anchor)
            assert store.get(anchor).level == MemoryLevel.EPISODIC

    def test_promote_stable_to_nuclear_by_usage(self):
        store = HierarchicalStore()
        store.add(StoredItem("s1", "concept A", MemoryLevel.STABLE, usage_count=10))
        store.add(StoredItem("s2", "concept B", MemoryLevel.STABLE, usage_count=1))
        comp = HierarchicalCompressor(store, nuclear_usage_threshold=5)
        promoted = comp.promote_stable_to_nuclear()
        assert len(promoted) == 1
        assert promoted[0].parent_id == "s1"
        assert promoted[0].level == MemoryLevel.NUCLEAR

    def test_ingest_episodes_creates_episodic(self):
        store = HierarchicalStore()
        comp = HierarchicalCompressor(store)
        eps = [
            Episode("pregunta 1", "respuesta 1"),
            Episode("pregunta 2", "respuesta 2"),
        ]
        created = comp.ingest_episodes(eps)
        assert len(created) == 2
        assert all(c.level == MemoryLevel.EPISODIC for c in created)
        assert len(store.list_by_level(MemoryLevel.EPISODIC)) == 2

    def test_end_to_end_pipeline(self):
        """Ingesta → cluster → stable → nuclear."""
        store = HierarchicalStore()
        comp = HierarchicalCompressor(
            store,
            Clusterer(threshold=0.3),
            nuclear_usage_threshold=2,
        )
        eps = [
            Episode("python fast types", "ok"),
            Episode("python types strict", "ok"),
            Episode("python strict types", "ok"),
        ]
        comp.ingest_episodes(eps)
        stable = comp.compress_episodic_to_stable()
        assert len(stable) == 1
        # Simular uso y promover
        store.mark_used(stable[0].id)
        store.mark_used(stable[0].id)
        nuclear = comp.promote_stable_to_nuclear()
        assert len(nuclear) == 1
        assert nuclear[0].level == MemoryLevel.NUCLEAR


# ════════════════════════════════════════════════════════════════════════════
# SleepCycle integration
# ════════════════════════════════════════════════════════════════════════════

class TestHierarchicalStorePersistence:
    def test_save_load_roundtrip(self, tmp_path):
        s1 = HierarchicalStore()
        s1.add(StoredItem("ep_1", "hola mundo", MemoryLevel.EPISODIC, usage_count=3))
        s1.add(StoredItem(
            "stable_1", "mundo ancla", MemoryLevel.STABLE,
            anchor_ids=["ep_1"], parent_id=None, usage_count=5,
        ))
        # Forzar counter a un valor no trivial para validar su round-trip.
        _ = s1.new_id("ep")
        _ = s1.new_id("stable")
        path = tmp_path / "hierarchy.jsonl"
        s1.save_jsonl(path)

        s2 = HierarchicalStore()
        s2.load_jsonl(path)
        assert len(s2) == 2
        ep = s2.get("ep_1")
        assert ep.level == MemoryLevel.EPISODIC
        assert ep.usage_count == 3
        st = s2.get("stable_1")
        assert st.anchor_ids == ["ep_1"]
        # El counter debe continuar donde quedó
        next_id = s2.new_id("ep")
        assert next_id.startswith("ep_") and int(next_id.split("_")[1]) > 2

    def test_load_missing_file_noop(self, tmp_path):
        s = HierarchicalStore()
        s.load_jsonl(tmp_path / "none.jsonl")
        assert len(s) == 0


class TestSleepIntegration:
    def test_compress_hook_in_cycle(self):
        buf = EpisodicBuffer()
        buf.add(Episode("python types", "ok"))
        buf.add(Episode("python types strict", "ok"))
        buf.add(Episode("rust borrow", "ok"))

        store = HierarchicalStore()
        compressor = HierarchicalCompressor(store, Clusterer(threshold=0.3))
        cycle = SleepCycle(buf, compress_hook=sleep_compress_hook(compressor))
        log = cycle.run()

        cp = log.phase("compress")
        assert cp.data["source"] == "compressor"
        assert cp.data["ingested"] == 3
        # Los 2 de python deberían formar un cluster
        assert cp.data["stable_created"] >= 1
