"""Tests for memory/semantic_store.py"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
from memory.semantic_store import SemanticStore


class TestSemanticStoreNoEncoder:
    """Test with fallback bag-of-words embeddings (no encoder)."""

    def test_store_and_get(self):
        mem = SemanticStore()
        mem.store("fact1", "Python was created by Guido van Rossum", "forge_c")
        assert mem.get("fact1") == "Python was created by Guido van Rossum"

    def test_get_missing(self):
        mem = SemanticStore()
        assert mem.get("nonexistent") is None

    def test_search_finds_relevant(self):
        mem = SemanticStore(similarity_threshold=0.1)
        mem.store("python_creator", "Python was created by Guido van Rossum", "forge_c")
        mem.store("java_creator", "Java was created by James Gosling", "forge_c")
        mem.store("weather", "It is sunny today", "general")

        results = mem.search("Who created Python?")
        assert len(results) > 0
        # Python-related should rank higher than weather
        keys = [r[0] for r in results]
        assert "python_creator" in keys

    def test_search_empty_store(self):
        mem = SemanticStore()
        results = mem.search("anything")
        assert results == []

    def test_search_by_domain(self):
        mem = SemanticStore(similarity_threshold=0.1)
        mem.store("fact1", "Python is great", "forge_c")
        mem.store("fact2", "Python is a snake", "general")

        results = mem.search("Python", domain="forge_c")
        assert all(mem._entries[r[0]].domain == "forge_c" for r in results)

    def test_learn(self):
        mem = SemanticStore()
        mem.learn("new_fact", "The sky is blue", "cora")
        entry = mem._entries["new_fact"]
        assert entry.source == "learned"
        assert entry.domain == "cora"

    def test_delete(self):
        mem = SemanticStore()
        mem.store("key", "value")
        assert mem.delete("key") is True
        assert mem.get("key") is None
        assert mem.delete("key") is False

    def test_len(self):
        mem = SemanticStore()
        assert len(mem) == 0
        mem.store("a", "value a")
        mem.store("b", "value b")
        assert len(mem) == 2

    def test_stats(self):
        mem = SemanticStore()
        mem.store("a", "val", "cora")
        mem.store("b", "val", "axiom")
        mem.store("c", "val", "cora")
        stats = mem.stats()
        assert stats["total_entries"] == 3
        assert stats["domains"]["cora"] == 2
        assert stats["domains"]["axiom"] == 1

    def test_search_as_context(self):
        mem = SemanticStore(similarity_threshold=0.1)
        mem.store("fact", "Python was made by Guido", "forge_c")
        ctx = mem.search_as_context("Python creator")
        assert "MEM" in ctx
        assert "Guido" in ctx

    def test_save_and_load(self, tmp_path):
        mem = SemanticStore()
        mem.store("k1", "value one", "cora")
        mem.store("k2", "value two", "axiom")
        path = str(tmp_path / "mem.json")
        mem.save(path)

        mem2 = SemanticStore()
        count = mem2.load(path)
        assert count == 2
        assert mem2.get("k1") == "value one"
        assert mem2.get("k2") == "value two"

    def test_access_count(self):
        mem = SemanticStore(similarity_threshold=0.1)
        mem.store("fact", "test value")
        mem.get("fact")
        mem.get("fact")
        mem.search("test")
        assert mem._entries["fact"].access_count >= 2


class TestSemanticStoreWithEncoder:
    """Test with real AION-C encoder for semantic embeddings."""

    @pytest.fixture
    def mem_with_encoder(self):
        from router.pipeline import MoSEConfig, MoSEPipeline
        from experiments.train_production import build_tokenizer

        tok = build_tokenizer(32_000)
        cfg = MoSEConfig(
            hidden_dim=64, vocab_size=tok.vocab_size,
            enc_n_layers=2, enc_state_dim=4, enc_expand=2,
            enc_d_conv=4, enc_ffn_mult=2,
            orch_mlp_hidden=32, orch_max_motors=3, orch_min_confidence=0.3,
            motor_max_nodes=8, motor_n_heads=4, motor_threshold=0.01,
            unif_n_heads=4,
            dec_n_layers=2, dec_n_heads=4, dec_max_seq_len=128,
            dec_state_dim=4, dec_expand=2, dec_d_conv=4, dec_ffn_mult=2,
        )
        pipeline = MoSEPipeline(cfg)
        mem = SemanticStore(
            encoder=pipeline.encoder,
            tokenizer=tok,
            similarity_threshold=0.0,  # low threshold for untrained encoder
        )
        return mem

    def test_store_with_encoder(self, mem_with_encoder):
        mem = mem_with_encoder
        mem.store("fact", "Python was created by Guido van Rossum", "forge_c")
        assert len(mem) == 1
        entry = mem._entries["fact"]
        assert entry.embedding.shape == (64,)  # hidden_dim
        assert torch.isfinite(entry.embedding).all()

    def test_search_semantic(self, mem_with_encoder):
        mem = mem_with_encoder
        mem.store("python", "Python was created by Guido van Rossum", "forge_c")
        mem.store("rain", "Rain causes wet soil and flooding", "cora")
        mem.store("sad", "My friend is sad because of a loss", "empathy")

        results = mem.search("Who made Python?", top_k=3)
        assert len(results) > 0
        # With real encoder, semantic similarity should work
        print(f"Search 'Who made Python?': {[(k, s) for k, _, s in results]}")

    def test_mem_integration_scenario(self, mem_with_encoder):
        """The full scenario from the plan: store a fact, then find it."""
        mem = mem_with_encoder

        # Store
        mem.store("python_creator", "Python was created by Guido van Rossum in 1991", "forge_c")

        # Search
        results = mem.search("Who created Python?")
        print(f"Results: {results}")

        # Should find it
        assert len(results) > 0
        assert "Guido" in results[0][1]
        print(f"  PASS: MEM found 'Guido' for 'Who created Python?'")

        # Context injection
        ctx = mem.search_as_context("Who created Python?")
        assert "Guido" in ctx
        assert "MEM" in ctx
        print(f"  PASS: Context injection works: {ctx[:60]}")
