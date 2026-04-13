"""
memory/semantic_store.py — SemanticStore: MEM externa para AION-C
================================================================

MEM basada en embeddings del encoder de AION-C.
Busqueda por cosine similarity, no necesita modelo externo.

Uso:
    mem = SemanticStore(encoder)
    mem.store("python_creator", "Python was created by Guido van Rossum", domain="forge_c")
    results = mem.search("Who created Python?", top_k=3)
    # → [("python_creator", "Python was created by...", 0.87)]
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class MemEntry:
    """One entry in semantic memory."""
    key: str
    value: str
    domain: str
    embedding: torch.Tensor  # [D]
    timestamp: float = field(default_factory=time.time)
    source: str = "manual"  # manual, learned, web
    access_count: int = 0


class SemanticStore:
    """
    Semantic memory store using encoder embeddings for similarity search.

    The encoder converts text → concept vectors → mean pool → embedding.
    Search uses cosine similarity between query embedding and stored embeddings.
    No external model needed — reuses the AION-C encoder.

    Args:
        encoder: the StreamEncoder from AION-C pipeline (or None for text-only mode)
        tokenizer: the tokenizer for encoding text
        similarity_threshold: minimum cosine similarity to return a result (default 0.5)
    """

    def __init__(
        self,
        encoder=None,
        tokenizer=None,
        similarity_threshold: float = 0.5,
    ):
        self._entries: Dict[str, MemEntry] = {}
        self._encoder = encoder
        self._tok = tokenizer
        self._threshold = similarity_threshold
        self._device = torch.device("cpu")

    def set_encoder(self, encoder, tokenizer, device=None):
        """Set or update the encoder (call after pipeline is built)."""
        self._encoder = encoder
        self._tok = tokenizer
        if device:
            self._device = device

    @torch.no_grad()
    def _embed(self, text: str) -> torch.Tensor:
        """Encode text to embedding vector [D] using the AION-C encoder."""
        if self._encoder is None or self._tok is None:
            # Fallback: bag-of-words hash embedding
            words = text.lower().split()
            D = 64
            emb = torch.zeros(D)
            for w in words:
                h = hash(w) % D
                emb[h] += 1.0
            return F.normalize(emb, dim=0)

        try:
            ids = self._tok.encode(text, 128)
        except TypeError:
            ids = self._tok.encode(text)[:128]

        ids_t = torch.tensor([ids], dtype=torch.long, device=self._device)
        self._encoder.eval()
        concepts = self._encoder(ids_t)  # [1, L, D]
        embedding = concepts.mean(dim=1).squeeze(0)  # [D]
        return F.normalize(embedding.float().cpu(), dim=0)

    # ── CRUD ──────────────────────────────────────────────────────────────────

    def store(self, key: str, value: str, domain: str = "general", source: str = "manual") -> None:
        """Store a fact in memory with its embedding."""
        embedding = self._embed(value)
        self._entries[key] = MemEntry(
            key=key, value=value, domain=domain,
            embedding=embedding, source=source,
        )

    def learn(self, key: str, value: str, domain: str = "general") -> None:
        """Auto-learn: store a newly discovered fact."""
        self.store(key, value, domain=domain, source="learned")

    def get(self, key: str) -> Optional[str]:
        """Get value by exact key."""
        entry = self._entries.get(key)
        if entry:
            entry.access_count += 1
            return entry.value
        return None

    def delete(self, key: str) -> bool:
        """Delete an entry."""
        if key in self._entries:
            del self._entries[key]
            return True
        return False

    # ── Search ────────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 5, domain: Optional[str] = None) -> List[Tuple[str, str, float]]:
        """
        Search memory by semantic similarity.

        Args:
            query: natural language query
            top_k: max results to return
            domain: filter by domain (None = all)

        Returns:
            List of (key, value, similarity_score) sorted by relevance
        """
        if not self._entries:
            return []

        q_emb = self._embed(query)

        results = []
        for key, entry in self._entries.items():
            if domain and entry.domain != domain:
                continue
            sim = F.cosine_similarity(q_emb.unsqueeze(0), entry.embedding.unsqueeze(0)).item()
            if sim >= self._threshold:
                results.append((key, entry.value, sim))
                entry.access_count += 1

        results.sort(key=lambda x: -x[2])
        return results[:top_k]

    def search_as_context(self, query: str, top_k: int = 3, domain: Optional[str] = None) -> str:
        """Search and format results as context string for the motor."""
        results = self.search(query, top_k, domain)
        if not results:
            return ""
        parts = []
        for key, value, sim in results:
            parts.append(f"[MEM {sim:.2f}] {value}")
        return "\n".join(parts)

    # ── Stats ─────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._entries)

    def stats(self) -> Dict[str, Any]:
        """Return memory statistics."""
        domains = {}
        for e in self._entries.values():
            domains[e.domain] = domains.get(e.domain, 0) + 1
        return {
            "total_entries": len(self._entries),
            "domains": domains,
            "total_accesses": sum(e.access_count for e in self._entries.values()),
        }

    def list_entries(self, domain: Optional[str] = None) -> List[Dict]:
        """List all entries, optionally filtered by domain."""
        results = []
        for key, entry in self._entries.items():
            if domain and entry.domain != domain:
                continue
            results.append({
                "key": key, "value": entry.value[:80],
                "domain": entry.domain, "source": entry.source,
                "accesses": entry.access_count,
            })
        return results

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save memory to JSON (without embeddings — they get recomputed on load)."""
        data = []
        for key, entry in self._entries.items():
            data.append({
                "key": entry.key, "value": entry.value,
                "domain": entry.domain, "source": entry.source,
                "timestamp": entry.timestamp,
            })
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load(self, path: str) -> int:
        """Load memory from JSON, recomputing embeddings. Returns count loaded."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        count = 0
        for entry in data:
            self.store(
                key=entry["key"], value=entry["value"],
                domain=entry.get("domain", "general"),
                source=entry.get("source", "loaded"),
            )
            count += 1
        return count
