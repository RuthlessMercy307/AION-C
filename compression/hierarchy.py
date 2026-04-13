"""
compression/hierarchy.py — Niveles EPISODIC/STABLE/NUCLEAR y clustering.

API principal:
    HierarchicalStore       — contenedor con items por nivel
    Clusterer               — agrupa items por similarity_fn
    HierarchicalCompressor  — promueve clusters a nivel superior
    sleep_compress_hook     — adaptador para Parte 23 (fase 4)

Similarity por defecto: Jaccard de tokens (determinista, sin deps). Se puede
reemplazar por una función basada en embeddings cuando estén disponibles.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple


# ════════════════════════════════════════════════════════════════════════════
# Modelo
# ════════════════════════════════════════════════════════════════════════════

class MemoryLevel(str, Enum):
    EPISODIC = "episodic"
    STABLE = "stable"
    NUCLEAR = "nuclear"


@dataclass
class StoredItem:
    """Un item en la memoria jerárquica.

    id:           identificador único dentro del store
    text:         texto representativo (summary para niveles superiores)
    level:        EPISODIC / STABLE / NUCLEAR
    usage_count:  cuántas veces fue accedido (señal para promotion)
    created_at:   unix ts
    parent_id:    id del cluster/item superior del que deriva, si aplica
    anchor_ids:   para niveles STABLE/NUCLEAR, ids de los episodios ancla
                  (evidencia cruda conservada)
    meta:         dict libre para extensiones
    """
    id: str
    text: str
    level: MemoryLevel
    usage_count: int = 0
    created_at: float = field(default_factory=time.time)
    parent_id: Optional[str] = None
    anchor_ids: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["level"] = self.level.value
        return d


@dataclass
class Cluster:
    """Grupo de items EPISODIC que forman un cluster temático."""
    cluster_id: str
    member_ids: List[str]
    anchor_ids: List[str]           # 1-2 representativos
    summary: str                    # resumen textual del cluster
    similarity: float               # similitud intra-cluster media

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ════════════════════════════════════════════════════════════════════════════
# Store
# ════════════════════════════════════════════════════════════════════════════

class HierarchicalStore:
    """Contenedor en memoria para items jerárquicos. NO persistente.

    Para persistencia, una capa superior envuelve esto y escribe a disco.
    """

    def __init__(self) -> None:
        self._items: Dict[str, StoredItem] = {}
        self._counter = 0

    def add(self, item: StoredItem) -> None:
        if item.id in self._items:
            raise KeyError(f"item id already exists: {item.id}")
        self._items[item.id] = item

    def new_id(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}_{self._counter}"

    def get(self, item_id: str) -> StoredItem:
        return self._items[item_id]

    def has(self, item_id: str) -> bool:
        return item_id in self._items

    def list_by_level(self, level: MemoryLevel) -> List[StoredItem]:
        return [it for it in self._items.values() if it.level == level]

    def all(self) -> List[StoredItem]:
        return list(self._items.values())

    def mark_used(self, item_id: str) -> None:
        it = self._items.get(item_id)
        if it is not None:
            it.usage_count += 1

    def demote_to_episodic(self, item_id: str) -> None:
        """Baja un item a EPISODIC — útil cuando un nuclear entra en duda."""
        it = self._items[item_id]
        it.level = MemoryLevel.EPISODIC

    def __len__(self) -> int:
        return len(self._items)

    # ── Persistencia JSONL ────────────────────────────────────────────────
    def save_jsonl(self, path) -> None:
        """Serializa todos los items (uno por línea) a un archivo JSONL.

        Se guarda también el `_counter` en una primera línea especial con
        clave "_meta" para que new_id() siga produciendo IDs únicos al reload.
        """
        import json
        from pathlib import Path as _Path
        p = _Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps({"_meta": {"counter": self._counter}})]
        for item in self._items.values():
            lines.append(json.dumps(item.to_dict()))
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def load_jsonl(self, path) -> None:
        """Reemplaza el estado actual con el contenido de un JSONL."""
        import json
        from pathlib import Path as _Path
        p = _Path(path)
        if not p.exists():
            return
        self._items.clear()
        self._counter = 0
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if "_meta" in d:
                self._counter = int(d["_meta"].get("counter", 0))
                continue
            # Reconstituir StoredItem — necesitamos parsear el enum level.
            d["level"] = MemoryLevel(d["level"])
            item = StoredItem(**d)
            self._items[item.id] = item


# ════════════════════════════════════════════════════════════════════════════
# Similarity
# ════════════════════════════════════════════════════════════════════════════

SimilarityFn = Callable[[str, str], float]


def _tokens(text: str) -> Set[str]:
    return {t for t in text.lower().split() if t}


def jaccard_similarity(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


# ════════════════════════════════════════════════════════════════════════════
# Clusterer
# ════════════════════════════════════════════════════════════════════════════

class Clusterer:
    """Agrupa items por similitud mediante greedy matching.

    Algoritmo:
        1. Para cada item sin asignar, abrir un cluster nuevo con él como seed.
        2. Mirar todos los otros items sin asignar — los que tengan
           sim(seed, other) >= threshold entran al cluster.
        3. Repetir hasta agotar items.
        4. Descartar clusters con < min_size miembros.
    """

    def __init__(
        self,
        similarity_fn: SimilarityFn = jaccard_similarity,
        threshold: float = 0.4,
        min_size: int = 2,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        if min_size < 2:
            raise ValueError("min_size must be >= 2 (cluster requires ≥2 items)")
        self.similarity_fn = similarity_fn
        self.threshold = threshold
        self.min_size = min_size

    def cluster(self, items: List[StoredItem]) -> List[Cluster]:
        if len(items) < self.min_size:
            return []
        unassigned: List[StoredItem] = list(items)
        out: List[Cluster] = []
        cid = 0
        while unassigned:
            seed = unassigned.pop(0)
            members = [seed]
            sims = [1.0]
            remaining: List[StoredItem] = []
            for other in unassigned:
                s = self.similarity_fn(seed.text, other.text)
                if s >= self.threshold:
                    members.append(other)
                    sims.append(s)
                else:
                    remaining.append(other)
            unassigned = remaining
            if len(members) >= self.min_size:
                cid += 1
                # anchor = seed + el miembro más similar (hasta 2)
                paired = sorted(
                    zip(members[1:], sims[1:]),
                    key=lambda x: x[1],
                    reverse=True,
                )
                anchors = [members[0].id]
                if paired:
                    anchors.append(paired[0][0].id)
                summary = _summarize_cluster([m.text for m in members])
                out.append(Cluster(
                    cluster_id=f"cluster_{cid}",
                    member_ids=[m.id for m in members],
                    anchor_ids=anchors,
                    summary=summary,
                    similarity=float(sum(sims) / len(sims)),
                ))
        return out


def _summarize_cluster(texts: List[str]) -> str:
    """Resumen determinista: las N palabras más frecuentes del cluster."""
    from collections import Counter
    counter: Counter = Counter()
    for t in texts:
        counter.update(_tokens(t))
    top = [w for w, _ in counter.most_common(5)]
    return " ".join(top) if top else "<empty>"


# ════════════════════════════════════════════════════════════════════════════
# HierarchicalCompressor — promoción entre niveles
# ════════════════════════════════════════════════════════════════════════════

class HierarchicalCompressor:
    """Orquesta clusterer + store para promover items entre niveles."""

    def __init__(
        self,
        store: HierarchicalStore,
        clusterer: Optional[Clusterer] = None,
        nuclear_usage_threshold: int = 5,
    ) -> None:
        self.store = store
        self.clusterer = clusterer or Clusterer()
        self.nuclear_usage_threshold = nuclear_usage_threshold

    # ── Episodic → Stable ────────────────────────────────────────────────
    def compress_episodic_to_stable(self) -> List[StoredItem]:
        """Forma clusters de nivel EPISODIC y crea items STABLE por cada uno.

        Devuelve los nuevos STABLE creados. Los episodios originales NO se
        borran — el pruning (Parte 24) se encarga después.
        """
        episodic = self.store.list_by_level(MemoryLevel.EPISODIC)
        clusters = self.clusterer.cluster(episodic)
        created: List[StoredItem] = []
        for c in clusters:
            stable_id = self.store.new_id("stable")
            item = StoredItem(
                id=stable_id,
                text=c.summary,
                level=MemoryLevel.STABLE,
                anchor_ids=list(c.anchor_ids),
                meta={
                    "member_ids": list(c.member_ids),
                    "cluster_id": c.cluster_id,
                    "similarity": c.similarity,
                },
            )
            self.store.add(item)
            created.append(item)
        return created

    # ── Stable → Nuclear ─────────────────────────────────────────────────
    def promote_stable_to_nuclear(self) -> List[StoredItem]:
        """Promueve STABLE a NUCLEAR si su usage_count supera el umbral."""
        stable = self.store.list_by_level(MemoryLevel.STABLE)
        promoted: List[StoredItem] = []
        for item in stable:
            if item.usage_count >= self.nuclear_usage_threshold:
                nuclear_id = self.store.new_id("nuclear")
                new_item = StoredItem(
                    id=nuclear_id,
                    text=f"concept: {item.text}",
                    level=MemoryLevel.NUCLEAR,
                    parent_id=item.id,
                    anchor_ids=list(item.anchor_ids),
                    usage_count=item.usage_count,
                    meta={"source_stable": item.id},
                )
                self.store.add(new_item)
                promoted.append(new_item)
        return promoted

    # ── Ingesta desde episodios del sleep buffer ──────────────────────────
    def ingest_episodes(self, episodes: Iterable[Any]) -> List[StoredItem]:
        """Convierte episodios del sleep buffer en StoredItems EPISODIC."""
        created: List[StoredItem] = []
        for ep in episodes:
            eid = self.store.new_id("ep")
            text = f"{ep.user_text} → {ep.aion_response}"
            item = StoredItem(
                id=eid,
                text=text,
                level=MemoryLevel.EPISODIC,
                meta={"motor_sequence": list(getattr(ep, "motor_sequence", []))},
            )
            self.store.add(item)
            created.append(item)
        return created


# ════════════════════════════════════════════════════════════════════════════
# Sleep cycle adaptador
# ════════════════════════════════════════════════════════════════════════════

def sleep_compress_hook(
    compressor: Optional[HierarchicalCompressor] = None,
):
    """Devuelve un compress_hook compatible con SleepCycle (Parte 23).

    Se usa:
        store = HierarchicalStore()
        compressor = HierarchicalCompressor(store)
        cycle = SleepCycle(buf, compress_hook=sleep_compress_hook(compressor))

    Efectos:
        - Ingesta los episodios del ciclo actual como EPISODIC items.
        - Forma clusters y los promueve a STABLE.
        - Devuelve stats para el phase log.
    """
    comp = compressor or HierarchicalCompressor(HierarchicalStore())

    def _hook(episodes, prev):
        ingested = comp.ingest_episodes(episodes)
        created = comp.compress_episodic_to_stable()
        promoted_nuclear = comp.promote_stable_to_nuclear()
        return {
            "clusters": len(created),
            "ingested": len(ingested),
            "stable_created": len(created),
            "nuclear_promoted": len(promoted_nuclear),
            "anchors": {c.id: c.anchor_ids for c in created},
            "source": "compressor",
        }

    return _hook
