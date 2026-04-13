"""
agent/memory_bridge.py — MemoryBridge: interfaz entre AgentLoop y memoria
==========================================================================

El MemoryBridge conecta el AgentLoop con una fuente de memoria externa.

Funciones principales:
  - load(key)         : carga un fragmento de memoria por clave
  - store(key, value) : guarda un valor en memoria
  - search(query)     : busca fragmentos relevantes (fuzzy)
  - as_context()      : devuelve el contexto de memoria para el motor

El motor de memoria real puede ser un archivo de texto, una base de datos
vectorial, o el sistema de memoria de AION-C. Por defecto se usa memoria
en RAM (dict) para facilitar los tests.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# MEMORY BRIDGE
# ─────────────────────────────────────────────────────────────────────────────

class MemoryBridge:
    """
    Interfaz de memoria para el AgentLoop.

    Por defecto usa un dict en RAM. Sustituible por cualquier backend
    (archivo JSONL, FAISS, base de datos) cambiando el backend opcional.

    Uso:
        mem = MemoryBridge()
        mem.store("task_context", "Working on AION-C training loop")
        mem.load("task_context")  # → "Working on AION-C training loop"
        mem.search("training")    # → [("task_context", "Working...")]
    """

    def __init__(self, backend: Optional[Dict[str, Any]] = None) -> None:
        self._store: Dict[str, Any] = backend if backend is not None else {}

    # ── CRUD ─────────────────────────────────────────────────────────────────

    def store(self, key: str, value: Any) -> None:
        """Guarda un valor bajo una clave."""
        self._store[key] = value

    def load(self, key: str, default: Any = None) -> Any:
        """Carga un valor por clave. Retorna default si no existe."""
        return self._store.get(key, default)

    def delete(self, key: str) -> bool:
        """Elimina una clave. Retorna True si existía."""
        existed = key in self._store
        self._store.pop(key, None)
        return existed

    def keys(self) -> List[str]:
        """Lista de claves disponibles."""
        return list(self._store.keys())

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, key: str) -> bool:
        return key in self._store

    # ── Learn (auto-save de conocimiento) ───────────────────────────────────

    def learn(self, key: str, content: Any, source: str = "unknown") -> None:
        """
        El modelo llama a learn() cuando aprende algo nuevo.

        Guarda contenido con metadata de fuente y timestamp.
        Usado por instruction tuning con [GUARDAR_MEM: ...].

        Args:
            key:     Clave descriptiva del conocimiento.
            content: Contenido a guardar.
            source:  Fuente del conocimiento (web, user, inference, etc.).
        """
        import time
        entry = {
            "content": content,
            "source":  source,
            "learned_at": time.time(),
        }
        self._store[key] = entry

    def get_learned(self, key: str) -> Optional[Any]:
        """Recupera contenido aprendido, extrayendo solo el content si es dict con metadata."""
        val = self._store.get(key)
        if isinstance(val, dict) and "content" in val:
            return val["content"]
        return val

    def search_learned(self, query: str, max_results: int = 5) -> List[tuple]:
        """Busca en conocimiento aprendido. Retorna (key, content, source)."""
        q    = query.lower()
        hits: List[tuple] = []
        for k, v in self._store.items():
            searchable = k.lower()
            if isinstance(v, dict) and "content" in v:
                searchable += " " + str(v["content"]).lower()
            else:
                searchable += " " + str(v).lower()
            if q in searchable:
                content = v["content"] if isinstance(v, dict) and "content" in v else v
                source  = v.get("source", "direct") if isinstance(v, dict) else "direct"
                hits.append((k, content, source))
                if len(hits) >= max_results:
                    break
        return hits

    # ── Búsqueda ──────────────────────────────────────────────────────────────

    def search(self, query: str, max_results: int = 5) -> List[tuple]:
        """
        Búsqueda simple por substring (case-insensitive).

        Args:
            query:       Término de búsqueda.
            max_results: Máximo de resultados a retornar.

        Returns:
            Lista de (key, value) donde la clave o el valor contienen el query.
        """
        q    = query.lower()
        hits: List[tuple] = []
        for k, v in self._store.items():
            if q in k.lower() or q in str(v).lower():
                hits.append((k, v))
                if len(hits) >= max_results:
                    break
        return hits

    # ── Contexto para el motor ────────────────────────────────────────────────

    def as_context(self, max_chars: int = 2000) -> str:
        """
        Devuelve el contenido de la memoria como string de contexto.

        Útil para incluir en el prompt del motor.

        Args:
            max_chars: límite de caracteres (trunca si es necesario).

        Returns:
            String con pares clave: valor, separados por newlines.
        """
        if not self._store:
            return "(no memory)"

        lines: List[str] = []
        for k, v in self._store.items():
            lines.append(f"{k}: {v}")

        context = "\n".join(lines)
        if len(context) > max_chars:
            context = context[:max_chars] + "...[truncated]"
        return context

    # ── Serialización ─────────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Exporta la memoria como dict."""
        return dict(self._store)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryBridge":
        """Crea un MemoryBridge desde un dict existente."""
        return cls(backend=dict(data))

    def __repr__(self) -> str:
        return f"MemoryBridge(keys={self.keys()!r})"
