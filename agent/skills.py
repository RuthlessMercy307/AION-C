"""
agent/skills.py — Skills system (Parte 3 del MEGA-PROMPT)
==========================================================

3.1 IMPLEMENTACIÓN
  - Carpeta: skills/
  - Cada .md es un "manual" de cómo hacer algo (Python best practices,
    debugging, escritura empática, etc.)
  - Al iniciar, los .md se cargan en MEM con domain="skill"
  - Antes de responder, se buscan skills relevantes (cosine > threshold)
  - El contenido encontrado se inyecta como `[SKILL: ...]` en el contexto
    siguiendo el formato canónico de la Parte 15.

Diseño:
  - SkillsLoader es agnóstico al backend de MEM: solo necesita un objeto
    con `.store(key, value, domain)` y `.search(query, top_k, domain)`.
  - Frontmatter YAML simple opcional al inicio del archivo (---/---) para
    permitir metadata como `domain`, `priority`, `tags`. Si no está, se usa
    el nombre del archivo como key.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


SKILL_DOMAIN = "skill"
DEFAULT_THRESHOLD = 0.5
DEFAULT_TOP_K = 3


@dataclass
class Skill:
    """Una skill cargada desde disco."""
    key:      str            # nombre canónico (sin .md)
    content:  str            # cuerpo del .md (sin frontmatter)
    metadata: Dict[str, Any] = field(default_factory=dict)
    path:     Optional[Path] = None

    @property
    def title(self) -> str:
        return self.metadata.get("title") or self.key.replace("_", " ").title()


def _parse_frontmatter(text: str) -> tuple:
    """
    Parsea frontmatter `---\\nkey: value\\n---\\n` mínimo (sin dependencia de PyYAML).
    Devuelve (metadata_dict, body_str). Si no hay frontmatter, devuelve ({}, text).
    """
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text
    raw = text[3:end].strip()
    body_start = end + len("\n---")
    if body_start < len(text) and text[body_start] == "\n":
        body_start += 1
    metadata: Dict[str, Any] = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        # parse list literal "[a, b, c]"
        if v.startswith("[") and v.endswith("]"):
            items = [x.strip().strip('"').strip("'") for x in v[1:-1].split(",")]
            metadata[k] = [i for i in items if i]
        elif v.lower() in ("true", "false"):
            metadata[k] = v.lower() == "true"
        else:
            try:
                metadata[k] = int(v)
            except ValueError:
                try:
                    metadata[k] = float(v)
                except ValueError:
                    metadata[k] = v
    return metadata, text[body_start:]


class SkillsLoader:
    """
    Carga, indexa y busca skills.

    Uso:
        loader = SkillsLoader()
        loader.load_dir("skills/")
        loader.attach_to_mem(mem)
        # más adelante, antes de generar:
        skills = loader.search("how do I write a python decorator?", mem)
        context = loader.format_for_injection(skills)
    """

    def __init__(self, threshold: float = DEFAULT_THRESHOLD) -> None:
        self.threshold = threshold
        self.skills: Dict[str, Skill] = {}

    # ── carga ──────────────────────────────────────────────────────────

    def load_file(self, path) -> Skill:
        p = Path(path)
        text = p.read_text(encoding="utf-8")
        metadata, body = _parse_frontmatter(text)
        key = metadata.get("name") or p.stem
        skill = Skill(key=key, content=body.strip(), metadata=metadata, path=p)
        self.skills[key] = skill
        return skill

    def load_dir(self, dir_path) -> List[Skill]:
        d = Path(dir_path)
        if not d.exists() or not d.is_dir():
            return []
        loaded: List[Skill] = []
        for p in sorted(d.glob("*.md")):
            try:
                loaded.append(self.load_file(p))
            except Exception:
                continue
        return loaded

    def add_skill(self, key: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> Skill:
        skill = Skill(key=key, content=content, metadata=metadata or {})
        self.skills[key] = skill
        return skill

    # ── persistencia en MEM ────────────────────────────────────────────

    def attach_to_mem(self, mem: Any) -> int:
        """
        Vuelca todas las skills cargadas a MEM con domain="skill".
        Devuelve el número de skills almacenadas.
        """
        if mem is None:
            return 0
        n = 0
        for skill in self.skills.values():
            try:
                mem.store(skill.key, skill.content, domain=SKILL_DOMAIN)
                n += 1
            except Exception:
                continue
        return n

    # ── búsqueda ───────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        mem: Any,
        top_k: int = DEFAULT_TOP_K,
        threshold: Optional[float] = None,
    ) -> List[tuple]:
        """
        Busca skills relevantes para `query` en MEM.

        Devuelve la lista filtrada por score >= threshold,
        ordenada de mayor a menor.
        Cada elemento es (key, content, score).
        """
        if mem is None or not query:
            return []
        thr = self.threshold if threshold is None else threshold
        try:
            results = mem.search(query, top_k=max(top_k * 2, top_k), domain=SKILL_DOMAIN)
        except TypeError:
            # backend sin soporte de dominio: filtramos manualmente
            results = mem.search(query, top_k=max(top_k * 2, top_k))
            results = [
                r for r in results
                if isinstance(r, tuple) and len(r) >= 2 and r[0] in self.skills
            ]
        filtered = [
            r for r in results
            if isinstance(r, tuple) and len(r) >= 3 and float(r[2]) >= thr
        ]
        return filtered[:top_k]

    # ── formato canónico para inyección ────────────────────────────────

    def format_for_injection(self, skills: List[tuple]) -> str:
        """
        Genera el bloque `[SKILL: ...]` que se pre-pendea al query.
        Cumple la Parte 15 (formato canónico).

        Acepta:
          - lista de tuplas (key, content, score)
          - lista de objetos Skill
        """
        lines: List[str] = []
        for item in skills:
            if isinstance(item, Skill):
                lines.append(f"[SKILL: {item.content}]")
            elif isinstance(item, tuple) and len(item) >= 2:
                content = item[1]
                lines.append(f"[SKILL: {content}]")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self.skills)


__all__ = [
    "SKILL_DOMAIN",
    "DEFAULT_THRESHOLD",
    "DEFAULT_TOP_K",
    "Skill",
    "SkillsLoader",
]
