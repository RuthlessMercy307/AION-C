"""
memory/user_model.py — User model persistence (Parte 8.3 del MEGA-PROMPT)
==========================================================================

Guarda en MEM persistente atributos del usuario para personalizar respuestas:
  - name (si lo dice)
  - preferred_language (es | en | mixed)
  - technical_level (beginner | intermediate | advanced)
  - preferred_tone (formal | casual)
  - projects (lista de proyectos actuales)
  - facts (key→value libre, p.ej. {"editor": "vscode"})

El UserModel se serializa a JSON y se almacena en SemanticStore con
domain="user_model" bajo la clave canónica "user_profile".
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


USER_MODEL_DOMAIN = "user_model"
USER_MODEL_KEY    = "user_profile"

VALID_LANGUAGES = ("es", "en", "mixed")
VALID_LEVELS    = ("beginner", "intermediate", "advanced", "unknown")
VALID_TONES     = ("formal", "casual", "neutral")


@dataclass
class UserModel:
    """Modelo persistente del usuario."""
    name:               Optional[str] = None
    preferred_language: str = "mixed"
    technical_level:    str = "unknown"
    preferred_tone:     str = "neutral"
    projects:           List[str] = field(default_factory=list)
    facts:              Dict[str, str] = field(default_factory=dict)

    # ── validators / setters ──────────────────────────────────────────

    def set_name(self, name: str) -> None:
        if name and isinstance(name, str):
            self.name = name.strip()

    def set_language(self, lang: str) -> None:
        if lang in VALID_LANGUAGES:
            self.preferred_language = lang

    def set_technical_level(self, level: str) -> None:
        if level in VALID_LEVELS:
            self.technical_level = level

    def set_tone(self, tone: str) -> None:
        if tone in VALID_TONES:
            self.preferred_tone = tone

    def add_project(self, project: str) -> None:
        p = (project or "").strip()
        if p and p not in self.projects:
            self.projects.append(p)

    def remove_project(self, project: str) -> None:
        if project in self.projects:
            self.projects.remove(project)

    def set_fact(self, key: str, value: str) -> None:
        if key and value:
            self.facts[str(key)] = str(value)

    def get_fact(self, key: str) -> Optional[str]:
        return self.facts.get(key)

    # ── serialización ──────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserModel":
        return cls(
            name=data.get("name"),
            preferred_language=data.get("preferred_language", "mixed"),
            technical_level=data.get("technical_level", "unknown"),
            preferred_tone=data.get("preferred_tone", "neutral"),
            projects=list(data.get("projects", [])),
            facts=dict(data.get("facts", {})),
        )

    @classmethod
    def from_json(cls, text: str) -> "UserModel":
        return cls.from_dict(json.loads(text))

    # ── persistencia en MEM ────────────────────────────────────────────

    def save_to_mem(self, mem: Any, key: str = USER_MODEL_KEY) -> None:
        if mem is None:
            return
        mem.store(key, self.to_json(), domain=USER_MODEL_DOMAIN)

    @classmethod
    def load_from_mem(cls, mem: Any, key: str = USER_MODEL_KEY) -> Optional["UserModel"]:
        if mem is None:
            return None
        try:
            results = mem.search(key, top_k=5)
        except Exception:
            return None
        for item in results or []:
            if isinstance(item, tuple) and len(item) >= 2 and item[0] == key:
                try:
                    return cls.from_json(item[1])
                except Exception:
                    return None
        return None

    # ── render para inyección en prompts ────────────────────────────────

    def render_for_context(self) -> str:
        """Texto compacto para inyectar en el contexto del modelo."""
        parts = []
        if self.name:
            parts.append(f"name={self.name}")
        if self.preferred_language != "mixed":
            parts.append(f"lang={self.preferred_language}")
        if self.technical_level != "unknown":
            parts.append(f"level={self.technical_level}")
        if self.preferred_tone != "neutral":
            parts.append(f"tone={self.preferred_tone}")
        if self.projects:
            parts.append(f"projects=[{', '.join(self.projects[:3])}]")
        return "user: " + ", ".join(parts) if parts else ""


__all__ = [
    "UserModel",
    "USER_MODEL_DOMAIN",
    "USER_MODEL_KEY",
    "VALID_LANGUAGES",
    "VALID_LEVELS",
    "VALID_TONES",
]
