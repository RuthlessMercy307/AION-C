"""
synth/canonical_format.py — Formato canónico de la Parte 15 del MEGA-PROMPT
=============================================================================

Formato:
    [SKILL: contenido]    (opcional)
    [MEM:   contenido]    (opcional)
    [USER:  pregunta]
    [TOOL:  json call]    (opcional)
    [RESULT: salida]      (opcional)
    [AION:  respuesta]
    [EOS]

Reglas:
  - SIEMPRE termina con [EOS]
  - SKILL/MEM al principio (en ese orden), antes del primer USER
  - TOOL puede aparecer entre USER y AION (modelo decidió usar tool)
  - RESULT siempre acompaña a un TOOL anterior
  - Multi-turn: secuencia [USER:] [AION:] [USER:] [AION:] ... [EOS]
  - El [EOS] solo aparece UNA vez, al final del ejemplo

Este módulo NO genera datos. Provee:
  - CanonicalRecord — record con texto + metadata
  - format_record() — convierte campos estructurados → texto canónico
  - parse_canonical() — parsea texto canónico → estructura
  - has_eos() / strip_eos() — utilidades
  - canonicalize_legacy() — convierte el formato {input, output, ...} antiguo
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


EOS_MARKER = "[EOS]"

# Tags reconocidos en el orden canónico
TAGS_PRE_USER  = ("SKILL", "MEM")
TAG_USER       = "USER"
TAG_TOOL       = "TOOL"
TAG_RESULT     = "RESULT"
TAG_AION       = "AION"


@dataclass
class CanonicalTurn:
    """Un turno en una conversación (puede llevar tool/result intermedios)."""
    user:    str
    aion:    str
    tool:    Optional[str] = None
    result:  Optional[str] = None


@dataclass
class CanonicalRecord:
    """Un ejemplo en formato canónico, con texto y metadata."""
    text:        str
    has_skill:   bool = False
    has_mem:     bool = False
    has_tool:    bool = False
    is_multi_turn: bool = False
    turn_count:  int = 1
    domain:      str = "general"
    language:    str = "en"
    type:        str = "single"   # single | multi_turn | tool | skill | mem | identity
    metadata:    Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text":          self.text,
            "has_skill":     self.has_skill,
            "has_mem":       self.has_mem,
            "has_tool":      self.has_tool,
            "is_multi_turn": self.is_multi_turn,
            "turn_count":    self.turn_count,
            "domain":        self.domain,
            "language":      self.language,
            "type":          self.type,
            "metadata":      dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CanonicalRecord":
        return cls(
            text=data["text"],
            has_skill=data.get("has_skill", False),
            has_mem=data.get("has_mem", False),
            has_tool=data.get("has_tool", False),
            is_multi_turn=data.get("is_multi_turn", False),
            turn_count=data.get("turn_count", 1),
            domain=data.get("domain", "general"),
            language=data.get("language", "en"),
            type=data.get("type", "single"),
            metadata=dict(data.get("metadata", {})),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Builders
# ─────────────────────────────────────────────────────────────────────────────


def _bracket(tag: str, content: str) -> str:
    """Devuelve '[TAG: content]' con el content normalizado a una sola línea."""
    if content is None:
        return ""
    s = str(content).replace("\r", " ").strip()
    # No colapsamos newlines internos para que código/poemas mantengan estructura,
    # pero sí garantizamos que no haya '[' sin escapar al inicio (no estricto).
    return f"[{tag}: {s}]"


def format_record(
    user: str,
    aion: str,
    *,
    skill: Optional[str] = None,
    mem:   Optional[str] = None,
    tool:  Optional[str] = None,
    result: Optional[str] = None,
    extra_turns: Optional[List[CanonicalTurn]] = None,
) -> str:
    """
    Formatea un ejemplo en texto canónico SIEMPRE terminando con [EOS].

    Para multi-turn, usar `extra_turns` con CanonicalTurn adicionales.
    El primer turno se forma con `user` + (opcional tool/result) + `aion`.
    """
    parts: List[str] = []
    if skill:
        parts.append(_bracket("SKILL", skill))
    if mem:
        parts.append(_bracket("MEM", mem))
    parts.append(_bracket(TAG_USER, user))
    if tool:
        parts.append(_bracket(TAG_TOOL, tool))
    if result:
        parts.append(_bracket(TAG_RESULT, result))
    parts.append(_bracket(TAG_AION, aion))
    for t in extra_turns or []:
        parts.append(_bracket(TAG_USER, t.user))
        if t.tool:
            parts.append(_bracket(TAG_TOOL, t.tool))
        if t.result:
            parts.append(_bracket(TAG_RESULT, t.result))
        parts.append(_bracket(TAG_AION, t.aion))
    parts.append(EOS_MARKER)
    return "\n".join(parts)


def build_record(
    user: str,
    aion: str,
    *,
    skill: Optional[str] = None,
    mem:   Optional[str] = None,
    tool:  Optional[str] = None,
    result: Optional[str] = None,
    extra_turns: Optional[List[CanonicalTurn]] = None,
    domain: str = "general",
    language: str = "en",
    type:   str = "single",
    metadata: Optional[Dict[str, Any]] = None,
) -> CanonicalRecord:
    """Crea un CanonicalRecord listo para serializar."""
    text = format_record(
        user=user, aion=aion,
        skill=skill, mem=mem,
        tool=tool, result=result,
        extra_turns=extra_turns,
    )
    n_turns = 1 + (len(extra_turns) if extra_turns else 0)
    return CanonicalRecord(
        text=text,
        has_skill=bool(skill),
        has_mem=bool(mem),
        has_tool=bool(tool) or any(t.tool for t in (extra_turns or [])),
        is_multi_turn=n_turns > 1,
        turn_count=n_turns,
        domain=domain,
        language=language,
        type=type,
        metadata=dict(metadata or {}),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Verificadores
# ─────────────────────────────────────────────────────────────────────────────


def has_eos(text: str) -> bool:
    """True si el texto termina con [EOS] (después de strippear)."""
    if not text:
        return False
    return text.rstrip().endswith(EOS_MARKER)


def strip_eos(text: str) -> str:
    """Devuelve el texto sin el [EOS] terminal."""
    if has_eos(text):
        return text.rstrip()[: -len(EOS_MARKER)].rstrip()
    return text


def count_tags(text: str) -> Dict[str, int]:
    """Cuenta cuántas veces aparece cada tag en el texto."""
    out = {tag: 0 for tag in (*TAGS_PRE_USER, TAG_USER, TAG_TOOL, TAG_RESULT, TAG_AION)}
    for tag in out:
        out[tag] = len(re.findall(rf"\[{tag}\s*:", text))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Parser (inverso de format_record)
# ─────────────────────────────────────────────────────────────────────────────


_BLOCK_RE = re.compile(
    r"\[(SKILL|MEM|USER|TOOL|RESULT|AION)\s*:\s*",
    re.IGNORECASE,
)


def parse_canonical(text: str) -> List[Tuple[str, str]]:
    """
    Parsea el texto canónico en una lista de (tag, content) en orden.

    Tolerante: ignora [EOS] al final. No falla si falta algún tag.
    Usa balanceo de corchetes para soportar contenidos con '[' o ']' anidados.
    """
    if not text:
        return []
    body = strip_eos(text)
    out: List[Tuple[str, str]] = []
    pos = 0
    while True:
        m = _BLOCK_RE.search(body, pos)
        if not m:
            break
        tag = m.group(1).upper()
        content_start = m.end()
        # Encontrar el ']' que cierra este bloque, contemplando anidados
        depth = 1
        i = content_start
        while i < len(body) and depth > 0:
            c = body[i]
            if c == "[":
                depth += 1
            elif c == "]":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        if depth != 0:
            break  # malformado, abortamos
        content = body[content_start:i].strip()
        out.append((tag, content))
        pos = i + 1
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Conversión de formato legacy {input, output, domain, ...} → canónico
# ─────────────────────────────────────────────────────────────────────────────


def canonicalize_legacy(record: Dict[str, Any]) -> CanonicalRecord:
    """
    Convierte un record legacy (como dataset_50k.jsonl) al formato canónico.

    Legacy: {"input": ..., "output": ..., "domain": ..., "graph": ..., "language": ...}
    Canónico: [USER: input] [AION: output] [EOS]
    Si hay graph con nodos, lo serializa como [MEM: ...] (opcional).
    """
    inp = record.get("input", "")
    out = record.get("output", "")
    domain = record.get("domain", "general")
    language = record.get("language", "en")
    graph = record.get("graph") or {}

    # Si el grafo tiene contenido lo convertimos en MEM textual breve
    mem_text: Optional[str] = None
    nodes = graph.get("nodes") if isinstance(graph, dict) else None
    edges = graph.get("edges") if isinstance(graph, dict) else None
    if nodes:
        node_labels = [str(n.get("label", n.get("id", ""))) for n in nodes][:6]
        edge_strs: List[str] = []
        if edges:
            for e in edges[:6]:
                src = e.get("source", "?")
                tgt = e.get("target", "?")
                rel = e.get("relation", "?")
                edge_strs.append(f"{src}-{rel}->{tgt}")
        parts = []
        if node_labels:
            parts.append("nodes=" + ",".join(node_labels))
        if edge_strs:
            parts.append("edges=" + "; ".join(edge_strs))
        if parts:
            mem_text = " | ".join(parts)

    return build_record(
        user=str(inp),
        aion=str(out),
        mem=mem_text,
        domain=str(domain),
        language=str(language),
        type="legacy",
        metadata={"source": "dataset_50k", "domain_id": record.get("domain_id")},
    )


__all__ = [
    "EOS_MARKER",
    "TAGS_PRE_USER", "TAG_USER", "TAG_TOOL", "TAG_RESULT", "TAG_AION",
    "CanonicalTurn", "CanonicalRecord",
    "format_record", "build_record",
    "has_eos", "strip_eos", "count_tags",
    "parse_canonical", "canonicalize_legacy",
]
