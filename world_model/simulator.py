"""
world_model/simulator.py — Simuladores por motor (Parte 19.1, 19.3)
=====================================================================

Cada motor tiene un WorldSimulator que ejecuta el query SOBRE el scratch pad,
poblando los slots según el schema correspondiente.

Los simuladores aquí son HEURÍSTICOS — no necesitan modelo neural. La idea es
que cuando exista un motor real, se inyecte como `model_fn` opcional. Pero
para AION-C la *capacidad* de simular calculadoramente (sin haber visto el
patrón en training) es lo que diferencia el world model de un LLM puro.

El AxiomSimulator implementa el ejemplo emblemático de la Parte 19.3:
  "15% de 240 = ?"
  → 0.15 × 240
  → 0.15 × 200 = 30
  → 0.15 × 40 = 6
  → 30 + 6 = 36
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .scratch_pad import (
    ScratchPad, ScratchPadSchema,
    FORGE_C_SCHEMA, AXIOM_SCHEMA, CORA_SCHEMA, MUSE_SCHEMA, EMPATHY_SCHEMA,
)


# ─────────────────────────────────────────────────────────────────────────────
# Base
# ─────────────────────────────────────────────────────────────────────────────


class WorldSimulator:
    """Base ABC. Cada motor implementa `simulate`."""

    motor: str = ""
    schema: Optional[ScratchPadSchema] = None

    def fresh_pad(self) -> ScratchPad:
        return ScratchPad(schema=self.schema)

    def simulate(self, query: str, pad: Optional[ScratchPad] = None) -> ScratchPad:
        if pad is None:
            pad = self.fresh_pad()
        return self._simulate(query, pad)

    def _simulate(self, query: str, pad: ScratchPad) -> ScratchPad:
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
# AXIOM — el caso emblemático
# ─────────────────────────────────────────────────────────────────────────────


_PERCENT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%\s*(?:de|of)\s*(\d+(?:\.\d+)?)")
_ARITH_RE   = re.compile(r"(\d+(?:\.\d+)?)\s*([+\-*x×÷/])\s*(\d+(?:\.\d+)?)")


class AxiomSimulator(WorldSimulator):
    motor = "axiom"
    schema = AXIOM_SCHEMA

    def _simulate(self, query: str, pad: ScratchPad) -> ScratchPad:
        q = (query or "").strip().lower()

        # Caso 1: porcentaje "X% de Y"
        m = _PERCENT_RE.search(q)
        if m:
            return self._simulate_percent(float(m.group(1)), float(m.group(2)), pad)

        # Caso 2: aritmética binaria
        m = _ARITH_RE.search(q)
        if m:
            return self._simulate_arith(float(m.group(1)), m.group(2), float(m.group(3)), pad)

        # Caso 3: prueba simbólica genérica (placeholder)
        pad.set_by_name("proven", [])
        pad.set_by_name("to_prove", [q])
        return pad

    def _simulate_percent(self, p: float, n: float, pad: ScratchPad) -> ScratchPad:
        decimal = p / 100.0
        # Descomponer n en cientos y resto para mostrar el paso a paso
        hundreds = (int(n) // 100) * 100
        rest = n - hundreds
        h_part = decimal * hundreds
        r_part = decimal * rest
        result = h_part + r_part
        if result.is_integer():
            result_str = str(int(result))
        else:
            result_str = f"{result:.4f}".rstrip("0").rstrip(".")
        proven = [
            f"{p}% = {decimal}",
            f"{decimal} × {hundreds} = {h_part:g}",
            f"{decimal} × {rest:g} = {r_part:g}",
            f"{h_part:g} + {r_part:g} = {result_str}",
        ]
        pad.set_by_name("proven", proven)
        pad.set_by_name("to_prove", [])
        pad.set_by_name("hypotheses", [f"{p}% of {n}"])
        return pad

    def _simulate_arith(self, a: float, op: str, b: float, pad: ScratchPad) -> ScratchPad:
        op_map = {"+": "+", "-": "-", "*": "*", "x": "*", "×": "*", "÷": "/", "/": "/"}
        py_op = op_map[op]
        try:
            result = {
                "+": a + b,
                "-": a - b,
                "*": a * b,
                "/": (a / b) if b != 0 else float("nan"),
            }[py_op]
        except Exception:
            result = float("nan")
        if isinstance(result, float) and result.is_integer():
            result_str = str(int(result))
        else:
            result_str = f"{result}"
        proven = [
            f"{a:g} {op} {b:g}",
            f"= {result_str}",
        ]
        pad.set_by_name("proven", proven)
        pad.set_by_name("to_prove", [])
        return pad


# ─────────────────────────────────────────────────────────────────────────────
# FORGE-C
# ─────────────────────────────────────────────────────────────────────────────


class ForgeCSimulator(WorldSimulator):
    motor = "forge_c"
    schema = FORGE_C_SCHEMA

    def _simulate(self, query: str, pad: ScratchPad) -> ScratchPad:
        q = (query or "").strip()
        # Heurística simple: detectar funciones definidas
        funcs = re.findall(r"def\s+(\w+)", q)
        variables: Dict[str, Any] = {}
        # Detectar asignaciones simples "x = 5"
        for m in re.finditer(r"(\w+)\s*=\s*(\d+(?:\.\d+)?|\".*?\"|'.*?'|None|null)", q):
            name = m.group(1)
            raw = m.group(2)
            try:
                if raw in ("None", "null"):
                    val: Any = None
                elif raw.startswith(("'", '"')):
                    val = raw.strip("'\"")
                elif "." in raw:
                    val = float(raw)
                else:
                    val = int(raw)
            except ValueError:
                val = raw
            variables[name] = val
        pad.set_by_name("variables", variables)
        pad.set_by_name("call_stack", funcs or ["main"])

        # Detección heurística de errores típicos
        if "null" in q.lower() and "." in q and "if" not in q.lower():
            pad.set_by_name("error", "possible null dereference")
        return pad


# ─────────────────────────────────────────────────────────────────────────────
# CORA — propagación causal
# ─────────────────────────────────────────────────────────────────────────────


_CAUSAL_RE = re.compile(
    r"(?:si|if)\s+(.+?)\s+(?:causa|causes|then|entonces)\s+(.+?)(?:[,.;\?]|$)",
    re.IGNORECASE,
)


class CoraSimulator(WorldSimulator):
    motor = "cora"
    schema = CORA_SCHEMA

    # Mini-grafo causal heurístico para cadenas conocidas
    KNOWN_CHAINS = {
        "lluvia": ["suelo mojado", "deslizamiento"],
        "rain":   ["wet soil", "landslide"],
        "fuego":  ["humo", "evacuación"],
        "fire":   ["smoke", "evacuation"],
    }

    def _simulate(self, query: str, pad: ScratchPad) -> ScratchPad:
        q = (query or "").strip().lower()

        # Caso 1: hay una afirmación causal explícita "si X causa Y"
        m = _CAUSAL_RE.search(q)
        if m:
            cause = m.group(1).strip()
            effect = m.group(2).strip()
            pad.set_by_name("causes", [cause])
            pad.set_by_name("direct_effects", [effect])
            chain = self._extend(effect)
            pad.set_by_name("indirect_effects", chain)
            if chain:
                pad.set_by_name("prediction", f"sí, {cause} causa {chain[-1]} vía {effect}")
            else:
                pad.set_by_name("prediction", f"sí, {cause} causa {effect}")
            return pad

        # Caso 2: cadena conocida
        for trigger, chain in self.KNOWN_CHAINS.items():
            if trigger in q:
                pad.set_by_name("causes", [trigger])
                pad.set_by_name("direct_effects", [chain[0]])
                pad.set_by_name("indirect_effects", chain[1:] if len(chain) > 1 else [])
                pad.set_by_name("prediction", f"sí, {trigger} → {' → '.join(chain)}")
                return pad

        # Default: no se detectó causa
        pad.set_by_name("causes", [])
        pad.set_by_name("direct_effects", [])
        pad.set_by_name("prediction", "no se detectó relación causal en el query")
        return pad

    def _extend(self, effect: str) -> List[str]:
        for trigger, chain in self.KNOWN_CHAINS.items():
            if trigger in effect or effect in chain:
                idx = chain.index(effect) if effect in chain else -1
                return chain[idx + 1:] if idx >= 0 else chain
        return []


# ─────────────────────────────────────────────────────────────────────────────
# MUSE — estado narrativo
# ─────────────────────────────────────────────────────────────────────────────


class MuseSimulator(WorldSimulator):
    motor = "muse"
    schema = MUSE_SCHEMA

    def _simulate(self, query: str, pad: ScratchPad) -> ScratchPad:
        q = (query or "").strip().lower()
        # Detectar palabras-tensor
        tension_markers = ("muerte", "death", "lucha", "fight", "perder", "lose", "amor", "love")
        tension = 0.5
        for m in tension_markers:
            if m in q:
                tension = min(1.0, tension + 0.15)
        # Conflictos detectados muy crudo
        conflicts: List[str] = []
        if any(w in q for w in ("vs", "contra", "against")):
            conflicts.append("oposición explícita")
        if "robot" in q and any(w in q for w in ("hombre", "human", "humano")):
            conflicts.append("robot vs humano")
        if not conflicts:
            conflicts.append("interno: protagonista vs sus dudas")
        pad.set_by_name("tension", float(tension))
        pad.set_by_name("conflicts", conflicts)
        pad.set_by_name("expectation", "el lector espera resolución")
        if tension >= 0.8:
            pad.set_by_name("subversion", "la resolución no es la esperada")
        return pad


# ─────────────────────────────────────────────────────────────────────────────
# EMPATHY — modelo mental del usuario
# ─────────────────────────────────────────────────────────────────────────────


class EmpathySimulator(WorldSimulator):
    motor = "empathy"
    schema = EMPATHY_SCHEMA

    EMOTION_MARKERS = {
        "frustración": ("no funciona", "broken", "doesn't work", "harto", "tired"),
        "tristeza":    ("triste", "sad", "perdí", "lost", "solo", "alone"),
        "ansiedad":    ("ansioso", "anxious", "miedo", "afraid", "preocupado"),
        "alegría":     ("feliz", "happy", "logré", "achieved", "gracias"),
        "enojo":       ("enojado", "angry", "molesto", "furioso"),
    }

    def _simulate(self, query: str, pad: ScratchPad) -> ScratchPad:
        q = (query or "").strip().lower()
        emotion = "neutral"
        for emo, markers in self.EMOTION_MARKERS.items():
            if any(m in q for m in markers):
                emotion = emo
                break
        # Causa probable: heurística cruda buscando "porque"/"because"
        cause = ""
        m = re.search(r"(?:porque|because)\s+(.+?)(?:[.\?!]|$)", q)
        if m:
            cause = m.group(1).strip()
        else:
            cause = "no identificada"
        # Necesidad
        need_map = {
            "frustración": "validación + diagnóstico técnico",
            "tristeza":    "presencia + validación, sin solucionar",
            "ansiedad":    "respiración + reframe + paso pequeño",
            "alegría":     "celebrar con el usuario",
            "enojo":       "validar enojo, no defender",
            "neutral":     "responder al contenido directamente",
        }
        strategy_map = {
            "frustración": "empatía primero, solución después",
            "tristeza":    "acompañar antes de aconsejar",
            "ansiedad":    "calma + un solo siguiente paso",
            "alegría":     "celebrar y reforzar",
            "enojo":       "validar sin escalar",
            "neutral":     "respuesta directa",
        }
        pad.set_by_name("emotion", emotion)
        pad.set_by_name("probable_cause", cause)
        pad.set_by_name("need", need_map[emotion])
        pad.set_by_name("response_strategy", strategy_map[emotion])
        return pad


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────


def build_default_simulators() -> Dict[str, WorldSimulator]:
    return {
        "forge_c": ForgeCSimulator(),
        "axiom":   AxiomSimulator(),
        "cora":    CoraSimulator(),
        "muse":    MuseSimulator(),
        "empathy": EmpathySimulator(),
    }


__all__ = [
    "WorldSimulator",
    "ForgeCSimulator",
    "AxiomSimulator",
    "CoraSimulator",
    "MuseSimulator",
    "EmpathySimulator",
    "build_default_simulators",
]
