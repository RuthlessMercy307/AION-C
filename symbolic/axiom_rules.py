"""
symbolic/axiom_rules.py — Reglas formales para AXIOM (Parte 20.1)
==================================================================

  - TransitivityRule:  si A→B y B→C entonces A→C
  - ContradictionRule: si A y ¬A coexisten → conflicto
  - SubstitutionRule:  si x=5, reemplaza x por 5 en labels/props
  - ArithmeticRule:    nodos con label "3 + 4" → calcula y crea nodo "7"

Las reglas usan las siguientes convenciones de relación:
  "implies"  — implicación lógica (A→B en sentido formal)
  "equals"   — equivalencia/asignación (x = 5)
  "negates"  — negación (A es ¬B)
  "computes" — vincula una expresión a su resultado
"""

from __future__ import annotations

import ast
import operator
import re
from typing import Optional

from .graph import SymbolicGraph, SymbolicNode, SymbolicEdge
from .rules import SymbolicRule, RuleResult


class TransitivityRule(SymbolicRule):
    """Si hay A--implies-->B y B--implies-->C, agrega A--implies-->C."""

    name  = "axiom.transitivity"
    motor = "axiom"

    def applies_to(self, graph: SymbolicGraph) -> bool:
        return len(graph.edges_with_relation("implies")) >= 2

    def apply(self, graph: SymbolicGraph) -> RuleResult:
        result = RuleResult()
        implies_edges = graph.edges_with_relation("implies")
        existing_pairs = {(e.source, e.target) for e in implies_edges}
        for e1 in implies_edges:
            for e2 in implies_edges:
                if e1.target == e2.source and e1.source != e2.target:
                    pair = (e1.source, e2.target)
                    if pair not in existing_pairs:
                        new_edge = SymbolicEdge(
                            source=e1.source, target=e2.target,
                            relation="implies",
                            props={"derived_by": "transitivity"},
                        )
                        graph.add_edge(new_edge)
                        result.added_edges.append(new_edge)
                        result.modified = True
                        existing_pairs.add(pair)
                        result.notes.append(
                            f"transitivity: {e1.source} → {e1.target} → {e2.target}"
                        )
        return result


class ContradictionRule(SymbolicRule):
    """
    Detecta nodos con label "A" y "¬A" (o A--negates-->B y A--implies-->B)
    como conflicto lógico.
    """

    name  = "axiom.contradiction"
    motor = "axiom"

    def apply(self, graph: SymbolicGraph) -> RuleResult:
        result = RuleResult()
        # Caso 1: nodos con label P y ¬P (o not P)
        labels = {n.id: (n.label or n.id) for n in graph.nodes}
        for node_id, label in labels.items():
            stripped = self._strip_negation(label)
            if stripped is None:
                continue
            for other_id, other_label in labels.items():
                if other_id == node_id:
                    continue
                if other_label.strip() == stripped.strip():
                    msg = f"contradiction: '{label}' and '{other_label}' both present"
                    if msg not in result.conflicts:
                        result.conflicts.append(msg)
                        result.modified = True
        # Caso 2: A implica B y A niega B simultáneamente
        for e in graph.edges_with_relation("implies"):
            if graph.has_edge(e.source, e.target, relation="negates"):
                result.conflicts.append(
                    f"contradiction: {e.source} both implies and negates {e.target}"
                )
                result.modified = True
        return result

    @staticmethod
    def _strip_negation(label: str) -> Optional[str]:
        if not label:
            return None
        s = label.strip()
        if s.startswith("¬"):
            return s[1:].strip()
        if s.lower().startswith("not "):
            return s[4:].strip()
        return None


class SubstitutionRule(SymbolicRule):
    """
    Si existe una arista X--equals-->5, reemplaza el label "X" en cualquier
    nodo cuyo label coincida o lo contenga (substitución textual).
    """

    name  = "axiom.substitution"
    motor = "axiom"

    def applies_to(self, graph: SymbolicGraph) -> bool:
        return any(e.relation == "equals" for e in graph.edges)

    def apply(self, graph: SymbolicGraph) -> RuleResult:
        result = RuleResult()
        for e in graph.edges_with_relation("equals"):
            var_node = graph.find_node(e.source)
            val_node = graph.find_node(e.target)
            if var_node is None or val_node is None:
                continue
            var_label = var_node.label or var_node.id
            val_label = val_node.label or val_node.id
            if not var_label or var_label == val_label:
                continue
            # Sustituir en otros nodos
            for n in graph.nodes:
                if n.id == var_node.id:
                    continue
                if not n.label:
                    continue
                new_label = re.sub(r"\b" + re.escape(var_label) + r"\b", val_label, n.label)
                if new_label != n.label:
                    n.props.setdefault("original_label", n.label)
                    n.label = new_label
                    result.modified = True
                    result.notes.append(
                        f"substituted '{var_label}' → '{val_label}' in {n.id}"
                    )
        return result


_ARITH_LABEL_RE = re.compile(
    r"^\s*([\d\.\-]+)\s*([+\-*/×x÷])\s*([\d\.\-]+)\s*$"
)


class ArithmeticRule(SymbolicRule):
    """
    Si un nodo tiene label como "3 + 4" o "0.15 * 240", calcula el resultado
    y crea una arista 'computes' al nodo resultado (creando uno si hace falta).
    """

    name  = "axiom.arithmetic"
    motor = "axiom"

    OP_MAP = {
        "+": operator.add,
        "-": operator.sub,
        "*": operator.mul, "x": operator.mul, "×": operator.mul,
        "/": operator.truediv, "÷": operator.truediv,
    }

    def apply(self, graph: SymbolicGraph) -> RuleResult:
        result = RuleResult()
        for n in list(graph.nodes):
            label = (n.label or n.id).strip()
            m = _ARITH_LABEL_RE.match(label)
            if not m:
                continue
            try:
                a = float(m.group(1))
                op = m.group(2)
                b = float(m.group(3))
            except ValueError:
                continue
            if op not in self.OP_MAP:
                continue
            try:
                value = self.OP_MAP[op](a, b)
            except ZeroDivisionError:
                result.conflicts.append(f"arithmetic: division by zero in '{label}'")
                continue
            value_str = str(int(value)) if isinstance(value, float) and value.is_integer() else str(value)
            result_id = f"result_{n.id}"
            if not graph.has_node(result_id):
                new_node = SymbolicNode(id=result_id, label=value_str, type="number")
                graph.add_node(new_node)
                result.added_nodes.append(new_node)
                result.modified = True
            if not graph.has_edge(n.id, result_id, "computes"):
                new_edge = SymbolicEdge(
                    source=n.id, target=result_id, relation="computes",
                    props={"value": value_str},
                )
                graph.add_edge(new_edge)
                result.added_edges.append(new_edge)
                result.modified = True
            result.notes.append(f"arithmetic: {label} = {value_str}")
        return result


AXIOM_RULES = [
    TransitivityRule(),
    ContradictionRule(),
    SubstitutionRule(),
    ArithmeticRule(),
]


__all__ = [
    "TransitivityRule",
    "ContradictionRule",
    "SubstitutionRule",
    "ArithmeticRule",
    "AXIOM_RULES",
]
