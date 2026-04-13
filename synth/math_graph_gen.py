"""
synth/math_graph_gen.py — Generador de Grafos de Prueba Matemática para AXIOM
===============================================================================

Motor de datos sintéticos para el motor AXIOM (razonamiento matemático).
Genera problemas con RESPUESTAS VERIFICABLES PROGRAMÁTICAMENTE:

  - Aritmética/álgebra:    eval() de expresiones Python
  - Ecuaciones lineales:   ax + b = c → x = (c - b) / a  (aritmética exacta)
  - Desigualdades:         transitivity chain a > b > c → a > c
  - Paridad/divisibilidad: n % 2 == 0 → par, propiedades algebraicas
  - Pruebas básicas:       suma de pares es par, producto de pares es par

Tres niveles de complejidad:

  Nivel 1 — Paso único (2-3 nodos), aritmética directa
             '¿Cuánto es 3 × 4 + 2?'  → eval("3*4+2") = 14
             '2x + 3 = 7, ¿cuánto es x?' → x = (7-3)/2 = 2.0
             'Si a > b y b > c, ¿a > c?' → transitivity directa

  Nivel 2 — Dos pasos (3-4 nodos), ecuaciones, desigualdades compuestas
             'Resuelve: 3x - 5 = 10' → x = 5.0 (verificado con eval)
             'a > b, b > c, c > d → ¿a > d?' (cadena 3 saltos)
             '¿n=12 es divisible por 3?' → 12 % 3 == 0 → True

  Nivel 3 — Tres+ pasos (4-6 nodos), pruebas algebraicas formales
             'Demuestra que la suma de dos pares es par'
             'Simplifica (x+1)(x-1) = x²-1' (identidad algebraica)
             'Si x²=9 y x>0, entonces x=3' (acotación + solución)

Contrato de cada MathExample:
  - problem_text:     descripción del problema en lenguaje natural
  - graph:            CausalGraph con MathNode/MathEdge
  - answer:           respuesta verificable (string o número)
  - numeric_answer:   Optional[float] — para verificación con eval()
  - complexity_level: 1-3
  - answer_type:      MathAnswerType
  - metadata:         parámetros para verify_math_example()
  - verifiable:       True siempre

Uso básico:
    gen = MathGraphGenerator(seed=42)
    ex  = gen.generate(level=1)
    res = verify_math_example(ex)
    assert res.passed

    batch = gen.generate_batch(n=100, level_distribution={1: 0.4, 2: 0.4, 3: 0.2})
"""

from __future__ import annotations

import math
import random
import uuid
from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
from typing import Dict, List, Optional, Tuple

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.graph import CausalGraph
from motors.axiom.relations import (
    MATH_RELATIONS,
    MathEdge,
    MathNode,
    MathNodeType,
    MathRelation,
)


# ─────────────────────────────────────────────────────────────────────────────
# TIPOS DE RESPUESTA
# ─────────────────────────────────────────────────────────────────────────────

class MathAnswerType(str, Enum):
    ARITHMETIC      = "arithmetic"      # eval() de una expresión numérica
    LINEAR_EQUATION = "linear_equation" # ax + b = c → x = ?
    INEQUALITY      = "inequality"      # transitivity / ordering
    DIVISIBILITY    = "divisibility"    # n % k == 0?
    PARITY_PROOF    = "parity_proof"    # suma/producto de pares/impares
    ALGEBRAIC_ID    = "algebraic_id"    # identidad algebraica (a+b)²=a²+2ab+b²
    BOUND           = "bound"           # si x²=k y x>0 → x=√k


# ─────────────────────────────────────────────────────────────────────────────
# RESULTADO DE VERIFICACIÓN
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MathVerificationResult:
    passed:  bool
    reason:  str
    details: Dict = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.passed

    def __repr__(self) -> str:
        return f"MathVerificationResult({'PASS' if self.passed else 'FAIL'}: {self.reason})"


# ─────────────────────────────────────────────────────────────────────────────
# MATH EXAMPLE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MathExample:
    """
    Unidad atómica de entrenamiento para AXIOM.

    numeric_answer:  para verificación exacta con eval() o comparación directa.
                     None para respuestas de tipo bool (p.ej. "¿a > c?").
    bool_answer:     para respuestas True/False (transitivity, divisibility, parity).
    """
    problem_text:     str
    graph:            CausalGraph
    answer:           str
    complexity_level: int
    answer_type:      MathAnswerType
    verifiable:       bool = True
    metadata:         Dict = field(default_factory=dict)
    example_id:       str  = field(default_factory=lambda: str(uuid.uuid4())[:12])
    numeric_answer:   Optional[float] = None
    bool_answer:      Optional[bool]  = None

    def __repr__(self) -> str:
        return (
            f"MathExample(level={self.complexity_level}, "
            f"type={self.answer_type.value}, "
            f"nodes={len(self.graph)}, id={self.example_id})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS — CONSTRUCCIÓN DE GRAFOS
# ─────────────────────────────────────────────────────────────────────────────

def _math_node(nid: str, label: str, ntype: MathNodeType, conf: float = 1.0) -> MathNode:
    return MathNode(node_id=nid, label=label, node_type=ntype,
                    confidence=conf, grounded=True)


def _math_edge(src: str, tgt: str, rel: MathRelation,
               strength: float = 1.0, conf: float = 1.0) -> MathEdge:
    return MathEdge(source_id=src, target_id=tgt, relation=rel,
                    strength=strength, confidence=conf)


# ─────────────────────────────────────────────────────────────────────────────
# GENERADOR NIVEL 1 — PASO ÚNICO (2-3 NODOS)
# ─────────────────────────────────────────────────────────────────────────────

class _Level1Generator:
    """
    Nivel 1: Operación o inferencia de un único paso.

    Subtypes:
      arithmetic:      eval() de expresiones con +, -, *, //
      linear_eq_1:     x + b = c  (coef = 1, solución entera)
      transitivity_2:  a > b y b > c → ¿a > c?
      divisibility_1:  ¿n es divisible por k? (respuesta True/False)
    """

    _SUBTYPES = ["arithmetic", "linear_eq_1", "transitivity_2", "divisibility_1"]

    def generate(self, rng: random.Random) -> MathExample:
        sub = rng.choice(self._SUBTYPES)
        if sub == "arithmetic":
            return self._arithmetic(rng)
        elif sub == "linear_eq_1":
            return self._linear_eq_1(rng)
        elif sub == "transitivity_2":
            return self._transitivity_2(rng)
        else:
            return self._divisibility_1(rng)

    # ── Aritmética simple ──────────────────────────────────────────────────

    def _arithmetic(self, rng) -> MathExample:
        """eval() de 3+4*2, etc."""
        ops   = ['+', '-', '*', '//']
        a, b, c = rng.randint(1, 15), rng.randint(1, 10), rng.randint(1, 10)
        op1   = rng.choice(ops)
        op2   = rng.choice(['+', '-', '*'])
        # Evitar divisiones por cero
        expr_str = f"{a} {op1} {b} {op2} {c}"
        try:
            result = int(eval(expr_str))
        except ZeroDivisionError:
            expr_str = f"{a} + {b} * {c}"
            result   = int(eval(expr_str))

        # Grafo: EXPRESSION --REDUCES_TO--> EQUALITY
        g = CausalGraph()
        expr_node = _math_node("expr",  f"{expr_str}",    MathNodeType.EXPRESSION)
        eq_node   = _math_node("eq",    f"= {result}",    MathNodeType.EQUALITY)
        g.add_node(expr_node).add_node(eq_node)
        g.add_edge(_math_edge("expr", "eq", MathRelation.REDUCES_TO))
        g.root_question = f"¿Cuánto es {expr_str}?"

        return MathExample(
            problem_text=f"Calcula: {expr_str}.",
            graph=g, answer=str(result),
            complexity_level=1, answer_type=MathAnswerType.ARITHMETIC,
            numeric_answer=float(result),
            metadata={"expr": expr_str, "expected": result},
        )

    # ── Ecuación lineal x + b = c ──────────────────────────────────────────

    def _linear_eq_1(self, rng) -> MathExample:
        """x + b = c → x = c - b (solución entera garantizada)."""
        b = rng.randint(-10, 10)
        x = rng.randint(-8, 8)
        c = x + b

        sign = '+' if b >= 0 else '-'
        abs_b = abs(b)
        eq_str  = f"x {sign} {abs_b} = {c}" if abs_b != 0 else f"x = {c}"
        sol_str = f"x = {x}"

        g = CausalGraph()
        axiom_node = _math_node("ax",  f"x {sign} {abs_b} = {c}", MathNodeType.AXIOM)
        expr_node  = _math_node("mid", f"x = {c} - ({b})",        MathNodeType.EXPRESSION)
        eq_node    = _math_node("sol", sol_str,                    MathNodeType.EQUALITY)
        g.add_node(axiom_node).add_node(expr_node).add_node(eq_node)
        g.add_edge(_math_edge("ax",  "mid", MathRelation.REDUCES_TO))
        g.add_edge(_math_edge("mid", "sol", MathRelation.EQUIVALENT_TO))
        g.root_question = f"Resuelve: {eq_str}."

        return MathExample(
            problem_text=f"Resuelve: {eq_str}.",
            graph=g, answer=sol_str,
            complexity_level=1, answer_type=MathAnswerType.LINEAR_EQUATION,
            numeric_answer=float(x),
            metadata={"equation": eq_str, "expected_x": x, "b": b, "c": c},
        )

    # ── Transitivity simple (2 desigualdades) ─────────────────────────────

    def _transitivity_2(self, rng) -> MathExample:
        """a > b, b > c → ¿a > c? (Siempre True — generamos solo positivos)."""
        names = ["a", "b", "c"]
        vals  = sorted(rng.sample(range(1, 21), 3), reverse=True)
        a, b, c = vals[0], vals[1], vals[2]

        g = CausalGraph()
        ax1  = _math_node("ax1", f"{names[0]} > {names[1]} ({a} > {b})", MathNodeType.AXIOM)
        ax2  = _math_node("ax2", f"{names[1]} > {names[2]} ({b} > {c})", MathNodeType.AXIOM)
        conc = _math_node("conc", f"{names[0]} > {names[2]}",             MathNodeType.THEOREM)
        g.add_node(ax1).add_node(ax2).add_node(conc)
        g.add_edge(_math_edge("ax1", "conc", MathRelation.IMPLIES))
        g.add_edge(_math_edge("ax2", "conc", MathRelation.IMPLIES))
        g.root_question = f"Si {names[0]} > {names[1]} y {names[1]} > {names[2]}, ¿es {names[0]} > {names[2]}?"

        answer_bool = a > c   # siempre True por construcción
        return MathExample(
            problem_text=(
                f"Sabemos que {names[0]} > {names[1]} ({a} > {b}) "
                f"y {names[1]} > {names[2]} ({b} > {c}). "
                f"¿Es {names[0]} > {names[2]}?"
            ),
            graph=g,
            answer=f"Sí, {names[0]} > {names[2]} ({a} > {c}) por transitividad.",
            complexity_level=1, answer_type=MathAnswerType.INEQUALITY,
            bool_answer=True,
            metadata={
                "a": a, "b": b, "c": c,
                "a_name": names[0], "c_name": names[2],
                "expected_result": True,
            },
        )

    # ── Divisibilidad directa ──────────────────────────────────────────────

    def _divisibility_1(self, rng) -> MathExample:
        """¿n es divisible por k?"""
        k       = rng.choice([2, 3, 4, 5, 6])
        divisible = rng.random() < 0.6
        if divisible:
            n = k * rng.randint(1, 12)
        else:
            n = k * rng.randint(1, 12) + rng.randint(1, k - 1)

        result     = (n % k == 0)
        answer_str = f"{'Sí' if result else 'No'}, {n} {'es' if result else 'no es'} divisible por {k}."
        if result:
            answer_str += f" ({n} = {k} × {n // k})"

        g = CausalGraph()
        hyp  = _math_node("hyp",  f"n = {n}",            MathNodeType.AXIOM)
        div  = _math_node("div",  f"{n} mod {k}",         MathNodeType.EXPRESSION)
        conc = _math_node("conc", f"divisible={result}",  MathNodeType.EQUALITY)
        g.add_node(hyp).add_node(div).add_node(conc)
        g.add_edge(_math_edge("hyp", "div",  MathRelation.REDUCES_TO))
        g.add_edge(_math_edge("div", "conc", MathRelation.EQUIVALENT_TO))
        g.root_question = f"¿{n} es divisible por {k}?"

        return MathExample(
            problem_text=f"¿{n} es divisible por {k}?",
            graph=g, answer=answer_str,
            complexity_level=1, answer_type=MathAnswerType.DIVISIBILITY,
            bool_answer=result,
            metadata={"n": n, "k": k, "expected_divisible": result,
                      "remainder": n % k},
        )


# ─────────────────────────────────────────────────────────────────────────────
# GENERADOR NIVEL 2 — DOS PASOS (3-4 NODOS)
# ─────────────────────────────────────────────────────────────────────────────

class _Level2Generator:
    """
    Nivel 2: Ecuaciones ax+b=c, cadenas de 3 desigualdades, divisibilidad compuesta.

    Subtypes:
      linear_eq_2:     ax + b = c  (a ≠ 1, solución racional)
      transitivity_3:  a > b > c > d → ¿a > d? (3 saltos)
      divisibility_2:  ¿n divisible por k? con n = producto o suma
      arithmetic_2op:  expr con paréntesis (a + b) * c
    """

    _SUBTYPES = ["linear_eq_2", "transitivity_3", "divisibility_2", "arithmetic_2op"]

    def generate(self, rng: random.Random) -> MathExample:
        sub = rng.choice(self._SUBTYPES)
        if sub == "linear_eq_2":
            return self._linear_eq_2(rng)
        elif sub == "transitivity_3":
            return self._transitivity_3(rng)
        elif sub == "divisibility_2":
            return self._divisibility_2(rng)
        else:
            return self._arithmetic_2op(rng)

    def _linear_eq_2(self, rng) -> MathExample:
        """ax + b = c. Garantizamos solución racional con denominador ≤ 8."""
        a = rng.choice([2, 3, 4, 5])
        x = rng.randint(-5, 5)          # solución entera
        b = rng.randint(-10, 10)
        c = a * x + b

        sign   = '+' if b >= 0 else '-'
        abs_b  = abs(b)
        eq_str = f"{a}x {sign} {abs_b} = {c}"

        g = CausalGraph()
        ax1  = _math_node("ax",  eq_str,                   MathNodeType.AXIOM)
        mid  = _math_node("mid", f"{a}x = {c - b}",        MathNodeType.EXPRESSION)
        sol  = _math_node("sol", f"x = {x}",               MathNodeType.EQUALITY)
        g.add_node(ax1).add_node(mid).add_node(sol)
        g.add_edge(_math_edge("ax",  "mid", MathRelation.REDUCES_TO))
        g.add_edge(_math_edge("mid", "sol", MathRelation.REDUCES_TO))
        g.root_question = f"Resuelve {eq_str}."

        # Verificar con eval
        check_val = (c - b) / a
        return MathExample(
            problem_text=f"Resuelve: {eq_str}.",
            graph=g, answer=f"x = {x}",
            complexity_level=2, answer_type=MathAnswerType.LINEAR_EQUATION,
            numeric_answer=float(x),
            metadata={"a": a, "b": b, "c": c, "expected_x": x,
                      "verify_expr": f"({c} - ({b})) / {a}"},
        )

    def _transitivity_3(self, rng) -> MathExample:
        """a > b > c > d. ¿a > d?"""
        names = ["a", "b", "c", "d"]
        vals  = sorted(rng.sample(range(1, 31), 4), reverse=True)
        a, b, c, d = vals[0], vals[1], vals[2], vals[3]

        g = CausalGraph()
        ax1  = _math_node("ax1", f"a > b ({a}>{b})", MathNodeType.AXIOM)
        ax2  = _math_node("ax2", f"b > c ({b}>{c})", MathNodeType.AXIOM)
        ax3  = _math_node("ax3", f"c > d ({c}>{d})", MathNodeType.AXIOM)
        lem1 = _math_node("lem", f"a > c",            MathNodeType.LEMMA)
        thm  = _math_node("thm", f"a > d",            MathNodeType.THEOREM)
        g.add_node(ax1).add_node(ax2).add_node(ax3).add_node(lem1).add_node(thm)
        g.add_edge(_math_edge("ax1", "lem", MathRelation.IMPLIES))
        g.add_edge(_math_edge("ax2", "lem", MathRelation.IMPLIES))
        g.add_edge(_math_edge("lem", "thm", MathRelation.IMPLIES))
        g.add_edge(_math_edge("ax3", "thm", MathRelation.IMPLIES))
        g.root_question = f"Si a>b>c>d, ¿a>d?"

        return MathExample(
            problem_text=(
                f"Dado que a > b ({a} > {b}), b > c ({b} > {c}), c > d ({c} > {d}). "
                f"Demuestra que a > d."
            ),
            graph=g,
            answer=f"Sí. Por transitividad: a > b > c > d, luego a > d ({a} > {d}).",
            complexity_level=2, answer_type=MathAnswerType.INEQUALITY,
            bool_answer=True,
            metadata={"a": a, "b": b, "c": c, "d": d,
                      "expected_result": True,
                      "chain": [a, b, c, d]},
        )

    def _divisibility_2(self, rng) -> MathExample:
        """¿(a × b) es divisible por c?"""
        c = rng.choice([2, 3, 4, 6])
        divisible = rng.random() < 0.6
        if divisible:
            # Garantizar a×b divisible por c
            a = rng.randint(1, 8)
            b = c * rng.randint(1, 4)
        else:
            a = rng.choice([p for p in range(1, 10) if p % c != 0 and p > 1])
            b = rng.choice([p for p in range(1, 10) if p % c != 0 and p > 1])
        n = a * b

        result     = (n % c == 0)
        answer_str = (
            f"{'Sí' if result else 'No'}, {a}×{b} = {n}, "
            f"que {'es' if result else 'no es'} divisible por {c}."
        )

        g = CausalGraph()
        ax1  = _math_node("ax1", f"a = {a}",     MathNodeType.AXIOM)
        ax2  = _math_node("ax2", f"b = {b}",     MathNodeType.AXIOM)
        prod = _math_node("prod", f"a×b = {n}",  MathNodeType.EXPRESSION)
        conc = _math_node("conc", f"div by {c} = {result}", MathNodeType.EQUALITY)
        g.add_node(ax1).add_node(ax2).add_node(prod).add_node(conc)
        g.add_edge(_math_edge("ax1", "prod", MathRelation.APPLIES))
        g.add_edge(_math_edge("ax2", "prod", MathRelation.APPLIES))
        g.add_edge(_math_edge("prod", "conc", MathRelation.REDUCES_TO))
        g.root_question = f"¿{a}×{b} es divisible por {c}?"

        return MathExample(
            problem_text=f"¿{a} × {b} es divisible por {c}?",
            graph=g, answer=answer_str,
            complexity_level=2, answer_type=MathAnswerType.DIVISIBILITY,
            bool_answer=result,
            numeric_answer=float(n),
            metadata={"a": a, "b": b, "n": n, "c": c,
                      "expected_divisible": result,
                      "verify_expr": f"({a} * {b}) % {c} == 0"},
        )

    def _arithmetic_2op(self, rng) -> MathExample:
        """Expresión con paréntesis: (a + b) * c - d."""
        a = rng.randint(1, 10)
        b = rng.randint(1, 10)
        c = rng.randint(1, 8)
        d = rng.randint(0, 10)
        expr_str = f"({a} + {b}) * {c} - {d}"
        result   = (a + b) * c - d

        g = CausalGraph()
        inner = _math_node("inner", f"{a} + {b} = {a+b}", MathNodeType.EXPRESSION)
        outer = _math_node("outer", f"{a+b} * {c} = {(a+b)*c}", MathNodeType.EXPRESSION)
        final = _math_node("final", f"= {result}", MathNodeType.EQUALITY)
        g.add_node(inner).add_node(outer).add_node(final)
        g.add_edge(_math_edge("inner", "outer", MathRelation.REDUCES_TO))
        g.add_edge(_math_edge("outer", "final", MathRelation.REDUCES_TO))
        g.root_question = f"Calcula {expr_str}."

        return MathExample(
            problem_text=f"Calcula: {expr_str}.",
            graph=g, answer=str(result),
            complexity_level=2, answer_type=MathAnswerType.ARITHMETIC,
            numeric_answer=float(result),
            metadata={"expr": expr_str, "expected": result,
                      "verify_expr": expr_str},
        )


# ─────────────────────────────────────────────────────────────────────────────
# GENERADOR NIVEL 3 — PRUEBAS FORMALES (4-6 NODOS)
# ─────────────────────────────────────────────────────────────────────────────

class _Level3Generator:
    """
    Nivel 3: Pruebas algebraicas formales con 4-6 pasos.

    Subtypes:
      parity_sum:      la suma de dos pares es par
      parity_product:  el producto de dos pares es par
      algebraic_id:    (a+b)² = a² + 2ab + b² (verifica con valores concretos)
      bound_sqrt:      si x² = k y x > 0, entonces x = √k
      linear_system:   sistema 2×2: ax+by=c, dx+ey=f → solución única
    """

    _SUBTYPES = ["parity_sum", "parity_product", "algebraic_id",
                 "bound_sqrt", "linear_system"]

    def generate(self, rng: random.Random) -> MathExample:
        sub = rng.choice(self._SUBTYPES)
        if sub == "parity_sum":
            return self._parity_sum(rng)
        elif sub == "parity_product":
            return self._parity_product(rng)
        elif sub == "algebraic_id":
            return self._algebraic_id(rng)
        elif sub == "bound_sqrt":
            return self._bound_sqrt(rng)
        else:
            return self._linear_system(rng)

    def _parity_sum(self, rng) -> MathExample:
        """
        Demuestra que la suma de dos números pares es par.
        Instancia concreta: a=2m, b=2n → a+b=2(m+n).
        """
        m = rng.randint(1, 10)
        n = rng.randint(1, 10)
        a = 2 * m
        b = 2 * n
        s = a + b

        g = CausalGraph()
        def1 = _math_node("d1",  f"a = 2m = {a} (par)",       MathNodeType.DEFINITION)
        def2 = _math_node("d2",  f"b = 2n = {b} (par)",       MathNodeType.DEFINITION)
        step = _math_node("s1",  f"a+b = 2m+2n = 2(m+n)",     MathNodeType.LEMMA)
        conc = _math_node("thm", f"a+b = {s} es par",         MathNodeType.THEOREM)
        check= _math_node("chk", f"{s} mod 2 = 0",            MathNodeType.EQUALITY)
        g.add_node(def1).add_node(def2).add_node(step).add_node(conc).add_node(check)
        g.add_edge(_math_edge("d1",  "s1",  MathRelation.DERIVES))
        g.add_edge(_math_edge("d2",  "s1",  MathRelation.DERIVES))
        g.add_edge(_math_edge("s1",  "thm", MathRelation.IMPLIES))
        g.add_edge(_math_edge("thm", "chk", MathRelation.EQUIVALENT_TO))
        g.root_question = "Demuestra que la suma de dos pares es par."

        return MathExample(
            problem_text=(
                f"Sea a = {a} (par) y b = {b} (par). "
                f"Demuestra que a + b es par."
            ),
            graph=g,
            answer=(
                f"Como a = 2×{m} y b = 2×{n}, "
                f"a + b = 2×{m} + 2×{n} = 2×({m}+{n}) = {s}. "
                f"Como {s} = 2×{m+n}, es par."
            ),
            complexity_level=3, answer_type=MathAnswerType.PARITY_PROOF,
            bool_answer=True,
            numeric_answer=float(s),
            metadata={
                "a": a, "b": b, "m": m, "n": n, "sum": s,
                "expected_parity": "even",
                "expected_result": True,
                "verify_expr": f"({a} + {b}) % 2 == 0",
            },
        )

    def _parity_product(self, rng) -> MathExample:
        """El producto de dos pares es par (y divisible por 4)."""
        m = rng.randint(1, 8)
        n = rng.randint(1, 8)
        a = 2 * m
        b = 2 * n
        p = a * b

        g = CausalGraph()
        def1 = _math_node("d1",  f"a = 2m = {a}",      MathNodeType.DEFINITION)
        def2 = _math_node("d2",  f"b = 2n = {b}",      MathNodeType.DEFINITION)
        step = _math_node("s1",  f"a×b = 4mn",         MathNodeType.LEMMA)
        conc = _math_node("thm", f"a×b = {p} es par",  MathNodeType.THEOREM)
        div4 = _math_node("d4",  f"{p} div by 4",      MathNodeType.EQUALITY)
        g.add_node(def1).add_node(def2).add_node(step).add_node(conc).add_node(div4)
        g.add_edge(_math_edge("d1",  "s1",  MathRelation.DERIVES))
        g.add_edge(_math_edge("d2",  "s1",  MathRelation.DERIVES))
        g.add_edge(_math_edge("s1",  "thm", MathRelation.IMPLIES))
        g.add_edge(_math_edge("thm", "d4",  MathRelation.EQUIVALENT_TO))
        g.root_question = "¿El producto de dos pares es divisible por 4?"

        return MathExample(
            problem_text=(
                f"Sea a = {a} (par) y b = {b} (par). "
                f"Demuestra que a × b es divisible por 4."
            ),
            graph=g,
            answer=(
                f"a×b = 2{m} × 2{n} = 4×{m*n} = {p}. "
                f"Como {p} = 4×{m*n}, es divisible por 4."
            ),
            complexity_level=3, answer_type=MathAnswerType.PARITY_PROOF,
            bool_answer=True,
            numeric_answer=float(p),
            metadata={
                "a": a, "b": b, "m": m, "n": n, "product": p,
                "expected_result": True,
                "verify_expr": f"({a} * {b}) % 4 == 0",
            },
        )

    def _algebraic_id(self, rng) -> MathExample:
        """
        Verifica (a+b)² = a² + 2ab + b² con valores concretos.
        Grafo: AXIOM(identity) → SPECIALIZES → EXPRESSION(LHS) → EQUIVALENT_TO → EXPRESSION(RHS).
        """
        a = rng.randint(1, 8)
        b = rng.randint(1, 8)
        lhs = (a + b) ** 2
        rhs = a**2 + 2*a*b + b**2

        g = CausalGraph()
        ax   = _math_node("ax",  "(a+b)² = a²+2ab+b² (identidad)", MathNodeType.AXIOM)
        spec = _math_node("sp",  f"a={a}, b={b}",                   MathNodeType.EXPRESSION)
        lhs_ = _math_node("lhs", f"({a}+{b})² = {(a+b)}² = {lhs}", MathNodeType.EXPRESSION)
        rhs_ = _math_node("rhs", f"{a}²+2·{a}·{b}+{b}² = {rhs}",  MathNodeType.EXPRESSION)
        eq   = _math_node("eq",  f"{lhs} = {rhs}",                 MathNodeType.EQUALITY)
        g.add_node(ax).add_node(spec).add_node(lhs_).add_node(rhs_).add_node(eq)
        g.add_edge(_math_edge("ax",  "sp",  MathRelation.GENERALIZES))
        g.add_edge(_math_edge("sp",  "lhs", MathRelation.SPECIALIZES))
        g.add_edge(_math_edge("sp",  "rhs", MathRelation.SPECIALIZES))
        g.add_edge(_math_edge("lhs", "eq",  MathRelation.EQUIVALENT_TO))
        g.add_edge(_math_edge("rhs", "eq",  MathRelation.EQUIVALENT_TO))
        g.root_question = f"Verifica (a+b)² = a²+2ab+b² para a={a}, b={b}."

        return MathExample(
            problem_text=(
                f"Verifica la identidad (a+b)² = a²+2ab+b² para a={a}, b={b}."
            ),
            graph=g,
            answer=(
                f"LHS: ({a}+{b})² = {a+b}² = {lhs}. "
                f"RHS: {a}²+2·{a}·{b}+{b}² = {a**2}+{2*a*b}+{b**2} = {rhs}. "
                f"LHS = RHS = {lhs}. ✓"
            ),
            complexity_level=3, answer_type=MathAnswerType.ALGEBRAIC_ID,
            bool_answer=True,
            numeric_answer=float(lhs),
            metadata={
                "a": a, "b": b, "lhs": lhs, "rhs": rhs,
                "expected_equal": True,
                "verify_lhs": f"({a}+{b})**2",
                "verify_rhs": f"{a}**2 + 2*{a}*{b} + {b}**2",
            },
        )

    def _bound_sqrt(self, rng) -> MathExample:
        """Si x² = k (k = perfecto) y x > 0, entonces x = √k."""
        r = rng.randint(1, 9)
        k = r * r

        g = CausalGraph()
        ax1  = _math_node("ax1", f"x² = {k}",          MathNodeType.AXIOM)
        ax2  = _math_node("ax2", f"x > 0",              MathNodeType.AXIOM)
        bnd  = _math_node("bnd", f"x = ±√{k} = ±{r}", MathNodeType.LEMMA)
        conc = _math_node("thm", f"x = {r}",           MathNodeType.THEOREM)
        eq   = _math_node("eq",  f"{r}² = {k} ✓",     MathNodeType.EQUALITY)
        g.add_node(ax1).add_node(ax2).add_node(bnd).add_node(conc).add_node(eq)
        g.add_edge(_math_edge("ax1", "bnd", MathRelation.DERIVES))
        g.add_edge(_math_edge("bnd", "thm", MathRelation.BOUNDS))
        g.add_edge(_math_edge("ax2", "thm", MathRelation.ASSUMES))
        g.add_edge(_math_edge("thm", "eq",  MathRelation.EQUIVALENT_TO))
        g.root_question = f"Si x²={k} y x>0, ¿cuánto es x?"

        return MathExample(
            problem_text=f"Si x² = {k} y x > 0, ¿cuánto es x?",
            graph=g,
            answer=f"x = {r} (ya que {r}² = {k} y {r} > 0).",
            complexity_level=3, answer_type=MathAnswerType.BOUND,
            numeric_answer=float(r),
            metadata={"k": k, "r": r, "expected_x": r,
                      "verify_expr": f"int({k}**0.5)"},
        )

    def _linear_system(self, rng) -> MathExample:
        """
        Sistema 2×2 con solución entera única.
        x + y = s, x - y = d → x = (s+d)/2, y = (s-d)/2.
        Garantizamos s y d de la misma paridad para solución entera.
        """
        x = rng.randint(-5, 5)
        y = rng.randint(-5, 5)
        s = x + y    # x + y = s
        d = x - y    # x - y = d

        sign_s = '+' if y >= 0 else '-'
        abs_y  = abs(y)

        g = CausalGraph()
        eq1  = _math_node("eq1", f"x + y = {s}",                   MathNodeType.AXIOM)
        eq2  = _math_node("eq2", f"x - y = {d}",                   MathNodeType.AXIOM)
        sum_ = _math_node("sum", f"2x = {s+d} → x = {x}",         MathNodeType.LEMMA)
        sub_ = _math_node("sub", f"2y = {s-d} → y = {y}",         MathNodeType.LEMMA)
        sol  = _math_node("sol", f"x={x}, y={y}",                  MathNodeType.THEOREM)
        chk  = _math_node("chk", f"✓ {x}+{y}={s}, {x}-{y}={d}",  MathNodeType.EQUALITY)
        g.add_node(eq1).add_node(eq2).add_node(sum_).add_node(sub_).add_node(sol).add_node(chk)
        g.add_edge(_math_edge("eq1", "sum", MathRelation.DERIVES))
        g.add_edge(_math_edge("eq2", "sum", MathRelation.DERIVES))
        g.add_edge(_math_edge("eq1", "sub", MathRelation.DERIVES))
        g.add_edge(_math_edge("eq2", "sub", MathRelation.DERIVES))
        g.add_edge(_math_edge("sum", "sol", MathRelation.IMPLIES))
        g.add_edge(_math_edge("sub", "sol", MathRelation.IMPLIES))
        g.add_edge(_math_edge("sol", "chk", MathRelation.EQUIVALENT_TO))
        g.root_question = f"Resuelve: x+y={s}, x-y={d}."

        return MathExample(
            problem_text=f"Resuelve el sistema: x + y = {s}  y  x - y = {d}.",
            graph=g,
            answer=f"x = {x}, y = {y}.",
            complexity_level=3, answer_type=MathAnswerType.LINEAR_EQUATION,
            numeric_answer=float(x),
            metadata={
                "s": s, "d": d, "expected_x": x, "expected_y": y,
                "verify_x": f"({s} + {d}) / 2",
                "verify_y": f"({s} - {d}) / 2",
            },
        )


# ─────────────────────────────────────────────────────────────────────────────
# VERIFICACIÓN
# ─────────────────────────────────────────────────────────────────────────────

def verify_math_example(ex: MathExample) -> MathVerificationResult:
    """
    Verifica que el grafo y la respuesta del ejemplo son matemáticamente correctos.

    Estrategia por tipo:
      ARITHMETIC:      eval(metadata["expr"]) == numeric_answer
      LINEAR_EQUATION: solve_linear(metadata) == numeric_answer
      INEQUALITY:      all values in chain are strictly ordered
      DIVISIBILITY:    n % k == 0 ↔ bool_answer
      PARITY_PROOF:    eval(verify_expr) == True
      ALGEBRAIC_ID:    eval(verify_lhs) == eval(verify_rhs)
      BOUND:           eval(verify_expr) == numeric_answer
    """
    meta  = ex.metadata
    atype = ex.answer_type

    try:
        # ── Verificación del grafo ─────────────────────────────────────────
        if len(ex.graph) < 2:
            return MathVerificationResult(
                False, f"Graph has only {len(ex.graph)} nodes (expected >= 2)"
            )
        if len(ex.graph.edges) < 1:
            return MathVerificationResult(
                False, "Graph has no edges"
            )

        # ── Verificación matemática por tipo ──────────────────────────────
        if atype == MathAnswerType.ARITHMETIC:
            expr   = meta.get("expr") or meta.get("verify_expr", "")
            expected = meta.get("expected")
            if expr and expected is not None:
                computed = int(eval(expr))  # seguro: solo enteros y ops aritméticas
                if computed != int(expected):
                    return MathVerificationResult(
                        False,
                        f"eval({expr!r}) = {computed} ≠ {expected}",
                        {"computed": computed, "expected": expected},
                    )

        elif atype == MathAnswerType.LINEAR_EQUATION:
            expected_x = meta.get("expected_x")
            if expected_x is not None:
                # Para sistema 2x2
                if "verify_x" in meta:
                    computed = eval(meta["verify_x"])
                    if abs(computed - expected_x) > 1e-9:
                        return MathVerificationResult(
                            False,
                            f"x = {computed} ≠ {expected_x}",
                            {"computed_x": computed, "expected_x": expected_x},
                        )
                # Para ecuación lineal simple
                elif "a" in meta and "b" in meta and "c" in meta:
                    a, b, c = meta["a"], meta["b"], meta["c"]
                    computed = (c - b) / a
                    if abs(computed - expected_x) > 1e-9:
                        return MathVerificationResult(
                            False,
                            f"({c}-{b})/{a} = {computed} ≠ {expected_x}",
                            {"computed": computed, "expected": expected_x},
                        )
                # Verificar numérico directo
                if ex.numeric_answer is not None and expected_x is not None:
                    if abs(ex.numeric_answer - expected_x) > 1e-9:
                        return MathVerificationResult(
                            False,
                            f"numeric_answer {ex.numeric_answer} ≠ expected_x {expected_x}",
                        )

        elif atype == MathAnswerType.INEQUALITY:
            chain = meta.get("chain")
            if chain:
                for i in range(len(chain) - 1):
                    if chain[i] <= chain[i + 1]:
                        return MathVerificationResult(
                            False,
                            f"Chain broken: {chain[i]} <= {chain[i+1]}",
                            {"chain": chain},
                        )
            elif "a" in meta and "c" in meta:
                a, c = meta["a"], meta["c"]
                expected = meta.get("expected_result", True)
                actual   = a > c
                if actual != expected:
                    return MathVerificationResult(
                        False, f"a > c: {a} > {c} = {actual} ≠ {expected}"
                    )

        elif atype == MathAnswerType.DIVISIBILITY:
            n   = meta.get("n", meta.get("a", 0))
            k   = meta.get("k", meta.get("c", 1))
            exp = meta.get("expected_divisible")
            if exp is not None:
                actual = (int(n) % int(k) == 0)
                if actual != exp:
                    return MathVerificationResult(
                        False,
                        f"{n} % {k} == 0 → {actual} ≠ {exp}",
                        {"n": n, "k": k, "remainder": int(n) % int(k)},
                    )
            # Verificar con verify_expr si existe
            if "verify_expr" in meta:
                result = eval(meta["verify_expr"])
                if result != ex.bool_answer:
                    return MathVerificationResult(
                        False,
                        f"eval({meta['verify_expr']!r}) = {result} ≠ {ex.bool_answer}",
                    )

        elif atype == MathAnswerType.PARITY_PROOF:
            ve = meta.get("verify_expr", "")
            if ve:
                result = eval(ve)
                if result != ex.bool_answer:
                    return MathVerificationResult(
                        False,
                        f"eval({ve!r}) = {result} ≠ {ex.bool_answer}",
                    )
            # Verify numeric if present
            if ex.numeric_answer is not None:
                n = ex.numeric_answer
                if int(n) % 2 != 0:
                    return MathVerificationResult(
                        False,
                        f"Parity proof result {int(n)} is odd (expected even)",
                    )

        elif atype == MathAnswerType.ALGEBRAIC_ID:
            lhs_expr = meta.get("verify_lhs", "")
            rhs_expr = meta.get("verify_rhs", "")
            if lhs_expr and rhs_expr:
                lhs_val = eval(lhs_expr)
                rhs_val = eval(rhs_expr)
                if abs(lhs_val - rhs_val) > 1e-9:
                    return MathVerificationResult(
                        False,
                        f"LHS {lhs_val} ≠ RHS {rhs_val}",
                        {"lhs": lhs_val, "rhs": rhs_val},
                    )
            if ex.numeric_answer is not None and lhs_expr:
                computed = eval(lhs_expr)
                if abs(computed - ex.numeric_answer) > 1e-9:
                    return MathVerificationResult(
                        False,
                        f"LHS {computed} ≠ numeric_answer {ex.numeric_answer}",
                    )

        elif atype == MathAnswerType.BOUND:
            ve = meta.get("verify_expr", "")
            if ve and ex.numeric_answer is not None:
                computed = eval(ve)
                if abs(float(computed) - ex.numeric_answer) > 1e-9:
                    return MathVerificationResult(
                        False,
                        f"eval({ve!r}) = {computed} ≠ {ex.numeric_answer}",
                    )

        return MathVerificationResult(True, "All mathematical constraints satisfied", {})

    except Exception as exc:
        return MathVerificationResult(
            False, f"Verification error: {exc}", {"exception": str(exc)}
        )


# ─────────────────────────────────────────────────────────────────────────────
# GENERADOR PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

class MathGraphGenerator:
    """
    Generador de grafos de prueba matemática para AXIOM.

    Misma interfaz que CausalGraphGenerator y CodeGraphGenerator:
        gen = MathGraphGenerator(seed=42)
        ex  = gen.generate(level=2)
        assert verify_math_example(ex).passed

        batch = gen.generate_batch(n=200, level_distribution={1: 0.4, 2: 0.4, 3: 0.2})
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng  = random.Random(seed)
        self._gens = {
            1: _Level1Generator(),
            2: _Level2Generator(),
            3: _Level3Generator(),
        }

    def generate(self, level: int = 1) -> MathExample:
        if level not in self._gens:
            raise ValueError(f"level must be 1-3, got {level}")
        return self._gens[level].generate(self._rng)

    def generate_batch(
        self,
        n: int = 100,
        level_distribution: Optional[Dict[int, float]] = None,
    ) -> List[MathExample]:
        if level_distribution is None:
            level_distribution = {1: 0.4, 2: 0.4, 3: 0.2}
        levels  = list(level_distribution.keys())
        weights = list(level_distribution.values())
        return [
            self.generate(self._rng.choices(levels, weights=weights, k=1)[0])
            for _ in range(n)
        ]
