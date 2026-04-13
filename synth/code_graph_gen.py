"""
synth/code_graph_gen.py — Generador de Grafos de Código para FORGE-C
======================================================================

Motor de datos sintéticos para el motor FORGE-C (razonamiento sobre código).
Genera problemas tipo:
  - "¿Qué función se ve afectada si cambio X?" (análisis de impacto)
  - "¿De dónde viene este dato?" (trazabilidad de datos)
  - "¿Por qué falla este test?" (diagnóstico de errores)
  - "¿Qué módulo importa B?" (dependencias)

Tres niveles de complejidad (equivalentes a niveles 1-3 de causal_graph_gen.py):

  Nivel 1 — Cadenas de llamada (2-3 nodos), preguntas directas
             "¿Qué función llama a B?" / "¿Qué lee la función A?"

  Nivel 2 — Fan-out / fan-in (3-5 nodos), análisis de impacto
             "¿Qué funciones se verían afectadas si cambio X?"
             "¿De qué depende el módulo A?"

  Nivel 3 — Flujo de datos multi-salto (4-7 nodos), trazabilidad
             "¿De dónde viene el dato que usa C?" (recorre DATA_FLOWS_TO)
             "¿Qué falla si Y lanza una excepción?" (THROWS propagado)

Contrato de cada CodeExample:
  - problem_text:     descripción del problema en lenguaje natural
  - graph:            CausalGraph con CodeNode/CodeEdge
  - answer:           respuesta correcta verificable
  - complexity_level: 1-3
  - answer_type:      CodeAnswerType
  - metadata:         parámetros para verify_code_example()
  - example_id:       UUID reproducible

Uso básico:
    gen = CodeGraphGenerator()
    ex  = gen.generate(level=1)
    res = verify_code_example(ex)
    assert res.passed

    batch = gen.generate_batch(n=100, level_distribution={1: 0.4, 2: 0.4, 3: 0.2})
"""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.graph import CausalGraph
from motors.forge_c.relations import (
    CODE_RELATIONS,
    CodeEdge,
    CodeNode,
    CodeNodeType,
    CodeRelation,
)


# ─────────────────────────────────────────────────────────────────────────────
# TIPOS DE RESPUESTA
# ─────────────────────────────────────────────────────────────────────────────

class CodeAnswerType(str, Enum):
    DIRECT_CALLER    = "direct_caller"    # ¿Qué llama directamente a X?
    DIRECT_CALLEE    = "direct_callee"    # ¿A qué llama directamente X?
    IMPACT_ANALYSIS  = "impact_analysis"  # ¿Qué se ve afectado si cambio X?
    DATA_SOURCE      = "data_source"      # ¿De dónde viene el dato que usa X?
    DEPENDENCY       = "dependency"       # ¿De qué depende X?
    ERROR_PROPAGATION= "error_propagation"# ¿Qué falla si Y lanza excepción?
    READS_WRITES     = "reads_writes"     # ¿Qué variables lee/escribe X?
    INHERITANCE      = "inheritance"      # ¿Qué hereda X? / ¿Quién hereda de X?


# ─────────────────────────────────────────────────────────────────────────────
# RESULTADO DE VERIFICACIÓN
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CodeVerificationResult:
    """
    Resultado de verify_code_example().

    passed:  True si el grafo y la respuesta son lógicamente consistentes
    reason:  Explicación en lenguaje natural del resultado
    details: Datos cuantitativos para debugging
    """
    passed: bool
    reason: str
    details: Dict = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.passed

    def __repr__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"CodeVerificationResult({status}: {self.reason})"


# ─────────────────────────────────────────────────────────────────────────────
# CÓDIGO EXAMPLE — UNIDAD ATÓMICA DE ENTRENAMIENTO
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CodeExample:
    """
    Unidad atómica de entrenamiento para FORGE-C.

    Misma filosofía que CausalExample: grafo + pregunta + respuesta verificable.
    El grafo usa CodeNode (con CodeNodeType) y CodeEdge (con CodeRelation).

    metadata siempre incluye las claves necesarias para verify_code_example():
      - answer_type-specific: expected_callers, expected_callees, etc.
    """
    problem_text:     str
    graph:            CausalGraph
    answer:           str
    complexity_level: int
    answer_type:      CodeAnswerType
    verifiable:       bool = True
    metadata:         Dict = field(default_factory=dict)
    example_id:       str  = field(default_factory=lambda: str(uuid.uuid4())[:12])

    def __repr__(self) -> str:
        return (
            f"CodeExample(level={self.complexity_level}, "
            f"type={self.answer_type.value}, "
            f"nodes={len(self.graph)}, edges={len(self.graph.edges)}, "
            f"id={self.example_id})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# POOLS DE CÓDIGO — CONTENIDO SEMÁNTICO
# ─────────────────────────────────────────────────────────────────────────────

# (node_id, label, CodeNodeType)
_CodeDesc = Tuple[str, str, CodeNodeType]

# Módulos de una aplicación web típica
WEB_APP_NODES: Dict[str, List[_CodeDesc]] = {
    "api": [
        ("route_handler",  "route_handler",  CodeNodeType.FUNCTION),
        ("auth_middleware","auth_middleware", CodeNodeType.FUNCTION),
        ("validator",      "validate_input", CodeNodeType.FUNCTION),
        ("serializer",     "serialize_response", CodeNodeType.FUNCTION),
        ("error_handler",  "handle_error",   CodeNodeType.FUNCTION),
    ],
    "db": [
        ("db_connect",     "db_connect",     CodeNodeType.FUNCTION),
        ("query_builder",  "build_query",    CodeNodeType.FUNCTION),
        ("orm_model",      "UserModel",      CodeNodeType.CLASS),
        ("connection_pool","connection_pool",CodeNodeType.VARIABLE),
        ("db_error",       "DatabaseError",  CodeNodeType.ERROR),
    ],
    "service": [
        ("user_service",   "UserService",    CodeNodeType.CLASS),
        ("auth_service",   "AuthService",    CodeNodeType.CLASS),
        ("email_service",  "EmailService",   CodeNodeType.CLASS),
        ("cache_service",  "CacheService",   CodeNodeType.CLASS),
        ("notification_fn","send_notification", CodeNodeType.FUNCTION),
    ],
    "data": [
        ("user_record",    "user_record",    CodeNodeType.VARIABLE),
        ("auth_token",     "auth_token",     CodeNodeType.VARIABLE),
        ("request_body",   "request_body",   CodeNodeType.VARIABLE),
        ("response_data",  "response_data",  CodeNodeType.VARIABLE),
        ("config_settings","config_settings",CodeNodeType.CONFIG),
    ],
    "test": [
        ("test_auth",      "test_auth",      CodeNodeType.TEST),
        ("test_user_crud", "test_user_crud", CodeNodeType.TEST),
        ("test_email",     "test_email",     CodeNodeType.TEST),
        ("mock_db",        "MockDatabase",   CodeNodeType.CLASS),
        ("assert_response","assert_response",CodeNodeType.FUNCTION),
    ],
    "module": [
        ("routes_module",  "routes",         CodeNodeType.MODULE),
        ("models_module",  "models",         CodeNodeType.MODULE),
        ("services_module","services",       CodeNodeType.MODULE),
        ("utils_module",   "utils",          CodeNodeType.MODULE),
        ("config_module",  "config",         CodeNodeType.MODULE),
    ],
}

# Cadenas de relaciones predefinidas por dominio (src_idx, tgt_idx, CodeRelation)
WEB_APP_CHAINS: Dict[str, List[Tuple[int, int, CodeRelation]]] = {
    "api": [
        (0, 1, CodeRelation.CALLS),
        (1, 2, CodeRelation.CALLS),
        (0, 3, CodeRelation.CALLS),
        (0, 4, CodeRelation.THROWS),
    ],
    "db": [
        (0, 3, CodeRelation.READS),
        (1, 2, CodeRelation.CALLS),
        (1, 4, CodeRelation.THROWS),
        (2, 3, CodeRelation.READS),
    ],
    "service": [
        (0, 2, CodeRelation.CALLS),
        (0, 3, CodeRelation.CALLS),
        (1, 0, CodeRelation.CALLS),
        (0, 4, CodeRelation.CALLS),
    ],
    "data": [
        (2, 0, CodeRelation.DATA_FLOWS_TO),
        (0, 1, CodeRelation.DATA_FLOWS_TO),
        (1, 3, CodeRelation.DATA_FLOWS_TO),
        (4, 0, CodeRelation.DATA_FLOWS_TO),
    ],
    "test": [
        (0, 3, CodeRelation.CALLS),
        (1, 3, CodeRelation.CALLS),
        (0, 4, CodeRelation.CALLS),
        (2, 3, CodeRelation.CALLS),
    ],
    "module": [
        (0, 1, CodeRelation.IMPORTS),
        (2, 1, CodeRelation.IMPORTS),
        (0, 2, CodeRelation.IMPORTS),
        (0, 3, CodeRelation.IMPORTS),
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS INTERNOS
# ─────────────────────────────────────────────────────────────────────────────

def _code_node(desc: _CodeDesc, prefix: str = "") -> CodeNode:
    nid, label, ntype = desc
    full_id = f"{prefix}{nid}" if prefix else nid
    return CodeNode(
        node_id=full_id,
        label=label,
        node_type=ntype,
        confidence=1.0,
        grounded=True,
    )


def _build_code_chain(
    domain: str,
    node_indices: List[int],
    rng: random.Random,
    prefix: str = "",
) -> Tuple[CausalGraph, List[CodeNode]]:
    """Construye un CausalGraph de código desde un subconjunto de nodos del dominio."""
    nodes_desc = WEB_APP_NODES[domain]
    chain_edges = WEB_APP_CHAINS[domain]

    g = CausalGraph(root_question="")
    nodes: List[CodeNode] = []

    for idx in node_indices:
        n = _code_node(nodes_desc[idx], prefix=prefix)
        g.add_node(n)
        nodes.append(n)

    for src_idx, tgt_idx, rel in chain_edges:
        if src_idx in node_indices and tgt_idx in node_indices:
            src_desc = nodes_desc[src_idx]
            tgt_desc = nodes_desc[tgt_idx]
            src_id = f"{prefix}{src_desc[0]}" if prefix else src_desc[0]
            tgt_id = f"{prefix}{tgt_desc[0]}" if prefix else tgt_desc[0]
            if src_id in g and tgt_id in g:
                e = CodeEdge(
                    source_id=src_id,
                    target_id=tgt_id,
                    relation=rel,
                    strength=round(rng.uniform(0.8, 1.0), 2),
                    confidence=round(rng.uniform(0.85, 1.0), 2),
                )
                g.add_edge(e)

    return g, nodes


def _rel_text(rel: CodeRelation) -> str:
    """Texto legible para una CodeRelation."""
    return {
        CodeRelation.CALLS:         "llama a",
        CodeRelation.IMPORTS:       "importa",
        CodeRelation.INHERITS:      "hereda de",
        CodeRelation.MUTATES:       "modifica",
        CodeRelation.READS:         "lee",
        CodeRelation.RETURNS:       "retorna",
        CodeRelation.THROWS:        "puede lanzar",
        CodeRelation.DEPENDS_ON:    "depende de",
        CodeRelation.TESTS:         "verifica",
        CodeRelation.IMPLEMENTS:    "implementa",
        CodeRelation.OVERRIDES:     "sobreescribe",
        CodeRelation.DATA_FLOWS_TO: "fluye datos a",
    }.get(rel, rel.value)


def _successors_with_rel(
    graph: CausalGraph,
    node_id: str,
    relation: Optional[CodeRelation] = None,
) -> List[str]:
    """Retorna los IDs de nodos destino de las aristas salientes del nodo dado."""
    result = []
    for edge in graph.edges:
        if edge.source_id == node_id:
            if relation is None or edge.relation == relation:
                result.append(edge.target_id)
    return result


def _predecessors_with_rel(
    graph: CausalGraph,
    node_id: str,
    relation: Optional[CodeRelation] = None,
) -> List[str]:
    """Retorna los IDs de nodos fuente de las aristas entrantes al nodo dado."""
    result = []
    for edge in graph.edges:
        if edge.target_id == node_id:
            if relation is None or edge.relation == relation:
                result.append(edge.source_id)
    return result


def _all_reachable(
    graph: CausalGraph,
    start_id: str,
    relation: Optional[CodeRelation] = None,
) -> Set[str]:
    """BFS: todos los nodos alcanzables desde start_id siguiendo aristas (o relación específica)."""
    visited = set()
    queue   = [start_id]
    while queue:
        cur = queue.pop(0)
        if cur in visited:
            continue
        visited.add(cur)
        for nid in _successors_with_rel(graph, cur, relation):
            if nid not in visited:
                queue.append(nid)
    visited.discard(start_id)
    return visited


# ─────────────────────────────────────────────────────────────────────────────
# GENERADOR NIVEL 1 — CADENAS DIRECTAS DE LLAMADA (2-3 NODOS)
# ─────────────────────────────────────────────────────────────────────────────

class _Level1Generator:
    """
    Nivel 1: Relaciones directas entre 2-3 elementos de código.

    Subtypes:
      direct_caller:  A llama a B, ¿qué llama a B?
      direct_callee:  A llama a B, ¿a qué llama A?
      reads_writes:   A lee/escribe V, ¿qué variable lee/escribe A?
      chain_impact:   A→B→C (CALLS), ¿A llega a C indirectamente?
    """

    _SUBTYPES = ["direct_caller", "direct_callee", "reads_writes", "chain_impact"]

    def generate(self, rng: random.Random, domain: Optional[str] = None) -> CodeExample:
        domain   = domain or rng.choice(list(WEB_APP_NODES.keys()))
        subtype  = rng.choice(self._SUBTYPES)
        nodes_desc = WEB_APP_NODES[domain]

        if subtype == "direct_caller":
            return self._direct_caller(rng, domain, nodes_desc)
        elif subtype == "direct_callee":
            return self._direct_callee(rng, domain, nodes_desc)
        elif subtype == "reads_writes":
            return self._reads_writes(rng, domain, nodes_desc)
        else:
            return self._chain_impact(rng, domain, nodes_desc)

    def _direct_caller(self, rng, domain, nodes_desc) -> CodeExample:
        """¿Qué función llama directamente a X?"""
        idxs = rng.sample(range(len(nodes_desc)), k=min(2, len(nodes_desc)))
        a_desc, b_desc = nodes_desc[idxs[0]], nodes_desc[idxs[1]]

        g = CausalGraph()
        a = _code_node(a_desc)
        b = _code_node(b_desc)
        g.add_node(a).add_node(b)
        g.add_edge(CodeEdge(a.node_id, b.node_id, CodeRelation.CALLS,
                            strength=1.0, confidence=1.0))

        problem = (
            f"En el sistema, '{a.label}' {_rel_text(CodeRelation.CALLS)} '{b.label}'. "
            f"Pregunta: ¿Qué función llama directamente a '{b.label}'?"
        )
        answer = f"'{b.label}' es llamada directamente por '{a.label}'."
        g.root_question = problem
        return CodeExample(
            problem_text=problem, graph=g, answer=answer,
            complexity_level=1, answer_type=CodeAnswerType.DIRECT_CALLER,
            metadata={
                "domain": domain, "variant": "direct_caller",
                "target_id": b.node_id,
                "expected_callers": [a.node_id],
            },
        )

    def _direct_callee(self, rng, domain, nodes_desc) -> CodeExample:
        """¿A qué llama directamente X?"""
        idxs = rng.sample(range(len(nodes_desc)), k=min(2, len(nodes_desc)))
        a_desc, b_desc = nodes_desc[idxs[0]], nodes_desc[idxs[1]]

        g = CausalGraph()
        a = _code_node(a_desc)
        b = _code_node(b_desc)
        g.add_node(a).add_node(b)
        rel = rng.choice([CodeRelation.CALLS, CodeRelation.DATA_FLOWS_TO])
        g.add_edge(CodeEdge(a.node_id, b.node_id, rel, strength=1.0, confidence=1.0))

        problem = (
            f"En el sistema, '{a.label}' {_rel_text(rel)} '{b.label}'. "
            f"Pregunta: ¿A qué {_rel_text(rel).split()[0]} directamente '{a.label}'?"
        )
        answer = f"'{a.label}' {_rel_text(rel)} '{b.label}'."
        g.root_question = problem
        return CodeExample(
            problem_text=problem, graph=g, answer=answer,
            complexity_level=1, answer_type=CodeAnswerType.DIRECT_CALLEE,
            metadata={
                "domain": domain, "variant": "direct_callee",
                "source_id": a.node_id,
                "expected_callees": [b.node_id],
                "relation": rel.value,
            },
        )

    def _reads_writes(self, rng, domain, nodes_desc) -> CodeExample:
        """¿Qué variable lee o escribe la función X?"""
        # Busca un par función+variable en el dominio
        fn_descs  = [d for d in nodes_desc if d[2] in (CodeNodeType.FUNCTION, CodeNodeType.CLASS)]
        var_descs = [d for d in nodes_desc if d[2] in (CodeNodeType.VARIABLE, CodeNodeType.CONFIG)]

        if not fn_descs or not var_descs:
            # Fallback: dos nodos cualesquiera
            return self._direct_caller(rng, domain, nodes_desc)

        fn_desc  = rng.choice(fn_descs)
        var_desc = rng.choice(var_descs)
        rel = rng.choice([CodeRelation.READS, CodeRelation.MUTATES])

        g = CausalGraph()
        fn  = _code_node(fn_desc)
        var = _code_node(var_desc)
        g.add_node(fn).add_node(var)
        g.add_edge(CodeEdge(fn.node_id, var.node_id, rel, strength=1.0, confidence=1.0))

        action = "lee" if rel == CodeRelation.READS else "modifica"
        problem = (
            f"'{fn.label}' {action} '{var.label}'. "
            f"Pregunta: ¿Qué variable {action} '{fn.label}'?"
        )
        answer = f"'{fn.label}' {action} la variable '{var.label}'."
        g.root_question = problem
        return CodeExample(
            problem_text=problem, graph=g, answer=answer,
            complexity_level=1, answer_type=CodeAnswerType.READS_WRITES,
            metadata={
                "domain": domain, "variant": "reads_writes",
                "source_id": fn.node_id,
                "expected_variable_id": var.node_id,
                "relation": rel.value,
            },
        )

    def _chain_impact(self, rng, domain, nodes_desc) -> CodeExample:
        """A llama a B que llama a C. ¿A llega a C?"""
        if len(nodes_desc) < 3:
            return self._direct_caller(rng, domain, nodes_desc)
        idxs = rng.sample(range(len(nodes_desc)), k=3)
        a_desc, b_desc, c_desc = nodes_desc[idxs[0]], nodes_desc[idxs[1]], nodes_desc[idxs[2]]

        g = CausalGraph()
        a = _code_node(a_desc)
        b = _code_node(b_desc)
        c = _code_node(c_desc)
        g.add_node(a).add_node(b).add_node(c)
        g.add_edge(CodeEdge(a.node_id, b.node_id, CodeRelation.CALLS, strength=1.0, confidence=1.0))
        g.add_edge(CodeEdge(b.node_id, c.node_id, CodeRelation.CALLS, strength=1.0, confidence=1.0))

        problem = (
            f"'{a.label}' llama a '{b.label}', y '{b.label}' llama a '{c.label}'. "
            f"Pregunta: ¿Puede '{a.label}' llegar (indirectamente) a '{c.label}'?"
        )
        answer = (
            f"Sí. Existe la cadena: '{a.label}' → '{b.label}' → '{c.label}'. "
            f"Por transitividad, '{a.label}' puede alcanzar indirectamente '{c.label}'."
        )
        g.root_question = problem
        return CodeExample(
            problem_text=problem, graph=g, answer=answer,
            complexity_level=1, answer_type=CodeAnswerType.IMPACT_ANALYSIS,
            metadata={
                "domain": domain, "variant": "chain_impact",
                "source_id": a.node_id,
                "target_id": c.node_id,
                "expected_reachable": True,
                "path": [a.node_id, b.node_id, c.node_id],
            },
        )


# ─────────────────────────────────────────────────────────────────────────────
# GENERADOR NIVEL 2 — ANÁLISIS DE IMPACTO / DEPENDENCIAS (3-5 NODOS)
# ─────────────────────────────────────────────────────────────────────────────

class _Level2Generator:
    """
    Nivel 2: Estructuras de fan-out/fan-in para análisis de impacto y dependencias.

    Subtypes:
      fan_out_impact: A llama a B y C. ¿Qué se ve afectado si cambio A?
      fan_in_dep:     B y C llaman a D. ¿De qué depende D transitivamente?
      module_deps:    M1 importa M2 y M3. ¿Cuáles son las dependencias de M1?
      error_fan_out:  A throws E; B y C usan A. ¿Qué falla si A lanza E?
    """

    def generate(self, rng: random.Random, domain: Optional[str] = None) -> CodeExample:
        domain  = domain or rng.choice(list(WEB_APP_NODES.keys()))
        subtype = rng.choice(["fan_out_impact", "fan_in_dep", "module_deps", "error_fan_out"])

        if subtype == "fan_out_impact":
            return self._fan_out_impact(rng, domain)
        elif subtype == "fan_in_dep":
            return self._fan_in_dep(rng, domain)
        elif subtype == "module_deps":
            return self._module_deps(rng)
        else:
            return self._error_fan_out(rng, domain)

    def _fan_out_impact(self, rng, domain) -> CodeExample:
        """A calls B and C. ¿Qué se ve afectado si cambio A?"""
        nodes_desc = WEB_APP_NODES[domain]
        if len(nodes_desc) < 3:
            nodes_desc = WEB_APP_NODES["api"]
        idxs = rng.sample(range(len(nodes_desc)), k=min(3, len(nodes_desc)))
        a_desc, b_desc, c_desc = nodes_desc[idxs[0]], nodes_desc[idxs[1]], nodes_desc[idxs[2]]

        g = CausalGraph()
        a = _code_node(a_desc)
        b = _code_node(b_desc)
        c = _code_node(c_desc)
        g.add_node(a).add_node(b).add_node(c)
        g.add_edge(CodeEdge(a.node_id, b.node_id, CodeRelation.CALLS, strength=1.0, confidence=1.0))
        g.add_edge(CodeEdge(a.node_id, c.node_id, CodeRelation.CALLS, strength=1.0, confidence=1.0))

        affected = [b.node_id, c.node_id]
        problem = (
            f"'{a.label}' llama a '{b.label}' y a '{c.label}'. "
            f"Pregunta: ¿Qué funciones se verían directamente afectadas si cambia '{a.label}'?"
        )
        answer = (
            f"Si '{a.label}' cambia, se ven afectadas directamente: "
            f"'{b.label}' y '{c.label}'."
        )
        g.root_question = problem
        return CodeExample(
            problem_text=problem, graph=g, answer=answer,
            complexity_level=2, answer_type=CodeAnswerType.IMPACT_ANALYSIS,
            metadata={
                "domain": domain, "variant": "fan_out_impact",
                "source_id": a.node_id,
                "expected_affected_count": 2,
                "expected_affected_ids": affected,
            },
        )

    def _fan_in_dep(self, rng, domain) -> CodeExample:
        """B y C llaman a D. ¿Quiénes usan D?"""
        nodes_desc = WEB_APP_NODES[domain]
        if len(nodes_desc) < 3:
            nodes_desc = WEB_APP_NODES["service"]
        idxs = rng.sample(range(len(nodes_desc)), k=min(3, len(nodes_desc)))
        b_desc, c_desc, d_desc = nodes_desc[idxs[0]], nodes_desc[idxs[1]], nodes_desc[idxs[2]]

        g = CausalGraph()
        b = _code_node(b_desc)
        c = _code_node(c_desc)
        d = _code_node(d_desc)
        g.add_node(b).add_node(c).add_node(d)
        g.add_edge(CodeEdge(b.node_id, d.node_id, CodeRelation.CALLS, strength=1.0, confidence=1.0))
        g.add_edge(CodeEdge(c.node_id, d.node_id, CodeRelation.CALLS, strength=1.0, confidence=1.0))

        callers = [b.node_id, c.node_id]
        problem = (
            f"'{b.label}' y '{c.label}' ambas llaman a '{d.label}'. "
            f"Pregunta: ¿Qué funciones dependen directamente de '{d.label}'?"
        )
        answer = (
            f"'{d.label}' es usada directamente por '{b.label}' y '{c.label}'."
        )
        g.root_question = problem
        return CodeExample(
            problem_text=problem, graph=g, answer=answer,
            complexity_level=2, answer_type=CodeAnswerType.DEPENDENCY,
            metadata={
                "domain": domain, "variant": "fan_in_dep",
                "target_id": d.node_id,
                "expected_callers": callers,
                "expected_caller_count": 2,
            },
        )

    def _module_deps(self, rng) -> CodeExample:
        """M1 importa M2 y M3. ¿Cuáles son las dependencias de M1?"""
        mod_descs = WEB_APP_NODES["module"]
        if len(mod_descs) < 3:
            return self._fan_out_impact(rng, "api")
        idxs = rng.sample(range(len(mod_descs)), k=3)
        m1_desc, m2_desc, m3_desc = mod_descs[idxs[0]], mod_descs[idxs[1]], mod_descs[idxs[2]]

        g = CausalGraph()
        m1 = _code_node(m1_desc)
        m2 = _code_node(m2_desc)
        m3 = _code_node(m3_desc)
        g.add_node(m1).add_node(m2).add_node(m3)
        g.add_edge(CodeEdge(m1.node_id, m2.node_id, CodeRelation.IMPORTS, strength=1.0, confidence=1.0))
        g.add_edge(CodeEdge(m1.node_id, m3.node_id, CodeRelation.IMPORTS, strength=1.0, confidence=1.0))

        deps = [m2.node_id, m3.node_id]
        problem = (
            f"El módulo '{m1.label}' importa '{m2.label}' e importa '{m3.label}'. "
            f"Pregunta: ¿De qué módulos depende directamente '{m1.label}'?"
        )
        answer = (
            f"'{m1.label}' depende directamente de: '{m2.label}' y '{m3.label}'."
        )
        g.root_question = problem
        return CodeExample(
            problem_text=problem, graph=g, answer=answer,
            complexity_level=2, answer_type=CodeAnswerType.DEPENDENCY,
            metadata={
                "variant": "module_deps",
                "source_id": m1.node_id,
                "expected_deps": deps,
                "expected_dep_count": 2,
            },
        )

    def _error_fan_out(self, rng, domain) -> CodeExample:
        """A puede lanzar E; B y C llaman a A. ¿Qué falla si A lanza E?"""
        fn_descs = WEB_APP_NODES["api"]
        err_descs = [d for d in WEB_APP_NODES["db"] if d[2] == CodeNodeType.ERROR]
        if not err_descs or len(fn_descs) < 3:
            return self._fan_out_impact(rng, domain)

        idxs  = rng.sample(range(len(fn_descs)), k=3)
        a_desc = fn_descs[idxs[0]]
        b_desc = fn_descs[idxs[1]]
        c_desc = fn_descs[idxs[2]]
        e_desc = err_descs[0]

        g = CausalGraph()
        a = _code_node(a_desc)
        b = _code_node(b_desc)
        c = _code_node(c_desc)
        e = _code_node(e_desc)
        g.add_node(a).add_node(b).add_node(c).add_node(e)
        g.add_edge(CodeEdge(a.node_id, e.node_id, CodeRelation.THROWS, strength=0.9, confidence=0.9))
        g.add_edge(CodeEdge(b.node_id, a.node_id, CodeRelation.CALLS, strength=1.0, confidence=1.0))
        g.add_edge(CodeEdge(c.node_id, a.node_id, CodeRelation.CALLS, strength=1.0, confidence=1.0))

        problem = (
            f"'{a.label}' puede lanzar '{e.label}'. "
            f"'{b.label}' y '{c.label}' llaman a '{a.label}'. "
            f"Pregunta: ¿Qué funciones fallarían si '{a.label}' lanza '{e.label}'?"
        )
        answer = (
            f"Si '{a.label}' lanza '{e.label}', fallarían directamente: "
            f"'{b.label}' y '{c.label}' (ambas la llaman)."
        )
        g.root_question = problem
        return CodeExample(
            problem_text=problem, graph=g, answer=answer,
            complexity_level=2, answer_type=CodeAnswerType.ERROR_PROPAGATION,
            metadata={
                "domain": domain, "variant": "error_fan_out",
                "error_source_id": a.node_id,
                "error_id": e.node_id,
                "expected_affected_ids": [b.node_id, c.node_id],
                "expected_affected_count": 2,
            },
        )


# ─────────────────────────────────────────────────────────────────────────────
# GENERADOR NIVEL 3 — TRAZABILIDAD DE DATOS Y PROPAGACIÓN DE ERRORES (4-7 NODOS)
# ─────────────────────────────────────────────────────────────────────────────

class _Level3Generator:
    """
    Nivel 3: Grafos de flujo de datos multi-salto y propagación de errores.

    Subtypes:
      data_source:      ¿De dónde viene el dato que usa C? (remonta DATA_FLOWS_TO)
      call_chain_break: A→B→C→D: ¿qué pasa si eliminamos B? (¿llega A a D?)
      multi_caller:     Grafo mixto, ¿cuántos callers alcanza X transitivamente?
      inheritance:      A hereda de B que hereda de C. ¿A tiene acceso a C?
    """

    def generate(self, rng: random.Random, domain: Optional[str] = None) -> CodeExample:
        subtype = rng.choice(["data_source", "call_chain_break", "multi_caller", "inheritance"])
        if subtype == "data_source":
            return self._data_source(rng)
        elif subtype == "call_chain_break":
            return self._call_chain_break(rng, domain or rng.choice(list(WEB_APP_NODES.keys())))
        elif subtype == "multi_caller":
            return self._multi_caller(rng, domain or rng.choice(list(WEB_APP_NODES.keys())))
        else:
            return self._inheritance(rng)

    def _data_source(self, rng) -> CodeExample:
        """
        request_body → validate_input → auth_token → route_handler
        ¿De dónde viene el dato que usa route_handler?
        """
        data_descs = WEB_APP_NODES["data"]
        api_descs  = WEB_APP_NODES["api"]
        if len(data_descs) < 2 or len(api_descs) < 2:
            return self._multi_caller(rng, "api")

        req_desc = data_descs[2]   # request_body
        tok_desc = data_descs[1]   # auth_token
        fn1_desc = api_descs[2]    # validate_input
        fn2_desc = api_descs[0]    # route_handler

        g = CausalGraph()
        req = _code_node(req_desc)
        tok = _code_node(tok_desc)
        fn1 = _code_node(fn1_desc)
        fn2 = _code_node(fn2_desc)
        g.add_node(req).add_node(fn1).add_node(tok).add_node(fn2)
        g.add_edge(CodeEdge(req.node_id, fn1.node_id, CodeRelation.DATA_FLOWS_TO,
                            strength=1.0, confidence=1.0))
        g.add_edge(CodeEdge(fn1.node_id, tok.node_id, CodeRelation.DATA_FLOWS_TO,
                            strength=1.0, confidence=1.0))
        g.add_edge(CodeEdge(tok.node_id, fn2.node_id, CodeRelation.DATA_FLOWS_TO,
                            strength=1.0, confidence=1.0))

        problem = (
            f"El flujo de datos es: '{req.label}' → '{fn1.label}' → '{tok.label}' → '{fn2.label}'. "
            f"Pregunta: ¿De dónde viene el dato que usa '{fn2.label}'?"
        )
        answer = (
            f"El dato que usa '{fn2.label}' viene de '{tok.label}', "
            f"que a su vez recibe de '{fn1.label}', que lee de '{req.label}'. "
            f"La fuente original es '{req.label}'."
        )
        g.root_question = problem
        return CodeExample(
            problem_text=problem, graph=g, answer=answer,
            complexity_level=3, answer_type=CodeAnswerType.DATA_SOURCE,
            metadata={
                "variant": "data_source",
                "consumer_id": fn2.node_id,
                "direct_source_id": tok.node_id,
                "root_source_id": req.node_id,
                "path": [req.node_id, fn1.node_id, tok.node_id, fn2.node_id],
            },
        )

    def _call_chain_break(self, rng, domain) -> CodeExample:
        """A→B→C→D (CALLS). Si eliminamos B, ¿A puede llegar a D?"""
        nodes_desc = WEB_APP_NODES[domain]
        if len(nodes_desc) < 4:
            nodes_desc = WEB_APP_NODES["api"]
        idxs = rng.sample(range(len(nodes_desc)), k=4)
        a_d, b_d, c_d, d_d = [nodes_desc[i] for i in idxs]

        has_alt = rng.random() < 0.4   # 40% chance hay camino alternativo A→D directo

        g = CausalGraph()
        a = _code_node(a_d)
        b = _code_node(b_d)
        c = _code_node(c_d)
        d = _code_node(d_d)
        g.add_node(a).add_node(b).add_node(c).add_node(d)
        g.add_edge(CodeEdge(a.node_id, b.node_id, CodeRelation.CALLS, strength=1.0, confidence=1.0))
        g.add_edge(CodeEdge(b.node_id, c.node_id, CodeRelation.CALLS, strength=1.0, confidence=1.0))
        g.add_edge(CodeEdge(c.node_id, d.node_id, CodeRelation.CALLS, strength=1.0, confidence=1.0))
        if has_alt:
            g.add_edge(CodeEdge(a.node_id, d.node_id, CodeRelation.DATA_FLOWS_TO,
                                strength=0.8, confidence=0.9))

        problem = (
            f"'{a.label}' llama a '{b.label}', que llama a '{c.label}', que llama a '{d.label}'. "
            + (f"Además, '{a.label}' también envía datos a '{d.label}' directamente. " if has_alt else "")
            + f"Pregunta: Si '{b.label}' es eliminada, ¿'{a.label}' aún puede llegar a '{d.label}'?"
        )
        if has_alt:
            answer = (
                f"Sí. Aunque '{b.label}' sea eliminada, existe un camino alternativo: "
                f"'{a.label}' envía datos directamente a '{d.label}'."
            )
        else:
            answer = (
                f"No. Si '{b.label}' es eliminada, la única cadena de llamada es "
                f"'{a.label}'→'{b.label}'→'{c.label}'→'{d.label}', que queda rota. "
                f"'{a.label}' no puede llegar a '{d.label}'."
            )
        g.root_question = problem
        return CodeExample(
            problem_text=problem, graph=g, answer=answer,
            complexity_level=3, answer_type=CodeAnswerType.IMPACT_ANALYSIS,
            metadata={
                "domain": domain, "variant": "call_chain_break",
                "source_id": a.node_id,
                "target_id": d.node_id,
                "removed_id": b.node_id,
                "has_alternate_path": has_alt,
                "expected_reachable_without_b": has_alt,
            },
        )

    def _multi_caller(self, rng, domain) -> CodeExample:
        """A llama a B y C; B llama a D. ¿Cuántos nodos alcanza A transitivamente?"""
        nodes_desc = WEB_APP_NODES[domain]
        if len(nodes_desc) < 4:
            nodes_desc = WEB_APP_NODES["service"]
        idxs = rng.sample(range(len(nodes_desc)), k=min(4, len(nodes_desc)))
        a_d, b_d, c_d, d_d = [nodes_desc[i] for i in idxs]

        g = CausalGraph()
        a = _code_node(a_d)
        b = _code_node(b_d)
        c = _code_node(c_d)
        d = _code_node(d_d)
        g.add_node(a).add_node(b).add_node(c).add_node(d)
        g.add_edge(CodeEdge(a.node_id, b.node_id, CodeRelation.CALLS, strength=1.0, confidence=1.0))
        g.add_edge(CodeEdge(a.node_id, c.node_id, CodeRelation.CALLS, strength=1.0, confidence=1.0))
        g.add_edge(CodeEdge(b.node_id, d.node_id, CodeRelation.CALLS, strength=1.0, confidence=1.0))

        reachable = {b.node_id, c.node_id, d.node_id}
        problem = (
            f"'{a.label}' llama a '{b.label}' y a '{c.label}'. "
            f"'{b.label}' llama a '{d.label}'. "
            f"Pregunta: ¿Cuántos nodos puede alcanzar '{a.label}' transitivamente?"
        )
        answer = (
            f"'{a.label}' puede alcanzar transitivamente 3 nodos: "
            f"'{b.label}', '{c.label}' (directos) y '{d.label}' (a través de '{b.label}')."
        )
        g.root_question = problem
        return CodeExample(
            problem_text=problem, graph=g, answer=answer,
            complexity_level=3, answer_type=CodeAnswerType.IMPACT_ANALYSIS,
            metadata={
                "domain": domain, "variant": "multi_caller",
                "source_id": a.node_id,
                "expected_reachable_ids": list(reachable),
                "expected_reachable_count": len(reachable),
            },
        )

    def _inheritance(self, rng) -> CodeExample:
        """A hereda de B que hereda de C. ¿A tiene acceso a métodos de C?"""
        svc_descs = WEB_APP_NODES["service"]
        if len(svc_descs) < 3:
            return self._data_source(rng)
        idxs = rng.sample(range(len(svc_descs)), k=3)
        a_d, b_d, c_d = svc_descs[idxs[0]], svc_descs[idxs[1]], svc_descs[idxs[2]]

        g = CausalGraph()
        a = _code_node(a_d)
        b = _code_node(b_d)
        c = _code_node(c_d)
        g.add_node(a).add_node(b).add_node(c)
        g.add_edge(CodeEdge(a.node_id, b.node_id, CodeRelation.INHERITS,
                            strength=1.0, confidence=1.0))
        g.add_edge(CodeEdge(b.node_id, c.node_id, CodeRelation.INHERITS,
                            strength=1.0, confidence=1.0))

        problem = (
            f"'{a.label}' hereda de '{b.label}', y '{b.label}' hereda de '{c.label}'. "
            f"Pregunta: ¿'{a.label}' tiene acceso (por herencia) a los métodos de '{c.label}'?"
        )
        answer = (
            f"Sí. '{a.label}' hereda de '{b.label}', que a su vez hereda de '{c.label}'. "
            f"Por herencia transitiva, '{a.label}' tiene acceso a los métodos de '{c.label}'."
        )
        g.root_question = problem
        return CodeExample(
            problem_text=problem, graph=g, answer=answer,
            complexity_level=3, answer_type=CodeAnswerType.INHERITANCE,
            metadata={
                "variant": "inheritance",
                "leaf_id": a.node_id,
                "root_id": c.node_id,
                "expected_inherits_transitively": True,
                "inheritance_path": [a.node_id, b.node_id, c.node_id],
            },
        )


# ─────────────────────────────────────────────────────────────────────────────
# VERIFICACIÓN
# ─────────────────────────────────────────────────────────────────────────────

def verify_code_example(ex: CodeExample) -> CodeVerificationResult:
    """
    Verifica que el grafo de código y la respuesta del ejemplo son consistentes.

    Contratos por tipo de pregunta:
      DIRECT_CALLER:   predecessors(target_id, CALLS) == expected_callers
      DIRECT_CALLEE:   successors(source_id, rel) == expected_callees
      READS_WRITES:    successors(source_id, rel) includes expected_variable_id
      IMPACT_ANALYSIS: reachable(source_id) includes expected_affected_ids
      DEPENDENCY:      predecessors(target_id, CALLS|IMPORTS) == expected_callers
      ERROR_PROPAGATION: predecessors(error_source_id, CALLS) == expected_affected_ids
      DATA_SOURCE:     path exists from root_source_id to consumer_id
      INHERITANCE:     path exists via INHERITS from leaf_id to root_id
    """
    g    = ex.graph
    meta = ex.metadata
    atype = ex.answer_type

    try:
        if atype == CodeAnswerType.DIRECT_CALLER:
            target_id = meta.get("target_id")
            expected  = set(meta.get("expected_callers", []))
            actual    = set(_predecessors_with_rel(g, target_id, CodeRelation.CALLS))
            if actual != expected:
                return CodeVerificationResult(
                    False,
                    f"Expected callers {expected}, got {actual}",
                    {"expected": expected, "actual": actual},
                )

        elif atype == CodeAnswerType.DIRECT_CALLEE:
            source_id = meta.get("source_id")
            expected  = set(meta.get("expected_callees", []))
            rel_str   = meta.get("relation", CodeRelation.CALLS.value)
            rel       = CodeRelation(rel_str)
            actual    = set(_successors_with_rel(g, source_id, rel))
            if actual != expected:
                return CodeVerificationResult(
                    False,
                    f"Expected callees {expected}, got {actual}",
                    {"expected": expected, "actual": actual},
                )

        elif atype == CodeAnswerType.READS_WRITES:
            source_id  = meta.get("source_id")
            expected_v = meta.get("expected_variable_id")
            rel_str    = meta.get("relation", CodeRelation.READS.value)
            rel        = CodeRelation(rel_str)
            actual     = set(_successors_with_rel(g, source_id, rel))
            if expected_v not in actual:
                return CodeVerificationResult(
                    False,
                    f"Expected variable {expected_v} in {actual}",
                    {"expected": expected_v, "actual": actual},
                )

        elif atype == CodeAnswerType.IMPACT_ANALYSIS:
            source_id = meta.get("source_id")
            if source_id:
                expected  = set(meta.get("expected_affected_ids", []))
                actual    = _all_reachable(g, source_id)
                if not expected.issubset(actual):
                    missing = expected - actual
                    return CodeVerificationResult(
                        False,
                        f"Expected {expected} reachable, missing {missing}",
                        {"expected": expected, "actual": actual},
                    )

        elif atype == CodeAnswerType.DEPENDENCY:
            target_id = meta.get("target_id")
            source_id = meta.get("source_id")
            if target_id:
                expected = set(meta.get("expected_callers", []))
                actual   = set(_predecessors_with_rel(g, target_id))
                if actual != expected:
                    return CodeVerificationResult(
                        False,
                        f"Expected predecessors {expected}, got {actual}",
                        {"expected": expected, "actual": actual},
                    )
            elif source_id:
                expected = set(meta.get("expected_deps", []))
                actual   = set(_successors_with_rel(g, source_id))
                if actual != expected:
                    return CodeVerificationResult(
                        False,
                        f"Expected deps {expected}, got {actual}",
                        {"expected": expected, "actual": actual},
                    )

        elif atype == CodeAnswerType.ERROR_PROPAGATION:
            err_src_id = meta.get("error_source_id")
            expected   = set(meta.get("expected_affected_ids", []))
            actual     = set(_predecessors_with_rel(g, err_src_id, CodeRelation.CALLS))
            if actual != expected:
                return CodeVerificationResult(
                    False,
                    f"Expected callers of error source {expected}, got {actual}",
                    {"expected": expected, "actual": actual},
                )

        elif atype == CodeAnswerType.DATA_SOURCE:
            consumer_id   = meta.get("consumer_id")
            root_source_id = meta.get("root_source_id")
            if consumer_id and root_source_id:
                # Verificar que existe camino de root_source_id a consumer_id
                reachable_from_root = _all_reachable(g, root_source_id, CodeRelation.DATA_FLOWS_TO)
                if consumer_id not in reachable_from_root:
                    return CodeVerificationResult(
                        False,
                        f"No DATA_FLOWS_TO path from {root_source_id} to {consumer_id}",
                        {"root": root_source_id, "consumer": consumer_id,
                         "reachable": reachable_from_root},
                    )

        elif atype == CodeAnswerType.INHERITANCE:
            leaf_id = meta.get("leaf_id")
            root_id = meta.get("root_id")
            if leaf_id and root_id:
                reachable = _all_reachable(g, leaf_id, CodeRelation.INHERITS)
                if root_id not in reachable:
                    return CodeVerificationResult(
                        False,
                        f"No INHERITS path from {leaf_id} to {root_id}",
                        {"leaf": leaf_id, "root": root_id, "reachable": reachable},
                    )

        return CodeVerificationResult(
            True, "All graph-answer constraints satisfied", {}
        )

    except Exception as exc:
        return CodeVerificationResult(
            False, f"Verification error: {exc}", {"exception": str(exc)}
        )


# ─────────────────────────────────────────────────────────────────────────────
# GENERADOR PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

class CodeGraphGenerator:
    """
    Generador de grafos de código sintéticos para FORGE-C.

    Misma interfaz que CausalGraphGenerator:
        gen = CodeGraphGenerator(seed=42)
        ex  = gen.generate(level=2)
        assert verify_code_example(ex).passed

        batch = gen.generate_batch(n=200, level_distribution={1: 0.4, 2: 0.4, 3: 0.2})
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng  = random.Random(seed)
        self._gens = {
            1: _Level1Generator(),
            2: _Level2Generator(),
            3: _Level3Generator(),
        }

    def generate(
        self,
        level: int = 1,
        domain: Optional[str] = None,
    ) -> CodeExample:
        """
        Genera un CodeExample del nivel indicado.

        Args:
            level:  1-3 (complejidad creciente)
            domain: dominio de código (None → aleatorio)

        Returns:
            CodeExample verificable
        """
        if level not in self._gens:
            raise ValueError(f"level must be 1-3, got {level}")
        return self._gens[level].generate(self._rng, domain)

    def generate_batch(
        self,
        n: int = 100,
        level_distribution: Optional[Dict[int, float]] = None,
        domain: Optional[str] = None,
    ) -> List[CodeExample]:
        """
        Genera un batch de n CodeExamples con la distribución de niveles indicada.

        Args:
            n:                    número de ejemplos
            level_distribution:   {level: probability} (deben sumar 1.0)
                                  Default: {1: 0.4, 2: 0.4, 3: 0.2}
            domain:               dominio fijo (None → aleatorio por ejemplo)

        Returns:
            List[CodeExample] de longitud n
        """
        if level_distribution is None:
            level_distribution = {1: 0.4, 2: 0.4, 3: 0.2}

        levels  = list(level_distribution.keys())
        weights = list(level_distribution.values())
        examples: List[CodeExample] = []

        for _ in range(n):
            level = self._rng.choices(levels, weights=weights, k=1)[0]
            examples.append(self.generate(level, domain=domain))

        return examples

    def available_domains(self) -> List[str]:
        """Lista de dominios disponibles para generación."""
        return list(WEB_APP_NODES.keys())
