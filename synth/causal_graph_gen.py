"""
synth/causal_graph_gen.py — Generador de Grafos Causales para AION-C
======================================================================

Motor de datos sintéticos para el CEC (Causal Energy Core).
Implementa el Generador B del plan FORGE-SYNTH:

  "El grafo se genera aleatoriamente pero con estructura lógica.
   Preguntas generadas automáticamente. Todas con respuestas verificables."

Cinco niveles de complejidad siguiendo el curriculum de AION-C:

  Nivel 1 — Cadenas lineales (2-3 nodos), preguntas de transitividad
  Nivel 2 — Bifurcaciones (fan-out / fan-in / diamante), 3-5 nodos
  Nivel 3 — Contradicciones intencionales que el modelo debe detectar
  Nivel 4 — Razonamiento contrafactual (si X no hubiera pasado...)
  Nivel 5 — Grafos multi-dominio, 8-15 nodos, relaciones mixtas

Contrato de cada CausalExample:
  - problem_text:     texto natural del problema
  - graph:            CausalGraph esperado (usando core/graph.py)
  - answer:           respuesta correcta en texto natural
  - complexity_level: 1-5
  - answer_type:      tipo de pregunta (TRANSITIVITY, BRANCHING, etc.)
  - verifiable:       True siempre — verify_example() lo confirma
  - metadata:         parámetros para verify_example()
  - example_id:       UUID reproducible con seed

Uso básico:
    gen = CausalGraphGenerator()
    ex  = gen.generate(level=1)
    res = verify_example(ex)
    assert res.passed

    batch = gen.generate_batch(n=100, level_distribution={1: 0.3, 2: 0.3, 3: 0.2, 4: 0.1, 5: 0.1})
"""

from __future__ import annotations

import copy
import random
import re
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.graph import (
    CausalEdge,
    CausalGraph,
    CausalNode,
    CausalRelation,
    NodeType,
    INHIBITORY_RELATIONS,
    POSITIVE_RELATIONS,
    CONTRADICTION_PAIRS,
)


# ─────────────────────────────────────────────────────────────────────────────
# TIPOS DE RESPUESTA
# ─────────────────────────────────────────────────────────────────────────────

class AnswerType(str, Enum):
    TRANSITIVITY  = "transitivity"   # ¿A lleva (indirectamente) a C?
    DIRECT_CAUSE  = "direct_cause"   # ¿Qué causa directamente X?
    BRANCHING     = "branching"      # ¿Qué efectos directos tiene A?
    CONTRADICTION = "contradiction"  # ¿Hay una contradicción en el sistema?
    COUNTERFACTUAL = "counterfactual" # ¿Si no hubiera X, qué pasaría con Z?
    CRITICAL_PATH  = "critical_path" # ¿Cuál es el camino causal más largo?
    MULTI_HOP      = "multi_hop"     # Razonamiento de varios saltos entre dominios


# ─────────────────────────────────────────────────────────────────────────────
# RESULTADO DE VERIFICACIÓN
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class VerificationResult:
    """
    Resultado de verify_example().

    passed:  True si el grafo y la respuesta son consistentes
    reason:  Explicación en lenguaje natural
    details: Datos cuantitativos de la verificación (para debugging)
    """
    passed: bool
    reason: str
    details: Dict = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.passed

    def __repr__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"VerificationResult({status}: {self.reason})"


# ─────────────────────────────────────────────────────────────────────────────
# EJEMPLO CAUSAL — EL CONTRATO DE DATOS DE ENTRENAMIENTO
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CausalExample:
    """
    Unidad atómica de entrenamiento para el CEC.

    Cada ejemplo contiene:
    - El problema en lenguaje natural (input al modelo)
    - El grafo causal esperado (target del GraphConstructor)
    - La respuesta correcta (target del SequenceDecoder)
    - Metadatos para verify_example()

    Compatible con FORGE SID (Synthetic Infinite Dataset):
    - Cero almacenamiento masivo — generado en CPU en tiempo real
    - 100% verificable — verify_example() confirma consistencia
    - Dificultad progresiva — complexity_level 1-5

    entity_spans: posiciones (start, end_exclusive) de cada nodo del grafo en
    los tokens de problem_text (tokenización word-level = split()).
    Span (-1, -1) indica que ese nodo no se encontró en el texto.
    Usado por loss_node_detection para dar supervisión posicional directa
    al NodeDetector en lugar de solo supervisión de conteo.
    """
    problem_text: str
    graph: CausalGraph
    answer: str
    complexity_level: int
    answer_type: AnswerType
    verifiable: bool = True
    metadata: Dict = field(default_factory=dict)
    example_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    entity_spans: List[Tuple[int, int]] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"CausalExample(level={self.complexity_level}, "
            f"type={self.answer_type.value}, "
            f"nodes={len(self.graph)}, id={self.example_id})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# ENTITY SPAN COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

_PUNCT_RE = re.compile(r"(?:^[^\w\u00C0-\u024F]+|[^\w\u00C0-\u024F]+$)")


def _strip_punct(s: str) -> str:
    """Elimina puntuación al inicio y al final de un token (preserva acentos)."""
    return _PUNCT_RE.sub("", s)


def compute_entity_spans(
    problem_text: str,
    graph: "CausalGraph",
) -> List[Tuple[int, int]]:
    """
    Mapea cada nodo del grafo a su span de tokens word-level en problem_text.

    Tokenización: str.lower().split() — idéntica a SimpleVocab.encode().
    La comparación se hace sobre tokens con puntuación inicial/final eliminada
    para manejar casos como "'Desempleo'" → "desempleo".
    Las posiciones devueltas corresponden a los tokens originales (sin limpiar).
    Cada nodo produce un span (start, end_exclusive).
    Si el nodo no aparece en el texto, produce (-1, -1).
    No se solapan spans: el primer match no ocupado por otro nodo gana.

    Ejemplo:
        problem_text = "Sabemos que: 'Lluvia' causa 'Suelo húmedo'."
        nodes = [Lluvia, Suelo húmedo]
        → [(3, 4), (6, 8)]    # "lluvia" en pos 3, "suelo húmedo" en pos 6-7
    """
    raw_tokens   = problem_text.lower().split()
    clean_tokens = [_strip_punct(t) for t in raw_tokens]
    spans: List[Tuple[int, int]] = []
    occupied: set = set()

    for node in graph.nodes:
        label_toks = [_strip_punct(t) for t in node.label.lower().split()]
        n = len(label_toks)
        found = False
        for i in range(len(clean_tokens) - n + 1):
            if clean_tokens[i:i + n] == label_toks and i not in occupied:
                spans.append((i, i + n))
                occupied.update(range(i, i + n))
                found = True
                break
        if not found:
            spans.append((-1, -1))

    return spans


# ─────────────────────────────────────────────────────────────────────────────
# POOLS DE DOMINIO — CONTENIDO SEMÁNTICO PARA LLENAR TEMPLATES
# ─────────────────────────────────────────────────────────────────────────────

# Formato de cada nodo: (node_id, label, NodeType, descripcion_larga)
_NodeDesc = Tuple[str, str, NodeType, str]

DOMAIN_NODES: Dict[str, List[_NodeDesc]] = {
    "clima": [
        ("lluvia",      "Lluvia",             NodeType.EVENT,  "precipitaciones intensas"),
        ("suelo_hum",   "Suelo húmedo",       NodeType.STATE,  "suelo saturado de agua"),
        ("inundacion",  "Inundación",         NodeType.EVENT,  "desbordamiento de ríos"),
        ("dano_agric",  "Daño agrícola",      NodeType.STATE,  "cosechas arrasadas"),
        ("dano_infra",  "Daño infraestructura",NodeType.STATE, "carreteras y puentes dañados"),
    ],
    "economia": [
        ("inflacion",   "Inflación",          NodeType.STATE,  "subida generalizada de precios"),
        ("alza_tipos",  "Alza de tipos",      NodeType.ACTION, "subida de tipos de interés"),
        ("baja_cred",   "Caída del crédito",  NodeType.STATE,  "reducción del crédito bancario"),
        ("baja_inv",    "Caída de inversión", NodeType.STATE,  "reducción de la inversión privada"),
        ("recesion",    "Recesión",           NodeType.EVENT,  "contracción económica sostenida"),
        ("desempleo",   "Desempleo",          NodeType.STATE,  "aumento del desempleo"),
    ],
    "salud": [
        ("virus",       "Virus",              NodeType.ENTITY, "agente patógeno viral"),
        ("infeccion",   "Infección",          NodeType.EVENT,  "contagio y propagación"),
        ("fiebre",      "Fiebre",             NodeType.STATE,  "temperatura corporal elevada"),
        ("debilidad",   "Debilidad",          NodeType.STATE,  "fatiga y reducción de defensas"),
        ("recup",       "Recuperación",       NodeType.EVENT,  "superación de la enfermedad"),
        ("inmunidad",   "Inmunidad",          NodeType.STATE,  "resistencia adquirida"),
    ],
    "tecnologia": [
        ("bug",         "Bug",                NodeType.EVENT,  "error en el código fuente"),
        ("error_rt",    "Error en runtime",   NodeType.EVENT,  "excepción no controlada"),
        ("fallo_sis",   "Fallo del sistema",  NodeType.EVENT,  "caída del servicio"),
        ("perd_datos",  "Pérdida de datos",   NodeType.STATE,  "datos corrompidos o eliminados"),
        ("tiempo_inact","Tiempo de inactividad",NodeType.STATE,"servicio no disponible"),
    ],
    "fisica": [
        ("calor",       "Calor excesivo",     NodeType.STATE,  "temperatura por encima del límite"),
        ("expansion",   "Expansión",          NodeType.EVENT,  "dilatación térmica del material"),
        ("presion",     "Presión elevada",    NodeType.STATE,  "presión por encima del límite"),
        ("ruptura",     "Ruptura",            NodeType.EVENT,  "fallo estructural del material"),
        ("fuga",        "Fuga",               NodeType.EVENT,  "escape del contenido"),
    ],
    "social": [
        ("paro",        "Desempleo",          NodeType.STATE,  "pérdida masiva de empleos"),
        ("pobreza",     "Pobreza",            NodeType.STATE,  "reducción del nivel de vida"),
        ("tension",     "Tensión social",     NodeType.STATE,  "malestar ciudadano creciente"),
        ("protesta",    "Protesta",           NodeType.EVENT,  "movilización ciudadana"),
        ("pol_social",  "Política social",    NodeType.ACTION, "medidas gubernamentales de apoyo"),
    ],
    "medioambiente": [
        ("contamin",    "Contaminación",      NodeType.STATE,  "emisiones tóxicas al ambiente"),
        ("cambio_clim", "Cambio climático",   NodeType.STATE,  "alteración del clima global"),
        ("sequia",      "Sequía",             NodeType.EVENT,  "ausencia prolongada de lluvias"),
        ("escasez",     "Escasez de agua",    NodeType.STATE,  "reducción de reservas hídricas"),
        ("incendio",    "Incendio forestal",  NodeType.EVENT,  "fuego no controlado en bosques"),
    ],
}

# Conexiones predefinidas dentro de cada dominio (nodo_idx_src → nodo_idx_tgt, relación)
# Se usan índices en la lista DOMAIN_NODES[domain]
DOMAIN_CHAINS: Dict[str, List[Tuple[int, int, CausalRelation]]] = {
    "clima":         [(0,1,CausalRelation.CAUSES), (1,2,CausalRelation.ENABLES),
                      (2,3,CausalRelation.CAUSES), (1,4,CausalRelation.CAUSES)],
    "economia":      [(0,1,CausalRelation.LEADS_TO), (1,2,CausalRelation.CAUSES),
                      (2,3,CausalRelation.CAUSES), (3,4,CausalRelation.ENABLES),
                      (4,5,CausalRelation.CAUSES)],
    "salud":         [(0,1,CausalRelation.CAUSES), (1,2,CausalRelation.CAUSES),
                      (2,3,CausalRelation.LEADS_TO), (3,4,CausalRelation.ENABLES),
                      (4,5,CausalRelation.ENABLES)],
    "tecnologia":    [(0,1,CausalRelation.CAUSES), (1,2,CausalRelation.LEADS_TO),
                      (2,3,CausalRelation.CAUSES), (2,4,CausalRelation.CAUSES)],
    "fisica":        [(0,1,CausalRelation.CAUSES), (1,2,CausalRelation.LEADS_TO),
                      (2,3,CausalRelation.CAUSES), (3,4,CausalRelation.ENABLES)],
    "social":        [(0,1,CausalRelation.CAUSES), (1,2,CausalRelation.LEADS_TO),
                      (2,3,CausalRelation.ENABLES), (3,4,CausalRelation.PREVENTS)],
    "medioambiente": [(0,1,CausalRelation.LEADS_TO), (1,2,CausalRelation.CAUSES),
                      (2,3,CausalRelation.CAUSES), (0,4,CausalRelation.ENABLES)],
}

# Conexiones cross-dominio para Level 5
# (dominio_src, idx_src, dominio_tgt, idx_tgt, relación)
CROSS_DOMAIN_EDGES: List[Tuple[str, int, str, int, CausalRelation]] = [
    ("clima",    2, "economia",       3, CausalRelation.LEADS_TO),   # inundación → baja_inv
    ("clima",    3, "economia",       4, CausalRelation.ENABLES),    # dano_agric → recesión
    ("economia", 4, "social",         0, CausalRelation.CAUSES),     # recesión → paro
    ("economia", 5, "social",         1, CausalRelation.CAUSES),     # desempleo → pobreza
    ("social",   1, "salud",          1, CausalRelation.ENABLES),    # pobreza → infección
    ("fisica",   2, "tecnologia",     2, CausalRelation.CAUSES),     # presión → fallo_sis
    ("tecnologia",2,"economia",       3, CausalRelation.LEADS_TO),   # fallo_sis → baja_inv
    ("medioambiente",1,"clima",       0, CausalRelation.LEADS_TO),   # cambio_clim → lluvia
    ("medioambiente",2,"social",      0, CausalRelation.ENABLES),    # sequía → paro (agrícola)
    ("contamin","fisica",0,0, CausalRelation.LEADS_TO),              # placeholder (no usado directamente)
]


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS INTERNOS
# ─────────────────────────────────────────────────────────────────────────────

def _node_from_desc(desc: _NodeDesc, prefix: str = "") -> CausalNode:
    """Crea un CausalNode desde un descriptor del pool de dominios."""
    nid, label, ntype, _ = desc
    return CausalNode(
        node_id=f"{prefix}{nid}" if prefix else nid,
        label=label,
        node_type=ntype,
        confidence=1.0,
        grounded=True,
    )


def _build_chain(
    domain: str,
    node_indices: List[int],
    rng: random.Random,
    prefix: str = "",
) -> Tuple[CausalGraph, List[CausalNode]]:
    """
    Construye un CausalGraph a partir de un subconjunto de nodos y las
    aristas del DOMAIN_CHAINS que conectan esos nodos entre sí.
    """
    nodes_desc = DOMAIN_NODES[domain]
    chain_edges = DOMAIN_CHAINS[domain]

    g = CausalGraph(root_question="")
    nodes: List[CausalNode] = []

    # Crear nodos en el orden dado
    for idx in node_indices:
        n = _node_from_desc(nodes_desc[idx], prefix=prefix)
        g.add_node(n)
        nodes.append(n)

    # Agregar aristas predefinidas entre los nodos seleccionados
    for src_idx, tgt_idx, rel in chain_edges:
        if src_idx in node_indices and tgt_idx in node_indices:
            src_id = f"{prefix}{nodes_desc[src_idx][0]}" if prefix else nodes_desc[src_idx][0]
            tgt_id = f"{prefix}{nodes_desc[tgt_idx][0]}" if prefix else nodes_desc[tgt_idx][0]
            if src_id in g and tgt_id in g:
                strength = round(rng.uniform(0.7, 1.0), 2)
                conf = round(rng.uniform(0.75, 1.0), 2)
                g.add_edge(CausalEdge(src_id, tgt_id, rel, strength=strength, confidence=conf))

    return g, nodes


def _longest_path(graph: CausalGraph) -> List[str]:
    """
    Encuentra el camino más largo (en número de aristas) en el grafo dirigido.
    Usa memoización sobre el grafo acíclico.
    Devuelve lista de node_ids del camino.
    """
    memo: Dict[str, List[str]] = {}

    def dfs(node_id: str) -> List[str]:
        if node_id in memo:
            return memo[node_id]
        best: List[str] = [node_id]
        for succ in graph.successors(node_id):
            candidate = [node_id] + dfs(succ.node_id)
            if len(candidate) > len(best):
                best = candidate
        memo[node_id] = best
        return best

    overall_best: List[str] = []
    for n in graph.nodes:
        path = dfs(n.node_id)
        if len(path) > len(overall_best):
            overall_best = path
    return overall_best


# ─────────────────────────────────────────────────────────────────────────────
# GENERADOR NIVEL 1 — CADENAS LINEALES (TRANSITIVIDAD)
# ─────────────────────────────────────────────────────────────────────────────

class _Level1Generator:
    """
    Nivel 1: Cadenas lineales A→B (→C), 2-3 nodos.
    Pregunta: ¿A lleva (directa o indirectamente) a C?
    El modelo aprende transitividad causal básica.
    """

    # (n_nodos, answer_type, subtipo)
    _SUBTYPES = [
        (2, AnswerType.DIRECT_CAUSE,  "direct"),       # A→B, ¿qué causa B?
        (2, AnswerType.TRANSITIVITY,  "two_reachable"), # A→B, ¿A lleva a B?
        (3, AnswerType.TRANSITIVITY,  "chain_yes"),     # A→B→C, ¿A lleva a C?
        (3, AnswerType.DIRECT_CAUSE,  "chain_direct"),  # A→B→C, ¿qué causa directamente B?
    ]

    def generate(self, rng: random.Random, domain: Optional[str] = None) -> CausalExample:
        domain = domain or rng.choice(list(DOMAIN_NODES.keys()))
        nodes_desc = DOMAIN_NODES[domain]
        n_available = len(DOMAIN_CHAINS[domain])

        subtype = rng.choice(self._SUBTYPES)
        n_nodes, answer_type, variant = subtype

        # Seleccionar nodos consecutivos del chain del dominio
        max_start = len(nodes_desc) - n_nodes
        start = rng.randint(0, max(0, max_start))
        node_indices = list(range(start, start + n_nodes))

        g, nodes = _build_chain(domain, node_indices, rng)

        if len(g.edges) == 0 or len(g) < 2:
            # Fallback: usar primeros nodos disponibles
            node_indices = list(range(min(n_nodes, len(nodes_desc))))
            g, nodes = _build_chain(domain, node_indices, rng)

        # Garantizar al menos una arista
        if len(g.edges) == 0:
            e = CausalEdge(nodes[0].node_id, nodes[-1].node_id, CausalRelation.CAUSES,
                           strength=1.0, confidence=1.0)
            g.add_edge(e)

        src_node = nodes[0]
        tgt_node = nodes[-1]
        middle_nodes = nodes[1:-1]

        # Construir contexto textual
        edge_descs = []
        for edge in g.edges:
            src_lbl = g.get_node(edge.source_id).label
            tgt_lbl = g.get_node(edge.target_id).label
            edge_descs.append(f"'{src_lbl}' {_rel_text(edge.relation)} '{tgt_lbl}'")
        context = "; ".join(edge_descs)

        if variant in ("direct",):
            causes = [e for e in g.in_edges(tgt_node.node_id)]
            cause_labels = [g.get_node(e.source_id).label for e in causes]
            problem = (
                f"Sabemos que: {context}. "
                f"Pregunta: ¿Qué causa directamente '{tgt_node.label}'?"
            )
            answer = (
                f"'{tgt_node.label}' es causado directamente por: "
                f"{', '.join(repr(l) for l in cause_labels)}."
            )
            meta = {
                "target_id": tgt_node.node_id,
                "expected_direct_causes": [g.get_node(e.source_id).node_id for e in causes],
            }

        elif variant == "two_reachable":
            reachable = g.has_path(src_node.node_id, tgt_node.node_id)
            if not reachable:
                reachable = True
                g.add_edge(CausalEdge(src_node.node_id, tgt_node.node_id,
                                      CausalRelation.CAUSES, strength=0.9, confidence=0.9))
            problem = (
                f"Sabemos que: {context}. "
                f"Pregunta: ¿'{src_node.label}' puede llevar a '{tgt_node.label}'?"
            )
            answer = (
                f"Sí, '{src_node.label}' lleva directamente a '{tgt_node.label}'."
                if reachable else
                f"No, no existe camino causal de '{src_node.label}' a '{tgt_node.label}'."
            )
            meta = {
                "source_id":        src_node.node_id,
                "target_id":        tgt_node.node_id,
                "expected_reachable": reachable,
            }

        elif variant == "chain_yes":
            reachable = g.has_path(src_node.node_id, tgt_node.node_id)
            via = " → ".join(n.label for n in nodes)
            problem = (
                f"Sabemos que: {context}. "
                f"Pregunta: ¿'{src_node.label}' puede (indirectamente) llevar a '{tgt_node.label}'?"
            )
            answer = (
                f"Sí. Existe la cadena causal: {via}. "
                f"Por transitividad, '{src_node.label}' puede llevar a '{tgt_node.label}'."
            ) if reachable else (
                f"No. No existe camino causal de '{src_node.label}' a '{tgt_node.label}'."
            )
            meta = {
                "source_id":        src_node.node_id,
                "target_id":        tgt_node.node_id,
                "expected_reachable": reachable,
            }

        else:  # chain_direct
            if middle_nodes:
                mid = middle_nodes[0]
                direct_causes = [g.get_node(e.source_id).label for e in g.in_edges(mid.node_id)]
                problem = (
                    f"Sabemos que: {context}. "
                    f"Pregunta: ¿Qué causa directamente '{mid.label}'?"
                )
                answer = (
                    f"'{mid.label}' es causado directamente por: "
                    f"{', '.join(repr(l) for l in direct_causes)}."
                )
                meta = {
                    "target_id": mid.node_id,
                    "expected_direct_causes": [g.get_node(e.source_id).node_id
                                               for e in g.in_edges(mid.node_id)],
                }
            else:
                # Fallback a transitivity
                problem = (
                    f"Sabemos que: {context}. "
                    f"Pregunta: ¿'{src_node.label}' puede llevar a '{tgt_node.label}'?"
                )
                answer = f"Sí, '{src_node.label}' lleva a '{tgt_node.label}'."
                meta = {"source_id": src_node.node_id, "target_id": tgt_node.node_id,
                        "expected_reachable": True}

        g.root_question = problem
        return CausalExample(
            problem_text=problem,
            graph=g,
            answer=answer,
            complexity_level=1,
            answer_type=answer_type,
            verifiable=True,
            metadata={**meta, "domain": domain, "variant": variant},
        )


# ─────────────────────────────────────────────────────────────────────────────
# GENERADOR NIVEL 2 — BIFURCACIONES (3-5 NODOS)
# ─────────────────────────────────────────────────────────────────────────────

class _Level2Generator:
    """
    Nivel 2: Fan-out (A→B, A→C), fan-in (A→C, B→C), diamante.
    3-5 nodos. Preguntas sobre efectos múltiples y causas convergentes.
    """

    _SUBTYPES = ["fan_out", "fan_in", "diamond", "mixed"]

    def generate(self, rng: random.Random, domain: Optional[str] = None) -> CausalExample:
        domain = domain or rng.choice(list(DOMAIN_NODES.keys()))
        nodes_desc = DOMAIN_NODES[domain]
        subtype = rng.choice(self._SUBTYPES)

        if subtype == "fan_out":
            return self._fan_out(rng, domain, nodes_desc)
        elif subtype == "fan_in":
            return self._fan_in(rng, domain, nodes_desc)
        elif subtype == "diamond":
            return self._diamond(rng, domain, nodes_desc)
        else:
            return self._mixed(rng, domain, nodes_desc)

    def _fan_out(self, rng, domain, nodes_desc) -> CausalExample:
        """Un nodo causa dos efectos distintos."""
        if len(nodes_desc) < 3:
            nodes_desc = DOMAIN_NODES["economia"]
        idxs = rng.sample(range(len(nodes_desc)), k=min(3, len(nodes_desc)))
        src_desc, b_desc, c_desc = nodes_desc[idxs[0]], nodes_desc[idxs[1]], nodes_desc[idxs[2]]

        g = CausalGraph()
        src = _node_from_desc(src_desc)
        b   = _node_from_desc(b_desc)
        c   = _node_from_desc(c_desc)
        g.add_node(src).add_node(b).add_node(c)

        rel_ab = rng.choice([CausalRelation.CAUSES, CausalRelation.ENABLES, CausalRelation.LEADS_TO])
        rel_ac = rng.choice([CausalRelation.CAUSES, CausalRelation.ENABLES, CausalRelation.LEADS_TO])
        g.add_edge(CausalEdge(src.node_id, b.node_id, rel_ab,
                              strength=round(rng.uniform(0.7, 1.0), 2)))
        g.add_edge(CausalEdge(src.node_id, c.node_id, rel_ac,
                              strength=round(rng.uniform(0.7, 1.0), 2)))

        problem = (
            f"Sabemos que '{src.label}' {_rel_text(rel_ab)} '{b.label}' "
            f"y que '{src.label}' {_rel_text(rel_ac)} '{c.label}'. "
            f"Pregunta: ¿Qué efectos directos tiene '{src.label}'?"
        )
        answer = (
            f"'{src.label}' tiene dos efectos directos: "
            f"(1) '{b.label}' y (2) '{c.label}'."
        )
        g.root_question = problem
        return CausalExample(
            problem_text=problem, graph=g, answer=answer,
            complexity_level=2, answer_type=AnswerType.BRANCHING,
            metadata={
                "domain": domain, "variant": "fan_out",
                "source_id": src.node_id,
                "expected_successor_count": 2,
                "expected_successor_ids": [b.node_id, c.node_id],
            },
        )

    def _fan_in(self, rng, domain, nodes_desc) -> CausalExample:
        """Dos causas distintas convergen en un efecto."""
        if len(nodes_desc) < 3:
            nodes_desc = DOMAIN_NODES["economia"]
        idxs = rng.sample(range(len(nodes_desc)), k=min(3, len(nodes_desc)))
        a_desc, b_desc, tgt_desc = nodes_desc[idxs[0]], nodes_desc[idxs[1]], nodes_desc[idxs[2]]

        g = CausalGraph()
        a   = _node_from_desc(a_desc)
        b   = _node_from_desc(b_desc)
        tgt = _node_from_desc(tgt_desc)
        g.add_node(a).add_node(b).add_node(tgt)

        rel_a = rng.choice([CausalRelation.CAUSES, CausalRelation.ENABLES])
        rel_b = rng.choice([CausalRelation.CAUSES, CausalRelation.ENABLES])
        g.add_edge(CausalEdge(a.node_id, tgt.node_id, rel_a,
                              strength=round(rng.uniform(0.6, 1.0), 2)))
        g.add_edge(CausalEdge(b.node_id, tgt.node_id, rel_b,
                              strength=round(rng.uniform(0.6, 1.0), 2)))

        problem = (
            f"Sabemos que '{a.label}' {_rel_text(rel_a)} '{tgt.label}' "
            f"y que '{b.label}' {_rel_text(rel_b)} '{tgt.label}'. "
            f"Pregunta: ¿Cuáles son las causas directas de '{tgt.label}'?"
        )
        answer = (
            f"'{tgt.label}' tiene dos causas directas: "
            f"(1) '{a.label}' y (2) '{b.label}'."
        )
        g.root_question = problem
        return CausalExample(
            problem_text=problem, graph=g, answer=answer,
            complexity_level=2, answer_type=AnswerType.DIRECT_CAUSE,
            metadata={
                "domain": domain, "variant": "fan_in",
                "target_id": tgt.node_id,
                "expected_predecessor_count": 2,
                "expected_predecessor_ids": [a.node_id, b.node_id],
            },
        )

    def _diamond(self, rng, domain, nodes_desc) -> CausalExample:
        """Diamante: A→B, A→C, B→D, C→D."""
        if len(nodes_desc) < 4:
            nodes_desc = DOMAIN_NODES["economia"]
        idxs = rng.sample(range(len(nodes_desc)), k=min(4, len(nodes_desc)))
        a_d, b_d, c_d, d_d = [nodes_desc[i] for i in idxs[:4]]

        g = CausalGraph()
        a, b, c, d = [_node_from_desc(x) for x in [a_d, b_d, c_d, d_d]]
        for n in [a, b, c, d]:
            g.add_node(n)

        rel1 = rng.choice([CausalRelation.CAUSES, CausalRelation.ENABLES])
        rel2 = rng.choice([CausalRelation.CAUSES, CausalRelation.ENABLES])
        g.add_edge(CausalEdge(a.node_id, b.node_id, rel1))
        g.add_edge(CausalEdge(a.node_id, c.node_id, rel1))
        g.add_edge(CausalEdge(b.node_id, d.node_id, rel2))
        g.add_edge(CausalEdge(c.node_id, d.node_id, rel2))

        problem = (
            f"'{a.label}' causa tanto '{b.label}' como '{c.label}'. "
            f"Además, tanto '{b.label}' como '{c.label}' causan '{d.label}'. "
            f"Pregunta: ¿'{a.label}' puede llevar a '{d.label}'?"
        )
        answer = (
            f"Sí. '{a.label}' lleva a '{d.label}' por dos caminos independientes: "
            f"(1) {a.label} → {b.label} → {d.label} "
            f"y (2) {a.label} → {c.label} → {d.label}."
        )
        g.root_question = problem
        return CausalExample(
            problem_text=problem, graph=g, answer=answer,
            complexity_level=2, answer_type=AnswerType.TRANSITIVITY,
            metadata={
                "domain": domain, "variant": "diamond",
                "source_id": a.node_id,
                "target_id": d.node_id,
                "expected_reachable": True,
                "n_paths": 2,
            },
        )

    def _mixed(self, rng, domain, nodes_desc) -> CausalExample:
        """Cadena con una bifurcación: A→B→D y A→C→D, 4-5 nodos."""
        if len(nodes_desc) < 4:
            nodes_desc = DOMAIN_NODES["tecnologia"]
        n = rng.randint(4, min(5, len(nodes_desc)))
        idxs = rng.sample(range(len(nodes_desc)), k=n)
        node_descs = [nodes_desc[i] for i in idxs]

        g = CausalGraph()
        ns = [_node_from_desc(d) for d in node_descs]
        for node in ns:
            g.add_node(node)

        # A→B, A→C, B→D (si hay 4)
        rels = [rng.choice([CausalRelation.CAUSES, CausalRelation.ENABLES, CausalRelation.LEADS_TO])
                for _ in range(3)]
        g.add_edge(CausalEdge(ns[0].node_id, ns[1].node_id, rels[0]))
        g.add_edge(CausalEdge(ns[0].node_id, ns[2].node_id, rels[1]))
        g.add_edge(CausalEdge(ns[1].node_id, ns[3].node_id, rels[2]))
        if n == 5:
            g.add_edge(CausalEdge(ns[2].node_id, ns[4].node_id, rels[0]))

        succs = {n.node_id for n in g.successors(ns[0].node_id)}
        edge_lines = [
            f"'{g.get_node(e.source_id).label}' {_rel_text(e.relation)} '{g.get_node(e.target_id).label}'"
            for e in g.edges
        ]
        problem = (
            f"Sistema causal: {'; '.join(edge_lines)}. "
            f"Pregunta: ¿Cuántos efectos directos tiene '{ns[0].label}'?"
        )
        answer = (
            f"'{ns[0].label}' tiene {len(succs)} efecto(s) directo(s): "
            f"{', '.join(repr(g.get_node(nid).label) for nid in sorted(succs))}."
        )
        g.root_question = problem
        return CausalExample(
            problem_text=problem, graph=g, answer=answer,
            complexity_level=2, answer_type=AnswerType.BRANCHING,
            metadata={
                "domain": domain, "variant": "mixed",
                "source_id": ns[0].node_id,
                "expected_successor_count": len(succs),
                "expected_successor_ids": list(succs),
            },
        )


# ─────────────────────────────────────────────────────────────────────────────
# GENERADOR NIVEL 3 — CONTRADICCIONES INTENCIONALES
# ─────────────────────────────────────────────────────────────────────────────

class _Level3Generator:
    """
    Nivel 3: Grafos con contradicciones intencionales.
    El modelo debe detectar cuándo el sistema causal es lógicamente inconsistente.
    ~70% de ejemplos tienen contradicción, ~30% son consistentes (negativos).
    """

    def generate(self, rng: random.Random, domain: Optional[str] = None) -> CausalExample:
        domain = domain or rng.choice(list(DOMAIN_NODES.keys()))
        has_contradiction = rng.random() < 0.70

        if has_contradiction:
            return self._with_contradiction(rng, domain)
        else:
            return self._without_contradiction(rng, domain)

    def _with_contradiction(self, rng, domain) -> CausalExample:
        nodes_desc = DOMAIN_NODES[domain]
        if len(nodes_desc) < 2:
            nodes_desc = DOMAIN_NODES["economia"]

        idxs = rng.sample(range(len(nodes_desc)), k=2)
        a_desc, b_desc = nodes_desc[idxs[0]], nodes_desc[idxs[1]]

        # Elegir un par contradictorio
        pair = rng.choice(CONTRADICTION_PAIRS)
        rel_positive, rel_negative = pair

        # Construir grafo base (puede tener otros nodos)
        n_extra = rng.randint(0, 2)
        extra_descs = []
        remaining = [i for i in range(len(nodes_desc)) if i not in idxs]
        if remaining and n_extra > 0:
            extra_idxs = rng.sample(remaining, k=min(n_extra, len(remaining)))
            extra_descs = [nodes_desc[i] for i in extra_idxs]

        g = CausalGraph()
        a = _node_from_desc(a_desc)
        b = _node_from_desc(b_desc)
        g.add_node(a).add_node(b)

        extras = []
        for ed in extra_descs:
            en = _node_from_desc(ed)
            g.add_node(en)
            extras.append(en)
            # Arista de contexto inocua
            rel_ctx = rng.choice([CausalRelation.REQUIRES, CausalRelation.PRECEDES,
                                  CausalRelation.CORRELATES])
            g.add_edge(CausalEdge(en.node_id, a.node_id, rel_ctx,
                                   strength=round(rng.uniform(0.5, 0.9), 2)))

        # La contradicción
        g.add_edge(CausalEdge(a.node_id, b.node_id, rel_positive, strength=0.9, confidence=0.85))
        g.add_edge(CausalEdge(a.node_id, b.node_id, rel_negative, strength=0.8, confidence=0.8))

        context_lines = [
            f"'{g.get_node(e.source_id).label}' {_rel_text(e.relation)} "
            f"'{g.get_node(e.target_id).label}'"
            for e in g.edges
        ]
        problem = (
            f"Un analista describe el siguiente sistema: {'; '.join(context_lines)}. "
            f"Pregunta: ¿Hay alguna contradicción lógica en este sistema causal?"
        )
        answer = (
            f"Sí, hay una contradicción: '{a.label}' no puede simultáneamente "
            f"'{rel_positive.value}' y '{rel_negative.value}' a '{b.label}'. "
            f"Estas dos relaciones son mutuamente excluyentes."
        )
        g.root_question = problem
        return CausalExample(
            problem_text=problem, graph=g, answer=answer,
            complexity_level=3, answer_type=AnswerType.CONTRADICTION,
            metadata={
                "domain": domain, "has_contradiction": True,
                "contradiction_source_id": a.node_id,
                "contradiction_target_id": b.node_id,
                "expected_has_contradiction": True,
                "expected_n_contradictions": 1,
            },
        )

    def _without_contradiction(self, rng, domain) -> CausalExample:
        nodes_desc = DOMAIN_NODES[domain]
        n = rng.randint(2, min(4, len(nodes_desc)))
        idxs = rng.sample(range(len(nodes_desc)), k=n)

        g = CausalGraph()
        ns = [_node_from_desc(nodes_desc[i]) for i in idxs]
        for node in ns:
            g.add_node(node)

        # Solo relaciones positivas no contradictorias
        safe_rels = [r for r in CausalRelation
                     if r.value in POSITIVE_RELATIONS or r == CausalRelation.PRECEDES]
        for i in range(len(ns) - 1):
            rel = rng.choice(safe_rels)
            g.add_edge(CausalEdge(ns[i].node_id, ns[i+1].node_id, rel,
                                   strength=round(rng.uniform(0.6, 1.0), 2)))

        context_lines = [
            f"'{g.get_node(e.source_id).label}' {_rel_text(e.relation)} "
            f"'{g.get_node(e.target_id).label}'"
            for e in g.edges
        ]
        problem = (
            f"Un analista describe: {'; '.join(context_lines)}. "
            f"Pregunta: ¿Hay alguna contradicción lógica en este sistema causal?"
        )
        answer = (
            "No, este sistema causal es lógicamente consistente. "
            "Todas las relaciones son compatibles entre sí."
        )
        g.root_question = problem
        return CausalExample(
            problem_text=problem, graph=g, answer=answer,
            complexity_level=3, answer_type=AnswerType.CONTRADICTION,
            metadata={
                "domain": domain, "has_contradiction": False,
                "expected_has_contradiction": False,
                "expected_n_contradictions": 0,
            },
        )


# ─────────────────────────────────────────────────────────────────────────────
# GENERADOR NIVEL 4 — RAZONAMIENTO CONTRAFACTUAL
# ─────────────────────────────────────────────────────────────────────────────

class _Level4Generator:
    """
    Nivel 4: Razonamiento contrafactual — "Si X no hubiera pasado, ¿qué ocurriría con Z?"
    El modelo aprende a simular intervenciones (do-calculus simplificado).

    Tres variantes:
      simple:    A→B→C, si no A, entonces no C (camino único)
      middle:    A→B→C, si no B, entonces no C (intervención en el medio)
      alternate: A→B→C y A→C directo: si no B, ¿C? (sí, via A→C)
    """

    def generate(self, rng: random.Random, domain: Optional[str] = None) -> CausalExample:
        domain = domain or rng.choice(list(DOMAIN_NODES.keys()))
        variant = rng.choice(["simple", "middle", "alternate"])

        if variant == "simple":
            return self._simple(rng, domain)
        elif variant == "middle":
            return self._middle(rng, domain)
        else:
            return self._alternate(rng, domain)

    def _simple(self, rng, domain) -> CausalExample:
        """A→B→C. Si no A, entonces no C (único camino)."""
        nodes_desc = DOMAIN_NODES[domain]
        if len(nodes_desc) < 3:
            nodes_desc = DOMAIN_NODES["salud"]
        idxs = rng.sample(range(len(nodes_desc)), k=3)
        a_d, b_d, c_d = [nodes_desc[i] for i in idxs]

        g = CausalGraph()
        a, b, c = [_node_from_desc(d) for d in [a_d, b_d, c_d]]
        g.add_node(a).add_node(b).add_node(c)
        rel1 = rng.choice([CausalRelation.CAUSES, CausalRelation.LEADS_TO])
        rel2 = rng.choice([CausalRelation.CAUSES, CausalRelation.ENABLES])
        g.add_edge(CausalEdge(a.node_id, b.node_id, rel1, strength=0.9))
        g.add_edge(CausalEdge(b.node_id, c.node_id, rel2, strength=0.85))

        problem = (
            f"Sabemos que '{a.label}' {_rel_text(rel1)} '{b.label}', "
            f"y '{b.label}' {_rel_text(rel2)} '{c.label}'. "
            f"Pregunta contrafactual: Si '{a.label}' NO hubiera ocurrido, "
            f"¿habría ocurrido '{c.label}'?"
        )
        answer = (
            f"No. Si '{a.label}' no hubiera ocurrido, '{b.label}' tampoco habría ocurrido "
            f"(ya que '{a.label}' es su única causa en este sistema). "
            f"Sin '{b.label}', '{c.label}' tampoco habría podido ocurrir. "
            f"La cadena '{a.label}' → '{b.label}' → '{c.label}' se habría roto desde el inicio."
        )
        g.root_question = problem
        return CausalExample(
            problem_text=problem, graph=g, answer=answer,
            complexity_level=4, answer_type=AnswerType.COUNTERFACTUAL,
            metadata={
                "domain": domain, "variant": "simple",
                "counterfactual_removed_id": a.node_id,
                "source_id": a.node_id,
                "target_id": c.node_id,
                "expected_path_blocked": True,
            },
        )

    def _middle(self, rng, domain) -> CausalExample:
        """A→B→C. Intervención en el medio: si no B (aunque A ocurra), ¿C?"""
        nodes_desc = DOMAIN_NODES[domain]
        if len(nodes_desc) < 3:
            nodes_desc = DOMAIN_NODES["tecnologia"]
        idxs = rng.sample(range(len(nodes_desc)), k=3)
        a_d, b_d, c_d = [nodes_desc[i] for i in idxs]

        g = CausalGraph()
        a, b, c = [_node_from_desc(d) for d in [a_d, b_d, c_d]]
        g.add_node(a).add_node(b).add_node(c)
        rel1 = CausalRelation.CAUSES
        rel2 = CausalRelation.LEADS_TO
        g.add_edge(CausalEdge(a.node_id, b.node_id, rel1, strength=0.9))
        g.add_edge(CausalEdge(b.node_id, c.node_id, rel2, strength=0.8))

        problem = (
            f"'{a.label}' causa '{b.label}', y '{b.label}' lleva a '{c.label}'. "
            f"Pregunta contrafactual: Asumamos que '{a.label}' sí ocurrió, pero "
            f"que '{b.label}' fue EVITADO por una intervención externa. "
            f"¿Habría ocurrido '{c.label}'?"
        )
        answer = (
            f"No. Aunque '{a.label}' ocurrió, al evitar '{b.label}' mediante intervención, "
            f"se rompe la cadena causal hacia '{c.label}'. "
            f"'{c.label}' no tiene otra fuente causal en este sistema, "
            f"por lo que no habría ocurrido."
        )
        g.root_question = problem
        return CausalExample(
            problem_text=problem, graph=g, answer=answer,
            complexity_level=4, answer_type=AnswerType.COUNTERFACTUAL,
            metadata={
                "domain": domain, "variant": "middle",
                "counterfactual_removed_id": b.node_id,
                "source_id": b.node_id,
                "target_id": c.node_id,
                "expected_path_blocked": True,
            },
        )

    def _alternate(self, rng, domain) -> CausalExample:
        """A→B→C y A→C. Si no B, ¿C? Sí, vía A→C (camino alternativo)."""
        nodes_desc = DOMAIN_NODES[domain]
        if len(nodes_desc) < 3:
            nodes_desc = DOMAIN_NODES["fisica"]
        idxs = rng.sample(range(len(nodes_desc)), k=3)
        a_d, b_d, c_d = [nodes_desc[i] for i in idxs]

        g = CausalGraph()
        a, b, c = [_node_from_desc(d) for d in [a_d, b_d, c_d]]
        g.add_node(a).add_node(b).add_node(c)
        g.add_edge(CausalEdge(a.node_id, b.node_id, CausalRelation.CAUSES, strength=0.9))
        g.add_edge(CausalEdge(b.node_id, c.node_id, CausalRelation.LEADS_TO, strength=0.8))
        g.add_edge(CausalEdge(a.node_id, c.node_id, CausalRelation.ENABLES, strength=0.7))

        problem = (
            f"'{a.label}' causa '{b.label}', '{b.label}' lleva a '{c.label}', "
            f"y además '{a.label}' también posibilita directamente '{c.label}'. "
            f"Pregunta contrafactual: Si '{b.label}' fuera eliminado del sistema, "
            f"¿podría aún ocurrir '{c.label}'?"
        )
        answer = (
            f"Sí, posiblemente. Aunque eliminar '{b.label}' bloquea el camino "
            f"'{a.label}' → '{b.label}' → '{c.label}', existe un camino alternativo: "
            f"'{a.label}' → '{c.label}' directamente. "
            f"Por lo tanto, '{c.label}' podría seguir ocurriendo si '{a.label}' está presente."
        )
        g.root_question = problem
        return CausalExample(
            problem_text=problem, graph=g, answer=answer,
            complexity_level=4, answer_type=AnswerType.COUNTERFACTUAL,
            metadata={
                "domain": domain, "variant": "alternate",
                "counterfactual_removed_id": b.node_id,
                "source_id": a.node_id,
                "target_id": c.node_id,
                "expected_path_blocked": False,   # ← hay camino alternativo
                "alternate_path_exists": True,
            },
        )


# ─────────────────────────────────────────────────────────────────────────────
# GENERADOR NIVEL 5 — MULTI-DOMINIO (8-15 NODOS, RELACIONES MIXTAS)
# ─────────────────────────────────────────────────────────────────────────────

class _Level5Generator:
    """
    Nivel 5: Grafos multi-dominio con 8-15 nodos y relaciones mixtas.
    Combina 2-3 dominios con conexiones cross-dominio.
    Preguntas sobre caminos críticos e inferencias de múltiples saltos.
    """

    # Pares de dominios que interactúan naturalmente
    _DOMAIN_COMBOS = [
        ["clima",    "economia"],
        ["economia", "social"],
        ["salud",    "social",  "economia"],
        ["fisica",   "tecnologia"],
        ["medioambiente", "clima", "social"],
        ["tecnologia", "economia", "social"],
    ]

    # Conexiones cross-dominio estructuradas
    # (dom1, idx1, dom2, idx2, relación)
    _CROSS = [
        ("clima",    2, "economia",       3, CausalRelation.LEADS_TO),
        ("clima",    3, "economia",       4, CausalRelation.ENABLES),
        ("economia", 4, "social",         0, CausalRelation.CAUSES),
        ("economia", 5, "social",         1, CausalRelation.CAUSES),
        ("social",   1, "salud",          1, CausalRelation.ENABLES),
        ("fisica",   2, "tecnologia",     2, CausalRelation.CAUSES),
        ("tecnologia",2,"economia",       3, CausalRelation.LEADS_TO),
        ("medioambiente",1,"clima",       0, CausalRelation.LEADS_TO),
        ("medioambiente",2,"social",      0, CausalRelation.ENABLES),
        ("social",   2, "economia",       5, CausalRelation.LEADS_TO),
        ("salud",    2, "social",         2, CausalRelation.LEADS_TO),
        ("economia", 4, "medioambiente",  2, CausalRelation.ENABLES),
    ]

    def generate(self, rng: random.Random, domain: Optional[str] = None) -> CausalExample:
        combo = rng.choice(self._DOMAIN_COMBOS)
        n_per_domain = rng.randint(3, 5)
        question_type = rng.choice(["critical_path", "multi_hop"])

        g = CausalGraph()
        domain_node_map: Dict[str, List[CausalNode]] = {}

        # Construir nodos de cada dominio
        for dom in combo:
            nodes_desc = DOMAIN_NODES[dom]
            max_n = min(n_per_domain, len(nodes_desc))
            n = rng.randint(max(2, max_n - 1), max_n)
            idxs = list(range(n))  # Usar primeros n nodos del dominio
            prefix = f"{dom[:3]}_"

            sub_g, nodes = _build_chain(dom, idxs, rng, prefix=prefix)
            domain_node_map[dom] = nodes
            for node in sub_g.nodes:
                g.add_node(node)
            for edge in sub_g.edges:
                if edge.source_id in g and edge.target_id in g:
                    g.add_edge(edge)

        # Añadir conexiones cross-dominio relevantes para este combo
        added_cross = 0
        rng.shuffle(list(range(len(self._CROSS))))  # mezclar orden
        for dom1, idx1, dom2, idx2, rel in self._CROSS:
            if dom1 not in combo or dom2 not in combo:
                continue
            nodes1 = domain_node_map.get(dom1, [])
            nodes2 = domain_node_map.get(dom2, [])
            if idx1 < len(nodes1) and idx2 < len(nodes2):
                src = nodes1[idx1]
                tgt = nodes2[idx2]
                if src.node_id != tgt.node_id and src.node_id in g and tgt.node_id in g:
                    try:
                        strength = round(rng.uniform(0.6, 0.95), 2)
                        g.add_edge(CausalEdge(src.node_id, tgt.node_id, rel,
                                              strength=strength, confidence=0.8))
                        added_cross += 1
                    except Exception:
                        pass

        # Garantizar al menos 8 nodos
        if len(g) < 8:
            # Añadir nodos extra de un dominio adicional
            extra_dom = rng.choice([d for d in DOMAIN_NODES if d not in combo])
            extra_nodes_desc = DOMAIN_NODES[extra_dom]
            prefix = f"{extra_dom[:3]}_"
            for i, nd in enumerate(extra_nodes_desc[:3]):
                en = _node_from_desc(nd, prefix=prefix)
                g.add_node(en)
                if i > 0:
                    prev_id = f"{prefix}{extra_nodes_desc[i-1][0]}"
                    if prev_id in g:
                        g.add_edge(CausalEdge(prev_id, en.node_id, CausalRelation.CAUSES))

        # Generar pregunta
        if question_type == "critical_path":
            return self._critical_path_question(g, combo, domain_node_map, rng)
        else:
            return self._multi_hop_question(g, combo, domain_node_map, rng)

    def _critical_path_question(self, g, combo, domain_node_map, rng) -> CausalExample:
        path = _longest_path(g)
        n_domains = len(combo)
        domains_str = " + ".join(combo)

        all_edges = [
            f"'{g.get_node(e.source_id).label}' {_rel_text(e.relation)} '{g.get_node(e.target_id).label}'"
            for e in g.edges
        ]
        # Truncar para no hacer el problema demasiado largo
        if len(all_edges) > 12:
            shown = all_edges[:12]
            omitted = len(all_edges) - 12
            context = "; ".join(shown) + f" (y {omitted} relaciones más)"
        else:
            context = "; ".join(all_edges)

        path_labels = " → ".join(g.get_node(nid).label for nid in path)
        summary = g.summary()

        problem = (
            f"Sistema multi-dominio ({domains_str}) con {len(g)} nodos y "
            f"{len(g.edges)} relaciones causales: {context}. "
            f"Pregunta: ¿Cuál es el camino causal más largo en este sistema? "
            f"¿Desde qué evento hasta qué consecuencia?"
        )
        answer = (
            f"El camino causal más largo tiene {len(path) - 1} eslabones: "
            f"{path_labels}. "
            f"Este camino atraviesa {n_domains} dominio(s) ({domains_str}), "
            f"mostrando cómo efectos en un área se propagan a otras."
        )
        g.root_question = problem
        return CausalExample(
            problem_text=problem, graph=g, answer=answer,
            complexity_level=5, answer_type=AnswerType.CRITICAL_PATH,
            metadata={
                "domains": combo,
                "expected_n_nodes": len(g),
                "expected_path": path,
                "expected_path_length": len(path) - 1,
                "n_cross_domain_edges": len([
                    e for e in g.edges
                    if e.source_id.split("_")[0] != e.target_id.split("_")[0]
                ]),
            },
        )

    def _multi_hop_question(self, g, combo, domain_node_map, rng) -> CausalExample:
        # Elegir dos nodos de dominios distintos
        if len(combo) < 2:
            return self._critical_path_question(g, combo, domain_node_map, rng)

        dom_a, dom_b = rng.sample(combo, k=2)
        nodes_a = domain_node_map.get(dom_a, [])
        nodes_b = domain_node_map.get(dom_b, [])
        if not nodes_a or not nodes_b:
            return self._critical_path_question(g, combo, domain_node_map, rng)

        src_node = nodes_a[0]
        tgt_node = nodes_b[-1]
        reachable = g.has_path(src_node.node_id, tgt_node.node_id)

        all_edges = [
            f"'{g.get_node(e.source_id).label}' {_rel_text(e.relation)} '{g.get_node(e.target_id).label}'"
            for e in g.edges
        ]
        if len(all_edges) > 12:
            context = "; ".join(all_edges[:12]) + f" (y {len(all_edges) - 12} más)"
        else:
            context = "; ".join(all_edges)

        problem = (
            f"Sistema multi-dominio ({', '.join(combo)}) — {len(g)} nodos: {context}. "
            f"Pregunta: ¿Puede '{src_node.label}' (dominio '{dom_a}') "
            f"provocar (directa o indirectamente) '{tgt_node.label}' (dominio '{dom_b}')?"
        )

        if reachable:
            answer = (
                f"Sí. Existe una cadena causal inter-dominio de '{src_node.label}' "
                f"('{dom_a}') hasta '{tgt_node.label}' ('{dom_b}'). "
                f"Los efectos en el dominio '{dom_a}' se propagan a través de conexiones "
                f"cross-dominio hasta '{dom_b}'."
            )
        else:
            answer = (
                f"No. No existe un camino causal directo o indirecto de '{src_node.label}' "
                f"('{dom_a}') hasta '{tgt_node.label}' ('{dom_b}') en este sistema. "
                f"Los dos dominios no están causalmente conectados en esta dirección."
            )
        g.root_question = problem
        return CausalExample(
            problem_text=problem, graph=g, answer=answer,
            complexity_level=5, answer_type=AnswerType.MULTI_HOP,
            metadata={
                "domains": combo,
                "expected_n_nodes": len(g),
                "source_id": src_node.node_id,
                "target_id": tgt_node.node_id,
                "source_domain": dom_a,
                "target_domain": dom_b,
                "expected_reachable": reachable,
            },
        )


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN DE VERIFICACIÓN
# ─────────────────────────────────────────────────────────────────────────────

def verify_example(example: CausalExample) -> VerificationResult:
    """
    Verifica que el CausalGraph y la respuesta del ejemplo son consistentes.

    Para cada AnswerType aplica la verificación programática correspondiente:
    - TRANSITIVITY / MULTI_HOP: comprueba has_path(source, target)
    - DIRECT_CAUSE:  comprueba predecessores directos del target
    - BRANCHING:     comprueba número de sucesores del source
    - CONTRADICTION: comprueba find_contradictions()
    - COUNTERFACTUAL: simula la eliminación del nodo y comprueba si el camino se rompe
    - CRITICAL_PATH:  comprueba que el grafo tenga n_nodes esperados y sin ciclos

    Devuelve VerificationResult con passed=True si todo es consistente.
    """
    g = example.graph
    meta = example.metadata
    atype = example.answer_type

    # ── Validez estructural básica (aplica a todos) ──────────────────────────
    struct_issues = []
    if len(g) == 0:
        struct_issues.append("grafo vacío")
    for edge in g.edges:
        if edge.source_idx < 0 or edge.target_idx < 0:
            struct_issues.append(f"arista {edge.edge_id} sin índices asignados")
        if edge.source_idx >= len(g) or edge.target_idx >= len(g):
            struct_issues.append(f"arista {edge.edge_id} con índice fuera de rango")
    if struct_issues:
        return VerificationResult(
            passed=False,
            reason=f"Problemas estructurales: {'; '.join(struct_issues)}",
            details={"structural_issues": struct_issues},
        )

    # ── TRANSITIVITY / MULTI_HOP ─────────────────────────────────────────────
    if atype in (AnswerType.TRANSITIVITY, AnswerType.MULTI_HOP):
        src_id  = meta.get("source_id")
        tgt_id  = meta.get("target_id")
        expected = meta.get("expected_reachable")
        if src_id is None or tgt_id is None or expected is None:
            return VerificationResult(
                passed=False,
                reason="Metadatos incompletos: falta source_id, target_id o expected_reachable",
                details=meta,
            )
        if src_id not in g:
            return VerificationResult(passed=False,
                reason=f"source_id {src_id!r} no existe en el grafo", details=meta)
        if tgt_id not in g:
            return VerificationResult(passed=False,
                reason=f"target_id {tgt_id!r} no existe en el grafo", details=meta)
        actual = g.has_path(src_id, tgt_id)
        ok = actual == expected
        return VerificationResult(
            passed=ok,
            reason=(f"has_path({src_id!r}, {tgt_id!r}) = {actual}, expected {expected}"
                    + ("" if ok else " ← FALLO")),
            details={"actual_reachable": actual, "expected_reachable": expected},
        )

    # ── DIRECT_CAUSE ─────────────────────────────────────────────────────────
    if atype == AnswerType.DIRECT_CAUSE:
        tgt_id   = meta.get("target_id")
        exp_preds = set(meta.get("expected_direct_causes", []))
        exp_succs = set(meta.get("expected_successor_ids", []))
        exp_count = meta.get("expected_predecessor_count") or meta.get("expected_successor_count")

        if tgt_id is not None and tgt_id in g:
            actual_preds = {g.get_node(e.source_id).node_id for e in g.in_edges(tgt_id)}
            if exp_preds:
                ok = actual_preds == exp_preds
                return VerificationResult(
                    passed=ok,
                    reason=(f"predecessors({tgt_id!r}) = {actual_preds}, expected {exp_preds}"
                            + ("" if ok else " ← FALLO")),
                    details={"actual": actual_preds, "expected": exp_preds},
                )
            if exp_count is not None:
                ok = len(actual_preds) == exp_count
                return VerificationResult(
                    passed=ok,
                    reason=f"len(predecessors) = {len(actual_preds)}, expected {exp_count}",
                    details={"actual_count": len(actual_preds), "expected_count": exp_count},
                )

        src_id = meta.get("source_id")
        if src_id is not None and src_id in g:
            actual_succs = {n.node_id for n in g.successors(src_id)}
            if exp_succs:
                ok = actual_succs == exp_succs
                return VerificationResult(
                    passed=ok,
                    reason=f"successors({src_id!r}) = {actual_succs}, expected {exp_succs}",
                    details={"actual": actual_succs, "expected": exp_succs},
                )
        return VerificationResult(passed=True, reason="DIRECT_CAUSE sin metadatos verificables",
                                  details=meta)

    # ── BRANCHING ─────────────────────────────────────────────────────────────
    if atype == AnswerType.BRANCHING:
        src_id    = meta.get("source_id")
        exp_count = meta.get("expected_successor_count")
        exp_ids   = set(meta.get("expected_successor_ids", []))

        if src_id is None or src_id not in g:
            return VerificationResult(passed=False,
                reason=f"source_id {src_id!r} no existe en el grafo", details=meta)

        actual_succs = {n.node_id for n in g.successors(src_id)}
        checks = []

        if exp_count is not None:
            count_ok = len(actual_succs) == exp_count
            checks.append(count_ok)

        if exp_ids:
            ids_ok = actual_succs == exp_ids
            checks.append(ids_ok)

        if not checks:
            return VerificationResult(passed=True, reason="BRANCHING sin metadatos de count/ids",
                                      details=meta)
        ok = all(checks)
        return VerificationResult(
            passed=ok,
            reason=(f"successors({src_id!r}) = {actual_succs} "
                    f"(count={len(actual_succs)}, expected={exp_count})"),
            details={"actual": actual_succs, "actual_count": len(actual_succs),
                     "expected_count": exp_count, "expected_ids": exp_ids},
        )

    # ── CONTRADICTION ─────────────────────────────────────────────────────────
    if atype == AnswerType.CONTRADICTION:
        expected_has  = meta.get("expected_has_contradiction")
        expected_n    = meta.get("expected_n_contradictions")

        contradictions = g.find_contradictions()
        actual_has = len(contradictions) > 0
        actual_n   = len(contradictions)

        if expected_has is not None:
            if actual_has != expected_has:
                return VerificationResult(
                    passed=False,
                    reason=(f"has_contradiction = {actual_has} (n={actual_n}), "
                            f"expected {expected_has}"),
                    details={"actual_n": actual_n, "expected_has": expected_has},
                )
        if expected_n is not None:
            if actual_n != expected_n:
                return VerificationResult(
                    passed=False,
                    reason=f"n_contradictions = {actual_n}, expected {expected_n}",
                    details={"actual_n": actual_n, "expected_n": expected_n},
                )
        return VerificationResult(
            passed=True,
            reason=f"CONTRADICTION OK: {actual_n} contradiccion(es), expected_has={expected_has}",
            details={"actual_n": actual_n},
        )

    # ── COUNTERFACTUAL ────────────────────────────────────────────────────────
    if atype == AnswerType.COUNTERFACTUAL:
        removed_id     = meta.get("counterfactual_removed_id")
        src_id         = meta.get("source_id")
        tgt_id         = meta.get("target_id")
        expected_blocked = meta.get("expected_path_blocked")

        if removed_id is None or tgt_id is None or expected_blocked is None:
            return VerificationResult(
                passed=False,
                reason="Metadatos incompletos para COUNTERFACTUAL",
                details=meta,
            )
        if removed_id not in g:
            return VerificationResult(passed=False,
                reason=f"counterfactual_removed_id {removed_id!r} no existe en el grafo",
                details=meta)
        if tgt_id not in g:
            return VerificationResult(passed=False,
                reason=f"target_id {tgt_id!r} no existe en el grafo", details=meta)

        # Simular eliminación sin modificar el grafo original
        g_cf = copy.deepcopy(g)
        g_cf.remove_node(removed_id)

        # Si el tgt también fue eliminado (era el mismo nodo), blocked = True trivialmente
        if tgt_id not in g_cf:
            actual_blocked = True
        else:
            # Buscar cualquier camino al target desde cualquier nodo no eliminado
            # (el src_id puede ser el removed, en ese caso blocked es True)
            if src_id and src_id not in g_cf:
                actual_blocked = True
            elif src_id and src_id in g_cf:
                actual_blocked = not g_cf.has_path(src_id, tgt_id)
            else:
                # Verificar si el target tiene algún predecesor
                actual_blocked = len(g_cf.in_edges(tgt_id)) == 0

        ok = actual_blocked == expected_blocked
        return VerificationResult(
            passed=ok,
            reason=(f"path_blocked_after_removing({removed_id!r}) = {actual_blocked}, "
                    f"expected {expected_blocked}" + ("" if ok else " ← FALLO")),
            details={"actual_blocked": actual_blocked, "expected_blocked": expected_blocked,
                     "removed_id": removed_id, "target_id": tgt_id},
        )

    # ── CRITICAL_PATH ─────────────────────────────────────────────────────────
    if atype == AnswerType.CRITICAL_PATH:
        expected_n   = meta.get("expected_n_nodes")
        expected_path = meta.get("expected_path", [])

        issues = []
        if expected_n is not None and len(g) != expected_n:
            issues.append(f"n_nodes={len(g)}, expected={expected_n}")

        cycles = g.detect_cycles()
        if cycles:
            issues.append(f"grafo tiene {len(cycles)} ciclo(s) inesperado(s)")

        if expected_path:
            actual_path = _longest_path(g)
            if len(actual_path) != len(expected_path):
                issues.append(
                    f"camino más largo tiene {len(actual_path)} nodos, "
                    f"expected {len(expected_path)}"
                )

        ok = len(issues) == 0
        return VerificationResult(
            passed=ok,
            reason=("CRITICAL_PATH OK" if ok else f"FALLO: {'; '.join(issues)}"),
            details={"n_nodes": len(g), "n_cycles": len(cycles),
                     "expected_n_nodes": expected_n},
        )

    # Default — tipo no reconocido
    return VerificationResult(
        passed=True,
        reason=f"AnswerType {atype!r} sin verificador específico — aceptado por defecto",
        details={},
    )


# ─────────────────────────────────────────────────────────────────────────────
# GENERADOR PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

class CausalGraphGenerator:
    """
    Generador principal de grafos causales para AION-C.

    Implementa el Generador B del plan FORGE-SYNTH:
    datos de razonamiento infinitos, verificables, generados en CPU.

    Compatible con CurriculumScheduler: complexity_level 1-5 se mapea
    directamente a los niveles del curriculum de FORGE.

    Uso:
        gen = CausalGraphGenerator(seed=42)

        # Un ejemplo
        ex = gen.generate(level=3)
        result = verify_example(ex)

        # Batch con distribución de niveles
        batch = gen.generate_batch(
            n=1000,
            level_distribution={1: 0.3, 2: 0.25, 3: 0.2, 4: 0.15, 5: 0.1}
        )

        # Stream infinito (para FORGE SID)
        for ex in gen.stream(level=2):
            train(ex)
    """

    def __init__(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)
        self._seed = seed
        self._generators = {
            1: _Level1Generator(),
            2: _Level2Generator(),
            3: _Level3Generator(),
            4: _Level4Generator(),
            5: _Level5Generator(),
        }
        self._counters = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

    def generate(
        self,
        level: int,
        domain: Optional[str] = None,
    ) -> CausalExample:
        """
        Genera un ejemplo de complejidad `level` (1-5).

        Args:
            level:  Nivel de complejidad 1-5.
            domain: Dominio semántico (opcional). Si None, aleatorio.

        Returns:
            CausalExample verificable con verify_example().
        """
        if level not in self._generators:
            raise ValueError(f"level debe ser 1-5, recibido {level}")
        gen = self._generators[level]
        ex = gen.generate(self._rng, domain=domain)
        self._counters[level] += 1
        # Computar entity_spans si no los tienen ya (los level generators no los añaden)
        if not ex.entity_spans:
            ex.entity_spans = compute_entity_spans(ex.problem_text, ex.graph)
        return ex

    def generate_batch(
        self,
        n: int,
        level_distribution: Optional[Dict[int, float]] = None,
        verify: bool = True,
    ) -> List[CausalExample]:
        """
        Genera un batch de n ejemplos según la distribución de niveles.

        Args:
            n:                  Número de ejemplos a generar.
            level_distribution: Dict {nivel: fracción}. Default: distribución uniforme.
            verify:             Si True, verifica cada ejemplo y filtra los fallidos.

        Returns:
            Lista de CausalExample (todos pasando verify_example si verify=True).
        """
        if level_distribution is None:
            level_distribution = {1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2, 5: 0.2}

        # Normalizar distribución
        total = sum(level_distribution.values())
        dist  = {k: v / total for k, v in level_distribution.items()}

        levels_list = list(dist.keys())
        weights     = [dist[l] for l in levels_list]

        examples: List[CausalExample] = []
        attempts = 0
        max_attempts = n * 5

        while len(examples) < n and attempts < max_attempts:
            level = self._rng.choices(levels_list, weights=weights, k=1)[0]
            ex = self.generate(level)
            if verify:
                result = verify_example(ex)
                if result.passed:
                    examples.append(ex)
            else:
                examples.append(ex)
            attempts += 1

        return examples[:n]

    def stream(self, level: int, domain: Optional[str] = None):
        """
        Generador infinito de ejemplos para el nivel dado.
        Para usar con FORGE SID (Synthetic Infinite Dataset).

        Usage:
            for example in gen.stream(level=2):
                train_step(example)
                if done: break
        """
        while True:
            yield self.generate(level, domain=domain)

    @property
    def stats(self) -> Dict:
        """Estadísticas de generación por nivel."""
        return {
            "total": sum(self._counters.values()),
            "by_level": dict(self._counters),
            "seed": self._seed,
        }


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS DE TEXTO
# ─────────────────────────────────────────────────────────────────────────────

_REL_TEXT: Dict[CausalRelation, str] = {
    CausalRelation.CAUSES:       "causa",
    CausalRelation.ENABLES:      "posibilita",
    CausalRelation.PREVENTS:     "impide",
    CausalRelation.LEADS_TO:     "lleva a",
    CausalRelation.IMPLIES:      "implica",
    CausalRelation.FOLLOWS_FROM: "se sigue de",
    CausalRelation.CONTRADICTS:  "contradice",
    CausalRelation.EQUIVALENT:   "es equivalente a",
    CausalRelation.SUPPORTS:     "apoya",
    CausalRelation.WEAKENS:      "debilita",
    CausalRelation.REQUIRES:     "requiere",
    CausalRelation.PRECEDES:     "precede a",
    CausalRelation.PART_OF:      "es parte de",
    CausalRelation.INSTANCE_OF:  "es instancia de",
    CausalRelation.CORRELATES:   "se correlaciona con",
    CausalRelation.ANALOGOUS_TO: "es análogo a",
}


def _rel_text(rel: CausalRelation) -> str:
    """Texto en español para una relación causal."""
    return _REL_TEXT.get(rel, rel.value)
