"""
synth/diverse_graph_gen.py — Generador de grafos VERDADERAMENTE diversos
=========================================================================

Los generadores existentes (CodeGraphGenerator, etc.) usan templates fijos:
  Code: 43 topologias unicas de 5000 intentos (ratio=0.009)
  Math: 11, Narrative: 9, Social: 10

Este modulo genera grafos ALEATORIOS con relaciones tipadas por dominio.
Cada grafo tiene topologia unica: n_nodes, n_edges, y conexiones aleatorias.

Cada dominio tiene sus propios node_types y relation_types del plan v4.
"""

from __future__ import annotations

import random
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# DOMAIN VOCABULARIES (from plan v4)
# ─────────────────────────────────────────────────────────────────────────────

DOMAINS = {
    "cora": {
        "node_types": ["entity", "event", "state", "action", "hypothesis", "fact", "question"],
        "relations": [
            "causes", "enables", "prevents", "inhibits", "implies",
            "contradicts", "supports", "refutes", "precedes", "follows",
            "concurrent", "part_of", "depends_on", "analogous_to",
        ],
        "entities_en": [
            "rain", "drought", "flood", "erosion", "fire", "storm", "earthquake",
            "inflation", "recession", "unemployment", "investment", "production",
            "demand", "supply", "debt", "growth", "pollution", "deforestation",
            "migration", "education", "poverty", "urbanization", "health",
            "innovation", "corruption", "inequality", "photosynthesis",
            "mutation", "evolution", "extinction", "adaptation", "metabolism",
            "stress", "motivation", "automation", "latency", "optimization",
        ],
        "entities_es": [
            "lluvia", "sequia", "inundacion", "erosion", "incendio", "tormenta",
            "inflacion", "recesion", "desempleo", "inversion", "produccion",
            "demanda", "oferta", "deuda", "crecimiento", "contaminacion",
            "migracion", "educacion", "pobreza", "urbanizacion", "salud",
            "innovacion", "corrupcion", "desigualdad", "fotosintesis",
            "mutacion", "evolucion", "extincion", "adaptacion", "metabolismo",
            "estres", "motivacion", "automatizacion", "latencia", "optimizacion",
        ],
        "q_templates_en": [
            "If {A} {rel} {B}, what happens to {C}?",
            "Does {A} {rel} {B}?",
            "What is the relationship between {A} and {B}?",
            "If {A} didn't occur, what would change?",
            "Is there a path from {A} to {B}?",
            "How many nodes does {A} affect?",
            "What causes {A}?",
            "What does {A} lead to?",
        ],
        "q_templates_es": [
            "Si {A} {rel} {B}, que pasa con {C}?",
            "{A} {rel} {B}?",
            "Cual es la relacion entre {A} y {B}?",
            "Si {A} no ocurriera, que cambiaria?",
            "Hay un camino de {A} a {B}?",
            "A cuantos nodos afecta {A}?",
            "Que causa {A}?",
            "A que lleva {A}?",
        ],
    },
    "forge_c": {
        "node_types": ["function", "class", "module", "variable", "expression", "error", "test", "config"],
        "relations": [
            "calls", "imports", "inherits", "mutates", "reads", "returns",
            "throws", "depends_on", "tests", "implements", "overrides", "data_flows_to",
        ],
        "entities_en": [
            "parse", "validate", "transform", "filter", "aggregate", "serialize",
            "compress", "encrypt", "cache", "route", "render", "compile",
            "optimize", "migrate", "backup", "authenticate", "authorize",
            "sanitize", "throttle", "retry", "deploy", "rollback", "monitor",
            "log", "hash", "sort", "merge", "split", "format", "connect",
            "disconnect", "subscribe", "publish", "fetch", "store",
        ],
        "entities_es": None,  # Code uses English names
        "q_templates_en": [
            "What does {A}() depend on?",
            "Who calls {A}()?",
            "If {A}() fails, what breaks?",
            "What is the execution order?",
            "Bug: {A}() returns null but {B}() expects a string. Fix?",
            "What functions need review after changing {A}()?",
            "Trace the data flow from {A}() to {B}().",
        ],
        "q_templates_es": [
            "De que depende {A}()?",
            "Quien llama a {A}()?",
            "Si {A}() falla, que se rompe?",
            "Cual es el orden de ejecucion?",
            "Bug: {A}() retorna null pero {B}() espera string. Solucion?",
            "Que funciones necesitan revision al cambiar {A}()?",
            "Traza el flujo de datos de {A}() a {B}().",
        ],
    },
    "axiom": {
        "node_types": ["axiom", "definition", "theorem", "lemma", "hypothesis", "expression", "equality", "set"],
        "relations": [
            "derives", "assumes", "contradicts", "generalizes", "specializes",
            "applies", "reduces_to", "bounds", "equivalent_to", "implies",
        ],
        "entities_en": None,  # Math uses generated expressions
        "entities_es": None,
        "q_templates_en": [
            "Solve: {expr}",
            "Calculate {A} {op} {B}",
            "Is {A} divisible by {B}?",
            "Simplify: {expr}",
            "What comes next in the series: {series}?",
            "Prove that {statement}",
            "What is {A} mod {B}?",
            "Find x if {equation}",
        ],
        "q_templates_es": [
            "Resuelve: {expr}",
            "Calcula {A} {op} {B}",
            "Es {A} divisible entre {B}?",
            "Simplifica: {expr}",
            "Que sigue en la serie: {series}?",
            "Demuestra que {statement}",
            "Cuanto es {A} mod {B}?",
            "Encuentra x si {equation}",
        ],
    },
    "muse": {
        "node_types": ["character", "event", "emotion", "theme", "symbol", "setting", "conflict", "resolution"],
        "relations": [
            "motivates", "conflicts_with", "develops_into", "symbolizes",
            "parallels", "contrasts", "foreshadows", "resolves", "intensifies", "subverts",
        ],
        "entities_en": [
            "hope", "betrayal", "redemption", "loss", "discovery", "sacrifice",
            "fear", "courage", "loneliness", "connection", "freedom", "duty",
            "innocence", "wisdom", "change", "memory", "truth", "illusion",
        ],
        "entities_es": [
            "esperanza", "traicion", "redencion", "perdida", "descubrimiento",
            "sacrificio", "miedo", "coraje", "soledad", "conexion", "libertad",
            "deber", "inocencia", "sabiduria", "cambio", "memoria", "verdad", "ilusion",
        ],
        "q_templates_en": [
            "Write a scene where {A} leads to {B}.",
            "Create a metaphor for {A}.",
            "What is the central theme?",
            "Generate a plot twist involving {A}.",
            "How does {A} contrast with {B}?",
            "Write a short story about {A}.",
        ],
        "q_templates_es": [
            "Escribe una escena donde {A} lleva a {B}.",
            "Crea una metafora para {A}.",
            "Cual es el tema central?",
            "Genera un plot twist con {A}.",
            "Como contrasta {A} con {B}?",
            "Escribe un cuento corto sobre {A}.",
        ],
    },
    "empathy": {
        "node_types": ["person", "intention", "belief", "emotion", "norm", "context", "relationship", "expectation"],
        "relations": [
            "wants", "believes", "feels", "expects", "violates_norm",
            "empathizes", "persuades", "trusts", "misunderstands", "reciprocates",
        ],
        "entities_en": [
            "Ana", "Carlos", "Sofia", "Diego", "Luna", "Pablo", "Marta",
            "Lucas", "Elena", "Javier", "Maria", "Roberto", "Carmen", "David",
        ],
        "entities_es": None,  # Names are the same
        "q_templates_en": [
            "How does {A} feel about the situation?",
            "What does {A} believe about {B}?",
            "Was a social norm violated?",
            "What should {A} do?",
            "How can {A} and {B} resolve their conflict?",
            "What does {A} expect from {B}?",
        ],
        "q_templates_es": [
            "Como se siente {A} sobre la situacion?",
            "Que cree {A} sobre {B}?",
            "Se violo alguna norma social?",
            "Que deberia hacer {A}?",
            "Como pueden {A} y {B} resolver su conflicto?",
            "Que espera {A} de {B}?",
        ],
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH-FIRST GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

class DiverseGraphGenerator:
    """Generates examples GRAPH-FIRST: random topology → text from graph."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self._seen: Set[str] = set()

    def _random_graph(
        self,
        domain: str,
        n_nodes: int,
        n_edges: int,
        lang: str,
    ) -> Tuple[List[Dict], List[Dict], List[str], str]:
        """Generate a random graph with domain-specific types.

        Returns: (nodes, edges, entity_names, topology_hash)
        """
        cfg = DOMAINS[domain]
        node_types = cfg["node_types"]
        relations = cfg["relations"]

        # Pick entity names
        if domain == "axiom":
            # Math: generate symbolic names
            entity_names = [f"x_{i}" for i in range(n_nodes)]
        elif domain == "forge_c":
            pool = cfg["entities_en"]
            entity_names = self.rng.sample(pool, min(n_nodes, len(pool)))
            while len(entity_names) < n_nodes:
                entity_names.append(f"func_{len(entity_names)}")
        elif lang == "es" and cfg.get("entities_es"):
            pool = cfg["entities_es"]
            entity_names = self.rng.sample(pool, min(n_nodes, len(pool)))
            while len(entity_names) < n_nodes:
                entity_names.append(pool[self.rng.randrange(len(pool))])
        elif cfg.get("entities_en"):
            pool = cfg["entities_en"]
            entity_names = self.rng.sample(pool, min(n_nodes, len(pool)))
            while len(entity_names) < n_nodes:
                entity_names.append(pool[self.rng.randrange(len(pool))])
        else:
            entity_names = [f"node_{i}" for i in range(n_nodes)]

        # Create nodes
        nodes = []
        for i, name in enumerate(entity_names):
            nodes.append({
                "id": f"n{i}",
                "label": name,
                "type": self.rng.choice(node_types),
            })

        # Create random edges (unique pairs, no self-loops)
        possible = [(i, j) for i in range(n_nodes) for j in range(n_nodes) if i != j]
        self.rng.shuffle(possible)
        actual_edges = min(n_edges, len(possible))

        edges = []
        for s, t in possible[:actual_edges]:
            edges.append({
                "source": f"n{s}",
                "target": f"n{t}",
                "relation": self.rng.choice(relations),
            })

        # Compute topology hash
        edge_sig = "|".join(f"{e['source']}-{e['relation']}-{e['target']}" for e in sorted(edges, key=lambda x: (x["source"], x["target"])))
        topo_hash = hashlib.md5(f"{n_nodes}:{edge_sig}".encode()).hexdigest()[:12]

        return nodes, edges, entity_names, topo_hash

    def generate(self, domain: str, lang: str = "en", difficulty: str = "medium") -> Optional[Dict]:
        """Generate one example with a unique graph topology."""
        cfg = DOMAINS[domain]

        # Difficulty controls graph size
        if difficulty == "easy":
            n_nodes = self.rng.randint(2, 3)
            n_edges = self.rng.randint(1, 3)
        elif difficulty == "medium":
            n_nodes = self.rng.randint(3, 5)
            n_edges = self.rng.randint(2, 6)
        else:  # hard
            n_nodes = self.rng.randint(4, 8)
            n_edges = self.rng.randint(4, 12)

        # Generate unique topology
        for _ in range(20):
            nodes, edges, names, topo_hash = self._random_graph(domain, n_nodes, n_edges, lang)
            if topo_hash not in self._seen:
                self._seen.add(topo_hash)
                break

        # Build question from graph
        q_templates = cfg["q_templates_es"] if lang == "es" else cfg["q_templates_en"]
        template = self.rng.choice(q_templates)

        A = names[0] if names else "X"
        B = names[1] if len(names) > 1 else "Y"
        C = names[2] if len(names) > 2 else "Z"

        # Special handling for math
        if domain == "axiom":
            return self._generate_math(lang, difficulty, nodes, edges)

        # Fill template
        rel = edges[0]["relation"] if edges else "relates to"
        question = template.format(A=A, B=B, C=C, rel=rel,
                                    expr=f"{A} + {B}", op="+",
                                    series="1, 2, 3", statement=f"{A} implies {B}",
                                    equation=f"{A} = {B}")

        # Build context from edges
        context_parts = []
        for e in edges:
            src_name = next((n["label"] for n in nodes if n["id"] == e["source"]), "?")
            tgt_name = next((n["label"] for n in nodes if n["id"] == e["target"]), "?")
            context_parts.append(f"{src_name} {e['relation']} {tgt_name}")
        context = ". ".join(context_parts) + "." if context_parts else ""

        # Generate answer based on graph structure
        answer = self._answer_from_graph(domain, nodes, edges, names, question, lang)

        return {
            "input": f"{context} {question}" if context else question,
            "output": answer,
            "domain": domain,
            "domain_id": {"cora": 0, "forge_c": 1, "axiom": 2, "muse": 3, "empathy": 4}.get(domain, 5),
            "graph": {"nodes": nodes, "edges": edges},
            "difficulty": difficulty,
            "language": lang,
        }

    def _generate_math(self, lang: str, difficulty: str, nodes, edges) -> Dict:
        """Generate a math example with a real answer."""
        if difficulty == "easy":
            a, b = self.rng.randint(1, 99), self.rng.randint(1, 99)
            op = self.rng.choice(["+", "-", "*"])
            result = a + b if op == "+" else a - b if op == "-" else a * b
            q = f"Calcula {a} {op} {b}" if lang == "es" else f"Calculate {a} {op} {b}"
            ans = str(result)
        elif difficulty == "medium":
            x = self.rng.randint(-20, 20)
            a = self.rng.randint(1, 12)
            b = self.rng.randint(-50, 50)
            c = a * x + b
            q = f"Resuelve: {a}x + {b} = {c}" if lang == "es" else f"Solve: {a}x + {b} = {c}"
            ans = f"x = {x}"
        else:
            n = self.rng.randint(10, 500)
            d = self.rng.choice([2, 3, 5, 7, 11, 13])
            q = f"Es {n} divisible entre {d}?" if lang == "es" else f"Is {n} divisible by {d}?"
            r = n % d
            if r == 0:
                ans = f"Si, {n}/{d} = {n // d}" if lang == "es" else f"Yes, {n}/{d} = {n // d}"
            else:
                ans = f"No, {n}/{d} = {n // d} resto {r}" if lang == "es" else f"No, {n}/{d} = {n // d} remainder {r}"

        return {
            "input": q, "output": ans,
            "domain": "axiom", "domain_id": 2,
            "graph": {"nodes": nodes, "edges": edges},
            "difficulty": difficulty, "language": lang,
        }

    def _answer_from_graph(self, domain, nodes, edges, names, question, lang) -> str:
        """Generate answer by analyzing the graph structure."""
        if not edges:
            return "No hay relaciones." if lang == "es" else "No relationships."

        # BFS for reachability
        adj = {}
        for e in edges:
            adj.setdefault(e["source"], []).append((e["target"], e["relation"]))

        q_lower = question.lower()

        # "What causes X" / "Que causa X"
        if "cause" in q_lower or "causa" in q_lower:
            target = names[0] if names else "?"
            causes = [
                next((n["label"] for n in nodes if n["id"] == e["source"]), "?")
                for e in edges
                if next((n["label"] for n in nodes if n["id"] == e["target"]), "") == target
            ]
            if causes:
                return ", ".join(causes) + f" {'causan' if lang == 'es' else 'cause'} {target}."
            return f"No hay causas directas de {target}." if lang == "es" else f"No direct causes of {target}."

        # "If X fails/didn't occur"
        if "fail" in q_lower or "falla" in q_lower or "didn't" in q_lower or "no ocurr" in q_lower:
            source = names[0]
            source_id = f"n0"
            affected = set()
            queue = [source_id]
            visited = set()
            while queue:
                curr = queue.pop(0)
                if curr in visited:
                    continue
                visited.add(curr)
                for tgt, rel in adj.get(curr, []):
                    name = next((n["label"] for n in nodes if n["id"] == tgt), tgt)
                    affected.add(name)
                    queue.append(tgt)
            if affected:
                prefix = "Se afectarian:" if lang == "es" else "Affected:"
                return f"{prefix} {', '.join(affected)}."
            return "No hay efectos directos." if lang == "es" else "No direct effects."

        # "Does X relate to Y" / yes-no
        if "?" in question and len(names) >= 2:
            src_id, tgt_id = "n0", "n1"
            direct = any(e["source"] == src_id and e["target"] == tgt_id for e in edges)
            if direct:
                rel = next(e["relation"] for e in edges if e["source"] == src_id and e["target"] == tgt_id)
                return f"Si, {names[0]} {rel} {names[1]}." if lang == "es" else f"Yes, {names[0]} {rel} {names[1]}."
            # Check indirect
            visited = set()
            queue = [src_id]
            while queue:
                curr = queue.pop(0)
                if curr == tgt_id:
                    return "Si, hay relacion indirecta." if lang == "es" else "Yes, indirect relationship."
                if curr in visited:
                    continue
                visited.add(curr)
                for tgt, _ in adj.get(curr, []):
                    queue.append(tgt)
            return "No hay relacion." if lang == "es" else "No relationship."

        # Default: describe the graph
        parts = []
        for e in edges[:3]:
            src = next((n["label"] for n in nodes if n["id"] == e["source"]), "?")
            tgt = next((n["label"] for n in nodes if n["id"] == e["target"]), "?")
            parts.append(f"{src} {e['relation']} {tgt}")
        return ". ".join(parts) + "."

    @property
    def n_unique(self) -> int:
        return len(self._seen)
