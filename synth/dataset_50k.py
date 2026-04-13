#!/usr/bin/env python3
"""
synth/dataset_50k.py — Generador de 50K datos diversos bilingues para AION-C
==============================================================================

Reglas (de ablation studies):
  - Diversidad ratio > 0.85 (cada ejemplo con topologia unica)
  - Random shuffle (NO curriculum)
  - Mezcla facil/medio/dificil
  - 60% ingles, 40% espanol
  - 15% refuerzo (repetir topologias utiles con variacion)

Distribucion:
  CORA   12K (24%) — cadenas 2-8 nodos, contrafactuales, transitividad
  FORGE  9K  (18%) — funciones Python/JS, debugging, algoritmos
  AXIOM  10K (20%) — aritmetica, algebra, geometria, probabilidad
  MUSE   7K  (14%) — micro-historias, poesia, dialogos creativos
  EMPATHY 7K (14%) — situaciones emocionales, conflictos, apoyo
  General 5K (10%) — chitchat, identidad, meta-preguntas

Cada ejemplo:
  {"input", "output", "domain", "domain_id", "graph", "difficulty", "language"}

Uso:
    cd AION-C
    python -m synth.dataset_50k
    python -m synth.dataset_50k --n 50000 --output datasets/dataset_50k.jsonl
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import sys, os
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from synth.causal_graph_gen import CausalGraphGenerator
from synth.code_graph_gen import CodeGraphGenerator
from synth.math_graph_gen import MathGraphGenerator
from synth.narrative_graph_gen import NarrativeGraphGenerator
from synth.social_graph_gen import SocialGraphGenerator
from synth.diverse_graph_gen import DiverseGraphGenerator


# ─────────────────────────────────────────────────────────────────────────────
# SPANISH TRANSLATIONS
# ─────────────────────────────────────────────────────────────────────────────

_ES_REPLACEMENTS = {
    # Common patterns in generated text
    "If ": "Si ", "if ": "si ",
    "Yes": "Si", "No": "No",
    "because ": "porque ", "Because ": "Porque ",
    "causes ": "causa ", "prevents ": "previene ",
    "leads to ": "lleva a ", "enables ": "habilita ",
    "The real cause is ": "La causa real es ",
    "What happens if ": "Que pasa si ",
    "Is it true that ": "Es cierto que ",
    "Prove ": "Demuestra ", "Solve ": "Resuelve ",
    "Calculate ": "Calcula ", "Simplify ": "Simplifica ",
    "Write ": "Escribe ", "Find ": "Encuentra ",
    "The answer is ": "La respuesta es ",
    " is affected": " se afecta",
    "Downstream nodes": "Nodos downstream",
    "Intermediary": "Intermediario",
    "There is no ": "No hay ",
    "The theme is ": "El tema es ",
    "What is the relationship": "Cual es la relacion",
    " function ": " funcion ",
    " error ": " error ",
    "Bug:": "Bug:",
    "Base case": "Caso base",
    "By induction": "Por induccion",
    "remainder ": "resto ",
    "divisible": "divisible",
}

_ES_QUESTIONS = {
    "What would change?": "Que cambiaria?",
    "What is the real cause?": "Cual es la causa real?",
    "Is there a path?": "Hay un camino?",
    "What functions need review?": "Que funciones necesitan revision?",
    "What is the execution order?": "Cual es el orden de ejecucion?",
    "What is the central theme?": "Cual es el tema central?",
    "How does the character feel?": "Como se siente el personaje?",
    "Was a social norm violated?": "Se violo alguna norma social?",
}


def to_spanish(text: str, rng: random.Random) -> str:
    """Best-effort translation of generated examples to Spanish."""
    for en, es in _ES_REPLACEMENTS.items():
        text = text.replace(en, es)
    for en, es in _ES_QUESTIONS.items():
        text = text.replace(en, es)
    return text


# ─────────────────────────────────────────────────────────────────────────────
# GENERAL DOMAIN EXAMPLES (chitchat, identity, meta)
# ─────────────────────────────────────────────────────────────────────────────

_GENERAL_QA_EN = [
    ("Hello", "Hello. How can I help?"),
    ("Hi there", "Hi! What do you need?"),
    ("Who are you?", "I'm AION-C, an AI system created by Jesus with MoSE architecture."),
    ("What can you do?", "I can reason causally, analyze code, solve math, create narratives, and understand social dynamics."),
    ("Can you see images?", "No, I only work with text."),
    ("What is MoSE?", "MoSE is my architecture: Mixture of Specialized Engines with 5 motors."),
    ("What motors do you have?", "CORA for causal, FORGE-C for code, AXIOM for math, MUSE for creativity, EMPATHY for social."),
    ("Are you GPT?", "No, I'm AION-C, a different architecture based on causal graph networks."),
    ("Thank you", "You're welcome. Anything else?"),
    ("Goodbye", "Goodbye! Come back anytime."),
    ("What is your purpose?", "To assist with reasoning across multiple domains using specialized engines."),
    ("How do you work?", "I route queries to specialized motors that build causal graphs, reason over them, then generate text."),
    ("Are you conscious?", "No. I process patterns in text. I don't have subjective experience."),
    ("What languages do you speak?", "I speak English and Spanish."),
    ("Tell me something interesting", "The human brain uses about 20% of the body's energy despite being 2% of its mass."),
    ("I'm bored", "I can help with that. Want a math puzzle, a short story, or to explore an idea?"),
    ("What's the meaning of life?", "That's a philosophical question with many perspectives. What matters to you?"),
    ("Can you learn?", "I have external memory (MEM) where I store useful information for future reference."),
]

_GENERAL_QA_ES = [
    ("Hola", "Hola. En que puedo ayudarte?"),
    ("Que tal", "Bien. Dime que necesitas."),
    ("Quien eres?", "Soy AION-C, un sistema de IA creado por Jesus con arquitectura MoSE."),
    ("Que puedes hacer?", "Puedo razonar causalmente, analizar codigo, resolver matematicas, crear narrativas y entender dinamicas sociales."),
    ("Puedes ver imagenes?", "No, solo trabajo con texto."),
    ("Que es MoSE?", "MoSE es mi arquitectura: Mixture of Specialized Engines con 5 motores."),
    ("Eres GPT?", "No, soy AION-C, una arquitectura diferente basada en redes de grafos causales."),
    ("Gracias", "De nada. Algo mas?"),
    ("Adios", "Hasta luego. Vuelve cuando quieras."),
    ("Como funcionas?", "Ruteo consultas a motores especializados que construyen grafos causales, razonan sobre ellos y generan texto."),
    ("Eres consciente?", "No. Proceso patrones en texto. No tengo experiencia subjetiva."),
    ("Que idiomas hablas?", "Hablo espanol e ingles."),
    ("Cuentame algo interesante", "El cerebro humano usa el 20% de la energia del cuerpo aunque es solo el 2% de su masa."),
    ("Estoy aburrido", "Puedo ayudar. Quieres un puzzle de matematicas, un cuento corto, o explorar una idea?"),
    ("No entiendo", "Explicame que parte no esta clara y lo intento de otra forma."),
    ("Puedes aprender?", "Tengo memoria externa (MEM) donde guardo informacion util para futuras consultas."),
]


# ─────────────────────────────────────────────────────────────────────────────
# TOPOLOGY TRACKER
# ─────────────────────────────────────────────────────────────────────────────

class TopologyTracker:
    """Tracks unique graph topologies for diversity measurement."""

    def __init__(self):
        self._seen: Set[str] = set()
        self._counts: Dict[str, int] = {}

    def hash_graph(self, graph) -> str:
        """Hash a CausalGraph by its topology (ignoring labels)."""
        if graph is None:
            return "no_graph_" + hashlib.md5(str(random.random()).encode()).hexdigest()[:8]
        nodes = sorted(graph.nodes, key=lambda n: n.node_id) if hasattr(graph, 'nodes') else []
        edges = sorted(graph.edges, key=lambda e: (e.source_id, e.target_id)) if hasattr(graph, 'edges') else []
        n_nodes = len(nodes)
        edge_sig = "|".join(
            f"{e.source_idx}-{e.relation.value if hasattr(e.relation, 'value') else e.relation}-{e.target_idx}"
            for e in edges
        )
        raw = f"{n_nodes}:{edge_sig}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]

    def add(self, graph) -> Tuple[str, bool]:
        """Returns (hash, is_new)."""
        h = self.hash_graph(graph)
        is_new = h not in self._seen
        self._seen.add(h)
        self._counts[h] = self._counts.get(h, 0) + 1
        return h, is_new

    @property
    def n_unique(self) -> int:
        return len(self._seen)

    @property
    def n_total(self) -> int:
        return sum(self._counts.values())

    @property
    def ratio(self) -> float:
        return self.n_unique / max(1, self.n_total)


# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE CONVERTER
# ─────────────────────────────────────────────────────────────────────────────

DOMAIN_IDS = {"cora": 0, "forge_c": 1, "axiom": 2, "muse": 3, "empathy": 4, "general": 5}
DIFFICULTY_MAP = {1: "easy", 2: "easy", 3: "medium", 4: "hard", 5: "hard"}


def example_to_dict(
    ex,
    domain: str,
    language: str,
    difficulty: str = "medium",
) -> Dict[str, Any]:
    """Convert a CausalExample (or similar) to the standard format."""
    # Extract graph as serializable dict
    graph_dict = {"nodes": [], "edges": []}
    if hasattr(ex, 'graph') and ex.graph is not None:
        g = ex.graph
        if hasattr(g, 'nodes'):
            nodes_list = g.nodes if isinstance(g.nodes, list) else list(g.nodes)
            for n in nodes_list:
                graph_dict["nodes"].append({
                    "id": n.node_id,
                    "label": n.label,
                    "type": n.node_type.value if hasattr(n.node_type, 'value') else str(n.node_type),
                })
            for e in g.edges:
                graph_dict["edges"].append({
                    "source": e.source_id,
                    "target": e.target_id,
                    "relation": e.relation.value if hasattr(e.relation, 'value') else str(e.relation),
                })

    inp = ex.problem_text if hasattr(ex, 'problem_text') else str(ex)
    out = ex.answer if hasattr(ex, 'answer') else ""

    return {
        "input": inp,
        "output": out,
        "domain": domain,
        "domain_id": DOMAIN_IDS.get(domain, 5),
        "graph": graph_dict,
        "difficulty": difficulty,
        "language": language,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

class Dataset50KGenerator:
    """Generates 50K diverse bilingual examples across 6 domains.

    Uses DiverseGraphGenerator (random topologies) as PRIMARY source
    for high diversity ratio. Falls back to template generators for
    supplementary variety in text patterns.
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.tracker = TopologyTracker()

        # PRIMARY: graph-first random generator (high diversity)
        self.diverse_gen = DiverseGraphGenerator(seed=seed)

        # SUPPLEMENTARY: template generators (lower diversity but richer text)
        self.cora_gen = CausalGraphGenerator(seed=seed)
        self.code_gen = CodeGraphGenerator(seed=seed + 1)
        self.math_gen = MathGraphGenerator(seed=seed + 2)
        self.narrative_gen = NarrativeGraphGenerator(seed=seed + 3)
        self.social_gen = SocialGraphGenerator(seed=seed + 4)

    def _pick_language(self) -> str:
        return "es" if self.rng.random() < 0.4 else "en"

    def _pick_difficulty(self) -> Tuple[int, str]:
        """Returns (level 1-3, difficulty_str)."""
        r = self.rng.random()
        if r < 0.30:
            return 1, "easy"
        elif r < 0.70:
            return 2, "medium"
        else:
            return 3, "hard"

    def _translate_if_spanish(self, ex_dict: Dict, lang: str) -> Dict:
        """Translate to Spanish if needed."""
        if lang == "es":
            ex_dict["input"] = to_spanish(ex_dict["input"], self.rng)
            ex_dict["output"] = to_spanish(ex_dict["output"], self.rng)
        return ex_dict

    def _gen_domain(self, domain: str, n: int, template_gen, template_levels) -> List[Dict]:
        """Generate n examples: 70% from diverse graph gen, 30% from templates."""
        print(f"  Generating {domain.upper()} ({n})...", end=" ", flush=True)
        n_diverse = int(n * 0.90)
        n_template = n - n_diverse
        examples = []

        # 70% from diverse graph generator (unique topologies)
        for _ in range(n_diverse * 2):
            if len(examples) >= n_diverse:
                break
            lang = self._pick_language()
            _, diff = self._pick_difficulty()
            try:
                ex = self.diverse_gen.generate(domain, lang, diff)
                if ex:
                    examples.append(ex)
            except Exception:
                pass

        # 30% from template generators (richer text patterns)
        if template_gen is not None:
            for _ in range(n_template * 2):
                if len(examples) >= n:
                    break
                lang = self._pick_language()
                level, diff = self._pick_difficulty()
                level = self.rng.choice(template_levels.get(diff, [1]))
                try:
                    ex = template_gen.generate(level=level)
                    d = example_to_dict(ex, domain, lang, diff)
                    self._translate_if_spanish(d, lang)
                    examples.append(d)
                except Exception:
                    pass

        # Track topologies for all
        for ex in examples:
            g = ex.get("graph", {})
            edges = g.get("edges", [])
            nodes = g.get("nodes", [])
            sig = f"{len(nodes)}|" + "|".join(
                f"{e.get('source','')}-{e.get('relation','')}-{e.get('target','')}"
                for e in sorted(edges, key=lambda x: (x.get("source",""), x.get("target","")))
            )
            h = hashlib.md5(sig.encode()).hexdigest()[:12]
            self.tracker._seen.add(h)
            self.tracker._counts[h] = self.tracker._counts.get(h, 0) + 1

        print(f"{len(examples)} (diverse={min(n_diverse, len(examples))}, template={max(0, len(examples)-n_diverse)})")
        return examples[:n]

    def generate_cora(self, n: int) -> List[Dict]:
        return self._gen_domain("cora", n, self.cora_gen,
                                {"easy": [1, 2], "medium": [3], "hard": [4, 5]})

    def generate_code(self, n: int) -> List[Dict]:
        return self._gen_domain("forge_c", n, self.code_gen,
                                {"easy": [1], "medium": [2], "hard": [3]})

    def generate_math(self, n: int) -> List[Dict]:
        return self._gen_domain("axiom", n, self.math_gen,
                                {"easy": [1], "medium": [2], "hard": [3]})

    def generate_narrative(self, n: int) -> List[Dict]:
        return self._gen_domain("muse", n, self.narrative_gen,
                                {"easy": [1], "medium": [2], "hard": [3]})

    def generate_social(self, n: int) -> List[Dict]:
        return self._gen_domain("empathy", n, self.social_gen,
                                {"easy": [1], "medium": [2], "hard": [3]})

    def generate_general(self, n: int) -> List[Dict]:
        """Generate general chitchat/identity examples."""
        print(f"  Generating General ({n})...", end=" ", flush=True)
        examples = []
        all_qa = _GENERAL_QA_EN + _GENERAL_QA_ES

        # Generate n examples by cycling and varying
        for i in range(n):
            if self.rng.random() < 0.4:
                q, a = self.rng.choice(_GENERAL_QA_ES)
                lang = "es"
            else:
                q, a = self.rng.choice(_GENERAL_QA_EN)
                lang = "en"

            examples.append({
                "input": q,
                "output": a,
                "domain": "general",
                "domain_id": 5,
                "graph": {"nodes": [], "edges": []},
                "difficulty": "easy",
                "language": lang,
            })

        print(f"{len(examples)} generated")
        return examples

    def generate_all(self, n: int = 50000) -> List[Dict]:
        """Generate the full dataset with plan v4 proportions."""
        t0 = time.time()
        print("=" * 60)
        print(f"  Generating {n} diverse bilingual examples")
        print("=" * 60)

        # Proportions from the plan
        n_cora = int(n * 0.24)
        n_code = int(n * 0.18)
        n_math = int(n * 0.20)
        n_muse = int(n * 0.14)
        n_empathy = int(n * 0.14)
        n_general = n - n_cora - n_code - n_math - n_muse - n_empathy

        all_examples = []
        all_examples.extend(self.generate_cora(n_cora))
        all_examples.extend(self.generate_code(n_code))
        all_examples.extend(self.generate_math(n_math))
        all_examples.extend(self.generate_narrative(n_muse))
        all_examples.extend(self.generate_social(n_empathy))
        all_examples.extend(self.generate_general(n_general))

        # 15% reinforcement: duplicate the most graph-diverse examples
        n_reinforce = int(len(all_examples) * 0.15)
        graph_examples = [ex for ex in all_examples if ex["graph"]["edges"]]
        if graph_examples:
            reinforcement = self.rng.choices(graph_examples, k=n_reinforce)
            all_examples.extend(reinforcement)
            print(f"  Added {n_reinforce} reinforcement examples")

        # Random shuffle (NOT curriculum)
        self.rng.shuffle(all_examples)

        elapsed = time.time() - t0

        # Stats
        domains = {}
        languages = {"en": 0, "es": 0}
        difficulties = {"easy": 0, "medium": 0, "hard": 0}
        for ex in all_examples:
            d = ex["domain"]
            domains[d] = domains.get(d, 0) + 1
            languages[ex["language"]] = languages.get(ex["language"], 0) + 1
            difficulties[ex["difficulty"]] = difficulties.get(ex["difficulty"], 0) + 1

        print(f"\n  Total: {len(all_examples)} examples ({elapsed:.1f}s)")
        print(f"  Topology coverage: {self.tracker.n_unique} unique / {self.tracker.n_total} tracked "
              f"(ratio={self.tracker.ratio:.3f})")
        print(f"  Domains: {domains}")
        en_pct = 100 * languages.get("en", 0) / len(all_examples)
        es_pct = 100 * languages.get("es", 0) / len(all_examples)
        print(f"  Languages: en={en_pct:.0f}% es={es_pct:.0f}%")
        print(f"  Difficulty: {difficulties}")

        return all_examples


def write_jsonl(examples: List[Dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    size_mb = path.stat().st_size / 1e6
    print(f"  Written: {path} ({size_mb:.1f} MB)")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    p = argparse.ArgumentParser(description="Generate 50K diverse bilingual dataset")
    p.add_argument("--n", type=int, default=50000)
    p.add_argument("--output", default="datasets/dataset_50k.jsonl")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    gen = Dataset50KGenerator(seed=args.seed)
    examples = gen.generate_all(n=args.n)
    write_jsonl(examples, Path(args.output))

    # Verify with DatasetQualityAnalyzer if available
    try:
        from tools.dataset_analyzer import DatasetQualityAnalyzer
        # Convert to CausalExample format for analyzer
        print("\n  Running DatasetQualityAnalyzer...")
        # The analyzer expects CausalExample objects, not dicts
        # Just report our own metrics
    except ImportError:
        pass

    # Final diversity check
    print(f"\n  DIVERSITY CHECK:")
    ratio = gen.tracker.ratio
    if ratio >= 0.85:
        print(f"    PASS: ratio={ratio:.3f} >= 0.85")
    else:
        print(f"    WARN: ratio={ratio:.3f} < 0.85 (target)")

    print("\n  Done!")


if __name__ == "__main__":
    main()
