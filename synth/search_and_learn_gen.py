"""
synth/search_and_learn_gen.py — Ciclo completo: search → learn → respond.

Patrón enseñado:
    1. Usuario pregunta sobre algo no integrado.
    2. Modelo emite [TOOL: search_web] con query afinada.
    3. [RESULT:] trae info simulada.
    4. Modelo decide iniciar auto-learn con [TOOL: auto_learn].
    5. [RESULT:] confirma creación del adapter.
    6. Modelo responde con la info aprendida.

Esto entrena el ciclo de auto-mejora end-to-end: el modelo aprende a
decidir cuándo una información merece ser consolidada como adapter
permanente, no sólo citada una vez.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

from synth.canonical_format import CanonicalRecord, CanonicalTurn, format_record


# ════════════════════════════════════════════════════════════════════════════
# Content pools
# ════════════════════════════════════════════════════════════════════════════

_LEARNABLE_TOPICS_EN = [
    ("the PPO algorithm in reinforcement learning",
     "ppo reinforcement learning algorithm",
     "Proximal Policy Optimization by Schulman et al. 2017, a policy gradient method that uses a clipped surrogate objective for stability.",
     "PPO is a reinforcement learning algorithm from 2017 that improves on TRPO by using a simpler clipped objective. It's the default choice for many RL problems because it balances sample efficiency and simplicity.",
     "cora"),
    ("how Bloom filters work",
     "bloom filter data structure",
     "Bloom filter: probabilistic data structure by Burton Howard Bloom in 1970. Uses k hash functions and a bit array. False positives possible, false negatives impossible.",
     "A Bloom filter is a probabilistic set membership data structure. It uses k hash functions that map each element to positions in a bit array. To check membership, you test those positions. False positives are possible, but false negatives are not.",
     "forge_c"),
    ("the Raft consensus algorithm",
     "raft consensus algorithm",
     "Raft: consensus algorithm by Ongaro and Ousterhout 2014. Designed to be more understandable than Paxos. Leader election + log replication.",
     "Raft is a consensus algorithm from 2014 by Ongaro and Ousterhout, designed as a more understandable alternative to Paxos. It works by electing a leader who then replicates log entries to followers.",
     "cora"),
    ("the SQLite WAL mode",
     "sqlite wal write ahead log",
     "SQLite WAL mode: instead of modifying the db file directly, changes are appended to a write-ahead log. Readers don't block writers and vice versa.",
     "SQLite's WAL (Write-Ahead Logging) mode logs changes to a separate file instead of writing directly to the database. This allows concurrent readers and writers without blocking each other, at the cost of a checkpoint step to fold the log back.",
     "forge_c"),
    ("how gradient accumulation works in deep learning",
     "gradient accumulation deep learning",
     "Gradient accumulation: compute gradients on multiple small batches and sum them before calling optimizer.step(). Simulates a larger effective batch size.",
     "Gradient accumulation lets you simulate large batch training on limited memory: you compute gradients on several small batches and add them up before updating the weights. The effective batch size is then N times the physical batch size.",
     "forge_c"),
    ("the central limit theorem",
     "central limit theorem probability",
     "CLT: the distribution of sample means approaches normal as sample size grows, regardless of the original distribution.",
     "The central limit theorem states that the distribution of the mean of many independent identically distributed random variables approaches a normal distribution as the sample size grows, regardless of the shape of the original distribution.",
     "axiom"),
    ("how public-key cryptography works",
     "public key cryptography explained",
     "Public-key crypto: each user has a key pair (public + private). Anyone can encrypt with the public key, only the holder of the private key can decrypt.",
     "Public-key cryptography uses a pair of keys: the public key can be shared freely and is used to encrypt, while the private key stays with the owner and is used to decrypt. It also supports digital signatures: sign with private, verify with public.",
     "cora"),
    ("what a kernel is in an operating system",
     "operating system kernel",
     "Kernel: the core of an OS. Manages CPU, memory, I/O, system calls. Runs in privileged (kernel) mode, separate from user space.",
     "The kernel is the core of an operating system. It manages hardware resources (CPU scheduling, memory, devices) and provides system calls to user programs. It runs in a privileged CPU mode separated from user space.",
     "forge_c"),
    ("the Nash equilibrium in game theory",
     "nash equilibrium game theory",
     "Nash equilibrium: a solution concept where no player can improve their payoff by unilaterally changing strategy.",
     "A Nash equilibrium is a state in a game where no player can gain by changing their own strategy if the others keep theirs fixed. Named after John Nash, who proved existence for games with finite strategies.",
     "axiom"),
    ("the Kalman filter",
     "kalman filter explained",
     "Kalman filter: an algorithm for estimating the state of a linear dynamic system from noisy measurements. Invented by Rudolf Kalman 1960.",
     "A Kalman filter estimates the true state of a system that is partially observed through noisy measurements. It works iteratively: predict the next state, update the prediction with a new measurement, repeat. Widely used in navigation and control.",
     "cora"),
]

_LEARNABLE_TOPICS_ES = [
    ("el algoritmo de Dijkstra",
     "algoritmo dijkstra camino minimo",
     "Algoritmo de Dijkstra: publicado en 1959 por Edsger Dijkstra. Calcula el camino más corto desde un nodo origen a todos los otros en un grafo con pesos no negativos.",
     "El algoritmo de Dijkstra encuentra el camino más corto desde un nodo origen a todos los demás nodos en un grafo con pesos no negativos. Usa una cola de prioridad para procesar los nodos en orden de distancia creciente. Complejidad O((V+E) log V) con heap.",
     "forge_c"),
    ("cómo funciona una red neuronal básica",
     "red neuronal basica explicacion",
     "Red neuronal: capas de neuronas conectadas, cada una computa una suma ponderada + activación no lineal. Se entrena con backpropagation.",
     "Una red neuronal básica está formada por capas de neuronas. Cada neurona calcula una suma ponderada de sus entradas más un sesgo, y aplica una función de activación no lineal. Se entrena ajustando los pesos via backpropagation para minimizar una función de pérdida.",
     "forge_c"),
    ("el teorema de Bayes",
     "teorema de bayes probabilidad",
     "Teorema de Bayes: P(A|B) = P(B|A) * P(A) / P(B). Relaciona probabilidades condicionales.",
     "El teorema de Bayes es una fórmula fundamental de la probabilidad que expresa cómo actualizar una creencia dada nueva evidencia. La fórmula es P(A|B) = P(B|A) * P(A) / P(B). Es la base del razonamiento bayesiano y de muchos métodos en estadística y aprendizaje automático.",
     "axiom"),
    ("qué es un sistema distribuido",
     "sistema distribuido concepto",
     "Sistema distribuido: conjunto de computadoras independientes que coordinan vía red para aparecer como un solo sistema.",
     "Un sistema distribuido es un conjunto de computadoras independientes que cooperan a través de una red para ofrecer a sus usuarios la apariencia de un solo sistema coherente. Los desafíos clásicos incluyen consistencia, tolerancia a fallos, y latencia.",
     "cora"),
    ("cómo funciona la memoria virtual",
     "memoria virtual sistema operativo",
     "Memoria virtual: el SO da a cada proceso un espacio de direcciones propio mapeado a RAM física mediante tablas de páginas.",
     "La memoria virtual es una técnica del sistema operativo que da a cada proceso la ilusión de tener su propia memoria continua y grande, mapeando direcciones virtuales a direcciones físicas de RAM mediante tablas de páginas. Permite usar más memoria que la RAM física vía swap a disco.",
     "forge_c"),
    ("el algoritmo QuickSort",
     "algoritmo quicksort ordenamiento",
     "QuickSort: algoritmo de ordenamiento por divide y vencerás. Elige un pivote, particiona el array, y recursa. Complejidad promedio O(n log n).",
     "QuickSort es un algoritmo de ordenamiento por divide y vencerás. Elige un elemento pivote, particiona el array en elementos menores y mayores que el pivote, y aplica recursivamente el mismo proceso a cada mitad. Complejidad promedio O(n log n), peor caso O(n²) si el pivote es mal elegido.",
     "forge_c"),
    ("qué es la entropía en información",
     "entropia teoria de la informacion",
     "Entropía de Shannon: medida de la incertidumbre de una variable aleatoria. H(X) = -sum p(x) log p(x).",
     "La entropía en teoría de la información, definida por Shannon, mide la cantidad de incertidumbre o información de una variable aleatoria. La fórmula es H(X) = -sum p(x) log p(x). Un valor bajo indica predictibilidad, un valor alto indica aleatoriedad.",
     "axiom"),
    ("cómo funciona el DNS",
     "dns sistema de nombres dominio",
     "DNS: sistema de nombres de dominio, traduce nombres legibles como example.com a direcciones IP.",
     "El DNS (Sistema de Nombres de Dominio) es una base de datos distribuida jerárquica que traduce nombres de dominio legibles para humanos (como example.com) a direcciones IP numéricas. Cuando visitas una web, tu dispositivo consulta servidores DNS para obtener la IP del servidor real.",
     "cora"),
]


# ════════════════════════════════════════════════════════════════════════════
# Record builder
# ════════════════════════════════════════════════════════════════════════════

def _build_cycle_record(
    topic_title: str,
    search_query: str,
    search_result: str,
    final_answer: str,
    motor_target: str,
    language: str,
) -> CanonicalRecord:
    """Construye el ciclo search → learn → respond como multi-turn canonical."""
    if language == "en":
        user1 = f"Can you tell me about {topic_title}?"
        user2 = "yes, please learn it permanently so it's faster next time."
        aion1 = (
            f"I don't have {topic_title} integrated in my knowledge yet. Let me search and see what I find."
        )
        aion1_after_tool = (
            f"Based on the search: {search_result} Would you like me to run an auto-learn cycle so I keep this knowledge permanently?"
        )
        auto_learn_result = f"auto_learn completed: created adapter {motor_target}:{_slug(topic_title)} (rank=4, exam_pass_rate=1.0)"
        aion2 = (
            f"Done — the adapter is now attached to the {motor_target} motor and the canonical exam still passes at 100%. "
            f"To answer your original question: {final_answer}"
        )
    else:
        user1 = f"¿Me puedes contar sobre {topic_title}?"
        user2 = "sí, por favor apréndelo permanentemente para que la próxima sea más rápido."
        aion1 = (
            f"No tengo {topic_title} integrado en mi conocimiento todavía. Déjame buscarlo."
        )
        aion1_after_tool = (
            f"Según la búsqueda: {search_result} ¿Quieres que ejecute un ciclo de auto-learn para guardar este conocimiento permanentemente?"
        )
        auto_learn_result = f"auto_learn completado: adapter {motor_target}:{_slug(topic_title)} creado (rank=4, exam_pass_rate=1.0)"
        aion2 = (
            f"Listo — el adapter quedó enganchado al motor {motor_target} y el examen canónico sigue pasando al 100%. "
            f"Respondiendo a tu pregunta original: {final_answer}"
        )

    # Primer turno: search
    first_tool = json.dumps({"tool": "search_web", "query": search_query})
    # Segundo turno: auto-learn
    second_tool = json.dumps({"tool": "auto_learn", "concept": _slug(topic_title),
                              "motor": motor_target,
                              "content": search_result})

    # Construcción manual del texto porque es un ciclo con DOS tool calls.
    text_parts = [
        f"[USER: {user1}]",
        f"[TOOL: {first_tool}]",
        f"[RESULT: {search_result}]",
        f"[AION: {aion1_after_tool}]",
        f"[USER: {user2}]",
        f"[TOOL: {second_tool}]",
        f"[RESULT: {auto_learn_result}]",
        f"[AION: {aion2}]",
        "[EOS]",
    ]
    text = "\n".join(text_parts)

    return CanonicalRecord(
        text=text,
        has_skill=False,
        has_mem=False,
        has_tool=True,
        is_multi_turn=True,
        turn_count=2,
        domain="metacognitive",
        language=language,
        type="search_and_learn",
        metadata={
            "subcategory": "search_and_learn",
            "uses_tools": ["search_web", "auto_learn"],
            "target_motor": motor_target,
            "expected_motor_sequence": ["cora"],
        },
    )


def _slug(s: str) -> str:
    out = []
    for ch in s.lower():
        if ch.isalnum():
            out.append(ch)
        elif ch in " -_":
            out.append("_")
    slug = "".join(out)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")[:40]


def generate_all(target_total: int = 2000, seed: int = 2024) -> List[CanonicalRecord]:
    rng = random.Random(seed)
    pool_en = _LEARNABLE_TOPICS_EN
    pool_es = _LEARNABLE_TOPICS_ES
    out: List[CanonicalRecord] = []
    for i in range(target_total):
        if i % 2 == 0:
            topic_title, query, result, answer, motor = rng.choice(pool_en)
            lang = "en"
        else:
            topic_title, query, result, answer, motor = rng.choice(pool_es)
            lang = "es"
        out.append(_build_cycle_record(topic_title, query, result, answer, motor, lang))
    return out


def write_jsonl(records: List[CanonicalRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")


def main() -> None:  # pragma: no cover
    p = argparse.ArgumentParser()
    p.add_argument("--total", type=int, default=2000)
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--out", type=str, default="datasets/search_and_learn_2k.jsonl")
    args = p.parse_args()
    records = generate_all(target_total=args.total, seed=args.seed)
    out_path = Path(args.out)
    write_jsonl(records, out_path)
    by_lang = {}
    by_motor = {}
    for r in records:
        by_lang[r.language] = by_lang.get(r.language, 0) + 1
        m = r.metadata.get("target_motor", "?")
        by_motor[m] = by_motor.get(m, 0) + 1
    print(f"Generated {len(records)} search+learn records -> {out_path}")
    print(f"  by language: {by_lang}")
    print(f"  by target motor: {by_motor}")


if __name__ == "__main__":  # pragma: no cover
    main()
