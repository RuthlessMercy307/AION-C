"""
experiments/cora_ablation.py — Ablación del CRE
================================================

Responde dos preguntas existenciales de CORA con datos del CausalGraphGenerator:

  EXPERIMENTO 1 — ¿El CRE converge con más iteraciones?
    • 100 CausalExamples (niveles 1-3)
    • Node features random pero consistentes (hash del node_id como semilla)
    • CRE con 1..20 iteraciones; mide delta = ||h_i - h_{i-1}||
    • Reporta: ¿delta decrece? ¿explosión? ¿colapso?
    • Gráfica ASCII del delta medio vs iteración

  EXPERIMENTO 2 — ¿Weight sharing es estable a 50 iteraciones?
    • Mismos datos, max_iterations=50
    • Mide norma media de node features en cada iteración
    • Mide cosine similarity entre pares en iter 1 vs iter 50
    • Reporta: ¿explosión? ¿over-smoothing?

Uso:
    python -m experiments.cora_ablation                  # 100 ejemplos, config default
    python -m experiments.cora_ablation --n 20 --dim 32  # rápido para desarrollo
    python -m experiments.cora_ablation --out results.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── Ruta del proyecto ────────────────────────────────────────────────────────
_PROJECT_ROOT = str(Path(__file__).parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
import torch.nn.functional as F

from synth.causal_graph_gen import CausalGraphGenerator, CausalExample
from cre.engine import CausalReasoningEngine, CREOutput
from cre.config import CREConfig
from core.graph import CausalGraph


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN DEL EXPERIMENTO
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CORA CRE Ablation Study")
    p.add_argument("--n",    type=int,   default=100,   help="Número de ejemplos")
    p.add_argument("--dim",  type=int,   default=64,    help="Dimensión de node features")
    p.add_argument("--seed", type=int,   default=42,    help="Semilla del generador")
    p.add_argument("--out",  type=str,   default="experiments/ablation_results.json",
                   help="Archivo JSON de salida")
    p.add_argument("--quiet", action="store_true", help="Reducir output (sin gráfica)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def make_node_features(graph: CausalGraph, dim: int) -> torch.Tensor:
    """
    Crea node features random pero CONSISTENTES: mismo graph → mismo tensor.
    Usa el hash de node_id como semilla del generador PyTorch.
    Normaliza a norma unitaria para comparaciones estables.
    """
    if len(graph.nodes) == 0:
        return torch.zeros(0, dim)
    feats = []
    for node in graph.nodes:
        # hash positivo y acotado a uint32
        seed = abs(hash(node.node_id)) % (2 ** 31)
        rng = torch.Generator()
        rng.manual_seed(seed)
        f = torch.randn(dim, generator=rng)
        f = F.normalize(f, dim=0)   # norma unitaria inicial
        feats.append(f)
    return torch.stack(feats, dim=0)   # [N, dim]


def get_per_iteration_snapshots(
    engine:        CausalReasoningEngine,
    graph:         CausalGraph,
    node_features: torch.Tensor,
    n_iterations:  int,
) -> List[torch.Tensor]:
    """
    Ejecuta el CRE con return_history=True y extrae el snapshot
    al FINAL de cada iteración completa (tras todas las capas de esa iteración).

    Returns:
        Lista de n_iterations tensores [N, D], uno por iteración.
        El tensor en posición i corresponde al estado tras la iteración i+1.
    """
    with torch.no_grad():
        output = engine(
            graph          = graph,
            node_features  = node_features,
            n_iterations   = n_iterations,
            return_history = True,
        )

    n_layers = engine.config.n_message_layers
    history  = output.layer_outputs    # len = n_iterations * n_layers

    # history[n_layers*i + (n_layers-1)] = estado al final de la iteración i
    snapshots = [
        history[(i + 1) * n_layers - 1]
        for i in range(n_iterations)
        if (i + 1) * n_layers - 1 < len(history)
    ]
    return snapshots


def mean_pairwise_cosine(h: torch.Tensor) -> float:
    """
    Media de la cosine similarity entre todos los pares de nodos.
    h: [N, D]. Devuelve nan si N < 2.
    """
    N = h.shape[0]
    if N < 2:
        return float("nan")
    h_n = F.normalize(h.float(), dim=-1)   # [N, D]
    sim = h_n @ h_n.T                      # [N, N]
    idx = torch.triu_indices(N, N, offset=1)
    return float(sim[idx[0], idx[1]].mean().item())


def is_monotone_decrease(values: List[float], tol: float = 1e-4) -> bool:
    """True si la secuencia es no-creciente (con tolerancia tol)."""
    return all(values[i] >= values[i + 1] - tol for i in range(len(values) - 1))


# ─────────────────────────────────────────────────────────────────────────────
# CARGA DE DATOS
# ─────────────────────────────────────────────────────────────────────────────

def load_examples(n: int, seed: int) -> List[CausalExample]:
    """
    Genera n CausalExamples de niveles 1-3.
    Filtra grafos vacíos o de un solo nodo (insuficientes para MP).
    """
    gen = CausalGraphGenerator(seed=seed)
    dist = {1: 0.40, 2: 0.40, 3: 0.20}
    raw  = gen.generate_batch(n * 2, level_distribution=dist, verify=False)
    # Filtrar: necesitamos ≥ 2 nodos para que el message passing sea interesante
    usable = [ex for ex in raw if len(ex.graph.nodes) >= 2]
    return usable[:n]


# ─────────────────────────────────────────────────────────────────────────────
# FORMATEO DE TABLAS
# ─────────────────────────────────────────────────────────────────────────────

def _bar(value: float, max_val: float, width: int = 30) -> str:
    """Barra ASCII proporcional al valor."""
    if max_val <= 0 or not math.isfinite(value):
        return "░" * width
    frac  = min(value / max_val, 1.0)
    filled = int(round(frac * width))
    return "█" * filled + "░" * (width - filled)


def print_section(title: str) -> None:
    width = 70
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def print_table_header(cols: List[Tuple[str, int]]) -> None:
    header = " │ ".join(f"{name:>{w}}" for name, w in cols)
    sep    = "─┼─".join("─" * w for _, w in cols)
    print(f"  {header}")
    print(f"  {sep}")


def print_table_row(cols: List[Tuple[str, int]], values: List) -> None:
    row = " │ ".join(f"{str(v):>{w}}" for (_, w), v in zip(cols, values))
    print(f"  {row}")


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENTO 1 — Convergencia del CRE
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment_1(
    examples:  List[CausalExample],
    node_dim:  int,
    n_iter:    int = 20,
    quiet:     bool = False,
) -> Dict:
    """
    Para cada ejemplo:
      • Crea node features consistentes
      • Extrae snapshots por iteración (1..n_iter)
      • Calcula delta_i = ||h_i - h_{i-1}||_F / N

    Devuelve un dict con estadísticas por iteración y resumen global.
    """
    cre_cfg = CREConfig(
        node_dim         = node_dim,
        edge_dim         = max(node_dim // 2, 16),
        message_dim      = node_dim,
        n_message_layers = 2,
        max_iterations   = n_iter,
    )
    engine = CausalReasoningEngine(cre_cfg)
    engine.eval()

    # ── Recolectar datos por iteración ────────────────────────────────────────
    # all_deltas[i] = lista de deltas en la iteración i+1 (uno por ejemplo)
    all_deltas: List[List[float]] = [[] for _ in range(n_iter)]
    all_norms:  List[List[float]] = [[] for _ in range(n_iter)]  # norma en iter i
    monotone_flags: List[bool]    = []
    init_norm_list: List[float]   = []

    for ex in examples:
        graph = ex.graph
        N     = len(graph.nodes)
        h0    = make_node_features(graph, node_dim)   # [N, D]

        snaps = get_per_iteration_snapshots(engine, graph, h0, n_iter)
        if len(snaps) < n_iter:
            continue    # grafo sin aristas → saltar (sería trivial)

        init_norm_list.append(float(h0.norm(dim=-1).mean().item()))
        h_prev = h0
        deltas_this: List[float] = []

        for i, h_i in enumerate(snaps):
            # Delta = norma Frobenius del cambio, normalizado por N
            delta = float((h_i - h_prev).norm().item()) / N
            norm  = float(h_i.norm(dim=-1).mean().item())

            all_deltas[i].append(delta)
            all_norms[i].append(norm)
            deltas_this.append(delta)
            h_prev = h_i

        monotone_flags.append(is_monotone_decrease(deltas_this))

    # ── Estadísticas por iteración ────────────────────────────────────────────
    iter_stats = []
    for i in range(n_iter):
        deltas = all_deltas[i]
        norms  = all_norms[i]
        if not deltas:
            continue
        t = torch.tensor(deltas)
        n = torch.tensor(norms)
        iter_stats.append({
            "iteration":   i + 1,
            "mean_delta":  float(t.mean()),
            "std_delta":   float(t.std()),
            "min_delta":   float(t.min()),
            "max_delta":   float(t.max()),
            "mean_norm":   float(n.mean()),
            "exploded":    int((n > 1000).sum()),   # norma muy grande
            "collapsed":   int((n < 0.01).sum()),   # colapso a cero
        })

    # ── Resumen global ────────────────────────────────────────────────────────
    n_valid   = len(monotone_flags)
    n_mono    = sum(monotone_flags)
    init_norm = float(torch.tensor(init_norm_list).mean()) if init_norm_list else 0.0

    explosion_cnt = sum(s["exploded"] for s in iter_stats)
    collapse_cnt  = sum(s["collapsed"] for s in iter_stats)

    summary = {
        "n_examples":         n_valid,
        "n_iterations":       n_iter,
        "node_dim":           node_dim,
        "init_norm_mean":     init_norm,
        "monotone_count":     n_mono,
        "monotone_fraction":  n_mono / max(n_valid, 1),
        "explosion_cells":    explosion_cnt,
        "collapse_cells":     collapse_cnt,
    }

    # ── Imprimir ──────────────────────────────────────────────────────────────
    if not quiet:
        print_section("EXPERIMENTO 1 — Convergencia del CRE")
        print(f"  Datos:  {n_valid} grafos válidos, niveles 1-3")
        print(f"  Config: node_dim={node_dim}, n_layers=2 (weight shared), {n_iter} iteraciones")
        print(f"  Norma inicial media: {init_norm:.4f}")
        print()

        cols = [
            ("Iter", 4), ("Delta μ", 9), ("Delta σ", 9),
            ("Delta min", 9), ("Delta max", 9), ("Norma μ", 9),
            ("Expl.", 5), ("Colap.", 5),
        ]
        print_table_header(cols)

        # Mostrar todas las iteraciones
        for s in iter_stats:
            print_table_row(cols, [
                s["iteration"],
                f"{s['mean_delta']:.5f}",
                f"{s['std_delta']:.5f}",
                f"{s['min_delta']:.5f}",
                f"{s['max_delta']:.5f}",
                f"{s['mean_norm']:.4f}",
                s["exploded"],
                s["collapsed"],
            ])

        # Gráfica ASCII: delta medio vs iteración
        print()
        print("  Delta medio por iteración (gráfica ASCII):")
        print()
        mean_deltas = [s["mean_delta"] for s in iter_stats]
        max_delta   = max(mean_deltas) if mean_deltas else 1.0
        for s in iter_stats:
            bar = _bar(s["mean_delta"], max_delta, width=35)
            print(f"  iter {s['iteration']:>2} │{bar}│ {s['mean_delta']:.5f}")

        # Resumen
        print()
        print("  RESUMEN:")
        pct_mono = summary["monotone_fraction"] * 100
        print(f"  • Delta decrece monótonamente en {n_mono}/{n_valid} "
              f"ejemplos ({pct_mono:.1f}%)")
        if explosion_cnt > 0:
            print(f"  ⚠  Explosión detectada (norma>1000): {explosion_cnt} celdas")
        else:
            print(f"  ✓  Sin explosión detectada (norma nunca > 1000)")
        if collapse_cnt > 0:
            print(f"  ⚠  Colapso detectado (norma<0.01):   {collapse_cnt} celdas")
        else:
            print(f"  ✓  Sin colapso detectado (norma nunca < 0.01)")

        final_norm = iter_stats[-1]["mean_norm"] if iter_stats else 0.0
        ratio = final_norm / init_norm if init_norm > 0 else float("nan")
        print(f"  • Norma final / inicial: {final_norm:.4f} / {init_norm:.4f} "
              f"= {ratio:.3f}x")

    return {"iter_stats": iter_stats, "summary": summary}


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENTO 2 — Estabilidad con weight sharing a 50 iteraciones
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment_2(
    examples: List[CausalExample],
    node_dim: int,
    n_iter:   int = 50,
    quiet:    bool = False,
) -> Dict:
    """
    Mide norma y cosine similarity de los node features en cada iteración hasta 50.
    Detecta over-smoothing (todos los nodos se vuelven iguales) y explosión.
    """
    cre_cfg = CREConfig(
        node_dim         = node_dim,
        edge_dim         = max(node_dim // 2, 16),
        message_dim      = node_dim,
        n_message_layers = 2,
        max_iterations   = n_iter,
    )
    engine = CausalReasoningEngine(cre_cfg)
    engine.eval()

    # Checkpoints para reporting
    CHECKPOINTS = [1, 5, 10, 20, 30, 40, 50]
    checkpoint_set = set(CHECKPOINTS)

    # all_norms[i] = norma media en iteración i+1, lista sobre ejemplos
    all_norms:   List[List[float]] = [[] for _ in range(n_iter)]
    all_cosines: List[List[float]] = [[] for _ in range(n_iter)]

    for ex in examples:
        graph = ex.graph
        N     = len(graph.nodes)
        h0    = make_node_features(graph, node_dim)

        snaps = get_per_iteration_snapshots(engine, graph, h0, n_iter)
        if len(snaps) < n_iter:
            continue

        for i, h_i in enumerate(snaps):
            norm = float(h_i.norm(dim=-1).mean().item())
            cos  = mean_pairwise_cosine(h_i)
            all_norms[i].append(norm)
            if not math.isnan(cos):
                all_cosines[i].append(cos)

    # ── Estadísticas en checkpoints ───────────────────────────────────────────
    ckpt_stats = []
    for i in range(n_iter):
        if (i + 1) not in checkpoint_set:
            continue
        norms   = torch.tensor(all_norms[i]) if all_norms[i] else torch.zeros(1)
        cosines = torch.tensor(all_cosines[i]) if all_cosines[i] else torch.zeros(1)
        ckpt_stats.append({
            "iteration":    i + 1,
            "mean_norm":    float(norms.mean()),
            "std_norm":     float(norms.std()),
            "max_norm":     float(norms.max()),
            "mean_cosine":  float(cosines.mean()),
            "std_cosine":   float(cosines.std()),
            "exploded":     int((norms > 1000).sum()),
            "smoothed":     int((cosines > 0.999).sum()),  # over-smoothing
        })

    # ── Resumen: comparación iter 1 vs iter 50 ────────────────────────────────
    norm_iter1  = float(torch.tensor(all_norms[0]).mean())  if all_norms[0]  else 0.0
    norm_iter50 = float(torch.tensor(all_norms[-1]).mean()) if all_norms[-1] else 0.0
    cos_iter1   = float(torch.tensor(all_cosines[0]).mean())  if all_cosines[0]  else float("nan")
    cos_iter50  = float(torch.tensor(all_cosines[-1]).mean()) if all_cosines[-1] else float("nan")

    any_explosion   = any(s["exploded"]  > 0 for s in ckpt_stats)
    any_oversmooth  = any(s["smoothed"]  > 0 for s in ckpt_stats)
    total_explosion = sum(s["exploded"]  for s in ckpt_stats)
    total_smooth    = sum(s["smoothed"]  for s in ckpt_stats)

    # ── Varianza entre nodos (por ejemplo, en iter 1 y iter 50) ──────────────
    # Over-smoothing se puede medir como caída en varianza inter-nodo
    var_iter1_list, var_iter50_list = [], []
    for ex in examples:
        graph = ex.graph
        N = len(graph.nodes)
        h0 = make_node_features(graph, node_dim)
        snaps = get_per_iteration_snapshots(engine, graph, h0, n_iter)
        if len(snaps) < n_iter:
            continue
        # Varianza de las features entre nodos (cuánto difieren entre sí)
        h1  = snaps[0].float()
        h50 = snaps[-1].float()
        if N >= 2:
            var_iter1_list.append(float(h1.var(dim=0).mean().item()))
            var_iter50_list.append(float(h50.var(dim=0).mean().item()))

    var1  = float(torch.tensor(var_iter1_list).mean())  if var_iter1_list  else 0.0
    var50 = float(torch.tensor(var_iter50_list).mean()) if var_iter50_list else 0.0

    summary = {
        "n_examples":        len(examples),
        "n_iterations":      n_iter,
        "node_dim":          node_dim,
        "norm_iter1":        norm_iter1,
        "norm_iter50":       norm_iter50,
        "norm_ratio_50_to_1": norm_iter50 / norm_iter1 if norm_iter1 > 0 else float("nan"),
        "cosine_iter1":      cos_iter1,
        "cosine_iter50":     cos_iter50,
        "cosine_delta":      cos_iter50 - cos_iter1 if not math.isnan(cos_iter1) else float("nan"),
        "variance_iter1":    var1,
        "variance_iter50":   var50,
        "variance_ratio":    var50 / var1 if var1 > 0 else float("nan"),
        "any_explosion":     any_explosion,
        "any_oversmoothing": any_oversmooth,
        "total_explosion_cells":  total_explosion,
        "total_smoothed_cells":   total_smooth,
    }

    # ── Imprimir ──────────────────────────────────────────────────────────────
    if not quiet:
        print_section("EXPERIMENTO 2 — Estabilidad de Weight Sharing (50 iteraciones)")
        print(f"  Datos:  {len(examples)} grafos, niveles 1-3")
        print(f"  Config: node_dim={node_dim}, n_layers=2 (weight shared), {n_iter} iteraciones")
        print()

        cols = [
            ("Iter", 4), ("Norma μ", 9), ("Norma σ", 9), ("Norma max", 9),
            ("Cos μ", 8), ("Cos σ", 8), ("Expl.", 5), ("Smooth.", 6),
        ]
        print_table_header(cols)
        for s in ckpt_stats:
            cos_str = f"{s['mean_cosine']:.5f}" if not math.isnan(s["mean_cosine"]) else "  nan  "
            cos_s   = f"{s['std_cosine']:.5f}"  if not math.isnan(s["std_cosine"])  else "  nan  "
            print_table_row(cols, [
                s["iteration"],
                f"{s['mean_norm']:.5f}",
                f"{s['std_norm']:.5f}",
                f"{s['max_norm']:.3f}",
                cos_str,
                cos_s,
                s["exploded"],
                s["smoothed"],
            ])

        # Gráfica ASCII: norma media vs iteración
        print()
        print("  Norma media de node features por iteración (checkpoints):")
        print()
        max_norm = max((s["mean_norm"] for s in ckpt_stats), default=1.0) or 1.0
        for s in ckpt_stats:
            bar = _bar(s["mean_norm"], max_norm, width=35)
            cos_str = f"{s['mean_cosine']:.4f}" if not math.isnan(s["mean_cosine"]) else " n/a "
            print(f"  iter {s['iteration']:>2} │{bar}│ norm={s['mean_norm']:.4f}  cos={cos_str}")

        # Comparación iter 1 vs 50
        print()
        print("  COMPARACIÓN iteración 1 vs iteración 50:")
        print(f"  {'Métrica':<25} {'Iter 1':>10} {'Iter 50':>10} {'Cambio':>10}")
        print(f"  {'─'*25} {'─'*10} {'─'*10} {'─'*10}")

        def _fmt(v):
            return f"{v:.5f}" if not math.isnan(v) else "    nan"

        norm_change = (norm_iter50 - norm_iter1) / norm_iter1 * 100 if norm_iter1 > 0 else float("nan")
        cos_change  = cos_iter50 - cos_iter1 if not math.isnan(cos_iter1) else float("nan")
        var_change  = (var50 - var1) / var1 * 100 if var1 > 0 else float("nan")

        print(f"  {'Norma media':<25} {_fmt(norm_iter1):>10} {_fmt(norm_iter50):>10} "
              f"  {norm_change:>+.1f}%" if not math.isnan(norm_change) else
              f"  {'Norma media':<25} {_fmt(norm_iter1):>10} {_fmt(norm_iter50):>10}")
        print(f"  {'Cosine similarity':<25} {_fmt(cos_iter1):>10} {_fmt(cos_iter50):>10} "
              f"  {cos_change:>+.5f}" if not math.isnan(cos_change) else
              f"  {'Cosine similarity':<25} {_fmt(cos_iter1):>10} {_fmt(cos_iter50):>10}")
        print(f"  {'Varianza inter-nodo':<25} {_fmt(var1):>10} {_fmt(var50):>10} "
              f"  {var_change:>+.1f}%" if not math.isnan(var_change) else
              f"  {'Varianza inter-nodo':<25} {_fmt(var1):>10} {_fmt(var50):>10}")

        # Veredictos
        print()
        print("  VEREDICTOS:")
        print(f"  • Explosión de gradientes: {'SÍ ⚠' if any_explosion else 'NO ✓'}")
        print(f"  • Over-smoothing (cos>0.999): {'SÍ ⚠' if any_oversmooth else 'NO ✓'}")
        var_ratio = summary["variance_ratio"]
        if not math.isnan(var_ratio):
            if var_ratio < 0.1:
                print(f"  ⚠  Varianza cayó a {var_ratio:.3f}x — posible over-smoothing progresivo")
            elif var_ratio > 10.0:
                print(f"  ⚠  Varianza creció a {var_ratio:.3f}x — posible divergencia")
            else:
                print(f"  ✓  Varianza estable: {var_ratio:.3f}x al final vs inicio")
        cos_delta = summary["cosine_delta"]
        if not math.isnan(cos_delta):
            if cos_delta > 0.1:
                print(f"  ⚠  Cosine similarity subió +{cos_delta:.4f} — los nodos se homogenizan")
            elif cos_delta < -0.1:
                print(f"  ✓  Cosine similarity bajó {cos_delta:.4f} — los nodos se diferencian")
            else:
                print(f"  ✓  Cosine similarity estable (Δ={cos_delta:+.4f})")

    return {"checkpoint_stats": ckpt_stats, "summary": summary}


# ─────────────────────────────────────────────────────────────────────────────
# PUNTO DE ENTRADA
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # Forzar UTF-8 en stdout para que los caracteres de tabla funcionen
    # en terminales Windows que por defecto usan cp1252.
    import io
    import warnings
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
        )

    warnings.filterwarnings("ignore")

    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║      CORA ABLATION STUDY — CausalReasoningEngine                    ║")
    print("║      ¿Converge el CRE? ¿Es estable a 50 iteraciones?               ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print(f"\n  Configuración: n={args.n}, dim={args.dim}, seed={args.seed}")

    # ── Cargar datos ──────────────────────────────────────────────────────────
    print(f"\n  Generando {args.n} ejemplos causales (niveles 1-3)...", end="", flush=True)
    t0 = time.time()
    examples = load_examples(args.n, args.seed)
    print(f" {len(examples)} ejemplos en {time.time()-t0:.1f}s")

    if len(examples) == 0:
        print("  ERROR: No se pudieron generar ejemplos válidos.")
        sys.exit(1)

    # Estadísticas de los datos
    node_counts = [len(ex.graph.nodes) for ex in examples]
    edge_counts = [len(ex.graph.edges) for ex in examples]
    lvl_counts  = {1: 0, 2: 0, 3: 0}
    for ex in examples:
        lvl_counts[ex.complexity_level] = lvl_counts.get(ex.complexity_level, 0) + 1
    print(f"  Distribución de niveles: {lvl_counts}")
    print(f"  Nodos por grafo: min={min(node_counts)}, "
          f"media={sum(node_counts)/len(node_counts):.1f}, max={max(node_counts)}")
    print(f"  Aristas por grafo: min={min(edge_counts)}, "
          f"media={sum(edge_counts)/len(edge_counts):.1f}, max={max(edge_counts)}")

    # ── Experimento 1 ─────────────────────────────────────────────────────────
    print(f"\n  Ejecutando Experimento 1 (20 iteraciones)...", end="", flush=True)
    t1 = time.time()
    res1 = run_experiment_1(examples, node_dim=args.dim, n_iter=20, quiet=args.quiet)
    if args.quiet:
        print(f" {time.time()-t1:.1f}s")

    # ── Experimento 2 ─────────────────────────────────────────────────────────
    print(f"\n  Ejecutando Experimento 2 (50 iteraciones)...", end="", flush=True)
    t2 = time.time()
    res2 = run_experiment_2(examples, node_dim=args.dim, n_iter=50, quiet=args.quiet)
    if args.quiet:
        print(f" {time.time()-t2:.1f}s")

    # ── Guardar JSON ──────────────────────────────────────────────────────────
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        "meta": {
            "n_examples": len(examples),
            "node_dim":   args.dim,
            "seed":       args.seed,
            "level_distribution": lvl_counts,
            "node_counts_mean":   sum(node_counts) / len(node_counts),
            "edge_counts_mean":   sum(edge_counts) / len(edge_counts),
        },
        "experiment_1": res1,
        "experiment_2": res2,
    }

    # Convertir floats nan a null para JSON válido
    def _clean(obj):
        if isinstance(obj, float):
            return None if math.isnan(obj) or math.isinf(obj) else obj
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        return obj

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(_clean(results), f, indent=2, ensure_ascii=False)

    print(f"\n  Resultados guardados en: {output_path.resolve()}")

    # ── Conclusión ────────────────────────────────────────────────────────────
    s1 = res1["summary"]
    s2 = res2["summary"]
    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║                       CONCLUSIONES                                  ║")
    print("╠══════════════════════════════════════════════════════════════════════╣")
    pct = s1["monotone_fraction"] * 100
    print(f"║  EXP 1 — Delta monótono:      {pct:>5.1f}% de los ejemplos              ║")
    exp1_ok = not s1["explosion_cells"] and not s1["collapse_cells"]
    exp1_str = "ESTABLE ✓" if exp1_ok else "INESTABLE ⚠"
    print(f"║           Estabilidad:        {exp1_str:<41}║")
    exp2_ok = not s2["any_explosion"] and not s2["any_oversmoothing"]
    exp2_str = "ESTABLE ✓" if exp2_ok else "INESTABLE ⚠"
    print(f"║  EXP 2 — Weight sharing 50it: {exp2_str:<41}║")
    var_r = s2.get("variance_ratio")
    var_str = f"{var_r:.3f}x" if var_r is not None else "n/a"
    print(f"║           Var inter-nodo:     {var_str:<41}║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()


if __name__ == "__main__":
    main()
