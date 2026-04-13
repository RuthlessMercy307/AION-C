"""
visualization/graph_viewer.py — Visualización de CausalGraph
=============================================================

Dos modos:
  1. ASCII — texto para terminal/CLI con nodos, flechas y relaciones tipadas
  2. HTML  — interactivo con vis.js para abrir en browser

Incluye visualización del scratch pad (slots después de cada iteración CRE).

Uso:
    from core.graph import CausalGraph
    from visualization.graph_viewer import GraphViewer, ascii_graph, html_graph

    viewer = GraphViewer(graph)
    print(viewer.to_ascii())
    viewer.to_html("graph.html")
"""

from __future__ import annotations

import html as html_lib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.graph import CausalGraph, CausalNode, CausalEdge


# ─────────────────────────────────────────────────────────────────────────────
# ASCII RENDERER
# ─────────────────────────────────────────────────────────────────────────────

def ascii_graph(graph: CausalGraph, title: str = "CausalGraph") -> str:
    """
    Renderiza un CausalGraph como texto ASCII para terminal.

    Muestra nodos con su tipo y confianza, y aristas con relaciones tipadas.
    """
    n_nodes = len(graph)
    n_edges = len(graph.edges)

    lines: List[str] = []
    lines.append(f"{'=' * 60}")
    lines.append(f"  {title}")
    lines.append(f"  Nodes: {n_nodes}  Edges: {n_edges}")
    lines.append(f"{'=' * 60}")

    # Nodos (graph.nodes is a List[CausalNode])
    lines.append("\n  NODES:")
    for node in graph.nodes:
        conf = f" (conf={node.confidence:.2f})" if node.confidence is not None else ""
        lines.append(f"    [{node.node_id}] {node.label} ({node.node_type.value}){conf}")

    # Build lookup for labels
    node_map = {n.node_id: n for n in graph.nodes}

    # Aristas con ASCII art
    lines.append("\n  EDGES:")
    if not graph.edges:
        lines.append("    (none)")
    for edge in graph.edges:
        src = node_map.get(edge.source_id)
        tgt = node_map.get(edge.target_id)
        src_label = src.label if src else edge.source_id
        tgt_label = tgt.label if tgt else edge.target_id
        rel = edge.relation.value if hasattr(edge.relation, 'value') else str(edge.relation)
        strength = f" [{edge.strength:.2f}]" if edge.strength is not None else ""
        lines.append(f"    {src_label} --[{rel}]{strength}--> {tgt_label}")

    lines.append(f"\n{'=' * 60}")
    return "\n".join(lines)


def ascii_scratch_pad(
    pad_states: List[Any],
    slot_labels: Optional[List[str]] = None,
) -> str:
    """
    Renderiza los estados del scratch pad como ASCII.

    Args:
        pad_states: Lista de tensores/arrays [n_slots, slot_dim] por iteración.
        slot_labels: Nombres opcionales para cada slot.
    """
    lines = ["  SCRATCH PAD STATE:"]
    try:
        import torch
        for i, state in enumerate(pad_states):
            if isinstance(state, torch.Tensor):
                state = state.detach().cpu()
            lines.append(f"\n  Iteration {i}:")
            n_slots = state.shape[0] if hasattr(state, 'shape') else len(state)
            for s in range(n_slots):
                slot = state[s] if hasattr(state, '__getitem__') else state
                if hasattr(slot, 'norm'):
                    norm = slot.norm().item()
                elif hasattr(slot, '__len__'):
                    norm = sum(x*x for x in slot) ** 0.5
                else:
                    norm = 0.0
                label = slot_labels[s] if slot_labels and s < len(slot_labels) else f"slot_{s}"
                bar = "#" * min(40, int(norm * 10))
                lines.append(f"    {label:>10}: |{bar:<40}| norm={norm:.3f}")
    except ImportError:
        lines.append("    (torch not available for pad visualization)")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# HTML RENDERER (vis.js)
# ─────────────────────────────────────────────────────────────────────────────

def _build_html(n_nodes: int, n_edges: int, nodes_json: str, edges_json: str, scratch_pad_html: str) -> str:
    """Build HTML string with vis.js graph visualization."""
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>AION-C Graph Viewer</title>
<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<style>
  body {{ font-family: monospace; margin: 0; background: #1a1a2e; color: #eee; }}
  #graph {{ width: 100%; height: 70vh; border: 1px solid #333; }}
  #info {{ padding: 10px; }}
  h2 {{ color: #0f3460; }}
  .pad-table {{ border-collapse: collapse; margin: 10px; }}
  .pad-table td, .pad-table th {{ border: 1px solid #444; padding: 4px 8px; font-size: 12px; }}
  .pad-table th {{ background: #16213e; }}
</style>
</head>
<body>
<div id="info">
  <h2 style="color:#e94560">AION-C Graph Viewer</h2>
  <p>Nodes: {n_nodes} | Edges: {n_edges}</p>
</div>
<div id="graph"></div>
{scratch_pad_html}
<script>
var nodes = new vis.DataSet({nodes_json});
var edges = new vis.DataSet({edges_json});
var container = document.getElementById('graph');
var data = {{ nodes: nodes, edges: edges }};
var options = {{
  nodes: {{ shape: 'box', font: {{ color: '#fff', size: 14 }}, color: {{ background: '#0f3460', border: '#e94560' }} }},
  edges: {{ arrows: 'to', color: {{ color: '#e94560' }}, font: {{ color: '#aaa', size: 10, align: 'middle' }} }},
  physics: {{ solver: 'forceAtlas2Based', forceAtlas2Based: {{ gravitationalConstant: -50 }} }},
  layout: {{ improvedLayout: true }}
}};
new vis.Network(container, data, options);
</script>
</body>
</html>"""


def html_graph(
    graph: CausalGraph,
    pad_states: Optional[List[Any]] = None,
    output_path: Optional[str] = None,
) -> str:
    """
    Genera un HTML interactivo con vis.js para el CausalGraph.

    Args:
        graph:       CausalGraph a visualizar.
        pad_states:  Lista de estados del scratch pad por iteración.
        output_path: Si se provee, guarda el HTML en este path.

    Returns:
        String HTML completo.
    """
    # Nodos para vis.js
    nodes_list = []
    for node in graph.nodes:
        conf = f"\nconf={node.confidence:.2f}" if node.confidence is not None else ""
        nodes_list.append({
            "id": node.node_id,
            "label": f"{node.label}\n({node.node_type.value}){conf}",
        })

    # Aristas para vis.js
    edges_list = []
    for i, edge in enumerate(graph.edges):
        rel = edge.relation.value if hasattr(edge.relation, 'value') else str(edge.relation)
        edges_list.append({
            "from": edge.source_id,
            "to": edge.target_id,
            "label": rel,
            "id": f"e{i}",
        })

    # Scratch pad HTML table
    scratch_html = ""
    if pad_states:
        try:
            import torch
            rows = []
            for i, state in enumerate(pad_states):
                if isinstance(state, torch.Tensor):
                    state = state.detach().cpu()
                n_slots = state.shape[0] if hasattr(state, 'shape') else 0
                slot_norms = []
                for s in range(n_slots):
                    norm = state[s].norm().item() if hasattr(state[s], 'norm') else 0.0
                    slot_norms.append(f"{norm:.3f}")
                rows.append(f"<tr><td>Iter {i}</td>" +
                           "".join(f"<td>{n}</td>" for n in slot_norms) + "</tr>")
            n_slots = pad_states[0].shape[0] if hasattr(pad_states[0], 'shape') else 0
            headers = "".join(f"<th>Slot {s}</th>" for s in range(n_slots))
            scratch_html = (
                '<div id="info"><h3 style="color:#e94560">Scratch Pad (norm per slot)</h3>'
                f'<table class="pad-table"><tr><th>Iteration</th>{headers}</tr>'
                + "".join(rows) + '</table></div>'
            )
        except ImportError:
            pass

    html_str = _build_html(
        n_nodes=len(graph),
        n_edges=len(graph.edges),
        nodes_json=json.dumps(nodes_list),
        edges_json=json.dumps(edges_list),
        scratch_pad_html=scratch_html,
    )

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_str)

    return html_str


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH VIEWER CLASS
# ─────────────────────────────────────────────────────────────────────────────

class GraphViewer:
    """
    Wrapper para visualización de CausalGraph.

    Uso:
        viewer = GraphViewer(graph, pad_states=[...])
        print(viewer.to_ascii())
        viewer.to_html("output.html")
    """

    def __init__(
        self,
        graph: CausalGraph,
        pad_states: Optional[List[Any]] = None,
        title: str = "CausalGraph",
    ) -> None:
        self.graph = graph
        self.pad_states = pad_states or []
        self.title = title

    def to_ascii(self) -> str:
        text = ascii_graph(self.graph, self.title)
        if self.pad_states:
            text += "\n" + ascii_scratch_pad(self.pad_states)
        return text

    def to_html(self, output_path: Optional[str] = None) -> str:
        return html_graph(self.graph, self.pad_states, output_path)
