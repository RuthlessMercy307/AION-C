"""
cre/engine.py — CausalReasoningEngine
=======================================

El motor de razonamiento iterativo de AION-C.

RAZONAMIENTO COMO ITERACIÓN CONVERGENTE:
    El CRE toma un CausalGraph con node features iniciales y lo "refina"
    aplicando message passing iterativamente con WEIGHT SHARING.

    No es un transformer que pasa N veces por N capas DISTINTAS.
    Es el mismo sistema dinámico (mismos pesos) aplicado N veces —
    como resolver una EDO numéricamente o como el algoritmo de
    belief propagation en grafos probabilísticos.

    Analogía:
        Transformer: leer el problema, responder de una vez
        CRE:         leer el problema, iterar hasta que converge

WEIGHT SHARING:
    Los mismos n_message_layers se reusan en cada iteración:
        iteración 1: layers[0], layers[1]
        iteración 2: layers[0], layers[1]  ← mismos pesos
        ...
        iteración 20: layers[0], layers[1]  ← mismos pesos

    Implicación: 20 iteraciones no cuestan 20x más parámetros.
    Cuestan 20x más FLOPs, que es más barato (memoria constante, compute lineal).

PARADA ADAPTATIVA (use_convergence_gate=True):
    Cuando está activada, el loop usa WeaknessDetector + ConvergenceGate:

    1. WeaknessDetector analiza el grafo tras cada iteración:
       - Detecta 5 tipos de debilidades (low_confidence, missing_cause, ...)
       - Genera focus_mask [N] bool: qué nodos necesitan más refinamiento
       - Calcula confidence scores para la ConvergenceGate

    2. Focus mask application:
       - Solo los nodos débiles reciben su delta de actualización completo
       - Nodos sin debilidades mantienen su estado (ahorra compute efectivo)
       - Ecuación: h = h_old + focus_weight * (h_new - h_old)
       - Todos los nodos PARTICIPAN como fuentes de mensajes, pero solo los
         focalizados ACTUALIZAN su estado. Esto preserva la propagación global.

    3. ConvergenceGate decide cuándo parar:
       - delta_norm < threshold: features estabilizados
       - alta confianza Y pocas debilidades: grafo resuelto
       - max_iterations: safety cap siempre activo
       - min_iterations: safety floor (nunca parar demasiado pronto)

    Resultado: queries simples convergen en 1-3 iteraciones; queries
    complejas usan lo que necesiten hasta max_iterations.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch
import torch.nn as nn

from core.graph import CAUSAL_RELATIONS, CausalGraph
from .batching import BatchedGraph
from .config import CREConfig
from .convergence import ConvergenceGate, ConvergenceDecision
from .message_passing import CausalMessagePassingLayer
from .moe import MoEOutput, SparseMoE
from .weakness import WeaknessDetector, WeaknessReport

if TYPE_CHECKING:
    from .scratch_pad import DifferentiableScratchPad


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT CONTAINER
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CREOutput:
    """
    Resultado del CausalReasoningEngine.

    node_features: [N, node_dim]    — representaciones refinadas (entrada al SD)
    edge_features: [E, edge_dim]    — representaciones refinadas de aristas
    iterations_run: int             — iteraciones ejecutadas
    layer_outputs:  List[Tensor]    — [iter × layer] snapshots (solo si return_history=True)
    stop_reason:    str             — por qué paró ("max_iterations", "delta_stable", ...)
    n_weaknesses_initial: int       — debilidades en la primera iteración (0 si sin gate)
    n_weaknesses_final:   int       — debilidades en la última iteración (0 si sin gate)
    focus_mask_final: Optional[Tensor] — [N] bool, último focus_mask (None si sin gate)
    """
    node_features:       torch.Tensor
    edge_features:       torch.Tensor
    iterations_run:      int
    layer_outputs:       List[torch.Tensor]        # vacía por defecto
    stop_reason:         str                       = "max_iterations"
    n_weaknesses_initial: int                      = 0
    n_weaknesses_final:  int                       = 0
    focus_mask_final:    Optional[torch.Tensor]    = None
    load_balance_loss:   Optional[torch.Tensor]    = None  # acumulada del MoE


# ─────────────────────────────────────────────────────────────────────────────
# CAUSAL REASONING ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class CausalReasoningEngine(nn.Module):
    """
    Motor de razonamiento iterativo con weight sharing.

    Uso:
        config = CREConfig()
        engine = CausalReasoningEngine(config)

        # graph con 5 nodos y 8 aristas
        node_feats = torch.randn(5, 256)   # features iniciales
        output = engine(graph, node_feats)
        # output.node_features: [5, 256] — refinados después de 20 iteraciones

        # Menos iteraciones (testing rápido)
        output = engine(graph, node_feats, n_iterations=3)

        # Con historial para análisis
        output = engine(graph, node_feats, return_history=True)
        # output.layer_outputs: lista de tensores [N, D] por cada (iter, layer)
    """

    def __init__(self, config: CREConfig, relation_keys: Optional[List[str]] = None) -> None:
        super().__init__()
        self.config = config

        # ── Vocabulario de relaciones ─────────────────────────────────────────
        # Por defecto usa CAUSAL_RELATIONS (16 relaciones causales).
        # Pasar relation_keys distinto para motores con vocabulario propio
        # (p.ej. CODE_RELATIONS para FORGE-C, con 12 relaciones de código).
        self.relation_keys: List[str] = (
            relation_keys if relation_keys is not None else CAUSAL_RELATIONS
        )

        # ── Capas compartidas ─────────────────────────────────────────────────
        # WEIGHT SHARING: estas n_layers capas se reusan en CADA iteración.
        self.layers: nn.ModuleList = nn.ModuleList([
            CausalMessagePassingLayer(config, self.relation_keys)
            for _ in range(config.n_message_layers)
        ])

        # ── Embeddings de aristas ─────────────────────────────────────────────
        # Cuando relation_keys es explícito, el tamaño del embedding es len(relation_keys).
        # Cuando relation_keys usa el default (CAUSAL_RELATIONS), respeta config.n_relation_types
        # para backwards compatibility con tests que verifican parámetros por config.
        n_rel_emb = (
            len(relation_keys) if relation_keys is not None
            else config.n_relation_types
        )
        self.edge_type_embedding: nn.Embedding = nn.Embedding(
            n_rel_emb, config.edge_dim
        )

        # ── Proyector de entrada para edge scalars ────────────────────────────
        self.edge_feat_projector: nn.Linear = nn.Linear(
            config.edge_dim + 2, config.edge_dim, bias=False
        )

        nn.init.normal_(self.edge_type_embedding.weight, std=0.02)
        nn.init.normal_(self.edge_feat_projector.weight, std=0.02)

        # ── SparseMoE (opcional) ──────────────────────────────────────────────
        # Activo cuando use_moe=True. El MoE especializa representaciones de
        # nodos después del message passing en cada iteración.
        # El GRU de nivel engine estabiliza cross-iteración cuando hay MoE.
        if config.use_moe:
            self.moe: Optional[SparseMoE] = SparseMoE(config)
            self.moe_gru: Optional[nn.GRUCell] = nn.GRUCell(
                input_size  = config.node_dim,
                hidden_size = config.node_dim,
            )
            nn.init.orthogonal_(self.moe_gru.weight_hh)
        else:
            self.moe     = None
            self.moe_gru = None

        # ── WeaknessDetector + ConvergenceGate (opcionales) ──────────────────
        # Solo se instancian cuando use_convergence_gate=True para no aumentar
        # el tamaño del modelo cuando no se usan.
        if config.use_convergence_gate:
            self.weakness_detector: Optional[WeaknessDetector] = WeaknessDetector(
                node_dim             = config.node_dim,
                confidence_threshold = config.weakness_conf_threshold,
                confidence_hidden    = config.weakness_conf_hidden,
                norm_eps             = config.norm_eps,
            )
            self.convergence_gate: Optional[ConvergenceGate] = ConvergenceGate(
                delta_threshold    = config.conv_delta_threshold,
                conf_threshold     = config.conv_conf_threshold,
                weakness_threshold  = config.conv_weakness_threshold,
                min_iterations     = config.min_iterations,
                norm_eps           = config.norm_eps,
            )
        else:
            self.weakness_detector = None
            self.convergence_gate  = None

    # ── Forward principal ─────────────────────────────────────────────────────

    def forward(
        self,
        graph:           CausalGraph,
        node_features:   torch.Tensor,    # [N, node_dim]
        n_iterations:    int | None = None,
        return_history:  bool = False,
        scratch_pad:     "Optional[DifferentiableScratchPad]" = None,
    ) -> CREOutput:
        """
        Aplica iteraciones de message passing con weight sharing.

        Con use_convergence_gate=False (default):
            Itera exactamente n_iterations veces (comportamiento original).

        Con use_convergence_gate=True:
            Itera hasta que ConvergenceGate dice parar O se alcanza max_iterations.
            WeaknessDetector guía qué nodos reciben updates prioritarios (focus_mask).

        Args:
            graph:          CausalGraph con source_idx/target_idx ya asignados
            node_features:  [N, node_dim] — features iniciales de los nodos
            n_iterations:   safety cap (default: config.max_iterations)
            return_history: si True, guarda snapshots de node_features
            scratch_pad:    DifferentiableScratchPad opcional

        Returns:
            CREOutput con node_features refinados, stop_reason, y métricas de
            debilidades (cuando use_convergence_gate=True)
        """
        if n_iterations is None:
            n_iterations = self.config.max_iterations

        h         = node_features
        h_initial = node_features.detach().clone()   # para coverage_ratio
        e         = self._init_edge_features(graph, h.device, h.dtype)

        pad_state: "Optional[torch.Tensor]" = None
        if scratch_pad is not None:
            pad_state = scratch_pad.init_state(device=h.device, dtype=h.dtype)

        history:      List[torch.Tensor] = []
        stop_reason:  str               = "max_iterations"
        n_weak_init:  int               = 0
        n_weak_final: int               = 0
        focus_mask:   Optional[torch.Tensor] = None   # [N] bool
        iterations_done: int            = n_iterations
        lb_loss_accum:   Optional[torch.Tensor] = None  # acumular MoE LB loss

        use_gate = (
            self.config.use_convergence_gate
            and self.weakness_detector is not None
            and self.convergence_gate is not None
        )

        for iteration in range(n_iterations):
            h_prev = h.detach().clone()   # snapshot antes de este paso

            # ── Message passing (todos los nodos participan como fuentes) ──────
            h_new = h
            for layer in self.layers:
                h_new, e = layer(h_new, e, graph)
                if return_history:
                    history.append(h_new.detach().clone())

            # ── SparseMoE: especialización post-MP ────────────────────────────
            # Flujo: moe_output = SparseMoE(h_new) → GRUCell(moe_output, h_new)
            if self.moe is not None and self.moe_gru is not None:
                moe_result: MoEOutput = self.moe(h_new)
                h_new = self.moe_gru(moe_result.output, h_new)   # [N, D]
                # Acumular load balance loss
                if lb_loss_accum is None:
                    lb_loss_accum = moe_result.load_balance_loss
                else:
                    lb_loss_accum = lb_loss_accum + moe_result.load_balance_loss

            # ── Focus mask: solo nodos débiles reciben su delta completo ───────
            # Ecuación: h = h_old + weight * (h_new - h_old)
            # weight[i] = 1.0 si nodo i está en el focus, 0.0 si no.
            # Sin gate: weight = 1.0 para todos (comportamiento original).
            if use_gate and focus_mask is not None:
                weight = focus_mask.float().unsqueeze(-1)   # [N, 1]
                h = h_prev + weight * (h_new - h_prev)
            else:
                h = h_new

            # ── Scratch pad ───────────────────────────────────────────────────
            if scratch_pad is not None:
                h = scratch_pad.read(h, pad_state)
                pad_state = scratch_pad.update(h, pad_state)

            # ── WeaknessDetector + ConvergenceGate ────────────────────────────
            if use_gate:
                report: WeaknessReport = self.weakness_detector(graph, h, e)

                # Primera iteración: establecer baseline de debilidades
                if iteration == 0:
                    n_weak_init = max(report.n_weaknesses, 1)

                n_weak_final = report.n_weaknesses
                focus_mask   = report.focus_mask   # [N] bool para próxima iteración

                # Verificar convergencia (a partir de la 2ª iteración para tener h_prev)
                if iteration >= 1:
                    decision: ConvergenceDecision = self.convergence_gate.check(
                        h_current   = h,
                        h_prev      = h_prev,
                        h_initial   = h_initial,
                        report      = report,
                        n_weak_init = n_weak_init,
                        iteration   = iteration,
                    )
                    if decision.should_stop:
                        stop_reason     = decision.reason
                        iterations_done = iteration + 1
                        break

            iterations_done = iteration + 1   # actualizar en cada iteración completada

        return CREOutput(
            node_features        = h,
            edge_features        = e,
            iterations_run       = iterations_done,
            layer_outputs        = history,
            stop_reason          = stop_reason,
            n_weaknesses_initial = n_weak_init,
            n_weaknesses_final   = n_weak_final,
            focus_mask_final     = focus_mask,
            load_balance_loss    = lb_loss_accum,
        )

    # ── Inicialización de features de aristas ─────────────────────────────────

    def _init_edge_features(
        self,
        graph:  CausalGraph,
        device: torch.device,
        dtype:  torch.dtype,
    ) -> torch.Tensor:
        """
        Inicializa edge features combinando:
            1. Embedding aprendido del tipo de relación
            2. strength y confidence de cada arista (de CausalEdge)

        Returns: [E, edge_dim]
        """
        if not graph.edges:
            return torch.zeros(0, self.config.edge_dim, device=device, dtype=dtype)

        # Índices de relación: para cada arista, su posición en relation_keys
        rel_indices = torch.tensor(
            [self.relation_keys.index(e.relation.value) for e in graph.edges],
            dtype=torch.long,
            device=device,
        )
        rel_emb = self.edge_type_embedding(rel_indices)  # [E, edge_dim]

        # Scalars por arista: strength y confidence
        strengths   = torch.tensor(
            [e.strength    for e in graph.edges], dtype=dtype, device=device
        ).unsqueeze(-1)   # [E, 1]
        confidences = torch.tensor(
            [e.confidence  for e in graph.edges], dtype=dtype, device=device
        ).unsqueeze(-1)   # [E, 1]

        # Proyectar todo junto → edge_dim
        combined = torch.cat([rel_emb, strengths, confidences], dim=-1)  # [E, edge_dim+2]
        return self.edge_feat_projector(combined)                         # [E, edge_dim]

    # ── Batch forward ─────────────────────────────────────────────────────────

    def forward_batch(
        self,
        graphs:              List[CausalGraph],
        node_features_list:  List[torch.Tensor],   # list of [N_i, node_dim]
        n_iterations:        Optional[int] = None,
    ) -> List[CREOutput]:
        """
        Batch forward: pads all graphs to max_nodes, runs message passing in a
        single flat GPU forward (B * max_nodes nodes), then unpacks per-graph results.

        Converts N sequential CRE forwards into 1 parallel forward.
        Vanilla-only: MoE, ConvergenceGate, and ScratchPad are not applied.

        Args:
            graphs:             B CausalGraphs
            node_features_list: B tensors of shape [N_i, node_dim]
            n_iterations:       iterations to run (default: config.max_iterations)

        Returns: list of B CREOutputs with correct node/edge features per graph.
        """
        if n_iterations is None:
            n_iterations = self.config.max_iterations

        B = len(graphs)
        if B == 0:
            return []

        device = node_features_list[0].device
        dtype  = node_features_list[0].dtype
        D      = self.config.node_dim

        n_nodes_list = [f.shape[0] for f in node_features_list]
        max_nodes    = max(n_nodes_list)

        # 1. Pad node features → flat tensor [B * max_nodes, D]
        padded_nodes: List[torch.Tensor] = []
        for feats in node_features_list:
            n = feats.shape[0]
            if n < max_nodes:
                pad = torch.zeros(max_nodes - n, D, device=device, dtype=dtype)
                padded_nodes.append(torch.cat([feats, pad], dim=0))
            else:
                padded_nodes.append(feats)
        h = torch.cat(padded_nodes, dim=0)   # [B * max_nodes, D]

        # 2. Build batched edge tensors with per-graph node-index offsets
        all_src:         List[int]   = []
        all_tgt:         List[int]   = []
        all_rel_vals:    List[str]   = []
        all_strengths:   List[float] = []
        all_confidences: List[float] = []
        edge_counts:     List[int]   = []

        for b, graph in enumerate(graphs):
            offset = b * max_nodes
            edge_counts.append(len(graph.edges))
            for edge in graph.edges:
                all_src.append(edge.source_idx + offset)
                all_tgt.append(edge.target_idx + offset)
                all_rel_vals.append(edge.relation.value)
                all_strengths.append(edge.strength)
                all_confidences.append(edge.confidence)

        E_total = sum(edge_counts)
        N_total = B * max_nodes

        if E_total == 0:
            return [
                CREOutput(
                    node_features  = node_features_list[b],
                    edge_features  = torch.zeros(
                        0, self.config.edge_dim, device=device, dtype=dtype
                    ),
                    iterations_run = 0,
                    layer_outputs  = [],
                    stop_reason    = "no_edges",
                )
                for b in range(B)
            ]

        src_idx = torch.tensor(all_src, dtype=torch.long, device=device)
        tgt_idx = torch.tensor(all_tgt, dtype=torch.long, device=device)

        # 3. Initialize batched edge features [E_total, edge_dim]
        rel_indices = torch.tensor(
            [self.relation_keys.index(r) for r in all_rel_vals],
            dtype=torch.long, device=device,
        )
        rel_emb     = self.edge_type_embedding(rel_indices)
        strengths   = torch.tensor(all_strengths,   dtype=dtype, device=device).unsqueeze(-1)
        confidences = torch.tensor(all_confidences, dtype=dtype, device=device).unsqueeze(-1)
        e = self.edge_feat_projector(
            torch.cat([rel_emb, strengths, confidences], dim=-1)
        )   # [E_total, edge_dim]

        # 4. Run n_iterations on the flat batched graph (1 GPU forward per iteration per layer)
        for _ in range(n_iterations):
            for layer in self.layers:
                h, e = layer.forward_tensors(
                    h, e, src_idx, tgt_idx, all_rel_vals, N_total
                )

        # 5. Unpack per-graph results — strip padding from node features
        outputs:     List[CREOutput] = []
        edge_offset: int             = 0
        for b in range(B):
            n         = n_nodes_list[b]
            node_out  = h[b * max_nodes : b * max_nodes + n]   # [N_i, D]
            ec        = edge_counts[b]
            edge_out  = e[edge_offset : edge_offset + ec]       # [E_b, edge_dim]
            edge_offset += ec

            outputs.append(CREOutput(
                node_features  = node_out,
                edge_features  = edge_out,
                iterations_run = n_iterations,
                layer_outputs  = [],
                stop_reason    = "max_iterations",
            ))

        return outputs

    # ── Forward batched (PyG-style) ───────────────────────────────────────────

    def forward_batched(
        self,
        batched:      "BatchedGraph",
        n_iterations: Optional[int] = None,
    ) -> List[CREOutput]:
        """
        Forward pass sobre un super-grafo concatenado estilo PyG.

        Procesa B grafos en UN SOLO forward pass sin padding con nodos dummy.
        El scatter_add con edge_index offseteados respeta los límites entre grafos
        automáticamente — los grafos no tienen edges entre sí, así que el message
        passing NO cruza entre grafos.

        Resultado MATEMÁTICAMENTE IDÉNTICO a B llamadas individuales de forward().
        Speedup: ~13x con batch=16, ~21x con batch=32 (ver AION-C-plan-v4 §16.2.3).

        Args:
            batched:      BatchedGraph del PyGStyleBatcher (con offsets en edge_index)
            n_iterations: iteraciones a correr (default: config.max_iterations)

        Returns:
            List of B CREOutputs — uno por grafo original, con features correctos.

        Nota: esta implementación es vanilla-only (sin MoE, ConvergenceGate, ScratchPad)
        para mantener equivalencia exacta con forward() en modo estándar.
        Para features avanzados, usar forward() individual.
        """
        if n_iterations is None:
            n_iterations = self.config.max_iterations

        device = batched.device
        dtype  = batched.node_features.dtype
        B      = batched.n_graphs

        if B == 0:
            return []

        N_total = batched.n_nodes
        E_total = batched.n_edges

        h = batched.node_features   # [N_total, D] — vivo en GPU, sin copias

        # ── Inicializar edge features del super-grafo ─────────────────────────
        # Los índices de relación se construyen aquí usando self.relation_keys,
        # que es el vocabulario propio de este CRE (CAUSAL_RELATIONS para CORA,
        # CODE_RELATIONS para FORGE-C, MATH_RELATIONS para AXIOM, etc.).
        # El batcher almacena solo strings (edge_rel_vals) — agnóstico al motor.
        if E_total == 0:
            e = torch.zeros(0, self.config.edge_dim, device=device, dtype=dtype)
        else:
            rel_indices = torch.tensor(
                [self.relation_keys.index(r) for r in batched.edge_rel_vals],
                dtype=torch.long, device=device,
            )   # [E_total]
            rel_emb     = self.edge_type_embedding(rel_indices)     # [E_total, edge_dim]
            strengths   = batched.edge_strengths.unsqueeze(-1)      # [E_total, 1]
            confidences = batched.edge_confidences.unsqueeze(-1)    # [E_total, 1]
            e = self.edge_feat_projector(
                torch.cat([rel_emb, strengths, confidences], dim=-1)
            )   # [E_total, edge_dim]

        # ── src_idx / tgt_idx del super-grafo ────────────────────────────────
        if E_total > 0:
            src_idx = batched.edge_index[0]   # [E_total] — ya con offsets
            tgt_idx = batched.edge_index[1]   # [E_total] — ya con offsets
            edge_rel_vals = batched.edge_rel_vals
        else:
            src_idx = torch.zeros(0, dtype=torch.long, device=device)
            tgt_idx = torch.zeros(0, dtype=torch.long, device=device)
            edge_rel_vals = []

        # ── n_iterations pasadas sobre el super-grafo ─────────────────────────
        # Los grafos no se mezclan porque no hay edges entre ellos.
        # scatter_add en forward_tensors usa tgt_idx como destino — como los
        # índices de Grafo B son ≥ offset(B), los mensajes nunca van a nodos de A.
        for _ in range(n_iterations):
            for layer in self.layers:
                h, e = layer.forward_tensors(
                    h, e, src_idx, tgt_idx, edge_rel_vals, N_total
                )

        # ── Desempaquetar resultados por grafo ────────────────────────────────
        outputs: List[CREOutput] = []
        node_offset = 0
        edge_offset = 0

        for b in range(B):
            n_nodes = batched.nodes_per_graph[b]
            n_edges = batched.edges_per_graph[b]

            node_out = h[node_offset : node_offset + n_nodes]    # [N_i, D]
            edge_out = e[edge_offset : edge_offset + n_edges]    # [E_i, edge_dim]

            outputs.append(CREOutput(
                node_features  = node_out,
                edge_features  = edge_out,
                iterations_run = n_iterations,
                layer_outputs  = [],
                stop_reason    = "max_iterations",
            ))

            node_offset += n_nodes
            edge_offset += n_edges

        return outputs

    # ── Utilidades ────────────────────────────────────────────────────────────

    def count_parameters(self) -> int:
        """Número total de parámetros entrenables."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def parameter_breakdown(self) -> dict:
        """Desglose de parámetros por sub-módulo."""
        layers_total = sum(p.numel() for layer in self.layers for p in layer.parameters())
        return {
            "layers_shared":        layers_total,
            "edge_type_embedding":  self.edge_type_embedding.weight.numel(),
            "edge_feat_projector":  self.edge_feat_projector.weight.numel(),
            "total":                self.count_parameters(),
            # Nota: layers_shared son los mismos pesos reutilizados N veces
            # — no se multiplican por max_iterations en el conteo de parámetros.
            "effective_at_max_iter": layers_total * self.config.max_iterations,
        }
