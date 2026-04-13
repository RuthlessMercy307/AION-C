"""
backend/app_fastapi.py — FastAPI + WebSocket backend (Parte 12 del MEGA-PROMPT)
================================================================================

Cumple con la Parte 12:
  12.1  React + Vite (usamos React desde CDN — sin build step) + FastAPI + WebSocket
  12.2  Chat con streaming token-by-token via WebSocket
  12.3  Panel lateral: routing scores, MEM, system state, stats
  12.4  Grafos interactivos por mensaje (vis-network desde CDN)
  12.5  Thinking indicator (solo NORMAL/DEEP)
  12.6  Plan view
  12.7  Tool log

Diseño:
  - `AppState` contiene model + tok + MEM + lifecycle + sessions + planner
  - `create_app(state)` crea la FastAPI con todas las rutas
  - El modelo es OPCIONAL: si no se inyecta, usa un mock que ecoa el input.
    Esto hace la app testeable sin cargar 5.5M params.
  - Los tests usan TestClient con state mockeado.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

# Module-level imports — pydantic 2.x necesita los tipos resueltos en el
# scope donde se define el endpoint, no dentro de create_app(). Si los
# importamos en una función, los ForwardRef quedan sin resolver.
from fastapi import (
    FastAPI, WebSocket, WebSocketDisconnect,
    UploadFile, File, HTTPException,
)
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse


# ─────────────────────────────────────────────────────────────────────────────
# Tipos del wire protocol del WebSocket
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class RoutingScores:
    """Scores normalizados por motor (0-1)."""
    cora:    float = 0.0
    forge_c: float = 0.0
    muse:    float = 0.0
    axiom:   float = 0.0
    empathy: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "cora": self.cora, "forge_c": self.forge_c,
            "muse": self.muse, "axiom": self.axiom, "empathy": self.empathy,
        }


@dataclass
class ChatStreamMessage:
    """
    Mensaje en el wire del WebSocket. type:
      'token'    — un token nuevo del stream
      'meta'     — info del motor activo, scores, level
      'thinking' — indicador de razonamiento (solo NORMAL/DEEP)
      'plan'     — Plan render (si la query disparó un plan)
      'tool'     — entry del tool log
      'graph'    — grafo causal de la respuesta
      'done'     — fin del stream
      'error'    — error
    """
    type:    str
    payload: Dict[str, Any]

    def to_json(self) -> str:
        return json.dumps({"type": self.type, "payload": self.payload}, ensure_ascii=False)


# ─────────────────────────────────────────────────────────────────────────────
# AppState
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class AppState:
    """
    Estado compartido del backend. Inyectable para tests.

    Args:
        model, tokenizer, mem    — pipeline + MEM real (o None para mock)
        sessions                 — dict session_id → ConversationHistory
        output_dir               — sandbox para upload/download
        n_params                 — param count del modelo (para /api/info)
        skills_loader            — SkillsLoader cargado con skills/*.md
        response_cache           — ResponseCache con LRU+TTL
        user_models              — dict session_id → UserModel
        lifecycle                — LifecycleManager (ACTIVE/IDLE/...)
        tool_executor            — ToolExecutor con todas las tools registradas
        planner                  — Planner para descomposición de tareas
        self_checker             — SelfChecker para validación de respuestas
        world_simulators         — dict motor_name → WorldSimulator
        symbolic_engines         — dict motor_name → SymbolicEngine
        goals_manager            — GoalsManager (Parte 17)
        identity_skill_text      — contenido del skills/identity.md (siempre inyectado)
    """
    model:      Any = None
    tokenizer:  Any = None
    mem:        Any = None
    sessions:   Dict[str, Any] = field(default_factory=dict)
    output_dir: Path = field(default_factory=lambda: Path("output"))
    n_params:   int = 0
    motor_names: List[str] = field(default_factory=lambda: ["cora", "forge_c", "muse", "axiom", "empathy"])
    lifecycle:  Any = None
    skills_loader: Any = None
    response_cache: Any = None
    user_models: Dict[str, Any] = field(default_factory=dict)
    tool_executor: Any = None
    planner:    Any = None
    self_checker: Any = None
    world_simulators: Dict[str, Any] = field(default_factory=dict)
    symbolic_engines: Dict[str, Any] = field(default_factory=dict)
    goals_manager: Any = None
    identity_skill_text: str = ""
    # ── Fase F / Parte 22: adapters ────────────────────────────────────
    adapter_registry: Any = None   # growth.AdapterRegistry (opcional)
    # ── Fase F / Parte 22.5: trayectorias compuestas ──────────────────
    trajectory_planner: Any = None  # composition.TrajectoryPlanner
    # ── Fase F / Parte 23: sleep cycle ────────────────────────────────
    episodic_buffer: Any = None     # sleep.EpisodicBuffer
    sleep_daemon: Any = None        # sleep.SleepDaemon
    # ── Fase F / Parte 25: reward probabilístico ──────────────────────
    reward_estimator: Any = None    # reward.RewardEstimator
    reward_ledger: Any = None       # reward.RewardLedger
    # ── Fase F / Parte 26: compresión jerárquica ──────────────────────
    hierarchical_store: Any = None  # compression.HierarchicalStore
    # ── Fase F / Parte 27: activación esparsa ─────────────────────────
    sparsity_tracker: Any = None    # sparse.SparsityTracker
    # ── Persistencia Fase F ───────────────────────────────────────────
    reward_ledger_path: Any = None  # Path a brain/v1/reward_ledger.jsonl
    hierarchy_path: Any = None      # Path a brain/v1/hierarchy.jsonl
    # ── SleepDaemon background loop ──────────────────────────────────
    sleep_poll_interval: float = 300.0  # chequea cada 5 min
    _sleep_task: Any = None

    def save_reward_ledger(self) -> None:
        if self.reward_ledger is not None and self.reward_ledger_path is not None:
            try:
                self.reward_ledger.save_jsonl(self.reward_ledger_path)
            except Exception:
                pass

    def save_hierarchy(self) -> None:
        if self.hierarchical_store is not None and self.hierarchy_path is not None:
            try:
                self.hierarchical_store.save_jsonl(self.hierarchy_path)
            except Exception:
                pass
    # ── Injection flags (Bug fix #1: tiny canonical generates better
    # con prompts mínimos. Default OFF para identity/skills/mem.
    # Cuando se entrene un modelo más grande, activarlos.) ──
    inject_identity: bool = False
    inject_skills:   bool = False
    inject_mem:      bool = False
    inject_summary:  bool = True   # ConversationHistory summary sí (multi-turn)

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Generación: real o mock
# ─────────────────────────────────────────────────────────────────────────────


async def _sleep_daemon_loop(state: "AppState") -> None:
    """Loop de background que corre maybe_run() del SleepDaemon periódicamente.

    Llamado desde startup hook con asyncio.create_task. Se detiene con
    task.cancel() en el shutdown hook.

    El intervalo de chequeo es `state.sleep_poll_interval` (default 5 min).
    El daemon decide internamente si el trigger aplica (inactividad u overflow).
    """
    if state.sleep_daemon is None:
        return
    try:
        while True:
            await asyncio.sleep(state.sleep_poll_interval)
            try:
                log = state.sleep_daemon.maybe_run()
                if log is not None:
                    state.save_hierarchy()
                    state.save_reward_ledger()
                    print(
                        f"[sleep] auto cycle: trigger={log.trigger}, "
                        f"eps={log.episodes_processed}, {log.duration_ms:.0f}ms"
                    )
            except Exception as exc:
                print(f"[sleep] maybe_run error: {exc}")
    except asyncio.CancelledError:
        return


async def _stream_echo(text: str) -> "asyncio.Generator[str, None, None]":
    """
    Mock streaming. Si el text contiene un bloque [USER: ...], extrae solo
    la pregunta del usuario y produce una respuesta corta. Si no, ecoa las
    primeras 5 palabras como mucho. Esto evita que el mock emita el prompt
    canónico completo (con SKILL/MEM/identity) durante los tests.
    """
    import re
    raw = text or ""
    m = re.search(r"\[USER:\s*([^\]]+)\]", raw)
    if m:
        user_q = m.group(1).strip()
        # Respuesta sintética corta para tests
        words = ("OK: " + user_q).split()[:6]
    else:
        words = raw.split()[:5]
    for w in words:
        await asyncio.sleep(0)
        yield w + " "


async def _stream_real_model(state: AppState, text: str, max_new: int = 60):
    """
    Streaming real con el modelo MoSE. Hace greedy decode emitiendo
    cada token decodificado a medida que se genera.
    """
    import torch
    from synth.canonical_dataloader import EOS_TOKEN_ID

    pipeline = state.model
    tok = state.tokenizer
    if pipeline is None or tok is None:
        async for w in _stream_echo(text):
            yield w
        return

    pipeline.eval()
    try:
        ids = tok.encode(text, 96)
    except TypeError:
        ids = tok.encode(text)[:96]
    cur = torch.tensor([ids], dtype=torch.long)
    plen = len(ids)
    last_decoded_len = 0

    with torch.no_grad():
        out = pipeline(cur)
        for _ in range(max_new):
            nxt = int(out.logits[0, -1].argmax().item())
            if nxt in (0, EOS_TOKEN_ID):
                break
            cur = torch.cat([cur, torch.tensor([[nxt]])], dim=1)
            try:
                full = tok.decode(cur[0, plen:].tolist())
            except Exception:
                full = ""
            new_part = full[last_decoded_len:]
            last_decoded_len = len(full)
            if new_part:
                yield new_part
                await asyncio.sleep(0)  # yield to event loop
            if cur.shape[1] >= 128:
                break
            out = pipeline(cur)


def _routing_for(state: AppState, text: str) -> RoutingScores:
    """Calcula scores del orchestrator. Mock-friendly."""
    if state.model is None or state.tokenizer is None:
        # Heurística mock: prioriza dominio según palabras clave
        low = (text or "").lower()
        if any(w in low for w in ("def ", "function", "código", "python", "javascript", "code")):
            return RoutingScores(forge_c=0.8, cora=0.05, muse=0.05, axiom=0.05, empathy=0.05)
        if any(w in low for w in ("%", "calcular", "math", "*", "+", "ecuación", "calcula")):
            return RoutingScores(axiom=0.8, cora=0.05, muse=0.05, forge_c=0.05, empathy=0.05)
        if any(w in low for w in ("triste", "feliz", "siento", "feel", "lonely", "frustrad")):
            return RoutingScores(empathy=0.8, cora=0.05, muse=0.05, forge_c=0.05, axiom=0.05)
        if any(w in low for w in ("poema", "historia", "story", "poem", "describe")):
            return RoutingScores(muse=0.8, cora=0.05, axiom=0.05, forge_c=0.05, empathy=0.05)
        return RoutingScores(cora=0.6, forge_c=0.1, muse=0.1, axiom=0.1, empathy=0.1)

    import torch
    pipeline = state.model
    tok = state.tokenizer
    try:
        ids = tok.encode(text, 96)
    except TypeError:
        ids = tok.encode(text)[:96]
    ids_t = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        concepts = pipeline.encoder(ids_t)
        pooled = concepts.mean(1).mean(0, keepdim=True)
        logits = pipeline.orchestrator.classifier(pooled)
        probs = torch.softmax(logits.squeeze(0), dim=-1).tolist()
    # Orden: cora, forge_c, muse, axiom, empathy (consistente con MOTOR_NAMES)
    return RoutingScores(
        cora=float(probs[0]),
        forge_c=float(probs[1]),
        muse=float(probs[2]),
        axiom=float(probs[3]),
        empathy=float(probs[4]),
    )


def _decide_level(text: str) -> Dict[str, Any]:
    """Decide reasoning level usando agent.reasoning_levels.LevelDecider."""
    from agent.reasoning_levels import LevelDecider
    decider = LevelDecider()
    decision = decider.decide(text)
    return {
        "level":          int(decision.level),
        "label":          decision.level.label,
        "show_thinking":  decision.level.show_thinking_indicator,
        "iterations":     list(decision.level.iterations),
        "reason":         decision.reason,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Wiring helpers — los 16 puntos del wiring se apoyan en estas funciones
# ─────────────────────────────────────────────────────────────────────────────


def _build_canonical_prompt(
    state: AppState,
    sid: str,
    user_text: str,
) -> str:
    """
    Construye el prompt en formato canónico inyectando:
      [SKILL: identity]               (siempre, Punto 10/16)
      [SKILL: contenido relevante]    (si hay match cosine > 0.5, Punto 1)
      [MEM: hechos relevantes]        (top-3 de la búsqueda en MEM)
      [SUMMARY: resumen de turnos]    (de ConversationHistory, Punto 3)
      [USER: texto del usuario]
    El [AION: ] se deja abierto para que el modelo lo complete.
    """
    parts: List[str] = []

    # 1) Identity skill (Punto 10/16) — opt-in via state.inject_identity
    #    Bug fix: tiny canonical entrenó solo en 2K records con [SKILL:],
    #    inyectarlo siempre lo saca de distribution → respuestas truncadas.
    if state.inject_identity and state.identity_skill_text:
        parts.append(f"[SKILL: {state.identity_skill_text}]")

    # 2) Skills relevantes — opt-in via state.inject_skills (cosine > 0.6)
    if state.inject_skills and state.skills_loader is not None and state.mem is not None:
        try:
            hits = state.skills_loader.search(user_text, state.mem, top_k=1, threshold=0.6)
            for k, content, score in hits:
                if k == "identity":
                    continue  # ya inyectado
                parts.append(f"[SKILL: {content[:200]}]")
        except Exception:
            pass

    # 3) MEM top-1 — opt-in via state.inject_mem (cosine > 0.6)
    if state.inject_mem and state.mem is not None:
        try:
            mem_hits = state.mem.search(user_text, top_k=1) or []
            for item in mem_hits[:1]:
                if isinstance(item, tuple) and len(item) >= 3:
                    k, v, score = item[0], item[1], item[2]
                    if score >= 0.6 and not k.startswith("skill") and k not in ("identity",):
                        parts.append(f"[MEM: {k}: {str(v)[:100]}]")
        except Exception:
            pass

    # 4) ConversationHistory: summary de turnos pasados (Punto 3)
    if state.inject_summary:
        history = state.sessions.get(sid)
        if history is not None and hasattr(history, "summary_block"):
            try:
                summary = history.summary_block()
                if summary:
                    parts.append(f"[SUMMARY: {summary[:200]}]")
            except Exception:
                pass

    # 5) UserModel del usuario actual (Punto 8) — incluido como [MEM:] solo
    #    si inject_mem está activo
    if state.inject_mem:
        user_model = state.user_models.get(sid)
        if user_model is not None and hasattr(user_model, "render_for_context"):
            try:
                ctx = user_model.render_for_context()
                if ctx:
                    parts.append(f"[MEM: {ctx}]")
            except Exception:
                pass

    parts.append(f"[USER: {user_text}]")
    parts.append("[AION:")
    return "\n".join(parts)


def _maybe_update_user_model(state: AppState, sid: str, user_text: str) -> None:
    """
    Detecta auto-identificación en el input y actualiza el UserModel
    de la sesión. Reglas heurísticas mínimas (Punto 8).
    """
    from memory.user_model import UserModel
    if sid not in state.user_models:
        state.user_models[sid] = UserModel()
    um = state.user_models[sid]
    low = user_text.lower()

    # Idioma: si hay tildes/ñ asume es; si predomina inglés, en
    has_es_chars = any(c in user_text for c in "áéíóúñ¿¡")
    has_en_words = any(w in low.split() for w in ("the", "is", "are", "what", "how", "you"))
    if has_es_chars and not has_en_words:
        um.set_language("es")
    elif has_en_words and not has_es_chars:
        um.set_language("en")

    # Nombre
    import re
    m = re.search(r"(?:mi nombre es|me llamo|my name is|i'm)\s+([A-Za-zÁÉÍÓÚÑáéíóúñ]+)", user_text, re.IGNORECASE)
    if m:
        um.set_name(m.group(1))

    # Nivel técnico
    if any(w in low for w in ("soy principiante", "i'm a beginner", "noob")):
        um.set_technical_level("beginner")
    elif any(w in low for w in ("soy senior", "i'm experienced", "advanced")):
        um.set_technical_level("advanced")


def _detect_multi_step(text: str) -> bool:
    """Heurística simple: detecta si el query parece ser una tarea multi-step."""
    low = text.lower().strip()
    # Triggers: separadores comunes + verbos imperativos múltiples
    multi_seps = (" luego ", " después ", " then ", " and then ", " y después ",
                  ";", "1)", "2)", "1.", "2.")
    if any(s in low for s in multi_seps):
        return True
    # Multi-line con bullets
    lines = [l for l in text.splitlines() if l.strip().startswith(("-", "*", "•"))]
    if len(lines) >= 2:
        return True
    return False


def _build_graph_for_query(state: AppState, user_text: str, motor: str) -> Dict[str, Any]:
    """
    Construye un SymbolicGraph mínimo a partir del query y el motor activo,
    luego lo pasa por el SymbolicEngine del motor (Punto 13). Devuelve dict
    listo para el frontend.
    """
    from symbolic.graph import SymbolicGraph, SymbolicNode, SymbolicEdge

    g = SymbolicGraph()
    # Construcción heurística por motor: extraer tokens y armar relaciones simples
    words = [w.strip(".,!?¡¿").lower() for w in user_text.split() if len(w) > 2]
    words = [w for w in words if w not in
             ("the", "and", "for", "with", "que", "para", "como", "este", "esta", "esto", "what", "how")]
    words = words[:6]

    # Nodos
    for i, w in enumerate(words):
        g.add_node(SymbolicNode(id=f"n{i}", label=w, type="concept"))

    # Aristas heurísticas según motor
    if motor == "cora":
        # Cadena causal
        for i in range(len(words) - 1):
            g.add_edge(SymbolicEdge(f"n{i}", f"n{i+1}", "causes"))
    elif motor == "axiom":
        # Implicación lógica encadenada
        for i in range(len(words) - 1):
            g.add_edge(SymbolicEdge(f"n{i}", f"n{i+1}", "implies"))
    elif motor == "forge_c":
        # Llamadas (calls)
        for i in range(len(words) - 1):
            g.add_edge(SymbolicEdge(f"n{i}", f"n{i+1}", "calls"))
    else:
        # Por defecto: relación genérica
        for i in range(len(words) - 1):
            g.add_edge(SymbolicEdge(f"n{i}", f"n{i+1}", "leads_to"))

    # Pasar por el engine simbólico si hay uno para el motor
    engine = state.symbolic_engines.get(motor)
    conflicts: List[str] = []
    notes: List[str] = []
    if engine is not None:
        try:
            result = engine.apply_all(g, max_iters=3)
            conflicts = list(result.conflicts)
            notes = list(result.notes)
        except Exception:
            pass

    return {
        "nodes": [{"id": n.id, "label": n.label, "type": n.type} for n in g.nodes],
        "edges": [{"source": e.source, "target": e.target, "relation": e.relation} for e in g.edges],
        "conflicts": conflicts,
        "notes": notes,
    }


def _run_world_model_simulation(state: AppState, user_text: str, motor: str) -> Optional[Dict[str, Any]]:
    """
    Punto 12: corre el WorldSimulator del motor activo en el scratch pad,
    pasa por verifier, hace el loop simulate→verify→re-simulate.
    Devuelve un dict con el scratch pad final + iteraciones, o None si no aplica.
    """
    sim = state.world_simulators.get(motor)
    if sim is None:
        return None
    try:
        from world_model.verifier import SimulationLoop
        loop = SimulationLoop(simulator=sim, max_iters=3)
        outcome = loop.run(user_text)
        return {
            "motor":      motor,
            "coherent":   outcome.coherent,
            "iterations": outcome.iterations,
            "scratchpad": outcome.pad.as_dict(),
            "issues":     outcome.last_result.issues,
        }
    except Exception:
        return None


def _execute_tools_in_response(state: AppState, response_text: str) -> List[Dict[str, Any]]:
    """
    Punto 2: parsea [TOOL: ...] del response del modelo, ejecuta cada tool
    via ToolExecutor y devuelve la lista de resultados para emitir al UI.
    """
    if state.tool_executor is None:
        return []
    try:
        records = state.tool_executor.run_from_text(response_text)
    except Exception:
        return []
    out = []
    for rec in records:
        out.append({
            "action":  rec.call.action,
            "input":   rec.call.input if not isinstance(rec.call.input, dict) else "...",
            "success": rec.success,
            "summary": rec.result.as_text()[:120],
        })
    return out


def _self_check(state: AppState, user_text: str, response_text: str) -> Dict[str, Any]:
    """Punto 5: corre SelfChecker y devuelve dict con confidence/policy."""
    if state.self_checker is None:
        return {"passed": True, "confidence": None, "policy": "respond_directly", "issues": []}
    try:
        from agent.self_check import classify_confidence, policy_for_confidence
        # Probabilidades sintéticas: la longitud razonable del response da confianza media-alta
        n = max(1, len(response_text.split()))
        synth_probs = [min(0.95, 0.5 + n * 0.02)] * 5
        result = state.self_checker.check(user_text, response_text, probs=synth_probs)
        level = result.confidence_level
        return {
            "passed":     result.passed,
            "confidence": result.confidence,
            "level":      level.value if level else None,
            "policy":     policy_for_confidence(level) if level else "respond_directly",
            "issues":     result.issues,
        }
    except Exception:
        return {"passed": True, "confidence": None, "policy": "respond_directly", "issues": []}


# ─────────────────────────────────────────────────────────────────────────────
# Frontend HTML (React desde CDN, sin build step)
# ─────────────────────────────────────────────────────────────────────────────


def _frontend_html() -> str:
    """Devuelve el HTML del frontend React. Cargado de backend/static/index.html."""
    here = Path(__file__).resolve().parent
    html_path = here / "static" / "index.html"
    if html_path.exists():
        return html_path.read_text(encoding="utf-8")
    return "<!doctype html><html><body><h1>AION-C</h1><p>frontend html missing</p></body></html>"


# ─────────────────────────────────────────────────────────────────────────────
# create_app — la fábrica
# ─────────────────────────────────────────────────────────────────────────────


def create_app(state: Optional[AppState] = None):
    """
    Crea la FastAPI app con todas las rutas y un AppState inyectable.

    Si `state` es None, se crea uno vacío (modo mock).
    Devuelve la instancia FastAPI.
    """
    if state is None:
        state = AppState()

    app = FastAPI(title="AION-C")
    app.state.aion = state

    # ── HTTP endpoints ────────────────────────────────────────────────

    @app.get("/", response_class=HTMLResponse)
    async def root() -> HTMLResponse:
        return HTMLResponse(content=_frontend_html())

    @app.get("/api/info")
    async def api_info() -> JSONResponse:
        params = int(state.n_params)
        return JSONResponse({
            "name":         "AION-C",
            "params":       params,
            "params_human": (f"{params/1e9:.2f}B" if params >= 1e9
                             else f"{params/1e6:.2f}M" if params >= 1e6
                             else f"{params}"),
            "motors":       state.motor_names,
            "model_loaded": state.model is not None,
            "n_sessions":   len(state.sessions),
            "lifecycle_state": getattr(state.lifecycle, "state", None) and state.lifecycle.state.value or "idle",
        })

    @app.post("/api/session")
    async def api_new_session() -> JSONResponse:
        sid = str(uuid.uuid4())[:8]
        _ensure_session(state, sid)
        return JSONResponse({"session_id": sid})

    @app.get("/api/sessions")
    async def api_list_sessions() -> JSONResponse:
        out = []
        for sid, history in state.sessions.items():
            turns = _history_to_list(history)
            first = turns[0]["content"][:40] if turns else "(empty)"
            out.append({"id": sid, "title": first, "messages": len(turns)})
        return JSONResponse(out)

    @app.get("/api/session/{sid}")
    async def api_get_session(sid: str) -> JSONResponse:
        history = state.sessions.get(sid)
        return JSONResponse(_history_to_list(history))

    @app.delete("/api/session/{sid}")
    async def api_delete_session(sid: str) -> JSONResponse:
        existed = sid in state.sessions
        state.sessions.pop(sid, None)
        return JSONResponse({"deleted": existed})

    @app.get("/api/mem")
    async def api_mem() -> JSONResponse:
        if state.mem is None:
            return JSONResponse({"entries": [], "total": 0})
        try:
            entries = state.mem.list_entries()
            stats = {}
            if hasattr(state.mem, "stats"):
                try:
                    stats = state.mem.stats()
                except Exception:
                    stats = {}
            return JSONResponse({"entries": entries[:50], "total": len(entries), "stats": stats})
        except Exception as exc:
            return JSONResponse({"entries": [], "total": 0, "error": str(exc)})

    @app.post("/api/upload")
    async def api_upload(file: UploadFile = File(...)) -> JSONResponse:
        if not file or not file.filename:
            raise HTTPException(status_code=400, detail="no file provided")
        # Sanitize filename — solo basename, no paths
        safe_name = Path(file.filename).name
        target = (state.output_dir / safe_name).resolve()
        # Reject path traversal
        try:
            target.relative_to(state.output_dir.resolve())
        except ValueError:
            raise HTTPException(status_code=400, detail="invalid filename")
        content = await file.read()
        target.write_bytes(content)
        return JSONResponse({
            "filename": safe_name,
            "bytes":    len(content),
            "path":     str(target),
        })

    @app.get("/api/download/{filename}")
    async def api_download(filename: str):
        safe_name = Path(filename).name
        target = (state.output_dir / safe_name).resolve()
        try:
            target.relative_to(state.output_dir.resolve())
        except ValueError:
            raise HTTPException(status_code=400, detail="invalid filename")
        if not target.exists():
            raise HTTPException(status_code=404, detail="not found")
        return FileResponse(str(target), filename=safe_name)

    @app.get("/api/files")
    async def api_list_files() -> JSONResponse:
        files = []
        for p in state.output_dir.glob("*"):
            if p.is_file():
                files.append({
                    "name":  p.name,
                    "bytes": p.stat().st_size,
                })
        return JSONResponse({"files": files})

    # ── Punto 6: cache stats ──────────────────────────────────────────

    @app.get("/api/cache/stats")
    async def api_cache_stats() -> JSONResponse:
        if state.response_cache is None:
            return JSONResponse({"enabled": False})
        try:
            stats = state.response_cache.stats()
            return JSONResponse({"enabled": True, **stats})
        except Exception as exc:
            return JSONResponse({"enabled": False, "error": str(exc)})

    # ── Punto 7: lifecycle ──────────────────────────────────────────────

    @app.get("/api/lifecycle")
    async def api_lifecycle() -> JSONResponse:
        if state.lifecycle is None:
            return JSONResponse({"state": "idle", "history": []})
        try:
            stats = state.lifecycle.stats()
            return JSONResponse(stats)
        except Exception as exc:
            return JSONResponse({"state": "idle", "error": str(exc)})

    # ── Punto 8: user model por sesión ──────────────────────────────────

    @app.get("/api/user/{sid}")
    async def api_user_model(sid: str) -> JSONResponse:
        um = state.user_models.get(sid)
        if um is None:
            return JSONResponse({"exists": False})
        return JSONResponse({"exists": True, **um.to_dict()})

    # ── Punto 15: goals system ──────────────────────────────────────────

    @app.get("/api/goals")
    async def api_goals() -> JSONResponse:
        if state.goals_manager is None:
            return JSONResponse({"enabled": False})
        try:
            return JSONResponse({"enabled": True, **state.goals_manager.snapshot()})
        except Exception as exc:
            return JSONResponse({"enabled": False, "error": str(exc)})

    @app.post("/api/goals/add")
    async def api_goal_add(payload: Dict[str, Any]) -> JSONResponse:
        if state.goals_manager is None:
            raise HTTPException(status_code=503, detail="goals not enabled")
        title = str(payload.get("title", "")).strip()
        if not title:
            raise HTTPException(status_code=400, detail="title required")
        kind = payload.get("kind", "goal")
        from agent.goals import GoalSource
        source = payload.get("source", GoalSource.USER.value)
        if kind == "task":
            t = state.goals_manager.add_task(title=title, source=source)
            return JSONResponse({"created": "task", **t.to_dict()})
        if kind == "mission":
            m = state.goals_manager.add_mission(title=title)
            return JSONResponse({"created": "mission", **m.to_dict()})
        g = state.goals_manager.add_goal(title=title, source=source)
        return JSONResponse({"created": "goal", **g.to_dict()})

    @app.post("/api/goals/approve/{gid}")
    async def api_goal_approve(gid: str) -> JSONResponse:
        if state.goals_manager is None:
            raise HTTPException(status_code=503, detail="goals not enabled")
        ok = state.goals_manager.approve_goal(gid)
        return JSONResponse({"approved": ok})

    @app.post("/api/goals/reject/{gid}")
    async def api_goal_reject(gid: str) -> JSONResponse:
        if state.goals_manager is None:
            raise HTTPException(status_code=503, detail="goals not enabled")
        ok = state.goals_manager.reject_goal(gid)
        return JSONResponse({"rejected": ok})

    # ── Parte 22: adapters ──────────────────────────────────────────
    @app.get("/api/adapters")
    async def api_adapters_list(motor: Optional[str] = None) -> JSONResponse:
        """Lista adapters registrados. Filtrable por motor."""
        if state.adapter_registry is None:
            return JSONResponse({"enabled": False, "adapters": []})
        try:
            metas = state.adapter_registry.list(motor_name=motor)
            items = [
                {
                    "concept": m.concept_name,
                    "motor": m.motor_name,
                    "rank": m.rank,
                    "alpha": m.alpha,
                    "params": m.num_params,
                    "size_bytes": m.size_bytes,
                    "created_at": m.created_at,
                    "usage_count": m.usage_count,
                    "reward_score": m.reward_score,
                    "tags": m.tags,
                }
                for m in metas
            ]
            return JSONResponse({
                "enabled": True,
                "count": len(items),
                "total_bytes": sum(m.size_bytes for m in metas),
                "adapters": items,
            })
        except Exception as exc:
            return JSONResponse({"enabled": False, "error": str(exc)})

    @app.delete("/api/adapters/{motor}/{concept}")
    async def api_adapters_delete(motor: str, concept: str) -> JSONResponse:
        if state.adapter_registry is None:
            raise HTTPException(status_code=503, detail="adapter registry not enabled")
        ok = state.adapter_registry.delete(motor, concept)
        return JSONResponse({"deleted": ok, "motor": motor, "concept": concept})

    # ── Parte 27: sparse activation ────────────────────────────────
    @app.get("/api/sparse/report")
    async def api_sparse_report() -> JSONResponse:
        """Devuelve el último reporte de sparsity por capa (si hay tracker)."""
        if state.sparsity_tracker is None:
            return JSONResponse({"enabled": False})
        try:
            return JSONResponse({"enabled": True, **state.sparsity_tracker.collect()})
        except Exception as exc:
            return JSONResponse({"enabled": False, "error": str(exc)})

    # ── Parte 26: hierarchical memory ──────────────────────────────
    @app.get("/api/memory/hierarchy")
    async def api_memory_hierarchy() -> JSONResponse:
        """Resumen de memoria jerárquica por nivel."""
        store = state.hierarchical_store
        if store is None:
            return JSONResponse({"enabled": False})
        from compression import MemoryLevel
        levels = {}
        for level in MemoryLevel:
            items = store.list_by_level(level)
            levels[level.value] = {
                "count": len(items),
                "items": [it.to_dict() for it in items[:20]],
            }
        return JSONResponse({"enabled": True, "levels": levels, "total": len(store)})

    # ── Parte 25: reward ───────────────────────────────────────────
    @app.post("/api/feedback")
    async def api_feedback(payload: Dict[str, Any]) -> JSONResponse:
        """Registra feedback explícito (👍/👎/corrección) sobre un mensaje.

        Body:
            {
              "vote": "up"|"down"|"correction",
              "motor": "forge_c",                # opcional
              "adapter": "python",                # opcional
              "episode_offset": 0                 # índice negativo del buffer
            }

        Efectos:
            - Actualiza el RewardLedger por clave motor[:adapter]
            - Si hay un episodio en el buffer, le escribe user_feedback
            - Si el adapter existe en el AdapterRegistry, actualiza su
              reward_score con media móvil simple.
        """
        if state.reward_estimator is None:
            return JSONResponse({"enabled": False})
        vote = str(payload.get("vote", "")).lower()
        if vote not in ("up", "down", "correction"):
            raise HTTPException(status_code=400, detail="vote must be up/down/correction")
        motor = payload.get("motor")
        adapter = payload.get("adapter")
        episode_offset = int(payload.get("episode_offset", -1))

        from reward import (
            RewardSignals, ExplicitSignal, ImplicitSignals, IntrinsicSignals,
        )
        explicit = {
            "up": ExplicitSignal.UP,
            "down": ExplicitSignal.DOWN,
            "correction": ExplicitSignal.CORRECTION,
        }[vote]
        estimate = state.reward_estimator.compute(RewardSignals(
            explicit=explicit,
            implicit=ImplicitSignals(),
            intrinsic=IntrinsicSignals(),
        ))

        # Ledger
        key = motor or "unknown"
        if adapter:
            key = f"{motor or 'unknown'}:{adapter}"
        if state.reward_ledger is not None:
            state.reward_ledger.add(key, estimate)
            state.save_reward_ledger()

        # Buffer feedback retroactivo
        if state.episodic_buffer is not None and len(state.episodic_buffer) > 0:
            try:
                eps = state.episodic_buffer._episodes
                idx = episode_offset if episode_offset >= 0 else (len(eps) + episode_offset)
                if 0 <= idx < len(eps):
                    eps[idx].user_feedback = vote
            except Exception:
                pass

        # Adapter registry reward_score update (EMA con alpha=0.3)
        if state.adapter_registry is not None and motor and adapter:
            try:
                meta = state.adapter_registry.get_meta(motor, adapter)
                alpha = 0.3
                meta.reward_score = (1 - alpha) * meta.reward_score + alpha * estimate.mean
                meta.usage_count += 1
                state.adapter_registry.update_meta(meta)
            except Exception:
                pass

        return JSONResponse({
            "enabled": True,
            "vote": vote,
            "key": key,
            "estimate": estimate.to_dict(),
            "ledger_mean": (
                state.reward_ledger.mean_for(key)
                if state.reward_ledger is not None else None
            ),
        })

    @app.get("/api/feedback/ledger")
    async def api_feedback_ledger() -> JSONResponse:
        if state.reward_ledger is None:
            return JSONResponse({"enabled": False})
        return JSONResponse({"enabled": True, "ledger": state.reward_ledger.snapshot()})

    # ── Parte 23: sleep cycle ──────────────────────────────────────
    @app.post("/api/sleep")
    async def api_sleep_force() -> JSONResponse:
        """Dispara manualmente un sleep cycle (ritual de 6 preguntas)."""
        if state.sleep_daemon is None:
            return JSONResponse({"enabled": False})
        log = state.sleep_daemon.force_run()
        # Persistir hierarchy tras el ciclo (compress_hook muta el store)
        state.save_hierarchy()
        return JSONResponse({"enabled": True, **log.to_dict()})

    @app.get("/api/sleep/last")
    async def api_sleep_last() -> JSONResponse:
        """Devuelve el log del último sleep cycle, si existe."""
        if state.sleep_daemon is None:
            return JSONResponse({"enabled": False})
        log = state.sleep_daemon.last_log
        if log is None:
            buf = state.episodic_buffer
            buf_size = len(buf) if buf is not None else 0
            return JSONResponse({
                "enabled": True, "last_log": None, "buffer_size": buf_size,
            })
        buf_size = len(state.episodic_buffer) if state.episodic_buffer is not None else 0
        return JSONResponse({
            "enabled": True,
            "last_log": log.to_dict(),
            "buffer_size": buf_size,
            "inactivity_seconds": state.sleep_daemon.inactivity_seconds,
        })

    @app.post("/api/sleep/episode")
    async def api_sleep_add_episode(payload: Dict[str, Any]) -> JSONResponse:
        """Añade un episodio al buffer manualmente (útil para tests/UI)."""
        if state.episodic_buffer is None:
            return JSONResponse({"enabled": False})
        from sleep import Episode
        ep = Episode(
            user_text=str(payload.get("user_text", "")),
            aion_response=str(payload.get("aion_response", "")),
            motor_sequence=list(payload.get("motor_sequence", [])),
            user_feedback=payload.get("user_feedback"),
        )
        state.episodic_buffer.add(ep)
        if state.sleep_daemon is not None:
            state.sleep_daemon.notify_activity()
        return JSONResponse({"enabled": True, "buffer_size": len(state.episodic_buffer)})

    # ── Parte 22.5: trayectorias compuestas ─────────────────────────
    @app.post("/api/trajectory/plan")
    async def api_trajectory_plan(payload: Dict[str, Any]) -> JSONResponse:
        """Planea (sin ejecutar) una trayectoria compuesta para una query.

        Devuelve la secuencia de motores + sub-goals + rationale para que
        la UI la muestre ANTES de ejecutar.
        """
        if state.trajectory_planner is None:
            return JSONResponse({"enabled": False})
        query = str(payload.get("query", "")).strip()
        if not query:
            raise HTTPException(status_code=400, detail="query required")
        try:
            traj = state.trajectory_planner.plan(query)
            return JSONResponse({"enabled": True, **traj.to_dict()})
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.post("/api/trajectory/execute")
    async def api_trajectory_execute(payload: Dict[str, Any]) -> JSONResponse:
        """Planea + ejecuta una trayectoria compuesta con el modelo cargado.

        Si no hay modelo real usa una generate_fn de eco determinista.
        """
        if state.trajectory_planner is None:
            return JSONResponse({"enabled": False})
        query = str(payload.get("query", "")).strip()
        if not query:
            raise HTTPException(status_code=400, detail="query required")
        try:
            traj = state.trajectory_planner.plan(query)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        from composition import CompositeOrchestrator

        def _echo_gen(motor: str, prompt: str, max_tokens: int) -> str:
            # Fallback sin modelo real.
            return f"[{motor}] {query[: max_tokens]}"

        gen_fn = _echo_gen
        if state.model is not None and state.tokenizer is not None:
            # Wrapper sync sobre el pipeline real (no-streaming, decode greedy).
            def _real_gen(motor: str, prompt: str, max_tokens: int) -> str:
                try:
                    import torch
                    from synth.canonical_dataloader import EOS_TOKEN_ID
                    pipeline = state.model
                    tok = state.tokenizer
                    try:
                        ids = tok.encode(prompt, 96)
                    except TypeError:
                        ids = tok.encode(prompt)[:96]
                    cur = torch.tensor([ids], dtype=torch.long)
                    plen = len(ids)
                    pipeline.eval()
                    with torch.no_grad():
                        out = pipeline(cur)
                        for _ in range(max_tokens):
                            nxt = int(out.logits[0, -1].argmax().item())
                            if nxt in (0, EOS_TOKEN_ID):
                                break
                            cur = torch.cat([cur, torch.tensor([[nxt]])], dim=1)
                            if cur.shape[1] >= 128:
                                break
                            out = pipeline(cur)
                    try:
                        return tok.decode(cur[0, plen:].tolist())
                    except Exception:
                        return ""
                except Exception as exc:
                    return f"[gen error: {exc}]"
            gen_fn = _real_gen

        result = CompositeOrchestrator(gen_fn).execute(traj)
        return JSONResponse({"enabled": True, **result.to_dict()})

    # ── Lifecycle hooks: SleepDaemon background loop ──────────────
    @app.on_event("startup")
    async def _startup_sleep_daemon() -> None:
        if state.sleep_daemon is not None and state._sleep_task is None:
            state._sleep_task = asyncio.create_task(_sleep_daemon_loop(state))

    @app.on_event("shutdown")
    async def _shutdown_sleep_daemon() -> None:
        task = state._sleep_task
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            state._sleep_task = None

    # ── WebSocket: streaming chat ────────────────────────────────────

    @app.websocket("/ws/chat/{sid}")
    async def ws_chat(websocket: WebSocket, sid: str) -> None:
        await websocket.accept()
        try:
            while True:
                raw = await websocket.receive_text()
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    await websocket.send_text(
                        ChatStreamMessage("error", {"detail": "invalid json"}).to_json()
                    )
                    continue
                user_text = str(msg.get("text", "")).strip()
                if not user_text:
                    continue

                # Punto 7: Lifecycle ACTIVE
                if state.lifecycle is not None:
                    try:
                        if state.lifecycle.state.value != "active":
                            state.lifecycle.start_responding("user query")
                    except Exception:
                        pass

                # Asegurar sesión + ConversationHistory (Punto 3)
                _ensure_session(state, sid)

                # Punto 8: actualizar UserModel desde el input
                _maybe_update_user_model(state, sid, user_text)

                # Agregar turno user al historial
                state.sessions[sid].add_user(user_text)

                # Decide nivel + routing
                level_info = _decide_level(user_text)
                routing = _routing_for(state, user_text)
                top_motor = max(routing.to_dict().items(), key=lambda x: x[1])[0]

                # Punto 6: ResponseCache hit?
                cached_response = None
                if state.response_cache is not None:
                    try:
                        cached_response = state.response_cache.get(user_text)
                    except Exception:
                        cached_response = None

                # Emitir meta
                await websocket.send_text(
                    ChatStreamMessage("meta", {
                        "level":   level_info,
                        "scores":  routing.to_dict(),
                        "motor":   top_motor,
                        "user":    user_text,
                        "cached":  cached_response is not None,
                    }).to_json()
                )

                if level_info["show_thinking"]:
                    await websocket.send_text(
                        ChatStreamMessage("thinking", {
                            "label":      level_info["label"],
                            "iterations": level_info["iterations"],
                        }).to_json()
                    )

                # Punto 11: ¿es tarea multi-step? → Planner
                plan_dict = None
                if state.planner is not None and _detect_multi_step(user_text):
                    try:
                        plan = state.planner.plan(user_text)
                        plan_dict = plan.to_dict()
                        await websocket.send_text(
                            ChatStreamMessage("plan", plan_dict).to_json()
                        )
                    except Exception:
                        pass

                # Punto 12: World Model — simulate→verify→re-simulate
                sim_result = _run_world_model_simulation(state, user_text, top_motor)
                if sim_result is not None:
                    await websocket.send_text(
                        ChatStreamMessage("scratchpad", sim_result).to_json()
                    )

                # Punto 1, 3, 8, 10, 16: build canonical prompt
                canonical_prompt = _build_canonical_prompt(state, sid, user_text)

                # Stream tokens (cache o real)
                response_parts: List[str] = []
                if cached_response is not None:
                    # Emite el cache token-by-token simulando streaming
                    for chunk in cached_response.split():
                        response_parts.append(chunk + " ")
                        await websocket.send_text(
                            ChatStreamMessage("token", {"text": chunk + " "}).to_json()
                        )
                        await asyncio.sleep(0)
                else:
                    async for token in _stream_real_model(state, canonical_prompt):
                        response_parts.append(token)
                        await websocket.send_text(
                            ChatStreamMessage("token", {"text": token}).to_json()
                        )
                full_response = "".join(response_parts).strip()

                # Punto 6: cache la respuesta
                if state.response_cache is not None and cached_response is None:
                    try:
                        state.response_cache.set(user_text, full_response)
                    except Exception:
                        pass

                # Punto 2: ejecutar tools que el modelo emitió en el output
                tool_records = _execute_tools_in_response(state, full_response)
                for tr in tool_records:
                    await websocket.send_text(
                        ChatStreamMessage("tool", tr).to_json()
                    )

                # Punto 4 + 13: graph causal del query (con SymbolicEngine aplicado)
                graph_dict = _build_graph_for_query(state, user_text, top_motor)
                if graph_dict["nodes"]:
                    await websocket.send_text(
                        ChatStreamMessage("graph", graph_dict).to_json()
                    )

                # Punto 5: SelfChecker + confidence policy
                check = _self_check(state, user_text, full_response)
                await websocket.send_text(
                    ChatStreamMessage("check", check).to_json()
                )

                # Persist assistant turn en ConversationHistory
                state.sessions[sid].add_assistant(
                    full_response,
                    motor=top_motor,
                    level=level_info["label"],
                    scores=routing.to_dict(),
                )

                # Parte 23: añadir episodio al buffer + notify activity al daemon
                if state.episodic_buffer is not None:
                    try:
                        from sleep import Episode
                        state.episodic_buffer.add(Episode(
                            user_text=user_text,
                            aion_response=full_response,
                            motor_sequence=[top_motor],
                        ))
                        if state.sleep_daemon is not None:
                            state.sleep_daemon.notify_activity()
                    except Exception:
                        pass

                # done — incluye scores y level para fixear desync de UI
                await websocket.send_text(
                    ChatStreamMessage("done", {
                        "response":   full_response,
                        "motor":      top_motor,
                        "scores":     routing.to_dict(),
                        "level":      level_info["label"],
                        "graph":      graph_dict,
                        "scratchpad": sim_result,
                        "check":      check,
                        "tools":      tool_records,
                        "plan":       plan_dict,
                    }).to_json()
                )

                # Punto 7: Lifecycle IDLE
                if state.lifecycle is not None:
                    try:
                        state.lifecycle.stop_responding("response complete")
                    except Exception:
                        pass
        except WebSocketDisconnect:
            return
        except Exception as exc:
            try:
                await websocket.send_text(
                    ChatStreamMessage("error", {"detail": str(exc)}).to_json()
                )
            except Exception:
                pass

    return app


def _ensure_session(state: AppState, sid: str) -> None:
    """Crea una ConversationHistory para la sesión si no existe (Punto 3)."""
    from memory.conversation_history import ConversationHistory
    if sid not in state.sessions:
        state.sessions[sid] = ConversationHistory(recent_window=6, mid_window=20)


def _history_to_list(history: Any) -> List[Dict[str, Any]]:
    """
    Convierte una ConversationHistory en lista de turnos serializables
    (compatibilidad con el formato que el frontend esperaba antes).
    """
    if history is None:
        return []
    if isinstance(history, list):
        return history  # backwards compat para tests viejos
    out: List[Dict[str, Any]] = []
    if hasattr(history, "turns"):
        for t in history.turns:
            out.append({
                "role":    t.role,
                "content": t.content,
                "ts":      t.timestamp,
                **{k: v for k, v in (t.metadata or {}).items()},
            })
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


def build_full_state(load_model: bool = True) -> AppState:
    """
    Construye un AppState COMPLETAMENTE WIRED con todos los componentes
    de los puntos 1-16 conectados al backend.

    Args:
        load_model: si True intenta cargar checkpoints/tiny_canonical.pt
                    si False usa modo mock (sin torch)

    Wiring:
      Punto 1   → SkillsLoader cargado con skills/*.md, attach_to_mem
      Punto 2   → ToolExecutor con todas las tools registradas
      Punto 3   → ConversationHistory por sesión (al crearse)
      Punto 4   → graph builder + emit en el WS handler
      Punto 5   → SelfChecker
      Punto 6   → ResponseCache LRU
      Punto 7   → LifecycleManager (idle inicial)
      Punto 8   → user_models dict por sesión
      Punto 10/16 → identity_skill_text cargado de skills/identity.md
      Punto 11  → Planner
      Punto 12  → 5 WorldSimulators (build_default_simulators)
      Punto 13  → SymbolicEngine por motor (axiom/forge_c/cora)
      Punto 15  → GoalsManager con misión permanente
    """
    from pathlib import Path as _Path
    repo = _Path(__file__).resolve().parent.parent

    # ── Modelo + tokenizer (opcional) ──────────────────────────────────
    pipeline = None
    tok = None
    n_params = 0
    if load_model:
        try:
            import torch
            from router.pipeline import MoSEPipeline, MoSEConfig
            from experiments.train_production import build_tokenizer
            tok = build_tokenizer(32_000)
            cfg = MoSEConfig(
                hidden_dim=64, vocab_size=tok.vocab_size,
                enc_n_layers=2, enc_state_dim=4, enc_expand=2, enc_d_conv=4, enc_ffn_mult=2,
                orch_mlp_hidden=32, orch_max_motors=3, orch_min_confidence=0.3,
                motor_max_nodes=8, motor_n_heads=4, motor_threshold=0.01, unif_n_heads=4,
                dec_n_layers=2, dec_n_heads=4, dec_max_seq_len=128,
                dec_state_dim=4, dec_expand=2, dec_d_conv=4, dec_ffn_mult=2,
            )
            pipeline = MoSEPipeline(cfg)
            ckpt_path = repo / "checkpoints" / "tiny_canonical.pt"
            if ckpt_path.exists():
                ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
                pipeline.load_state_dict(ck["model_state"], strict=False)
                print(f"Loaded {ckpt_path.name} (routing_acc={ck.get('routing_acc', 0):.1f}%)")
            else:
                print(f"WARN: {ckpt_path} not found, using random weights")
            pipeline.eval()
            n_params = sum(p.numel() for p in pipeline.parameters())
        except Exception as exc:
            print(f"WARN: model load failed: {exc}")
            pipeline = None
            tok = None

    # ── MEM (SemanticStore o mock) ─────────────────────────────────────
    mem = None
    try:
        from memory.semantic_store import SemanticStore
        if pipeline is not None:
            mem = SemanticStore(encoder=pipeline.encoder, tokenizer=tok, similarity_threshold=0.0)
            mem._device = "cpu"
        else:
            mem = SemanticStore(encoder=None, tokenizer=None, similarity_threshold=0.0)
    except Exception:
        mem = None

    # ── Punto 1, 10, 16: SkillsLoader + identity ───────────────────────
    skills_loader = None
    identity_skill_text = ""
    try:
        from agent.skills import SkillsLoader
        skills_dir = repo / "skills"
        skills_loader = SkillsLoader(threshold=0.5)
        loaded = skills_loader.load_dir(skills_dir)
        if mem is not None:
            skills_loader.attach_to_mem(mem)
        # Identity siempre disponible (Punto 10/16)
        if "identity" in skills_loader.skills:
            identity_skill_text = skills_loader.skills["identity"].content[:400]
        print(f"Loaded {len(loaded)} skills (identity: {'yes' if identity_skill_text else 'no'})")
    except Exception as exc:
        print(f"WARN: skills load failed: {exc}")

    # ── Punto 2: ToolExecutor con tools registradas ────────────────────
    tool_executor = None
    try:
        from agent.tools import build_tool_registry
        from agent.tool_executor import ToolExecutor
        registry = build_tool_registry(
            output_root=repo / "output",
            allowed_domains=set(),  # deny-all por seguridad
            mem=mem,
        )
        tool_executor = ToolExecutor(registry)
    except Exception as exc:
        print(f"WARN: tool executor init failed: {exc}")

    # ── Punto 5: SelfChecker ───────────────────────────────────────────
    self_checker = None
    try:
        from agent.self_check import SelfChecker
        self_checker = SelfChecker()
    except Exception:
        pass

    # ── Punto 6: ResponseCache ─────────────────────────────────────────
    response_cache = None
    try:
        from memory.response_cache import ResponseCache
        response_cache = ResponseCache(max_size=256)
    except Exception:
        pass

    # ── Punto 7: LifecycleManager ──────────────────────────────────────
    lifecycle = None
    try:
        from agent.lifecycle import LifecycleManager, SystemState
        lifecycle = LifecycleManager(initial_state=SystemState.IDLE)
    except Exception:
        pass

    # ── Punto 11: Planner ──────────────────────────────────────────────
    planner = None
    try:
        from agent.planner import Planner
        planner = Planner()
    except Exception:
        pass

    # ── Punto 12: WorldSimulators por motor ────────────────────────────
    world_simulators: Dict[str, Any] = {}
    try:
        from world_model.simulator import build_default_simulators
        world_simulators = build_default_simulators()
    except Exception:
        pass

    # ── Punto 13: SymbolicEngine por motor ─────────────────────────────
    symbolic_engines: Dict[str, Any] = {}
    try:
        from symbolic.engine import build_engine_for_motor
        for motor in ("axiom", "forge_c", "cora"):
            symbolic_engines[motor] = build_engine_for_motor(motor)
    except Exception:
        pass

    # ── Parte 22: AdapterRegistry (brain/adapters/) ────────────────────
    adapter_registry = None
    try:
        from growth import AdapterRegistry
        adapter_registry = AdapterRegistry(repo / "brain")
    except Exception as exc:
        print(f"WARN: adapter registry init failed: {exc}")

    # ── Parte 22.5: TrajectoryPlanner ──────────────────────────────────
    trajectory_planner = None
    try:
        from composition import TrajectoryPlanner
        trajectory_planner = TrajectoryPlanner()
    except Exception as exc:
        print(f"WARN: trajectory planner init failed: {exc}")

    # ── Parte 23+24+25+26: Sleep Cycle con pruner/reward/compressor ───
    episodic_buffer = None
    sleep_daemon = None
    reward_estimator = None
    reward_ledger = None
    hierarchical_store = None
    hierarchical_compressor = None
    brain_v1 = repo / "brain" / "v1"
    reward_ledger_path = brain_v1 / "reward_ledger.jsonl"
    hierarchy_path = brain_v1 / "hierarchy.jsonl"
    try:
        from sleep import EpisodicBuffer, SleepCycle, SleepDaemon
        from pruning import MemoryPruner, sleep_prune_hook
        from reward import RewardEstimator, RewardLedger, sleep_reward_hook
        from compression import (
            HierarchicalStore, HierarchicalCompressor, Clusterer,
            sleep_compress_hook,
        )
        episodic_buffer = EpisodicBuffer(max_size=1000)
        reward_estimator = RewardEstimator()
        reward_ledger = RewardLedger()
        # Restore ledger desde disco si existe
        try:
            reward_ledger.load_jsonl(reward_ledger_path)
            if reward_ledger.keys():
                print(f"Loaded reward ledger: {len(reward_ledger.keys())} keys from {reward_ledger_path.name}")
        except Exception as exc:
            print(f"WARN: reward ledger load failed: {exc}")

        hierarchical_store = HierarchicalStore()
        try:
            hierarchical_store.load_jsonl(hierarchy_path)
            if len(hierarchical_store) > 0:
                print(f"Loaded hierarchy: {len(hierarchical_store)} items from {hierarchy_path.name}")
        except Exception as exc:
            print(f"WARN: hierarchy load failed: {exc}")

        hierarchical_compressor = HierarchicalCompressor(
            hierarchical_store,
            Clusterer(threshold=0.3),
        )
        sleep_cycle = SleepCycle(
            episodic_buffer,
            reward_hook=sleep_reward_hook(reward_estimator),
            prune_hook=sleep_prune_hook(MemoryPruner()),
            compress_hook=sleep_compress_hook(hierarchical_compressor),
        )
        sleep_daemon = SleepDaemon(
            sleep_cycle,
            inactivity_seconds=1800.0,  # 30 min (configurable vía env AION_SLEEP_INACTIVITY)
            overflow_threshold=500,
        )
        import os as _os
        try:
            sleep_daemon.inactivity_seconds = float(
                _os.environ.get("AION_SLEEP_INACTIVITY", sleep_daemon.inactivity_seconds)
            )
        except Exception:
            pass
    except Exception as exc:
        print(f"WARN: sleep daemon init failed: {exc}")

    # ── Punto 15: GoalsManager + misión permanente ─────────────────────
    goals_manager = None
    try:
        from agent.goals import GoalsManager, GoalSource
        goals_manager = GoalsManager()
        # Tarea housekeeping inicial: indexar MEM
        goals_manager.add_housekeeping_task("indexar MEM al arrancar")
        goals_manager.log_routine_entry("startup", "system initialized")
    except Exception:
        pass

    return AppState(
        model=pipeline,
        tokenizer=tok,
        mem=mem,
        n_params=n_params,
        skills_loader=skills_loader,
        response_cache=response_cache,
        tool_executor=tool_executor,
        self_checker=self_checker,
        lifecycle=lifecycle,
        planner=planner,
        world_simulators=world_simulators,
        symbolic_engines=symbolic_engines,
        goals_manager=goals_manager,
        identity_skill_text=identity_skill_text,
        adapter_registry=adapter_registry,
        trajectory_planner=trajectory_planner,
        episodic_buffer=episodic_buffer,
        sleep_daemon=sleep_daemon,
        reward_estimator=reward_estimator,
        reward_ledger=reward_ledger,
        hierarchical_store=hierarchical_store,
        reward_ledger_path=reward_ledger_path,
        hierarchy_path=hierarchy_path,
    )


# Backwards compatibility
def _build_real_state() -> AppState:
    return build_full_state(load_model=True)


def main() -> None:
    import uvicorn
    state = _build_real_state()
    app = create_app(state)
    print("AION-C web UI: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
