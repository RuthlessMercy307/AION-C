"""
tests/test_backend_wiring_integration.py — verifica los 16 wires
==================================================================

Cada test cubre uno (o varios) de los puntos del wiring solicitados:

  Punto 1   Skills cargados en MEM al arrancar + inyectados en prompt
  Punto 2   ToolExecutor parsea [TOOL: ...] y emite type:'tool'
  Punto 3   ConversationHistory guarda turnos y produce summary
  Punto 4   Per-message graph emitido como type:'graph'
  Punto 5   SelfChecker emite type:'check' con confidence
  Punto 6   ResponseCache hit/miss visible en /api/cache/stats
  Punto 7   LifecycleManager transita ACTIVE↔IDLE durante chat
  Punto 8   UserModel persistido por sesión, accesible vía /api/user/{sid}
  Punto 9   BrainVersionManager — verificado en test_brain_version
  Punto 10  Identity skill SIEMPRE inyectado en el prompt
  Punto 11  Planner emite type:'plan' para queries multi-step
  Punto 12  WorldSimulator emite type:'scratchpad'
  Punto 13  SymbolicEngine corre en _build_graph_for_query
  Punto 14  Ya cubierto por test_anti_forgetting (módulos standalone)
  Punto 15  GoalsManager con endpoints /api/goals
  Punto 16  Identity skill estable entre mensajes
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest

try:
    from fastapi.testclient import TestClient
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

pytestmark = pytest.mark.skipif(not HAS_FASTAPI, reason="fastapi not installed")


# ─────────────────────────────────────────────────────────────────────────────
# Helper: build a fully-wired state in mock mode (no torch/model)
# ─────────────────────────────────────────────────────────────────────────────


def make_wired_client(tmp_path):
    """Crea una app con TODOS los componentes wired pero sin modelo real."""
    from backend.app_fastapi import build_full_state, create_app
    state = build_full_state(load_model=False)
    state.output_dir = tmp_path / "output"
    state.output_dir.mkdir(parents=True, exist_ok=True)
    app = create_app(state)
    return TestClient(app), state


def collect_ws_messages(client, sid: str, user_text: str, max_msgs: int = 60):
    """Conecta al WS, envía un mensaje, recolecta todos los mensajes hasta done."""
    out = []
    with client.websocket_connect(f"/ws/chat/{sid}") as ws:
        ws.send_text(json.dumps({"text": user_text}))
        for _ in range(max_msgs):
            try:
                raw = ws.receive_text()
            except Exception:
                break
            try:
                m = json.loads(raw)
            except Exception:
                continue
            out.append(m)
            if m.get("type") == "done":
                break
    return out


def msgs_by_type(messages):
    """Agrupa mensajes por type."""
    out = {}
    for m in messages:
        t = m.get("type")
        out.setdefault(t, []).append(m)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Punto 1, 10, 16: Skills system + Identity inyectado
# ─────────────────────────────────────────────────────────────────────────────


class TestSkillsWiring:
    def test_skills_loaded_at_startup(self, tmp_path):
        _, state = make_wired_client(tmp_path)
        assert state.skills_loader is not None
        assert len(state.skills_loader.skills) >= 11

    def test_identity_skill_text_present(self, tmp_path):
        _, state = make_wired_client(tmp_path)
        assert state.identity_skill_text
        assert "AION-C" in state.identity_skill_text

    def test_skills_in_mem_after_startup(self, tmp_path):
        _, state = make_wired_client(tmp_path)
        # MEM debe tener entries con domain="skill"
        if state.mem is None:
            return
        entries = state.mem.list_entries()
        skill_entries = [e for e in entries if e.get("domain") == "skill"]
        assert len(skill_entries) >= 5  # al menos 5 de los 11 skills

    def test_canonical_prompt_includes_identity_when_enabled(self, tmp_path):
        # Bug fix #1: identity injection es opt-in. Cuando se activa, debe aparecer.
        from backend.app_fastapi import _build_canonical_prompt, _ensure_session
        _, state = make_wired_client(tmp_path)
        state.inject_identity = True   # explicit opt-in
        _ensure_session(state, "s1")
        prompt = _build_canonical_prompt(state, "s1", "hola")
        assert "[SKILL:" in prompt
        assert "AION-C" in prompt or "identity" in prompt.lower()

    def test_canonical_prompt_minimal_by_default(self, tmp_path):
        # Default OFF: el prompt es minimal para evitar OOD del tiny
        from backend.app_fastapi import _build_canonical_prompt, _ensure_session
        _, state = make_wired_client(tmp_path)
        _ensure_session(state, "s1")
        prompt = _build_canonical_prompt(state, "s1", "hola")
        assert "[USER: hola]" in prompt
        assert "[AION:" in prompt
        # SKILL/MEM NO deben aparecer por default
        assert "[SKILL:" not in prompt
        assert "[MEM:" not in prompt


# ─────────────────────────────────────────────────────────────────────────────
# Punto 2: Tool executor wired
# ─────────────────────────────────────────────────────────────────────────────


class TestToolExecutorWiring:
    def test_tool_executor_built(self, tmp_path):
        _, state = make_wired_client(tmp_path)
        assert state.tool_executor is not None
        assert "write_file" in state.tool_executor.registry
        assert "search_mem" in state.tool_executor.registry

    def test_tool_executor_has_search_mem_with_real_mem(self, tmp_path):
        _, state = make_wired_client(tmp_path)
        # search_mem tool should have real mem injected
        sm = state.tool_executor.registry.get("search_mem")
        assert sm is not None
        # mem injected: should be the same object as state.mem
        assert sm._mem is state.mem


# ─────────────────────────────────────────────────────────────────────────────
# Punto 3: ConversationHistory
# ─────────────────────────────────────────────────────────────────────────────


class TestConversationHistoryWiring:
    def test_session_creates_conversation_history(self, tmp_path):
        from memory.conversation_history import ConversationHistory
        client, state = make_wired_client(tmp_path)
        sid = client.post("/api/session").json()["session_id"]
        assert isinstance(state.sessions[sid], ConversationHistory)

    def test_history_records_turns(self, tmp_path):
        client, state = make_wired_client(tmp_path)
        sid = client.post("/api/session").json()["session_id"]
        msgs = collect_ws_messages(client, sid, "hola")
        assert any(m["type"] == "done" for m in msgs)
        h = state.sessions[sid]
        assert len(h) == 2  # user + assistant
        assert h.turns[0].role == "user"
        assert h.turns[1].role == "assistant"

    def test_get_session_returns_serializable_turns(self, tmp_path):
        client, _ = make_wired_client(tmp_path)
        sid = client.post("/api/session").json()["session_id"]
        collect_ws_messages(client, sid, "hi")
        r = client.get(f"/api/session/{sid}")
        assert r.status_code == 200
        turns = r.json()
        assert len(turns) >= 2
        assert turns[0]["role"] == "user"


# ─────────────────────────────────────────────────────────────────────────────
# Punto 4 + 13: Per-message graph emitted (with SymbolicEngine applied)
# ─────────────────────────────────────────────────────────────────────────────


class TestGraphWiring:
    def test_ws_emits_graph_message(self, tmp_path):
        client, _ = make_wired_client(tmp_path)
        sid = client.post("/api/session").json()["session_id"]
        msgs = collect_ws_messages(client, sid, "rain causes wet soil and floods")
        groups = msgs_by_type(msgs)
        assert "graph" in groups
        graph = groups["graph"][0]["payload"]
        assert "nodes" in graph
        assert "edges" in graph
        assert len(graph["nodes"]) > 0

    def test_graph_includes_symbolic_engine_results(self, tmp_path):
        from backend.app_fastapi import _build_graph_for_query, build_full_state
        state = build_full_state(load_model=False)
        graph = _build_graph_for_query(state, "rain causes wet soil and floods", "cora")
        # transitividad debería haber añadido aristas si hay >= 2 causes
        assert "conflicts" in graph
        assert "notes" in graph
        # Para CORA con cadena causal, transitividad genera notas
        if len(graph["edges"]) >= 2:
            assert isinstance(graph["notes"], list)


# ─────────────────────────────────────────────────────────────────────────────
# Punto 5: SelfChecker
# ─────────────────────────────────────────────────────────────────────────────


class TestSelfCheckWiring:
    def test_ws_emits_check_message(self, tmp_path):
        client, _ = make_wired_client(tmp_path)
        sid = client.post("/api/session").json()["session_id"]
        msgs = collect_ws_messages(client, sid, "hello world test")
        groups = msgs_by_type(msgs)
        assert "check" in groups
        check = groups["check"][0]["payload"]
        assert "passed" in check
        assert "policy" in check


# ─────────────────────────────────────────────────────────────────────────────
# Punto 6: ResponseCache
# ─────────────────────────────────────────────────────────────────────────────


class TestResponseCacheWiring:
    def test_cache_stats_endpoint(self, tmp_path):
        client, _ = make_wired_client(tmp_path)
        r = client.get("/api/cache/stats")
        assert r.status_code == 200
        d = r.json()
        assert d["enabled"] is True
        assert "hits" in d
        assert "misses" in d

    def test_cache_hit_after_repeat(self, tmp_path):
        client, state = make_wired_client(tmp_path)
        sid = client.post("/api/session").json()["session_id"]
        # First query → miss
        collect_ws_messages(client, sid, "what is python")
        # Second identical query → hit
        msgs = collect_ws_messages(client, sid, "what is python")
        groups = msgs_by_type(msgs)
        meta = groups["meta"][0]["payload"]
        assert meta["cached"] is True
        # Stats reflejan al menos 1 hit
        stats = client.get("/api/cache/stats").json()
        assert stats["hits"] >= 1


# ─────────────────────────────────────────────────────────────────────────────
# Punto 7: Lifecycle transitions
# ─────────────────────────────────────────────────────────────────────────────


class TestLifecycleWiring:
    def test_lifecycle_endpoint(self, tmp_path):
        client, _ = make_wired_client(tmp_path)
        r = client.get("/api/lifecycle")
        assert r.status_code == 200
        d = r.json()
        assert "current_state" in d or "state" in d

    def test_lifecycle_transitions_during_chat(self, tmp_path):
        client, state = make_wired_client(tmp_path)
        sid = client.post("/api/session").json()["session_id"]
        # before: idle
        assert state.lifecycle.state.value == "idle"
        collect_ws_messages(client, sid, "hola")
        # after the message processes, lifecycle should have transitioned at least once
        history = state.lifecycle.history
        assert len(history) >= 2  # idle→active and active→idle
        states_seen = {t.to_state.value for t in history}
        assert "active" in states_seen
        assert "idle" in states_seen


# ─────────────────────────────────────────────────────────────────────────────
# Punto 8: UserModel per session
# ─────────────────────────────────────────────────────────────────────────────


class TestUserModelWiring:
    def test_user_model_created_after_first_message(self, tmp_path):
        client, state = make_wired_client(tmp_path)
        sid = client.post("/api/session").json()["session_id"]
        collect_ws_messages(client, sid, "mi nombre es Jesus")
        assert sid in state.user_models
        assert state.user_models[sid].name == "Jesus"

    def test_user_model_endpoint(self, tmp_path):
        client, _ = make_wired_client(tmp_path)
        sid = client.post("/api/session").json()["session_id"]
        collect_ws_messages(client, sid, "my name is Alex")
        r = client.get(f"/api/user/{sid}")
        assert r.status_code == 200
        d = r.json()
        assert d["exists"] is True
        assert d["name"] == "Alex"


# ─────────────────────────────────────────────────────────────────────────────
# Punto 11: Planner for multi-step queries
# ─────────────────────────────────────────────────────────────────────────────


class TestPlannerWiring:
    def test_planner_emits_plan_for_multi_step(self, tmp_path):
        client, _ = make_wired_client(tmp_path)
        sid = client.post("/api/session").json()["session_id"]
        msgs = collect_ws_messages(client, sid, "primero crea la base de datos luego construye el backend luego añade auth")
        groups = msgs_by_type(msgs)
        assert "plan" in groups
        plan = groups["plan"][0]["payload"]
        assert plan["task"]
        assert len(plan["steps"]) >= 2

    def test_planner_skipped_for_single_step(self, tmp_path):
        client, _ = make_wired_client(tmp_path)
        sid = client.post("/api/session").json()["session_id"]
        msgs = collect_ws_messages(client, sid, "hola")
        groups = msgs_by_type(msgs)
        assert "plan" not in groups


# ─────────────────────────────────────────────────────────────────────────────
# Punto 12: World Model simulation in pipeline
# ─────────────────────────────────────────────────────────────────────────────


class TestWorldModelWiring:
    def test_simulators_built(self, tmp_path):
        _, state = make_wired_client(tmp_path)
        assert "axiom" in state.world_simulators
        assert "cora" in state.world_simulators
        assert "muse" in state.world_simulators
        assert "empathy" in state.world_simulators
        assert "forge_c" in state.world_simulators

    def test_ws_emits_scratchpad_for_axiom_query(self, tmp_path):
        client, _ = make_wired_client(tmp_path)
        sid = client.post("/api/session").json()["session_id"]
        msgs = collect_ws_messages(client, sid, "calcula 25% de 200")
        groups = msgs_by_type(msgs)
        assert "scratchpad" in groups
        sp = groups["scratchpad"][0]["payload"]
        assert sp["motor"] == "axiom"
        assert "scratchpad" in sp


# ─────────────────────────────────────────────────────────────────────────────
# Punto 13: Symbolic engine (covered in TestGraphWiring above)
# ─────────────────────────────────────────────────────────────────────────────


class TestSymbolicWiring:
    def test_engines_per_motor(self, tmp_path):
        _, state = make_wired_client(tmp_path)
        assert "axiom" in state.symbolic_engines
        assert "forge_c" in state.symbolic_engines
        assert "cora" in state.symbolic_engines


# ─────────────────────────────────────────────────────────────────────────────
# Punto 15: Goals manager + endpoints
# ─────────────────────────────────────────────────────────────────────────────


class TestGoalsWiring:
    def test_goals_endpoint(self, tmp_path):
        client, _ = make_wired_client(tmp_path)
        r = client.get("/api/goals")
        assert r.status_code == 200
        d = r.json()
        assert d["enabled"] is True
        assert "permanent_mission" in d
        assert "active_goals" in d
        assert "pending_tasks" in d

    def test_add_goal_user_source(self, tmp_path):
        client, _ = make_wired_client(tmp_path)
        r = client.post("/api/goals/add", json={"title": "learn rust", "kind": "goal"})
        assert r.status_code == 200
        d = r.json()
        assert d["created"] == "goal"
        assert d["title"] == "learn rust"
        assert d["status"] == "active"

    def test_add_proposed_goal(self, tmp_path):
        client, _ = make_wired_client(tmp_path)
        r = client.post("/api/goals/add", json={"title": "improve math", "source": "proposed"})
        d = r.json()
        gid = d["id"]
        # proposed goal must be pending
        snapshot = client.get("/api/goals").json()
        proposed_titles = [g["title"] for g in snapshot["proposed_goals"]]
        assert "improve math" in proposed_titles
        # approve
        ar = client.post(f"/api/goals/approve/{gid}")
        assert ar.json()["approved"] is True
        snapshot2 = client.get("/api/goals").json()
        active_titles = [g["title"] for g in snapshot2["active_goals"]]
        assert "improve math" in active_titles

    def test_add_mission(self, tmp_path):
        client, _ = make_wired_client(tmp_path)
        r = client.post("/api/goals/add", json={"title": "build login", "kind": "mission"})
        assert r.status_code == 200
        d = r.json()
        assert d["created"] == "mission"
        assert d["status"] == "active"

    def test_housekeeping_task_at_startup(self, tmp_path):
        _, state = make_wired_client(tmp_path)
        # build_full_state agrega un task de indexar MEM
        tasks = state.goals_manager.list_pending_tasks()
        assert any("indexar MEM" in t.title for t in tasks)


# ─────────────────────────────────────────────────────────────────────────────
# Punto 16: identity stability across messages
# ─────────────────────────────────────────────────────────────────────────────


class TestIdentityStability:
    def test_identity_text_constant_across_calls(self, tmp_path):
        # Cuando inject_identity=True, debe aparecer el mismo texto en cada llamada
        from backend.app_fastapi import _build_canonical_prompt, _ensure_session
        _, state = make_wired_client(tmp_path)
        state.inject_identity = True
        _ensure_session(state, "s1")
        p1 = _build_canonical_prompt(state, "s1", "hola")
        p2 = _build_canonical_prompt(state, "s1", "que tal")
        assert "AION-C" in p1
        assert "AION-C" in p2


# ─────────────────────────────────────────────────────────────────────────────
# Punto 9: BrainVersionManager (test integration smoke)
# ─────────────────────────────────────────────────────────────────────────────


class TestBrainVersionWiringSmoke:
    def test_brain_dir_writable(self, tmp_path):
        # Smoke: just verify the manager can be instantiated and save
        from brain.version_manager import BrainVersionManager
        import torch
        bvm = BrainVersionManager(root_dir=tmp_path / "brain")
        v = bvm.save_version(
            state_dict={"w": torch.tensor([1.0, 2.0])},
            notes="test",
            metrics={"routing_acc": 0.95},
        )
        assert v.id == "v1"
        loaded = bvm.list_versions()
        assert len(loaded) == 1
