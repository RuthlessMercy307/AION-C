"""
tests/test_backend_fastapi.py — Tests para Fase D backend
============================================================

Cubre la FastAPI app sin cargar el modelo real (usa AppState mockeado).
TestClient de starlette/fastapi maneja HTTP y WebSocket sintéticamente.

Cubre:
  - GET /                 → sirve HTML
  - GET /api/info         → metadata del modelo
  - POST/GET sesiones      → ciclo de vida de sesiones
  - GET /api/mem          → MEM listing
  - POST /api/upload, GET /api/download, GET /api/files → file sandbox
  - WS /ws/chat/{sid}     → handshake + streaming + message types
  - Path traversal rejected en upload/download
  - ChatStreamMessage / RoutingScores serialization
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest

# Detect optional dep
try:
    from fastapi.testclient import TestClient
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


pytestmark = pytest.mark.skipif(not HAS_FASTAPI, reason="fastapi not installed")


# ─────────────────────────────────────────────────────────────────────────────
# Mocks del state
# ─────────────────────────────────────────────────────────────────────────────


class FakeMem:
    def __init__(self):
        self._entries = [
            {"key": "identity", "value": "I am AION-C", "domain": "general"},
            {"key": "fact1",    "value": "rust is fast", "domain": "forge_c"},
        ]
    def list_entries(self, domain=None):
        return list(self._entries)
    def stats(self):
        return {"total_entries": len(self._entries)}
    def search(self, query, top_k=5, domain=None):
        return [(e["key"], e["value"], 0.9) for e in self._entries[:top_k]]
    def store(self, key, value, domain="general", source="test"):
        self._entries.append({"key": key, "value": value, "domain": domain})


def make_state(tmp_path):
    from backend.app_fastapi import AppState
    return AppState(
        model=None,        # mock echo mode
        tokenizer=None,
        mem=FakeMem(),
        output_dir=tmp_path / "output",
        n_params=5_555_449,
    )


def make_client(tmp_path):
    from backend.app_fastapi import create_app
    state = make_state(tmp_path)
    app = create_app(state)
    return TestClient(app), state


# ─────────────────────────────────────────────────────────────────────────────
# Wire types
# ─────────────────────────────────────────────────────────────────────────────


class TestWireTypes:
    def test_routing_scores_to_dict(self):
        from backend.app_fastapi import RoutingScores
        s = RoutingScores(cora=0.8, forge_c=0.05, muse=0.05, axiom=0.05, empathy=0.05)
        d = s.to_dict()
        assert set(d.keys()) == {"cora", "forge_c", "muse", "axiom", "empathy"}
        assert d["cora"] == 0.8

    def test_chat_stream_message_json(self):
        from backend.app_fastapi import ChatStreamMessage
        m = ChatStreamMessage(type="token", payload={"text": "hola"})
        j = json.loads(m.to_json())
        assert j["type"] == "token"
        assert j["payload"]["text"] == "hola"

    def test_chat_stream_message_unicode(self):
        from backend.app_fastapi import ChatStreamMessage
        m = ChatStreamMessage("token", {"text": "qué tal — niño"})
        j = json.loads(m.to_json())
        assert "qué tal" in j["payload"]["text"]


# ─────────────────────────────────────────────────────────────────────────────
# HTTP endpoints
# ─────────────────────────────────────────────────────────────────────────────


class TestRoot:
    def test_serves_html(self, tmp_path):
        client, _ = make_client(tmp_path)
        r = client.get("/")
        assert r.status_code == 200
        assert "AION-C" in r.text


class TestApiInfo:
    def test_info_basic(self, tmp_path):
        client, _ = make_client(tmp_path)
        r = client.get("/api/info")
        assert r.status_code == 200
        d = r.json()
        assert d["name"] == "AION-C"
        assert d["params"] == 5555449
        assert "M" in d["params_human"]
        assert d["motors"] == ["cora", "forge_c", "muse", "axiom", "empathy"]
        assert d["model_loaded"] is False  # mock

    def test_info_with_model_flag(self, tmp_path):
        from backend.app_fastapi import create_app, AppState
        state = AppState(model=object(), tokenizer=None, n_params=42, output_dir=tmp_path / "o")
        client = TestClient(create_app(state))
        d = client.get("/api/info").json()
        assert d["model_loaded"] is True


class TestSessions:
    def test_create_and_get(self, tmp_path):
        client, _ = make_client(tmp_path)
        r = client.post("/api/session")
        assert r.status_code == 200
        sid = r.json()["session_id"]
        assert sid

        r2 = client.get(f"/api/session/{sid}")
        assert r2.status_code == 200
        assert r2.json() == []

    def test_list_sessions(self, tmp_path):
        client, state = make_client(tmp_path)
        client.post("/api/session")
        client.post("/api/session")
        r = client.get("/api/sessions")
        assert r.status_code == 200
        assert len(r.json()) == 2

    def test_delete_session(self, tmp_path):
        client, state = make_client(tmp_path)
        sid = client.post("/api/session").json()["session_id"]
        r = client.delete(f"/api/session/{sid}")
        assert r.status_code == 200
        assert r.json()["deleted"] is True
        # Confirm gone
        r2 = client.delete(f"/api/session/{sid}")
        assert r2.json()["deleted"] is False


class TestMem:
    def test_mem_listing(self, tmp_path):
        client, _ = make_client(tmp_path)
        r = client.get("/api/mem")
        assert r.status_code == 200
        d = r.json()
        assert d["total"] == 2
        assert "entries" in d
        keys = {e["key"] for e in d["entries"]}
        assert "identity" in keys

    def test_mem_no_backend(self, tmp_path):
        from backend.app_fastapi import create_app, AppState
        state = AppState(mem=None, output_dir=tmp_path / "o")
        client = TestClient(create_app(state))
        d = client.get("/api/mem").json()
        assert d["entries"] == []


class TestFiles:
    def test_upload_and_list(self, tmp_path):
        client, state = make_client(tmp_path)
        files = {"file": ("hello.txt", b"hello world", "text/plain")}
        r = client.post("/api/upload", files=files)
        assert r.status_code == 200
        d = r.json()
        assert d["filename"] == "hello.txt"
        assert d["bytes"] == 11

        # File appears in /api/files
        listing = client.get("/api/files").json()
        assert any(f["name"] == "hello.txt" for f in listing["files"])

        # Download works
        r2 = client.get("/api/download/hello.txt")
        assert r2.status_code == 200
        assert r2.content == b"hello world"

    def test_upload_path_traversal_rejected(self, tmp_path):
        client, _ = make_client(tmp_path)
        # filename with traversal — Path.name strips it, so we get "etc"
        files = {"file": ("../../etc/passwd", b"x", "text/plain")}
        r = client.post("/api/upload", files=files)
        # Should succeed but with safe basename
        assert r.status_code == 200
        assert r.json()["filename"] == "passwd"  # basename only

    def test_download_missing_returns_404(self, tmp_path):
        client, _ = make_client(tmp_path)
        r = client.get("/api/download/nonexistent.txt")
        assert r.status_code == 404


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket
# ─────────────────────────────────────────────────────────────────────────────


class TestWebSocket:
    def test_handshake(self, tmp_path):
        client, state = make_client(tmp_path)
        sid = client.post("/api/session").json()["session_id"]
        with client.websocket_connect(f"/ws/chat/{sid}") as ws:
            ws.send_text(json.dumps({"text": "hola"}))
            # Primer mensaje: meta
            meta_raw = ws.receive_text()
            meta = json.loads(meta_raw)
            assert meta["type"] == "meta"
            assert "scores" in meta["payload"]
            assert "motor" in meta["payload"]
            assert meta["payload"]["user"] == "hola"

    def test_streaming_tokens(self, tmp_path):
        client, state = make_client(tmp_path)
        sid = client.post("/api/session").json()["session_id"]
        with client.websocket_connect(f"/ws/chat/{sid}") as ws:
            ws.send_text(json.dumps({"text": "hello world test"}))
            seen_types = set()
            full_response = ""
            for _ in range(50):
                msg = json.loads(ws.receive_text())
                seen_types.add(msg["type"])
                if msg["type"] == "token":
                    full_response += msg["payload"]["text"]
                if msg["type"] == "done":
                    break
            assert "meta" in seen_types
            assert "token" in seen_types
            assert "done" in seen_types
            # Mock echo: la respuesta completa contiene el input
            assert "hello" in full_response.lower()

    def test_session_persists_after_ws(self, tmp_path):
        client, state = make_client(tmp_path)
        sid = client.post("/api/session").json()["session_id"]
        with client.websocket_connect(f"/ws/chat/{sid}") as ws:
            ws.send_text(json.dumps({"text": "test message"}))
            for _ in range(50):
                msg = json.loads(ws.receive_text())
                if msg["type"] == "done":
                    break
        # La sesión ahora es una ConversationHistory
        history = state.sessions[sid]
        assert len(history) == 2  # user + assistant
        assert history.turns[0].role == "user"
        assert history.turns[0].content == "test message"
        assert history.turns[1].role == "assistant"

    def test_ws_creates_session_if_missing(self, tmp_path):
        client, state = make_client(tmp_path)
        # No POST /api/session — usamos un sid arbitrario
        sid = "newsessid"
        with client.websocket_connect(f"/ws/chat/{sid}") as ws:
            ws.send_text(json.dumps({"text": "x"}))
            for _ in range(20):
                msg = json.loads(ws.receive_text())
                if msg["type"] == "done":
                    break
        assert sid in state.sessions

    def test_ws_invalid_json(self, tmp_path):
        client, _ = make_client(tmp_path)
        sid = client.post("/api/session").json()["session_id"]
        with client.websocket_connect(f"/ws/chat/{sid}") as ws:
            ws.send_text("not json {")
            err = json.loads(ws.receive_text())
            assert err["type"] == "error"

    def test_ws_thinking_for_complex_query(self, tmp_path):
        client, _ = make_client(tmp_path)
        sid = client.post("/api/session").json()["session_id"]
        with client.websocket_connect(f"/ws/chat/{sid}") as ws:
            # Query con trigger DEEP → debe emitir thinking
            ws.send_text(json.dumps({"text": "demuestra el teorema de pitágoras paso a paso"}))
            seen_thinking = False
            for _ in range(50):
                msg = json.loads(ws.receive_text())
                if msg["type"] == "thinking":
                    seen_thinking = True
                if msg["type"] == "done":
                    break
            assert seen_thinking, "expected thinking message for DEEP-level query"

    def test_ws_no_thinking_for_simple_greeting(self, tmp_path):
        client, _ = make_client(tmp_path)
        sid = client.post("/api/session").json()["session_id"]
        with client.websocket_connect(f"/ws/chat/{sid}") as ws:
            ws.send_text(json.dumps({"text": "hola"}))
            seen_thinking = False
            for _ in range(50):
                msg = json.loads(ws.receive_text())
                if msg["type"] == "thinking":
                    seen_thinking = True
                if msg["type"] == "done":
                    break
            assert not seen_thinking, "INSTANT level should not show thinking indicator"


# ─────────────────────────────────────────────────────────────────────────────
# Routing heuristic (mock mode)
# ─────────────────────────────────────────────────────────────────────────────


class TestRoutingHeuristic:
    def test_code_query_routes_forge_c(self, tmp_path):
        from backend.app_fastapi import _routing_for, AppState
        s = AppState()
        scores = _routing_for(s, "write a python function")
        assert scores.forge_c > 0.5

    def test_math_query_routes_axiom(self, tmp_path):
        from backend.app_fastapi import _routing_for, AppState
        s = AppState()
        scores = _routing_for(s, "calcula 25% de 200")
        assert scores.axiom > 0.5

    def test_emotion_query_routes_empathy(self, tmp_path):
        from backend.app_fastapi import _routing_for, AppState
        s = AppState()
        scores = _routing_for(s, "estoy triste, me siento solo")
        assert scores.empathy > 0.5

    def test_creative_query_routes_muse(self, tmp_path):
        from backend.app_fastapi import _routing_for, AppState
        s = AppState()
        scores = _routing_for(s, "escribe un poema sobre la luna")
        assert scores.muse > 0.5

    def test_default_routes_cora(self, tmp_path):
        from backend.app_fastapi import _routing_for, AppState
        s = AppState()
        scores = _routing_for(s, "what is the meaning of life")
        # Default goes to cora
        assert scores.cora >= 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Level decision integration
# ─────────────────────────────────────────────────────────────────────────────


class TestLevelDecision:
    def test_decide_level_instant_for_greeting(self):
        from backend.app_fastapi import _decide_level
        d = _decide_level("hola")
        assert d["label"] == "instant"
        assert d["show_thinking"] is False

    def test_decide_level_deep_for_demuestra(self):
        from backend.app_fastapi import _decide_level
        d = _decide_level("demuestra el teorema de pitágoras")
        assert d["label"] == "deep"
        assert d["show_thinking"] is True

    def test_decide_level_normal_for_medium(self):
        from backend.app_fastapi import _decide_level
        d = _decide_level("escribe una función python que sume dos números")
        assert d["label"] == "normal"
        assert d["show_thinking"] is True


# ─────────────────────────────────────────────────────────────────────────────
# Parte 22: adapter endpoints
# ─────────────────────────────────────────────────────────────────────────────


class TestAdaptersEndpoints:
    def _client_with_registry(self, tmp_path):
        from backend.app_fastapi import AppState, create_app
        from growth import AdapterRegistry
        state = AppState(
            model=None, tokenizer=None, mem=FakeMem(),
            output_dir=tmp_path / "output", n_params=0,
            adapter_registry=AdapterRegistry(tmp_path / "brain"),
        )
        return TestClient(create_app(state)), state

    def test_list_empty(self, tmp_path):
        client, _ = self._client_with_registry(tmp_path)
        r = client.get("/api/adapters")
        assert r.status_code == 200
        data = r.json()
        assert data["enabled"] is True
        assert data["count"] == 0
        assert data["adapters"] == []

    def test_list_after_save(self, tmp_path):
        import torch.nn as nn
        from growth import LoRAConfig, build_adapter_pack

        client, state = self._client_with_registry(tmp_path)
        # Construir un motor minimo y guardar un adapter real
        motor = nn.Sequential()
        motor.add_module("proj", nn.Linear(4, 4))
        pack = build_adapter_pack(
            motor, ["proj"], LoRAConfig(rank=2, alpha=4), "sql", "forge_c"
        )
        state.adapter_registry.save(pack, tags=["database"])

        r = client.get("/api/adapters")
        data = r.json()
        assert data["count"] == 1
        a = data["adapters"][0]
        assert a["concept"] == "sql"
        assert a["motor"] == "forge_c"
        assert a["rank"] == 2
        assert a["tags"] == ["database"]
        assert data["total_bytes"] > 0

    def test_list_filter_by_motor(self, tmp_path):
        import torch.nn as nn
        from growth import LoRAConfig, build_adapter_pack
        client, state = self._client_with_registry(tmp_path)
        motor = nn.Sequential()
        motor.add_module("proj", nn.Linear(4, 4))
        reg = state.adapter_registry
        reg.save(build_adapter_pack(motor, ["proj"], LoRAConfig(), "a", "forge_c"))
        reg.save(build_adapter_pack(motor, ["proj"], LoRAConfig(), "b", "axiom"))
        r = client.get("/api/adapters?motor=axiom")
        assert r.json()["count"] == 1
        assert r.json()["adapters"][0]["motor"] == "axiom"

    def test_delete(self, tmp_path):
        import torch.nn as nn
        from growth import LoRAConfig, build_adapter_pack
        client, state = self._client_with_registry(tmp_path)
        motor = nn.Sequential()
        motor.add_module("proj", nn.Linear(4, 4))
        state.adapter_registry.save(
            build_adapter_pack(motor, ["proj"], LoRAConfig(), "gone", "forge_c")
        )
        r = client.delete("/api/adapters/forge_c/gone")
        assert r.status_code == 200
        assert r.json()["deleted"] is True
        # Segunda vez → no existe
        r2 = client.delete("/api/adapters/forge_c/gone")
        assert r2.json()["deleted"] is False

    def test_disabled_when_no_registry(self, tmp_path):
        from backend.app_fastapi import AppState, create_app
        state = AppState(model=None, tokenizer=None, output_dir=tmp_path / "o")
        client = TestClient(create_app(state))
        r = client.get("/api/adapters")
        assert r.status_code == 200
        assert r.json()["enabled"] is False


# ─────────────────────────────────────────────────────────────────────────────
# Parte 22.5: trajectory endpoints
# ─────────────────────────────────────────────────────────────────────────────


class TestTrajectoryEndpoints:
    def _client_with_planner(self, tmp_path):
        from backend.app_fastapi import AppState, create_app
        from composition import TrajectoryPlanner
        state = AppState(
            model=None, tokenizer=None, mem=FakeMem(),
            output_dir=tmp_path / "output",
            trajectory_planner=TrajectoryPlanner(),
        )
        return TestClient(create_app(state)), state

    def test_plan_single_motor(self, tmp_path):
        client, _ = self._client_with_planner(tmp_path)
        r = client.post("/api/trajectory/plan", json={"query": "me siento triste"})
        assert r.status_code == 200
        data = r.json()
        assert data["enabled"] is True
        assert data["motor_sequence"] == ["empathy"]

    def test_plan_code_as_story(self, tmp_path):
        client, _ = self._client_with_planner(tmp_path)
        r = client.post(
            "/api/trajectory/plan",
            json={"query": "explica este código como cuento"},
        )
        data = r.json()
        assert data["motor_sequence"] == ["forge_c", "muse"]
        assert "transform-as" in data["rationale"]

    def test_plan_empty_query_400(self, tmp_path):
        client, _ = self._client_with_planner(tmp_path)
        r = client.post("/api/trajectory/plan", json={"query": "   "})
        assert r.status_code == 400

    def test_execute_uses_echo_without_model(self, tmp_path):
        client, _ = self._client_with_planner(tmp_path)
        r = client.post(
            "/api/trajectory/execute",
            json={"query": "explica este código como cuento"},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["enabled"] is True
        assert len(data["steps"]) == 2
        assert data["steps"][0]["motor_name"] == "forge_c"
        assert data["steps"][1]["motor_name"] == "muse"
        assert "[muse]" in data["fused_output"]

    def test_disabled_when_no_planner(self, tmp_path):
        from backend.app_fastapi import AppState, create_app
        state = AppState(model=None, tokenizer=None, output_dir=tmp_path / "o")
        client = TestClient(create_app(state))
        r = client.post("/api/trajectory/plan", json={"query": "test"})
        assert r.status_code == 200
        assert r.json()["enabled"] is False


# ─────────────────────────────────────────────────────────────────────────────
# Parte 23: sleep cycle endpoints
# ─────────────────────────────────────────────────────────────────────────────


class TestSleepEndpoints:
    def _client_with_sleep(self, tmp_path):
        from backend.app_fastapi import AppState, create_app
        from sleep import EpisodicBuffer, SleepCycle, SleepDaemon
        buf = EpisodicBuffer()
        daemon = SleepDaemon(
            SleepCycle(buf), inactivity_seconds=3600, overflow_threshold=500
        )
        state = AppState(
            model=None, tokenizer=None, output_dir=tmp_path / "o",
            episodic_buffer=buf, sleep_daemon=daemon,
        )
        return TestClient(create_app(state)), state

    def test_add_episode_increments_buffer(self, tmp_path):
        client, state = self._client_with_sleep(tmp_path)
        r = client.post("/api/sleep/episode", json={
            "user_text": "hola", "aion_response": "hi",
            "motor_sequence": ["cora"], "user_feedback": "up",
        })
        assert r.status_code == 200
        assert r.json()["buffer_size"] == 1
        assert len(state.episodic_buffer) == 1

    def test_last_no_log_initially(self, tmp_path):
        client, _ = self._client_with_sleep(tmp_path)
        r = client.get("/api/sleep/last")
        data = r.json()
        assert data["enabled"] is True
        assert data["last_log"] is None
        assert data["buffer_size"] == 0

    def test_force_run_cycle(self, tmp_path):
        client, state = self._client_with_sleep(tmp_path)
        # Cargar algunos episodios
        for _ in range(3):
            client.post("/api/sleep/episode", json={
                "user_text": "python", "aion_response": "ok",
                "user_feedback": "up",
            })
        r = client.post("/api/sleep")
        assert r.status_code == 200
        log = r.json()
        assert log["enabled"] is True
        assert log["episodes_processed"] == 3
        assert len(log["phases"]) == 6
        assert [p["name"] for p in log["phases"]] == [
            "recollect", "score", "prune", "compress", "consolidate", "followups"
        ]
        # Buffer vacío tras el ciclo
        assert len(state.episodic_buffer) == 0

    def test_last_log_after_run(self, tmp_path):
        client, _ = self._client_with_sleep(tmp_path)
        client.post("/api/sleep/episode", json={"user_text": "q", "aion_response": "a"})
        client.post("/api/sleep")
        r = client.get("/api/sleep/last")
        data = r.json()
        assert data["last_log"] is not None
        assert data["last_log"]["episodes_processed"] == 1

    def test_disabled_when_no_daemon(self, tmp_path):
        from backend.app_fastapi import AppState, create_app
        state = AppState(model=None, tokenizer=None, output_dir=tmp_path / "o")
        client = TestClient(create_app(state))
        assert client.post("/api/sleep").json()["enabled"] is False
        assert client.get("/api/sleep/last").json()["enabled"] is False

    def test_background_loop_auto_triggers_on_inactivity(self, tmp_path):
        """El asyncio background loop debe disparar maybe_run cuando pasa el tiempo.

        Usamos intervalo y inactivity muy cortos para no esperar segundos reales.
        """
        import asyncio as _asyncio
        from sleep import EpisodicBuffer, Episode, SleepCycle, SleepDaemon
        from backend.app_fastapi import _sleep_daemon_loop, AppState

        buf = EpisodicBuffer()
        buf.add(Episode("test query", "test response"))
        daemon = SleepDaemon(SleepCycle(buf), inactivity_seconds=0.01, overflow_threshold=999)
        daemon.notify_activity(ts=0.0)  # pretendemos que ya pasó mucho tiempo
        state = AppState(
            model=None, tokenizer=None, output_dir=tmp_path / "o",
            episodic_buffer=buf, sleep_daemon=daemon,
            sleep_poll_interval=0.01,
        )

        async def _drive():
            task = _asyncio.create_task(_sleep_daemon_loop(state))
            # Dar tiempo al loop para correr al menos 1 ciclo
            await _asyncio.sleep(0.10)
            task.cancel()
            try:
                await task
            except _asyncio.CancelledError:
                pass

        _asyncio.get_event_loop().run_until_complete(_drive())
        # El daemon debió correr al menos una vez
        assert daemon.last_log is not None
        assert daemon.last_log.episodes_processed == 1

    def test_background_loop_noop_when_no_daemon(self, tmp_path):
        import asyncio as _asyncio
        from backend.app_fastapi import _sleep_daemon_loop, AppState
        state = AppState(
            model=None, tokenizer=None, output_dir=tmp_path / "o",
            sleep_poll_interval=0.01,
        )
        # Sin daemon, el loop retorna inmediatamente
        _asyncio.get_event_loop().run_until_complete(_sleep_daemon_loop(state))


# ─────────────────────────────────────────────────────────────────────────────
# Parte 25: feedback endpoints
# ─────────────────────────────────────────────────────────────────────────────


class TestFeedbackEndpoints:
    def _client_with_reward(self, tmp_path):
        from backend.app_fastapi import AppState, create_app
        from reward import RewardEstimator, RewardLedger
        from sleep import EpisodicBuffer
        from growth import AdapterRegistry
        state = AppState(
            model=None, tokenizer=None, output_dir=tmp_path / "o",
            reward_estimator=RewardEstimator(),
            reward_ledger=RewardLedger(),
            episodic_buffer=EpisodicBuffer(),
            adapter_registry=AdapterRegistry(tmp_path / "brain"),
        )
        return TestClient(create_app(state)), state

    def test_feedback_up_updates_ledger(self, tmp_path):
        client, state = self._client_with_reward(tmp_path)
        r = client.post("/api/feedback", json={"vote": "up", "motor": "forge_c"})
        assert r.status_code == 200
        data = r.json()
        assert data["enabled"] is True
        assert data["key"] == "forge_c"
        assert data["estimate"]["mean"] > 0.6
        # ledger record
        snap = state.reward_ledger.snapshot()
        assert "forge_c" in snap
        assert snap["forge_c"]["n"] == 1

    def test_feedback_invalid_vote(self, tmp_path):
        client, _ = self._client_with_reward(tmp_path)
        r = client.post("/api/feedback", json={"vote": "meh"})
        assert r.status_code == 400

    def test_feedback_updates_episode_buffer(self, tmp_path):
        from sleep import Episode
        client, state = self._client_with_reward(tmp_path)
        state.episodic_buffer.add(Episode("q", "a"))
        client.post("/api/feedback", json={"vote": "up", "motor": "cora"})
        assert state.episodic_buffer.snapshot()[-1].user_feedback == "up"

    def test_feedback_updates_adapter_meta(self, tmp_path):
        import torch.nn as nn
        from growth import LoRAConfig, build_adapter_pack
        client, state = self._client_with_reward(tmp_path)
        motor = nn.Sequential()
        motor.add_module("proj", nn.Linear(4, 4))
        state.adapter_registry.save(
            build_adapter_pack(motor, ["proj"], LoRAConfig(), "python", "forge_c")
        )
        client.post("/api/feedback", json={
            "vote": "up", "motor": "forge_c", "adapter": "python"
        })
        meta = state.adapter_registry.get_meta("forge_c", "python")
        assert meta.usage_count == 1
        assert meta.reward_score > 0.0

    def test_ledger_endpoint(self, tmp_path):
        client, _ = self._client_with_reward(tmp_path)
        client.post("/api/feedback", json={"vote": "up", "motor": "cora"})
        client.post("/api/feedback", json={"vote": "down", "motor": "muse"})
        r = client.get("/api/feedback/ledger")
        snap = r.json()["ledger"]
        assert "cora" in snap
        assert "muse" in snap
        assert snap["cora"]["mean"] > snap["muse"]["mean"]

    def test_disabled_when_no_estimator(self, tmp_path):
        from backend.app_fastapi import AppState, create_app
        state = AppState(model=None, tokenizer=None, output_dir=tmp_path / "o")
        client = TestClient(create_app(state))
        assert client.post("/api/feedback", json={"vote": "up"}).json()["enabled"] is False
        assert client.get("/api/feedback/ledger").json()["enabled"] is False

    def test_feedback_persists_to_disk(self, tmp_path):
        """Feedback debe escribirse a reward_ledger.jsonl para sobrevivir restart."""
        from backend.app_fastapi import AppState, create_app
        from reward import RewardEstimator, RewardLedger
        ledger_path = tmp_path / "brain" / "v1" / "reward_ledger.jsonl"
        state = AppState(
            model=None, tokenizer=None, output_dir=tmp_path / "o",
            reward_estimator=RewardEstimator(),
            reward_ledger=RewardLedger(),
            reward_ledger_path=ledger_path,
        )
        client = TestClient(create_app(state))
        client.post("/api/feedback", json={"vote": "up", "motor": "forge_c"})
        assert ledger_path.exists()
        # Nuevo state carga del mismo path
        state2 = AppState(
            model=None, tokenizer=None, output_dir=tmp_path / "o2",
            reward_estimator=RewardEstimator(),
            reward_ledger=RewardLedger(),
            reward_ledger_path=ledger_path,
        )
        state2.reward_ledger.load_jsonl(ledger_path)
        assert state2.reward_ledger.count_for("forge_c") == 1


# ─────────────────────────────────────────────────────────────────────────────
# Parte 26: hierarchical memory endpoint
# ─────────────────────────────────────────────────────────────────────────────


class TestHierarchyEndpoint:
    def test_empty_store(self, tmp_path):
        from backend.app_fastapi import AppState, create_app
        from compression import HierarchicalStore
        state = AppState(
            model=None, tokenizer=None, output_dir=tmp_path / "o",
            hierarchical_store=HierarchicalStore(),
        )
        client = TestClient(create_app(state))
        r = client.get("/api/memory/hierarchy")
        data = r.json()
        assert data["enabled"] is True
        assert data["total"] == 0
        assert data["levels"]["episodic"]["count"] == 0

    def test_store_with_items(self, tmp_path):
        from backend.app_fastapi import AppState, create_app
        from compression import HierarchicalStore, StoredItem, MemoryLevel
        store = HierarchicalStore()
        store.add(StoredItem("1", "a", MemoryLevel.EPISODIC))
        store.add(StoredItem("2", "b", MemoryLevel.STABLE))
        store.add(StoredItem("3", "c", MemoryLevel.NUCLEAR))
        state = AppState(
            model=None, tokenizer=None, output_dir=tmp_path / "o",
            hierarchical_store=store,
        )
        client = TestClient(create_app(state))
        data = client.get("/api/memory/hierarchy").json()
        assert data["total"] == 3
        assert data["levels"]["episodic"]["count"] == 1
        assert data["levels"]["stable"]["count"] == 1
        assert data["levels"]["nuclear"]["count"] == 1

    def test_disabled_when_no_store(self, tmp_path):
        from backend.app_fastapi import AppState, create_app
        state = AppState(model=None, tokenizer=None, output_dir=tmp_path / "o")
        client = TestClient(create_app(state))
        assert client.get("/api/memory/hierarchy").json()["enabled"] is False


# ─────────────────────────────────────────────────────────────────────────────
# Parte 27: sparse report endpoint
# ─────────────────────────────────────────────────────────────────────────────


class TestSparseEndpoint:
    def test_report_with_tracker(self, tmp_path):
        import torch
        import torch.nn as nn
        from backend.app_fastapi import AppState, create_app
        from sparse import attach_sparse_gates, SparseConfig, SparsityTracker

        motor = nn.Sequential()
        motor.add_module("proj", nn.Linear(8, 32))
        attach_sparse_gates(motor, ["proj"], SparseConfig(gate_hidden=4))
        motor(torch.randn(2, 8))
        tracker = SparsityTracker(motor)
        state = AppState(
            model=None, tokenizer=None, output_dir=tmp_path / "o",
            sparsity_tracker=tracker,
        )
        client = TestClient(create_app(state))
        r = client.get("/api/sparse/report")
        data = r.json()
        assert data["enabled"] is True
        assert "avg_density" in data
        assert 0 <= data["active_percent"] <= 100

    def test_disabled_when_no_tracker(self, tmp_path):
        from backend.app_fastapi import AppState, create_app
        state = AppState(model=None, tokenizer=None, output_dir=tmp_path / "o")
        client = TestClient(create_app(state))
        assert client.get("/api/sparse/report").json()["enabled"] is False
