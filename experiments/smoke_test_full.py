"""
experiments/smoke_test_full.py — Smoke Test Completo Pre-H200
=============================================================

Verifica que TODAS las piezas del sistema AION-C funcionan end-to-end
antes de gastar en GPU cloud.

Ejecuta con --config tiny en CPU. Si CUALQUIER pieza falla, reporta exactamente cuál.

Piezas verificadas:
  1. Phase 0-4 completas con 50 steps cada una
  2. Interactive eval con 3 preguntas automáticas
  3. MEM store y search
  4. Agent Loop con 3 turns mockeados
  5. Visualización de grafo en ASCII
  6. Cuantización INT4 del modelo tiny
  7. Inferencia del modelo cuantizado
  8. Conversation history de 3 turnos
  9. System prompt

Uso:
    cd AION-C
    python -m experiments.smoke_test_full
    # Exit code 0 = todo OK, 1 = algo falló
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path
from typing import List, Tuple

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch


# ----------------------��-------------------------��----------------------------
# TEST RUNNER
# -----------------------------------------------------------------------------

class SmokeTestRunner:
    """Ejecuta todos los smoke tests y reporta resultados."""

    def __init__(self):
        self.results: List[Tuple[str, bool, str]] = []
        self.pipeline = None
        self.tok = None
        self.mose_cfg = None
        self.device = torch.device("cpu")
        self.run_dir = _ROOT / "runs" / "smoke_test"

    def test(self, name: str):
        """Decorator para registrar un test."""
        def decorator(fn):
            def wrapper():
                print(f"\n{'-' * 50}", flush=True)
                print(f"  TEST: {name}", flush=True)
                print(f"{'-' * 50}", flush=True)
                t0 = time.perf_counter()
                try:
                    fn()
                    elapsed = time.perf_counter() - t0
                    print(f"  PASS ({elapsed:.1f}s)", flush=True)
                    self.results.append((name, True, f"{elapsed:.1f}s"))
                except Exception as e:
                    elapsed = time.perf_counter() - t0
                    tb = traceback.format_exc()
                    print(f"  FAIL ({elapsed:.1f}s): {e}", flush=True)
                    print(tb, flush=True)
                    self.results.append((name, False, str(e)))
            wrapper._test_name = name
            return wrapper
        return decorator

    def summary(self) -> bool:
        """Imprime resumen y retorna True si todo pasó."""
        print(f"\n{'=' * 60}", flush=True)
        print("  SMOKE TEST SUMMARY", flush=True)
        print(f"{'=' * 60}", flush=True)

        passed = sum(1 for _, ok, _ in self.results if ok)
        total  = len(self.results)

        for name, ok, detail in self.results:
            status = "PASS" if ok else "FAIL"
            print(f"  [{status}] {name:45s} {detail}", flush=True)

        print(f"\n  {passed}/{total} tests passed", flush=True)
        all_ok = passed == total
        if all_ok:
            print("  ALL CLEAR — Ready for H200!", flush=True)
        else:
            failed = [name for name, ok, _ in self.results if not ok]
            print(f"  FAILED: {', '.join(failed)}", flush=True)
            print("  DO NOT PROCEED with H200 until all tests pass.", flush=True)
        print(f"{'=' * 60}", flush=True)
        return all_ok


runner = SmokeTestRunner()


# -----------------------------------------------------------------------------
# TEST 1: BUILD PIPELINE
# --------------------------------------��--------------------------------------

@runner.test("1. Build MoSE Pipeline (tiny)")
def test_build_pipeline():
    from experiments.train_production import build_pipeline_and_tok
    runner.run_dir.mkdir(parents=True, exist_ok=True)
    runner.pipeline, runner.tok, runner.mose_cfg = build_pipeline_and_tok("tiny", runner.device)
    params = sum(p.numel() for p in runner.pipeline.parameters())
    print(f"    Params: {params:,}", flush=True)
    assert params > 0, "Pipeline has 0 parameters"

    # Forward pass smoke test
    ids = torch.randint(1, runner.mose_cfg.vocab_size, (1, 16))
    out = runner.pipeline(ids)
    assert out.logits.shape[0] == 1, f"Bad logits shape: {out.logits.shape}"
    print(f"    Forward pass OK. Logits: {out.logits.shape}", flush=True)


# -----------------------------------------------------------------------------
# TEST 2: PHASE 0-3 (50 steps each)
# -----------------------------------------------------------------------------

@runner.test("2. Phase 0-3 Training (50 steps each)")
def test_phase0_3():
    import copy
    from experiments.train_production import (
        _TrainHparams, run_phase0, run_phase1, run_phase2, run_phase3,
        load_all_datasets,
    )

    hparams = _TrainHparams()
    max_steps = 50

    datasets = load_all_datasets(max_examples=100, eval_size=10)

    # Phase 0
    r0 = run_phase0(
        runner.pipeline, runner.mose_cfg, runner.tok, datasets, hparams,
        runner.device, runner.run_dir, max_steps_override=max_steps,
    )
    assert r0.steps_run > 0, f"Phase 0: 0 steps run"
    print(f"    Phase 0: {r0.steps_run} steps, loss={r0.final_loss:.4f}", flush=True)

    # Phase 1
    r1 = run_phase1(
        runner.pipeline, runner.mose_cfg, runner.tok, datasets, hparams,
        runner.device, runner.run_dir, max_steps_override=max_steps,
    )
    assert r1.steps_run > 0
    print(f"    Phase 1: {r1.steps_run} steps, loss={r1.final_loss:.4f}", flush=True)

    # Phase 2
    r2_list = run_phase2(
        runner.pipeline, runner.mose_cfg, runner.tok, datasets, hparams,
        runner.device, runner.run_dir, max_steps_override=max_steps,
    )
    print(f"    Phase 2: {len(r2_list)} motors trained", flush=True)

    # Phase 3
    r3 = run_phase3(
        runner.pipeline, runner.mose_cfg, runner.tok, datasets, hparams,
        runner.device, runner.run_dir, max_steps_override=max_steps,
    )
    assert r3.steps_run > 0
    print(f"    Phase 3: {r3.steps_run} steps, loss={r3.final_loss:.4f}", flush=True)


# -----------------------------------------------------------------------------
# TEST 3: PHASE 4 INSTRUCTION TUNING
# -----------------------------------------------------------------------------

@runner.test("3. Phase 4 Instruction Tuning (50 steps)")
def test_phase4():
    from experiments.train_production import run_phase4, _TrainHparams

    hparams = _TrainHparams()
    r4 = run_phase4(
        runner.pipeline, runner.mose_cfg, runner.tok, hparams,
        runner.device, runner.run_dir, max_steps_override=50,
        interactive=False,
    )
    assert r4.steps_run > 0, f"Phase 4: 0 steps"
    print(f"    Phase 4: {r4.steps_run} steps, loss={r4.final_loss:.4f}", flush=True)


# -----------------------------------------------------------------------------
# TEST 4: INTERACTIVE EVAL (automated)
# -----------------------------------------------------------------------------

@runner.test("4. Interactive Eval (3 automated questions)")
def test_interactive_eval():
    from experiments.train_production import interactive_eval

    questions = [
        "Quién eres?",
        "Resuelve 2+2",
        "Escribe hello world en Python",
    ]
    result = interactive_eval(runner.pipeline, runner.tok, runner.device, questions=questions)
    assert result is True, "Interactive eval returned False"
    print("    3 questions answered successfully", flush=True)


# -----------------------------------------------------------------------------
# TEST 5: MEM STORE AND SEARCH
# ----------------------------------------���------------------------------------

@runner.test("5. MEM Store, Search, and Learn")
def test_mem():
    from agent.memory_bridge import MemoryBridge

    mem = MemoryBridge()

    # Basic store/load
    mem.store("test_key", "test_value")
    assert mem.load("test_key") == "test_value"

    # Search
    results = mem.search("test")
    assert len(results) > 0, "Search returned 0 results"

    # Learn
    mem.learn("pytorch_version", "2.5", source="web_search")
    learned = mem.get_learned("pytorch_version")
    assert learned == "2.5", f"Learned value wrong: {learned}"

    # Search learned
    hits = mem.search_learned("pytorch")
    assert len(hits) > 0, "search_learned returned 0 results"
    assert hits[0][2] == "web_search", f"Source wrong: {hits[0][2]}"

    # Context
    ctx = mem.as_context()
    assert "test_key" in ctx
    print(f"    MEM has {len(mem)} entries. Context: {len(ctx)} chars", flush=True)


# -----------------------------------------------------------------------------
# TEST 6: AGENT LOOP (3 mocked turns)
# -----------------------------------------------------------------------------

@runner.test("6. Agent Loop (3 mocked turns)")
def test_agent_loop():
    from agent.loop import AgentLoop, MockMotor, MotorAction
    from agent.tools import build_tool_registry

    def mock_runner(cmd, **kwargs):
        from types import SimpleNamespace
        return SimpleNamespace(stdout="mocked output", stderr="", returncode=0)

    motor = MockMotor([
        MotorAction("bash", {"command": "echo hello"}, "Testing bash"),
        MotorAction("file_read", {"path": __file__}, "Reading self"),
        MotorAction("DONE", {}, "All done"),
    ])

    tools = build_tool_registry(
        runner=mock_runner,
        read_fn=lambda path: "file content mock",
    )

    loop = AgentLoop(motor=motor, tools=tools, max_turns=10)
    result = loop.run(task="Smoke test task")

    assert result.succeeded, f"Agent loop failed: {result.status}"
    assert result.turns_used == 3, f"Expected 3 turns, got {result.turns_used}"
    print(f"    Agent loop: {result.turns_used} turns, status={result.status}", flush=True)


# -----------------------------------------------------------------------------
# TEST 7: GRAPH VISUALIZATION (ASCII)
# -----------------------------------------------------------------------------

@runner.test("7. Graph Visualization (ASCII)")
def test_graph_viz():
    from core.graph import CausalGraph, CausalNode, CausalEdge, NodeType, CausalRelation
    from visualization.graph_viewer import GraphViewer

    g = CausalGraph()
    g.add_node(CausalNode(node_id="n1", label="Rain", node_type=NodeType.ENTITY, confidence=0.9))
    g.add_node(CausalNode(node_id="n2", label="Flood", node_type=NodeType.EVENT, confidence=0.7))
    g.add_node(CausalNode(node_id="n3", label="Damage", node_type=NodeType.STATE, confidence=0.6))
    g.add_edge(CausalEdge(source_id="n1", target_id="n2", relation=CausalRelation.CAUSES, strength=0.8))
    g.add_edge(CausalEdge(source_id="n2", target_id="n3", relation=CausalRelation.LEADS_TO, strength=0.7))

    viewer = GraphViewer(g, title="Smoke Test Graph")
    ascii_text = viewer.to_ascii()
    assert "Rain" in ascii_text
    assert "CAUSES" in ascii_text or "causes" in ascii_text
    assert "Flood" in ascii_text
    print(f"    ASCII graph rendered ({len(ascii_text)} chars)", flush=True)

    # HTML
    html = viewer.to_html()
    assert "<html>" in html.lower() or "<!doctype" in html.lower()
    print(f"    HTML graph rendered ({len(html)} chars)", flush=True)


# -----------------------------------------------------------------------------
# TEST 8: QUANTIZATION INT4
# -------------------------------------------���---------------------------------

@runner.test("8. Quantization INT4")
def test_quantize():
    from inference.quantize import quantize_state_dict, dequantize_state_dict

    state_dict = runner.pipeline.state_dict()
    quantized, stats = quantize_state_dict(state_dict, group_size=32)

    print(f"    Quantized: {stats['quantized']:,} params", flush=True)
    print(f"    Kept float: {stats['kept_float']:,} params", flush=True)

    # Dequantize and verify shapes match
    restored = dequantize_state_dict(quantized)
    for key in state_dict:
        assert key in restored, f"Missing key after dequantize: {key}"
        orig_shape = list(state_dict[key].shape)
        rest_shape = list(restored[key].shape)
        assert orig_shape == rest_shape, f"Shape mismatch for {key}: {orig_shape} vs {rest_shape}"

    print(f"    All {len(state_dict)} tensors round-tripped successfully", flush=True)

    # Save quantized checkpoint
    out_path = runner.run_dir / "tiny_int4.pt"
    torch.save({"quantized_state": quantized}, out_path)
    print(f"    Saved: {out_path}", flush=True)


# -----------------------------------------------------------------------------
# TEST 9: INFERENCE FROM QUANTIZED MODEL
# -----------------------------------------------------------------------------

@runner.test("9. Inference from Quantized Model")
def test_quantized_inference():
    from inference.quantize import quantize_state_dict, dequantize_state_dict
    from router.pipeline import MoSEPipeline, MoSEConfig

    # Quantize → dequantize → load into fresh model
    state_dict = runner.pipeline.state_dict()
    quantized, _ = quantize_state_dict(state_dict, group_size=32)
    restored = dequantize_state_dict(quantized)

    model2 = MoSEPipeline(runner.mose_cfg)
    model2.load_state_dict(restored, strict=False)
    model2.eval()

    ids = torch.randint(1, runner.mose_cfg.vocab_size, (1, 16))
    with torch.no_grad():
        out = model2(ids)

    assert out.logits.shape[0] == 1
    pred = out.logits[0].argmax(dim=-1)
    print(f"    Quantized model inference OK. Output shape: {out.logits.shape}", flush=True)


# -----------------------------------------------------------------------------
# TEST 10: CONVERSATION HISTORY (3 turns)
# -----------------------------------------------------------------------------

@runner.test("10. Conversation History (3 turns)")
def test_conversation_history():
    from agent.loop import AgentLoop, MockMotor, MotorAction

    def mock_runner(cmd, **kwargs):
        from types import SimpleNamespace
        return SimpleNamespace(stdout="ok", stderr="", returncode=0)

    from agent.tools import build_tool_registry

    motor = MockMotor([MotorAction("DONE", {}, "Done")])

    loop = AgentLoop(
        motor=motor,
        tools=build_tool_registry(runner=mock_runner),
        max_turns=5,
        max_history_tokens=4096,
        system_prompt="Eres AION-C, un asistente de IA.",
    )

    # Simulate 3 turns of conversation
    loop.add_user_message("Hola, quién eres?")
    loop.add_assistant_message("Soy AION-C.")
    loop.add_user_message("Qué puedes hacer?")
    loop.add_assistant_message("Puedo razonar, escribir código, y más.")
    loop.add_user_message("Genial, gracias.")
    loop.add_assistant_message("De nada!")

    history = loop.get_conversation_history()
    assert len(history) == 6, f"Expected 6 messages, got {len(history)}"

    # Verify system prompt in built history
    from agent.session import AgentSession
    session = AgentSession(task="test")
    built = loop._build_history(session)
    assert "AION-C" in built, "System prompt not in history"
    assert "Hola" in built, "User message not in history"
    print(f"    History: {len(history)} messages, built={len(built)} chars", flush=True)

    # Test truncation
    loop2 = AgentLoop(
        motor=motor,
        tools=build_tool_registry(runner=mock_runner),
        max_history_tokens=10,  # very small
    )
    for i in range(100):
        loop2.add_user_message(f"Message {i} with some content to fill tokens")
    h = loop2.get_conversation_history()
    total_tokens = sum(len(m["content"]) // 4 for m in h)
    assert total_tokens <= 20, f"History not truncated: {total_tokens} tokens"
    print(f"    Truncation OK: {len(h)} messages after 100 inputs", flush=True)


# -----------------------------------------------------------------------------
# TEST 11: SYSTEM PROMPT
# ------------------------------------------------------��----------------------

@runner.test("11. System Prompt")
def test_system_prompt():
    from agent.loop import AgentLoop, MockMotor, MotorAction
    from agent.tools import build_tool_registry

    def mock_runner(cmd, **kwargs):
        from types import SimpleNamespace
        return SimpleNamespace(stdout="ok", stderr="", returncode=0)

    system = "Eres AION-C en modo tutor de matemáticas."
    motor = MockMotor([MotorAction("DONE", {}, "Done")])
    loop = AgentLoop(
        motor=motor,
        tools=build_tool_registry(runner=mock_runner),
        system_prompt=system,
    )
    assert loop.system_prompt == system

    from agent.session import AgentSession
    session = AgentSession(task="math question")
    history = loop._build_history(session)
    assert "tutor de matemáticas" in history, f"System prompt not found in history: {history[:200]}"
    print(f"    System prompt correctly injected into history", flush=True)


# -----------------------------------------------------------------------------
# TEST 12: INSTRUCTION TUNING GENERATOR
# -----------------------------------------------------------------------------

@runner.test("12. Instruction Tuning Generator")
def test_instruction_gen():
    from synth.instruction_gen import InstructionGenerator

    gen = InstructionGenerator(seed=42)

    # Test each category generates without error
    categories = [
        ("identity", gen.gen_identity, 10),
        ("casual", gen.gen_casual, 10),
        ("reasoning", gen.gen_reasoning, 10),
        ("code", gen.gen_code, 10),
        ("math", gen.gen_math, 10),
        ("creativity", gen.gen_creativity, 10),
        ("social", gen.gen_social, 10),
        ("autonomy", gen.gen_autonomy, 10),
        ("self_verify", gen.gen_self_verify, 10),
        ("thinking_aloud", gen.gen_thinking_aloud, 10),
        ("proactive", gen.gen_proactive, 10),
        ("format", gen.gen_format, 10),
        ("safety", gen.gen_safety, 10),
        ("metacognition", gen.gen_metacognition, 10),
        ("mem_usage", gen.gen_mem_usage, 10),
        ("multi_turn", gen.gen_multi_turn, 10),
        ("system_prompt", gen.gen_system_prompt, 10),
    ]

    total = 0
    for name, fn, n in categories:
        examples = fn(n)
        assert len(examples) == n, f"{name}: expected {n}, got {len(examples)}"
        for ex in examples:
            assert "instruction" in ex, f"{name}: missing 'instruction'"
            assert "response" in ex, f"{name}: missing 'response'"
        total += len(examples)

    print(f"    {len(categories)} categories, {total} examples generated OK", flush=True)


# ------------------------------------------------------------���----------------
# TEST 13: NEW TOOLS (WebSearch, WebFetch, FileRead)
# -----------------------------------------------------------------------------

@runner.test("13. New Tools (WebSearch, WebFetch, FileRead)")
def test_new_tools():
    from agent.tools import WebSearchTool, WebFetchTool, FileReadTool

    # WebSearch with mock
    ws = WebSearchTool(search_fn=lambda q, n: f"Results for '{q}': item1, item2")
    r = ws.run({"query": "Python tutorial"})
    assert r.ok, f"WebSearch failed: {r.stderr}"
    assert "Python" in r.stdout

    # WebFetch with mock
    wf = WebFetchTool(fetch_fn=lambda url: f"<html>Content of {url}</html>")
    r = wf.run({"url": "https://example.com"})
    assert r.ok, f"WebFetch failed: {r.stderr}"
    assert "example.com" in r.stdout

    # FileRead (reads this file)
    fr = FileReadTool()
    r = fr.run({"path": __file__})
    assert r.ok, f"FileRead failed: {r.stderr}"
    assert "smoke_test_full" in r.stdout

    # FileRead with mock
    fr2 = FileReadTool(read_fn=lambda p: "mocked content")
    r = fr2.run({"path": "anything.txt"})
    assert r.ok
    assert r.stdout == "mocked content"

    print("    All 3 new tools work with mocks and real execution", flush=True)


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def main():
    print("=" * 60, flush=True)
    print("  AION-C FULL SMOKE TEST — Pre-H200 Verification", flush=True)
    print("=" * 60, flush=True)

    t_global = time.perf_counter()

    # Run all tests in order
    test_build_pipeline()
    test_instruction_gen()
    test_new_tools()
    test_mem()
    test_agent_loop()
    test_conversation_history()
    test_system_prompt()
    test_graph_viz()
    test_phase0_3()
    test_phase4()
    test_interactive_eval()
    test_quantize()
    test_quantized_inference()

    elapsed = time.perf_counter() - t_global
    print(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)", flush=True)

    all_ok = runner.summary()
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
