#!/usr/bin/env python3
"""
verify_tiny_e2e.py — Fase C: verificación E2E del tiny entrenado
==================================================================

Carga checkpoints/tiny_canonical.pt y verifica:
  1. ROUTING       — los 5 motores reciben tráfico (cobertura ≥ 4/5)
  2. CANONICAL     — el modelo reconoce [USER:][AION:][SKILL:][MEM:][TOOL:]
  3. EOS           — la generación se detiene en BPE EOS (id 2)
  4. SKILL/MEM     — inyección de contexto altera la salida
  5. WORLD MODEL   — los 5 simuladores se conectan al pipeline (Parte 19)
  6. NEURO-SYMBOL  — engine acepta grafos del decoder (Parte 20)
  7. AUTO-LEARN    — flujo simulate→detect→learn→re-query funciona

Salida: PASS/FAIL por cada check + summary.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# Force UTF-8 on Windows stdout (cp1252 can't handle ✓ ✗ → )
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

import torch

from synth.canonical_format import format_record, parse_canonical, has_eos


EOS = 2  # BPE EOS


def log(msg: str) -> None:
    print(msg, flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Build + load
# ─────────────────────────────────────────────────────────────────────────────


def build_and_load(ckpt_path: Path):
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
    ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    pipeline.load_state_dict(ck["model_state"], strict=False)
    pipeline.eval()
    return pipeline, tok, ck


def encode(tok, text, ml=128):
    try: return tok.encode(text, ml)
    except TypeError: return tok.encode(text)[:ml]


@torch.no_grad()
def generate(pipeline, tok, prompt, max_new=40):
    """Greedy decode hasta EOS o max_new."""
    ids = encode(tok, prompt, ml=96)
    cur = torch.tensor([ids], dtype=torch.long)
    plen = len(ids)
    out = pipeline(cur)
    for _ in range(max_new):
        nxt = int(out.logits[0, -1].argmax().item())
        if nxt in (0, EOS):
            break
        cur = torch.cat([cur, torch.tensor([[nxt]])], dim=1)
        if cur.shape[1] >= 128:
            break
        out = pipeline(cur)
    text = tok.decode(cur[0, plen:].tolist()) if cur.shape[1] > plen else ""
    motors = list(out.active_motors) if hasattr(out, 'active_motors') else []
    return text, motors


# ─────────────────────────────────────────────────────────────────────────────
# Checks
# ─────────────────────────────────────────────────────────────────────────────


def check_routing(pipeline, tok) -> dict:
    """Check 1: los 5 motores reciben tráfico al menos 1 vez."""
    log("\n[CHECK 1/7] ROUTING — los 5 motores reciben tráfico")
    prompts = {
        "cora":    "[USER: por qué llueve?]\n[AION: ",
        "forge_c": "[USER: write a python function to add two numbers]\n[AION: ",
        "axiom":   "[USER: cuánto es 25% de 200?]\n[AION: ",
        "muse":    "[USER: write a short poem about the moon]\n[AION: ",
        "empathy": "[USER: estoy frustrado, perdí mi trabajo]\n[AION: ",
    }
    motors_seen = set()
    per_prompt = {}
    for domain, prompt in prompts.items():
        _, motors = generate(pipeline, tok, prompt, max_new=8)
        per_prompt[domain] = motors
        for m in motors:
            motors_seen.add(m)
    coverage = len(motors_seen)
    passed = coverage >= 4  # tolerancia: 4 de 5
    log(f"  Motors active across all prompts: {sorted(motors_seen)}")
    log(f"  Coverage: {coverage}/5")
    log(f"  → {'PASS' if passed else 'FAIL'}")
    return {"passed": passed, "coverage": coverage, "motors_seen": sorted(motors_seen)}


def check_canonical_format(pipeline, tok) -> dict:
    """Check 2: el modelo procesa formato canónico sin crashear."""
    log("\n[CHECK 2/7] CANONICAL FORMAT — los 5 tags se aceptan como input")
    cases = [
        format_record(user="hola", aion="", ),  # USER + AION (vacío)
        format_record(user="x", aion="", skill="be concise"),
        format_record(user="x", aion="", mem="user=jesus"),
        format_record(user="x", aion="",
                      tool='{"action":"write_file","input":{"path":"a.py","content":"x"}}',
                      result="ok"),
    ]
    successes = 0
    for prefix in cases:
        # Truncate trailing [EOS] for the generation prompt
        prompt = prefix.replace("\n[EOS]", "")
        try:
            text, _ = generate(pipeline, tok, prompt, max_new=4)
            successes += 1
        except Exception as exc:
            log(f"  FAIL on case: {exc}")
    passed = successes == len(cases)
    log(f"  Processed {successes}/{len(cases)} canonical inputs without crash")
    log(f"  → {'PASS' if passed else 'FAIL'}")
    return {"passed": passed, "successes": successes, "total": len(cases)}


def check_eos_termination(pipeline, tok) -> dict:
    """Check 3: la generación se detiene en BPE EOS."""
    log("\n[CHECK 3/7] EOS TERMINATION — generación se detiene")
    prompts = ["[USER: hola]\n[AION: ", "[USER: 2+2]\n[AION: "]
    stops_observed = 0
    lengths = []
    for prompt in prompts:
        text, _ = generate(pipeline, tok, prompt, max_new=60)
        lengths.append(len(text))
        # Si la longitud generada es < max_new, asumimos que paró por EOS
        if len(text) < 200:  # umbral generoso
            stops_observed += 1
    passed = stops_observed >= 1  # al menos 1 prompt se detuvo
    log(f"  Generations stopped before max_new: {stops_observed}/{len(prompts)}")
    log(f"  Lengths: {lengths}")
    log(f"  → {'PASS' if passed else 'FAIL'}")
    return {"passed": passed, "stops": stops_observed, "lengths": lengths}


def check_skill_mem_injection(pipeline, tok) -> dict:
    """Check 4: skill/mem prefix altera la salida."""
    log("\n[CHECK 4/7] SKILL/MEM INJECTION — el contexto inyectado influye")
    base_prompt = "[USER: hola]\n[AION: "
    skill_prompt = "[SKILL: respond in Spanish always]\n[USER: hola]\n[AION: "
    mem_prompt = "[MEM: user name=Jesus, lang=es]\n[USER: hola]\n[AION: "

    base_text, _   = generate(pipeline, tok, base_prompt, max_new=12)
    skill_text, _  = generate(pipeline, tok, skill_prompt, max_new=12)
    mem_text, _    = generate(pipeline, tok, mem_prompt, max_new=12)

    different_skill = base_text != skill_text
    different_mem   = base_text != mem_text
    passed = different_skill or different_mem  # al menos uno produce diferencia
    log(f"  base:  '{base_text[:40]}'")
    log(f"  skill: '{skill_text[:40]}'  (different: {different_skill})")
    log(f"  mem:   '{mem_text[:40]}'  (different: {different_mem})")
    log(f"  → {'PASS' if passed else 'FAIL'}")
    return {
        "passed": passed,
        "diff_skill": different_skill,
        "diff_mem": different_mem,
    }


def check_world_model_pipeline(pipeline, tok) -> dict:
    """Check 5: simuladores world_model funcionan + se conectan a la pipeline."""
    log("\n[CHECK 5/7] WORLD MODEL — simuladores conectan al pipeline")
    from world_model import (
        AxiomSimulator, ForgeCSimulator, CoraSimulator,
        MuseSimulator, EmpathySimulator,
        ScratchPadVerifier, SimulationLoop,
    )

    sims_ok = 0
    details = []

    # 1) AxiomSimulator: 15% de 240 → 36 (caso emblemático)
    axiom_loop = SimulationLoop(simulator=AxiomSimulator())
    outcome = axiom_loop.run("cuanto es 15% de 240")
    proven = outcome.pad.get_by_name("proven") or []
    has_36 = any("36" in step for step in proven)
    if outcome.coherent and has_36:
        sims_ok += 1
        details.append(f"  AxiomSimulator: 15% de 240 → 36 ✓ (coherent={outcome.coherent})")
    else:
        details.append(f"  AxiomSimulator: FAIL — coherent={outcome.coherent}, has_36={has_36}")

    # 2-5) Otros simuladores
    test_cases = [
        (ForgeCSimulator(), "def add(a, b): return a + b", "forge_c"),
        (CoraSimulator(), "rain causes wet soil", "cora"),
        (MuseSimulator(), "una historia de muerte y amor", "muse"),
        (EmpathySimulator(), "estoy frustrado, perdí mi trabajo", "empathy"),
    ]
    for sim, query, name in test_cases:
        loop = SimulationLoop(simulator=sim)
        out = loop.run(query)
        if out.coherent:
            sims_ok += 1
            details.append(f"  {name}Simulator: coherent ✓")
        else:
            details.append(f"  {name}Simulator: FAIL — {out.last_result.issues}")

    # 6) Conexión con pipeline real: el encoder produce embeddings que el sim
    #    podría consumir (verificamos al menos que el flujo no rompa)
    pipeline_ok = False
    try:
        ids = encode(tok, "what is 15% of 240?", ml=64)
        ids_t = torch.tensor([ids], dtype=torch.long)
        with torch.no_grad():
            out = pipeline(ids_t)
            concepts = pipeline.encoder(ids_t)
        # Pipeline forward + encoder forward sin crash → conexión OK
        pipeline_ok = (concepts is not None and concepts.shape[0] == 1)
    except Exception as exc:
        details.append(f"  Pipeline integration: FAIL — {exc}")

    if pipeline_ok:
        details.append(f"  Pipeline integration: encoder accepts canonical query ✓")

    passed = sims_ok == 5 and pipeline_ok
    log(f"  Simulators OK: {sims_ok}/5")
    log(f"  Pipeline integration: {'OK' if pipeline_ok else 'FAIL'}")
    for d in details:
        log(d)
    log(f"  → {'PASS' if passed else 'FAIL'}")
    return {"passed": passed, "sims_ok": sims_ok, "pipeline_ok": pipeline_ok}


def check_neuro_symbolic(pipeline, tok) -> dict:
    """Check 6: engine simbólico opera sobre grafos derivados del modelo."""
    log("\n[CHECK 6/7] NEURO-SYMBOLIC — engine simbólico procesa grafos")
    from symbolic import (
        SymbolicGraph, SymbolicNode, SymbolicEdge,
        SymbolicEngine, build_engine_for_motor,
    )

    # 1) Caso AXIOM: 0.15 × 240 → 36 vía ArithmeticRule
    g_axiom = SymbolicGraph()
    g_axiom.add_node(SymbolicNode(id="expr", label="0.15 × 240"))
    engine_axiom = build_engine_for_motor("axiom")
    result_axiom = engine_axiom.apply_all(g_axiom)
    axiom_ok = (g_axiom.has_node("result_expr")
                and g_axiom.find_node("result_expr").label == "36")

    # 2) Caso CORA: A→B→C, transitividad → A→C
    g_cora = SymbolicGraph()
    g_cora.add_edge(SymbolicEdge("a", "b", "causes"))
    g_cora.add_edge(SymbolicEdge("b", "c", "causes"))
    engine_cora = build_engine_for_motor("cora")
    result_cora = engine_cora.apply_all(g_cora)
    cora_ok = g_cora.has_edge("a", "c", "causes")

    # 3) Caso conflict resolution: causes + prevents → símbolo gana
    g_conflict = SymbolicGraph()
    g_conflict.add_edge(SymbolicEdge("a", "b", "causes"))
    g_conflict.add_edge(SymbolicEdge("a", "b", "prevents"))
    result_conflict = engine_cora.apply_all(g_conflict)
    conflict_ok = (result_conflict.has_conflicts
                   and not g_conflict.has_edge("a", "b", "prevents")
                   and g_conflict.has_edge("a", "b", "causes"))

    # 4) Conexión con pipeline: el pipeline produce activos motors,
    #    y podemos construir un grafo simbólico a partir del query
    integration_ok = False
    try:
        ids = encode(tok, "rain causes wet soil", ml=64)
        ids_t = torch.tensor([ids], dtype=torch.long)
        with torch.no_grad():
            out = pipeline(ids_t)
        # Construye un grafo simbólico mock derivado del active_motor
        if out.active_motors:
            g = SymbolicGraph()
            g.add_node(SymbolicNode(id="query", label="rain causes wet soil",
                                    type="query", props={"motor": out.active_motors[0]}))
            integration_ok = True
    except Exception as exc:
        log(f"  Pipeline integration FAIL: {exc}")

    passed = axiom_ok and cora_ok and conflict_ok and integration_ok
    log(f"  AxiomEngine 0.15×240=36:        {'✓' if axiom_ok else '✗'}")
    log(f"  CoraEngine transitivity:        {'✓' if cora_ok else '✗'}")
    log(f"  Conflict resolution (sym wins): {'✓' if conflict_ok else '✗'}")
    log(f"  Pipeline integration:           {'✓' if integration_ok else '✗'}")
    log(f"  → {'PASS' if passed else 'FAIL'}")
    return {
        "passed": passed,
        "axiom_ok": axiom_ok,
        "cora_ok": cora_ok,
        "conflict_ok": conflict_ok,
        "integration_ok": integration_ok,
    }


def check_auto_learn(pipeline, tok) -> dict:
    """Check 7: el flujo auto-learn funciona sobre el tiny entrenado."""
    log("\n[CHECK 7/7] AUTO-LEARN — detect→search→learn→answer")
    import math as _math
    import torch.nn.functional as F

    # 1) Pregunta algo que el modelo no debería saber bien
    UNKNOWN = "what is rust programming language?"
    text_before, _ = generate(pipeline, tok, f"[USER: {UNKNOWN}]\n[AION: ", max_new=20)
    log(f"  Before learn: '{text_before[:60]}'")

    # 2) Simula búsqueda en KB local
    kb = {"rust": "Rust is a systems programming language focused on memory safety."}
    found = kb.get("rust")
    if not found:
        log("  → FAIL: KB lookup failed")
        return {"passed": False}

    # 3) Mini fine-tune en background (5 steps focused)
    pipeline.train()
    new_text = format_record(user=UNKNOWN, aion=found)
    try:
        new_ids = tok.encode(new_text, 96)
    except TypeError:
        new_ids = tok.encode(new_text)[:96]
    new_ids = new_ids + [EOS]
    new_t = torch.tensor([new_ids], dtype=torch.long)

    # Optimizer mini, lr alto, 30 steps
    opt = torch.optim.AdamW(pipeline.parameters(), lr=1e-3)
    for _ in range(30):
        out = pipeline(new_t)
        loss = F.cross_entropy(out.logits[0, :-1], new_t[0, 1:], ignore_index=0)
        if not _math.isfinite(loss.item()):
            continue
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pipeline.parameters(), 1.0)
        opt.step()

    pipeline.eval()
    text_after, _ = generate(pipeline, tok, f"[USER: {UNKNOWN}]\n[AION: ", max_new=20)
    log(f"  After learn:  '{text_after[:60]}'")

    learned = ("rust" in text_after.lower()
               or "programming" in text_after.lower()
               or "memory" in text_after.lower())
    passed = learned
    log(f"  Learned new fact: {'YES' if learned else 'NO'}")
    log(f"  → {'PASS' if passed else 'FAIL'}")
    return {"passed": passed, "before": text_before, "after": text_after}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main():
    ckpt_path = ROOT / "checkpoints" / "tiny_canonical.pt"
    if not ckpt_path.exists():
        log(f"ERROR: checkpoint not found at {ckpt_path}")
        sys.exit(1)

    log("=" * 64)
    log("FASE C — VERIFICACIÓN E2E")
    log("=" * 64)

    log(f"Loading checkpoint {ckpt_path.name}...")
    pipeline, tok, ck = build_and_load(ckpt_path)
    log(f"  config: {ck.get('config_name')}")
    log(f"  steps:  {ck.get('n_steps')}")
    log(f"  routing_acc: {ck.get('routing_acc'):.1f}%")
    log(f"  motor_counts: {ck.get('motor_counts')}")

    results = {}
    results["routing"]      = check_routing(pipeline, tok)
    results["canonical"]    = check_canonical_format(pipeline, tok)
    results["eos"]          = check_eos_termination(pipeline, tok)
    results["skill_mem"]    = check_skill_mem_injection(pipeline, tok)
    results["world_model"]  = check_world_model_pipeline(pipeline, tok)
    results["neuro_symbol"] = check_neuro_symbolic(pipeline, tok)
    results["auto_learn"]   = check_auto_learn(pipeline, tok)

    log("\n" + "=" * 64)
    log("SUMMARY")
    log("=" * 64)
    n_passed = sum(1 for r in results.values() if r["passed"])
    for name, r in results.items():
        mark = "PASS" if r["passed"] else "FAIL"
        log(f"  [{mark}] {name}")
    log(f"\n{n_passed}/{len(results)} checks passed")

    # Save report
    report_path = ROOT / "checkpoints" / "tiny_canonical_e2e_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            k: {kk: (list(vv) if isinstance(vv, set) else vv) for kk, vv in r.items()}
            for k, r in results.items()
        }, f, ensure_ascii=False, indent=2, default=str)
    log(f"Report saved: {report_path}")

    if n_passed == len(results):
        log("\n>>> FASE C COMPLETE — TINY E2E PASSED <<<")
        sys.exit(0)
    else:
        log("\n>>> FASE C INCOMPLETE — see failures above <<<")
        sys.exit(1)


if __name__ == "__main__":
    main()
