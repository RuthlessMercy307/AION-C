#!/usr/bin/env python3
"""
app.py — AION-C Web UI
=======================

Single-file web app: Flask backend + embedded HTML/CSS/JS frontend.
Dark theme with purple accents. Chat, routing, MEM, graph visualization.

Usage:
    cd AION-C
    python app.py
    # Open http://localhost:3000
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from flask import Flask, request, jsonify, Response

import torch

# ─── Model Loading ───────────────────────────────────────────────────────────

print("Loading AION-C...", flush=True)

from router.pipeline import MoSEPipeline, MoSEConfig
from experiments.train_production import build_tokenizer
from memory.semantic_store import SemanticStore

DEVICE = torch.device("cpu")
TOK = build_tokenizer(32_000)

CFG = MoSEConfig(
    hidden_dim=1024, vocab_size=TOK.vocab_size,
    enc_n_layers=12, enc_state_dim=16, enc_expand=2, enc_d_conv=4, enc_ffn_mult=4,
    orch_mlp_hidden=512, orch_max_motors=3, orch_min_confidence=0.3,
    motor_max_nodes=8, motor_n_heads=8, motor_threshold=0.01, unif_n_heads=8,
    dec_n_layers=16, dec_n_heads=8, dec_max_seq_len=512,
    dec_state_dim=16, dec_expand=2, dec_d_conv=4, dec_ffn_mult=4,
)
PIPELINE = MoSEPipeline(CFG).to(DEVICE)
PARAMS = sum(p.numel() for p in PIPELINE.parameters())

# Load fine-tuned checkpoint
_CKPT_PATH = _ROOT / "checkpoints" / "aion_1b_direct.pt"
if _CKPT_PATH.exists():
    print(f"  Loading checkpoint {_CKPT_PATH.name}...", flush=True)
    _ckpt = torch.load(str(_CKPT_PATH), map_location="cpu", weights_only=False)
    PIPELINE.load_state_dict(_ckpt["model_state"], strict=False)
    print(f"  Checkpoint loaded: step={_ckpt.get('step','?')} val_loss={_ckpt.get('val_loss','?')}", flush=True)
else:
    print(f"  WARNING: {_CKPT_PATH} not found - using random weights", flush=True)

MEM = SemanticStore(encoder=PIPELINE.encoder, tokenizer=TOK, similarity_threshold=0.0)
MEM._device = DEVICE
MEM.store("identity", "I am AION-C, an AI created by Jesus with MoSE architecture.", "general")
MEM.store("architecture", "MoSE uses 5 specialized motors: CORA, FORGE-C, AXIOM, MUSE, EMPATHY.", "general")
MEM.store("limitations", "I cannot process images, audio, or video. Text only.", "general")

MOTOR_NAMES = ["cora", "forge_c", "muse", "axiom", "empathy"]
MOTOR_COLORS = {"cora": "#a855f7", "forge_c": "#3b82f6", "axiom": "#f59e0b",
                "muse": "#ec4899", "empathy": "#10b981", "general": "#6b7280"}

# Chat sessions
SESSIONS = {}  # session_id -> [{"role","content","motor","scores","graph","mem_hits"}]

print(f"  Model: {PARAMS:,} params | MEM: {len(MEM)} entries | Ready!", flush=True)


# ─── Inference ───────────────────────────────────────────────────────────────

def generate(text: str, session_id: str):
    """Generate response — feeds the raw user text as the model was fine-tuned.

    NOTE: aion_1b_direct.pt was fine-tuned on raw `input + " " + output + EOS`
    pairs with no MEM tags or Q/A prefixes. Prepending context here destroys
    output quality. MEM search is still done for the side panel display only.
    """
    # MEM search — kept for side panel display, NOT injected into model input
    mem_results = MEM.search(text, top_k=3)

    # Feed only the raw user text — match training format exactly
    try:
        ids = TOK.encode(text, 96)
    except TypeError:
        ids = TOK.encode(text)[:96]

    plen = len(ids)
    cur = torch.tensor([ids], dtype=torch.long, device=DEVICE)

    PIPELINE.eval()
    with torch.no_grad():
        out = PIPELINE(cur)
        motors = list(out.active_motors)
        scores = out.orchestrator.scores.tolist() if hasattr(out.orchestrator, 'scores') else [0.2]*5

        for _ in range(48):
            nxt = int(out.logits[0, -1].argmax().item())
            if nxt in (0, 2):
                break
            cur = torch.cat([cur, torch.tensor([[nxt]], device=DEVICE)], dim=1)
            if cur.shape[1] - plen >= 3:
                try:
                    ts = TOK.decode([nxt])
                    if ts.rstrip().endswith(('.', '?', '!')):
                        break
                except Exception:
                    pass
            if cur.shape[1] < 128:
                out = PIPELINE(cur)

    pred_ids = cur[0, plen:].tolist()
    try:
        response = TOK.decode(pred_ids)
    except Exception:
        response = "(no response)"

    # Extract graph from crystallizer output with real token labels
    graph_data = {"nodes": [], "edges": []}
    try:
        if motors:
            motor_obj = PIPELINE.motors[motors[0]]
            concepts = PIPELINE.encoder(torch.tensor([ids], dtype=torch.long, device=DEVICE))
            cryst = motor_obj.build_graph(concepts)
            if cryst.graphs:
                g = cryst.graphs[0]
                # Decode each token individually for label lookup
                tok_strs = []
                for tid in ids:
                    try:
                        s = TOK.decode([tid]).strip()
                    except Exception:
                        s = ""
                    tok_strs.append(s if s else f"·{tid}")

                for n in g.nodes:
                    seq_pos = n.metadata.get("seq_pos") if hasattr(n, "metadata") and n.metadata else None
                    if seq_pos is not None and 0 <= seq_pos < len(tok_strs):
                        # Try to build a phrase from a small window around the token
                        win = tok_strs[max(0, seq_pos-1):seq_pos+2]
                        label = " ".join(w for w in win if w).strip() or tok_strs[seq_pos]
                        if len(label) > 28:
                            label = label[:25] + "…"
                    else:
                        label = n.label
                    graph_data["nodes"].append({
                        "id": n.node_id,
                        "label": label,
                        "type": n.node_type.value if hasattr(n.node_type, 'value') else str(n.node_type),
                        "confidence": round(getattr(n, "confidence", 1.0), 2),
                    })
                for e in g.edges:
                    graph_data["edges"].append({
                        "source": e.source_id,
                        "target": e.target_id,
                        "relation": e.relation.value if hasattr(e.relation, 'value') else str(e.relation),
                        "strength": round(getattr(e, "strength", 1.0), 2),
                    })
    except Exception as exc:
        graph_data["error"] = str(exc)[:100]

    # Auto-learn disabled: model is not trained to read MEM context,
    # and storing every QA pollutes the side-panel without helping generation.

    return {
        "response": response,
        "motors": motors,
        "scores": {MOTOR_NAMES[i]: round(s, 3) for i, s in enumerate(scores[:5])},
        "graph": graph_data,
        "mem_hits": [{"key": k, "value": v[:60], "sim": round(s, 2)} for k, v, s in mem_results],
    }


# ─── Flask App ───────────────────────────────────────────────────────────────

app = Flask(__name__)


@app.route("/")
def index():
    return Response(HTML_PAGE, mimetype="text/html")


@app.route("/api/info", methods=["GET"])
def api_info():
    ckpt_name = _CKPT_PATH.name if _CKPT_PATH.exists() else "(no checkpoint)"
    return jsonify({
        "params": PARAMS,
        "params_human": f"{PARAMS/1e9:.2f}B" if PARAMS >= 1e9 else f"{PARAMS/1e6:.1f}M",
        "checkpoint": ckpt_name,
        "device": str(DEVICE),
        "vocab_size": TOK.vocab_size,
    })


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json
    text = data.get("message", "")
    session_id = data.get("session_id", "default")

    if session_id not in SESSIONS:
        SESSIONS[session_id] = []

    SESSIONS[session_id].append({"role": "user", "content": text})

    result = generate(text, session_id)

    SESSIONS[session_id].append({
        "role": "assistant", "content": result["response"],
        "motors": result["motors"], "scores": result["scores"],
        "graph": result["graph"],
    })

    return jsonify(result)


@app.route("/api/sessions", methods=["GET"])
def list_sessions():
    summaries = []
    for sid, msgs in SESSIONS.items():
        first_msg = msgs[0]["content"][:40] if msgs else "New chat"
        summaries.append({"id": sid, "title": first_msg, "messages": len(msgs)})
    return jsonify(summaries)


@app.route("/api/session/<session_id>", methods=["GET"])
def get_session(session_id):
    return jsonify(SESSIONS.get(session_id, []))


@app.route("/api/session", methods=["POST"])
def new_session():
    sid = str(uuid.uuid4())[:8]
    SESSIONS[sid] = []
    return jsonify({"session_id": sid})


@app.route("/api/mem", methods=["GET"])
def get_mem():
    return jsonify({"entries": MEM.list_entries(), "stats": MEM.stats()})


@app.route("/api/mem", methods=["POST"])
def add_mem():
    data = request.json
    MEM.store(data["key"], data["value"], data.get("domain", "general"))
    return jsonify({"ok": True, "total": len(MEM)})


# ─── HTML/CSS/JS ─────────────────────────────────────────────────────────────

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AION-C</title>
<script src="https://unpkg.com/vis-network@9.1.9/standalone/umd/vis-network.min.js"></script>
<style>
* { margin:0; padding:0; box-sizing:border-box; }
:root {
  --bg: #0f0f1a; --surface: #1a1a2e; --surface2: #16213e;
  --text: #e2e8f0; --text2: #94a3b8; --purple: #a855f7;
  --purple2: #7c3aed; --border: #2d2d44; --input-bg: #1e1e32;
}
body { font-family: -apple-system, 'Segoe UI', sans-serif; background: var(--bg); color: var(--text); height: 100vh; display: flex; overflow: hidden; }

/* Sidebar */
.sidebar { width: 260px; background: var(--surface); border-right: 1px solid var(--border); display: flex; flex-direction: column; }
.sidebar-header { padding: 16px; border-bottom: 1px solid var(--border); }
.sidebar-header h1 { font-size: 20px; color: var(--purple); }
.sidebar-header p { font-size: 11px; color: var(--text2); margin-top: 4px; }
.new-chat-btn { width: 100%; padding: 10px; margin: 12px 0; background: var(--purple2); color: white; border: none; border-radius: 8px; cursor: pointer; font-size: 14px; }
.new-chat-btn:hover { background: var(--purple); }
.chat-list { flex: 1; overflow-y: auto; padding: 8px; }
.chat-item { padding: 10px 12px; border-radius: 8px; cursor: pointer; margin-bottom: 4px; font-size: 13px; color: var(--text2); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.chat-item:hover, .chat-item.active { background: var(--surface2); color: var(--text); }

/* Main chat */
.main { flex: 1; display: flex; flex-direction: column; }
.messages { flex: 1; overflow-y: auto; padding: 24px; }
.msg { margin-bottom: 20px; max-width: 85%; }
.msg.user { margin-left: auto; }
.msg.assistant { margin-right: auto; }
.msg-bubble { padding: 12px 16px; border-radius: 16px; font-size: 14px; line-height: 1.6; word-wrap: break-word; }
.msg.user .msg-bubble { background: var(--purple2); color: white; border-bottom-right-radius: 4px; }
.msg.assistant .msg-bubble { background: var(--surface2); color: var(--text); border-bottom-left-radius: 4px; }
.motor-badge { display: inline-block; margin-top: 6px; padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: 600; color: white; }
.mem-tag { font-size: 11px; color: var(--text2); margin-top: 4px; font-style: italic; }

/* Input */
.input-area { padding: 16px 24px; border-top: 1px solid var(--border); background: var(--surface); }
.input-wrap { display: flex; gap: 8px; }
.input-wrap input { flex: 1; padding: 12px 16px; background: var(--input-bg); border: 1px solid var(--border); border-radius: 12px; color: var(--text); font-size: 14px; outline: none; }
.input-wrap input:focus { border-color: var(--purple); }
.input-wrap button { padding: 12px 20px; background: var(--purple2); color: white; border: none; border-radius: 12px; cursor: pointer; font-size: 14px; }
.input-wrap button:hover { background: var(--purple); }
.input-wrap button:disabled { opacity: 0.5; cursor: not-allowed; }

/* Right panel */
.panel { width: 300px; background: var(--surface); border-left: 1px solid var(--border); overflow-y: auto; display: flex; flex-direction: column; }
.panel-section { padding: 16px; border-bottom: 1px solid var(--border); }
.panel-section h3 { font-size: 13px; color: var(--purple); margin-bottom: 10px; text-transform: uppercase; letter-spacing: 1px; }
.score-bar { display: flex; align-items: center; margin-bottom: 6px; font-size: 12px; }
.score-bar .name { width: 70px; color: var(--text2); }
.score-bar .bar-bg { flex: 1; height: 8px; background: var(--bg); border-radius: 4px; overflow: hidden; }
.score-bar .bar-fill { height: 100%; border-radius: 4px; transition: width 0.3s; }
.score-bar .val { width: 36px; text-align: right; color: var(--text2); font-size: 11px; }
.mem-entry { font-size: 12px; color: var(--text2); padding: 6px 0; border-bottom: 1px solid var(--border); }
.mem-entry .key { color: var(--purple); font-weight: 600; }
.graph-viz { min-height: 120px; position: relative; }
.graph-node { position: absolute; background: var(--purple2); color: white; padding: 4px 8px; border-radius: 6px; font-size: 10px; white-space: nowrap; }
.graph-edge { position: absolute; font-size: 9px; color: var(--text2); }
.typing { color: var(--text2); font-size: 13px; padding: 12px 16px; }

/* Per-message graph card */
.graph-card { margin-top: 8px; border: 1px solid var(--border); border-radius: 10px; background: var(--surface); overflow: hidden; }
.graph-card-header { padding: 8px 12px; cursor: pointer; display: flex; justify-content: space-between; align-items: center; font-size: 12px; color: var(--text2); user-select: none; }
.graph-card-header:hover { background: var(--surface2); color: var(--text); }
.graph-card-header .gc-title { display: flex; align-items: center; gap: 8px; }
.graph-card-header .gc-arrow { transition: transform 0.2s; }
.graph-card.open .gc-arrow { transform: rotate(90deg); }
.graph-card-body { display: none; height: 360px; background: #0a0a14; }
.graph-card.open .graph-card-body { display: block; }
.graph-card-empty { padding: 16px; font-size: 12px; color: var(--text2); text-align: center; }
.gc-meta { font-size: 10px; color: var(--text2); padding: 6px 12px; border-top: 1px solid var(--border); display: flex; gap: 12px; flex-wrap: wrap; }
.gc-meta span b { color: var(--purple); }

@media (max-width: 900px) { .sidebar, .panel { display: none; } }
</style>
</head>
<body>

<div class="sidebar">
  <div class="sidebar-header">
    <h1>AION-C</h1>
    <p id="modelInfo">Loading model info...</p>
  </div>
  <div style="padding: 0 12px;">
    <button class="new-chat-btn" onclick="newChat()">+ New Chat</button>
  </div>
  <div class="chat-list" id="chatList"></div>
</div>

<div class="main">
  <div class="messages" id="messages"></div>
  <div class="input-area">
    <div class="input-wrap">
      <input id="input" placeholder="Type a message... (EN or ES)" onkeydown="if(event.key==='Enter')send()" autofocus>
      <button id="sendBtn" onclick="send()">Send</button>
    </div>
  </div>
</div>

<div class="panel">
  <div class="panel-section">
    <h3>Routing Scores</h3>
    <div id="routingScores">
      <p style="font-size:12px;color:var(--text2)">Send a message to see routing</p>
    </div>
  </div>
  <div class="panel-section">
    <h3>Memory (MEM)</h3>
    <div id="memPanel">
      <p style="font-size:12px;color:var(--text2)">Loading...</p>
    </div>
  </div>
</div>

<script>
const COLORS = {cora:'#a855f7', forge_c:'#3b82f6', axiom:'#f59e0b', muse:'#ec4899', empathy:'#10b981'};
const RELATION_COLORS = {
  causes:       '#ef4444',  // red
  prevents:     '#f97316',  // orange
  enables:      '#22c55e',  // green
  leads_to:     '#dc2626',  // dark red
  implies:      '#3b82f6',  // blue
  follows_from: '#60a5fa',  // light blue
  contradicts:  '#a855f7',  // purple
  equivalent:   '#9ca3af',  // gray
  supports:     '#84cc16',  // lime
  weakens:      '#92400e',  // brown
  requires:     '#06b6d4',  // cyan
  precedes:     '#ec4899',  // pink
  part_of:      '#14b8a6',  // teal
  instance_of:  '#a16207',  // olive
  correlates:   '#eab308',  // yellow
  analogous_to: '#8b5cf6',  // violet
};
const NODE_TYPE_SHAPES = {
  entity: 'box', event: 'diamond', state: 'ellipse', action: 'triangle',
  hypothesis: 'dot', fact: 'star', question: 'square'
};
let sessionId = null;
let sending = false;
let graphCounter = 0;
const graphInstances = {};

async function loadModelInfo() {
  try {
    const r = await fetch('/api/info');
    const d = await r.json();
    document.getElementById('modelInfo').textContent =
      `MoSE • 5 Motors • ${d.params_human} (${d.params.toLocaleString()} params)`;
  } catch(e) {
    document.getElementById('modelInfo').textContent = 'Model info unavailable';
  }
}

async function newChat() {
  const r = await fetch('/api/session', {method:'POST'});
  const d = await r.json();
  sessionId = d.session_id;
  document.getElementById('messages').innerHTML = '';
  loadSessions();
}

async function loadSessions() {
  const r = await fetch('/api/sessions');
  const sessions = await r.json();
  const el = document.getElementById('chatList');
  el.innerHTML = sessions.map(s =>
    `<div class="chat-item ${s.id===sessionId?'active':''}" onclick="loadSession('${s.id}')">${s.title} (${s.messages})</div>`
  ).join('');
}

async function loadSession(sid) {
  sessionId = sid;
  const r = await fetch(`/api/session/${sid}`);
  const msgs = await r.json();
  const el = document.getElementById('messages');
  el.innerHTML = '';
  msgs.forEach(m => addMessage(m.role, m.content, m.motors, m.scores, m.graph));
  loadSessions();
}

function addMessage(role, content, motors, scores, graph) {
  const el = document.getElementById('messages');
  const div = document.createElement('div');
  div.className = `msg ${role}`;
  let badge = '';
  if (role === 'assistant' && motors && motors.length) {
    const m = motors[0];
    const c = COLORS[m] || '#6b7280';
    const s = scores && scores[m] ? scores[m] : '';
    badge = `<div><span class="motor-badge" style="background:${c}">${m.toUpperCase()} ${s}</span></div>`;
  }
  let graphCard = '';
  if (role === 'assistant') {
    const gid = 'g' + (++graphCounter);
    const nNodes = graph && graph.nodes ? graph.nodes.length : 0;
    const nEdges = graph && graph.edges ? graph.edges.length : 0;
    graphCard = `
      <div class="graph-card" id="card-${gid}">
        <div class="graph-card-header" onclick="toggleGraph('${gid}')">
          <span class="gc-title"><span class="gc-arrow">▶</span> Causal Graph</span>
          <span>${nNodes} nodes · ${nEdges} edges</span>
        </div>
        <div class="graph-card-body" id="body-${gid}"></div>
        <div class="gc-meta" id="meta-${gid}" style="display:none"></div>
      </div>`;
    if (graph) graphInstances[gid] = {data: graph, rendered: false};
  }
  div.innerHTML = `<div class="msg-bubble">${escHtml(content)}</div>${badge}${graphCard}`;
  el.appendChild(div);
  el.scrollTop = el.scrollHeight;
}

function toggleGraph(gid) {
  const card = document.getElementById('card-' + gid);
  const meta = document.getElementById('meta-' + gid);
  const wasOpen = card.classList.contains('open');
  card.classList.toggle('open');
  if (!wasOpen) {
    meta.style.display = 'flex';
    renderGraph(gid);
  } else {
    meta.style.display = 'none';
  }
}

function renderGraph(gid) {
  const inst = graphInstances[gid];
  const body = document.getElementById('body-' + gid);
  const meta = document.getElementById('meta-' + gid);
  if (!inst || !inst.data) {
    body.innerHTML = '<div class="graph-card-empty">No graph data</div>';
    return;
  }
  if (inst.rendered) return;
  const g = inst.data;
  if (!g.nodes || !g.nodes.length) {
    body.innerHTML = '<div class="graph-card-empty">Empty graph (no nodes detected for this prompt)</div>';
    inst.rendered = true;
    return;
  }
  const nodes = new vis.DataSet(g.nodes.map(n => ({
    id: n.id,
    label: n.label || n.id,
    shape: NODE_TYPE_SHAPES[n.type] || 'box',
    color: { background: '#1e1e32', border: '#a855f7', highlight: { background: '#2d2d44', border: '#c084fc' } },
    font: { color: '#e2e8f0', size: 14, face: 'Segoe UI' },
    title: `${n.type || 'node'} · conf ${n.confidence ?? '?'}`,
  })));
  const edges = new vis.DataSet((g.edges || []).map((e, i) => ({
    id: 'e' + i,
    from: e.source,
    to: e.target,
    label: e.relation,
    color: { color: RELATION_COLORS[e.relation] || '#64748b', highlight: '#ffffff' },
    font: { color: RELATION_COLORS[e.relation] || '#64748b', size: 11, strokeWidth: 0, align: 'middle' },
    arrows: 'to',
    width: 1 + (e.strength || 0.5) * 2,
    smooth: { type: 'curvedCW', roundness: 0.15 },
  })));
  const network = new vis.Network(body, { nodes, edges }, {
    physics: { enabled: true, solver: 'forceAtlas2Based',
      forceAtlas2Based: { gravitationalConstant: -50, springLength: 100, springConstant: 0.08 },
      stabilization: { iterations: 150 } },
    interaction: { dragNodes: true, dragView: true, zoomView: true, hover: true },
    nodes: { borderWidth: 2 },
  });
  inst.rendered = true;
  // Build legend in meta
  const relSet = new Set((g.edges || []).map(e => e.relation));
  const legend = Array.from(relSet).map(r =>
    `<span><b style="color:${RELATION_COLORS[r] || '#64748b'}">■</b> ${r}</span>`
  ).join('');
  meta.innerHTML = legend || '<span>no relations</span>';
}

function escHtml(t) { const d=document.createElement('div'); d.textContent=t; return d.innerHTML; }

async function send() {
  if (sending) return;
  const inp = document.getElementById('input');
  const text = inp.value.trim();
  if (!text) return;
  inp.value = '';

  if (!sessionId) await newChat();

  addMessage('user', text);
  sending = true;
  document.getElementById('sendBtn').disabled = true;

  // Typing indicator
  const msgs = document.getElementById('messages');
  const typing = document.createElement('div');
  typing.className = 'typing';
  typing.textContent = 'AION-C is thinking...';
  msgs.appendChild(typing);
  msgs.scrollTop = msgs.scrollHeight;

  try {
    const r = await fetch('/api/chat', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({message: text, session_id: sessionId})
    });
    const d = await r.json();

    typing.remove();
    addMessage('assistant', d.response, d.motors, d.scores, d.graph);
    updateRouting(d.scores);
    updateMem(d.mem_hits);
    loadSessions();
  } catch(e) {
    typing.textContent = 'Error: ' + e.message;
  }
  sending = false;
  document.getElementById('sendBtn').disabled = false;
  inp.focus();
}

function updateRouting(scores) {
  const el = document.getElementById('routingScores');
  if (!scores) return;
  el.innerHTML = Object.entries(scores).map(([name, val]) => {
    const c = COLORS[name] || '#6b7280';
    const w = Math.round(val * 100);
    return `<div class="score-bar">
      <span class="name">${name}</span>
      <div class="bar-bg"><div class="bar-fill" style="width:${w}%;background:${c}"></div></div>
      <span class="val">${(val*100).toFixed(0)}%</span>
    </div>`;
  }).join('');
}

function updateMem(hits) {
  const el = document.getElementById('memPanel');
  fetch('/api/mem').then(r=>r.json()).then(d => {
    let html = `<p style="font-size:11px;color:var(--text2);margin-bottom:8px">${d.stats.total_entries} entries</p>`;
    if (hits && hits.length) {
      html += '<p style="font-size:11px;color:var(--purple);margin-bottom:4px">Last search:</p>';
      hits.forEach(h => {
        html += `<div class="mem-entry"><span class="key">${h.key}</span> (${h.sim}) ${h.value}</div>`;
      });
    }
    d.entries.slice(0, 5).forEach(e => {
      html += `<div class="mem-entry"><span class="key">${e.key}</span> [${e.domain}] ${e.value}</div>`;
    });
    el.innerHTML = html;
  });
}

// Init
loadModelInfo();
newChat();
updateMem();
</script>
</body>
</html>"""


# ─── Run ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        import flask
    except ImportError:
        print("Installing Flask...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "flask", "-q"])

    print(f"\n  AION-C Web UI: http://localhost:3000\n")
    app.run(host="0.0.0.0", port=3000, debug=False)
