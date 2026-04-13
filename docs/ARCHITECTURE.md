# AION-C Architecture Deep Dive

This document describes every package in the AION-C codebase, the role it
plays in the system, its key design decisions, and the interfaces it
exposes. It is written for developers who need to read, modify, or extend
the code. For a higher-level view, see the main `README.md`. For the
cognitive layer specifically, see `COGNITIVE_LAYER.md`.

## Table of contents

1. [Design principles](#design-principles)
2. [The MoSE pipeline — end to end](#the-mose-pipeline)
3. [Core model packages](#core-model-packages)
4. [Agent layer](#agent-layer)
5. [Memory subsystems](#memory-subsystems)
6. [Training infrastructure](#training-infrastructure)
7. [World model and symbolic layer](#world-model-and-symbolic-layer)
8. [Data generation and format](#data-generation-and-format)
9. [Evaluation](#evaluation)
10. [Backend and UI](#backend-and-ui)
11. [Fase F cognitive packages](#fase-f-cognitive-packages)
12. [Testing philosophy](#testing-philosophy)
13. [Failure modes and invariants](#failure-modes-and-invariants)

---

## Design principles

### 1. Composable, not monolithic

Every subsystem is a standalone Python package with its own tests. The
MoSE pipeline imports and wires them, but each package can be used
independently. For example, `growth/` (LoRA adapters) works on any
`nn.Module` with Linear layers, not just motors from this codebase.

### 2. Tests are the specification

Before a package is considered done, its test file must cover the entire
public API with unit tests, plus at least one integration test against a
dependent package. The test files double as living documentation; reading
`tests/test_growth_adapters.py` is the fastest way to learn how the
adapter system works.

### 3. State dicts are first-class

Every component that has trainable parameters produces a clean
`state_dict` that can be saved, loaded, compared, and rolled back. The
BrainVersionManager relies on this to version the brain. The adapter
system relies on this to persist just the LoRA matrices without the base
weights. Always design with serialization in mind.

### 4. No frameworks that lock us in

PyTorch, FastAPI, and pytest are the only non-optional dependencies.
React is loaded from a CDN with no build step. There is no Hugging Face
Transformers dependency at training time, no database, no Docker. The
motivation is that anyone with Python and a Linux box can run everything.

### 5. Invariants are enforced, not assumed

Dataset must end with `[EOS]`? There is a test for it. Adapter detach
must restore base weights bit-a-bit? There is a test for it. Sleep cycle
must run all six phases in order? There is a test for it. If an
invariant matters, it gets a test.

### 6. Explicit is better than implicit

The canonical format is a string with named tags. The motor sequence in
a trajectory is an ordered list. The sleep cycle phases are a tuple in
hard-coded order. The sparse gate density target is a float in a config
dataclass. The system never relies on magic behavior or convention; it
enumerates what it does.

---

## The MoSE pipeline

The `router/pipeline.py` module defines `MoSEPipeline`, the top-level
`nn.Module` that represents the whole trainable system. Its `forward`
method is the clearest summary of the architecture.

```
def forward(self, token_ids, query_text=None):
    # 1. Encode
    concepts = self.encoder(token_ids)  # [B, L, D]

    # 2. Orchestrate (routing)
    orch_out = self.orchestrator(concepts, query_text)

    # 3. For each active motor:
    #    a. motor.build_graph(concepts) -> cryst_out
    #    b. motor.reason(graph, node_feats, n_iterations) -> cre_out
    #    c. motor.get_graph_repr(cre_out, k_nodes) -> [K, D]

    # 4. Unify motor representations
    #    unified = self.unifier(motor_reprs)  # [K, D]

    # 5. Decode
    #    decoder_out = self.decoder(token_ids, unified, concepts)
    return MoSEOutput(logits, ..., active_motors)
```

The five steps are independent: swapping the encoder for a different
architecture does not require changes in steps 2-5 as long as the output
shape stays `[B, L, D]`. Swapping a motor for a different one does not
require changes in steps 1, 2, 4, or 5 as long as it implements
`BaseMotor`.

### MoSEConfig

`MoSEConfig` is a dataclass with all the hyperparameters of the
pipeline. The notable fields:

- `hidden_dim` — the D in `[B, L, D]`. All motors must work in this
  dimensionality.
- `vocab_size` — set by the tokenizer at construction time. Typically
  32K for the BPE tokenizer used.
- `enc_n_layers`, `enc_state_dim`, `enc_expand`, `enc_d_conv`,
  `enc_ffn_mult` — Mamba encoder configuration.
- `orch_mlp_hidden`, `orch_max_motors`, `orch_min_confidence` —
  orchestrator configuration.
- `motor_max_nodes`, `motor_n_heads`, `motor_threshold` — motor
  internals.
- `unif_n_heads` — unifier attention heads.
- `dec_n_layers`, `dec_n_heads`, `dec_max_seq_len`, `dec_state_dim`,
  `dec_expand`, `dec_d_conv`, `dec_ffn_mult` — Mamba decoder.

### Parameter scaling

The tiny configuration has `hidden_dim=64` and produces 5.5M parameters.
The 1.1B configuration has `hidden_dim=1024` with 12 encoder layers and
16 decoder layers. The 3.5B configuration in `router/config_3_5b.py`
uses a formula-based parameter count and targets exactly 3.437B
parameters; it is not used for training today but exists for the future.

---

## Core model packages

### encoder/

```
encoder/
  __init__.py
  stream_encoder.py
  mamba_block.py
```

The encoder is a stack of Mamba state-space blocks. Each block has:

- A linear projection from `hidden_dim` to `expand * hidden_dim`
- A 1D convolution over sequence dimension with kernel `d_conv`
- A SiLU activation
- A selective state-space scan with learned parameters A, B, C, D
- A linear projection back to `hidden_dim`
- A residual connection and layer norm

Why Mamba instead of attention? State-space models have linear
complexity in sequence length, which matters for the 1024-token context
in the production 1.1B. They also tend to be more parameter-efficient
at small scales, which matters for the tiny model that must fit in 5M
parameters and still demonstrate the architecture.

The encoder is deliberately small relative to the rest of the system.
The hypothesis is that encoding tokens into a shared concept space is a
solved problem; the interesting work happens downstream in the motors.

### decoder/

Symmetric to the encoder, with one addition: cross-attention on the
unified motor graph representation. The decoder takes:

- `token_ids` of the response so far (during training, shifted input)
- `graph_repr [B, K, D]` from the unifier
- `concepts [B, L, D]` from the encoder (some motors pass these through)

And produces next-token logits. The cross-attention is what lets the
decoder read the motor's refined graph output rather than just the raw
token stream.

### motors/

```
motors/
  __init__.py
  base_motor.py
  cora/
    motor.py
    relations.py
  forge_c/
    motor.py
    relations.py
  axiom/
    motor.py
    relations.py
  muse/
    motor.py
    relations.py
  empathy/
    motor.py
    relations.py
```

`BaseMotor` is an `nn.Module` subclass that also inherits from `ABC`. It
defines five abstract methods that every motor must implement:

- `define_node_types() -> List[str]` — what kinds of nodes this motor
  recognizes (e.g. `entity`, `event`, `state` for CORA).
- `define_relations() -> List[str]` — what relations the motor tracks
  between nodes (e.g. `causes`, `enables`, `prevents`).
- `build_graph(concepts) -> CrystallizerOutput` — turn concept vectors
  into a typed graph.
- `reason(graph, node_features, n_iterations) -> CREOutput` — refine
  the graph via message passing.
- `get_graph_repr(cre_output, k_nodes) -> [K, D]` — project the refined
  graph to a fixed-size tensor for the decoder.

The five motors differ mainly in their relation vocabularies and in
some subclassing of the crystallizer. Their core `reason()` methods all
delegate to a shared `CausalReasoningEngine` but with different edge
functions.

### cre/

The `CausalReasoningEngine` is a GNN-style message passing module:

```
for iter in range(n_iterations):
    for each edge (src, dst, rel):
        msg = message_fn[rel](h[src], h[dst])
        agg[dst] += msg
    for each node:
        h[node] = GRU(h[node], agg[node])
    h = LayerNorm(h)
```

The message functions are typed: there is one `nn.Linear` per relation,
so `CORA` with 16 relations has 16 message functions and `FORGE-C` with
12 relations has 12. This is what allows the same architecture to
specialize per motor without sharing message weights across unrelated
edge types.

The GRU update and layer norm make the message passing stable over
many iterations. The adaptive reasoning levels in `agent/reasoning_levels`
exploit this by using 15-50 iterations for DEEP mode and just 1-3 for
LIGHT.

### crystallizer/

The crystallizer is the motor-specific subcomponent that turns the
concept stream `[B, L, D]` into a typed graph. It has three steps:

1. **Node detection.** A scorer assigns a probability to each position
   in the sequence of being a node. Positions above a threshold become
   nodes.
2. **Node typing.** A classifier assigns each node to one of the motor's
   node types.
3. **Relation scoring.** For each pair of nodes, a relation scorer
   produces logits over possible relation types. The top-k edges are
   kept.

The output is a `CrystallizerOutput` with the graph, node features, and
differentiable tensors needed for training (the node scores and
relation logits flow into the LM loss).

### orchestrator/

The orchestrator takes pooled concept vectors and produces a softmax
distribution over the five motors. It has two paths:

- During training, the distribution is supervised from the dataset's
  `domain` field. A cross-entropy loss called `routing_loss` pushes the
  orchestrator to pick the right motor for the domain.
- During inference, the orchestrator returns up to `orch_max_motors`
  active motors above `orch_min_confidence`. The pipeline then runs
  each active motor and passes their outputs to the unifier.

During training we also compute a `balance_loss` that penalizes
over-use of any single motor via an exponential moving average of
activation probabilities. This prevents the router from collapsing to
"always pick cora" which would break specialization.

### unifier/

The unifier takes a list of `[K, D]` tensors (one per active motor) and
produces a single `[K, D]` tensor. The implementation is a multi-head
cross-attention: each output position attends to all input positions
across all motors.

The unifier is also where the trajectory-compositional logic
eventually plugs in; today it handles the case where multiple motors
are active in a single forward pass, but the `composition/` package
extends this to handle sequences of motor invocations with explicit
dependencies.

### tokenizer/

Contains `aion_32k.model`, the 32K BPE SentencePiece model trained on
a mixed Spanish and English corpus. Character coverage is supplemented
with explicit ASCII and extended Latin characters so that the 32K
vocabulary does not miss common code and math symbols.

The tokenizer is loaded via `experiments.train_production.build_tokenizer`
which returns an object with an `encode(text, max_len)` method
(compatible with `sentencepiece.SentencePieceProcessor`).

### router/

`router/pipeline.py` contains `MoSEPipeline` and `MoSEConfig`. The
pipeline wiring is deliberately explicit so that someone reading the
code can trace the data flow without jumping between files. The
`forward()` method is roughly 80 lines and can be read top to bottom.

`router/config_3_5b.py` contains the formula-based 3.5B configuration.
It is not used today for training but serves as the target for the
next scale step after the 1.1B.

---

## Agent layer

### agent/tools.py + agent/tool_executor.py

The tool system is the interface through which the model can take
actions beyond generating text. There are six tools:

| Tool | Input | Output | Sandbox |
|------|-------|--------|---------|
| `WriteFileTool` | `path`, `content` | write status | path must be inside `output/` |
| `EditFileTool` | `path`, `old`, `new` | diff | path must exist under `output/` |
| `RunCodeTool` | `code`, `language` | stdout, stderr | subprocess with 60s cap, no network |
| `CallApiTool` | `url`, `method`, `headers`, `body` | response JSON | deny-all whitelist by default |
| `SearchMemTool` | `query`, `domain`, `top_k` | results | read-only MEM access |
| `StoreMemTool` | `key`, `value`, `domain` | success | writes to MEM |

The `ToolExecutor` handles parsing tool calls from model output. The
parser uses a balanced-brace JSON extractor which is more robust than
regex at handling nested JSON with strings that contain braces. It
runs on the full generated text, extracts every `[TOOL: {...}]` block,
and executes them in order.

Aliases: `search_web` maps to `CallApiTool` with a pre-configured
endpoint, and `read_file` maps to a read-only variant of
`EditFileTool`. These exist because the training data from Fase B uses
`search_web` as a tool name.

### agent/planner.py

```python
@dataclass
class PlanStep:
    id: str
    description: str
    motor_hint: Optional[str]
    status: str  # PENDING / RUNNING / DONE / FAILED / SKIPPED
    attempts: int
    last_error: Optional[str]
    result: Optional[StepResult]

@dataclass
class Plan:
    id: str
    query: str
    steps: List[PlanStep]
    status: str  # PLANNING / RUNNING / COMPLETE / TIMED_OUT / FAILED
    started_at: float
    deadline: Optional[float]
    replans_remaining: int
```

The planner has two retry loops:

1. **Per-step retries.** Each step can be attempted up to `max_attempts`
   times (default 2). If an attempt fails, the step goes back to
   `PENDING` and is retried.
2. **Replans.** If a step fails permanently, the planner can invoke
   `_replan` up to `max_replans` times (default 1). Replanning marks
   the failed step as `SKIPPED` and creates replacement steps. The
   `SKIPPED` state is preserved for auditing.

Timeouts are wall-clock: between steps the planner checks
`now > deadline` and if so sets the plan status to `TIMED_OUT` and
returns the partial state. This is deliberate: a plan that times out
should not silently discard its progress.

Persistence: `attach_to_mem(plan, mem, key='current_task')` serializes
the plan to MEM under the `planner` domain. `load_from_mem` reconstitutes
it. Plans survive across backend restarts.

### agent/skills.py

The skills system injects domain-specific guidance into the prompt
based on query similarity.

```python
@dataclass
class Skill:
    name: str
    description: str
    content: str
    tags: List[str]
    domain: str
    examples: List[str]
```

The 11 skill files live in `skills/*.md` with YAML frontmatter. The
`SkillsLoader` parses the frontmatter without a PyYAML dependency
(hand-rolled parser, approximately 30 lines). At query time it
computes a similarity score between the query and each skill's content
using an injectable similarity function; skills above threshold 0.5
are injected into the prompt as `[SKILL: ...]` blocks.

The 11 skills:

- `python_best_practices`
- `javascript_patterns`
- `causal_reasoning`
- `math_step_by_step`
- `creative_writing`
- `empathetic_response`
- `identity`
- `code_debugging`
- `spanish_responses`
- `sqlite_patterns`
- `web_development`

The identity skill is special: it is always injected regardless of
similarity because it defines the assistant's persona. This was a hard
lesson from the first tiny training run where the identity skill being
rare in the data caused OOD behavior when it was injected at inference
time.

### agent/self_check.py

```python
@dataclass
class SelfCheckResult:
    confidence: float       # [0, 1]
    flags: List[str]        # e.g. "too_short", "echo_detected"
    policy: str             # "respond_directly" / "respond_with_disclaimer" / "search_then_respond"
    notes: Dict[str, Any]
```

The checker runs a battery of static checks:

- `length`: is the response suspiciously short or long?
- `echo`: does the response repeat the user's query verbatim?
- `python_syntax`: if the response contains Python code, does it parse?
- `bracket_balance`: are parentheses, brackets, braces, and quotes
  balanced?
- `numeric_consistency`: if the response claims arithmetic, does it
  check out?

The `confidence_from_probs` function takes the first five token
probabilities from the decoder and computes a geometric mean as a
rough confidence signal. This is combined with the static check results
to produce a final policy.

There is also an `ErrorLog` that records when the self-checker flags an
issue. Each entry has `error`, `cause`, `prevention`, and `domain`. This
log is written to MEM and available for debugging why the model
produces bad outputs.

### agent/reasoning_levels.py

```python
class ReasoningLevel(IntEnum):
    INSTANT = 0   # 0 iterations, for greetings
    LIGHT = 1     # 1-3 iterations
    NORMAL = 2    # 5-10 iterations
    DEEP = 3      # 15-50 iterations

@dataclass
class LevelDecision:
    level: ReasoningLevel
    label: str
    iterations: int
    show_thinking: bool
    reason: str
```

The `LevelDecider.decide()` method takes the query, orchestrator scores,
whether a skill was injected, and whether MEM has relevant results.
It uses a combination of trigger words and heuristics:

- INSTANT triggers: `hola`, `hi`, `buenos días`, `how are you`
- DEEP triggers: `demuestra`, `prove`, `analyze`, `refactor`,
  `design from scratch`, `explain in detail`
- If a skill was injected, downshift one level (the assumption is that
  the skill already contains the domain knowledge so less internal
  reasoning is needed)
- Default to NORMAL

`show_thinking` is True for NORMAL and DEEP, False for INSTANT and
LIGHT. This controls whether the UI shows a "thinking" indicator to
the user.

### agent/lifecycle.py

A simple FSM with four states:

```
ACTIVE ----start_responding---> ACTIVE
  |
  +---stop_responding---> IDLE
  +---start_learning---> LEARNING
  +---go_to_sleep---> SLEEPING

IDLE ---start_responding---> ACTIVE
IDLE ---go_to_sleep---> SLEEPING

LEARNING ---stop_learning---> IDLE
SLEEPING ---wake_up---> IDLE
```

The `ALLOWED_TRANSITIONS` map explicitly enumerates which state-to-state
transitions are valid; any other attempt raises an exception. Each
transition can have `on_enter` and `on_exit` callbacks. Transitions are
logged to a history list that is truncated at a configurable length to
keep memory bounded.

This is used by the backend to know when it is safe to dispatch
queries, when a sleep cycle is running and must not be disturbed, and
when an auto-learn is in progress.

### agent/goals.py

```python
@dataclass
class Goal:
    id: str
    title: str
    description: str
    source: GoalSource     # USER / SYSTEM / PROPOSED
    status: str            # ACTIVE / PAUSED / DONE / REJECTED
    created_at: float
    parent_mission_id: Optional[str]
    progress: float

@dataclass
class Task:
    id: str
    title: str
    status: str
    kind: str              # WORK / HOUSEKEEPING
    created_at: float
```

The `GoalsManager` has:

- A permanent mission (`"serve the user helpfully, honestly, and safely"`)
  that cannot be removed or modified.
- Active missions created by the user.
- Proposed goals created by the system that require user approval.
- Active goals (user-approved).
- Pending tasks (work items that contribute to a goal).
- Housekeeping tasks (system maintenance like "index MEM at startup").
- A routine log for scheduled activities.

The manager is serializable and persists across backend restarts via
`snapshot()` and `from_snapshot()`.

---

## Memory subsystems

### memory/semantic_store.py

The main external memory. Facts and their embeddings live here.

```python
class SemanticStore:
    def __init__(self, encoder, tokenizer, similarity_threshold=0.0):
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.entries: List[MemoryEntry] = []

    def store(self, key, value, domain="general", source="user"):
        # compute embedding from value via encoder
        embedding = self._embed(value)
        self.entries.append(MemoryEntry(...))

    def search(self, query, top_k=5, domain=None):
        q_embed = self._embed(query)
        scored = [(e, cosine(q_embed, e.embedding)) for e in self.entries]
        if domain:
            scored = [(e, s) for e, s in scored if e.domain == domain]
        return sorted(scored, key=lambda x: -x[1])[:top_k]
```

The embedding is computed by pooling the encoder output over the
sequence dimension. This means the semantic store uses the same concept
space as the rest of the model, which is important: facts stored in the
SemanticStore are "understood" by the motors in the same way as raw
input.

There is no database. Entries live in a Python list and are persisted
via JSON pickling. For the 5.5M tiny model with thousands of entries,
linear search is fast enough. At 10K+ entries an index would be needed.

### memory/user_model.py

```python
@dataclass
class UserModel:
    name: Optional[str]
    language: Optional[str]
    level: Optional[str]  # beginner / intermediate / expert
    tone: Optional[str]   # formal / casual / playful
    projects: List[str]
    facts: Dict[str, Any]
```

Each field has a validator. Names are stripped of punctuation; language
must be one of a small allow-list; level must be one of three values;
tone must be one of three values.

The `update_from_text` method runs simple NLP heuristics on user input
to extract potential updates (e.g. "my name is Jesús" → `name="Jesús"`).
It is deliberately conservative: it only updates a field if it is
currently None or if the new value has strong evidence.

Persisted to MEM under `domain="user_model"`, key=`"user_profile"`.

### memory/response_cache.py

```python
class ResponseCache:
    def __init__(self, max_size=256, ttl_seconds=3600):
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        ...

    def get(self, query): ...
    def set(self, query, response): ...
    def invalidate_by_substring(self, substring): ...
    def invalidate_by_predicate(self, pred): ...
    def stats(self): ...
```

The key is a normalized version of the query (lowercased, whitespace
collapsed). LRU eviction when `max_size` is exceeded. TTL check on
`get` for lazy expiration.

`invalidate_by_substring` is useful when a fact changes: if the user
tells the system that X is actually Y, we invalidate every cached
response containing "X" so the next query gets a fresh one.

### memory/conversation_history.py

```python
class ConversationHistory:
    def __init__(self, recent_turns=4, mid_turns=6, old_threshold=10):
        self.turns: List[Turn] = []

    def add_user(self, text): ...
    def add_assistant(self, text, motor=None, level=None, scores=None): ...

    def summary_block(self, summarizer): ...   # uses injectable fn
    def key_facts(self): ...                    # rule-based extraction
    def render_context(self): ...               # [FACTS:][SUMMARY:][USER:][AION:]
```

The history is tiered: recent turns are shown verbatim, mid turns are
replaced with a summary, and old turns are dropped (their content lives
in extracted facts). The tier boundaries are configurable.

`summary_block` takes an injectable summarizer so that tests can use a
deterministic stub while production can use the real model.
`render_context` produces a prompt-ready string with the four tag
blocks.

---

## Training infrastructure

### training/anti_forgetting.py

The five layers of catastrophic-forgetting protection, each a separate
class that can be used independently.

#### MotorIsolation

```python
class MotorIsolation:
    def __init__(self, pipeline, active_motors: Set[str]):
        ...

    def __enter__(self):
        for name, motor in self.pipeline.motors.items():
            if name not in self.active_motors:
                for p in motor.parameters():
                    self._frozen.append((p, p.requires_grad))
                    p.requires_grad_(False)
        return self

    def __exit__(self, *args):
        for p, was_trainable in self._frozen:
            p.requires_grad_(was_trainable)
```

Used during auto-learn: when teaching a new concept to CORA, freeze
everything except CORA's parameters so no gradient flows into the other
motors.

#### WeightImportanceTracker

Running mean and variance of gradients per parameter, with a
`protection_factor(p)` method that returns a scalar multiplier. The
multiplier is small for weights with high importance (we want to protect
them) and large for weights with low importance (we let them move
freely). During backprop, the gradients are scaled by this factor.

This is a simplified implementation of Elastic Weight Consolidation
(Kirkpatrick et al.).

#### ExamRunner

```python
class ExamRunner:
    def __init__(self, items, generator_fn, matcher_fn):
        self.items = items
        self.generator_fn = generator_fn
        self.matcher_fn = matcher_fn

    def run(self) -> ExamResult:
        hits = 0
        for item in self.items:
            output = self.generator_fn(item.prompt)
            if self.matcher_fn(output, item.expected):
                hits += 1
        return ExamResult(score=hits / len(self.items), ...)
```

Runs a fixed set of test prompts through a generator and matches each
output against an expected answer. Returns a pass rate. Used before
and after training to detect regression.

#### RollbackManager

```python
class RollbackManager:
    def snapshot(self, pipeline): ...       # deep copy state_dict
    def rollback(self, pipeline): ...       # restore from snapshot
    def should_rollback(self, before, after, max_drop=0.02): ...
```

The `should_rollback` logic is "if the post-training exam dropped by
more than `max_drop`, revert the weights". This is a hard safety net
that prevents any training step from silently breaking what was
working before.

#### SelectiveReplay

Maintains a small buffer of gradient vectors from prior examples. When
training on a new example, computes the overlap between the current
weight delta and each stored gradient. Examples with high overlap are
replayed (re-trained on) to reinforce the patterns they represented.

This is a pragmatic approximation of experience replay that works with
small buffers (dozens of examples) rather than full replay buffers
(millions).

### brain/version_manager.py

```python
@dataclass
class BrainVersion:
    id: str                          # v1, v2, ...
    parent_id: Optional[str]
    created_at: float
    notes: str
    metrics: Dict[str, float]
    metadata: Dict[str, Any]

class BrainVersionManager:
    def __init__(self, root_dir: Path): ...

    def save_version(self, state_dict, notes, metrics, metadata) -> BrainVersion:
        version_id = self._next_id()
        version_dir = self.root / version_id
        torch.save(state_dict, version_dir / "weights.pt")
        (version_dir / "metadata.json").write_text(...)
        return BrainVersion(...)

    def load_version(self, version_id) -> Tuple[Dict, BrainVersion]: ...
    def list_versions(self) -> List[BrainVersion]: ...
    def latest(self) -> Optional[BrainVersion]: ...
    def compare(self, v1_id, v2_id) -> Dict[str, Tuple[float, float, float]]: ...
    def rollback(self, to_id) -> BrainVersion: ...
    def delete_version(self, version_id) -> bool: ...
```

Each version lives in its own subdirectory (`brain/v1/`, `brain/v2/`,
...). The `parent_id` forms a linear history (no branching in v1;
branching is a future extension). Metrics are per-version and can be
compared with `compare(v1, v2)` which returns a dict of
`{metric_name: (v1_value, v2_value, delta)}`.

Rollback moves the `latest` pointer back but does not delete intermediate
versions. Delete is a separate operation that permanently removes a
version directory.

---

## World model and symbolic layer

### world_model/scratch_pad.py

```python
@dataclass
class SlotSpec:
    index: int
    name: str
    expected_type: type
    required: bool
    description: str

@dataclass
class ScratchPadSchema:
    motor_name: str
    slots: List[SlotSpec]

class ScratchPad:
    def __init__(self, schema, n_slots=16): ...

    def set(self, index, value): ...
    def get(self, index): ...
    def set_by_name(self, name, value): ...
    def get_by_name(self, name): ...
    def validate(self) -> List[str]: ...  # returns error messages
```

Each motor has its own schema with named slots. For AXIOM, the slots
include `input_expression`, `parsed_number_1`, `parsed_number_2`,
`operation`, `intermediate_result`, `final_result`. The schema makes the
simulator's intermediate state inspectable in the UI.

### world_model/simulator.py

Five simulators, one per motor. Each implements a single `simulate`
method that takes a query and populates the scratch pad with the
reasoning trace.

The AxiomSimulator handles simple percentage and arithmetic queries:

```
"15% of 240"
  → parsed_number_1 = 15
  → parsed_number_2 = 240
  → operation = "percentage"
  → intermediate = 0.15 * 240
  → final_result = 36.0
```

Simulators are heuristic and deliberate: the point is not to be a full
symbolic solver, it is to provide a check for the neural motor's
answer. If the simulator says 36 and the decoder says 38, the
self-checker flags a numeric inconsistency.

### world_model/verifier.py

```python
class ScratchPadVerifier:
    def verify(self, pad: ScratchPad) -> VerificationResult:
        errors = []
        for slot in pad.schema.slots:
            if slot.required and pad.get(slot.index) is None:
                errors.append(f"required slot {slot.name} is None")
            elif pad.get(slot.index) is not None:
                val = pad.get(slot.index)
                if not isinstance(val, slot.expected_type):
                    errors.append(f"slot {slot.name} type mismatch")
        return VerificationResult(ok=(len(errors) == 0), errors=errors)
```

The verifier runs generic checks (required slots present, types correct)
plus motor-specific checks registered per simulator. If verification
fails, the `SimulationLoop` corrects the scratch pad and re-simulates.

### symbolic/

The symbolic layer is deliberately lightweight: no torch dependency, no
ML. Just dataclass-based graphs and rule functions.

```python
@dataclass
class SymbolicNode:
    id: str
    type: str
    value: Any

@dataclass
class SymbolicEdge:
    source: str
    target: str
    relation: str

class SymbolicGraph:
    def __init__(self): ...
    def add_node(self, node): ...
    def add_edge(self, edge): ...
    def has_path(self, src, dst) -> bool: ...
    def has_cycle(self) -> bool: ...
    def copy(self) -> SymbolicGraph: ...
    def remove_node(self, node_id): ...
```

Rules are functions that take a graph and return a modified graph plus
a list of applied rule names:

```python
def transitivity_rule(graph: SymbolicGraph) -> Tuple[SymbolicGraph, List[str]]:
    new_graph = graph.copy()
    applied = []
    for edge_ab in new_graph.edges:
        if edge_ab.relation == "implies":
            for edge_bc in new_graph.edges:
                if edge_bc.source == edge_ab.target and edge_bc.relation == "implies":
                    # A implies B, B implies C => A implies C
                    new_graph.add_edge(SymbolicEdge(edge_ab.source, edge_bc.target, "implies"))
                    applied.append(f"transitivity: {edge_ab.source} -> {edge_bc.target}")
    return new_graph, applied
```

The 11 rules across the three rule files implement:

**AXIOM:** Transitivity, Contradiction, Substitution, Arithmetic
**FORGE-C:** TypeCheck, NullCheck, LoopDetection, DeadCode
**CORA:** CausalTransitivity, CausalContradiction, Counterfactual

### symbolic/engine.py

```python
class SymbolicEngine:
    def __init__(self, rules: List[Callable]): ...

    def apply_all(self, graph, max_iters=5) -> EngineResult:
        for iter in range(max_iters):
            new_graph, applied_this_iter = self._apply_once(graph)
            if new_graph == graph:  # fixed point reached
                break
            graph = new_graph
        return EngineResult(
            final_graph=graph,
            applied_rules=all_applied,
            conflicts=conflicts,
            notes=notes,
            added_nodes=added_nodes,
            added_edges=added_edges,
            removed_edges=removed_edges,
        )
```

The engine iterates to a fixed point or `max_iters`. Each iteration
applies every rule and accumulates the changes. Conflicts are
detected when two rules produce contradictory edges (e.g.
`A causes B` and `A prevents B` both present). The conflict resolution
policy removes the "prevents" edge because the neural layer is more
likely to have made a mistake there than the "causes" edge.

The `build_engine_for_motor("axiom" | "forge_c" | "cora")` factory
returns an engine loaded with the rules appropriate for that motor.

---

## Data generation and format

### synth/canonical_format.py

The single source of truth for the training data format.

```
EXAMPLE := [SKILL: ...]? [MEM: ...]? TURN+ [EOS]
TURN    := [USER: ...] [TOOL: json]? [RESULT: ...]? [AION: ...]
```

Rules:

- Every example ends with `[EOS]` exactly once.
- `[SKILL:]` and `[MEM:]` appear before the first `[USER:]`.
- `[TOOL:]` is always followed by `[RESULT:]`.
- Multi-turn alternates `[USER:]` and `[AION:]` strictly.

`format_record()` builds the string from individual fields.
`parse_canonical()` is the inverse, returning a `CanonicalRecord` with
all the parts extracted. `canonicalize_legacy()` converts the old
`{input, output, graph, ...}` format from Fase B into the canonical
format, used once during the Fase B unification.

`CanonicalRecord` has:

- `text` (the canonical string)
- `has_skill`, `has_mem`, `has_tool` (booleans from parsing)
- `is_multi_turn`, `turn_count`
- `domain` ("cora" / "forge_c" / "muse" / "axiom" / "empathy" /
  "general" / "metacognitive")
- `language` ("en" / "es")
- `type` ("legacy" / "tool" / "skill" / "mem" / "identity" /
  "multi_turn" / "metacognitive")
- `metadata` (open dict for subcategory, expected_motor_sequence, etc.)

### synth/canonical_dataloader.py

```python
def load_canonical_records(path) -> List[CanonicalRecord]: ...
def encode_record(tok, record, max_len=1024) -> List[int]: ...
def weighted_sampler_indices(records, n_steps, target_ratio=0.5, seed=42) -> List[int]: ...
def domain_to_motor_idx(domain: str) -> int: ...

MOTOR_NAMES = ["cora", "forge_c", "muse", "axiom", "empathy"]
EOS_TOKEN_ID = <from BPE>
```

The weighted sampler balances examples with skill-or-mem against those
without. The target ratio is 0.5 by default so the model sees both
styles equally. Without the weighted sampler, the natural distribution
heavily favors examples with `[MEM:]` blocks because the 57.5K legacy
records all have them.

`encode_record` handles the multi-turn case by concatenating all turns
with sentinel tokens in the correct order.

### synth/generators

- **`conversational_gen.py`** — Multi-turn conversations. Seeded so
  output is deterministic. Each conversation has 2-6 turns with a
  consistent topic.
- **`tool_gen.py`** — Examples where the model uses one of the six
  tools. Covers all tools and all domains.
- **`skill_injected_gen.py`** — Examples where a `[SKILL:]` is present
  in the prompt and the response explicitly uses the skill's guidance.
- **`mem_injected_gen.py`** — Examples where a `[MEM:]` contains a
  fact and the response explicitly references it.
- **`identity_gen.py`** — Identity skill examples. The 500 in the base
  dataset plus 1000 to be added to reach the recommended 1500 minimum.
- **`metacognitive_gen.py`** — Five categories × 500 examples each:
  `out_of_knowledge`, `propose_learning`, `context_overflow`,
  `out_of_capability`, `low_confidence_disclaimer`. See
  `../TRAINING_SPEC.md` section 4 for the full specifications.

### synth/dataset_unifier.py

```python
def read_jsonl(path) -> Iterator[Dict]: ...
def write_jsonl(records, path) -> None: ...
def merge(sources: List[Path], dest: Path) -> None: ...
def fix_eos(record) -> Dict: ...
def compute_diversity_exact(records) -> DiversityStats: ...
def verify_eos_all(records) -> bool: ...
```

Used during the Fase B pipeline to merge the 12.5K new examples with
the 57.5K legacy records and to verify the canonical format invariants
(every example has exactly one `[EOS]`, etc.).

---

## Evaluation

### evaluation/eval_prompts.py

50 canonical eval prompts, 10 per motor domain, all out-of-sample
(not in the training data). Each prompt is:

```python
EvalPrompt(
    query="15% de 240",
    expected_substring="36",
    references=["36", "el 15 por ciento de 240 es 36", "15/100 * 240 = 36"],
    language="es",
    difficulty="easy",
    domain="axiom",
)
```

The 50 prompts are the benchmark we use to track training progress.
Every 200 training steps the trainer generates responses for all 50 and
computes `exact_match`, `bleu`, and `routing_accuracy`, then combines
them into the single `combined` score.

### evaluation/metrics.py

```python
def bleu_score(reference, hypothesis, n=2) -> float:
    # BLEU-1+2 with additive smoothing and brevity penalty
    # no external deps (no nltk, no sacrebleu)

def multi_reference_bleu(references, hypothesis) -> float:
    return max(bleu_score(ref, hypothesis) for ref in references)

def exact_match(expected_substring, response) -> float:
    return 1.0 if expected_substring.lower() in response.lower() else 0.0

def generation_quality_score(prompts, gen_fn) -> GenerationQualityResult:
    # Runs gen_fn on every prompt, computes each metric,
    # combines with weights 0.4 exact + 0.2 bleu + 0.4 routing,
    # returns per-domain breakdown and overall combined
```

The combined score weights are chosen so that routing accuracy (which
the tiny model already achieves at 98.2%) contributes 40%, exact match
(which requires the model to actually generate the right content) is
also 40%, and BLEU is a soft supplementary signal at 20%.

---

## Backend and UI

### backend/app_fastapi.py

The backend is a single FastAPI application with an injectable
`AppState` dataclass. The `create_app(state)` factory makes testing
easy: tests inject a mock state with `None` for the model and a
`FakeMem`, production calls `build_full_state()` which loads the
tiny checkpoint and wires all 16 cognitive components.

Endpoints:

```
GET  /                           → serves the static HTML
GET  /api/info                   → model metadata
POST /api/session                → create new session
GET  /api/sessions               → list sessions
GET  /api/session/{sid}          → get session turns
DELETE /api/session/{sid}        → delete session
GET  /api/mem                    → list MEM entries
POST /api/upload                 → upload file to sandbox
GET  /api/download/{file}        → download from sandbox
GET  /api/files                  → list sandbox files
GET  /api/cache/stats            → cache hit/miss stats
GET  /api/lifecycle              → lifecycle state + history
GET  /api/user/{sid}             → user model for session
GET  /api/goals                  → goals snapshot
POST /api/goals/add              → add goal / task / mission
POST /api/goals/approve/{gid}    → approve proposed goal
POST /api/goals/reject/{gid}     → reject proposed goal
GET  /api/adapters               → list adapters (Fase F)
DELETE /api/adapters/{motor}/{concept} → delete adapter
POST /api/trajectory/plan        → plan a composite trajectory
POST /api/trajectory/execute     → plan and execute
POST /api/sleep                  → force sleep cycle
GET  /api/sleep/last             → last sleep log
POST /api/sleep/episode          → add episode to buffer
POST /api/feedback               → record thumbs up/down/correction
GET  /api/feedback/ledger        → reward ledger snapshot
GET  /api/memory/hierarchy       → hierarchical memory by level
GET  /api/sparse/report          → sparse activation report
WS   /ws/chat/{sid}              → streaming chat
```

The WebSocket handler emits the following message types:

- `token` — a chunk of the response
- `meta` — routing scores, active motors, level decision
- `thinking` — thinking indicator on/off
- `graph` — per-message causal graph
- `scratchpad` — world model simulation result
- `tool` — tool execution record
- `check` — self-check result
- `plan` — planner output for multi-step queries
- `done` — end of stream with the complete response
- `error` — error detail

### backend/static/index.html

A single HTML file loaded once. React 18 and vis-network are pulled
from CDNs. The entire UI is roughly 1100 lines of JSX in `<script>`
tags.

The state management is simple: one top-level `App` component with
hooks for each piece of state (messages, routing scores, info,
adapters, trajectory, sleep log, ...). Child components
(`Sidebar`, `ChatArea`, `RightPanel`, `ChatMessage`, `FeedbackButtons`)
receive props.

The style uses CSS variables for the dark theme. No CSS framework. The
result is a single-file deployable static site that works behind any
reverse proxy.

---

## Fase F cognitive packages

These are documented in detail in `COGNITIVE_LAYER.md`. This section
gives only a one-paragraph summary per package.

### growth/

LoRA-style adapters. `LoRALinear` wraps an `nn.Linear` with two
low-rank matrices. `AdapterPack` groups LoRALinears across a motor.
`attach_adapter_pack(motor, pack)` monkey-patches the motor to route
through LoRA forwards. `detach_adapter_pack(motor, pack)` restores
the original Linears bit-a-bit. `AdapterRegistry` handles persistence
and lookup. `GrowthPolicy` decides adapter vs expansion vs sub-motor.

### composition/

Trajectory planner. `TrajectoryPlanner.plan(query)` produces a
`Trajectory` (a list of `TrajectoryStep` with motor name, sub-goal,
and dependencies). `CompositeOrchestrator.execute(trajectory)` runs
each step with a generate function, passes prior outputs to dependent
steps, and produces a `TrajectoryResult`. `TrajectoryUnifier.fuse()`
combines the final step outputs.

### sleep/

Sleep cycle. `EpisodicBuffer` holds episodes pending processing.
`SleepCycle.run()` executes six phases in strict order: recollect,
score, prune, compress, consolidate, followups. Each phase has an
injectable hook. `SleepDaemon` schedules runs based on inactivity,
overflow, or manual trigger. The backend runs it as an asyncio
background task.

### pruning/

Four-signal pruning. `PruneSignals` has frequency, recency, utility,
retrieval cost. `MemoryPruner.prune(items)` returns a `PruneReport`
with decisions (KEEP / PROMOTE / COMPRESS / DELETE) and dynamic TTLs.
Integrated into the sleep cycle via `sleep_prune_hook`.

### reward/

Probabilistic reward. `RewardSignals` combines explicit, implicit,
and intrinsic signals. `RewardEstimator.compute()` returns a
`RewardEstimate` with mean, std, and conservative lower bound.
`ImplicitDetector` parses conversation for thanks, re-asks, code
copies, abandonment, and no-correction continues. `RewardLedger`
accumulates per-motor or per-motor-adapter stats.

### compression/

Three-level hierarchical memory. `MemoryLevel` enum for EPISODIC /
STABLE / NUCLEAR. `HierarchicalStore` holds `StoredItem`s across
levels. `Clusterer` uses greedy Jaccard to group similar episodic
items into clusters. `HierarchicalCompressor.compress_episodic_to_stable()`
creates stable items from clusters with anchor preservation.
`promote_stable_to_nuclear()` promotes frequently-used stables to
abstract nuclear concepts.

### sparse/

Activation sparsity via gating. `SparseConfig` sets target density,
mode (continuous or binary), gate hidden dim, temperature, threshold.
`GateNetwork` is a small MLP producing an activation mask from the
input. `SparseLinear` wraps an `nn.Linear` with a gate.
`attach_sparse_gates(motor, targets, config)` installs gates.
`SparsityTracker` measures actual density after forward passes.
`sparsity_loss(pipeline, target)` produces the loss term.

### experiments/fase_f/

The five validation experiments from Parte 21. `common.py` has the
`FakeMotor`, `make_exam`, `exam_pass_rate`, and `ExperimentReport`
helpers. Each experiment has its own file. `run_all.py` runs all five
with FakeMotor (fast, for CI). `run_real.py` runs them against the real
tiny pipeline loaded from `checkpoints/tiny_canonical.pt`.
`real_pipeline.py` has the loader and real-motor exam helpers.

---

## Testing philosophy

The test suite is 2856 passing tests across 67 test files. Every
package has its own test file. Integration tests live in
`test_backend_wiring_integration.py` which exercises the full backend
with all components wired.

Key principles:

- **Tests must run in under 60 seconds.** Slow tests are either
  parallelized or marked `@pytest.mark.slow` and excluded from default
  runs. The full suite runs in about 52 seconds on a modern machine.
- **Tests use small synthetic data.** Real model checkpoints are not
  loaded in unit tests; the `FakeMotor` and `FakeMem` in
  `tests/test_backend_fastapi.py` let the backend run without PyTorch
  weights.
- **Invariants are tested exhaustively.** Every promise the system
  makes (adapter detach is bit-a-bit, pipeline forward preserves
  shape, sleep cycle runs all six phases in order) has a test that
  would fail if the promise were broken.
- **Regressions are caught at commit time.** The baseline count of
  passing tests is documented; every change must preserve it.

---

## Failure modes and invariants

This is the list of things that have gone wrong in the history of the
project, each with the resulting invariant that prevents recurrence.

### Failure: `val_loss=nan` in early training of the H200 3.6B

Cause: unstable initialization combined with the Mamba scan's exp of
large values. Fix: initialization scale reduced, gradient clipping
added, AMP autocast configured to fp16 with GradScaler. Invariant:
every training run has a finite-loss check at each step; if
`loss.item()` is non-finite the step is skipped.

### Failure: tiny model produced garbage when `[SKILL: identity]` was injected

Cause: the identity skill appeared in only 500 of the 70K dataset
records, a rate too low for the tiny model to learn it robustly.
Out-of-distribution inputs at inference time produced garbage.
Invariant: every category in `TRAINING_SPEC.md` has a coverage floor,
and the matrix is verified before training starts. The identity
category floor was raised to 1500 for the 1.1B training.

### Failure: EOS missing from some generated examples

Cause: the legacy `{input, output}` format did not always include a
clear termination marker, and the canonicalizer missed some records.
Fix: `fix_eos()` added to the dataset unifier, and `verify_eos_all()`
run on every dataset before training. Invariant: 100% EOS coverage
verified on every dataset.

### Failure: adapter weights leaked into base weights during training

Cause: the original design had the base Linear as a hidden attribute
not registered as a submodule, which meant `motor.parameters()`
returned only the adapter parameters after attach, confusing the
optimizer. Fix: base Linear is now a normal submodule; the
AdapterPack's `adapter_state_dict()` filters at save time to produce
just the LoRA matrices. Invariant: every adapter test verifies both
forward equivalence with the gate disabled and bit-a-bit restoration
after detach.

### Failure: sleep cycle skipped phases under exceptions

Cause: an exception in an early phase short-circuited the remaining
phases, leaving the system in an inconsistent state. Fix: phases are
wrapped in try/except that records the error in the log but still
runs subsequent phases. Invariant: the sleep cycle log always
contains six phases, even if some are error records.

### Failure: reward ledger lost on backend restart

Cause: the ledger was in-memory only. Every backend restart cleared
the feedback history. Fix: JSONL persistence added with load-on-boot
and save-on-update. Invariant: feedback survives restarts and the
ledger is verified by an integration test that starts a backend,
writes feedback, stops, starts a new backend, and reads the same
ledger back.

---

This document is long. It is long because the system is large. A
smaller summary would have forced me to choose between completeness
and precision, and neither is acceptable for documentation that must
serve as the reference for people maintaining this code. If you read
this far, you now know everything structural about AION-C; the
remaining files in `docs/` cover specific subsystems in even greater
depth.
