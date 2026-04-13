# The AION-C Cognitive Layer (Parts 22-27)

This document describes Fase F, the layer of capabilities that turns
AION-C from a competent modular model into a system that learns, forgets,
grows, and reasons compositionally over time. Each of the six parts
(22, 22.5, 23, 24, 25, 26, 27) is covered here with design rationale,
pseudocode, and integration notes.

For the higher-level rationale (why these six parts exist as a coherent
block), see the main `README.md`. For the architecture of the underlying
model that they extend, see `ARCHITECTURE.md`.

## Table of contents

1. [Why a cognitive layer at all](#why-a-cognitive-layer)
2. [Parte 22 — Growth: Adapters, Expansion, Sub-motors](#parte-22)
3. [Parte 22.5 — Compositional trajectories](#parte-225)
4. [Parte 23 — Sleep cycle ritual](#parte-23)
5. [Parte 24 — Four-signal pruning](#parte-24)
6. [Parte 25 — Probabilistic reward](#parte-25)
7. [Parte 26 — Hierarchical compression](#parte-26)
8. [Parte 27 — Sparse activation](#parte-27)
9. [How they compose](#how-they-compose)
10. [Integration with training](#integration-with-training)

---

## Why a cognitive layer

The Fase A through E parts build a competent modular model: encoder,
orchestrator, motors, unifier, decoder, plus agent features (tools,
planner, skills, self-check) and training infrastructure. But a model
that ships with fixed weights and fixed knowledge is still
fundamentally an LLM, just one with better routing.

Fase F is the step that turns the architecture into a *cognitive
system*. The distinguishing features are:

- **The system can learn new concepts after deployment without
  retraining.** This is what `growth/` (Parte 22) enables.
- **The system can route queries through multiple specialized motors
  in sequence, not just pick one.** This is what `composition/`
  (Parte 22.5) enables.
- **The system has explicit, scheduled cognitive maintenance cycles
  where it processes what it experienced during wake.** This is what
  `sleep/` (Parte 23) enables.
- **The system can forget selectively based on explicit signals about
  what is worth remembering.** This is what `pruning/` (Parte 24) plus
  `compression/` (Parte 26) enable.
- **The system can rank its own outputs based on user feedback and
  intrinsic confidence.** This is what `reward/` (Parte 25) enables.
- **The system can activate only a fraction of its weights per query,
  reducing compute without losing quality.** This is what `sparse/`
  (Parte 27) enables.

Together these form the cognitive layer. The system is no longer a
frozen function; it is an evolving process.

The lesson motivating all of this is the failure of every
retrieval-augmented generation system to produce genuinely continual
learning. RAG reads documents at inference time but does not modify
the model. Finetuning modifies the model but loses prior knowledge.
AION-C's cognitive layer is the attempt to have both: weights that
learn new things, plus protection against forgetting.

---

## Parte 22

**Growth of the model: Adapters, Expansion, Sub-motors.**

### Three levels of growth

When the system encounters a concept it does not know well, it has
three options, each with different cost and reversibility:

1. **Adapter (LoRA-style).** Cheapest. A small low-rank matrix pair
   (A and B) is added to a handful of Linear layers in one motor.
   Reversible: detach restores the motor bit-a-bit.
2. **Expansion of an existing motor.** Medium cost. Add layers,
   heads, or hidden dimensions to a saturated motor. Function-preserving
   init (net2net) so the expanded motor produces identical output at
   init. Not reversible without keeping the old weights.
3. **Sub-motor (new child motor).** Highest cost. Create a new motor
   dedicated to a structurally distinct domain. Inherits from the
   parent's weights and diverges. Registered in the MoSE config as an
   additional motor.

### 22.1 — Adapters

```python
@dataclass
class LoRAConfig:
    rank: int = 8            # cuello de botella
    alpha: float = 16.0      # factor de escala; delta = (alpha/rank) * B A x
    dropout: float = 0.0
    init_scale: float = 1.0

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, config: LoRAConfig):
        super().__init__()
        self.base = base                                      # referencia al Linear original
        self.lora_A = nn.Parameter(torch.zeros(rank, in_f))   # Kaiming init
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank))  # zero init
        self.enabled = True
        self._scaling = alpha / rank

    def forward(self, x):
        y = self.base(x)
        if not self.enabled or self._scaling == 0:
            return y
        delta = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B)
        return y + self._scaling * delta
```

Key design decisions:

- **Zero init on B.** The first forward produces exactly the base
  output because `B = 0` makes `delta = 0`. This means adding an
  adapter to an untrained model does not break its initial behavior.
- **`base` is a normal submodule.** Earlier drafts hid `base` via
  `object.__setattr__` to keep it out of `state_dict`, but that
  confused the optimizer (base weights were hidden from
  `motor.parameters()` after attach). Solution: base is a normal
  submodule, and `AdapterPack.adapter_state_dict()` filters at save
  time to emit just the LoRA matrices.
- **`enabled` flag allows hot-swapping.** Disabling a pack during
  inference preserves base behavior without detaching.

### AdapterPack

```python
class AdapterPack(nn.Module):
    def __init__(self, concept_name, motor_name, config):
        self.concept_name = concept_name
        self.motor_name = motor_name
        self.config = config
        self.layers = nn.ModuleDict()  # path -> LoRALinear
        self._target_paths = []

    def add_layer(self, path, lora_linear):
        self.layers[_escape(path)] = lora_linear
        self._target_paths.append(path)

    def adapter_state_dict(self):
        # only the LoRA matrices, not the base weights
        return {
            f"{key}.lora_A": layer.lora_A.detach().clone()
            for key, layer in self.layers.items()
        } | {
            f"{key}.lora_B": layer.lora_B.detach().clone()
            for key, layer in self.layers.items()
        }
```

### Attach / Detach

```python
def attach_adapter_pack(motor, pack):
    for path in pack.target_paths():
        parent, attr = _resolve_parent(motor, path)
        current = getattr(parent, attr)
        lora = pack.get(path)
        assert current is lora.base, "motor mutated since pack was built"
        setattr(parent, attr, lora)  # monkey-patch
    pack._attached = True

def detach_adapter_pack(motor, pack):
    for path in pack.target_paths():
        parent, attr = _resolve_parent(motor, path)
        current = getattr(parent, attr)
        assert current is pack.get(path), "detach path changed"
        setattr(parent, attr, pack.get(path).base)  # restore original
    pack._attached = False
```

The bit-a-bit guarantee of detach is the single most important
invariant of the adapter system. It means: no matter what a user
trains into an adapter, detaching that adapter restores the motor to
its pre-training state byte for byte. This is what makes continual
learning safe.

### Registry

```python
class AdapterRegistry:
    def __init__(self, root_dir):
        self.root = Path(root_dir)
        self.adapters_root = self.root / "adapters"

    def save(self, pack, parent_brain_version=None, tags=None):
        # writes adapter.pt + meta.json under adapters/<motor>/<concept>/
        ...

    def load_into(self, pack):
        # loads adapter.pt into an already-constructed pack
        ...

    def list(self, motor_name=None) -> List[AdapterMeta]: ...
    def route_by_query(self, query, motor_name, max_hits=3): ...
    def delete(self, motor_name, concept_name): ...
```

The registry is the persistence layer. Adapters live in
`brain/adapters/<motor>/<concept>/` with `adapter.pt` (the LoRA
matrices) and `meta.json` (metadata). The directory structure is
human-readable and can be inspected with `ls`.

### Growth policy (22.4)

```python
def decide_growth(baseline_accuracy, policy=None, domain_distinct=False,
                  current_adapters_in_motor=0):
    p = policy or GrowthPolicy()

    if baseline_accuracy >= p.adapter_upper:     # >= 0.70
        return GrowthDecision.NO_GROWTH

    if baseline_accuracy >= p.adapter_lower:     # >= 0.30
        if current_adapters_in_motor >= p.max_adapters_per_motor:
            return SUB_MOTOR if domain_distinct else EXPAND_MOTOR
        return GrowthDecision.ADAPTER

    # < 0.30
    return SUB_MOTOR if domain_distinct else EXPAND_MOTOR
```

Thresholds are the ones from the mega-prompt section 22.4. The policy
is deliberately simple; the important thing is that it is explicit
and testable, not fancy.

### Tests and integration

The growth tests cover:

- `LoRALinear.forward()` with zero init matches base output.
- `detach_adapter_pack` restores motor to bit-a-bit identical state.
- Double attach raises, double detach is a no-op.
- Registry save-load round trip is identity.
- Policy decisions match the mega-prompt table.
- Integration: attach adapter to real `CodeMotor` from the 1.1B
  pipeline, mutate, detach, verify base is unchanged.

---

## Parte 22.5

**Compositional trajectories — Orchestrator as director of a flow.**

The core move is elevating the orchestrator from selector to director.

### Before: one motor per query

```
query --> orchestrator --> motor --> output
```

### After: sequence of motors with dependencies

```
query --> planner --> [motor1, motor2, ..., motorN]
                           |        |           |
                           +--------+-----------+
                                    |
                                    v
                                  unifier --> output
```

### Models

```python
@dataclass
class TrajectoryStep:
    motor_name: str
    sub_goal: str
    depends_on: List[int]   # indices of prior steps
    max_tokens: int

@dataclass
class Trajectory:
    query: str
    steps: List[TrajectoryStep]
    rationale: str

    def motor_sequence(self) -> List[str]:
        return [s.motor_name for s in self.steps]
```

A trajectory is a DAG over motors, not just a list. Steps can depend
on any subset of prior steps. The planner validates that all
`depends_on` indices are less than the current step's index (no
forward references).

### Planner

The planner is heuristic in v1. It uses substring matching to detect
which motors apply, plus special patterns for compositional queries:

```python
class TrajectoryPlanner:
    def plan(self, query) -> Trajectory:
        low = query.lower()
        detected = self._detect_motors(low)  # by hints
        transform = self._detect_transform_as(low)  # "como cuento"

        if transform is not None:
            # "explica este código como cuento" -> [forge_c, muse]
            base = self._primary_technical(detected, low)
            return Trajectory(
                query=query,
                steps=[
                    TrajectoryStep(base, f"analyze: {query}"),
                    TrajectoryStep(transform, f"reexpress as story",
                                   depends_on=[0]),
                ],
                rationale=f"transform-as -> {base} -> {transform}",
            )

        if self._is_compositional(low) and len(detected) >= 2:
            # "python y rust, ¿por qué?" -> [forge_c, cora] with cora unifier
            steps = [TrajectoryStep(m, f"answer part {m}", []) for m in detected[:-1]]
            steps.append(TrajectoryStep("cora", "compare prior outputs",
                                        depends_on=list(range(len(steps)))))
            return Trajectory(query, steps, ...)

        # Default: single motor
        return Trajectory(query, [TrajectoryStep(detected[0] or "cora", ...)])
```

The v1 planner covers the emblematic cases from the spec:

- `"explica este código como cuento"` → `[forge_c, muse]`
- `"explica python y rust, por qué son diferentes"` → `[forge_c, cora]`
- `"me siento triste"` → `[empathy]`
- `"calcula 15% de 240"` → `[axiom]`

A future planner can be neural, learning a policy from examples. The
interface stays the same.

### CompositeOrchestrator

```python
class CompositeOrchestrator:
    def __init__(self, generate_fn):
        self.generate_fn = generate_fn  # (motor, prompt, max_tokens) -> str

    def execute(self, trajectory) -> TrajectoryResult:
        step_results = []
        for i, step in enumerate(trajectory.steps):
            prior_outputs = [step_results[d].output for d in step.depends_on]
            prompt = self._build_prompt(trajectory.query, step.sub_goal, prior_outputs)
            output = self.generate_fn(step.motor_name, prompt, step.max_tokens)
            step_results.append(StepResult(i, step.motor_name, ..., output))
        fused = TrajectoryUnifier().fuse(trajectory, step_results)
        return TrajectoryResult(trajectory, step_results, fused, ...)
```

The `generate_fn` is injectable so tests can use a stub
(returning deterministic text) while production uses the real pipeline.

### Trajectory Unifier

```python
class TrajectoryUnifier:
    def fuse(self, trajectory, step_results) -> str:
        if len(step_results) == 1:
            return step_results[0].output.strip()
        # The last step is the "leader" — its output is the final answer
        return step_results[-1].output.strip()
```

The v1 unifier is simple: the last step drives the answer, prior
steps are intermediate reasoning. Future versions will detect and
resolve contradictions across steps.

### Backend integration

The backend has `/api/trajectory/plan` (plan only, no execute) and
`/api/trajectory/execute` (plan and run). The UI shows the plan in
the right panel as a sequence of motor badges connected by arrows to
a `unifier` badge at the end.

---

## Parte 23

**Sleep cycle — ritual of six questions, inspired by sleep memory
consolidation.**

### The six phases, in order

| # | Phase | Question | Hook |
|---|-------|----------|------|
| 1 | recollect | ¿Qué viví desde el último sueño? | None (stub counts episodes) |
| 2 | score | ¿Qué fue útil y qué no? | `reward_hook` (Parte 25) |
| 3 | prune | ¿Qué debo olvidar? | `prune_hook` (Parte 24) |
| 4 | compress | ¿Qué debo comprimir? | `compress_hook` (Parte 26) |
| 5 | consolidate | ¿Qué debo consolidar en los pesos? | `consolidate_hook` (Parte 9 auto-learn) |
| 6 | followups | ¿Qué debo preguntarme mañana? | `followups_hook` (goals) |

The order is strict. Phases 2-6 depend on the output of prior phases
(phase 2 scores all episodes from phase 1, phase 3 uses those scores
to prune, etc.). The names are hard-coded in `SLEEP_QUESTIONS` tuple
and cannot be reordered without changing the code.

### EpisodicBuffer

The input to the cycle. An episode is one turn of the user's
interaction:

```python
@dataclass
class Episode:
    user_text: str
    aion_response: str
    timestamp: float
    motor_sequence: List[str]
    user_feedback: Optional[str]
    implicit_score: float
    meta: Dict[str, Any]

class EpisodicBuffer:
    def add(self, episode): ...
    def drain(self) -> List[Episode]: ...  # empties
    def snapshot(self) -> List[Episode]: ...  # read-only
```

FIFO with a max size; when full, the oldest episode is dropped.

### SleepCycle.run()

```python
def run(self, trigger="manual") -> SleepCycleLog:
    started = time.time()
    episodes = self.buffer.drain()
    phases = []
    prev = {}
    try:
        for name, question in SLEEP_QUESTIONS:
            result = self._run_phase(name, question, episodes, prev)
            phases.append(result)
            prev[name] = result.data
    except Exception as exc:
        error = str(exc)
    return SleepCycleLog(started, time.time(), trigger, len(episodes), phases, error)
```

Key invariant: six phases always run (or the log carries the error).

### SleepDaemon

```python
@dataclass
class SleepDaemon:
    cycle: SleepCycle
    inactivity_seconds: float = 1800.0
    overflow_threshold: int = 500
    _last_activity_ts: float = 0.0
    _last_log: Optional[SleepCycleLog] = None

    def notify_activity(self, ts=None): ...
    def should_run(self, now=None) -> Optional[SleepTrigger]: ...
    def maybe_run(self, now=None) -> Optional[SleepCycleLog]: ...
    def force_run(self) -> SleepCycleLog: ...
```

Three triggers:

- **MANUAL** — explicit `force_run()` call, e.g. from `/api/sleep`.
- **INACTIVITY** — `now - last_activity_ts >= inactivity_seconds` and
  the buffer has episodes to process.
- **OVERFLOW** — buffer size reached `overflow_threshold` regardless
  of activity.

The backend runs the daemon as an asyncio background task:

```python
async def _sleep_daemon_loop(state):
    while True:
        await asyncio.sleep(state.sleep_poll_interval)
        log = state.sleep_daemon.maybe_run()
        if log is not None:
            state.save_hierarchy()
            state.save_reward_ledger()
```

The poll interval is 300 seconds (5 minutes) by default. Every 5
minutes, the daemon checks whether a cycle should run, and if so,
runs it synchronously. This is a cooperative design: if the cycle is
long, the chat handler will see the daemon state and wait.

---

## Parte 24

**Four-signal pruning with dynamic TTL.**

### The four signals

| Signal | Source | Interpretation |
|--------|--------|----------------|
| S1 frequency | access count | how often the item was retrieved |
| S2 recency | time since last access | decayed exponentially with configurable half-life |
| S3 utility | reward aggregated | from the Parte 25 ledger |
| S4 retrieval cost | tokens, graph hops, time | expensive items get protected |

### Normalization

```python
def normalize(self, max_freq, max_cost, half_life_sec):
    s1 = min(self.frequency / max(max_freq, 1e-9), 1.0)
    s2 = math.exp(-self.last_access_age * math.log(2) / half_life_sec)
    s3 = max(0.0, min(1.0, self.utility))
    s4 = min(self.retrieval_cost / max(max_cost, 1e-9), 1.0)
    return s1, s2, s3, s4
```

All four are in `[0, 1]` after normalization. Frequency and cost are
normalized by the batch-wise maximum (so scoring is relative to the
items currently being pruned). Recency uses an absolute exponential
decay with a half-life defaulting to one week.

### Retention score

```python
retain_score = (w1*s1 + w2*s2 + w3*s3 + w4*s4) / sum_of_weights
```

Default weights: `w1=0.25, w2=0.25, w3=0.35, w4=0.15`. Utility gets
the highest weight because it is the closest proxy for "worth
keeping". Cost gets the lowest weight but is nonzero because items
that are expensive to reconstruct deserve protection.

### Four actions

```python
def _action_for(self, score):
    if score < self.delete_threshold:        # < 0.15
        return PruneAction.DELETE
    if score < self.compress_threshold:      # < 0.35
        return PruneAction.COMPRESS
    if score >= self.promote_threshold:      # >= 0.80
        return PruneAction.PROMOTE
    return PruneAction.KEEP
```

`PROMOTE` means move to a faster cache layer (hot memory). `KEEP`
means leave in place with an updated TTL. `COMPRESS` means hand off
to the Parte 26 compressor before eventual deletion. `DELETE` means
gone.

### Dynamic TTL

```python
def _ttl_for(self, score):
    clamped = max(0.0, min(1.0, score))
    return self.ttl_min_sec + clamped * (self.ttl_max_sec - self.ttl_min_sec)
```

Items with a high retention score get long TTLs (up to 30 days by
default), items with low scores get short ones (as low as 1 hour).
The TTL is recomputed on every sleep cycle, so an item that is
suddenly used a lot after being dormant will have its TTL restored.

### Stress test result

The stress test generates 1000 synthetic items with three populations:
10% "hot" (high frequency, fresh, high utility), 20% "medium", 70%
"cold" (old, never used, low utility). The pruner correctly classifies
them: most cold items go to DELETE, most hot items go to PROMOTE, and
the medium items go to KEEP or COMPRESS. Specifically, on 1000 items
the stress test asserts `deleted >= 500` and `50 <= promoted <= 200`,
and both pass consistently with random seeds.

---

## Parte 25

**Probabilistic reward combining three sources of signal.**

### The formula

```
reward = alpha*R_explicit + beta*R_implicit + gamma*R_intrinsic
```

Default weights: `alpha=0.55, beta=0.30, gamma=0.15`. Explicit signals
are the most reliable but the rarest (users rarely click thumbs
buttons). Implicit signals are plentiful but noisy. Intrinsic signals
are a regularizer.

### Explicit signals

```python
class ExplicitSignal(str, Enum):
    UP = "up"                # thumbs up
    DOWN = "down"            # thumbs down
    CORRECTION = "correction"  # user explicitly corrected
    NONE = "none"            # no explicit signal
```

Each signal maps to a mean and standard deviation:

```python
def _explicit_value(signal):
    if signal == UP:         return 1.0, 0.05
    if signal == DOWN:       return 0.0, 0.05
    if signal == CORRECTION: return 0.1, 0.10
    return 0.5, 0.35  # NONE - high variance, neutral mean
```

### Implicit signals

```python
@dataclass
class ImplicitSignals:
    no_correction_continue: bool   # user continued without correcting
    thanks: bool                   # user said "gracias"/"thanks"
    re_asked_similar: bool         # user re-asked a similar question
    code_copied: bool              # user appears to have copied the code
    abandoned: bool                # user left without replying
```

The key design point from the mega-prompt: `no_correction_continue`
weighs MORE than `thanks`. The rationale is that users often say
"thanks" out of politeness even when the answer was mediocre, but
they only continue the conversation naturally when the answer was
actually useful.

```python
def _implicit_score(self, s):
    score = 0.5  # neutral baseline
    if s.no_correction_continue: score += 0.40
    if s.thanks:                 score += 0.15
    if s.re_asked_similar:       score -= 0.35  # strong negative
    if s.code_copied:            score += 0.30
    if s.abandoned:              score -= 0.10
    return clamp(score, 0, 1)
```

### Intrinsic signals

```python
@dataclass
class IntrinsicSignals:
    token_entropy_mean: float       # lower is better
    symbolic_consistent: bool       # from symbolic engine
    unifier_agreement: float        # [0, 1] from cross-motor consensus

def to_mean_std(self):
    ent_scaled = max(0, min(1, 1 - self.token_entropy_mean / math.log(5)))
    sym = 1.0 if self.symbolic_consistent else 0.0
    ua = clamp(self.unifier_agreement, 0, 1)
    mean = 0.35*ent_scaled + 0.4*sym + 0.25*ua
    return mean, 0.15
```

These are zero-cost signals available at every inference without any
user interaction. They act as a regularizer that catches obviously
wrong outputs (contradict symbolic rules, high entropy, motors
disagree).

### ImplicitDetector

```python
class ImplicitDetector:
    def detect(self, assistant_response, next_user_text, previous_user_text,
               time_to_next_turn_sec=None) -> ImplicitSignals:
        if next_user_text is None:
            abandoned = (time_to_next_turn_sec is None or
                         time_to_next_turn_sec > self.abandon_threshold_sec)
            return ImplicitSignals(abandoned=abandoned)

        nxt_low = next_user_text.lower()
        thanks = bool(_THANKS_RE.search(nxt_low))
        re_asked = self._similarity(previous_user_text, next_user_text) >= 0.6
        correction_markers = ("no,", "incorrect", "equivocado", "mal", "wrong", "actually")
        has_correction = any(m in nxt_low for m in correction_markers)
        no_correction = not has_correction and not re_asked
        has_code = bool(_CODE_BLOCK_RE.search(assistant_response))
        code_copied = has_code and not has_correction
        return ImplicitSignals(no_correction, thanks, re_asked, code_copied, False)
```

The detector is heuristic, not learned. It is fast, predictable, and
good enough for a v1 signal that feeds the reward formula.

### RewardLedger

Persistent accumulator per key. Key can be a motor name (`"forge_c"`)
or a motor-adapter tuple (`"forge_c:python_typing"`). The ledger
stores `sum_mean`, `sum_var`, `n`, and computes running mean and
variance.

```python
class RewardLedger:
    def add(self, key, estimate): ...
    def mean_for(self, key) -> float: ...
    def count_for(self, key) -> int: ...
    def snapshot(self) -> Dict[str, Dict[str, float]]: ...
    def save_jsonl(self, path): ...
    def load_jsonl(self, path): ...
```

Persistence is JSONL with one line per key. On boot the backend
loads `brain/v1/reward_ledger.jsonl` if it exists, rebuilding the
ledger from disk. Every `add()` in an endpoint handler triggers a
`save_jsonl()` so the ledger survives crashes.

### UI integration

Thumbs up and thumbs down buttons appear on every assistant message.
Clicking sends `POST /api/feedback` with `{vote, motor, adapter,
episode_offset}`. The handler creates a `RewardSignals`, computes the
estimate, updates the ledger, tags the episode in the buffer with
`user_feedback`, and (if the motor and adapter are specified)
updates the adapter meta in the registry with an exponential moving
average of the reward.

---

## Parte 26

**Three-level hierarchical compression of memory.**

### The three levels

```
EPISODIC (raw)
  ↓ cluster
STABLE (clustered, anchors preserved)
  ↓ abstract
NUCLEAR (concept, metadata only)
```

Each level has a different TTL and a different purpose:

- **EPISODIC** items are individual conversations or facts, stored
  verbatim. Short TTL (hours to days). Everything starts here.
- **STABLE** items are clusters of similar episodic items, with a
  summary text and 1-2 preserved anchor episodes. Medium TTL (days
  to weeks).
- **NUCLEAR** items are abstract concepts that emerged from stable
  clusters, with pointers back to their anchors. Long TTL (weeks to
  permanent).

### StoredItem

```python
@dataclass
class StoredItem:
    id: str
    text: str
    level: MemoryLevel
    usage_count: int
    created_at: float
    parent_id: Optional[str]     # for NUCLEAR, points to source STABLE
    anchor_ids: List[str]        # for STABLE/NUCLEAR, preserved raw episodes
    meta: Dict[str, Any]
```

Anchor preservation is crucial: when a user challenges a nuclear
concept (e.g. "wait, that's not actually what I said"), the system
can follow the anchor pointers back to the raw episodic items and
re-examine the evidence.

### Clusterer

Greedy Jaccard clustering with no dependencies:

```python
class Clusterer:
    def cluster(self, items) -> List[Cluster]:
        unassigned = list(items)
        out = []
        while unassigned:
            seed = unassigned.pop(0)
            members = [seed]
            remaining = []
            for other in unassigned:
                if self.similarity_fn(seed.text, other.text) >= self.threshold:
                    members.append(other)
                else:
                    remaining.append(other)
            unassigned = remaining
            if len(members) >= self.min_size:
                out.append(Cluster(
                    member_ids=[m.id for m in members],
                    anchor_ids=self._pick_anchors(members),
                    summary=self._summarize(members),
                    similarity=avg_sim,
                ))
        return out
```

The threshold defaults to 0.4 for general use and 0.3 for the sleep
cycle integration (looser clustering at sleep time because the goal
is to find any family resemblance).

### HierarchicalCompressor

```python
class HierarchicalCompressor:
    def __init__(self, store, clusterer=None, nuclear_usage_threshold=5):
        ...

    def compress_episodic_to_stable(self) -> List[StoredItem]:
        episodic = self.store.list_by_level(EPISODIC)
        clusters = self.clusterer.cluster(episodic)
        created = []
        for c in clusters:
            item = StoredItem(
                id=self.store.new_id("stable"),
                text=c.summary,
                level=STABLE,
                anchor_ids=c.anchor_ids,
                meta={"member_ids": c.member_ids, "similarity": c.similarity},
            )
            self.store.add(item)
            created.append(item)
        return created

    def promote_stable_to_nuclear(self) -> List[StoredItem]:
        stable_items = self.store.list_by_level(STABLE)
        promoted = []
        for item in stable_items:
            if item.usage_count >= self.nuclear_usage_threshold:
                nuclear = StoredItem(
                    id=self.store.new_id("nuclear"),
                    text=f"concept: {item.text}",
                    level=NUCLEAR,
                    parent_id=item.id,
                    anchor_ids=item.anchor_ids,
                    usage_count=item.usage_count,
                )
                self.store.add(nuclear)
                promoted.append(nuclear)
        return promoted
```

The flow:

1. Episodic items enter via `ingest_episodes()` from the sleep cycle.
2. `compress_episodic_to_stable()` clusters them and creates stable
   items. Originals are NOT deleted; pruning (Parte 24) handles that.
3. `promote_stable_to_nuclear()` runs on each sleep cycle. Stable
   items with enough usage are promoted to nuclear.

### Persistence

`HierarchicalStore` has `save_jsonl()` and `load_jsonl()` with a
metadata line that preserves the `_counter` so `new_id()` continues
producing unique IDs after reload:

```python
def save_jsonl(self, path):
    lines = [json.dumps({"_meta": {"counter": self._counter}})]
    for item in self._items.values():
        lines.append(json.dumps(item.to_dict()))
    Path(path).write_text("\n".join(lines), encoding="utf-8")

def load_jsonl(self, path):
    self._items.clear()
    self._counter = 0
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        if "_meta" in d:
            self._counter = d["_meta"]["counter"]
            continue
        d["level"] = MemoryLevel(d["level"])
        self._items[d["id"]] = StoredItem(**d)
```

---

## Parte 27

**Conditional computation via sparse activation gates.**

### The idea in one sentence

A small gating network (~1% of the size of each motor) produces a
per-query activation mask that controls which neurons in the main
Linear layers are active for that query. The gates are trained from
step 0 so the motors learn to be robust to masking.

### SparseConfig

```python
@dataclass
class SparseConfig:
    target_density: float = 0.15      # fraction active neurons target
    mode: str = "continuous"          # or "binary"
    gate_hidden: int = 16             # hidden dim of the gate MLP
    temperature: float = 1.0          # sigmoid temperature
    threshold: float = 0.5            # binarization cutoff
```

Target density 0.15 means we want about 15% of neurons active per
query. This is an aggressive sparsity target. During training we often
start with target 0.5 (permissive) and anneal toward 0.15.

### GateNetwork

```python
class GateNetwork(nn.Module):
    def __init__(self, in_features, out_features, config):
        self.fc1 = nn.Linear(in_features, config.gate_hidden)
        self.fc2 = nn.Linear(config.gate_hidden, out_features)
        # Warm start: bias initialized so sigmoid(0) approximates target
        with torch.no_grad():
            offset = math.log(target / (1 - target))  # logit of target
            self.fc2.bias.fill_(offset)

    def forward(self, x):
        # Reduce to [B, in_features] by averaging over seq dim
        reduced = x
        while reduced.dim() > 2:
            reduced = reduced.mean(dim=1)
        h = F.relu(self.fc1(reduced))
        logits = self.fc2(h) / self.config.temperature
        probs = torch.sigmoid(logits)
        if self.config.mode == "continuous":
            return probs
        # Binary mode: straight-through estimator
        hard = (probs >= 0.5).float()
        return hard.detach() + probs - probs.detach()
```

The warm-start of the bias is critical. Without it, the gate
initializes with `sigmoid(0) = 0.5` meaning 50% density, and the
sparsity loss has to spend thousands of steps pulling it down to
target. With the warm-start, the first forward produces density close
to target, and the loss can focus on shaping the distribution rather
than adjusting the mean.

### SparseLinear

```python
class SparseLinear(nn.Module):
    def __init__(self, base: nn.Linear, config: SparseConfig):
        self.base = base
        self.gate = GateNetwork(base.in_features, base.out_features, config)
        self.enabled = True
        self._last_density = None

    def forward(self, x):
        y = self.base(x)
        if not self.enabled:
            return y
        mask = self.gate(x)              # [B, out]
        # Broadcast mask to match output shape
        while mask.dim() < y.dim():
            mask = mask.unsqueeze(-2)
        # Record actual density for metrics
        with torch.no_grad():
            active = (mask >= self.config.threshold).float().mean()
            self._last_density = active.item()
        return y * mask
```

The wrapper is transparent when `enabled=False`. Its parameters
include both the base and the gate, so `motor.parameters()` correctly
yields all trainable tensors.

### SparsityTracker

```python
class SparsityTracker:
    def __init__(self, root):
        self._layers = [(name, m) for name, m in root.named_modules()
                        if isinstance(m, SparseLinear)]

    def collect(self) -> Dict[str, Any]:
        per_layer = {name: m.last_density for name, m in self._layers}
        avg = mean(per_layer.values()) if per_layer else 0.0
        return {"per_layer": per_layer, "avg_density": avg,
                "active_percent": round(avg*100, 1),
                "layer_count": len(self._layers)}
```

### sparsity_loss

```python
def sparsity_loss(root, target, reduction="mean"):
    sparses = [m for m in root.modules() if isinstance(m, SparseLinear)]
    losses = []
    for s in sparses:
        if s._last_density is None:
            continue
        d = torch.tensor(s._last_density)
        losses.append((d - target) ** 2)
    if not losses:
        return torch.zeros(())
    return torch.stack(losses).mean() if reduction == "mean" else torch.stack(losses).sum()
```

This is the penalty that pushes the gate's observed density toward
the target. During training, this loss is added to the standard
language modeling loss:

```python
loss = lm_loss + routing_w*route_loss + balance_w*balance_loss + sparsity_w*sparsity_loss
```

The weight `sparsity_w` defaults to 0.1; increasing it makes the
gates converge to target density faster at the cost of some quality.

### Compatibility with adapters

The gate is applied AFTER the base Linear. If the base is actually a
`LoRALinear` (adapter attached), the gate sees `base_out + delta`.
This means a new adapter can "push" neurons above the threshold by
adding to their activation, even if the base would have gated them
off. Conversely, the gate can mask adapter contributions if it
decides they are not relevant for this query.

The mega-prompt specifies that the gate should act on the sum rather
than on each component separately, because the user might want
adapter contributions to be visible to the gate's decision. The
current implementation does this via the wrapper-based design.

### Smoke test on real CodeMotor

The tests verify:

- Attach gates to 5 Linear layers in a real `CodeMotor`
- Run forward with random input, record density
- Detach, verify base weights are unchanged byte-for-byte
- Measure `avg_density` is within 10% of target

---

## How they compose

The six parts are not independent. Here are the interactions:

### Adapters + Sparse

Adapters add to the Linear's output; gates mask the combined output.
An adapter can rescue neurons that the gate would have masked by
pushing them above the threshold. This is tested: attach a sparse
gate and a LoRA adapter on the same Linear, verify the forward is
stable.

### Adapters + Reward

When feedback arrives on a response generated by a specific adapter,
the reward updates both the RewardLedger for the key
`motor:adapter` and the adapter's `reward_score` in its meta via
an exponential moving average. Adapters with high reward get
promoted during sleep; adapters with low reward get pruned.

### Sleep + Pruning + Compression + Reward

The sleep cycle is the orchestration layer for the memory
maintenance parts:

- Phase 2 (score) uses the reward estimator as its hook.
- Phase 3 (prune) uses the pruner as its hook.
- Phase 4 (compress) uses the hierarchical compressor as its hook.
- Phases 5 (consolidate) and 6 (followups) are still stubs because
  they depend on auto-learn plus goals, which exist but are not
  fully wired.

A single sleep cycle processes every episode through all six
phases in sequence, with prior phase outputs available to later
phases.

### Trajectories + Reward

Each step in a trajectory produces an output that can receive
feedback. The UI attaches thumbs buttons to the final response,
but the reward goes to the motor sequence as a whole, not just the
last motor. This is still a v1 design; a future version will
distribute reward across the steps.

### Adapters + Compression

When an adapter's concept is superseded by a nuclear concept (say,
the "python_typing" adapter is folded into a broader
"static_type_systems" nuclear concept), the old adapter can be
deleted once the nuclear concept is well-established. This is a
manual decision today; it can be automated later.

---

## Integration with training

The integration of Fase F into the training loop is critical: the
cognitive layer must be exposed during training, not glued on afterward.
The lesson from the identity skill OOD is that features the model never
sees during training become broken at inference.

The `train_1b_canonical.py` script does this:

```python
def train(..., fase_f=True, sparsity_w=0.1, sparsity_target=0.5):
    pipeline, cfg = build_pipeline(config, vocab_size)

    if fase_f:
        per_motor_sparse, sparsity_tracker = attach_fase_f_to_pipeline(
            pipeline,
            sparsity_target=sparsity_target,
            gate_hidden=8,
            max_targets_per_motor=6,
        )

    for step in range(start_step + 1, n_steps + 1):
        for _ in range(grad_accum):
            out = pipeline(ids_t)
            lm = F.cross_entropy(out.logits[0, :-1], ids_t[0, 1:], ignore_index=0)
            # routing loss, balance loss as before

            sp = torch.zeros(())
            if fase_f:
                sp = sparsity_loss(pipeline, target=sparsity_target)

            loss = lm + routing_w*rl + balance_w*bl + sparsity_w*sp

            loss.backward()

    # At the end: smoke test that adapters still work on this trained model
    scaffold_report = verify_adapter_scaffolding(pipeline, device)
    # scaffold_report[motor_name] = {"ok": True, "n_targets": 6}
```

The `verify_adapter_scaffolding` function at the end runs an
attach/mutate/detach cycle on every motor and asserts that the base
weights are unchanged byte for byte. If this test fails, the training
is considered broken. It is a sanity check that the combination of
sparse gates and the training dynamics did not corrupt the adapter
system.

### Observations from the 50-step dry-run

On the tiny config with Fase F enabled:

- Base params: 5,555,449
- Params after gates: 5,578,849 (+23,400 gate params, 0.42% overhead)
- Initial density: 0.59 (warm-start from target 0.5)
- Observed density range over 50 steps: 0.57 - 0.60
- `sparsity_loss` stable around 0.008
- `lm_loss` decreasing from 10.388 to 10.323
- Final scaffolding verification: all 5 motors passed

This confirms the integration works. The 1.1B training will use the
same setup.

---

The cognitive layer is the most complex part of AION-C. It is also
what distinguishes it from a plain modular transformer. If you understand
this document, you understand what AION-C is betting on: that an
evolving, compositional, selectively-forgetting cognitive system can
match a frozen dense LLM while being smaller, cheaper, and more
honest about its own limitations.
