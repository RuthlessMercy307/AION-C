# AION-C — Investor Brief

This document is the non-technical version of the project. It exists
for people who want to understand the bet, the market, the progress,
and the ask, without reading code. For technical depth, see the main
`README.md` and the other files in `docs/`.

## Table of contents

1. [The one-paragraph pitch](#pitch)
2. [The problem with today's AI](#problem)
3. [The insight](#insight)
4. [What AION-C is](#what-it-is)
5. [What is working today](#working)
6. [The market](#market)
7. [Competitive landscape](#competitive)
8. [Financial position](#financial)
9. [The ask](#ask)
10. [Risks and how they are managed](#risks)
11. [Milestones for the next six months](#milestones)

---

## Pitch

AION-C is a Causal Reasoning Model — not a Large Language Model —
built around the thesis that the next generation of intelligent
systems will be modular, continually learning, and cheap to run,
rather than monolithic and expensive. The project has built and
validated a complete working prototype at small scale, with 2856
passing tests, 38 packages, and five scientific experiments that
confirm the architectural design. The next step is training a 1.1
billion parameter version for approximately $4 of cloud compute,
which the team is ready to execute this week.

---

## Problem

The dominant approach to AI today is to train a single very large
neural network on a large corpus of data, then freeze that network
and deploy it. Examples include GPT-4, Claude, Gemini, Llama, and
Mistral. These models are excellent at many tasks but have three
structural problems that cannot be solved by making them larger:

### 1. They cannot learn new things after deployment

If a model was trained in January and something new happens in
February, the only way to teach the model about February is to
retrain it from scratch. This is expensive (millions of dollars
per retrain) and risky (the retrained model can lose prior
capabilities).

The common workaround is retrieval-augmented generation (RAG),
where the model reads a database at inference time. RAG helps but
does not update the model's internal knowledge; the next query
has to look up the same information again.

### 2. They cannot forget selectively

If a model learned something wrong (a factual error, a biased
association, outdated information), there is no clean way to
remove just that one thing without retraining everything.

### 3. They are expensive to run

A 175 billion parameter model activates all 175 billion parameters
to answer the simplest question. The compute cost per query is
flat, independent of how easy or hard the question is.

The combined effect is that deploying a large model is expensive,
updating it is slow and risky, and running it is wasteful. For many
real-world applications — especially in regulated domains like
medicine, finance, and law where correctness matters and things
change — this is a poor fit.

---

## Insight

The bet behind AION-C is that these three problems are NOT
fundamental limits of AI. They are consequences of a specific
architectural choice (a single monolithic frozen model). A
different architecture can address all three.

Specifically:

- **Instead of one giant network, use many small networks (motors)
  that specialize in different kinds of reasoning.** A router picks
  which motor handles each query. Only the relevant motor runs.
  This is the Mixture of Experts idea, refined with explicit
  specialization rather than learned routing.

- **Instead of baking knowledge into weights, use an external
  memory.** The weights learn how to reason; the memory stores what
  is known. Updating a fact means writing to a database, not
  retraining the model.

- **Instead of freezing the model after training, support continual
  learning via small per-concept adapters.** When a user teaches
  the system something new, a small adapter (about 1% the size of
  a motor) is created in minutes of compute. The base model does
  not change; the adapter is additive and reversible. If training
  the adapter corrupts behavior, the system automatically rolls
  back.

- **Instead of running the full model on every query, activate only
  the parts that are needed.** Sparse gates produce per-query masks
  that turn off neurons that are not relevant. Target: 15% of
  weights active per query.

- **Instead of treating reasoning as implicit pattern matching,
  build explicit causal graphs.** Each motor builds a graph of the
  entities, relations, and causal chains in the query, refines it
  via message passing, and feeds the refined graph to a decoder.
  This makes reasoning inspectable.

These ideas are not new individually. Mixture of Experts is in
Mixtral and DeepSeek. LoRA adapters are standard. External memory
is an old research topic. Sparse activation is Google's sparse
expert work. Causal graphs have roots in neural theorem proving.

**What AION-C adds is putting all of them together in a single
coherent system that runs end to end, with explicit support for
continual learning, and demonstrating that they compose
productively even at small scale.**

This is not a paper demo. It is a working system with tests, a
UI, a training pipeline, and empirical results.

---

## What it is

AION-C consists of:

- **Five specialized reasoning engines** (motors): one for causal
  reasoning, one for code, one for math, one for creative writing,
  one for social and emotional content.
- **A router** (orchestrator) that picks which motor handles each
  query based on the query's content.
- **A shared backbone** (encoder, unifier, decoder) that all
  motors use for input processing and output generation.
- **An external memory** with embedding-based search, user
  profiles, conversation history, and response caching.
- **Six action tools** (write file, edit file, run code, call API,
  search memory, store memory) that the model can invoke via JSON
  in its output.
- **A planner** that breaks complex requests into steps with
  retries and replanning.
- **A skills system** with 11 markdown files of domain-specific
  guidance injected into the prompt when relevant.
- **A world model scratch pad** that shows intermediate reasoning
  steps for inspection.
- **A symbolic verification engine** with 11 rules (transitivity,
  contradiction detection, type checking, etc.) that can override
  the neural motors' output.
- **A complete cognitive layer** (the Fase F extension) with
  adapters for continual learning, compositional trajectories for
  multi-motor queries, a six-phase sleep cycle, four-signal
  pruning, probabilistic reward, hierarchical memory compression,
  and sparse activation.
- **A full web UI** (React + FastAPI + WebSocket) showing live
  routing, causal graphs, memory state, tool execution, and all
  the cognitive layer components.

All of this is implemented, tested (2856 passing tests), and
running today at small scale (5.5 million parameters). The next
step is scaling to 1.1 billion parameters using the packaged
training pipeline.

---

## Working

This is the honest list of what exists and runs today.

### Architecture

- 38 top-level packages, each with its own tests.
- 67 test files, 2856 tests passing, zero regressions introduced
  by any phase of the project.
- Five specialized motors with explicit causal graph reasoning,
  a router for selecting motors, a unifier for combining their
  outputs.
- Tools, planner, skills, self-check, memory, lifecycle manager,
  goals manager, all wired into the backend.

### Training

- A 5.5 million parameter tiny model trained on CPU in 44 minutes,
  achieving 98.2% routing accuracy and passing 7/7 end-to-end
  checks on the canonical evaluation set.
- A training pipeline for the 1.1 billion parameter production
  model, validated with a 50-step dry run that includes the
  cognitive layer active from step zero.
- A 72,500 example canonical dataset across 7 domains (code, math,
  causal, creative, emotional, general, metacognitive) with
  balanced English and Spanish coverage.

### Cognitive layer

Five experiments validate the cognitive architecture:

- **Sequential learning:** 100 concepts learned one after another,
  with the original 10-item exam still passing at exactly 100%
  after each step. This is a bit-for-bit mathematical guarantee,
  not a probabilistic one.
- **Cross-domain non-interference:** 15 adapters trained across
  5 motors, with each motor's exam still passing at 100% after
  all training is done. Zero contamination across motor
  boundaries.
- **Stress test:** The six-phase sleep cycle processes 1000
  synthetic episodes in 36 milliseconds. No crashes, no dropped
  data, all phases complete.
- **Compositional queries:** Four multi-domain queries (including
  "explain this code as a story", which should trigger
  code-motor → creative-motor in sequence) all planned correctly
  and executed without crashes against the real tiny pipeline.
- **Self-evaluation:** Eight scenarios spanning explicit feedback,
  implicit signals, and intrinsic confidence signals, with 6 out
  of 8 classified correctly at the 75% target threshold.

Five out of five experiments pass on both synthetic data and the
real tiny pipeline.

### Backend and UI

- A full FastAPI + WebSocket backend with 25+ endpoints.
- A React UI (served from a CDN with no build step) that shows
  chat, routing scores, memory entries, tool logs, goals, cache
  statistics, lifecycle state, adapter list, trajectory plan,
  sleep cycle log, feedback buttons, world model scratch pad,
  and the per-message causal graph.

### Deployment

- A 27.2 MB deployment zip (`aion_c_vast.zip`) ready to upload to
  a cloud GPU instance. Contains all 307 source files plus the
  dataset and configuration.
- A 5-command deployment procedure documented in `VAST_README.md`
  that takes the project from zip file to running 1.1 billion
  parameter training in about 5 minutes of setup.

---

## Market

This section is more speculative than the "Working" section. It
is the honest attempt to estimate where AION-C could fit
commercially, not a forecast.

### Why continual learning matters

The global AI market is dominated today by subscription access to
frontier models (ChatGPT, Claude, Gemini) and by fine-tuned
derivatives of open models (Llama, Mistral). Both approaches have
the same core problem: the model is frozen, and updating it is
expensive and risky.

For enterprises, this creates a structural mismatch. A company
deploying an AI assistant for customer support has to keep it up
to date with product changes, new FAQs, and evolving policies.
Today this is handled via RAG (reading documents at inference
time) which is slow and error-prone, or by periodic full
retraining which is expensive.

AION-C's continual learning mechanism is the first viable
alternative. A customer support deployment would ingest new
policies and product updates as LoRA adapters, with the
automatic rollback protection ensuring that updates cannot break
prior behavior. The cost per concept is a few minutes of compute;
the cost of a full retrain is days of engineering plus thousands
of dollars.

### Why interpretability matters

Regulated industries (healthcare, finance, law) face a growing
problem: their AI systems must be auditable for compliance
reasons, but dense LLMs are opaque. The EU AI Act, the FDA's
guidance on AI in medical devices, and the SEC's requirements
for financial AI all trend toward mandating explainability.

AION-C's explicit causal graphs, routing scores, and sleep cycle
logs are the kind of structured, inspectable output that
compliance teams need. A deployment in a regulated industry
could use AION-C not because it is the most powerful model
available but because it is the most auditable.

### Why cost matters

A 175 billion parameter model costs approximately $10 per million
tokens to run at retail API prices. A company running an
assistant that serves 10 million tokens per day is paying
$3,000 per day, $90,000 per month.

AION-C at 1.1 billion parameters has a per-token cost an order of
magnitude lower, with the additional factor of only activating
about 30% of parameters per query via sparse gates. A
back-of-envelope estimate puts the per-token cost at about 1% of
a frontier model, roughly $30 per day or $900 per month for the
same 10 million tokens.

This is the "quality at a fraction of the cost" pitch that
makes customers pay attention.

### Target verticals

Speculatively, the best-fit verticals for AION-C are:

1. **Customer support** (continual learning of product updates
   without retraining).
2. **Internal knowledge management** (interpretable reasoning
   over proprietary documents).
3. **Regulated industries** (audit trails and explicit
   reasoning).
4. **Edge deployment** (low per-query cost for on-device
   assistants).
5. **Research tools** (interpretability for scientific use cases
   where understanding the model's reasoning matters).

These are hypotheses, not validated markets. The first validation
will come from a post-Fase E pilot with a real user.

---

## Competitive

The honest list of who AION-C is competing with and how.

### Frontier models (GPT-4, Claude Opus, Gemini Pro)

AION-C does NOT compete with these on absolute generation quality
at any scale we can afford. The frontier models have trillions of
tokens of training data and hundreds of billions of parameters.
No 1-10 billion parameter model matches them on open-ended text.

Where AION-C competes is on:
- Cost per query (1-2 orders of magnitude cheaper)
- Continual learning (frontier models cannot)
- Interpretability (frontier models are black boxes)
- Deployability in regulated or cost-sensitive contexts

### Small open models (Gemma 2 2B, Phi-3 Mini 3.8B, Qwen 2.5 Coder)

These are the direct comparison points at AION-C's target
parameter scale. They are dense transformers with no continual
learning, no sparse activation, no explicit reasoning graphs.

AION-C's 1.1 billion parameter target is smaller than Gemma 2
(2B) or Phi-3 Mini (3.8B). The bet is that AION-C's
architectural advantages (continual learning, sparse activation,
modular specialization) offset the smaller raw size, producing
competitive benchmark scores plus qualitatively new capabilities.

Whether this bet pays off is what Fase E will determine.

### Other modular approaches

There are academic efforts on modular AI (various MoE variants,
the Pathways idea from Google, specialized expert models).
AION-C is distinctive in combining MoE routing with explicit
causal reasoning (CRE), external memory, and the continual
learning cognitive layer. No public research project puts all
of these together in a working system with tests and a UI.

---

## Financial

### Current investment

- Approximately $62 of cloud compute to date (Vast.ai GPU rentals
  for experiments, H200 training session for an earlier version).
- Zero external funding.
- Built by one person (Jesús) with Claude Opus as a development
  assistant. No employees, no contractors.
- Development time: approximately six months of part-time work.

### Cost of the next step

- Fase E (1.1 billion parameter training on RTX 4090): $4.
- Post-training analysis and benchmarks: approximately $10-20 in
  additional compute for eval runs.
- Documentation and first paper draft: zero direct cost, mostly
  time.

Total immediate cost to reach a trained, benchmarked 1.1 billion
parameter model with a paper-worthy validation set: under $100.

### Near-term needs

- $200-500 for A/B test training runs (Fase F on vs off, dataset
  ablations, different learning rates).
- $500-1000 for a 3.5 billion parameter training attempt on a
  higher-end GPU if the 1.1 billion scale looks promising.
- Compute for running the production backend (if a pilot user
  signs on): low, approximately $50 per month for a modest
  cloud instance.

### Longer-term needs

- If the project needs a second person to share maintenance and
  move faster, full-time compensation plus benefits is $150-250K
  per year depending on seniority and location.
- Serious research compute for the 10 billion parameter scale is
  approximately $5,000-20,000 depending on training duration.
- Conference travel and publication costs (registration fees,
  travel): $5,000-10,000 per year.

The total budget for a year of focused development including
one additional engineer and ongoing compute is in the range of
$200,000-$350,000.

---

## Ask

What the project would benefit from in priority order.

### 1. Validation partner

Someone willing to use AION-C on a real task after the 1.1B is
trained. This is worth more than money — it provides the
empirical validation that the architecture produces useful
outputs, not just benchmark numbers. Ideal profile: a small
business with a well-defined knowledge-management or
customer-support problem who can tolerate a prototype and give
feedback.

### 2. Technical advisor

Someone with research credentials in state-space models,
continual learning, or neuro-symbolic AI who can review the
architecture, suggest improvements, and help shape the
publication strategy. One or two hours per month would be
enormously helpful.

### 3. Compute credits or a small grant

$1,000-$5,000 in cloud credits would unlock A/B testing,
higher-scale experiments, and the 3.5 billion parameter target.
This is not a blocker — the project will proceed at the 1.1
billion scale regardless — but it would accelerate the timeline
by months.

### 4. Introduction to potential users

If the project finds product-market fit after Fase E, an
introduction to 2-3 candidate customers in regulated industries
(healthcare, finance, legal) would short-circuit months of cold
outreach.

### 5. Capital

This is the last item intentionally. The project does NOT need
venture capital at this stage. Taking institutional money now
would create pressure on the wrong metrics (revenue, user growth)
when the actual question is architectural validation. If a later
stage requires scaling the team and compute budget significantly,
that is when institutional capital makes sense.

---

## Risks

The honest list of what could go wrong and how it is managed.

### Technical risk 1: The 1.1B underperforms

**Risk.** AION-C 1.1 billion parameters scores significantly
below Gemma 2 2B on benchmarks, undermining the thesis.

**Management.** The project already has strong architectural
evidence (the 5 experiments all pass at small scale). If the
1.1B underperforms, the first response is to analyze why
(undertrained, dataset gap, architecture issue) rather than
give up. The dataset has a known multi-turn gap that can be
closed in a second iteration. Training recipe tuning is
cheaper than a redesign.

### Technical risk 2: The cognitive layer breaks at scale

**Risk.** The five experiments pass at 5.5 million parameters,
but something in the interaction between the sparse gates, the
LoRA adapters, and the full 1.1 billion model breaks.

**Management.** The dry run of the 1.1 billion training pipeline
on the tiny configuration already exercises all the cognitive
layer components. The scaffolding verification at the end of
training catches regressions automatically. The RollbackManager
prevents corrupted training from persisting.

### Technical risk 3: Real users produce queries the training data does not cover

**Risk.** The 72K canonical dataset is small and synthetic.
Real user queries may expose gaps that the experiments did not
anticipate.

**Management.** This is exactly the problem the continual
learning mechanism solves. When a gap is found, the system can
learn the concept via auto-learn without breaking existing
capabilities. The bit-a-bit guarantee makes this safe.

### Business risk 1: No market for a 1.1B model regardless of architecture

**Risk.** Customers are willing to pay for frontier-quality
(Claude, GPT-4) or for free open models (Llama, Mistral), but
there is no middle market for a 1-10B model with
architectural advantages.

**Management.** The target verticals (regulated industries,
continual learning use cases, cost-sensitive deployments) have
specific pain points that frontier models and open Llamas do
not address. A single validated pilot in one of these verticals
is the proof point.

### Business risk 2: One-person bus factor

**Risk.** The project has exactly one person. If that person is
unavailable for any reason, progress stops.

**Management.** Acknowledged. Documentation is extensive (this
file, the README, the docs/ directory) so that a successor
could pick up the work. The tests are comprehensive (2856
passing) so the system has a clear contract. The priority for
the next phase after Fase E is bringing on a second person.

### External risk: A better modular architecture appears

**Risk.** A frontier lab publishes a better modular architecture
that obsoletes AION-C's design before it reaches scale.

**Management.** The field is moving fast and this is a real
risk. The mitigation is to publish AION-C's findings quickly
(workshop paper after Fase E) so the architectural ideas are
on record, and to stay flexible — if a better approach appears,
adopt what works rather than defending the existing design.

---

## Milestones

The next six months in priority order, with rough timelines.

### Month 1

- Rent an RTX 4090 on Vast.ai, upload the deployment zip, run the
  1.1B training. Budget: $4. Duration: 5-17 hours.
- Evaluate the trained 1.1B on the 50 canonical prompts plus
  HumanEval, GSM8K, and MMLU if time permits.
- Document results in an updated `docs/COMPARISON.md` and the main
  `README.md`.
- First public release of a workshop paper draft.

### Month 2

- A/B test Fase F on vs off by retraining the 1.1B without the
  cognitive layer and comparing.
- Investigate dataset gaps (multi-turn, tool use, adversarial)
  and regenerate the weak categories.
- Run the five validation experiments against the 1.1B and
  record per-scale comparison.
- Publish the workshop paper on arXiv as a technical report.

### Month 3

- Auto-learn benchmark: teach 20 real concepts (Rust book, Python
  library, math chapter) and measure quality gain, exam pass
  rate retention, and inference quality against held-out queries.
- First pilot deployment, if a validation partner is identified.

### Month 4

- Integrate the RewardLedger as a training signal for an RLHF-style
  second-pass finetune.
- Re-run comparisons with the RLHF version.
- Start design work for the Parte 28 streaming implementation in
  preparation for larger scale.

### Month 5

- Decide whether to scale to 3.5 billion parameters based on Fase
  E results. If yes, rent a higher-end GPU and run the training.
- Add multi-modal support if there is a clear use case.
- Second workshop paper or conference submission.

### Month 6

- Implement Parte 28 (hierarchical streaming) if 3.5B is working
  and the next scale target requires it.
- Pilot deployment with a real user at production scale.
- Begin design for Parte 29 (neuromorphic hooks) as a research
  direction.

### What success looks like at month 6

- A trained, benchmarked 1.1 billion parameter model with
  measurable advantages over comparable dense models on at least
  two of: continual learning, compute efficiency, interpretability.
- Two published workshop papers plus a technical report.
- One validated pilot with a real user generating real queries.
- A second person on the project sharing maintenance.
- A clear path to the 3.5B or 10B scale based on empirical
  evidence, not speculation.

---

## One-line summary

AION-C is a research artifact that bets on modular continually-learning
architecture as the path to cheap, interpretable, continually-updating
AI systems, with a complete working prototype, five validation
experiments that pass, a $4 training cost for the next scale, and a
founder ready to execute the next step this week.

The next move is simple: rent a GPU, run the training, analyze the
results, publish. If it works, the thesis has its first empirical
backing. If it does not, we learn something and iterate.
