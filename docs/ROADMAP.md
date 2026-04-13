# Roadmap

This document is the honest plan for AION-C going forward. Dates are
relative to April 2026. The roadmap is split into phases that correspond
to decisions the project faces in order.

## Table of contents

1. [Current state — summary](#current-state)
2. [Fase E — Scale to 1.1B](#fase-e)
3. [Post Fase E — Analysis and decisions](#post-fase-e)
4. [Parte 28 — Hierarchical streaming](#parte-28)
5. [Parte 29 — Neuromorphic hooks](#parte-29)
6. [Publication strategy](#publication)
7. [Team and funding](#team-and-funding)
8. [Long-term research questions](#long-term-research)

---

## Current state

As of this document's writing:

- **Fase A** (infrastructure) complete. 12 items, +391 tests.
- **Fase B** (data pipeline) complete. 70K canonical dataset.
- **Fase C** (tiny training) complete. 5.5M params, 98.2% routing, 7/7 E2E.
- **Fase D** (backend + UI) complete. 25+ endpoints, React CDN UI.
- **Fase F** (cognitive layer) complete. 7 packages, +221 tests, 5/5 experiments pass.
- **Paso 0** (metacognitive dataset) complete. 2,500 new examples, dataset at 72.5K.
- **Paso 1** (cleanup) complete. AION-C/ duplicate removed, legacy files archived.
- **Paso 2** (persistence) complete. RewardLedger + HierarchicalStore JSONL.
- **Paso 3** (sleep daemon loop) complete. Asyncio background task in backend.
- **Paso 4** (real-tiny experiments) complete. 5/5 pass on the real pipeline.
- **Paso 5** (Fase F in training) complete. Sparse gates + scaffolding verification.
- **Paso 6** (repackage) complete. `aion_c_vast.zip` 27.2 MB with all Fase F packages.
- **Paso 7** (dry-run) complete. 50-step training on tiny with Fase F passes.

Test count: 2856 passing, 0 regressions.

The project is fully prepared for Fase E. All that remains is to
rent a GPU.

---

## Fase E

**Scale to 1.1B on RTX 4090.**

### Prerequisites (all met)

- `aion_c_vast.zip` packaged and ready (27.2 MB, 307 files)
- `train_1b_canonical.py` with Fase F integration active by default
- Dataset: `datasets/dataset_canonical_72_5k.jsonl` (72,500 records, EOS 100%)
- Dry-run passing on tiny config (scaffolding verification + density logging)
- Evaluation harness: 50 canonical prompts in `evaluation/eval_prompts.py`
- Budget: approximately $4 (Vast.ai 4090 at $0.23/hr × 17 hours)

### Plan

```bash
# 1. Rent a Vast.ai instance with RTX 4090
# Target: $0.23/hr or lower, 40 GB disk minimum

# 2. Upload and extract
scp -P <PORT> aion_c_vast.zip root@<HOST>:/root/
ssh -p <PORT> root@<HOST>
cd /root && unzip aion_c_vast.zip && cd AION-C

# 3. Install deps
pip install -r requirements_vast.txt

# 4. Sanity check GPU
python -c "import torch; print('CUDA:', torch.cuda.is_available(),
    torch.cuda.get_device_name(0))"

# 5. Run preflight tests
python -m pytest tests/test_evaluation.py \
    tests/test_canonical_dataloader.py \
    tests/test_brain_version.py -q

# 6. Sanity check training script at small scale
python train_tiny_canonical.py --steps 100

# 7. Start the real 1.1B training
python train_1b_canonical.py --config 1b --steps 15000 \
    2>&1 | tee train_1b.log

# 8. Monitor via the log file
# ~3 sps means ~75 min of compute for 15K steps plus eval time
```

### Expected timeline

| Hour | Activity |
|------|----------|
| 0 | Upload, install, sanity check |
| 0.5 | Run `train_tiny_canonical.py --steps 100` to verify GPU path |
| 1 | Start `train_1b_canonical.py --config 1b` |
| 1-3 | Warm-up and first eval checkpoints |
| 3-6 | Main training, routing accuracy climbs |
| 6-10 | Eval combined score converges |
| 10-17 | Either early stops or hits max_steps |
| 17-18 | Save brain/v1/, download checkpoint, shutdown instance |

Expected cost: $3.90 at $0.23/hr × 17h. If it early stops at hour 8,
cost is $1.84. Budget $4.00 gives a generous margin.

### Success criteria

- Training completes without crashes.
- `gen_quality.combined` at the best step exceeds the tiny's 0.4002
  by a meaningful margin (say, 0.55+).
- `routing_accuracy` maintains >95%.
- Scaffolding verification passes at the end (all 5 motors bit-a-bit
  after attach/detach test).
- `sparsity_loss` stays stable or decreases.
- No NaN losses.

### Failure modes and mitigations

- **Out of memory.** Expected VRAM use is ~18 GB for the 1.1B at
  batch 1 with context 1024 in fp16. If it overflows, reduce
  `dec_max_seq_len` or `hidden_dim`.
- **NaN loss.** Already mitigated with gradient clipping and AMP
  GradScaler. If it still happens, reduce learning rate or disable
  AMP.
- **Routing collapse.** If `routing_accuracy` drops below 80%,
  increase `balance_w` above 0.5.
- **Sparse loss diverges.** If `sparsity_loss` grows unbounded,
  reduce `sparsity_w` below 0.1 or temporarily disable Fase F with
  `--no-fase-f`.
- **Instance crashes.** Resume from the last checkpoint:
  `python train_1b_canonical.py --config 1b --steps 15000 --resume
  checkpoints/aion_1b_canonical.pt`.

---

## Post Fase E

**Weeks 3-8: analysis and decisions.**

### Immediate post-training work

- Download checkpoints and metrics JSON to local.
- Run the 50 canonical eval prompts fully and record per-domain
  breakdown.
- Run HumanEval if the model generates Python (forge_c domain).
- Run GSM8K if the model does arithmetic (axiom domain).
- Record baseline numbers for future comparison.

### Analysis questions

- **Did training quality improve with Fase F compared to a baseline
  without Fase F?** We can run a shadow training (same recipe minus
  `--no-fase-f`) if time and budget allow. This is an A/B test.
- **How much of the improvement comes from the 2500 metacognitive
  examples?** We can run a shadow training on the 70K baseline
  dataset (without metacognitive) and compare.
- **What is the actual per-query compute cost with sparse gates
  active?** Measure FLOPs before and after sparsity kicks in.
- **Do the 5 cognitive experiments still pass on the 1.1B?** Re-run
  `experiments/fase_f/run_real.py` against the new checkpoint.

### Decisions to make after analysis

- **Iterate on the 1.1B or scale to 3.5B?** If the 1.1B is clearly
  undertrained (loss still decreasing sharply at step 15K), we
  extend training. If it has plateaued, we scale up.
- **Which dataset gaps are most painful?** The multi-turn ratio is
  known to be low (7%). If the 1.1B is weak on multi-turn, that is
  the first thing to fix.
- **Ship as a demo or keep internal?** The backend UI is production-
  ready. At the 1.1B scale it becomes usable for real conversations.
  Whether to expose it publicly is a research-ethics and compute-cost
  question.

### Near-term improvements (weeks 3-8)

- **Multi-turn data regeneration.** Target 20-30% multi-turn
  distribution in the dataset.
- **Real auto-learn benchmark.** Teach 20 real concepts (a Rust
  book, a Python library doc, a math textbook chapter) and measure
  the quality gain, the time per concept, and the exam pass rate.
- **RLHF from the RewardLedger.** The ledger now persists real
  user feedback. Use it as a training signal for a second-pass
  finetune.
- **Formal publication.** Write a short paper on the architecture
  plus Fase F experiment results. Submit to a workshop.

---

## Parte 28

**Streaming jerárquico de motores — VRAM / RAM / SSD / predictive.**

This is the plan for the next scale jump after the 1.1B. Documented
in the mega-prompt section 28, not yet implemented.

### Why it matters

At the 1.1B scale, the model fits in VRAM with room to spare on a
4090. At 3.5B it starts getting tight. At 10B+ it does not fit at
all. Parte 28 is the mechanism that lets a large model run on a
modest GPU by keeping most of it in system RAM or SSD and streaming
only the active parts to VRAM.

### The four layers

1. **VRAM (Layer 1)** — the shared backbone (encoder, orchestrator,
   unifier, decoder). Approximately 30% of total params. Always
   resident on GPU.
2. **System RAM (Layer 2)** — the 5 motors. Transferred to VRAM on
   demand. Pinned memory for fast async copy.
3. **SSD NVMe (Layer 3)** — adapters, MEM, compressed concepts. Mmap
   or lazy load.
4. **Predictive (Layer 4)** — the orchestrator pre-loads motors
   while the user is typing, before the query is complete.

### Implementation plan

- Separate the backbone from the motors in the model definition.
- `MotorResidencyManager.load(motor) / evict(motor) / status()` API.
- Pinned RAM allocation for motor state dicts.
- CUDA streams for async copy without blocking inference.
- Orchestrator streaming hook that emits partial routing scores
  on each typed character.
- UI metric panel showing VRAM used, resident motors, hit/miss of
  the predictive preload.

### When

After Fase E is done and we have a 1.1B model validated. The
priority is Fase E first because without a working large model,
streaming is a solution looking for a problem.

Estimated implementation time: 2-4 weeks of focused work, once the
1.1B is in hand.

---

## Parte 29

**Neuromorphic hooks — not yet in the mega-prompt.**

This is speculative. The hypothesis is that the CRE's message passing
structure maps cleanly to spiking neural network substrates. A
message on an edge can be interpreted as a spike; the GRU update can
be approximated by leaky integrate-and-fire dynamics; the graph
topology is inherently sparse and localized.

### Why care

- Spiking networks run efficiently on neuromorphic hardware (Intel
  Loihi, BrainChip Akida, SpiNNaker). If AION-C's core can be
  expressed in SNN form, it could run at milliwatt power levels.
- The biological plausibility is a research story in its own right.
- The event-driven nature matches the lifecycle model (ACTIVE / IDLE
  / SLEEPING) — a neuromorphic AION-C would literally sleep.

### Status

Not a priority. It waits until the 1.1B is validated and the
streaming work is done. It is a reach goal for year 2.

### Prerequisites

- A working large AION-C (Fase E done).
- Basic familiarity with an SNN simulator (Nengo, Brian2) or real
  hardware.
- A collaborator who understands neuromorphic engineering at the
  hardware level.

---

## Publication

### Short-term papers

- **Workshop paper** (ICLR or NeurIPS workshop track). Architecture
  overview, Fase F cognitive layer, five validation experiments.
  Target: 6-8 pages. Timeline: month 2 after Fase E.
- **Technical report** (arXiv). Full architecture, all phases, all
  experiments. Target: 20-30 pages. Timeline: month 3 after Fase E.

### Medium-term papers

- **Cognitive layer paper.** Deep dive on the six phases of the
  sleep cycle, the four signals of pruning, and the probabilistic
  reward formula. Target: a full conference venue (NeurIPS, ICLR).
- **Auto-learn paper.** The bit-a-bit detach guarantee plus
  empirical results on real concepts. Target: ACL or EMNLP (NLP
  venues care about continual learning).
- **Comparison paper.** AION-C vs Gemma 2 2B vs Phi-3 Mini 3.8B on
  HumanEval, GSM8K, MMLU, and the continual learning benchmark we
  build. Target: any top-tier venue.

### Long-term

- **Architecture monograph.** Book-length treatment of the
  full system with all decisions explained. Target: an AI/ML
  publisher. Timeline: year 2.

---

## Team and funding

### Current team

One person (Jesús) plus Claude Opus as an assistant. Self-funded at
approximately $62 invested to date. No sponsors.

### Near-term needs

- A second person to share maintenance once the 1.1B is running.
- Compute budget for multiple 1.1B training runs (A/B tests of Fase
  F on/off, dataset ablations). Target: $200-500 for the next two
  months.
- Storage for multiple checkpoints (~2 GB each).

### What would accelerate the project

- A research collaborator who is familiar with state-space models
  (Mamba specifically) and can review the encoder/decoder code.
- A compute sponsor who can donate $1000-5000 in cloud credits for
  3.5B and 10B scale experiments.
- An advisor from a research lab who can help shape the publication
  strategy.

### What would NOT help

- Venture capital at this stage. The project is pre-paper and
  pre-traction; taking VC money now would put pressure on the
  wrong metrics.
- Productization effort. The system is a research artifact, not a
  product. Turning it into a SaaS would distract from the science.

---

## Long-term research

The five open questions from Parte 21 are the research agenda for
years 2-3. Each one is a paper's worth of experimental work at real
scale:

1. **Scaling sequential learning.** Can a single motor hold 1000
   adapters? 10,000? At what point does growth escalate from adapter
   to expansion? What is the recovery path from a bad auto-learn?
2. **Cross-domain interference at scale.** When 5 motors all have
   dozens of adapters, do they interfere through the shared
   backbone? If so, how to measure and mitigate?
3. **Compute cost at production scale.** 1000 adapters × 1.1B base
   × 100K queries/day. What is the infrastructure cost? Where are
   the bottlenecks?
4. **Compositional trajectory quality.** At scale, does the
   `forge_c → muse → unifier` trajectory actually produce better
   outputs than a single motor? Or is the overhead not worth it?
5. **Self-evaluation calibration.** Can the model reliably know
   when its output is wrong, as opposed to merely having low
   confidence? A calibration study with real users.

These questions cannot be answered at the 5.5M tiny scale and are
premature at the 1.1B. They become meaningful at the 10B+ scale
where the system's compute patterns match real deployments.

---

## What is explicitly NOT on the roadmap

Saying no is part of a good roadmap.

- **Chasing absolute-quality benchmarks against GPT-4 / Claude Opus
  / Gemini 2.** AION-C at 1-10B is not going to win MMLU against
  frontier models. That is not the contest.
- **Building a chatbot product.** This is a research artifact until
  proven otherwise.
- **Expanding the dataset to trillion-token scale.** The bet is on
  architecture and continual learning, not corpus size.
- **Supporting languages beyond English and Spanish initially.**
  Adding more languages is straightforward but distracts from
  architectural validation.
- **Multi-modality (vision, audio).** The architecture supports it
  in principle (the encoder is pluggable), but it is not the focus.

---

## Dates and milestones

An honest list of target dates. All dates are "best case if nothing
goes wrong" — actual slippage is expected.

| Milestone | Target | Confidence |
|-----------|--------|-----------|
| Fase E training complete | Week 1 | High |
| Post-training analysis | Week 2 | High |
| First workshop paper draft | Week 4 | Medium |
| Multi-turn dataset iteration | Week 6 | Medium |
| Real auto-learn benchmark | Week 8 | Medium |
| RLHF second-pass finetune | Week 10 | Low |
| 3.5B training decision | Week 12 | Low |
| Streaming (Parte 28) prototype | Month 4 | Very low |

The further out, the less confidence. That is normal.

---

## How to track progress

The `memory/project_mega_prompt_progress.md` file in the user's
memory directory is updated as each step completes. The git log of
this repository is the authoritative history. Each phase has a
section in the README updated with the new test count and metrics.

If you are reading this document to decide whether to contribute,
the best way to know the current state is to run the test suite and
the experiments:

```bash
python -m pytest tests/ -q       # should show 2856 passing
python -m experiments.fase_f.run_real  # should show 5/5 passing
```

If both produce the expected numbers, the project is in the state
this roadmap describes. If not, this roadmap is out of date and
you should check the git log for the latest changes.

---

The roadmap will be revised after Fase E produces actual results.
At that point we will either have a strong 1.1B model and a clear
path to scaling, or we will have learned something unexpected that
requires rethinking parts of the architecture. Both outcomes are
informative. The only bad outcome is not running Fase E at all.
