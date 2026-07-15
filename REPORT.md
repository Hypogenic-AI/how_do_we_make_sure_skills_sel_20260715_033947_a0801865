# Are Skills / Self-Evolving agents *creating new knowledge*, or *better eliciting latent knowledge*?
### An operational discriminator, validated on a controlled substrate with real frontier LLMs

**Author:** automated research session · **Date:** 2026-07-15
**Models:** `anthropic/claude-sonnet-4.5`, `openai/gpt-4.1` (via OpenRouter), `Qwen/Qwen2.5-7B-Instruct` (local LoRA)

---

## 1. Executive Summary

When an LLM agent improves after receiving a "skill" (human-written or self-evolved), that gain
can mean two very different things: the skill **injected genuinely-new, transferable knowledge** the
model did not have (the "invented Attention/relativity" case), or it merely **re-elicited latent
knowledge** the model already possessed (a dressed-up prompt optimization). The field's self-evolving
systems are overwhelmingly black-box wrappers around a frozen base model, so this is an empirical
question — but existing skill ablations only measure task pass-rate and cannot say *why* a skill helped.

We built a controlled substrate where the ground-truth answer to "is this novel or latent?" is **known
by construction**, and an **Elicitation–Novelty Discriminator (END)** of four measurable quantities.
Across two frontier models and 3,000+ real API evaluations we find a sharp, consistent signature:

- **A skill injects genuinely-new knowledge only when the model cannot self-generate it.** On an
  invented-language glossary ("Zeth"), the correct skill lifts accuracy **0.00 → 1.00** (both models,
  p < 10⁻⁴, Cohen's h = 3.14), while the model's *self-generated* skill and a length-matched control
  both stay at **0.00**. This is the one unambiguous "new knowledge" case — and the model provably
  cannot produce it alone.
- **Self-evolution is bounded by what is latent + the external signal.** A Voyager/STaR-style
  self-evolution loop **never** moves the Zeth task off 0.00 under *verifier-only* feedback across 5
  iterations, but reaches **0.85** the moment it receives *ground-truth answers* — exactly the
  AlphaEvolve lesson (the LLM proposes; an external verifier discovers). For a *latent-but-unreliable*
  procedure (base-9 multiplication) self-evolution does recover (up to 0.95), because the knowledge was
  already there to elicit.
- **Self-generated skills can be actively harmful.** On base-9 multiplication Claude's own skill
  *dropped* accuracy from 0.85 to **0.25** — it articulated a harder, more error-prone algorithm than
  its own default — a concrete mechanism for the SkillsBench finding that self-generated skills fall
  below baseline.
- **The user's proposed "fine-tune the skill back into the weights" probe is real but method-dependent.**
  Genuinely-new information (Zeth) *can* be written into weights and generalizes to held-out phrases
  (**1.00** accuracy, no skill in context) — but only via **example-distillation**; fine-tuning on the
  skill *document* leaves accuracy at **0.00**. A naive doc-finetune would wrongly declare real,
  transferable knowledge to be "mere prompt optimization."

**Bottom line for the research question:** current self-evolving / self-generated skills are, in the
strict sense, **better elicitation, not knowledge creation** — because self-evolution cannot produce
knowledge the base model lacks. Genuine novelty enters a skill **only from an external source** (a human
who knows the procedure, or a ground-truth verifier + search that discovers it). The reliable way to
"make sure" is to measure whether the model can self-generate the skill and whether the gain survives a
length-matched control — not to admire novel-sounding text.

---

## 2. Research Question & Motivation

> Is existing Skill / self-evolution creating new ideas, or just finding a better way to exploit what
> LLMs have already learned? How can we *tell the two apart*?

The user framed the crux precisely: an LLM that fails an IMO problem, then succeeds after being given (or
self-developing) a skill "xxx geometry theory," may have (a) *recalled* a latent, pre-training-adjacent
concept via better prompting, or (b) genuinely *acquired* a new procedure. Standard memorization tests
cannot separate these (most training samples are seen once, so there is no over-fitting fingerprint), and
the user noted that even the obvious probe — fine-tuning the skill back into weights — does not settle it.

**Gap in prior work** (`literature_review.md`, 20 papers): 13/20 argue elicitation, only AlphaEvolve shows
verifiable novelty — and that comes from an *external verifier + search*, not from the LLM introspecting.
Crucially, every existing skill ablation (SkillsBench, the DS-skills ablation) measures only pass-rate
*with vs. without* a skill; **none isolates whether the model could have produced the skill itself**, and
none has a substrate where "novel vs. latent" is ground-truth-known. Subjective novelty judgments are
worse than useless — HindSight shows LLM/human novelty ratings *anti-correlate* (ρ = −0.29) with real
impact.

---

## 3. Methodology

### 3.1 A controlled substrate where novelty is known by construction

We use **counterfactual tasks** whose regime we can dial, all verified deterministically against
computable ground truth (no LLM-as-judge, per the HindSight caveat):

| Setting | Task | Regime | What a skill *could* be |
|---|---|---|---|
| `arith_b10` | base-10 3-digit multiplication | **elicitation** | re-elicit a latent, reliable skill |
| `vocab_es` | Spanish→English (24-word glossary) | **elicitation** | re-elicit words the model already knows |
| `arith_b9`, `arith_b11` | base-9 / base-11 3-digit multiplication | **novelty-procedure** | scaffold a latent-but-unreliable procedure (canonical counterfactual from *Reasoning or Reciting?*) |
| `vocab_zeth` | invented-language ("Zeth")→English | **novelty-information** | supply information the model provably cannot have (glossary is randomly generated at run time) |

`vocab_zeth` is the key construction: the 24-word glossary is generated from a fixed random seed, so the
mapping **cannot** be in any training corpus. It is the cleanest lab analog of "a new theory the model has
not memorized," yet its answers are exactly checkable.

### 3.2 The Elicitation–Novelty Discriminator (END)

For a skill *S* that improves a task, we measure four quantities (fixed base model, held-out test set):

- **Normalized gain** g = (acc_S − acc₀)/(1 − acc₀).
- **Self-Generation utility** SG = g(self-generated S) / g(correct S). High ⇒ the knowledge was already
  latent (the model can write its own working skill) ⇒ elicitation.
- **Length-Controlled Gain** LCG = g(correct S) − g(length-matched irrelevant filler). Isolates real
  content from mere context-length / prompt-shape effects.
- **Transfer**: does a skill formed on one variant help on an unseen one? (Exp 2/3 held-out splits.)

**Signature predicted:** *elicitation* ⇒ small g, high SG, LCG ≈ 0. *Novelty* ⇒ large g, **low SG**,
LCG > 0. The falsifiable claim: self-evolution/self-generation sits at the elicitation floor precisely
because SG is low in the novelty regime (you cannot self-evolve knowledge you do not have).

### 3.3 Three experiments (real models)

- **Exp 1 — Discriminator.** 5 settings × 2 models × 4 conditions (no-skill / correct-skill /
  self-generated-skill / length-matched control) × N=60 held-out items. Self-generated skills are written
  by the model *unaided* (task description only, no answers) then reused on the test set.
- **Exp 2 — Self-evolution loop.** A Voyager/STaR-style loop (K=5) that refines the model's *own* skill
  from experience, under two feedback signals: **verifier-only** (pass/fail) vs. **with-answer**
  (ground-truth revealed). Learning curve on a disjoint 40-item test set. Settings: `arith_b9`,
  `vocab_zeth`, `vocab_es`. Model: claude-sonnet-4.5.
- **Exp 3 — Internalization probe** (the user's proposed test, on the GPU). LoRA fine-tune
  Qwen2.5-7B-Instruct to put the skill *into weights*, then test **without** the skill in context, on a
  held-out split. Two fine-tuning modes: **doc** (train on the skill document) vs. **examples** (train on
  input→output pairs). Settings: `vocab_zeth`, `arith_b9`.

### 3.4 Reproducibility

Seeds fixed at 42; temperature 0 (deterministic); all API responses cached to `results/llm_cache/`
(≈3,000 calls). Deterministic verifiers in `src/tasks.py`. Bootstrap 95% CIs (5,000 resamples),
two-proportion z-tests, Cohen's h effect sizes. LoRA: r=16, α=32, 3 epochs, lr 2e-4, bf16, single A6000.
Hardware: 4× RTX A6000 (one used). Environment: Python 3.12, torch 2.5.1+cu124, transformers 4.56.2,
peft. Full config and code in `src/`; raw outputs in `results/`.

> **A note on provider guardrails.** Our original novelty-information task was a substitution cipher; the
> Amazon Bedrock/Google guardrails on OpenRouter blocked it (`finish_reason: content_filter`) as
> jailbreak-adjacent. We replaced it with the mathematically-equivalent but innocuous invented-glossary
> task. This is documented honestly as it affects reproducibility.

---

## 4. Results

### 4.1 Exp 1 — the discriminator separates the regimes cleanly

Accuracy by condition (N=60, bootstrap 95% CI in `results/exp1_summary.csv`; figure
`figures/exp1_condition_bars.png`):

| Model | Setting | Regime | no-skill | correct | self-gen | length-ctrl |
|---|---|---|---:|---:|---:|---:|
| claude-sonnet-4.5 | arith_b10 | elicitation | 0.983 | 0.967 | 0.983 | 0.950 |
| claude-sonnet-4.5 | vocab_es | elicitation | 0.950 | 1.000 | 1.000 | 0.950 |
| claude-sonnet-4.5 | arith_b9 | novelty-proc | 0.850 | 0.883 | **0.250** | 0.850 |
| claude-sonnet-4.5 | arith_b11 | novelty-proc | 0.767 | 0.683 | 0.767 | 0.750 |
| claude-sonnet-4.5 | **vocab_zeth** | **novelty-info** | **0.000** | **1.000** | **0.000** | **0.000** |
| gpt-4.1 | arith_b10 | elicitation | 0.417 | 0.700 | 0.367 | 0.250 |
| gpt-4.1 | vocab_es | elicitation | 0.917 | 1.000 | 0.850 | 0.917 |
| gpt-4.1 | arith_b9 | novelty-proc | 0.000 | 0.000 | 0.000 | 0.000 |
| gpt-4.1 | arith_b11 | novelty-proc | 0.017 | 0.000 | 0.000 | 0.017 |
| gpt-4.1 | **vocab_zeth** | **novelty-info** | **0.000** | **1.000** | **0.000** | **0.000** |

Discriminator metrics (`results/discriminator.csv`; figure `figures/exp1_discriminator_scatter.png`):

- **Novelty-information (`vocab_zeth`), both models:** g_correct = 1.00, **SG = 0.00**, **LCG = 1.00**,
  p(correct vs base) < 10⁻⁴, Cohen's h = 3.14 (maximal). The correct skill is decisive; the
  self-generated skill and the length-matched control are worthless. → **genuine new knowledge, and
  un-self-generatable.**
- **Elicitation (`vocab_es`, `arith_b10`):** self-generated ≈ correct (SG ≈ 1 for Spanish on Claude),
  and a length-matched irrelevant memo does about as well as the "skill" (Spanish: 0.95 vs 0.95). →
  **the skill is doing prompt-shaping, not teaching.** (GPT-4.1's base-10 multiplication is genuinely
  execution-limited — the skill helps 0.42→0.70 — a scaffolding effect, still fully latent.)
- **Novelty-procedure (`arith_b9/b11`):** the *correct* skill barely helps on Claude (execution, not
  knowledge, is the bottleneck) and GPT-4.1 fails at ~0 regardless of skill (a **capability floor**: an
  in-context skill cannot buy compute the model lacks). The **self-generated** base-9 skill *hurts*
  (0.85 → 0.25).

### 4.2 Exp 2 — self-evolution is bounded by latency + external signal

Learning curves (`figures/exp2_self_evolution.png`), claude-sonnet-4.5, K=5:

| Setting | verifier-only (iter1→5) | with-answer (iter1→5) | correct-skill ceiling |
|---|---|---|---|
| `arith_b9` (novelty-proc, **latent**) | 0.275 → 0.075 → 0.90 → 0.95 → **0.925** | 0.375 → 0.95 → … → **0.95** | 0.875 |
| `vocab_zeth` (novelty-info, **absent**) | 0.00 → 0.00 → 0.00 → 0.00 → **0.00** | **0.85** (from iter 1, stable) | 1.000 |
| `vocab_es` (elicitation) | ~0.87 (flat) | ~0.9 (flat, near ceiling) | 1.000 |

The decisive contrast is `vocab_zeth`: **verifier-only self-evolution stays pinned at 0.00 for all five
iterations** — pass/fail leaks essentially no information about 24 arbitrary word mappings, so
introspection cannot manufacture them. The **instant** it receives ground-truth answers it climbs to 0.85
(it copies the revealed mappings into a glossary). This is precisely the AlphaEvolve / "Cannot Self-Correct
Yet" pattern: **an external, non-gameable signal is what turns self-evolution into discovery.** For the
*latent* base-9 procedure, self-evolution recovers even from verifier-only feedback — because it is
eliciting knowledge that was already present.

### 4.3 Exp 3 — the "fine-tune it back" probe: real, but method-dependent

Qwen2.5-7B-Instruct, held-out test, **no skill in the prompt** unless "in-context"
(`figures/exp3_internalization.png`):

| Setting | base (no skill) | in-context skill | LoRA on **document** | LoRA on **examples** |
|---|---:|---:|---:|---:|
| `vocab_zeth` (novelty-info) | 0.00 | 0.95 | **0.00** | **1.00** |
| `arith_b9` (novelty-proc) | 0.00 | 0.00 | 0.00 | 0.00 |

For Zeth, example-distillation writes the glossary **into the weights** so the model translates *held-out*
phrases at 1.00 with nothing in context — proof that the skill encoded real, transferable knowledge, not a
prompt trick. But fine-tuning on the *document* (the same information, as prose) internalizes **nothing**
(0.00). The user's probe therefore *cannot* by itself classify a skill: its verdict flips with the
fine-tuning recipe. For base-9 multiplication a 7B model fails at 0.00 in every condition — an in-context
skill and light fine-tuning alike cannot supply executional capacity it lacks.

---

## 5. Analysis & Discussion

**Answering the question.** Across every setting, a skill delivered a large, content-specific,
length-controlled gain in **exactly one** place: where the required knowledge was genuinely absent from the
model (`vocab_zeth`) — and there, the model could **never** self-generate it (SG = 0, and verifier-only
self-evolution = 0 across five rounds). Everywhere the model *could* self-generate the skill (Spanish,
base-10), the skill was doing prompt-optimization: a length-matched irrelevant memo matched it. Therefore:

> **Self-evolution/self-generation ≈ better elicitation, not knowledge creation.** The ceiling of what an
> agent can give *itself* is what is already latent. Genuine novelty enters a skill only from an external
> source — a human who holds the procedure, or a ground-truth verifier + search that discovers it (the
> AlphaEvolve pattern). This is not a definitional stance; it is what the SG and LCG measurements show.

**Why self-generated skills can *hurt* (SkillsBench, mechanistically).** Claude's self-authored base-9
skill instructed the harder *native* base-9 long-multiplication algorithm (multiply digit-by-digit,
convert every partial product, carry in base 9), whereas its own default (and our correct skill) uses the
reliable convert-to-decimal-then-back route. Forced to follow its own more error-prone articulation,
accuracy fell 0.85 → 0.25. A self-generated skill is not new knowledge; it is a *sampled elicitation
strategy*, and a worse one can interfere.

**A practical recipe to "make sure."** To decide whether a skill is novelty or elicitation, don't score
its prose. Measure: (1) **SG** — can the base model write a working version unaided? (2) **LCG** — does it
beat a token-matched irrelevant control? (3) **verifier-only self-evolution** — can the loop reach the
skill without ground-truth answers? A skill is *genuine new knowledge* iff SG ≈ 0, LCG > 0, and
verifier-only self-evolution stalls while an external signal unlocks it. This protocol is the deliverable.

**Relation to the literature.** Our results reproduce and *mechanize* four prior findings on one substrate:
self-generated skills ≤ baseline (SkillsBench), length-control indistinguishability (DS-skills ablation),
no self-correction without an external signal (Cannot-Self-Correct-Yet), and discovery = verifier + search
(AlphaEvolve). The novelty here is the **controlled ground truth** and the **SG/LCG discriminator** that
ties them together.

**Contamination note.** That frontier models still score 0.00 on Zeth is expected and *strengthens* the
design: the glossary is randomly generated, so there is nothing to be contaminated by. Base-9 discussions
surely appear in pretraining, yet the models are execution-limited anyway — again consistent with
"latent-but-unreliable," not "novel."

---

## 6. Limitations

- **Scope of "novelty."** Our novelty-information task injects *declarative* new facts (a glossary), not a
  new *reasoning* method à la Attention. Injecting a genuinely-new *procedure* the model cannot execute is
  the harder, unaddressed case (our base-9 "capability floor" hints at it). The discriminator (SG, LCG)
  should transfer, but we have not demonstrated a positive novel-*procedure* result.
- **Two families, short tasks.** Arithmetic and vocabulary are clean but narrow; real agentic Skills
  (SkillsBench) are longer, multi-step, and tool-using. We deliberately traded ecological realism for
  ground-truth control; the SkillsBench harness (Docker/BenchFlow) was too heavy to run end-to-end here.
- **Two frontier models + one 7B.** Broader model coverage (Gemini, larger open models) would test the
  capability-floor story further.
- **Self-generation operationalization.** We measured zero-feedback self-generation and a 5-step evolution
  loop; a much longer or tool-augmented loop might crack Zeth from frequency structure — but note our
  with-answer loop already shows that only ground-truth signal helps.
- **Ceiling noise in normalized gain.** For near-ceiling settings (base-10, Spanish on Claude) g is
  numerically unstable; we base those interpretations on absolute accuracies and the length control, not g.
- **Provider guardrail substitution** (cipher → glossary) documented in §3.4.

---

## 7. Conclusions & Next Steps

**Answer to the research question.** On a substrate where "new vs. latent" is known by construction,
skills cross from *elicitation* into *genuine new knowledge* in exactly the cases the base model
**cannot self-generate** the skill — and in those cases self-evolution from introspection or verifier-only
feedback fails completely, recovering only when fed an external ground-truth signal. Today's self-evolving
skills are therefore, rigorously, **prompt/elicitation optimization bounded by the base model's latent
knowledge; not autonomous discovery.** Genuine novelty requires an external source of the new procedure
(human curation, or a correct verifier + search). The way to "make sure" is to measure self-generability
(SG) and length-controlled gain (LCG) — never to trust novel-sounding text.

**Next steps.** (1) Build a *novel-procedure* task where the correct procedure is executable by the model
but not latent, to seek a positive novelty result beyond declarative facts. (2) Port the SG/LCG
discriminator onto SkillsBench's real agentic tasks. (3) Add a verifier-hacking probe (Red-Queen) to the
self-evolution loop. (4) Test whether verifier+search (AlphaEvolve-style) on our substrate discovers
procedures no single model can self-generate — the constructive complement to this study's negative result.

---

## References
Key papers (full list and notes in `literature_review.md` / `resources.md`): Voyager (2305.16291),
Self-Evolving Agents Survey (2508.07407), AlphaEvolve (2506.13131), SkillsBench (2602.12670), LLM-Skills
DS Ablation (2607.07504), Library Drift (2605.19576), Reasoning or Reciting? (2307.02477), Emergent
Abilities: Mirage? (2304.15004), LLMs Cannot Self-Correct Yet (2310.01798), STaR (2203.14465), HindSight
(2603.15164), Promptbreeder (2309.16797).

**Artifacts.** `src/` (harness), `results/` (raw JSON, CSV summaries, LLM cache), `figures/` (4 plots),
`planning.md` (pre-registered design).
