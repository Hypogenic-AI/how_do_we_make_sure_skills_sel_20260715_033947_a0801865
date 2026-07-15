# Research Plan: Is Skill/Self-Evolution *creating new knowledge* or *better eliciting latent knowledge*?

## Research Question
When an LLM agent gains performance from a "skill" (human-crafted or self-evolved), is the skill
(a) **injecting genuinely-new, transferable procedure** the model did not already possess (the
"invented Attention / General Relativity" case), or (b) **re-eliciting latent knowledge** the model
already had, i.e. a dressed-up prompt-optimization? How can we operationally *tell the two apart*?

---

## Motivation & Novelty Assessment

### Why This Research Matters
Self-evolving agents and "Skills" (Voyager, Darwin Gödel Machine, Agent-Skills/SKILL.md, Promptbreeder)
are being marketed as engines of discovery, but almost all are **black-box wrappers around a frozen base
model** — they add no weights and no new training signal beyond what a verifier/environment already
provides. If their gains are only better elicitation, the field is conflating *prompt optimization* with
*knowledge creation*. Getting this distinction right determines whether "self-improving AI" is a path to
new science or a plateau at the base model's latent ceiling.

### Gap in Existing Work
The literature (see `literature_review.md`) is lopsided: **13/20 papers** argue elicitation, only
**1/20 (AlphaEvolve)** shows verifiable novelty — and that comes from an *external verifier + search*, not
the LLM introspecting. Crucially, existing ablations (SkillsBench, the DS-skills ablation) measure only
**task pass-rate** with/without a skill. **None isolates *why* a skill helps** — none separates "the model
could have produced this itself" (latent) from "this is procedure the model cannot self-generate" (new).
The user's own proposed probe (fine-tune the skill back into weights) is acknowledged as *insufficient*.
There is no controlled substrate where "novel vs. latent" is **ground-truth-known**, so novelty claims rest
on subjective LLM/human judgments — which HindSight shows *anti-correlate* (ρ=−0.29) with real impact.

### Our Novel Contribution
1. **A controlled substrate with known ground truth for novelty.** We use *counterfactual* tasks
   (base-b arithmetic with b≠10; permuted-alphabet ciphers) where we can *dial* the regime:
   - **Elicitation regime** (base-10 / identity cipher): the required procedure is provably in-distribution
     (latent). A skill here can only *re-elicit*.
   - **Novelty regime** (base-9/11 / permuted cipher): the correct procedure exists and is verifiable, but
     is *out-of-distribution* — the closest lab analog to "a new theory the model has not memorized."
   Ground truth is computable ⇒ a **deterministic verifier**, no LLM-as-judge.
2. **The Elicitation–Novelty Discriminator (END)** — four measurable quantities that classify a skill's
   benefit:
   - **Normalized gain** g = (acc_skill − acc_base)/(1 − acc_base).
   - **Self-Generation rate SG** — can the model, *unaided*, produce a skill that actually works? (measured
     by whether its self-authored skill raises held-out accuracy). High SG ⇒ latent ⇒ elicitation.
   - **Length-Controlled Gain LCG** = g(real skill) − g(length-matched irrelevant filler). Isolates content
     from mere context-length/prompt-shape effects.
   - **Transfer Gain TG** — does a skill formed on variant A help on an *unseen* variant B (base-9→base-11)?
     Positive ⇒ transferable procedure; ~0 ⇒ task-specific recitation.
3. **An empirical test of the user's "fine-tune it back" probe** (Exp 3, GPU): we internalize a
   counterfactual skill into a small open model's weights (LoRA) and test whether that reproduces the
   in-context gain and whether it *transfers* — showing what the fine-tune probe can and cannot resolve.

### Falsifiable Predictions
- **Elicitation regime:** high baseline, small g, **high SG**, gain collapses under length control. → skill = prompt-opt.
- **Novelty regime:** baseline collapses (replicating Reasoning-or-Reciting), **large g** from the correct skill,
  **low SG** (the model cannot self-generate the new procedure), **positive LCG**, **positive TG**. → skill injects
  genuinely-new transferable procedure — but current *self-evolving* (self-generated) skills stay at the
  elicitation floor **precisely because SG is low** (you cannot self-evolve knowledge you don't have).

The unifying claim: **self-evolution is bounded by self-generation.** It equals elicitation whenever the
needed procedure is latent; genuine novelty requires an *external* source of the new procedure (a human
skill, or a verifier+search loop that discovers it), which self-introspection alone cannot supply.

### Experiment Justification
- **Exp 1 (core, real API):** Establishes the four END quantities across both regimes and ≥2 frontier models.
  Needed to demonstrate the discriminator *works* and to answer the question quantitatively.
- **Exp 2 (self-evolution loop, real API):** A Voyager-style iterative skill-refinement loop with verifier
  feedback. Tests whether *actual self-evolution* (not one-shot self-generation) ever crosses the SG barrier
  in the novelty regime. Directly tests the "self-evolving creates new ideas?" claim.
- **Exp 3 (internalization probe, GPU):** Implements and stress-tests the user's proposed fine-tune-back
  probe on Qwen2.5-7B. Needed because the user explicitly flagged it as an unresolved solution.

---

## Hypothesis Decomposition
- **H1 (regime effect):** Adding the *correct* skill yields larger normalized gain in the novelty regime
  than in the elicitation regime (a skill can only help a lot where latent knowledge is missing).
- **H2 (self-generation gap):** SG is high in the elicitation regime and low in the novelty regime.
- **H3 (content vs. length):** In the novelty regime, LCG > 0 (real skill beats length-matched filler);
  in the elicitation regime LCG ≈ 0.
- **H4 (transfer):** Correct skills show positive TG (transfer to unseen counterfactual variants);
  self-generated skills in the novelty regime do not.
- **H5 (self-evolution ceiling):** Iterative self-evolution in the novelty regime plateaus below the
  correct-skill ceiling and near the no-skill floor when the base model cannot self-generate the procedure.
- **H6 (internalization):** Fine-tuning the skill into weights reproduces part of the in-context gain but
  does not by itself distinguish elicitation from novelty (transfer to unseen variants is the tell).

## Proposed Methodology

### Approach
Controlled counterfactual experiments with **real LLM APIs** (via OpenRouter) + a small-model
**LoRA internalization** probe on local GPU. Deterministic, computable verifiers throughout — no
LLM-as-judge for the primary signals (per HindSight caveat).

### Tasks / Substrate
1. **Base-b arithmetic** (multi-digit addition/multiplication) — b ∈ {10 (elicitation), 9, 11 (novelty)}.
   Following *Reasoning or Reciting?* (base-9 is the canonical counterfactual). Answers computed exactly.
2. **Permuted-alphabet Caesar/substitution cipher decoding** — identity map (elicitation) vs. a fixed
   random permutation (novelty). Exact string-match verifier.
(Optional supplement: measure the Self-Generation gap on real **SkillsBench** human SKILL.md artifacts.)

### Conditions (base model fixed)
(a) No-Skill baseline · (b) Correct-Skill (explicit procedure) · (c) Self-Generated-Skill (model writes its
own, then reused on held-out items) · (d) Length-matched irrelevant-content control · (e) Self-Evolved-Skill
after K verifier-feedback iterations (Exp 2).

### Baselines
No-Skill prompting; length-matched control (Promptbreeder/prompt-opt stand-in); Correct-Skill as the
*upper reference* (the "external new procedure" oracle, AlphaEvolve-style verifier ground truth).

### Evaluation Metrics
Deterministic accuracy; normalized gain g; Self-Generation rate SG; Length-Controlled Gain LCG;
Transfer Gain TG; self-evolution learning curve. Report mean ± bootstrap 95% CI.

### Statistical Analysis Plan
- Per-cell N ≥ 100 problems; ≥2 models. Two-proportion z-tests / bootstrap for accuracy deltas.
- Effect sizes (Cohen's h for proportions). Bonferroni across the main hypothesis family.
- Seeds fixed (42); temperature 0 for determinism where supported; 3 seeds for self-generation stochastics.

## Expected Outcomes
Support for H1–H5 would show: skills *can* inject genuine novelty (novelty regime, correct skill), but
self-evolution/self-generation stays at the elicitation floor because it cannot produce knowledge the model
lacks. This is a crisp, falsifiable answer to the user's question with a reusable protocol.

## Timeline & Milestones
1. Env + harness (done/early). 2. Exp 1 core runs. 3. Exp 2 self-evolution loop. 4. Exp 3 GPU probe.
5. Analysis + figures. 6. REPORT.md + README.md. (~20-30% buffer for API/GPU debugging.)

## Potential Challenges
- **Model refuses/format drift** → strict parsing + regex extraction + retries.
- **API cost/rate limits** → temperature 0, caching to disk, modest N with adequate power.
- **Contamination** (model has seen base-9 discussions) → that *strengthens* the test: if it still can't do
  it, the novelty gap is real; we also use a *fresh* random cipher permutation per seed.
- **GPU contention** (2/4 A6000 busy) → use a single free GPU, 7B + LoRA, small N.

## Success Criteria
A working END harness; quantitative values for g, SG, LCG, TG across regimes/models with CIs; a clear,
statistically-supported verdict on when skills cross from elicitation to novelty; complete REPORT.md.
