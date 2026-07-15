# Literature Review

**Research question.** *How do we make sure Skills / Self-Evolving agents are creating something NEW (e.g., inventing General Relativity or the Attention mechanism) rather than finding a better way to EXPLOIT / elicit what LLMs have already learned (i.e., glorified prompt optimization)?*

This review synthesizes 20 papers (downloaded in `papers/`, per-paper structured notes in `notes_compact.json`). It (1) frames the question as an empirical one, (2) reports where the evidence currently falls, (3) catalogs the *measurement methods* that can actually distinguish the two hypotheses, and (4) recommends a concrete experimental design.

---

## 1. Research Area Overview

Two literatures collide here:

- **The "self-evolving" / "skills" systems literature** — agents that accumulate reusable procedural knowledge (Voyager's executable-code skill library [1]; the SKILL.md "agent skills" abstraction [3]), or that rewrite their own code/prompts (Darwin Gödel Machine [4], Promptbreeder [7], the AI Scientist [5], AlphaEvolve [9]). The self-evolving-agents survey [2] formalizes this as a feedback loop (System Inputs → Agent → Environment → Optimiser) and a paradigm shift from offline pretraining toward online, lifelong self-modification. Schmidhuber's Gödel Machine [6] is the theoretical ideal these approximate.
- **The "is it real?" evaluation literature** — work that asks whether apparent gains reflect *new* capability or *re-packaged latent* capability: counterfactual evaluation [16], the "emergent abilities are a metric mirage" critique [17], "LLMs cannot self-correct reasoning yet" [18], and STaR-style self-bootstrapping [19]. Plus a fast-growing *idea-novelty* measurement strand [10, 11, 12].

The central conceptual point, made explicit by several papers, is that **most self-evolving methods are black-box wrappers around a frozen base model.** They add no weights and no new training signal beyond what the environment/verifier provides; they *search over the model's existing behavior manifold*. Voyager's own authors describe it as harnessing "the world knowledge encapsulated in pre-trained LLMs" and as "an in-context form of novelty search" [1]. This makes the hypothesis — *elicitation/prompt-optimization vs. genuine novelty* — a testable empirical claim, not a matter of definition.

**Where the evidence falls (this corpus).** Coding each paper by which side of the question it supports: **13/20 argue elicitation / prompt-optimization**, **4/20 mixed-or-neutral**, **2/20 provide a measurement method**, and **only 1/20 (AlphaEvolve) argues for genuine, verifiable novelty** — and even that one owes its novelty to a ground-truth evaluator, not to the LLM inventing concepts unaided (see §4).

---

## 2. The case that self-evolving ≈ better elicitation / prompt optimization

The strongest evidence is from **controlled, paired ablations that hold the base model fixed** and vary only the "skill"/"evolution" component.

- **SkillsBench [14]** — 87 expertise-heavy, containerized tasks with deterministic verifiers, run *with vs. without* Skills across 18 agent×model configurations. **Human-curated** Skills raise average pass rate **33.9% → 50.5% (+16.6 pp)**, improving all 18 configs. But **agent self-generated Skills fall *below* the no-Skills baseline** (−8.1 to −11.5 pp). The lift comes from *human* procedural knowledge injected into context, not from anything the agent discovered about itself. There is **no novelty/discovery metric** — it measures task resolution only.
- **Library Drift [13]** — self-evolving skill libraries silently degrade ("drift") without outcome-driven lifecycle governance. Even the *working* fix produces skills that are re-elicited procedural shortcuts; the **no-injection ablation floor is +0.002 ± 0.005** (i.e., an accumulating bank of 42 skills that are never injected adds essentially nothing), and over-aggressive retirement drops *below* baseline. Gains depend entirely on outcome-driven curation, not on emergent capability.
- **"Do LLM-Generated Skills Make Better AI Data Scientists?" [15]** — a clean component ablation: across 5 conditions the total spread is **1.2 pp** (No-Skill 68.3% vs Full 67.5%), **none significant (all p ≥ 0.396)**, and Full is **statistically indistinguishable from a token-matched Length-Control** of irrelevant content (+0.6 pp, p=0.775). On some stages the LLM skill *harms* (Data Preparation 57.1→50.8). Low-curation LLM-generated skills add no reliable value over the base prompt.
- **Inefficiencies of Meta-Agents [8]** — ADAS-style meta-agents that "design" new agents **do not meaningfully learn from prior designs**, produce behaviorally near-duplicate agents, and the best "evolved" agent often only ties the best *initial-library* agent (MMLU: Initial 62.8 vs 66–68). Meta-designed agents are always *costlier* at inference, and break-even vs. human-designed agents only occurs at ~15,000 examples. The "evolution" mostly reshuffles known building blocks.
- **Promptbreeder [7]** is the honest limiting case: it is explicitly *prompt search*. It improves benchmarks by evolving task-prompts and mutation-prompts — a better way to elicit CoT, not a new reasoning capability.
- **Voyager [1]** — "novelty" is operationalized as *count of unique Minecraft items* within a fixed, human-designed tech tree (novel *to the agent*, not to human knowledge). Swapping GPT-4→GPT-3.5 yields **5.7× fewer items**: the ceiling is set by the base model's latent coding knowledge. Hallucinated non-existent items are logged as failures — the agent operates strictly *within* its learned distribution.

**Foundational reasons this is expected:**
- **Reasoning or Reciting? [16]** — on *counterfactual* task variants (same procedure, altered input–output mapping, e.g. base-9 arithmetic) performance collapses across 11 tasks / 4 models, while comprehension checks stay high. Much apparent "reasoning" is memorized recitation — so eliciting it harder ≠ new reasoning.
- **Emergent Abilities: A Mirage? [17]** — sharp "new" capabilities are often artifacts of discontinuous metrics + small test sets. A claimed novel ability can vanish under a linear metric. Any "self-evolving unlocked a new skill" claim must rule this out.
- **LLMs Cannot Self-Correct Reasoning Yet [18]** — *without an external signal*, self-refinement does **not** add correctness and often degrades it; prior gains trace to oracle labels or unfair compute comparisons. This is the crux: self-evolution needs an **external ground-truth signal** to add anything real.
- **STaR [19]** — even self-improvement that *does* work (bootstrapping correct rationales) is gated by an external correctness filter; it amplifies rationales the model can already produce and verify, i.e. elicitation under a verifier.

---

## 3. The idea-novelty strand: novel-*sounding* ≠ novel-*and-real*

- **Can LLMs Generate Novel Research Ideas? [10]** — in a blind study with 100+ NLP researchers, a simple RAG + overgenerate-and-rank agent produces ideas rated **more novel** than experts' (5.64 vs 4.84, p<0.01). *But* the same system hits a **diversity ceiling** (~200 unique ideas out of 4000 generated) and is slightly *less feasible* — consistent with recombination of retrieved knowledge rather than genuine conceptual leaps.
- **Nova [11]** raises measured novelty via iterative retrieval-planning — i.e., novelty scales with *how much external knowledge you feed in*, which is elicitation-by-retrieval, not intrinsic invention.
- **HindSight [12]** exposes *why* novelty ratings are untrustworthy: LLM-as-judge (and humans) reward novel-*sounding* framing that is **negatively correlated** with real future impact (Spearman ρ = −0.29 between its impact-grounded metric and judged Novelty). This is a direct warning: subjective novelty scores can be *inversely* related to genuine discovery.

---

## 4. The one clear counterexample — and what makes it different

**AlphaEvolve [9]** is the corpus's strongest case for *genuine, verifiable* new results: a rank-48 decomposition for 4×4 complex matrix multiplication (**first improvement over Strassen's 49 in 56 years**), improvements on 14 matrix-multiplication targets, surpassing best-known bounds on ~20% of 50+ open math problems (e.g., kissing number in 11D: 592→593), and deployed engineering wins (0.7% of Google's fleet-wide compute recovered; 23% matmul-kernel speedup).

Crucially, its novelty does **not** come from the LLM introspecting or "self-improving" in a vacuum. It comes from **(a) evolutionary search over program space + (b) an automated, ground-truth evaluator** (tensor rank + provable correctness) that provides an *external, non-gameable* signal. This aligns exactly with [18]: the LLM proposes; a verifier disposes; search does the discovery. The lesson for "how do we make sure": **genuine novelty appears where there is a cheap, correct, external verifier and a real search loop — not where an agent merely reorganizes its context.**

**Governance caveats.** The **Red Queen Gödel Machine [20]** and **Library Drift [13]** warn that once you *learn* the evaluator or let the library self-curate, you get **verifier-hacking / drift**: the agent optimizes the measurement, not the target. The **Gödel Machine [6]** ideal (only self-modify on a *proof* of improvement) and the AI Scientist [5] (which auto-generates *and* auto-reviews its own papers — a conflicted signal at ~$15/paper) bracket the design space between "provably grounded" and "self-graded and gameable."

---

## 5. Measurement methods — how to actually "make sure" (the deliverable)

The literature offers four complementary discriminators between *elicitation* and *novelty*:

1. **Paired with/without ablation, base model fixed** [14, 15] — the minimal control. If a "skill" helps, it must beat the *no-skill* baseline **and** a **token/length-matched irrelevant-content control** [15] (else you are only measuring context length).
2. **Human-curated vs. self-generated skill** [14] — if self-generated skills match no-skill (or lose), the system is not *discovering*; it is *re-eliciting* what humans already encode.
3. **Contamination / recitation controls** — counterfactual task variants [16] and **time-split / future-impact** evaluation [12] separate memorized-and-recited from transferable-and-new. Any novelty claim must survive a held-out, post-cutoff, or counterfactual split.
4. **Metric-artifact controls** [17] — report linear/continuous metrics and adequate test-set sizes so a "new ability" cannot be a thresholding illusion.
5. **External, non-gameable verifier + search** [9, 18, 19] vs. self-grading [5, 20] — attribute any gain to the *verifier + search*, and audit for verifier-hacking/drift [13, 20].

---

## 6. Recommendations for Our Experiment

**Recommended substrate: SkillsBench [14]** (cloned to `code/skillsbench/`, 87 tasks, 8 domains, deterministic verifiers, oracle solutions). It already ships the paired with/without-Skills harness and the crucial datum that human-curated skills help (+16.6 pp) while self-generated ones do not — the perfect testbed for the core question.

**Core experiment — a "novelty vs. elicitation" discriminator on SkillsBench:**
- **Conditions (fixed base model):** (a) No-Skill baseline; (b) Human-curated skill; (c) Agent self-generated skill; (d) **Length-matched irrelevant-content control** [15]; (e) self-generated skill after outcome-driven curation [13].
- **Primary metric:** deterministic-verifier pass rate + **normalized gain** g = (pass_skill − pass_vanilla)/(1 − pass_vanilla) [14]. Focus on the **17 "informative" tasks** (30–80% baseline) where the skill can actually move the needle [15].
- **Discriminating tests:** (i) does (c) beat (a) *and* (d)? (ii) does (c) close the gap to (b), or plateau at the elicitation floor? (iii) **transfer probe** — construct held-out / counterfactual [16] or post-cutoff [12] task variants and check whether the skill encodes *transferable new procedure* vs. task-specific recitation.
- **Attribution / anti-hacking:** track skill-invocation rate [14], per-skill contribution c(s) and drift verdicts [13]; use a *frozen external* verifier and explicitly test for verifier-hacking [20].

**Recommended baselines:** No-Skill prompting; human-curated Skills [14]; Promptbreeder-style prompt optimization [7] (to show "just prompt-opt"); AlphaEvolve-style search-with-verifier [9] as the *genuine-novelty upper reference* where a task admits a ground-truth evaluator.

**Recommended metrics:** verifier pass rate, normalized gain g, length-control delta, transfer/counterfactual gap, contribution/drift diagnostics — and **explicitly avoid LLM-as-judge novelty scores as a primary signal**, since [12] shows they anti-correlate with real impact.

**Expected contribution.** A crisp, reusable protocol (and result) showing *under what conditions* self-evolving skills cross from re-eliciting latent knowledge to encoding genuinely new, transferable procedure — with the falsifiable prediction (from [13, 14, 15]) that **without an external ground-truth verifier and real search, self-generated skills sit at the elicitation floor (≈ no-skill), and only human curation or verifier-driven search (AlphaEvolve-style) produces gains that survive length-control and transfer tests.**

---

## References (see `papers/README.md` for files & arXiv IDs)

[1] Voyager (2305.16291) · [2] Self-Evolving Agents Survey (2508.07407) · [3] Agent Skills Survey (2602.12430) · [4] Darwin Gödel Machine (2505.22954) · [5] The AI Scientist (2408.06292) · [6] Gödel Machines (cs/0309048) · [7] Promptbreeder (2309.16797) · [8] Inefficiencies of Meta-Agents (2510.06711) · [9] AlphaEvolve (2506.13131) · [10] Can LLMs Generate Novel Ideas? (2409.04109) · [11] Nova (2410.14255) · [12] HindSight (2603.15164) · [13] Library Drift (2605.19576) · [14] SkillsBench (2602.12670) · [15] LLM-Skills Data-Scientist Ablation (2607.07504) · [16] Reasoning or Reciting? (2307.02477) · [17] Emergent Abilities: Mirage? (2304.15004) · [18] Cannot Self-Correct Yet (2310.01798) · [19] STaR (2203.14465) · [20] Red Queen Gödel Machine (2606.26294)
