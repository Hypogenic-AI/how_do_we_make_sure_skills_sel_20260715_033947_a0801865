# Resources Catalog

## Summary
Resources gathered for: *How do we make sure Skills / Self-Evolving agents are creating something NEW rather than a better way to exploit what LLMs have already learned?*
**20 papers**, **3 cloned repos** (+ pointers to 4 more), **1 primary benchmark** (SkillsBench, with 2 optional datasets). Full synthesis in `literature_review.md`; per-paper structured notes in `notes_compact.json`.

## Papers (20)
Coded by which side of the core question each supports: **13 elicitation/prompt-opt · 4 mixed · 2 measurement-method · 1 genuine-novelty**.

| Title | Year | File | Stance / Key info |
|-------|------|------|-------------------|
| Voyager | 2023 | papers/2305.16291_Voyager.pdf | elicitation · skill library; "novelty"=unique Minecraft items; GPT-4→3.5 = 5.7× fewer |
| Self-Evolving Agents Survey | 2025 | papers/2508.07407_*.pdf | elicitation · taxonomy; feedback-loop framework; black-box wrappers on frozen model |
| Agent Skills Survey | 2026 | papers/2602.12430_*.pdf | elicitation · SKILL.md abstraction; human vs autonomous acquisition |
| Darwin Gödel Machine | 2025 | papers/2505.22954_*.pdf | elicitation · self-rewriting coding agent w/ empirical archive |
| The AI Scientist | 2024 | papers/2408.06292_*.pdf | mixed · end-to-end research + **self-review** (~$15/paper) — conflicted signal |
| Gödel Machines | 2003 | papers/cs0309048_*.pdf | mixed · theoretical: self-modify only on a *proof* of improvement |
| Promptbreeder | 2023 | papers/2309.16797_*.pdf | elicitation · explicit prompt search (task + mutation prompts) |
| Inefficiencies of Meta-Agents | 2025 | papers/2510.06711_*.pdf | elicitation · meta-agents don't learn; tie initial library; costlier |
| **AlphaEvolve** | 2025 | papers/2506.13131_*.pdf | **genuine novelty** · rank-48 4×4 matmul (beats Strassen 56 yrs); verifier+search |
| Can LLMs Generate Novel Ideas? | 2024 | papers/2409.04109_*.pdf | mixed · AI novelty 5.64>4.84 human but diversity ceiling ~200/4000 |
| Nova | 2024 | papers/2410.14255_*.pdf | elicitation · novelty scales with retrieved knowledge |
| HindSight | 2026 | papers/2603.15164_*.pdf | measurement · future-impact metric; judged novelty ρ=−0.29 vs impact |
| Library Drift | 2026 | papers/2605.19576_*.pdf | elicitation · no-injection floor +0.002; gains need outcome-driven curation |
| **SkillsBench** | 2026 | papers/2602.12670_*.pdf | elicitation · human Skills +16.6 pp; self-gen Skills **below** baseline |
| LLM-Skills DS Ablation | 2026 | papers/2607.07504_*.pdf | elicitation · 1.2 pp spread, all p≥0.396; = length-matched control |
| Reasoning or Reciting? | 2023 | papers/2307.02477_*.pdf | measurement · counterfactual tasks collapse → recitation |
| Emergent Abilities: Mirage? | 2023 | papers/2304.15004_*.pdf | elicitation · "new ability" = metric artifact |
| Cannot Self-Correct Yet | 2023 | papers/2310.01798_*.pdf | elicitation · no external signal → no gain (often worse) |
| STaR | 2022 | papers/2203.14465_*.pdf | elicitation · self-bootstrap gated by external correctness filter |
| Red Queen Gödel Machine | 2026 | papers/2606.26294_*.pdf | mixed · co-evolving evaluators → verifier-hacking risk |

See `papers/README.md` for authors, arXiv IDs, and grouping.

## Datasets / Benchmarks
| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| **SkillsBench** ⭐ | benchflow/skillsbench (HF + GitHub) | 87 tasks / ~495 MB | agentic tasks, deterministic verifiers | `code/skillsbench/tasks/` | paired with/without-Skills; primary substrate |
| AI-Researcher scores | NoviScl/AI-Researcher (+OSF) | small | idea novelty ratings | download on demand | optional, idea-novelty strand |
| counterfactual-eval | ZhaofengWu/counterfactual-evaluation | small | reasoning-vs-reciting | download on demand | optional, recitation control |

See `datasets/README.md` for download/loading instructions.

## Code Repositories
| Name | URL | Purpose | Location |
|------|-----|---------|----------|
| SkillsBench ⭐ | github.com/benchflow-ai/skillsbench | benchmark + verifiers + paired-Skills harness | `code/skillsbench/` |
| Voyager | github.com/MineDojo/Voyager | archetypal skill-library agent | `code/Voyager/` |
| AI-Scientist | github.com/SakanaAI/AI-Scientist | automated research pipeline | `code/AI-Scientist/` |

See `code/README.md` for run instructions, requirements, and inspected-but-not-cloned repos (AlphaEvolve results, AI-Researcher, counterfactual-evaluation).

## Resource Gathering Notes

### Search Strategy
Paper-finder service was **down** (HTTP 500) — fell back to the **arXiv API** (`arxiv_search.py`, kept in workspace) across ~13 targeted queries spanning the systems, the "just prompt-optimization?" skeptics, genuine-discovery evidence, novelty-measurement methods, and memorization/elicitation foundations; three canonical skeptical papers were pulled by exact ID. Repos found via the GitHub API. All 20 PDFs deep-read in parallel via a workflow (one reader-agent per paper) producing structured, question-oriented notes.

### Selection Criteria
Chosen to *span both sides* of the hypothesis and to include the papers with **controlled ablations** and **explicit measurement methods** — the only evidence that can actually distinguish elicitation from novelty. Prioritized recent (2024–2026) work plus the foundational skeptics (Reasoning-or-Reciting, Emergent-Mirage, Cannot-Self-Correct, STaR) and the one strong genuine-novelty case (AlphaEvolve).

### Challenges Encountered
- Paper-finder API unavailable (500) → arXiv fallback.
- arXiv keyword relevance is weak for negated/abstract queries (e.g., "reasoning or reciting") → resolved by direct arXiv-ID lookup.
- One old-format ID (`cs/0309048`) needed the category prefix.
- SkillsBench clone was 941 MB (446 MB `.git`) → pruned `.git` from all three repos; git-ignored the large dirs.

### Gaps and Workarounds
- **AlphaEvolve system** is not open-sourced (results only) → documented the results Colab; use its *design pattern* (external verifier + search) rather than its code.
- No single dataset directly labels "novel vs. elicited" → the contribution is a **protocol** (SkillsBench paired ablation + length control + transfer/counterfactual split) rather than a pre-labeled set.

## Recommendations for Experiment Design
1. **Primary substrate**: SkillsBench (`code/skillsbench/`) — paired with/without-Skills, deterministic verifiers, 87 tasks (focus on the 17 informative ones).
2. **Baselines**: No-Skill; human-curated Skill; agent self-generated Skill; **length-matched irrelevant-content control**; Promptbreeder-style prompt-opt; AlphaEvolve-style verifier+search as genuine-novelty upper reference.
3. **Metrics**: verifier pass rate, **normalized gain g**, length-control delta, transfer/counterfactual gap, skill-invocation rate, per-skill contribution/drift diagnostics. **Avoid LLM-as-judge novelty as a primary signal** (HindSight: it anti-correlates with real impact).
4. **Code to reuse**: SkillsBench harness (verifiers/oracles); counterfactual-evaluation for the recitation control; AI-Researcher scores if extending to idea-novelty.
5. **Falsifiable prediction**: without an external ground-truth verifier + real search, self-generated skills sit at the *elicitation floor* (≈ no-skill); only human curation or verifier-driven search yields gains surviving length-control and transfer tests.

See `literature_review.md` for full reasoning (§5 measurement methods, §6 experiment design).
