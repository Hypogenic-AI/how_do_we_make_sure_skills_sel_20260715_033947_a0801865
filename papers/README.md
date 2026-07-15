# Downloaded Papers

Curated for the research question: **How do we make sure Skills / Self-Evolving agents are creating something NEW (e.g., inventing Relativity or the Attention mechanism) rather than finding a better way to EXPLOIT / elicit what LLMs have already learned (i.e., glorified prompt optimization)?**

Papers are grouped by their role in the argument. See `../literature_review.md` for the synthesis.

## A. The systems under scrutiny (self-evolving agents & skill libraries)

1. **Voyager: An Open-Ended Embodied Agent with LLMs** — Wang et al., 2023. `2305.16291_Voyager.pdf` — arXiv:2305.16291.
   The archetypal self-evolving agent with an ever-growing skill library of executable code (Minecraft). Claims "novel discoveries."
2. **A Comprehensive Survey of Self-Evolving AI Agents** — 2025. `2508.07407_SelfEvolvingAgents_Survey.pdf` — arXiv:2508.07407.
   Taxonomy of agent-evolution techniques; framing and open problems.
3. **Agent Skills for LLMs: Architecture, Acquisition, Security, and the Path Forward** — 2026. `2602.12430_AgentSkills_Survey.pdf` — arXiv:2602.12430.
   Survey of the "skills" paradigm; human-authored vs LLM-acquired skills.
4. **Darwin Gödel Machine: Open-Ended Evolution of Self-Improving Agents** — 2025. `2505.22954_DarwinGodelMachine.pdf` — arXiv:2505.22954.
   Agent that empirically rewrites its own code and keeps an archive of variants.
5. **The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery** — Lu et al., 2024. `2408.06292_AI_Scientist.pdf` — arXiv:2408.06292.
   End-to-end automated research pipeline; a central "does it create new science?" case.
6. **Gödel Machines: Self-Referential Universal Problem Solvers** — Schmidhuber, 2003. `cs0309048_GodelMachines_Schmidhuber.pdf` — arXiv:cs/0309048.
   Theoretical foundation for provably-optimal self-improvement (the ideal the above approximate).

## B. Prompt / meta-agent optimization ("is it just prompt optimization?")

7. **Promptbreeder: Self-Referential Self-Improvement via Prompt Evolution** — Fernando et al., 2023. `2309.16797_Promptbreeder.pdf` — arXiv:2309.16797.
   Evolves prompts (and mutation-prompts) — the clearest case of "self-improvement = prompt search."
8. **Inefficiencies of Meta-Agents for Agent Design** — 2025. `2510.06711_Inefficiencies_MetaAgents.pdf` — arXiv:2510.06711.
   Skeptical study of meta-agents that design agents; where the "evolution" gains actually come from.

## C. Evidence FOR genuine discovery

9. **AlphaEvolve: A Coding Agent for Scientific and Algorithmic Discovery** — DeepMind, 2025. `2506.13131_AlphaEvolve.pdf` — arXiv:2506.13131.
   Strongest claim of verifiable *new* results (e.g., improved matrix-multiplication bounds) via LLM-driven evolution + evaluators.

## D. Measuring novelty vs. elicitation (core to "how do we make sure")

10. **Can LLMs Generate Novel Research Ideas? A Large-Scale Human Study with 100+ NLP Researchers** — Si et al., 2024. `2409.04109_CanLLMsGenerateNovelIdeas.pdf` — arXiv:2409.04109.
    Blinded human study; the reference point for measuring idea novelty.
11. **Nova: Iterative Planning and Search to Enhance Novelty and Diversity of LLM-Generated Ideas** — 2024. `2410.14255_Nova_NoveltyDiversity.pdf` — arXiv:2410.14255.
    How to push (and quantify) novelty/diversity.
12. **HindSight: Evaluating LLM-Generated Research Ideas via Future Impact** — 2026. `2603.15164_HindSight_FutureImpact.pdf` — arXiv:2603.15164.
    Time-split evaluation matching generated ideas to real future publications — a contamination-robust novelty metric.
13. **Library Drift: A Silent Failure Mode in Self-Evolving LLM Skill Libraries** — 2026. `2605.19576_LibraryDrift.pdf` — arXiv:2605.19576.
    Reports LLM-authored skills deliver **+0.0pp** vs human-curated **+16.2pp** (SkillsBench). Direct evidence on the core question.
14. **SkillsBench: Benchmarking How Well Agent Skills Work Across Diverse Tasks** — 2026. `2602.12670_SkillsBench.pdf` — arXiv:2602.12670.
    The benchmark (87 tasks, 8 domains, deterministic verifiers) — the experimental substrate.
15. **Do LLM-Generated Skills Make Better AI Data Scientists? A Component Ablation** — 2026. `2607.07504_LLMSkills_DataScientist_Ablation.pdf` — arXiv:2607.07504.
    Ablation isolating whether LLM-generated skills add value over base prompting.

## E. Foundations: novelty vs. memorization / elicitation

16. **Reasoning or Reciting? Exploring Capabilities and Limitations of LMs Through Counterfactual Tasks** — Wu et al., 2023. `2307.02477_ReasoningOrReciting.pdf` — arXiv:2307.02477.
    Counterfactual perturbations separate genuine reasoning from memorized recitation.
17. **Are Emergent Abilities of LLMs a Mirage?** — Schaeffer et al., 2023. `2304.15004_EmergentAbilities_Mirage.pdf` — arXiv:2304.15004.
    "New" capability can be a metric artifact — a caution about claiming novel abilities.
18. **Large Language Models Cannot Self-Correct Reasoning Yet** — Huang et al., 2023. `2310.01798_CannotSelfCorrect.pdf` — arXiv:2310.01798.
    Without an external signal, self-refinement does not add correctness — key to "verification."
19. **STaR: Bootstrapping Reasoning with Reasoning** — Zelikman et al., 2022. `2203.14465_STaR.pdf` — arXiv:2203.14465.
    Self-improvement by training on self-generated rationales (elicitation vs. new capability).
20. **The Red Queen Gödel Machine: Co-Evolving Agents and Their Evaluators** — 2026. `2606.26294_RedQueenGodelMachine.pdf` — arXiv:2606.26294.
    The moving-goalpost / verifier-hacking problem — why fixed verifiers mislead the "is it new?" test.
