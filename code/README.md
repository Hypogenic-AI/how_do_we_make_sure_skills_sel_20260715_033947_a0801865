# Cloned Repositories

Reference implementations for the research question (novelty vs. elicitation in self-evolving skills).
`.git` directories were removed to save space; re-clone from the URLs below for history.
These directories are git-ignored (see root `.gitignore`) — they are large and locally available for the experiment runner.

## Repo 1: SkillsBench  ⭐ PRIMARY EXPERIMENTAL SUBSTRATE
- **URL**: https://github.com/benchflow-ai/skillsbench
- **HuggingFace**: https://huggingface.co/datasets/benchflow/skillsbench
- **Location**: `code/skillsbench/`  (~495 MB; task assets)
- **Purpose**: The first benchmark for "do agent Skills actually help?" — 87 expertise-heavy, containerized tasks across 8 domains, each with a **deterministic verifier** and an **oracle** reference solution. Ships the *paired with-Skills vs. no-Skills* harness that is exactly the control our experiment needs.
- **Key layout** (per task under `tasks/<name>/`):
  - `task.md` — task spec / prompt
  - `environment/` — `Dockerfile` + input files (sandbox)
  - `oracle/` — reference solution (`solve.sh`, helper scripts); must pass before agent runs
  - `verifier/` — deterministic tests (`test.sh`, `test_outputs.py`) → binary pass/fail
  - (paired) curated `Skills/` where present
- **Registry**: `registry.json` (v1.0/1.1, 87-task roster, per-task git commit + sha256 digest).
- **Run** (from the paper's Quick Start):
  ```bash
  uv tool install benchflow
  cd code/skillsbench && uv sync --locked
  bench tasks check tasks/offer-letter-generator      # validate
  bench eval run --tasks-dir tasks/citation-check --agent oracle --sandbox docker
  ```
- **Requirements**: Docker (sandboxed tasks), the `benchflow` CLI, and API keys for the agent model. Oracle runs need no external API.
- **Relevance**: Directly encodes the core finding — human-curated Skills +16.6 pp; **self-generated Skills fall below no-Skills baseline**. Use its verifiers + paired design for our novelty-vs-elicitation discriminator (see `literature_review.md` §6).

## Repo 2: Voyager
- **URL**: https://github.com/MineDojo/Voyager
- **Location**: `code/Voyager/`  (~6 MB)
- **Purpose**: The archetypal self-evolving agent with an ever-growing **executable-code skill library** (Minecraft). `skill_library/` holds example accumulated skills; `voyager/` is the agent; `installation/` documents the (heavy) MineDojo/Mineflayer setup.
- **Relevance**: Canonical "LLM writes reusable skills" reference. Its "novelty = unique items in a fixed game" framing is the definitional pitfall our metrics must avoid. Full run needs Minecraft + Node/Mineflayer + OpenAI keys — inspection-only is sufficient for us.

## Repo 3: AI-Scientist
- **URL**: https://github.com/SakanaAI/AI-Scientist
- **Location**: `code/AI-Scientist/`  (~154 MB)
- **Purpose**: End-to-end automated research pipeline (idea → code → experiments → paper → **auto-review**). `launch_scientist.py` is the entry point; `templates/` and `example_papers/` show outputs; `review_ai_scientist/` is the self-review module.
- **Relevance**: The strongest "does it create new science?" claim in the corpus, but its **self-graded** review signal is exactly the conflicted-verifier risk [Red Queen GM / Cannot-Self-Correct]. Reference for how *not* to close the evaluation loop.

## Not cloned (inspected via paper only)
- **AlphaEvolve** — system not open-sourced; only *results* released:
  https://colab.research.google.com/github/google-deepmind/alphaevolve_results/blob/master/mathematical_results.ipynb
- **Promptbreeder** — no official code; several community reimplementations exist.
- **AI-Researcher** (idea-novelty study [10]) — https://github.com/NoviScl/AI-Researcher (agent + all human review scores).
- **counterfactual-evaluation** [16] — https://github.com/ZhaofengWu/counterfactual-evaluation (code + synthetic data + prompts/responses).
