# Datasets

Data files are **not** committed to git (see `.gitignore`). Follow the download instructions below.
For this research question the "dataset" is really a **benchmark of tasks + paired skills + deterministic verifiers**; the primary one (SkillsBench) is already present under `code/skillsbench/` from the repo clone.

---

## Dataset 1: SkillsBench  ⭐ PRIMARY

### Overview
- **Source**: https://huggingface.co/datasets/benchflow/skillsbench · repo: https://github.com/benchflow-ai/skillsbench
- **Size**: 87 tasks across 8 domains (Natural Science, Software Eng., Math/OR, Cybersecurity, Media, etc.); ~495 MB with task assets.
- **Format**: Per-task folders — `task.md`, `environment/` (Dockerfile + inputs), `oracle/` (reference solution), `verifier/` (deterministic pass/fail tests). Optional paired curated `Skills/`.
- **Task**: Agentic task completion (expertise-heavy workflows), scored by deterministic verifiers — **not** classification/generation.
- **Splits**: No train/test split; it is an evaluation suite. Paper convention: report over all 87, and focus analysis on the **17 "informative" tasks** with 30–80% baseline pass rate.
- **License**: See `code/skillsbench/LICENSE`.

### Download Instructions
Already available at `code/skillsbench/tasks/` (from the repo clone). To (re)fetch independently:

**Using HuggingFace:**
```python
from datasets import load_dataset
ds = load_dataset("benchflow/skillsbench")
```
**Or the full harness (recommended — includes verifiers/oracles):**
```bash
git clone https://github.com/benchflow-ai/skillsbench.git code/skillsbench
cd code/skillsbench && uv sync --locked
```

### Loading / Running
```bash
# validate a task, then run the oracle (must pass before any agent run)
bench tasks check code/skillsbench/tasks/citation-check
bench eval run --tasks-dir code/skillsbench/tasks/citation-check --agent oracle --sandbox docker
```
Task specs are plain files:
```python
spec = open("code/skillsbench/tasks/citation-check/task.md").read()
```

### Sample Data
See `samples/example_task_citation-check.md` (a full task spec copied for reference).
Task layout example (`tasks/citation-check/`): `task.md`, `environment/{Dockerfile,test.bib}`,
`oracle/{solve.sh,detect_fakes.py,crossref_snapshot.json}`, `verifier/{test.sh,test_outputs.py}`.

### Notes
- Requires **Docker** for the sandboxed environments; the `benchflow` CLI drives evaluation.
- Headline result to reproduce/build on: human-curated Skills **+16.6 pp** (33.9%→50.5%); agent self-generated Skills **below** no-Skills baseline (−8 to −11 pp).

---

## Dataset 2 (optional): AI-Researcher human review scores — for the idea-novelty strand
- **Source**: https://github.com/NoviScl/AI-Researcher (+ OSF cache https://osf.io/z6qa4)
- **Contents**: LLM- vs. human-written research ideas with blind expert review scores (Novelty/Excitement/Feasibility, 1–10) from the 100+ researcher study [10].
- **Use**: Only if the experiment extends to *idea* novelty; pair with **HindSight** [12]'s impact-grounded, contamination-robust metric rather than raw novelty ratings (which anti-correlate with real impact).
- **Download**:
  ```bash
  git clone https://github.com/NoviScl/AI-Researcher.git datasets/ai-researcher
  ```

## Dataset 3 (optional): Counterfactual evaluation — contamination/recitation control
- **Source**: https://github.com/ZhaofengWu/counterfactual-evaluation [16]
- **Contents**: 11 tasks with default vs. counterfactual variants + synthetic data + prompts/responses. Use to build the **transfer/recitation control** in our design (does a "skill" survive counterfactual perturbation?).
- **Download**:
  ```bash
  git clone https://github.com/ZhaofengWu/counterfactual-evaluation.git datasets/counterfactual-eval
  ```
