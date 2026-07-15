# Skills / Self-Evolution: new knowledge, or better elicitation?

A controlled study that operationally distinguishes whether an LLM "skill" **injects genuinely-new,
transferable knowledge** or merely **re-elicits latent knowledge** (a dressed-up prompt optimization),
using real frontier models. Full write-up: **[REPORT.md](REPORT.md)**.

## Key findings

- **A skill injects genuine new knowledge only where the model cannot self-generate it.** On an
  invented-language glossary the correct skill lifts accuracy **0.00 → 1.00** (Claude Sonnet 4.5 &
  GPT-4.1, p < 10⁻⁴), while the model's *self-generated* skill and a length-matched control stay at
  **0.00**. Everywhere the model *could* self-generate the skill (Spanish, base-10 arithmetic), an
  irrelevant length-matched memo helped about as much — i.e. the skill was prompt-shaping, not teaching.
- **Self-evolution is bounded by what is latent.** A Voyager/STaR-style loop never moves the invented-
  language task off 0.00 under verifier-only (pass/fail) feedback across 5 iterations, but jumps to
  **0.85** once given ground-truth answers — discovery needs an *external* signal, not introspection.
- **Self-generated skills can hurt.** Claude's own base-9 multiplication skill dropped accuracy
  **0.85 → 0.25** (it articulated a harder, error-prone algorithm) — a concrete mechanism for SkillsBench's
  "self-generated skills fall below baseline."
- **The "fine-tune the skill back" probe is real but method-dependent.** Genuinely-new info can be written
  into a 7B model's weights and generalizes to held-out items (**1.00**, no skill in context) — but only
  via example-distillation; fine-tuning on the skill *document* internalizes nothing (**0.00**).

**Takeaway:** today's self-evolving skills are better *elicitation*, not knowledge creation. To "make
sure," measure **self-generability (SG)** and **length-controlled gain (LCG)** — don't trust novel-sounding
text.

## Reproduce

```bash
uv venv && source .venv/bin/activate
uv add openai numpy pandas matplotlib scipy "torch==2.5.1" "transformers==4.56.2" peft accelerate datasets
export OPENROUTER_KEY=...     # for Exp 1 & 2 (real API, cached to results/llm_cache/)

python src/exp1_discriminator.py     # Exp 1: the discriminator (5 settings x 2 models x 4 conditions)
python src/exp2_self_evolution.py    # Exp 2: self-evolution loop (verifier-only vs with-answer)
python src/exp3_internalization.py   # Exp 3: LoRA internalization probe (needs 1 GPU)
python src/analyze.py                # stats + figures
```

API calls are cached, so re-runs are deterministic and free. Temperature 0, seeds fixed at 42.

## Structure

```
planning.md              Pre-registered design (motivation, hypotheses, metrics)
REPORT.md                Full research report with results, analysis, limitations
src/
  llm_client.py          OpenRouter client with on-disk caching + retries
  tasks.py               Counterfactual tasks + deterministic verifiers (base-b arith, vocab)
  exp1_discriminator.py  Exp 1
  exp2_self_evolution.py Exp 2
  exp3_internalization.py Exp 3 (Qwen2.5-7B LoRA)
  analyze.py             Bootstrap CIs, z-tests, Cohen's h, figures
results/                 Raw JSON, exp1_summary.csv, discriminator.csv, llm_cache/
figures/                 exp1_condition_bars, exp1_discriminator_scatter, exp2_self_evolution, exp3_internalization
literature_review.md     20-paper synthesis (pre-gathered)
```

## The discriminator (END)

For a skill *S* on a fixed base model: **g** = normalized gain; **SG** = g(self-generated S)/g(correct S)
(→1 means latent/elicitable); **LCG** = g(S) − g(length-matched filler) (→ real content). A skill is
*genuine new knowledge* iff **SG ≈ 0, LCG > 0, and verifier-only self-evolution stalls** while an external
ground-truth signal unlocks it. See REPORT.md §3.2.
