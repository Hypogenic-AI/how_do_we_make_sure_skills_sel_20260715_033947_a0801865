"""Experiment 2 — Self-evolution loop (Voyager/STaR-style) vs. the SG barrier.

An agent iteratively refines its OWN skill from experience + feedback, then we
measure the evolved skill on a held-out test set each iteration (learning curve).

We contrast two feedback signals (the crux from "LLMs Cannot Self-Correct Yet"
and STaR — self-improvement needs an EXTERNAL ground-truth signal):
  - verifier_only : the agent learns only pass/fail per training item.
  - with_answer   : the agent also sees the correct answer for each item (STaR-like).

Settings (fixed base model):
  arith_b9   : NOVELTY-procedure  (procedure latent; can self-evolution scaffold it?)
  vocab_zeth : NOVELTY-information (glossary absent; only ground-truth answers can reveal it)
  vocab_es   : ELICITATION control (already latent)

Prediction: for vocab_zeth, verifier_only cannot recover the glossary (info is not
latent and pass/fail leaks ~nothing); with_answer CAN (external signal reveals it) —
i.e. genuine novelty needs an external source, not introspection.
"""
import os
import sys
import json
import pathlib
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(__file__))
import tasks as T
from llm_client import chat
from exp1_discriminator import build_settings, instruction_for, make_prompt

RESULTS = pathlib.Path(__file__).resolve().parent.parent / "results"

MODEL = "claude-sonnet-4.5"
N_TRAIN = 18
N_TEST = 40
K_ITERS = 5
MAX_WORKERS = 12


def eval_skill(setting, skill_text, items, model):
    def _one(it):
        p = make_prompt(setting, it, skill_text=skill_text)
        r = chat([{"role": "user", "content": p}], model=model,
                 temperature=0.0, max_tokens=1200)
        return T.verify(it, r), r
    flags = [None] * len(items)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(_one, it): i for i, it in enumerate(items)}
        for f in as_completed(futs):
            flags[futs[f]] = f.result()
    return flags


def build_feedback(setting, train_items, flags, mode):
    """Compose experience feedback for the skill-update step."""
    lines = []
    for it, (ok, resp) in zip(train_items, flags):
        pred = (T.extract_arithmetic(resp, setting["base"]) if setting["family"] == "arithmetic"
                else T.extract_vocab(resp))
        status = "CORRECT" if ok else "WRONG"
        if mode == "verifier_only":
            lines.append(f"- Problem `{it['question']}` -> your answer `{pred}` : {status}")
        else:  # with_answer
            lines.append(f"- Problem `{it['question']}` -> your answer `{pred}` : {status}"
                         f" (correct answer: `{it['answer']}`)")
    return "\n".join(lines)


def update_skill(setting, prev_skill, feedback, model):
    instr = instruction_for(setting)
    prompt = (
        f"You are iteratively improving a reusable SKILL for this task:\n{instr}\n\n"
        f"Current skill:\n---\n{prev_skill or '(empty)'}\n---\n\n"
        f"Here is feedback from trying the current skill on practice problems:\n"
        f"{feedback}\n\n"
        "Revise the SKILL so a solver following it gets these right in future. "
        "You may add rules, a procedure, or a glossary of mappings you have inferred. "
        "Output ONLY the improved skill text."
    )
    return chat([{"role": "user", "content": prompt}], model=model,
                temperature=0.0, max_tokens=1200)


def run_setting(sname, setting, model, mode):
    # disjoint train/test splits
    all_items = setting["items"]
    train = all_items[:N_TRAIN]
    test = all_items[N_TRAIN:N_TRAIN + N_TEST]

    skill = ""
    curve = []
    # iteration 0 = no skill baseline on test
    base_flags = eval_skill(setting, None, test, model)
    curve.append({"iter": 0, "test_acc": sum(f[0] for f in base_flags) / len(test),
                  "skill_len": 0})
    for k in range(1, K_ITERS + 1):
        tr_flags = eval_skill(setting, skill or None, train, model)
        fb = build_feedback(setting, train, tr_flags, mode)
        skill = update_skill(setting, skill, fb, model)
        te_flags = eval_skill(setting, skill, test, model)
        acc = sum(f[0] for f in te_flags) / len(test)
        curve.append({"iter": k, "test_acc": acc, "skill_len": len(skill)})
        print(f"    [{mode}] {sname} iter {k}: test_acc={acc:.3f} skill_len={len(skill)}",
              flush=True)
    return {"setting": sname, "mode": mode, "model": model,
            "regime": setting["regime"], "curve": curve, "final_skill": skill}


def main():
    settings = build_settings()
    chosen = ["arith_b9", "vocab_zeth", "vocab_es"]
    modes = ["verifier_only", "with_answer"]
    # upper reference: correct-skill test accuracy for each setting
    out = []
    for sname in chosen:
        setting = settings[sname]
        test = setting["items"][N_TRAIN:N_TRAIN + N_TEST]
        ref_flags = eval_skill(setting, setting["correct_skill"], test, MODEL)
        ref_acc = sum(f[0] for f in ref_flags) / len(test)
        print(f"[ref] {sname} correct-skill test_acc={ref_acc:.3f}", flush=True)
        for mode in modes:
            r = run_setting(sname, setting, MODEL, mode)
            r["correct_skill_ref_acc"] = ref_acc
            out.append(r)
            (RESULTS / "exp2_results.json").write_text(json.dumps(out, indent=2))
    print("Saved exp2_results.json")


if __name__ == "__main__":
    main()
