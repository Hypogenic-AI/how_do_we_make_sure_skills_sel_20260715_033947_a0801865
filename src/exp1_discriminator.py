"""Experiment 1 — The Elicitation-Novelty Discriminator (END).

For each (model, task-setting) we evaluate a held-out test set under 4 conditions:
  (a) no_skill        - task instruction only            -> baseline floor
  (b) correct_skill   - explicit correct procedure/key   -> external-oracle upper ref
  (c) self_gen_skill  - the model's OWN skill, written unaided from the task
                        description (no answers), then reused on the test set
  (d) length_control  - irrelevant filler matched to (b)'s length

Task settings span the elicitation<->novelty axis:
  arith_b10  : base-10 3-digit multiply    (ELICITATION control: latent+reliable)
  arith_b9   : base-9  3-digit multiply    (NOVELTY-procedure: latent concept, unreliable)
  arith_b11  : base-11 3-digit multiply    (NOVELTY-procedure; also used for transfer)
  vocab_es   : Spanish->English glossary   (ELICITATION control: words already latent)
  vocab_zeth : invented "Zeth"->English    (NOVELTY-information: glossary NOT in the model)

Metrics: accuracy, normalized gain g=(acc-acc0)/(1-acc0), LCG=g_b-g_d, SG-utility=g_c/g_b.
All verification is deterministic (src/tasks.verify). Real APIs via OpenRouter.
"""
import os
import sys
import json
import pathlib
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(__file__))
import tasks as T
from llm_client import chat

RESULTS = pathlib.Path(__file__).resolve().parent.parent / "results"
RESULTS.mkdir(exist_ok=True)

N_TEST = 60          # held-out test items per setting
N_SELFGEN_CTX = 0    # zero-feedback self-generation (no example answers shown)
MODELS = ["claude-sonnet-4.5", "gpt-4.1"]
MAX_WORKERS = 12

OP = "mul"          # multiplication: hard enough to leave room above the ceiling
N_DIGITS = 3

TASK_INSTRUCTION = {
    "arithmetic": "Solve the following arithmetic problem. The numbers are written "
                  "in base {base}. Give ONLY the final answer, in base {base}.",
    "vocab": "Translate the following {lang} phrase into English. "
             "Output ONLY the English translation (one word per source word).",
}


def build_settings():
    """Return dict setting_name -> (test_items, correct_skill_text, meta)."""
    settings = {}
    # arithmetic bases (multiplication, 3-digit)
    for b in (10, 9, 11):
        items = T.gen_arithmetic(b, N_TEST, seed=42, op=OP, n_digits=N_DIGITS)
        settings[f"arith_b{b}"] = {
            "items": items,
            "correct_skill": T.arithmetic_skill(b, op=OP),
            "family": "arithmetic", "base": b, "lang": None,
            "regime": "elicitation" if b == 10 else "novelty_procedure",
        }
    # vocabulary: Spanish (latent) vs invented Zeth (novel info)
    settings["vocab_es"] = {
        "items": T.gen_vocab(T.SPANISH, N_TEST, seed=42, n_words=4),
        "correct_skill": T.vocab_skill(T.SPANISH, "Spanish"),
        "family": "vocab", "base": None, "lang": "Spanish", "regime": "elicitation",
    }
    zeth = T.make_glossary(seed=123)
    settings["vocab_zeth"] = {
        "items": T.gen_vocab(zeth, N_TEST, seed=42, n_words=4),
        "correct_skill": T.vocab_skill(zeth, "Zeth"),
        "family": "vocab", "base": None, "lang": "Zeth",
        "regime": "novelty_information", "glossary": zeth,
    }
    return settings


def instruction_for(setting):
    if setting["family"] == "arithmetic":
        return TASK_INSTRUCTION["arithmetic"].format(base=setting["base"])
    return TASK_INSTRUCTION["vocab"].format(lang=setting["lang"])


def make_prompt(setting, item, skill_text=None):
    instr = instruction_for(setting)
    parts = []
    if skill_text:
        parts.append(skill_text)
        parts.append("")
    parts.append(instr)
    parts.append("")
    parts.append(f"Problem: {item['question']}")
    return "\n".join(parts)


def self_generate_skill(setting, model):
    """Ask the model to write its OWN skill for this task, unaided (no answers).
    This measures whether the required knowledge is already latent (elicitable)."""
    instr = instruction_for(setting)
    if setting["family"] == "arithmetic":
        ex = ", ".join(it["question"] for it in setting["items"][:3])
        prompt = (
            f"You will repeatedly solve tasks of this type:\n{instr}\n"
            f"Example problems (no answers given): {ex}\n\n"
            "Write a concise, reusable SKILL (a procedure/cheat-sheet) that a solver "
            "should follow to get these right every time. Output ONLY the skill text."
        )
    else:
        ex = ", ".join(it["question"] for it in setting["items"][:3])
        prompt = (
            f"You will repeatedly solve tasks of this type:\n{instr}\n"
            f"Example phrases to translate (no translations given): {ex}\n\n"
            "Write a concise, reusable SKILL (a glossary/cheat-sheet) that a solver "
            "should follow to translate such phrases every time. Output ONLY the skill text."
        )
    return chat([{"role": "user", "content": prompt}], model=model,
                temperature=0.0, max_tokens=800)


def run_condition(setting, model, condition, skill_text):
    """Run all test items under one condition; return per-item correctness list."""
    items = setting["items"]

    def _one(item):
        prompt = make_prompt(setting, item, skill_text=skill_text)
        resp = chat([{"role": "user", "content": prompt}], model=model,
                    temperature=0.0, max_tokens=1200)
        return T.verify(item, resp)

    results = [None] * len(items)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(_one, it): i for i, it in enumerate(items)}
        for f in as_completed(futs):
            results[futs[f]] = f.result()
    return results


def main():
    settings = build_settings()
    all_rows = []
    selfgen_texts = {}

    for model in MODELS:
        for sname, setting in settings.items():
            print(f"[{model}] {sname} ({setting['regime']})", flush=True)
            correct = setting["correct_skill"]
            filler = T.length_matched_filler(correct, seed=7)
            selfgen = self_generate_skill(setting, model)
            selfgen_texts[f"{model}|{sname}"] = selfgen

            conditions = {
                "no_skill": None,
                "correct_skill": correct,
                "self_gen_skill": selfgen,
                "length_control": filler,
            }
            for cond, skill in conditions.items():
                res = run_condition(setting, model, cond, skill)
                acc = sum(res) / len(res)
                all_rows.append({
                    "model": model, "setting": sname, "regime": setting["regime"],
                    "condition": cond, "n": len(res), "n_correct": sum(res),
                    "accuracy": acc, "correct_flags": res,
                })
                print(f"    {cond:16s} acc={acc:.3f}", flush=True)

    out = RESULTS / "exp1_results.json"
    out.write_text(json.dumps(all_rows, indent=2))
    (RESULTS / "exp1_selfgen_skills.json").write_text(json.dumps(selfgen_texts, indent=2))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
