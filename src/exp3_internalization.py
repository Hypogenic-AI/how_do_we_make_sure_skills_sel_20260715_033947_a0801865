"""Experiment 3 — The internalization probe (the user's proposed "fine-tune the
skill back into the weights" test), on a small open model (Qwen2.5-7B-Instruct).

The user proposed: fine-tune the skill into the model and see if it then solves the
task WITHOUT the skill in context; if it still fails, the skill was "just prompt
optimization." We implement this and show what it can/cannot resolve.

Conditions on a held-out TEST set (no skill in the prompt unless stated):
  base_no_skill      : base model, no skill                 -> does it already know? (SG proxy)
  base_in_context    : base model, correct skill in prompt  -> in-context ceiling
  ft_doc             : LoRA-tuned on the SKILL DOCUMENT text (continued-pretraining style)
  ft_examples        : LoRA-tuned on input->output EXAMPLES that teach the mapping/procedure
                       (fine-tune split is DISJOINT from test => measures generalization)

Substrate: vocab_zeth (novelty-INFORMATION) and arith_b9 (novelty-PROCEDURE).
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")  # a free A6000
import sys
import json
import pathlib
import random
import torch

sys.path.insert(0, os.path.dirname(__file__))
import tasks as T

RESULTS = pathlib.Path(__file__).resolve().parent.parent / "results"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
N_FT = 220          # fine-tune examples (disjoint from test)
N_TEST = 60
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import Dataset

_tok = None


def load_base():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map={"": 0})
    return model, tok


def chat_prompt(tok, user):
    return tok.apply_chat_template(
        [{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def batch_generate(model, tok, users, max_new=200, bs=16):
    outs = []
    model.eval()
    for i in range(0, len(users), bs):
        chunk = users[i:i + bs]
        prompts = [chat_prompt(tok, u) for u in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                  max_length=2048).to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        for j in range(len(chunk)):
            new = gen[j][enc["input_ids"].shape[1]:]
            outs.append(tok.decode(new, skip_special_tokens=True))
    return outs


def make_task(kind):
    """Return (family, instruction_fn, correct_skill, ft_items, test_items)."""
    if kind == "zeth":
        gloss = T.make_glossary(seed=123)
        skill = T.vocab_skill(gloss, "Zeth")
        ft = T.gen_vocab(gloss, N_FT, seed=1, n_words=4)
        test = T.gen_vocab(gloss, N_TEST, seed=999, n_words=4)
        instr = ("Translate the following Zeth phrase into English. "
                 "Output ONLY the English translation.")
    else:  # arith_b9
        b = 9
        skill = T.arithmetic_skill(b, op="mul")
        ft = T.gen_arithmetic(b, N_FT, seed=1, op="mul", n_digits=3)
        test = T.gen_arithmetic(b, N_TEST, seed=999, op="mul", n_digits=3)
        instr = (f"Solve the following arithmetic problem. The numbers are written "
                 f"in base {b}. Give ONLY the final answer, in base {b}.")
    return kind, instr, skill, ft, test


def user_text(instr, question, skill=None):
    s = (skill + "\n\n") if skill else ""
    return f"{s}{instr}\nProblem: {question}"


def eval_acc(model, tok, instr, test, skill=None):
    users = [user_text(instr, it["question"], skill) for it in test]
    resps = batch_generate(model, tok, users)
    correct = [T.verify(it, r) for it, r in zip(test, resps)]
    return sum(correct) / len(correct), resps


def build_ft_dataset(tok, instr, skill, ft_items, mode):
    """mode='doc': teach the skill document. mode='examples': teach input->output."""
    rows = []
    if mode == "doc":
        # Repeat the skill document as an assistant 'knowledge' turn a few times,
        # plus a handful of solved examples so the format is learned.
        for _ in range(20):
            msgs = [{"role": "user", "content": f"Recall your reference for this task:\n{instr}"},
                    {"role": "assistant", "content": skill}]
            rows.append(tok.apply_chat_template(msgs, tokenize=False))
        # a few worked examples (10) so it can produce answers in-format
        for it in ft_items[:10]:
            msgs = [{"role": "user", "content": user_text(instr, it["question"])},
                    {"role": "assistant", "content": str(it["answer"])}]
            rows.append(tok.apply_chat_template(msgs, tokenize=False))
    else:  # examples: input->output only (NO skill shown) -> distill knowledge into weights
        for it in ft_items:
            msgs = [{"role": "user", "content": user_text(instr, it["question"])},
                    {"role": "assistant", "content": str(it["answer"])}]
            rows.append(tok.apply_chat_template(msgs, tokenize=False))
    return Dataset.from_dict({"text": rows})


def tokenize_ds(ds, tok, max_len=512):
    def _tok(b):
        # right-pad for training (correct RoPE positions); mask pads in labels.
        out = tok(b["text"], truncation=True, max_length=max_len,
                  padding="max_length", padding_side="right")
        labels = []
        for ids, am in zip(out["input_ids"], out["attention_mask"]):
            labels.append([tid if a == 1 else -100 for tid, a in zip(ids, am)])
        out["labels"] = labels
        return out
    return ds.map(_tok, batched=True, remove_columns=["text"])


def finetune(base_model, tok, ds, tag):
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                      task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(base_model, lora)
    args = TrainingArguments(
        output_dir=f"/tmp/ft_{tag}", per_device_train_batch_size=4,
        gradient_accumulation_steps=2, num_train_epochs=3, learning_rate=2e-4,
        logging_steps=10, save_strategy="no", bf16=True, report_to=[])
    tds = tokenize_ds(ds, tok)
    Trainer(model=model, args=args, train_dataset=tds).train()
    model.eval()
    return model


def run_kind(kind):
    print(f"\n===== Exp3 internalization: {kind} =====", flush=True)
    _, instr, skill, ft, test = make_task(kind)
    results = {}

    base, tok = load_base()
    acc_ns, _ = eval_acc(base, tok, instr, test, skill=None)
    acc_ic, _ = eval_acc(base, tok, instr, test, skill=skill)
    results["base_no_skill"] = acc_ns
    results["base_in_context"] = acc_ic
    print(f"  base_no_skill={acc_ns:.3f}  base_in_context={acc_ic:.3f}", flush=True)

    for mode in ["doc", "examples"]:
        ds = build_ft_dataset(tok, instr, skill, ft, mode)
        ft_model = finetune(base, tok, ds, f"{kind}_{mode}")
        acc_ft, _ = eval_acc(ft_model, tok, instr, test, skill=None)  # NO skill in prompt
        results[f"ft_{mode}"] = acc_ft
        print(f"  ft_{mode} (no skill in prompt) test_acc={acc_ft:.3f}", flush=True)
        # unload adapter for next mode
        ft_model = ft_model.unload()
        del ft_model
        torch.cuda.empty_cache()

    del base
    torch.cuda.empty_cache()
    return {"kind": kind, "results": results,
            "n_ft": len(ft), "n_test": len(test), "model": MODEL_NAME}


def main():
    out = []
    for kind in ["zeth", "arith_b9"]:
        out.append(run_kind(kind))
        (RESULTS / "exp3_results.json").write_text(json.dumps(out, indent=2))
    print("Saved exp3_results.json")


if __name__ == "__main__":
    main()
