"""LoRA fine-tune of a small chat model on dispatch decisions (loss on the commands only).

Usage: train/.venv/bin/python train/train_lora.py --data train/data/bc.jsonl --out train/out/bc-v1
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="Qwen/Qwen2.5-3B-Instruct")
ap.add_argument("--data", required=True, nargs="+")
ap.add_argument("--out", required=True)
ap.add_argument("--resume-adapter", default=None)
ap.add_argument("--epochs", type=float, default=1.0)
ap.add_argument("--max-examples", type=int, default=None)
ap.add_argument("--batch", type=int, default=4)
ap.add_argument("--accum", type=int, default=2)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--rank", type=int, default=16)
ap.add_argument("--max-len", type=int, default=3072)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--grad-checkpoint", action="store_true")
ap.add_argument("--max-steps", type=int, default=None)
ap.add_argument("--save-steps", type=int, nargs="*", default=[], help="also save the adapter at these steps (OUT/step-N)")
a = ap.parse_args()
random.seed(a.seed)
torch.manual_seed(a.seed)

tok = AutoTokenizer.from_pretrained(a.base)
tok.padding_side = "right"
rows = [json.loads(line) for f in a.data for line in open(f)]
random.shuffle(rows)
if a.max_examples:
    rows = rows[:a.max_examples]


def encode(row):
    msgs = row["messages"]
    prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
    p = tok(prompt, add_special_tokens=False)["input_ids"]
    t = tok(msgs[-1]["content"] + "<|im_end|>", add_special_tokens=False)["input_ids"]
    ids = (p + t)[-a.max_len:]
    labels = ([-100] * len(p) + t)[-a.max_len:]
    return ids, labels


data = [encode(r) for r in rows]
print(f"{len(data)} examples, mean {sum(len(d[0]) for d in data) // len(data)} tokens, max {max(len(d[0]) for d in data)}")

model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16, device_map={"": 0},
                                             attn_implementation="sdpa")
if a.grad_checkpoint:
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
if a.resume_adapter:
    model = PeftModel.from_pretrained(model, a.resume_adapter, is_trainable=True)
else:
    model = get_peft_model(model, LoraConfig(
        r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
model.print_trainable_parameters()
print("weights dtype", next(model.parameters()).dtype, flush=True)
core = model.get_base_model()
model.train()
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
steps_total = math.ceil(len(data) * a.epochs / a.batch)
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: min(1.0, (s + 1) / 30) * max(0.05, 1 - s / max(1, steps_total // a.accum)))

# batches of similar length to limit padding
order = sorted(range(len(data)), key=lambda i: len(data[i][0]))
batches = [order[i:i + a.batch] for i in range(0, len(order), a.batch)]
random.shuffle(batches)
batches = (batches * math.ceil(a.epochs))[:steps_total]
if a.max_steps:
    batches, steps_total = batches[:a.max_steps], min(steps_total, a.max_steps)
t0, running = time.time(), None
for step, idx in enumerate(batches, 1):
    width = max(len(data[i][0]) for i in idx)
    ids = torch.full((len(idx), width), tok.pad_token_id)
    lab = torch.full((len(idx), width), -100)
    att = torch.zeros((len(idx), width), dtype=torch.long)
    for r, i in enumerate(idx):
        x, y = data[i]
        ids[r, :len(x)] = torch.tensor(x)
        lab[r, :len(y)] = torch.tensor(y)
        att[r, :len(x)] = 1
    # Only the command tokens carry loss, so project just those positions to the vocabulary
    # (full-sequence logits over a 150k vocabulary would dominate memory).
    hidden = core.model(input_ids=ids.cuda(), attention_mask=att.cuda()).last_hidden_state[:, :-1]
    target = lab[:, 1:].cuda()
    keep = target != -100
    loss = torch.nn.functional.cross_entropy(core.lm_head(hidden[keep]).float(), target[keep])
    (loss / a.accum).backward()
    if step % a.accum == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
    running = loss.item() if running is None else 0.98 * running + 0.02 * loss.item()
    if step % 50 == 0 or step == steps_total or step == 5:
        el = time.time() - t0
        print(f"step {step}/{steps_total} loss {running:.4f} {el / step:.2f}s/step eta {el / step * (steps_total - step) / 60:.0f}min "
              f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f}GB", flush=True)
    if step % 1000 == 0:
        model.save_pretrained(a.out)
    if step in a.save_steps:
        model.save_pretrained(f"{a.out}/step-{step}")
model.save_pretrained(a.out)
Path(a.out, "train_args.json").write_text(json.dumps(vars(a), indent=1))
print(f"saved adapter to {a.out} in {(time.time() - t0) / 60:.1f} min")
