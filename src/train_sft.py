"""LoRA SFT on a recipe's dataset (loss on the completion only).

Usage: python src/train_sft.py --recipe R3 --langs en ar [--seed 1]
Saves the adapter to outputs/adapters/<recipe>_<langs>_s<seed>/
"""
import argparse
import random
import time

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from common import ROOT, load_config, read_jsonl
from sample import load_model


def collate(tok, max_len):
    def fn(batch):
        ids, labels = [], []
        for ex in batch:
            p = tok(ex["prompt"], add_special_tokens=True)["input_ids"]
            c = tok(" " + ex["completion"], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
            x = (p + c)[:max_len]
            y = ([-100] * len(p) + c)[:max_len]
            ids.append(x)
            labels.append(y)
        m = max(len(x) for x in ids)
        pad = tok.pad_token_id
        input_ids = torch.tensor([x + [pad] * (m - len(x)) for x in ids])
        lab = torch.tensor([y + [-100] * (m - len(y)) for y in labels])
        attn = torch.tensor([[1] * len(x) + [0] * (m - len(x)) for x in ids])
        return {"input_ids": input_ids, "attention_mask": attn, "labels": lab}
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", required=True, choices=["R1", "R2", "R3", "R4"])
    ap.add_argument("--langs", nargs="+", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="use only N examples (quick timing/pipeline test)")
    ap.add_argument("--bs", type=int, default=0, help="override training.batch_size (grad_accum is adjusted to keep the effective batch)")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    tc = dict(cfg["training"])
    if args.bs:
        tc["grad_accum"] = max(1, tc["batch_size"] * tc["grad_accum"] // args.bs)
        tc["batch_size"] = args.bs
    torch.manual_seed(args.seed)

    data = []
    for lang in args.langs:
        data += read_jsonl(ROOT / cfg["paths"]["processed"] / args.recipe / f"{lang}.jsonl")
    if args.limit:
        random.Random(args.seed).shuffle(data)
        data = data[: args.limit]
    print(f"{args.recipe}: {len(data)} training examples from {args.langs}")

    tok, model = load_model(cfg)
    tok.padding_side = "right"
    # a prompt+completion longer than max_seq_len would lose its completion (no loss signal): check up front
    n_cut = sum(len(tok(e["prompt"])["input_ids"]) + len(tok(" " + e["completion"], add_special_tokens=False)["input_ids"]) + 1
                > tc["max_seq_len"] for e in data)
    print(f"examples longer than max_seq_len={tc['max_seq_len']}: {n_cut}")
    assert n_cut == 0, "raise training.max_seq_len in config.yaml"
    if cfg["model"].get("load_in_4bit"):
        model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(
        r=tc["lora_r"], lora_alpha=tc["lora_alpha"], lora_dropout=tc["lora_dropout"],
        target_modules=tc["target_modules"], task_type="CAUSAL_LM"))
    # fp16 base model: keep the LoRA weights in fp32 and use a loss scaler (T4 has no bf16)
    for p_ in model.parameters():
        if p_.requires_grad:
            p_.data = p_.data.float()
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["model"]["dtype"] == "float16")
    model.print_trainable_parameters()
    model.train()

    dl = DataLoader(data, batch_size=tc["batch_size"], shuffle=True,
                    collate_fn=collate(tok, tc["max_seq_len"]))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=tc["lr"])
    steps = (len(dl) // tc["grad_accum"] + 1) * tc["epochs"]
    sched = get_cosine_schedule_with_warmup(opt, int(0.05 * steps), steps)

    params = [p_ for p_ in model.parameters() if p_.requires_grad]
    step, t0 = 0, time.time()
    for ep in range(tc["epochs"]):
        for i, b in enumerate(tqdm(dl, desc=f"epoch {ep + 1}")):
            b = {k: v.to(model.device) for k, v in b.items()}
            loss = model(**b).loss / tc["grad_accum"]
            scaler.scale(loss).backward()
            if (i + 1) % tc["grad_accum"] == 0 or (i + 1) == len(dl):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt); scaler.update(); sched.step(); opt.zero_grad(); step += 1
                if step % 10 == 0 or step == 1:
                    print(f"step {step}/{steps} loss {loss.item() * tc['grad_accum']:.4f} "
                          f"({time.time() - t0:.0f}s, peak mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB)")
    print(f"training time: {time.time() - t0:.0f}s")

    name = f"{args.recipe}_{'-'.join(args.langs)}_s{args.seed}"
    out = ROOT / cfg["paths"]["outputs"] / "adapters" / name
    model.save_pretrained(out)
    tok.save_pretrained(out)
    print(f"saved adapter to {out}")


if __name__ == "__main__":
    main()
