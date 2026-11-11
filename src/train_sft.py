"""LoRA SFT on a recipe's dataset (loss on the completion only).

Usage: python src/train_sft.py --recipe R3 --langs en ar [--seed 1]
Saves the adapter to outputs/adapters/<recipe>_<langs>_s<seed>/
"""
import argparse

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
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    tc = cfg["training"]
    torch.manual_seed(args.seed)

    data = []
    for lang in args.langs:
        data += read_jsonl(ROOT / cfg["paths"]["processed"] / args.recipe / f"{lang}.jsonl")
    print(f"{args.recipe}: {len(data)} training examples from {args.langs}")

    tok, model = load_model(cfg)
    tok.padding_side = "right"
    if cfg["model"].get("load_in_4bit"):
        model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(
        r=tc["lora_r"], lora_alpha=tc["lora_alpha"], lora_dropout=tc["lora_dropout"],
        target_modules=tc["target_modules"], task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    model.train()

    dl = DataLoader(data, batch_size=tc["batch_size"], shuffle=True,
                    collate_fn=collate(tok, tc["max_seq_len"]))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=tc["lr"])
    steps = (len(dl) // tc["grad_accum"] + 1) * tc["epochs"]
    sched = get_cosine_schedule_with_warmup(opt, int(0.05 * steps), steps)

    step = 0
    for ep in range(tc["epochs"]):
        for i, b in enumerate(tqdm(dl, desc=f"epoch {ep + 1}")):
            b = {k: v.to(model.device) for k, v in b.items()}
            loss = model(**b).loss / tc["grad_accum"]
            loss.backward()
            if (i + 1) % tc["grad_accum"] == 0:
                opt.step(); sched.step(); opt.zero_grad(); step += 1
                if step % 10 == 0:
                    print(f"step {step} loss {loss.item() * tc['grad_accum']:.4f}")

    name = f"{args.recipe}_{'-'.join(args.langs)}_s{args.seed}"
    out = ROOT / cfg["paths"]["outputs"] / "adapters" / name
    model.save_pretrained(out)
    tok.save_pretrained(out)
    print(f"saved adapter to {out}")


if __name__ == "__main__":
    main()
