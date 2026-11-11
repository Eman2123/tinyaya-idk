"""Sample answers (5 x temperature + greedy) and save hidden states of the last prompt token.

Usage: python src/sample.py --lang en [--limit 10]
Outputs:
  outputs/samples/<lang>.jsonl        id, split, question, gold, greedy, samples
  outputs/hidden/<lang>.npz           ids, layer arrays (n_questions x hidden)
"""
import argparse

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from common import build_prompt, load_config, load_questions, out_path, write_jsonl


def load_model(cfg):
    mc = cfg["model"]
    tok = AutoTokenizer.from_pretrained(mc["name"])
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kwargs = {"torch_dtype": getattr(torch, mc["dtype"]), "device_map": "auto"}
    if mc.get("load_in_4bit"):
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=getattr(torch, mc["dtype"])
        )
    model = AutoModelForCausalLM.from_pretrained(mc["name"], **kwargs)
    model.eval()
    return tok, model


@torch.no_grad()
def generate(tok, model, prompts, max_new, sample, temperature=0.7, n=1):
    enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
    kw = dict(max_new_tokens=max_new, pad_token_id=tok.pad_token_id)
    if sample:
        kw.update(do_sample=True, temperature=temperature, num_return_sequences=n)
    else:
        kw.update(do_sample=False)
    out = model.generate(**enc, **kw)
    gen = out[:, enc["input_ids"].shape[1]:]
    texts = tok.batch_decode(gen, skip_special_tokens=True)
    return [t.strip() for t in texts]


@torch.no_grad()
def last_token_hidden(tok, model, prompts, layers):
    enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
    hs = model(**enc, output_hidden_states=True).hidden_states
    # left padding -> last position is the last prompt token
    return [hs[l][:, -1, :].float().cpu().numpy() for l in layers]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    sc = cfg["sampling"]
    torch.manual_seed(sc["seed"])
    rows = load_questions(cfg, args.lang)
    if args.limit:
        rows = rows[: args.limit]

    tok, model = load_model(cfg)
    bs, n = sc["batch_size"], sc["n_samples"]
    layers = sc["probe_layers"]
    max_new = cfg["model"]["max_new_tokens"]

    results, hidden = [], [[] for _ in layers]
    for i in tqdm(range(0, len(rows), bs), desc=f"sampling {args.lang}"):
        batch = rows[i : i + bs]
        prompts = [build_prompt(cfg, r["question"]) for r in batch]
        greedy = generate(tok, model, prompts, max_new, sample=False)
        samp = generate(tok, model, prompts, max_new, sample=True,
                        temperature=sc["temperature"], n=n)
        hs = last_token_hidden(tok, model, prompts, layers)
        for j, r in enumerate(batch):
            results.append({
                "id": r["id"], "split": r["split"], "question": r["question"],
                "answers": r["answers"], "greedy": greedy[j],
                "samples": samp[j * n : (j + 1) * n],
            })
        for k in range(len(layers)):
            hidden[k].append(hs[k])

    write_jsonl(out_path(cfg, "samples", f"{args.lang}.jsonl"), results)
    np.savez_compressed(
        out_path(cfg, "hidden", f"{args.lang}.npz"),
        ids=np.array([r["id"] for r in results]),
        **{f"layer{layers[k]}": np.concatenate(hidden[k]) for k in range(len(layers))},
    )
    print(f"saved {len(results)} questions for {args.lang}")


if __name__ == "__main__":
    main()
