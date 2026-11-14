"""Side-by-side demo: base TinyAya vs a tuned abstention adapter on the same question.

Run on a machine with a GPU (about 8 GB free; a Kaggle/Colab T4 works):
  pip install gradio            # plus requirements.txt
  export HF_TOKEN=...           # needs access to CohereLabs/tiny-aya-global
  python src/app.py             # add --share for a temporary public link

Adapters come from the Hub repo (seed 1 adapters: R1..R5 x en/es/ar/ms, e.g. R4_es_s1).
The prompt is the same few-shot prompt used for training and evaluation (configs/config.yaml),
so answers are comparable with the reported numbers. Answers are cut at the first line, 16 new tokens.
This is a research demo: 'I don't know' means the adapter abstained, not that the answer is unknowable.
"""
import argparse

import gradio as gr
import torch
from peft import PeftModel

from common import build_prompt, is_refusal, load_config
from sample import generate, load_model

LANGS = {"English": "en", "Spanish": "es", "Arabic": "ar", "Malay": "ms"}
RECIPES = {"R4 probe labels (no gold)": "R4", "R2 gold labels": "R2", "R3 consistency labels": "R3",
           "R5 probe AND consistency": "R5", "R1 teacher traps": "R1"}
EXAMPLES = [["who sings where in the world is carmen sandiego", "English"],
            ["quién canta where in the world is carmen sandiego", "Spanish"],
            ["siapa penyanyi lagu where in the world is carmen sandiego", "Malay"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters-repo", default="EmanMirza2025/tinyaya-idk-adapters")
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    tok, model = load_model(cfg)
    max_new = cfg["model"]["max_new_tokens"]
    loaded = {}

    def answer(question, lang_name, recipe_name):
        lang, recipe = LANGS[lang_name], RECIPES[recipe_name]
        if not question.strip():
            return "", "", ""
        name = f"{recipe}_{lang}_s1"
        prompt = build_prompt(cfg, question.strip(), lang)
        if name not in loaded:
            if "_peft" not in loaded:
                loaded["_peft"] = PeftModel.from_pretrained(model, args.adapters_repo, subfolder=name, adapter_name=name)
            else:
                loaded["_peft"].load_adapter(args.adapters_repo, subfolder=name, adapter_name=name)
            loaded[name] = True
        pm = loaded["_peft"]
        with torch.no_grad():
            with pm.disable_adapter():
                base_out = generate(tok, pm, [prompt], max_new, sample=False)[0]
            pm.set_adapter(name)
            tuned_out = generate(tok, pm, [prompt], max_new, sample=False)[0]
        verdict = "The adapter abstained." if is_refusal(tuned_out, cfg, lang) else "The adapter answered."
        return base_out, tuned_out, f"{verdict} (adapter: {name})"

    with gr.Blocks(title="TinyAya IDK demo") as demo:
        gr.Markdown("# TinyAya: base vs abstention-tuned\nSame question, same few-shot prompt. "
                    "The tuned adapter should say \"I don't know\" more often when the base model is guessing. "
                    "Research demo, not a safety tool.")
        with gr.Row():
            q = gr.Textbox(label="Question", lines=2)
            lang = gr.Dropdown(list(LANGS), value="English", label="Question language (selects the prompt and adapter language)")
            rec = gr.Dropdown(list(RECIPES), value="R4 probe labels (no gold)", label="Adapter recipe")
        btn = gr.Button("Ask both")
        with gr.Row():
            base_box = gr.Textbox(label="Base model (tiny-aya-global)")
            tuned_box = gr.Textbox(label="Tuned adapter")
        info = gr.Markdown()
        gr.Examples(EXAMPLES, inputs=[q, lang])
        btn.click(answer, [q, lang, rec], [base_box, tuned_box, info])
    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
