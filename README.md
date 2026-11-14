# tinyaya-idk

Teaching TinyAya to say "I don't know" in languages with no labelled QA data.

Lexsi AI Safety and Alignment Hackathon project (deadline: 15 Nov 2026, 11:59 PM AoE (about 16 Nov, 4:59 PM PKT); re-verify on the hackathon page).

## Idea

Refusal-aware tuning needs gold answers to decide which questions a model "knows". Most languages have none.
We test whether cheap signals can replace gold labels when building abstention training data:

- **Consistency labels**: agreement across 5 sampled answers.
- **Probe labels**: logistic-regression probe on hidden states, trained on the anchor language only, transferred to other languages.

Recipes compared:

| Recipe | known/unknown source | Role |
|---|---|---|
| R0 | none (base model) | baseline |
| R1 | teacher-written trap questions | common shortcut |
| R2 | gold labels | upper bound |
| R3 | consistency labels | cheap method A |
| R4 | transferred-probe labels | cheap method B |
| R5 | probe AND consistency agree (ablation) | label-rule ablation, same size as R4 |

## Pipeline

```
data/raw/<lang>.jsonl                       # question, gold answers
  -> src/sample.py          -> outputs/samples/<lang>.jsonl, outputs/hidden/<lang>_<layer>.npy
  -> src/labels.py          -> outputs/labels/<lang>.jsonl   (gold, consistency)
  -> src/probe.py           -> outputs/labels/<lang>.jsonl   (adds probe labels), label-quality table
  -> src/build_datasets.py  -> data/processed/<recipe>/<lang>.jsonl
  -> src/train_sft.py       -> outputs/adapters/<recipe>_<lang_group>/
  -> src/evaluate.py        -> outputs/results/*.csv
```

## Setup

```bash
pip install -r requirements.txt
```

Edit `configs/config.yaml`: set `model.name` to the exact TinyAya Hugging Face ID (confirm on the Cohere Labs page) and the language list.

## Input format

`data/raw/<lang>.jsonl`, one JSON object per line:

```json
{"id": "q001", "question": "...", "answers": ["gold answer", "alias"], "split": "build|val|test"}
```

If `split` is missing, `src/common.py` assigns 1000/250/250 deterministically by id hash.

## Notebooks

`notebooks/colab.ipynb` and `notebooks/kaggle.ipynb` run the whole pipeline on free GPUs (clone repo, set model ID, smoke test,
sampling, labels, training, evaluation, tables). Both skip steps whose outputs already exist, so you can resume after a disconnect.
Set your Hugging Face token as a secret (`HF_TOKEN`) and accept the TinyAya licence first.

## Scripts

| Script | Purpose |
|---|---|
| `src/prepare_data.py` | Build `data/raw/<lang>.jsonl` from MKQA (no id/sw there; ms works) |
| `src/judge.py` | Optional LLM answer-equivalence judge + 50-item hand-check file |
| `src/generate_traps.py` | Teacher-written trap questions for R1 |
| `src/sanity_checks.py` | Safety refusal rate and general-capability check |
| `src/stats.py` | Seed summary, paired bootstrap, transfer matrix |
| `scripts/run_all.sh` | End-to-end run for one language group |
| `tests/test_pipeline.py` | CPU-only synthetic tests (`python tests/test_pipeline.py`) |

| `src/audit_eval.py` | AuditKIT pass (`ak.evaluate`) over the same test predictions; cross-check of `evaluate.py` |
| `src/curate.py` | CuratorKIT pass (dedup) over the SFT training sets |
| `scripts/run_loop.py` | Resumable train+eval loop over seeds/recipes/languages (`--sync-user` pushes each finished run to the Hub) |
| `scripts/sync_hf.py`, `scripts/restore_hf.py`, `scripts/hf_push.py` | Hub sync, restore after a Kaggle reset, final public push |

Not implemented: DPO stretch experiment, AlignTune (pins transformers/trl versions that clash with the training environment).

## Quick start

```bash
python src/sample.py --lang en --limit 10        # tiny end-to-end test
python src/labels.py --lang en
python src/probe.py --anchor en --targets es ar ms
python src/build_datasets.py --langs en es ar ms
python src/train_sft.py --recipe R3 --langs en ar
python src/evaluate.py --recipe R3 --langs en ar
```

## Notes

- Answer matching is normalized exact match. For non-Latin scripts use the optional judge (see `src/common.py`) and hand-check 50 items per language.
- Generation stops at the EOS token that `train_sft.py` appends to every target (`src/sample.py: generate`). Evaluations made before this fix (14 Nov 2026) are invalid and were re-run.
- Refusal templates in `configs/config.yaml` must be checked by a native speaker.
