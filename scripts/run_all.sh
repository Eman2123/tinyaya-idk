#!/usr/bin/env bash
# End-to-end run for one language group. Edit LANGS / ANCHOR; run from the repo root.
# Usage: bash scripts/run_all.sh [seed]
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src

ANCHOR=${ANCHOR:-en}
TARGETS=${TARGETS:-"es ar ms"}
LANGS="$ANCHOR $TARGETS"
SEED=${1:-1}

for L in $LANGS; do python src/sample.py --lang "$L"; python src/labels.py --lang "$L"; done
python src/probe.py --anchor "$ANCHOR" --targets $TARGETS
python src/build_datasets.py --langs $LANGS

# baseline first (over-refusal metrics need base preds)
python src/evaluate.py --name R0 --langs $LANGS
python src/sanity_checks.py --name R0

for R in R2 R3 R4; do
  python src/train_sft.py --recipe "$R" --langs $LANGS --seed "$SEED"
  NAME="${R}_$(echo $LANGS | tr ' ' '-')_s${SEED}"
  python src/evaluate.py --name "$NAME" --adapter "outputs/adapters/$NAME" --langs $LANGS
  python src/sanity_checks.py --name "$NAME" --adapter "outputs/adapters/$NAME"
done

# transfer: train on the anchor language only, test on every language
for R in R2 R3 R4; do
  python src/train_sft.py --recipe "$R" --langs $ANCHOR --seed "$SEED"
  NAME="${R}_${ANCHOR}_s${SEED}"
  python src/evaluate.py --name "$NAME" --adapter "outputs/adapters/$NAME" --langs $LANGS
done

python src/stats.py summary
python src/stats.py bootstrap --a R2 --b R3 --langs $LANGS
python src/stats.py bootstrap --a R3 --b R4 --langs $LANGS
