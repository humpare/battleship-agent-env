#!/bin/bash
# Small-model experiment end to end: test the untrained model, train, test the checkpoints.
# Usage: train/bs_pipeline.sh   (needs train/.venv with torch + peft and a 24 GB GPU; about 1.5 hours)
set -e
cd "$(dirname "$0")"
R=../results/small-model; mkdir -p $R out
PY=./.venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ -f data/bs5.jsonl ] || $PY bs_data.py --games 4000 --out data/bs5.jsonl
$PY bs_eval.py --out $R/step-0.json 2>&1 | grep -E "^(frozen30|extra100)|Error"
$PY train_lora.py --data data/bs5.jsonl --out out/bs5-v1 --batch 8 --accum 2 --max-len 512 --epochs 1 \
    --save-steps 200 800 2400 2>&1 | grep -E "examples|step |saved|Error"
for s in 200 800 2400 final; do
  A=out/bs5-v1/step-$s; [ $s = final ] && A=out/bs5-v1
  $PY bs_eval.py --adapter $A --out $R/step-$s.json 2>&1 | grep -E "^(frozen30|extra100)|Error"
done
echo PIPELINE DONE
