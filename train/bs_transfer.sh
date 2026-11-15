#!/bin/bash
# Does the trained small model carry over to rules it never saw? Same model, no further training,
# on other fleets and a larger board (the prompt states the changed rules). Compared with the
# untrained model. The model's own placement is usually illegal under changed rules, so the
# honest number here is shot quality and the score with a dealt fleet.
cd "$(dirname "$0")"
R=../results/small-model/transfer; mkdir -p $R
PY=./.venv/bin/python
run() {  # name, env...
  name=$1; shift
  for m in base trained; do
    A=""; [ $m = trained ] && A="--adapter out/bs5-v1"
    env "$@" $PY bs_eval.py $A --out $R/$name-$m.json 2>&1 | grep -E "^(frozen30|extra100)|Error" | sed "s/^/$name $m /"
  done
}
run fleet-3221   BATTLESHIP_FLEET=3,2,2,1
run fleet-22111  BATTLESHIP_FLEET=2,2,1,1,1
run board-6x6    BATTLESHIP_N=6
echo TRANSFER DONE
