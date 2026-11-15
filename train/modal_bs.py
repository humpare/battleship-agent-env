"""Mixed-rules small-model experiment on a Modal H100: train ONE model on several Battleship
rule sets, then test it (and the untrained model) on held-out boards of rule sets it saw and
of rule sets it never saw.

  ~/venvs/modal/bin/modal run train/modal_bs.py --name mix-v1
Results come back to results/small-model/modal-<name>.json; the adapter stays in the Modal volume
(fetch: modal volume get flatland-dispatch-train out/<name> train/out/).
"""
import json
import random
import subprocess
from pathlib import Path

import modal

HERE = Path(__file__).parent
app = modal.App("battleship-small-model")
vol = modal.Volume.from_name("flatland-dispatch-train", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "transformers", "peft", "accelerate", "huggingface_hub", "hf_transfer", "numpy")
    .env({"HF_HOME": "/vol/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_dir(HERE.parent / "src" / "battleship", "/root/src/battleship")
    .add_local_file(HERE / "train_lora.py", "/root/train/train_lora.py")
    .add_local_file(HERE / "bs_common.py", "/root/train/bs_common.py")
    .add_local_file(HERE / "bs_eval.py", "/root/train/bs_eval.py")
)
TRAIN = {"bs5.jsonl": 25000, "bs5-3211.jsonl": None, "bs5-2221.jsonl": None, "bs5-331.jsonl": None,
         "bs6-3321.jsonl": None, "bs6-42211.jsonl": None, "bs6-22211.jsonl": None}
# (label, board, fleet, seen in training?, board groups)
TESTS = [("5x5 fleet 3-2-2-1-1", "5", "3,2,2,1,1", True, ["frozen30", "extra100"]),
         ("5x5 fleet 3-2-1-1", "5", "3,2,1,1", True, ["frozen30"]),
         ("5x5 fleet 3-2-2-1", "5", "3,2,2,1", False, ["frozen30", "extra100"]),
         ("5x5 fleet 2-2-1-1-1", "5", "2,2,1,1,1", False, ["frozen30", "extra100"]),
         ("6x6 fleet 3-2-2-1-1", "6", "3,2,2,1,1", False, ["frozen30"])]


@app.function(image=image, gpu="H100", cpu=8, timeout=5 * 3600, volumes={"/vol": vol})
def run(name: str, data: bytes, epochs: float, tests: list):
    import os
    Path("/root/data.jsonl").write_bytes(data)
    out = f"/vol/out/{name}"
    subprocess.run(["python", "/root/train/train_lora.py", "--data", "/root/data.jsonl", "--out", out,
                    "--batch", "32", "--accum", "1", "--max-len", "512", "--epochs", str(epochs)], check=True)
    vol.commit()
    results = []
    for label, n, fleet, seen, groups in tests:
        for model in ("untrained", "trained"):
            dest = f"/tmp/{model}-{n}-{fleet}.json"
            cmd = ["python", "/root/train/bs_eval.py", "--out", dest, "--groups", *groups, "--batch", "130"]
            if model == "trained":
                cmd += ["--adapter", out]
            subprocess.run(cmd, check=True, cwd="/root/train", env={**os.environ, "BATTLESHIP_N": n, "BATTLESHIP_FLEET": fleet})
            results.append({"rules": label, "rules_seen_in_training": seen, "model": model,
                            **json.loads(Path(dest).read_text())})
            print(label, model, {g: {k: v for k, v in r.items() if k != "scores"} for g, r in results[-1]["groups"].items()}, flush=True)
    Path(f"{out}/eval.json").write_text(json.dumps(results))
    vol.commit()
    return results


@app.local_entrypoint()
def main(name: str = "mix-v1", epochs: float = 1.0):
    rng, rows = random.Random(0), []
    for f, cap in TRAIN.items():
        lines = (HERE / "data" / f).read_text().splitlines()
        rng.shuffle(lines)
        rows += lines[:cap] if cap else lines
    rng.shuffle(rows)
    print(len(rows), "training examples from", len(TRAIN), "rule sets")
    res = run.remote(name, ("\n".join(rows) + "\n").encode(), epochs, TESTS)
    dest = HERE.parent / "results" / "small-model" / f"modal-{name}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(res, indent=1))
    print("saved", dest)
