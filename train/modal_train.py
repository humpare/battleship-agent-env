"""Run the LoRA fine-tune on a Modal GPU and keep the adapter in a Modal volume.

  ~/venvs/modal/bin/modal run train/modal_train.py --name bc-v2 --data train/data/bc.jsonl
  ~/venvs/modal/bin/modal volume get flatland-dispatch-train out/bc-v2 train/out/   # fetch the adapter
"""
import subprocess
from pathlib import Path

import modal

HERE = Path(__file__).parent
app = modal.App("flatland-dispatch-train")
vol = modal.Volume.from_name("flatland-dispatch-train", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "transformers", "peft", "accelerate", "huggingface_hub", "hf_transfer")
    .env({"HF_HOME": "/vol/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_file(HERE / "train_lora.py", "/root/train_lora.py")
)


@app.function(image=image, gpu="H100", timeout=4 * 3600, volumes={"/vol": vol})
def train(name: str, data: bytes, extra: list[str]):
    Path("/root/data.jsonl").write_bytes(data)
    cmd = ["python", "/root/train_lora.py", "--data", "/root/data.jsonl", "--out", f"/vol/out/{name}", *extra]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    vol.commit()


@app.local_entrypoint()
def main(name: str, data: str = "train/data/bc.jsonl", batch: int = 16, epochs: float = 1.0,
         max_examples: int = 0, resume: str = "", lr: float = 2e-4):
    extra = ["--batch", str(batch), "--accum", "1", "--epochs", str(epochs), "--lr", str(lr)]
    if max_examples:
        extra += ["--max-examples", str(max_examples)]
    if resume:
        extra += ["--resume-adapter", f"/vol/out/{resume}"]
    train.remote(name, Path(data).read_bytes(), extra)
