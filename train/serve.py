"""Minimal OpenAI-compatible /v1/chat/completions server for a local HF model (+ optional LoRA adapter).

Usage: train/.venv/bin/python train/serve.py --port 8011 [--adapter train/out/bc-v1]
"""
import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="Qwen/Qwen2.5-3B-Instruct")
ap.add_argument("--adapter", default=None)
ap.add_argument("--port", type=int, default=8011)
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(a.base)
model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16, device_map={"": 0})
if a.adapter:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, a.adapter).merge_and_unload()
model.eval()
name = a.adapter or a.base


class H(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        ids = tok.apply_chat_template(body["messages"], add_generation_prompt=True, return_tensors="pt",
                                      return_dict=True).to(model.device)
        temp = float(body.get("temperature", 0) or 0)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=min(int(body.get("max_tokens", 512)), 768),
                                 do_sample=temp > 0, temperature=temp if temp > 0 else None,
                                 top_p=None, top_k=None, pad_token_id=tok.eos_token_id)
        n_in = ids["input_ids"].shape[1]
        text = tok.decode(out[0, n_in:], skip_special_tokens=True)
        res = {"model": name, "choices": [{"index": 0, "finish_reason": "stop",
                                           "message": {"role": "assistant", "content": text}}],
               "usage": {"prompt_tokens": n_in, "completion_tokens": int(out.shape[1] - n_in)}}
        data = json.dumps(res).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


print(f"serving {name} on :{a.port}", flush=True)
HTTPServer(("127.0.0.1", a.port), H).serve_forever()
