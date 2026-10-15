"""LoRA policy-gradient training step for Qwen3.5-4B on Modal.

Two entry points:
  modal run modal_train.py::make_adapter --name iter0
      creates an identity LoRA adapter (B = 0, so the model is unchanged)
  modal run modal_train.py::train_step --prev iter0 --new iter1 --data rl/iter1.jsonl
      one policy-gradient update from a trajectory file, saving a new adapter

Trajectory file (JSON Lines), one line per model call:
  {"prompt": <exact text the model read>,
   "segments": [{"text": ..., "train": true/false}, ...]  # what followed, in order;
                                   # train=false marks injected text that the model did not write
   "advantage": float, "group": str, "seat": str, "game": str}
Volumes: parley-hf-cache (model weights), parley-adapters (LoRA adapters),
parley-rl-data (trajectory files and training logs).
"""

import json
import math
import os
import time

import modal

MODEL = "Qwen/Qwen3.5-4B"
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("torch==2.9.1", "transformers>=4.57", "peft>=0.17", "accelerate", "huggingface_hub[hf_transfer]")
    # Fast kernels for Qwen3.5's gated-delta-net layers; without them transformers falls
    # back to reference PyTorch code and a training step takes an hour.
    .uv_pip_install("flash-linear-attention")
    .run_commands("uv pip install --system --no-build-isolation causal-conv1d || echo 'causal-conv1d build failed; using the slow fallback'")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/cache/hf"})
)
hf_cache = modal.Volume.from_name("parley-hf-cache", create_if_missing=True)
adapters = modal.Volume.from_name("parley-adapters", create_if_missing=True)
data = modal.Volume.from_name("parley-rl-data", create_if_missing=True)
app = modal.App("parley-train")
VOLUMES = {"/cache/hf": hf_cache, "/adapters": adapters, "/data": data}


def load_base(device):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map=device)
    return tokenizer, model


@app.function(image=image, volumes=VOLUMES, memory=24 * 1024, timeout=30 * 60)
def make_adapter(name: str, rank: int = 16, alpha: int = 32):
    """An identity LoRA adapter: lets the server load 'an adapter' before any training."""
    from peft import LoraConfig, get_peft_model
    tokenizer, model = load_base("cpu")
    config = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.0, bias="none",
                        target_modules=TARGET_MODULES, task_type="CAUSAL_LM")
    peft_model = get_peft_model(model, config)
    peft_model.print_trainable_parameters()
    path = f"/adapters/{name}"
    peft_model.save_pretrained(path)
    adapters.commit()
    return sorted(os.listdir(path))


@app.function(image=image, volumes=VOLUMES, gpu="H100", timeout=3 * 60 * 60)
def train_step(prev: str, new: str, data_file: str, lr: float = 1e-5, epochs: int = 1,
               clip: float = 0.2, clip_high: float = 0.28, kl_coef: float = 0.0,
               max_len: int = 8192, minibatches: int = 4, grad_accum: int = 0,
               loss_agg: str = "token", max_grad_norm: float = 1.0, kl_log_fraction: float = 0.1):
    """One policy-gradient update (PPO/GRPO-style clipped objective) on a LoRA adapter.

    Advantages come precomputed in the data file. With epochs=1 the ratio is 1
    on the first pass, so the clipping only matters for epochs>1 or when the
    sampler's distribution differed; "old" log-probs are recomputed here from
    the previous adapter so sampler/trainer numerics do not leak into the ratio.
    """
    import torch
    import torch.nn.functional as F
    from peft import PeftModel

    adapters.reload()
    data.reload()
    torch.manual_seed(0)
    tokenizer, base = load_base("cuda")
    base.gradient_checkpointing_enable()
    base.config.use_cache = False
    model = PeftModel.from_pretrained(base, f"/adapters/{prev}", is_trainable=True)
    model.print_trainable_parameters()

    rows = [json.loads(line) for line in open(f"/data/{data_file}") if line.strip()]
    end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    examples = []
    for row in rows:
        prompt_ids = tokenizer(row["prompt"], add_special_tokens=False)["input_ids"]
        gen_ids, train_mask = [], []
        segments = row.get("segments") or [{"text": row["generated"], "train": True}]
        for seg in segments:
            ids = tokenizer(seg["text"], add_special_tokens=False)["input_ids"]
            gen_ids += ids
            train_mask += [bool(seg.get("train", True))] * len(ids)
        gen_ids.append(end)
        train_mask.append(True)
        if len(prompt_ids) + len(gen_ids) > max_len:
            # keep the end of the prompt (the instructions) and the whole generation
            keep = max_len - len(gen_ids)
            if keep < 256:
                continue
            prompt_ids = prompt_ids[-keep:]
        examples.append({"ids": prompt_ids + gen_ids, "n_prompt": len(prompt_ids),
                         "mask": train_mask, "adv": float(row["advantage"]), "group": row.get("group", "")})
    n_gen_total = sum(sum(e["mask"]) for e in examples)
    print(f"{len(examples)} examples, {n_gen_total} generated tokens, "
          f"mean adv {sum(e['adv'] for e in examples) / max(len(examples), 1):.3f}")

    def logprobs_and_entropy(m, ex):
        ids = torch.tensor([ex["ids"]], device="cuda")
        logits = m(input_ids=ids).logits[0, ex["n_prompt"] - 1:-1].float()
        targets = ids[0, ex["n_prompt"]:]
        logp_all = F.log_softmax(logits, dim=-1)
        logp = logp_all.gather(1, targets[:, None])[:, 0]
        entropy = -(logp_all.exp() * logp_all).sum(-1)
        mask = torch.tensor(ex["mask"], device="cuda")
        return logp[mask], entropy[mask]

    if grad_accum <= 0:
        grad_accum = max(1, math.ceil(len(examples) / max(minibatches, 1)))
    # Old log-probs (policy before this update); reference (base model) log-probs for the
    # KL term if it is on, else for a logged sample only.
    old, ref = [], {}
    model.eval()
    with torch.no_grad():
        for i, ex in enumerate(examples):
            lp, _ = logprobs_and_entropy(model, ex)
            old.append(lp)
            if kl_coef > 0 or (kl_log_fraction > 0 and i % max(1, int(1 / kl_log_fraction)) == 0):
                with model.disable_adapter():
                    rl, _ = logprobs_and_entropy(model, ex)
                ref[i] = rl
    model.train()
    print(f"grad_accum {grad_accum} -> {math.ceil(len(examples) / grad_accum)} optimizer steps", flush=True)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
    log = {"loss": [], "entropy": [], "ratio_clipped": [], "kl": []}
    order = list(range(len(examples)))
    step = 0
    started = time.monotonic()
    for epoch in range(epochs):
        torch.manual_seed(epoch)
        order = torch.randperm(len(examples)).tolist()
        optimizer.zero_grad()
        for i, idx in enumerate(order):
            ex = examples[idx]
            lp, ent = logprobs_and_entropy(model, ex)
            ratio = torch.exp(lp - old[idx])
            adv = ex["adv"]
            surrogate = torch.minimum(ratio * adv, torch.clamp(ratio, 1 - clip, 1 + clip_high) * adv)
            if loss_agg == "token":
                # Dr. GRPO / DAPO style: sum over tokens, divided by a constant, so long and
                # short generations weigh by their tokens rather than per-sequence.
                loss = -surrogate.sum() / (n_gen_total / len(examples))
            else:
                loss = -surrogate.mean()
            kl = torch.zeros((), device="cuda")
            if idx in ref:
                # k3 estimator of KL(policy || base) as in GRPO
                d = ref[idx] - lp
                kl = (torch.exp(d) - d - 1).mean()
                if kl_coef > 0:
                    loss = loss + kl_coef * kl
            (loss / grad_accum).backward()
            log["loss"].append(loss.item())
            log["entropy"].append(ent.mean().item())
            log["ratio_clipped"].append(((ratio < 1 - clip) | (ratio > 1 + clip_high)).float().mean().item())
            if idx in ref:
                log["kl"].append(kl.item())
            if (i + 1) % grad_accum == 0 or i == len(order) - 1:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                step += 1
                if step % 10 == 0:
                    print(f"step {step} loss {sum(log['loss'][-grad_accum:]) / grad_accum:.4f} "
                          f"entropy {sum(log['entropy'][-grad_accum:]) / grad_accum:.3f} "
                          f"{time.monotonic() - started:.0f}s", flush=True)
    model.save_pretrained(f"/adapters/{new}")
    adapters.commit()
    summary = {
        "prev": prev, "new": new, "data_file": data_file, "examples": len(examples),
        "generated_tokens": n_gen_total, "optimizer_steps": step, "epochs": epochs, "lr": lr,
        "clip": [clip, clip_high], "kl_coef": kl_coef, "loss_agg": loss_agg,
        "mean_loss": sum(log["loss"]) / max(len(log["loss"]), 1),
        "mean_entropy": sum(log["entropy"]) / max(len(log["entropy"]), 1),
        "clipped_fraction": sum(log["ratio_clipped"]) / max(len(log["ratio_clipped"]), 1),
        "mean_kl_to_base_sampled": sum(log["kl"]) / max(len(log["kl"]), 1),
        "kl_samples": len(log["kl"]),
        "seconds": round(time.monotonic() - started),
    }
    with open(f"/data/train-{new}.json", "w") as f:
        json.dump(summary, f, indent=1)
    data.commit()
    print(json.dumps(summary))
    return summary
