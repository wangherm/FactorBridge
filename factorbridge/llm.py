"""Real local Qwen QLoRA SFT. No API teacher, no CPU training fallback."""
from __future__ import annotations

import gc
import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

from .contracts import encode_completion, messages, validate_output, SYSTEM
from .io import read_json, read_jsonl, write_json, write_jsonl, digest, sha256, environment, code_hash, lock_environment


def prepared(c, internal=False, public_test=False):
    root = Path(c["run_dir"]) / ("internal_prepared" if internal else "test_prepared" if public_test else "prepared")
    index = read_json(root / "index.json")
    if not index["complete"] or index["config_hash"] != digest(c):
        raise ValueError("Prepared configuration changed; create a new run, do not relabel old data")
    for path, expected in index["files"].items():
        if sha256(root / path) != expected:
            raise ValueError(f"Prepared artifact changed: {path}")
    return root, index


def signature(c):
    _, index = prepared(c)
    initial = {}
    if c["stage"] == "cross_species_identity":
        stage1 = read_json(c["stage1_config"])
        directory = Path(stage1["run_dir"]) / "training/adapter"
        initial = {p.name: sha256(p) for p in sorted(directory.glob("*")) if p.is_file()}
        if not initial: raise ValueError("Initial Stage 1 adapter missing")
    return digest({"config": c, "prepared": index, "code": code_hash(), "environment": environment(), "initial_adapter": initial})


def require_gpu():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable: Qwen load/train smoke test NOT performed. No CPU fallback.")
    return torch


def load_model(c, adapter=None, trainable=True):
    torch = require_gpu()
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
    if c["stage"] == "cross_species_identity" and trainable and adapter is None:
        from .cross_species import initial_adapter
        adapter = initial_adapter(c)
    if not re.fullmatch(r"[0-9a-f]{40}", c.get("model_revision") or ""):
        raise ValueError("Pin model_revision to a Hugging Face commit SHA before GPU run")
    dtype_name = c["compute_dtype"]
    dtype = torch.bfloat16 if dtype_name == "bfloat16" or (dtype_name == "auto" and torch.cuda.is_bf16_supported()) else torch.float16
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("Requested BF16 unsupported on actual GPU")
    if dtype_name not in {"auto", "bfloat16", "float16"}:
        raise ValueError("Unknown compute_dtype")
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype) if c["load_in_4bit"] else None
    tok = AutoTokenizer.from_pretrained(c["model_name"], revision=c["model_revision"], trust_remote_code=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(c["model_name"], revision=c["model_revision"],
             quantization_config=quant, dtype=dtype, device_map={"": 0}, trust_remote_code=False, attn_implementation="sdpa")
    base.config.use_cache = not trainable
    if c["load_in_4bit"]:
        # PEFT also upcasts non-Params4bit floating weights to FP32. Apply the
        # same precision recipe to train, reloaded adapters and frozen Qwen.
        base = prepare_model_for_kbit_training(base,
            use_gradient_checkpointing=trainable and c["gradient_checkpointing"],
            gradient_checkpointing_kwargs={"use_reentrant": False})
    elif trainable and c["gradient_checkpointing"]:
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if adapter:
        model = PeftModel.from_pretrained(base, str(adapter), is_trainable=trainable)
    elif trainable:
        lc = LoraConfig(r=c["lora_r"], lora_alpha=c["lora_alpha"], lora_dropout=c["lora_dropout"],
                        target_modules=c["lora_target_modules"], task_type="CAUSAL_LM", bias="none")
        model = get_peft_model(base, lc)
    else:
        model = base
    if trainable:
        names = [n for n, p in model.named_parameters() if p.requires_grad]
        if not names or any("lora_" not in n for n in names):
            raise RuntimeError("Unexpected trainable base parameters")
    return model, tok, str(dtype)


def training_data(c, tok):
    if c.get("pilot_only"):
        raise ValueError("Single-study pilot has no independent validation; add independent studies before Qwen training")
    root, _ = prepared(c)
    cards = {r["example_id"]: r["card"] for r in read_jsonl(root / "cards.jsonl")}
    label_rows = {r["example_id"]: r for r in read_jsonl(root / "private/labels.jsonl")}
    labels = {eid: r["target"] for eid, r in label_rows.items()}
    result = {"train": [], "validation": []}
    exports = {s: [] for s in result}
    lengths = []
    failures = []
    def encode_row(row):
        split, eid = row["split"], row["example_id"]
        if split not in result:
            return None
        try:
            encoded = encode_completion(tok, cards[eid], labels[eid], c["max_length"], c["min_support"])
            encoded["teacher_targets"] = label_rows[eid].get("distillation_targets", {})
        except ValueError as exc:
            return (split, eid, None, str(exc))
        return (split, eid, encoded, None)
    with ThreadPoolExecutor(max_workers=c.get("preprocessing_workers", 1)) as pool:
        encoded_rows = list(pool.map(encode_row, read_jsonl(root / "private/lineage.jsonl")))
    for row in encoded_rows:
        if row is None:
            continue
        split, eid, encoded, error = row
        if error:
            failures.append({"example_id": eid, "error": error})
            continue
        lengths.append(len(encoded["input_ids"]))
        result[split].append(encoded)
        exports[split].append({"prompt": messages(cards[eid]), "completion": [{"role": "assistant", "content": json.dumps(labels[eid], ensure_ascii=False)}]})
    target = Path(c["run_dir"]) / "sft"
    write_json(target / "lengths.json", {"count": len(lengths), "maximum": max(lengths, default=0), "rejected": failures, "truncation": False})
    if failures:
        raise ValueError("Overlong/invalid training examples recorded in sft/lengths.json; no silent dropping")
    if not all(result.values()):
        raise ValueError("Both train and validation examples required")
    for split, rows in exports.items():
        write_jsonl(target / f"{split}.jsonl", rows)
    return result


def batch(rows, torch, pad_id=0):
    rows = [rows] if isinstance(rows, dict) else rows
    width = max(len(r["input_ids"]) for r in rows)
    values = {}
    for key, pad in [("input_ids", pad_id), ("attention_mask", 0), ("labels", -100)]:
        values[key] = torch.tensor([r[key] + [pad] * (width - len(r[key])) for r in rows], device="cuda", dtype=torch.long)
    return values


def save(model, tok, path, c, head=None):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    # This pipeline never resizes the vocabulary or trains embedding weights.
    # Avoid PEFT's online vocabulary-size probe when saving an offline adapter.
    model.save_pretrained(path, safe_serialization=True, save_embedding_layers=False)
    tok.save_pretrained(path)
    write_json(path / "resolved_config.json", c)
    write_json(path / "contract.json", {"system_prompt": SYSTEM, "schema_version": "stage1-v1", "annotation_version": "none", "chat_template": tok.chat_template})
    from .distill import save_head
    save_head(head, path, c)


def step(model, opt, rows, torch, micro_batch_size=1, pad_id=0, c=None, head=None):
    from .distill import loss as combined_loss
    opt.zero_grad(set_to_none=True)
    losses = []
    blocks = [rows[i:i+micro_batch_size] for i in range(0, len(rows), micro_batch_size)]
    for block in blocks:
        loss, _, _ = combined_loss(model, batch(block, torch, pad_id), block, c or {}, head, torch)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite forward loss")
        (loss / len(blocks)).backward()
        losses.append(float(loss.detach().cpu()))
    parameters = [p for p in model.parameters() if p.requires_grad] + (list(head.parameters()) if head is not None else [])
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads or not all(torch.isfinite(g).all() for g in grads) or not any(torch.any(g != 0) for g in grads):
        raise RuntimeError("Missing, nonfinite, or all-zero adapter gradients")
    if head is not None and any(r.get("teacher_targets") for r in rows):
        if not any(p.grad is not None and torch.any(p.grad != 0) for p in head.parameters()):
            raise RuntimeError("Teacher targets present but auxiliary head received no gradient")
    torch.nn.utils.clip_grad_norm_(parameters, 1.0)
    opt.step()
    return float(np.mean(losses))


def reload_fingerprint(model, torch):
    """Record runtime precision and exact adapter bytes, without hashing all base weights."""
    parameters, adapters = {}, {}
    for name, p in model.named_parameters():
        parameters[name] = {"dtype": str(p.dtype), "shape": list(p.shape), "class": type(p).__name__}
        if "lora_" in name:
            raw = p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
            adapters[name] = {**parameters[name], "sha256": hashlib.sha256(raw).hexdigest()}
    if not adapters:
        raise RuntimeError("No LoRA tensors available for serialization check")
    return {"parameters": parameters, "adapter_tensors": adapters,
            "quantized_compute_dtypes": {n: str(m.compute_dtype) for n, m in model.named_modules() if hasattr(m, "compute_dtype")}}


def smoke(c):
    torch = require_gpu()
    root = Path(c["run_dir"]) / "smoke"
    if root.exists():
        raise FileExistsError("Smoke output exists; choose a new run_dir to preserve evidence")
    root.mkdir(parents=True)
    sig = signature(c)
    write_json(root / "status.json", {"status": "running", "signature": sig})
    torch.manual_seed(c["seed"])
    model, tok, dtype = load_model(c)
    from .distill import make_head
    head = make_head(model, c, torch)
    rows = training_data(c, tok)
    trainable = {n: p.numel() for n, p in model.named_parameters() if p.requires_grad}
    if head is not None:
        trainable.update({"auxiliary_head." + n: p.numel() for n, p in head.named_parameters()})
    write_json(root / "trainable_parameters.json", trainable)
    write_json(root / "environment.json", environment())
    lock_environment(root / "requirements.resolved.txt")
    torch.cuda.reset_peak_memory_stats()
    parameters = [p for p in model.parameters() if p.requires_grad] + (list(head.parameters()) if head is not None else [])
    opt = torch.optim.AdamW(parameters, lr=c["learning_rate"])
    logs = []
    model.train()
    longest = sorted(rows["train"], key=lambda r: len(r["input_ids"]), reverse=True)
    teacher_rows = [r for r in longest if r.get("teacher_targets")]
    if head is not None and not teacher_rows:
        raise ValueError("No distillation targets for configured teachers")
    for i in range(c["smoke_steps"]):
        pool = teacher_rows if head is not None and i == 0 else longest
        sample = [pool[j % len(pool)] for j in range(c["micro_batch_size"])]
        loss = step(model, opt, sample, torch, c["micro_batch_size"], tok.pad_token_id, c, head)
        logs.append({"step": i + 1, "loss": loss})
        write_jsonl(root / "steps.jsonl", logs)
    model.eval()
    b = batch(rows["train"][0], torch)
    with torch.no_grad():
        before = model(**b, use_cache=False).logits[:, -1, :].float().cpu()
    before_fingerprint = reload_fingerprint(model, torch)
    write_json(root / "before_reload_fingerprint.json", before_fingerprint)
    save(model, tok, root / "adapter", c, head)
    head_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()} if head is not None else None
    del opt, parameters, model, b, head
    gc.collect()
    torch.cuda.empty_cache()
    model, tok, _ = load_model(c, root / "adapter", trainable=False)
    if head_state is not None:
        from safetensors.torch import load_file
        reloaded_head = load_file(str(root / "adapter/distillation_head.safetensors"))
        if any(not torch.equal(v, reloaded_head[k]) for k, v in head_state.items()):
            raise RuntimeError("Distillation head save/reload mismatch")
    model.eval()
    with torch.no_grad():
        after = model(**batch(rows["train"][0], torch), use_cache=False).logits[:, -1, :].float().cpu()
    after_fingerprint = reload_fingerprint(model, torch)
    write_json(root / "after_reload_fingerprint.json", after_fingerprint)
    delta = float((before - after).abs().max())
    allclose = torch.allclose(before, after, rtol=0.01, atol=0.05)
    precision_equal = (before_fingerprint["parameters"] == after_fingerprint["parameters"] and
                       before_fingerprint["quantized_compute_dtypes"] == after_fingerprint["quantized_compute_dtypes"])
    adapter_equal = before_fingerprint["adapter_tensors"] == after_fingerprint["adapter_tensors"]
    np.savez_compressed(root / "reload_logits.npz", before=before.numpy(), after=after.numpy())
    write_json(root / "reload_diagnostics.json", {"parameter_structure_and_dtypes_equal": precision_equal,
               "adapter_tensor_hashes_equal": adapter_equal, "logits_allclose": allclose,
               "max_abs_delta": delta, "mean_abs_delta": float((before - after).abs().mean()),
               "rtol": 0.01, "atol": 0.05, "use_cache_for_both_forwards": False, "example_split": "train"})
    if not precision_equal or not adapter_equal:
        raise RuntimeError("Save/reload parameter precision or adapter bytes differ; see reload_diagnostics.json")
    if not allclose:
        raise RuntimeError(f"Save/reload logits differ: {delta}")
    write_json(root / "status.json", {"status": "passed", "signature": sig, "dtype": dtype,
               "reload_max_abs_delta": delta, "trainable_parameters": sum(trainable.values()),
               "peak_cuda_bytes": torch.cuda.max_memory_allocated(), "steps": c["smoke_steps"]})


def verify_training_smoke(c):
    """Reuse measured GPU evidence for a new dataset, never manufacture a smoke pass.

    The explicitly accepted previous release used the identical load/step/save
    implementation. Its hash is needed because signatures also hash data code.
    All actual evidence is read from the user's original run on this machine.
    """
    source = read_json(c["reuse_smoke_config"]) if c.get("reuse_smoke_config") else c
    root = Path(source["run_dir"]) / "smoke"
    state = read_json(root / "status.json")
    if state.get("status") != "passed":
        raise RuntimeError("Actual successful GPU smoke evidence required")
    _, index = prepared(source)
    env = environment()
    if read_json(root / "environment.json") != env:
        raise RuntimeError("GPU/dependencies changed since the measured smoke; evidence cannot be reused")
    allowed_code = [code_hash()]
    if source["stage"] == "noise_recovery":
        # ad51e79: measured Qwen NF4/BF16 load/backward/save/reload implementation.
        allowed_code.append("34dbe91c405ab862eb9bd201c4808a0fc10fa1866fb0b40d5cfa2747e98e3922")
    valid = [digest({"config": source, "prepared": index, "code": h,
                     "environment": env, "initial_adapter": {}}) for h in allowed_code]
    if source["stage"] != "noise_recovery":
        valid = [signature(source)]
    if state.get("signature") not in valid:
        raise RuntimeError("Smoke signature does not match its original data, configuration, code and environment")
    if c.get("reuse_smoke_config"):
        if source["stage"] != "noise_recovery" or c["stage"] != "noise_recovery":
            raise ValueError("Dataset-only smoke reuse is limited to Stage 1; new teachers require their own evidence")
        data_keys = {"manifest", "run_dir", "pilot_only", "defer_public_test", "independent_dataset_cards",
                     "internal_final_test_only", "reuse_smoke_config", "epochs", "rank", "noise_levels"}
        different = [k for k in set(c) | set(source) if k not in data_keys and c.get(k) != source.get(k)]
        if different:
            raise ValueError(f"Model/training recipe differs from passed smoke: {different}")
        diagnostic = read_json(root / "reload_diagnostics.json")
        if not all(diagnostic.get(k) for k in ["parameter_structure_and_dtypes_equal", "adapter_tensor_hashes_equal", "logits_allclose"]):
            raise RuntimeError("Original save/reload diagnostics did not pass")
        if read_json(root / "adapter/resolved_config.json") != source:
            raise RuntimeError("Original smoke adapter configuration changed")
    return {"mode": "reused_actual_gpu_smoke" if c.get("reuse_smoke_config") else "original_run_smoke",
            "source_run": source["run_dir"], "source_signature": state["signature"],
            "new_smoke_executed": False, "reload_max_abs_delta": state.get("reload_max_abs_delta"),
            "evidence_sha256": {str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob("*")) if p.is_file()}}


def train(c):
    torch = require_gpu()
    sig = signature(c)
    smoke_evidence = verify_training_smoke(c)
    root = Path(c["run_dir"]) / "training"
    if root.exists():
        raise FileExistsError("Training directory exists; never overwrite a run")
    root.mkdir(parents=True)
    write_json(root / "gpu_smoke_evidence.json", smoke_evidence)
    write_json(root / "status.json", {"status": "running", "signature": sig})
    torch.manual_seed(c["seed"])
    rng = np.random.default_rng(c["seed"])
    model, tok, dtype = load_model(c)
    from .distill import make_head, loss as combined_loss
    head = make_head(model, c, torch)
    rows = training_data(c, tok)
    trainable = {n: p.numel() for n, p in model.named_parameters() if p.requires_grad}
    if head is not None:
        trainable.update({"auxiliary_head." + n: p.numel() for n, p in head.named_parameters()})
    write_json(root / "trainable_parameters.json", trainable)
    parameters = [p for p in model.parameters() if p.requires_grad] + (list(head.parameters()) if head is not None else [])
    opt = torch.optim.AdamW(parameters, lr=c["learning_rate"])
    logs, best, steps = [], float("inf"), 0
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(c["epochs"]):
        order = rng.permutation(len(rows["train"]))
        model.train()
        accumulation_size = c["gradient_accumulation_steps"] * c["micro_batch_size"]
        for offset in range(0, len(order), accumulation_size):
            block = order[offset:offset+accumulation_size]
            loss = step(model, opt, [rows["train"][i] for i in block], torch, c["micro_batch_size"], tok.pad_token_id, c, head)
            steps += 1
            logs.append({"epoch": epoch + 1, "step": steps, "training_loss": loss, "elapsed_seconds": time.monotonic() - started})
            write_jsonl(root / "steps.jsonl", logs)
            print(json.dumps(logs[-1]), flush=True)
        model.eval()
        val, val_sft, val_aux = [], [], []
        with torch.no_grad():
            for row in rows["validation"]:
                loss, sft, auxiliary = combined_loss(model, batch(row, torch), [row], c, head, torch)
                val.append(float(loss.cpu())); val_sft.append(sft); val_aux.append(auxiliary)
        vl = float(np.mean(val))
        if not np.isfinite(vl):
            raise RuntimeError("Nonfinite validation loss")
        logs.append({"epoch": epoch + 1, "validation_loss": vl, "validation_sft_loss": float(np.mean(val_sft)), "validation_auxiliary_mse": float(np.mean(val_aux))})
        write_jsonl(root / "steps.jsonl", logs)
        print(json.dumps(logs[-1]), flush=True)
        if vl < best:
            best = vl
            save(model, tok, root / "adapter", c, head)
    lock_environment(root / "requirements.resolved.txt")
    write_json(root / "environment.json", environment())
    write_json(root / "status.json", {"status": "completed", "signature": sig, "best_validation_loss": best,
               "steps": steps, "dtype": dtype, "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
               "elapsed_seconds": time.monotonic() - started, "scientific_success": "not_evaluated"})


def infer(c, card_rows, method, adapter_override=None):
    torch = require_gpu()
    adapter = None
    if adapter_override is not None:
        adapter = Path(adapter_override)
        saved = read_json(adapter / "resolved_config.json")
        if any(saved[k] != c[k] for k in ("model_name", "model_revision")):
            raise ValueError("Regression adapter backbone mismatch")
    elif method == "qwen_finetuned":
        state = read_json(Path(c["run_dir"]) / "training/status.json")
        if state.get("status") != "completed" or state["signature"] != signature(c):
            raise ValueError("Matching completed training run required")
        adapter = Path(c["run_dir"]) / "training/adapter"
    model, tok, _ = load_model(c, adapter, trainable=False)
    model.eval()
    result = []
    for row in card_rows:
        ids = tok.apply_chat_template(messages(row["card"]), tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
        if ids.shape[1] + c["max_new_tokens"] > c["max_length"]:
            raise ValueError(f"Prompt plus generation budget exceeds max_length: {row['example_id']}")
        with torch.no_grad():
            output = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=False,
                                    max_new_tokens=c["max_new_tokens"], pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        raw = tok.decode(output[0, ids.shape[1]:], skip_special_tokens=True)
        try:
            target = validate_output(json.loads(raw), row["card"], c["min_support"])
            result.append({"example_id": row["example_id"], "valid": True, "output": target, "raw": raw})
        except (ValueError, TypeError, KeyError) as exc:
            result.append({"example_id": row["example_id"], "valid": False, "output": None, "raw": raw, "error": str(exc)})
    return result
