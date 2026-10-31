# License: see LICENSE.
"""
Verify, train, and evaluate candidates with reproducible checkpoint continuation.
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
import argparse
import ast
import hashlib
import importlib.util
import inspect
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import (
    GeneratedCodeChanged,
    check_candidate_source,
    check_generation_identity,
    load_config,
    model_file,
    peek_settings,
    read_json,
    task_label,
    training_dir,
    write_json,
)
from .data import SplitBatches, batch_loader, prepare_input


# Numerical precision

def configure_cuda(training: dict, device: torch.device) -> dict:
    """Apply and record CUDA choices that can change speed and numerical results."""
    if device.type != "cuda":
        return {}
    settings = training.get("cuda", {})
    torch.backends.cudnn.benchmark = settings.get("cudnn_benchmark", False)
    torch.backends.cudnn.benchmark_limit = settings.get("cudnn_benchmark_limit", 10)
    torch.backends.cudnn.deterministic = settings.get("cudnn_deterministic", True)
    torch.set_float32_matmul_precision(settings.get("float32_matmul_precision", "highest"))
    # Keep the TF32 controls on the same API family as set_float32_matmul_precision.
    torch.backends.cudnn.allow_tf32 = settings.get("cudnn_allow_tf32", True)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = settings.get(
        "bf16_reduced_precision_reduction", True
    )
    return {
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_benchmark_limit": torch.backends.cudnn.benchmark_limit,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "fused_optimizer": bool(training.get("fused_optimizer", False)),
    }


def model_dtype(device: torch.device, precision: str) -> torch.dtype:
    if device.type == "cpu":
        if precision != "fp32":
            raise ValueError("CPU experiments require precision=fp32")
        return torch.float32
    if device.type != "cuda" or precision not in ("bf16", "bf16_full"):
        raise ValueError("CUDA experiments require precision=bf16 or bf16_full")
    torch.cuda.set_device(device)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Configured BF16 is not supported by this CUDA device")
    return torch.bfloat16 if precision == "bf16_full" else torch.float32


def autocast(device: torch.device, precision: str):
    if precision == "bf16":
        return torch.autocast(device.type, dtype=torch.bfloat16)
    return nullcontext()


# Architecture verification

ALLOWED_IMPORTS = {"torch", "math"}
FORBIDDEN_CALLS = {"open", "eval", "exec", "compile", "__import__"}


def static_contract(source_path: Path) -> dict:
    source = source_path.read_text()
    tree = ast.parse(source, filename=str(source_path))
    imports = []
    forbidden = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in FORBIDDEN_CALLS:
                forbidden.append(node.func.id)
    invalid = sorted({name for name in imports if name.split(".", 1)[0] not in ALLOWED_IMPORTS})
    if invalid:
        raise ValueError(f"Disallowed imports: {invalid}")
    if forbidden:
        raise ValueError(f"Forbidden calls: {sorted(set(forbidden))}")
    return {"source_bytes": len(source.encode()), "imports": sorted(set(imports))}


def import_candidate(candidate: Path):
    source_path = model_file(candidate)
    module_name = f"comb_v2_candidate_{candidate.stem}_{abs(hash(source_path.resolve()))}"
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {source_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
    build_model = getattr(module, "build_model", None)
    if not callable(build_model):
        raise TypeError("model.py must define build_model")
    parameters = list(inspect.signature(build_model).parameters)
    if parameters != ["n_channels", "n_samples", "n_classes"]:
        raise TypeError("build_model signature must be (n_channels, n_samples, n_classes)")
    return module


def build_candidate(candidate: Path, report: dict, parameter_limit: int):
    # Standalone synthetic benchmark models are not HAD candidates.
    if candidate.parent.name == "candidates":
        check_candidate_source(candidate)
    static_contract(model_file(candidate))
    module = import_candidate(candidate)
    channels, samples = report["input_shape"]
    model = module.build_model(channels, samples, len(report["labels"]))
    if not isinstance(model, nn.Module):
        raise TypeError("build_model must return torch.nn.Module")
    forbidden = (nn.RNNBase, nn.RNNCellBase)
    for name, layer in model.named_modules():
        if isinstance(layer, forbidden):
            raise ValueError(f"Recurrent module is not allowed: {name} ({type(layer).__name__})")
    count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if count == 0:
        raise ValueError("Model has no trainable parameters")
    if count >= parameter_limit:
        raise ValueError(f"{count} parameters exceed strict limit {parameter_limit}")
    return model, count


def verify(
    candidate: Path,
    config: dict,
    report: dict,
    output: Path,
    device: str | torch.device,
) -> dict:
    """Check the model contract and run two configured-size optimization steps."""
    candidate, output = Path(candidate), Path(output)
    device = torch.device(device)
    training = config["training"]
    precision = training.get("precision", "fp32")
    dtype = model_dtype(device, precision)
    cuda_settings = configure_cuda(training, device)
    model, count = build_candidate(candidate, report, int(config["search"]["max_parameters"]))
    model.to(device=device, dtype=dtype)
    channels, samples = report["input_shape"]
    classes = len(report["labels"])

    before = [(name, id(parameter)) for name, parameter in model.named_parameters()]
    model.eval()
    for batch_size in (1, 2):
        x = torch.randn(batch_size, channels, samples, device=device).to(dtype)
        with torch.no_grad(), autocast(device, precision):
            first = model(x)
            second = model(x)
        if tuple(first.shape) != (batch_size, classes):
            raise ValueError(f"Output shape {tuple(first.shape)} is not {(batch_size, classes)}")
        if not torch.isfinite(first).all():
            raise FloatingPointError("Model produced non-finite logits")
        if not torch.equal(first, second):
            raise RuntimeError("Repeated evaluation is not deterministic")
    if before != [(name, id(parameter)) for name, parameter in model.named_parameters()]:
        raise RuntimeError("Forward created trainable parameters")

    optimizer_name = training.get("optimizer", "AdamW")
    if optimizer_name.lower() != "adamw":
        raise ValueError("comb_v2 uses AdamW")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        fused=bool(training.get("fused_optimizer", False)) and device.type == "cuda",
    )
    batch_size = int(training["batch_size"])
    clip = float(training["gradient_clip_norm"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    model.train()
    for _ in range(2):
        x = torch.randn(batch_size, channels, samples, device=device).to(dtype)
        y = torch.randint(classes, (batch_size,), device=device)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device, precision):
            logits = model(x)
        loss = nn.functional.cross_entropy(logits.float(), y)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite verification loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip, error_if_nonfinite=True)
        optimizer.step()
    if before != [(name, id(parameter)) for name, parameter in model.named_parameters()]:
        raise RuntimeError("Training forward created trainable parameters")
    parameter_dtypes = {parameter.dtype for parameter in model.parameters()}
    gradient_dtypes = {parameter.grad.dtype for parameter in model.parameters() if parameter.grad is not None}
    if parameter_dtypes != {dtype} or gradient_dtypes != {dtype}:
        raise RuntimeError("Model changed the configured parameter or gradient dtype")

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    result = {
        "status": "passed",
        "device": str(device),
        "precision": precision,
        "parameter_dtype": str(dtype),
        "gradient_dtype": str(dtype),
        "cuda_settings": cuda_settings,
        "batch_size": batch_size,
        "parameter_count": count,
        "two_steps_seconds": time.monotonic() - started,
    }
    if device.type == "cuda":
        result["peak_allocated_mib"] = torch.cuda.max_memory_allocated(device) / 2**20
    write_json(output / "verification.json", result)
    return result


# Metrics and held-out evaluation

def metrics(y: np.ndarray, prediction: np.ndarray, classes: int) -> dict:
    confusion = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(confusion, (y, prediction), 1)
    support = confusion.sum(1)
    predicted = confusion.sum(0)
    true_positive = np.diag(confusion).astype(float)
    recall = np.divide(true_positive, support, out=np.zeros(classes), where=support > 0)
    precision = np.divide(true_positive, predicted, out=np.zeros(classes), where=predicted > 0)
    f1 = np.divide(
        2 * recall * precision,
        recall + precision,
        out=np.zeros(classes),
        where=recall + precision > 0,
    )
    present = support > 0
    total = int(confusion.sum())
    accuracy = float(true_positive.sum() / total)
    expected = float((support * predicted).sum() / (total * total))
    return {
        "accuracy": accuracy,
        "balanced_accuracy": float(recall[present].mean()),
        "cohen_kappa": (accuracy - expected) / (1 - expected) if expected < 1 else 0.0,
        "macro_f1": float(f1[present].mean()),
        "weighted_f1": float((f1 * support).sum() / total),
        "n": total,
    }


def evaluate(
    model: torch.nn.Module,
    split,
    report: dict,
    batch_size: int,
    device: str | torch.device,
    precision: str = "bf16",
    *, loader=None,
) -> dict:
    """Evaluate one shared head, masking labels absent from each dataset."""
    device = torch.device(device)
    dtype = model_dtype(device, precision)
    names = list(report["datasets"])
    classes = len(report["labels"])
    mask = torch.zeros((len(names), classes), dtype=torch.bool, device=device)
    for source_id, name in enumerate(names):
        mask[source_id, report["source_classes"][name]] = True

    if loader is None:
        loader = batch_loader(split, batch_size, device)
    predictions = []
    model.eval()
    with torch.inference_mode():
        for x, _, source in loader:
            x = prepare_input(x, split, device, dtype)
            source = source.to(device, non_blocking=True)
            with autocast(device, precision):
                logits = model(x)
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Non-finite evaluation logits")
            logits = logits.masked_fill(~mask[source], float("-inf"))
            predictions.append(logits.argmax(1).cpu().numpy())

    prediction = np.concatenate(predictions)
    y = split.y.numpy()
    source = split.source.numpy()
    per_dataset = {}
    for source_id, name in enumerate(names):
        selected = source == source_id
        per_dataset[name] = metrics(y[selected], prediction[selected], classes)
    scores = [row["balanced_accuracy"] for row in per_dataset.values()]
    return {
        "mean_dataset_balanced_accuracy": float(np.mean(scores)),
        "worst_dataset_balanced_accuracy": float(min(scores)),
        "per_dataset": per_dataset,
    }


def evaluate_frozen_test(
    data,
    model: torch.nn.Module,
    best_checkpoint: Path,
    output: Path,
    device: str | torch.device,
) -> dict:
    """Load test EEG only after the caller has frozen the champion selection."""
    device = torch.device(device)
    precision = data.config["training"].get("precision", "bf16")
    dtype = model_dtype(device, precision)
    data.load(("test",), device=device)
    configure_cuda(data.config["training"], device)
    # Set the destination dtype before loading so BF16 checkpoints stay BF16.
    model.to(device=device, dtype=dtype)
    state = torch.load(best_checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    loading = data.config.get("data", {})
    loader = batch_loader(
        data.splits["test"], int(data.config["training"]["batch_size"]), device,
        num_workers=int(loading.get("num_workers", 0)),
        prefetch_factor=int(loading.get("prefetch_factor", 2)),
    )
    result = evaluate(
        model,
        data.splits["test"],
        data.report,
        int(data.config["training"]["batch_size"]),
        device,
        precision,
        loader=loader,
    )
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "test.json", result)
    return result


def test_champion(
    data,
    candidate: Path,
    checkpoint: Path,
    output: Path,
    device: str | torch.device,
) -> dict:
    """Build the frozen champion and perform the experiment's only test read."""

    model, _ = build_candidate(
        Path(candidate), data.report, int(data.config["search"]["max_parameters"])
    )
    return evaluate_frozen_test(data, model, Path(checkpoint), Path(output), device)


# Training and checkpoint continuation

class TrainingIdentityChanged(ValueError):
    """Saved training artifacts belong to another protocol or dataset."""


def _atomic_torch(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _seed_all(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)


def _rng_state(generator: torch.Generator, device: torch.device) -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "shuffle": generator.get_state(),
    }
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(device)
    return state


def _restore_rng(state: dict, generator: torch.Generator, device: torch.device) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    generator.set_state(state["shuffle"])
    if "cuda" in state:
        torch.cuda.set_rng_state(state["cuda"], device)


def training_identity(config: dict, report: dict) -> dict:
    """Return the portable protocol and data identity used by a checkpoint."""
    evaluation = config.get("evaluation", {})
    supported_evaluation = {
        "selection_metric": "mean_dataset_balanced_accuracy",
        "checkpoint_policy": "one_common_epoch",
        "mask_invalid_logits": "negative_infinity",
        "test_when": "explicit_after_search",
    }
    for name, expected in supported_evaluation.items():
        if name in evaluation and evaluation[name] != expected:
            raise TrainingIdentityChanged(
                f"Unsupported evaluation setting {name}={evaluation[name]!r}"
            )
    datasets = []
    for name, values in report["datasets"].items():
        fingerprint = {
            "name": name,
            "bytes": values.get("bytes"),
            "mtime_ns": values.get("mtime_ns"),
            "counts": values.get("counts"),
            "sample_identity_sha256": values.get("sample_identity_sha256"),
        }
        datasets.append(fingerprint)
    return {
        "training": config["training"],
        "evaluation": evaluation,
        "max_parameters": config["search"]["max_parameters"],
        "input_shape": report["input_shape"],
        "labels": report["labels"],
        "label_policy": report.get("label_policy"),
        "dataset_order": list(report["datasets"]),
        "source_classes": report["source_classes"],
        "split_counts": report.get("counts"),
        "datasets": datasets,
    }


_training_identity = training_identity


def _identity_sha256(identity: dict) -> str:
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, default=str).encode()
    ).hexdigest()


def training_identity_sha256(config: dict, report: dict) -> str:
    return _identity_sha256(training_identity(config, report))


def validate_saved_training(
    output: Path,
    candidate: Path,
    config: dict,
    report: dict,
    *,
    require_checkpoint: bool = True,
) -> None:
    """Reject saved training artifacts from another model, protocol, or dataset."""
    output, candidate = Path(output), Path(candidate)
    candidate_sha = hashlib.sha256(model_file(candidate).read_bytes()).hexdigest()
    identity_sha = training_identity_sha256(config, report)
    config_path = output / "run_config.json"
    if not config_path.is_file():
        raise TrainingIdentityChanged("Saved training is missing run_config.json")
    saved = read_json(config_path)
    # Older files stored the entire data identity under "config".
    saved_training = saved.get("training", saved.get("config", {}).get("training"))
    if saved_training != config["training"]:
        raise TrainingIdentityChanged(
            "Saved training does not match this model, protocol, and dataset"
        )
    if require_checkpoint:
        checkpoint_path = output / "latest.pt"
        if not checkpoint_path.is_file():
            raise TrainingIdentityChanged("Saved training is missing latest.pt")
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False, mmap=True
        )
        if (
            checkpoint.get("candidate_sha256") != candidate_sha
            or checkpoint.get("training_identity_sha256") != identity_sha
        ):
            raise TrainingIdentityChanged(
                "Checkpoint does not match this model, protocol, and dataset"
            )


def _result(candidate: Path, checkpoint: dict, full_epochs: int, count: int) -> dict:
    completed = int(checkpoint["epoch"])
    return {
        "status": "complete" if completed == full_epochs else "partial",
        "candidate_id": candidate.stem,
        "epochs_completed": completed,
        "best_score": float(checkpoint["best_score"]),
        "best_epoch": int(checkpoint["best_epoch"]),
        "parameter_count": count,
        "seconds": float(checkpoint["seconds"]),
        "history": checkpoint["history"],
        "test_used_for_selection": False,
    }


def run(
    data,
    candidate: Path,
    output: Path,
    device: str | torch.device,
    target_epochs: int,
) -> dict:
    """Continue training to ``target_epochs`` on a fixed full-length schedule."""
    candidate, output = Path(candidate), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(device)
    config = data.config
    training = config["training"]
    precision = training.get("precision", "fp32")
    dtype = model_dtype(device, precision)
    cuda_settings = configure_cuda(training, device)
    if training.get("optimizer", "AdamW").lower() != "adamw":
        raise ValueError("comb_v2 uses AdamW")
    full_epochs = int(training["epochs"])
    target_epochs = int(target_epochs)
    if not 1 <= target_epochs <= full_epochs:
        raise ValueError(f"target_epochs must be between 1 and {full_epochs}")
    if "train" not in data.splits or "valid" not in data.splits:
        raise ValueError("Load train and valid splits before training")
    if "test" in data.splits:
        raise ValueError("Test data must not be loaded during model selection")

    torch.set_num_threads(int(training.get("cpu_threads", 1)))
    seed = int(training["seed"])
    _seed_all(seed, device)
    model, count = build_candidate(
        candidate, data.report, int(config["search"]["max_parameters"])
    )
    # Cast before creating AdamW: its moment tensors follow the parameter dtype.
    model.to(device=device, dtype=dtype)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        fused=bool(training.get("fused_optimizer", False)) and device.type == "cuda",
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=full_epochs)
    generator = torch.Generator().manual_seed(seed)
    candidate_sha = hashlib.sha256(model_file(candidate).read_bytes()).hexdigest()
    identity = training_identity(config, data.report)
    identity_sha = _identity_sha256(identity)
    run_config = {
        "seed": seed,
        "device": str(device),
        "effective_precision": precision,
        "torch": torch.__version__,
        "cuda_settings": cuda_settings,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "training": training,
    }
    if config.get("_task"):
        run_config["task"] = config["_task"]
    config_path = output / "run_config.json"
    latest_path = output / "latest.pt"
    if config_path.exists():
        validate_saved_training(
            output, candidate, config, data.report, require_checkpoint=latest_path.exists()
        )
    else:
        write_json(config_path, run_config)

    history = []
    best_score = -1.0
    best_epoch = 0
    completed = 0
    previous_seconds = 0.0
    if latest_path.exists():
        checkpoint = torch.load(latest_path, map_location="cpu", weights_only=False)
        if checkpoint["candidate_sha256"] != candidate_sha or checkpoint["training_identity_sha256"] != identity_sha:
            raise TrainingIdentityChanged(
                "Checkpoint does not match this model, protocol, and dataset"
            )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        history = checkpoint["history"]
        # Older checkpoints may contain matrices; keep only scalar metrics on resume.
        for row in history:
            for values in row["validation"]["per_dataset"].values():
                values.pop("confusion_matrix", None)
        best_score = float(checkpoint["best_score"])
        best_epoch = int(checkpoint["best_epoch"])
        completed = int(checkpoint["epoch"])
        previous_seconds = float(checkpoint["seconds"])
        _restore_rng(checkpoint["rng"], generator, device)
        if completed >= target_epochs:
            result = _result(candidate, checkpoint, full_epochs, count)
            write_json(output / "result.json", result)
            return result

    batch_size = int(training["batch_size"])
    clip = float(training["gradient_clip_norm"])
    split = data.splits["train"]
    loading = config.get("data", {})
    loader_options = {
        "num_workers": 0 if getattr(split, "normalized", False) else int(loading.get("num_workers", 0)),
        "prefetch_factor": int(loading.get("prefetch_factor", 2)),
    }
    # Both phases reuse one loader; RAM batches stay inside the trainer process.
    batches = []
    loader = batch_loader(SplitBatches(data.splits), batch_size, device, batches=batches, **loader_options)
    saved_config = read_json(config_path)
    saved_config.setdefault("data_loading_history", []).append({
        "from_epoch": completed + 1,
        "backend": type(split).__name__,
        **loader_options,
    })
    write_json(config_path, saved_config)
    print(f"(worker {device} {task_label(candidate, config)}) [Epo{completed:02d}] ready; next epoch {completed + 1}/{target_epochs}", flush=True)
    started = time.monotonic()
    for epoch in range(completed + 1, target_epochs + 1):
        epoch_started = time.monotonic()
        order = torch.randperm(len(split), generator=generator).tolist()
        batches[:] = [("train", order[start:start + batch_size]) for start in range(0, len(split), batch_size)]
        model.train()
        total_loss = 0.0
        learning_rate = optimizer.param_groups[0]["lr"]
        for x, y, _ in loader:
            x = prepare_input(x, split, device, dtype)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device, precision):
                logits = model(x)
            # FP32 loss reduction; full-BF16 parameter gradients remain BF16.
            loss = F.cross_entropy(logits.float(), y)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip, error_if_nonfinite=True)
            optimizer.step()
            total_loss += float(loss.detach()) * len(y)

        valid_count = len(data.splits["valid"])
        batches[:] = [("valid", list(range(start, min(start + batch_size, valid_count))))
                      for start in range(0, valid_count, batch_size)]
        validation = evaluate(
            model, data.splits["valid"], data.report, batch_size, device, precision,
            loader=loader,
        )
        score = validation["mean_dataset_balanced_accuracy"]
        row = {
            "epoch": epoch,
            "learning_rate": float(learning_rate),
            "loss": total_loss / len(split),
            "validation": validation,
        }
        history.append(row)
        if score > best_score:
            best_score, best_epoch = score, epoch
            _atomic_torch(output / "best.pt", model.state_dict())
        scheduler.step()
        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "rng": _rng_state(generator, device),
            "history": history,
            "best_score": best_score,
            "best_epoch": best_epoch,
            "seconds": previous_seconds + time.monotonic() - started,
            "candidate_sha256": candidate_sha,
            "training_identity_sha256": identity_sha,
        }
        _atomic_torch(latest_path, checkpoint)
        write_json(output / "result.json", _result(candidate, checkpoint, full_epochs, count))
        print(
            f"(worker {device} {task_label(candidate, config)}) [Epo{epoch:02d}] "
            f"loss={row['loss']:.4f} val_bacc={score:.4f} "
            f"best={best_score:.4f} time={time.monotonic() - epoch_started:.1f}s",
            flush=True,
        )

    result = _result(candidate, checkpoint, full_epochs, count)
    write_json(output / "result.json", result)
    print(f"(worker {device} {task_label(candidate, config)}) [done] epochs={target_epochs} best_val_bacc={best_score:.4f}", flush=True)
    return result


def candidate_outcome(candidate, config, report):
    """Require either full training evidence or an explicit, unchanged-model failure."""
    from .run import _result_reaches

    check_candidate_source(candidate)
    output = training_dir(candidate, config)
    failure_path = output / "failure.json"
    if failure_path.exists():
        failure = read_json(failure_path)
        source_hash = hashlib.sha256(model_file(candidate).read_bytes()).hexdigest()
        if config.get("_task") and failure.get("task") != config["_task"]:
            raise RuntimeError("Failure record belongs to a different task")
        if (failure.get("stage") not in {"verification", "training"}
                or failure.get("candidate_sha256") != source_hash
                or failure.get("repair_attempted") is not False):
            raise RuntimeError(f"Invalid failure record: {candidate.stem}")
        if (output / "result.json").exists():
            if read_json(output / "result.json").get("status") == "complete":
                raise RuntimeError(f"Conflicting complete and failed results: {candidate.stem}")
            validate_saved_training(output, candidate, config, report)
        return "failed"

    full_epochs = int(config["training"]["epochs"])
    epochs = int(config.get("_target_epochs", full_epochs))
    if not _result_reaches(output / "result.json", epochs, full_epochs):
        raise RuntimeError(f"Interrupted or incomplete training: {candidate.stem}; automatic resume is disabled")
    validate_saved_training(output, candidate, config, report)
    result = read_json(output / "result.json")
    checkpoint = torch.load(output / "latest.pt", map_location="cpu", weights_only=False, mmap=True)
    if (result["status"] != ("complete" if epochs == full_epochs else "partial")
            or result["epochs_completed"] != epochs or checkpoint.get("epoch") != epochs
            or [row["epoch"] for row in result["history"]] != list(range(1, epochs + 1))
            or checkpoint.get("history") != result["history"]
            or checkpoint.get("best_epoch") != result["best_epoch"]
            or checkpoint.get("best_score") != result["best_score"]):
        raise RuntimeError(f"Result and checkpoint disagree: {candidate.stem}")
    scores = [row["validation"]["mean_dataset_balanced_accuracy"] for row in result["history"]]
    if any(not math.isfinite(score) or not 0 <= score <= 1 for score in scores):
        raise RuntimeError(f"Invalid validation score: {candidate.stem}")
    if result["best_score"] != max(scores) or result["best_epoch"] != scores.index(max(scores)) + 1:
        raise RuntimeError(f"Best validation result is inconsistent: {candidate.stem}")
    best = torch.load(output / "best.pt", map_location="cpu", weights_only=True, mmap=True)
    if not best or best.keys() != checkpoint["model"].keys():
        raise RuntimeError(f"Invalid best checkpoint: {candidate.stem}")
    return "complete"


def is_cuda_oom(error: Exception, device: str) -> bool:
    """Recognize explicit CUDA allocation failures, never host OOM or killed processes."""
    message = str(error).lower()
    return (torch.device(device).type == "cuda" and isinstance(error, RuntimeError)
            and ("cuda out of memory" in message or "cuda error: out of memory" in message))


def run_candidate(config: dict, candidate: Path, device: str) -> dict:
    """Verify and train one generated model directly, without a worker server."""
    from .data import CombinedData
    from .run import _result_reaches

    candidate = Path(candidate).resolve()
    root = Path(config["output"]["root"]) / config["search"]["name"]
    if candidate.parents[2] != root:
        raise ValueError("Candidate does not belong to the configured search")
    check_generation_identity(candidate.parent.parent, config)
    from .run import is_multitask, check_protocol, task_reports
    if is_multitask(config):
        raise ValueError("Select an independent training task with --task emotion, --task mi, or --task sleep")
    if config.get("_task"):
        check_protocol(candidate.parent.parent, config, task_reports(config))
    settings = peek_settings(config)
    target_epochs = int(config.get("_target_epochs", config["training"]["epochs"]))
    if settings["enabled"]:
        early_epochs = int(settings["early_epochs"])
        if target_epochs not in {early_epochs, int(config["training"]["epochs"])}:
            raise ValueError("PEEK workers require the early or full epoch budget")
        if target_epochs > early_epochs:
            from .AutoBCI_PEEK import _resume_promotion
            promotion = read_json(candidate.parent.parent / "peek/promotion.json")
            promoted = _resume_promotion(promotion, sorted(candidate.parent.glob('*.py')), settings["promote_count"])
            if candidate not in promoted:
                raise ValueError("Only PEEK-promoted candidates may continue to the full budget")
    check_candidate_source(candidate)
    print(f"(worker {device} {task_label(candidate, config)}) [load] preparing dataset metadata", flush=True)
    data = CombinedData(config)
    output = training_dir(candidate, config)
    full_epochs = int(config["training"]["epochs"])
    if _result_reaches(output / "result.json", target_epochs, full_epochs):
        candidate_outcome(candidate, config, data.report)
        print(f"(worker {device} {task_label(candidate, config)}) [skip] already completed", flush=True)
        return read_json(output / "result.json")
    previous_failure = read_json(output / "failure.json") if (output / "failure.json").exists() else {}
    oom_attempts = previous_failure.get("oom_attempts", [])
    continuing = False
    if settings["enabled"] and target_epochs == full_epochs and (output / "result.json").exists():
        early_config = {**config, "_target_epochs": int(settings["early_epochs"])}
        # OOM retries restore the frozen early checkpoint before continuing.
        if not previous_failure:
            continuing = candidate_outcome(candidate, early_config, data.report) == "complete"
        elif oom_attempts:
            validate_saved_training(output, candidate, config, data.report)
            continuing = read_json(output / "result.json")["epochs_completed"] == int(settings["early_epochs"])
    if previous_failure and not oom_attempts:
        raise RuntimeError("Failed candidates cannot be retrained")
    if (config.get("_task") or settings["enabled"]) and output.exists() and any(output.iterdir()):
        if not continuing and (not oom_attempts or set(path.name for path in output.iterdir()) != {"failure.json"}):
            raise RuntimeError("Interrupted or failed task; automatic resume is disabled")
    stage = "verification"
    try:
        verify(candidate, config, data.report, output, device)
        stage = "training"
        storage = config["data"].get("storage", "disk")
        print(f"(worker {device} {task_label(candidate, config)}) [load] preparing train/valid ({storage})", flush=True)
        load_started = time.monotonic()
        data.load(splits=("train", "valid"), device=device)
        print(f"(worker {device} {task_label(candidate, config)}) [load] ready in {time.monotonic() - load_started:.1f}s", flush=True)
        result = run(data, candidate, output, device, target_epochs)
        if oom_attempts:
            saved = read_json(output / "run_config.json")
            saved["oom_attempts"] = [
                {key: value for key, value in attempt.items() if key != "candidate_sha256"}
                for attempt in oom_attempts
            ]
            write_json(output / "run_config.json", saved)
        (output / "failure.json").unlink(missing_ok=True)
        return result
    except (GeneratedCodeChanged, TrainingIdentityChanged):
        raise
    except Exception as exc:
        print(f"(worker {device} {task_label(candidate, config)}) [fail] {stage}: {type(exc).__name__}: {exc}", flush=True)
        failure = {
            "status": "failed", "stage": stage, "error": f"{type(exc).__name__}: {exc}",
            "candidate_sha256": hashlib.sha256(model_file(candidate).read_bytes()).hexdigest(),
            "repair_attempted": False,
            "device": str(device),
            "failure_kind": "cuda_oom" if is_cuda_oom(exc, device) else "candidate_error",
        }
        if config.get("_task"):
            failure["task"] = config["_task"]
        if oom_attempts:
            failure["oom_attempts"] = oom_attempts
        write_json(output / "failure.json", failure)
        if stage == "verification":
            write_json(output / "verification.json", failure)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--search", help="override the run folder name")
    parser.add_argument("--task", choices=("emotion", "mi", "sleep"))
    parser.add_argument("--target-epochs", type=int, help="early or full training budget for a PEEK stage")
    args = parser.parse_args()
    try:
        from .run import select_task
        config = load_config(args.config)
        if args.search:
            config["search"]["name"] = args.search
        if args.task:
            config = select_task(config, args.task)
        if args.target_epochs is not None:
            config["_target_epochs"] = args.target_epochs
        run_candidate(config, args.candidate, args.device)
    except Exception as error:
        print(f"(worker {args.device} {args.candidate.stem}) [error] {type(error).__name__}: {error}", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
