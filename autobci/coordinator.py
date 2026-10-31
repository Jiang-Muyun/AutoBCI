# License: see LICENSE.
"""
Schedule candidate training across GPUs and recover interrupted training rounds.
"""

from datetime import datetime, timezone
from pathlib import Path
import argparse
import fcntl
import hashlib
import os
import re
import shutil
import signal
import subprocess
import sys
import time

import yaml

from .common import (
    PROJECT_ROOT as PROJECT,
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
from .data import CombinedData
from .run import (
    _validate_completed_round,
    check_protocol,
    is_multitask,
    peek_round,
    restore_early_checkpoint,
    stage_jobs,
    summarize_round,
    task_configs,
    task_reports,
)
from .train import candidate_outcome, validate_saved_training


SETTINGS_PATH = PROJECT / "config/coordinator.yaml"


def log(message):
    print(f"(coordinator) {message}", flush=True)


def find_config(model):
    """Match the run name to one active experiment YAML."""
    matches = []
    for path in sorted((PROJECT / "config/experiments").glob("*.yaml")):
        if path.name.startswith(("exclude_", "_exclude_")):
            continue
        config = load_config(path)
        if config["search"]["name"] == model:
            matches.append(config)
    if len(matches) != 1:
        raise ValueError(f"Expected one active config for {model}; found {len(matches)}")
    return matches[0]


def inspect_round(config, round_index, report):
    """Read-only preflight: validate prior rounds and distinguish fresh from interrupted."""
    if not 1 <= round_index <= int(config["search"]["rounds"]):
        raise ValueError("Round is outside the configured search")
    root = Path(config["output"]["root"]) / config["search"]["name"]
    previous_score = None
    for number in range(1, round_index + 1):
        folder = root / f"round_{number:02d}"
        check_generation_identity(folder, config)
        if is_multitask(config):
            check_protocol(folder, config, report)
        candidates = sorted(folder.glob("candidates/*.py"))
        if len(candidates) != int(config["search"]["new_candidates"]):
            raise RuntimeError(f"Incomplete generation: {folder.name}")
        for candidate in candidates:
            check_candidate_source(candidate)
        if (folder / "summary.json").exists():
            if not peek_settings(config)["enabled"]:
                for candidate in candidates:
                    for selected in task_configs(config):
                        task_report = report["tasks"][selected["_task"]] if is_multitask(config) else report
                        candidate_outcome(candidate, selected, task_report)
            summary = _validate_completed_round(root, number, config, report, previous_score)
            previous_score = summary["global_incumbent_score"]
            if number == round_index:
                return "complete", candidates
        else:
            if number < round_index:
                raise RuntimeError(f"Complete round_{number:02d} first")
            if peek_settings(config)["enabled"]:
                return "staged", candidates
            if any(training_dir(candidate, selected).exists()
                   for candidate in candidates for selected in task_configs(config)):
                raise RuntimeError(f"Interrupted round: {folder.name}; automatic resume is disabled")
            return "fresh", candidates


def stop_workers(processes):
    """Terminate private trainer process groups, including their data-loader children."""
    for process in processes:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 2
    for process in processes:
        try:
            process.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            pass
    for process in processes:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def gpu_memory(devices=None):
    """Query VRAM without CUDA contexts; map visible IDs to logical cuda:N names."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.free,memory.total", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    )
    rows = [tuple(part.strip() for part in line.split(","))
            for line in result.stdout.splitlines() if line.strip()]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    selected = rows
    if visible is not None:
        selected = []
        for token in visible.split(","):
            token = token.strip()
            if token in {"", "-1"}:
                break
            matches = [row for row in rows if row[0] == token or
                       (token.startswith("GPU-") and row[1].startswith(token))]
            if len(matches) != 1:
                raise ValueError(f"Cannot map CUDA_VISIBLE_DEVICES entry {token!r} using nvidia-smi")
            selected.append(matches[0])
    memory = {f"cuda:{index}": (int(row[2]) * 2**20, int(row[3]) * 2**20)
              for index, row in enumerate(selected)}
    return memory if devices is None else {device: memory[device] for device in devices}


def choose_gpu(memory, running, threshold=0.75):
    below = [device for device, (free, total) in memory.items()
             if (total - free) / total < threshold]
    if below:
        return min(below, key=lambda device: (running.get(device, 0), -memory[device][0]))
    return max(memory, key=lambda device: memory[device][0])


def discard_oom_attempt(candidate, failure, elapsed, config=None):
    """Keep failure evidence, but remove every checkpoint/result before a fresh retry."""
    output = training_dir(candidate, config)
    attempts = failure.get("oom_attempts", [])
    result = output / "result.json"
    attempts.append({
        "attempt": len(attempts) + 1,
        "ended_at": datetime.now(timezone.utc).isoformat(),
        "device": failure["device"],
        "stage": failure["stage"],
        "error": failure["error"],
        "candidate_sha256": failure["candidate_sha256"],
        "elapsed_seconds": elapsed,
        "epochs_completed": read_json(result).get("epochs_completed", 0) if result.exists() else 0,
    })
    failure["oom_attempts"] = attempts
    # This also marks the round interrupted if the coordinator is stopped while waiting.
    write_json(output / "failure.json", failure)
    for path in output.iterdir():
        if path.name != "failure.json":
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
    if config and peek_settings(config)["enabled"] and config.get("_target_epochs", 0) > int(peek_settings(config)["early_epochs"]):
        restore_early_checkpoint(candidate, config)


def record_training_timeout(worker, timeout_seconds, elapsed):
    """Keep completed epochs while marking a stopped worker as ineligible."""
    candidate, selected = worker["candidate"], worker["config"]
    output = training_dir(candidate, selected)
    failure_path = output / "failure.json"
    result_path = output / "result.json"
    target = selected.get("_target_epochs", selected.get("training", {}).get("epochs"))
    if failure_path.exists() or (result_path.exists() and (
            read_json(result_path).get("status") == "complete"
            or (target is not None and read_json(result_path).get("epochs_completed") == target))):
        return
    completed = read_json(result_path).get("epochs_completed", 0) if result_path.exists() else 0
    failure = {
        "status": "failed",
        "stage": "training",
        "failure_kind": "training_timeout",
        "error": f"Worker exceeded {timeout_seconds:g} seconds",
        "candidate_sha256": hashlib.sha256(model_file(candidate).read_bytes()).hexdigest(),
        "repair_attempted": False,
        "device": worker["device"],
        "timeout_seconds": timeout_seconds,
        "elapsed_seconds": elapsed,
        "epochs_completed": completed,
    }
    if selected.get("_task"):
        failure["task"] = selected["_task"]
    write_json(failure_path, failure)


def train_dynamic(config, candidates, devices, report, launch_delay=None, jobs=None):
    """Launch by live VRAM; only confirmed CUDA OOM receives a fresh-start retry."""
    settings = yaml.safe_load(SETTINGS_PATH.read_text())
    if launch_delay is None:
        launch_delay = float(settings["launch_delay_seconds"])
    threshold = float(settings["gpu_utilization_threshold"])
    max_tasks = int(settings["max_tasks_per_gpu"])
    capacity_wait = float(settings["capacity_wait_seconds"])
    timeout_seconds = float(settings["training_timeout_seconds"])
    if timeout_seconds <= 0:
        raise ValueError("training_timeout_seconds must be positive")
    if not devices:
        raise ValueError("At least one visible GPU is required")
    if launch_delay < 10:
        raise ValueError("Launch delay must be at least 10 seconds")
    log(f"Per candidate/task training timeout: {timeout_seconds:g}s from worker launch")
    pending = [(candidate, selected, 0.0) for candidate, selected in (
        jobs if jobs is not None else
        [(candidate, selected) for candidate in candidates for selected in task_configs(config)]
    )]
    active = []
    next_launch = 0.0

    def interrupt(signum, frame):
        raise SystemExit(128 + signum)

    def finish_worker(worker, code, timed_out=False):
        process, candidate, selected = worker["process"], worker["candidate"], worker["config"]
        task_report = report["tasks"][selected["_task"]] if is_multitask(config) else report
        outcome = candidate_outcome(candidate, selected, task_report)
        if not timed_out and (code == 0) != (outcome == "complete"):
            raise RuntimeError(f"Process exit {code} disagrees with saved outcome: {task_label(candidate, selected)}")
        active.remove(worker)
        failure_path = training_dir(candidate, selected) / "failure.json"
        failure = read_json(failure_path) if outcome == "failed" else {}
        if failure.get("failure_kind") == "cuda_oom":
            if failure.get("device") != worker["device"]:
                raise RuntimeError(f"OOM device record disagrees: {task_label(candidate, selected)}")
            if is_multitask(config) or "_target_epochs" in selected:
                discard_oom_attempt(candidate, failure, time.monotonic() - worker["started"], selected)
            else:
                discard_oom_attempt(candidate, failure, time.monotonic() - worker["started"])
            pending.append((candidate, selected, time.monotonic() + launch_delay))
            log(f"CUDA OOM: {task_label(candidate, selected)} on {worker['device']}; discarded attempt; "
                f"retry this training stage after at least {launch_delay:g}s: {failure['error']}")
        elif failure.get("failure_kind") == "training_timeout":
            log(f"{task_label(candidate, selected)}: timed out after {failure['elapsed_seconds']:.1f}s "
                f"(limit {timeout_seconds:g}s); completed epochs preserved")
        else:
            log(f"{task_label(candidate, selected)}: {outcome}; saved outcome verified")

    previous_handlers = {sig: signal.signal(sig, interrupt)
                         for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    try:
        while pending or active:
            for worker in list(active):
                process, candidate = worker["process"], worker["candidate"]
                code = process.poll()
                if code is None:
                    elapsed = time.monotonic() - worker["started"]
                    if elapsed >= timeout_seconds:
                        stop_workers([process])
                        record_training_timeout(worker, timeout_seconds, elapsed)
                        finish_worker(worker, None, timed_out=True)
                    continue
                if code < 0:
                    raise RuntimeError(f"Training process {process.pid} interrupted by signal {-code}")
                # A dead trainer may leave loader children behind; release them before retrying.
                stop_workers([process])
                finish_worker(worker, code)

            now = time.monotonic()
            ready = next((index for index, (_, _, after) in enumerate(pending) if after <= now), None)
            if ready is not None and now >= next_launch:
                running = {device: sum(worker["device"] == device for worker in active)
                           for device in devices}
                if any(count >= max_tasks for count in running.values()):
                    log(f"GPU task limit reached ({max_tasks}); waiting {capacity_wait:g}s")
                    time.sleep(capacity_wait)
                    continue
                memory = gpu_memory(devices)
                device = choose_gpu(memory, running, threshold)
                candidate, selected, _ = pending.pop(ready)
                output = training_dir(candidate, selected)
                if output.exists():
                    settings = peek_settings(selected)
                    continuation = (settings["enabled"] and selected.get("_target_epochs", 0) > settings["early_epochs"]
                                    and (output / "result.json").exists())
                    if continuation:
                        task_report = report["tasks"][selected["_task"]] if is_multitask(config) else report
                        validate_saved_training(output, candidate, selected, task_report)
                        if read_json(output / "result.json")["epochs_completed"] != settings["early_epochs"]:
                            raise RuntimeError("Continuation must start from the PEEK observation checkpoint")
                    else:
                        failure = read_json(output / "failure.json")
                        if not failure.get("oom_attempts") or list(output.iterdir()) != [output / "failure.json"]:
                            raise RuntimeError(f"Retry is not a fresh start: {task_label(candidate, selected)}")
                else:
                    output.mkdir(parents=True)
                free, total = memory[device]
                log(f"Launching {task_label(candidate, selected)} on {device}; {free / 2**30:.1f}/{total / 2**30:.1f} GiB free, "
                    f"{running[device]} own tasks running")
                command = [sys.executable, "-u", "-m", "autobci.train", "--config", config["_config_path"],
                           "--candidate", str(candidate), "--device", device]
                if selected.get("_task"):
                    command.extend(["--task", selected["_task"]])
                if config.get("search"):
                    command.extend(["--search", config["search"]["name"]])
                if "_target_epochs" in selected:
                    command.extend(["--target-epochs", str(selected["_target_epochs"])])
                process = subprocess.Popen(command, cwd=PROJECT, start_new_session=True)
                started = time.monotonic()
                active.append({"process": process, "candidate": candidate, "device": device, "started": started, "config": selected})
                next_launch = started + launch_delay
            if pending or active:
                time.sleep(0.2)
    finally:
        for sig in previous_handlers:
            signal.signal(sig, signal.SIG_IGN)
        if active:
            log("Stopping trainers and data-loader workers")
            stop_workers([worker["process"] for worker in active])
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


def resume_jobs(config, round_index, candidates, report):
    """Explicitly restart interrupted tasks while retaining their original files."""
    folder = Path(config["output"]["root"]) / config["search"]["name"] / f"round_{round_index:02d}"
    jobs = []
    for candidate in candidates:
        for selected in task_configs(config):
            task_report = report["tasks"][selected["_task"]] if is_multitask(config) else report
            output = training_dir(candidate, selected)
            if not output.exists():
                jobs.append((candidate, selected))
                continue
            try:
                outcome = candidate_outcome(candidate, selected, task_report)
            except RuntimeError as error:
                if not str(error).startswith("Interrupted or incomplete training:"):
                    raise
                if (output / "latest.pt").exists():
                    validate_saved_training(output, candidate, selected, task_report)
                archive = folder / "interrupted_training" / selected.get("_task", "combined") / output.name
                if archive.exists():
                    raise RuntimeError(f"Interrupted archive already exists: {archive}")
                archive.parent.mkdir(parents=True, exist_ok=True)
                output.rename(archive)
                log(f"Archived interrupted {task_label(candidate, selected)} at {archive}")
                jobs.append((candidate, selected))
            else:
                log(f"Keeping verified {outcome} outcome: {task_label(candidate, selected)}")
    return jobs


def coordinate(config, round_index, launch_delay=None, resume_interrupted=False):
    if launch_delay is None:
        settings = yaml.safe_load(SETTINGS_PATH.read_text())
        launch_delay = float(settings["launch_delay_seconds"])
    folder = Path(config["output"]["root"]) / config["search"]["name"] / f"round_{round_index:02d}"
    log(f"Checking {config['search']['name']} {folder.name}")
    if not folder.is_dir():
        raise ValueError(f"Generate {folder} before starting training")
    # Lock the existing directory itself; no lock/status/log files are created.
    lock = os.open(folder, os.O_RDONLY)
    try:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Another coordinator is already running {folder}") from None
        report = task_reports(config) if is_multitask(config) else CombinedData(config).report
        try:
            state, candidates = inspect_round(config, round_index, report)
        except RuntimeError as error:
            if not resume_interrupted or not str(error).startswith("Interrupted round:"):
                raise
            folder = Path(config["output"]["root"]) / config["search"]["name"] / f"round_{round_index:02d}"
            candidates = sorted(folder.glob("candidates/*.py"))
            state = "interrupted"
        if state == "complete":
            log(f"{config['search']['name']} {folder.name} already complete; saved results and checkpoints verified")
            return
        tasks = task_configs(config)
        names = ", ".join(task.get("_task", "combined") for task in tasks)
        log(f"{len(candidates)} architectures × {len(tasks)} tasks ({names}) = "
            f"{len(candidates) * len(tasks)} independent training jobs")
        devices = list(gpu_memory())
        log(f"{len(devices)} visible GPUs; dynamic VRAM scheduling; {launch_delay:g}s between all starts")
        if peek_settings(config)["enabled"]:
            def train_stage(paths, target):
                jobs = stage_jobs(config, paths, report, target, resume_interrupted)
                if jobs:
                    train_dynamic(config, paths, devices, report, launch_delay, jobs=jobs)
            peek_round(config, candidates, report, train_stage)
        else:
            jobs = resume_jobs(config, round_index, candidates, report) if state == "interrupted" else None
            train_dynamic(config, candidates, devices, report, launch_delay, jobs=jobs)
        summarize_round(config, round_index)
        inspect_round(config, round_index, report)
        log(f"{config['search']['name']} {folder.name} complete; results verified")
    finally:
        os.close(lock)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="run name, e.g. claude-opus-5-5")
    parser.add_argument("round", help="round folder, e.g. round_01")
    parser.add_argument("--launch-delay", type=float,
                        help="override launch_delay_seconds in config/coordinator.yaml")
    parser.add_argument("--search", help="separate run folder name (overrides search.name)")
    parser.add_argument("--resume-interrupted", action="store_true",
                        help="archive interrupted task files and restart only unfinished tasks")
    args = parser.parse_args()
    if not re.fullmatch(r"round_\d{2}", args.round):
        parser.error("Use a round name such as round_01")
    config = find_config(args.model)
    if args.search:
        config["search"]["name"] = args.search
    coordinate(config, int(args.round.split("_")[1]), args.launch_delay, args.resume_interrupted)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        log(f"ERROR: {type(error).__name__}: {error}")
        raise SystemExit(1)
