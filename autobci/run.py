"""
Orchestrate architecture search, independent task workflows, champion selection, and final evaluation.

Copyright 2022 Centre for Brain Computing Research (CBCR), College of Computing and Data Science (CCDS), Nanyang Technological University (NTU);
licensed under the CBCR License 1.0 (see LICENSE).
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Sequence
import argparse
import hashlib
import json
import shutil

from .common import (
    GeneratedCodeChanged,
    check_candidate_source,
    check_generation_identity,
    load_config,
    model_file,
    peek_settings,
    read_json,
    training_dir,
    validate_peek,
    write_json,
)
from .train import TrainingIdentityChanged


# Independent task workflows

def is_multitask(config):
    return bool(config.get("data", {}).get("tasks")) and not config.get("_task")


def select_task(config, task):
    selected = deepcopy(config)
    selected["data"]["catalog"] = selected["data"]["tasks"][task]
    selected["_task"] = task
    return selected


def task_configs(config):
    if not is_multitask(config):
        return [config]
    return [select_task(config, task) for task in config["data"]["tasks"]]


def task_reports(config):
    from .data import CombinedData

    return {"tasks": {
        task: CombinedData(select_task(config, task)).report
        for task in config["data"]["tasks"]
    }}


def early_checkpoint_dir(candidate, config):
    return (Path(candidate).parent.parent / "peek/early_checkpoints"
            / config.get("_task", "combined") / training_dir(candidate, config).name)


def restore_early_checkpoint(candidate, config):
    """Restore the exact checkpoint used for promotion after an interrupted continuation."""
    source = early_checkpoint_dir(candidate, config)
    if not source.is_dir():
        raise RuntimeError(f"Missing frozen early checkpoint: {source}")
    shutil.copytree(source, training_dir(candidate, config), dirs_exist_ok=True)


def stage_jobs(config, candidates, reports, target_epochs, resume_interrupted=False):
    """Skip completed stage outcomes and require explicit handling of interruptions."""
    from .train import candidate_outcome, validate_saved_training

    jobs = []
    early = int(peek_settings(config)["early_epochs"])
    for candidate in candidates:
        for selected in task_configs(config):
            selected = {**selected, "_target_epochs": target_epochs}
            report = reports["tasks"][selected["_task"]] if is_multitask(config) else reports
            output = training_dir(candidate, selected)
            if not output.exists() or not any(output.iterdir()):
                if target_epochs > early:
                    restore_early_checkpoint(candidate, selected)
                jobs.append((candidate, selected))
                continue
            try:
                candidate_outcome(candidate, selected, report)
                continue
            except RuntimeError as error:
                if not str(error).startswith("Interrupted or incomplete training:"):
                    raise
            result = read_json(output / "result.json") if (output / "result.json").exists() else {}
            if target_epochs > early and result.get("epochs_completed") == early:
                candidate_outcome(candidate, {**selected, "_target_epochs": early}, report)
            else:
                if not resume_interrupted:
                    raise RuntimeError(f"Interrupted round: {candidate.parent.parent.name}; use --resume-interrupted")
                if (output / "latest.pt").exists():
                    validate_saved_training(output, candidate, selected, report)
                archive = candidate.parent.parent / "interrupted_training" / selected.get("_task", "combined") / output.name
                if archive.exists():
                    raise RuntimeError(f"Interrupted archive already exists: {archive}")
                archive.parent.mkdir(parents=True, exist_ok=True)
                output.rename(archive)
                if target_epochs > early:
                    restore_early_checkpoint(candidate, selected)
            jobs.append((candidate, selected))
    return jobs


def peek_round(config, candidates, reports, train_stage, llm_call=None):
    """Observe all tasks, freeze one PEEK decision, then continue promoted models."""
    from .AutoBCI_PEEK import select_candidates
    from .train import candidate_outcome

    settings = validate_peek(config)
    folder = candidates[0].parent.parent
    early = int(settings["early_epochs"])
    if not (folder / "peek/promotion.json").exists():
        train_stage(candidates, early)
    promoted = select_candidates(folder, candidates, config, reports, llm_call)
    for candidate in promoted:
        for selected in task_configs(config):
            snapshot = early_checkpoint_dir(candidate, selected)
            if snapshot.exists():
                continue
            report = reports["tasks"][selected["_task"]] if is_multitask(config) else reports
            if candidate_outcome(candidate, {**selected, "_target_epochs": early}, report) != "complete":
                raise RuntimeError("Promoted candidate is missing successful early training")
            temporary = snapshot.with_name(snapshot.name + ".tmp")
            if temporary.exists():
                shutil.rmtree(temporary)
            shutil.copytree(training_dir(candidate, selected), temporary)
            temporary.rename(snapshot)
    train_stage(promoted, int(config["training"]["epochs"]))
    return promoted


def check_protocol(round_dir, config, report):
    """Never reinterpret older combined-data generations as current multi-task experiments."""
    from .AutoBCI_HAD import _generation_protocol

    prompt = Path(round_dir) / "generation" / "prompt.md"
    if not prompt.exists():
        raise ValueError("Missing generation protocol")
    protocol = json.loads(prompt.read_text().split("\n\nPROTOCOL\n", 1)[1]
                          .split("\n\nMEASURED PARENT EVIDENCE\n", 1)[0])

    def portable(value):
        if isinstance(value, dict):
            return {key: portable(item) for key, item in value.items()
                    if key not in {"path", "bytes", "mtime_ns", "pool_bytes", "sample_identity_sha256"}}
        if isinstance(value, list):
            return [portable(item) for item in value]
        return value

    round_index = int(Path(round_dir).name.removeprefix("round_"))
    expected = _generation_protocol(round_index, report, config)
    if portable(protocol) != portable(expected):
        raise ValueError("Generation protocol differs from configured task training; use a new search name. "
                         "Existing generated code and results are preserved.")


def round_summary(config, round_index, reports, previous_score=None):
    """A candidate is eligible only when ALL independent trainings completed."""
    from .train import candidate_outcome

    root = Path(config["output"]["root"]) / config["search"]["name"]
    folder = root / f"round_{round_index:02d}"
    check_generation_identity(folder, config)
    check_protocol(folder, config, reports)
    candidates = sorted(folder.glob("candidates/*.py"))
    if len(candidates) != int(config["search"]["new_candidates"]):
        raise RuntimeError("Round is missing model candidates")
    settings = peek_settings(config)
    promoted_ids = {candidate.stem for candidate in candidates}
    if settings["enabled"]:
        from .AutoBCI_PEEK import select_candidates
        if not (folder / "peek/promotion.json").is_file():
            raise RuntimeError("Round is missing its PEEK promotion")
        promoted_ids = {candidate.stem for candidate in select_candidates(folder, candidates, config, reports)}
    rows = []
    for candidate in candidates:
        metadata = read_json(candidate.with_suffix(".json"))
        if metadata["candidate_id"] != candidate.stem:
            raise ValueError("Candidate metadata does not match its filename")
        tasks = {}
        for selected in task_configs(config):
            task = selected["_task"]
            output = training_dir(candidate, selected)
            pruned = candidate.stem not in promoted_ids
            if pruned:
                selected = {**selected, "_target_epochs": int(settings["early_epochs"])}
            outcome = candidate_outcome(candidate, selected, reports["tasks"][task])
            if pruned and outcome == "complete":
                outcome = "pruned"
            result = read_json(output / "result.json") if (output / "result.json").exists() else {}
            tasks[task] = {
                "status": outcome,
                "best_score": result.get("best_score") if outcome == "complete" else None,
                "best_epoch": result.get("best_epoch") if outcome == "complete" else None,
                "epochs_completed": result.get("epochs_completed", 0),
                "parameter_count": result.get("parameter_count"),
                "seconds": result.get("seconds", 0.0),
                "failure": read_json(output / "failure.json") if outcome == "failed" else None,
            }
        complete = all(row["status"] == "complete" for row in tasks.values())
        rows.append({
            "candidate_id": candidate.stem, "parent_id": metadata.get("parent_id"),
            "status": "complete" if complete else ("failed" if any(t["status"] == "failed" for t in tasks.values()) else "pruned"), "tasks": tasks,
            "best_score": sum(row["best_score"] for row in tasks.values()) / len(tasks) if complete else None,
            "test_used_for_selection": False,
        })
    ranking = sorted((row for row in rows if row["status"] == "complete"),
                     key=lambda row: (-row["best_score"], row["candidate_id"]))
    if len(ranking) < int(config["search"]["parent_count"]):
        raise RuntimeError("Too few candidates completed both tasks to select HAD parents")
    best = ranking[0]["best_score"]
    return {
        "status": "complete", "round_index": round_index, "forecast_enabled": settings["enabled"],
        "selection_metric": "mean_task_best_validation_bacc",
        "candidates": rows, "full_budget_ranking": ranking,
        "promoted_ids": sorted(promoted_ids) if settings["enabled"] else [row["candidate_id"] for row in ranking],
        "parent_ids": sorted({row["parent_id"] for row in rows if row["parent_id"]}),
        "best_new_full_budget_candidate_id": ranking[0]["candidate_id"],
        "best_new_full_budget_score": best, "previous_global_incumbent_score": previous_score,
        "global_incumbent_score": best if previous_score is None else max(previous_score, best),
        "improved_global_incumbent": previous_score is None or best > previous_score,
        "compute": {
            "actual_epochs_trained": sum(task["epochs_completed"] for row in rows for task in row["tasks"].values()),
            "nominal_full_budget_epochs": len(rows) * len(config["data"]["tasks"]) * int(config["training"]["epochs"]),
            "observed_candidate_training_seconds": sum(task["seconds"] for row in rows for task in row["tasks"].values()),
        },
        "test_used_for_selection": False,
    }


def validate_round(config, round_index, reports, previous_score=None):
    root = Path(config["output"]["root"]) / config["search"]["name"]
    saved = read_json(root / f"round_{round_index:02d}" / "summary.json")
    expected = round_summary(config, round_index, reports, previous_score)
    if saved != expected:
        raise ValueError("Completed round summary is stale or inconsistent with the configured tasks")
    return saved


def _summarize_tasks(config, round_index):

    if not 1 <= round_index <= int(config["search"]["rounds"]):
        raise ValueError("Round is outside the configured search")
    reports = task_reports(config)
    root = Path(config["output"]["root"]) / config["search"]["name"]
    previous_score = None
    for number in range(1, round_index):
        previous_score = validate_round(config, number, reports, previous_score)["global_incumbent_score"]
    summary = round_summary(config, round_index, reports, previous_score)
    folder = root / f"round_{round_index:02d}"
    write_json(folder / "summary.json", summary)
    _finalize_search(config, reports)
    return folder


def completed_parents(search_dir, config, before_round=None):
    """Use the same global pool and equal task weighting across all finished rounds."""
    reports = task_reports(config)
    parents = []
    previous_score = None
    for number in range(1, int(config["search"]["rounds"]) + 1):
        if before_round is not None and number >= before_round:
            break
        folder = Path(search_dir) / f"round_{number:02d}"
        if not (folder / "summary.json").exists():
            break
        summary = validate_round(config, number, reports, previous_score)
        previous_score = summary["global_incumbent_score"]
        for row in summary["full_budget_ranking"]:
            candidate = folder / "candidates" / f"{row['candidate_id']}.py"
            parents.append({
                "candidate_path": candidate, "candidate_id": candidate.stem,
                "round_index": number, "best_score": row["best_score"],
                "metadata": read_json(candidate.with_suffix(".json")),
                "model_code": candidate.read_text(),
                "task_results": {selected["_task"]: read_json(training_dir(candidate, selected) / "result.json")
                                 for selected in task_configs(config)},
            })
    return sorted(parents, key=lambda row: (-row["best_score"], row["candidate_id"]))


def _search_tasks(config, *, llm_call=None, stop_after_round=None, generate_only=False, device="cuda"):
    from .AutoBCI_HAD import generate_round
    from .train import run_candidate

    settings = validate_peek(config)
    root = Path(config["output"]["root"]) / config["search"]["name"]
    root.mkdir(parents=True, exist_ok=True)
    reports = task_reports(config)
    previous_score = None
    for number in range(1, int(config["search"]["rounds"]) + 1):
        folder = root / f"round_{number:02d}"
        if (folder / "summary.json").exists():
            previous_score = validate_round(config, number, reports, previous_score)["global_incumbent_score"]
        else:
            if folder.exists() and (folder / "generation" / "prompt.md").exists():
                check_protocol(folder, config, reports)
            candidates = generate_round(root, number, config, reports, llm_call=llm_call)
            if generate_only:
                return root
            if settings["enabled"]:
                def train_stage(paths, target):
                    for candidate, selected in stage_jobs(config, paths, reports, target):
                        try:
                            run_candidate(selected, candidate, device)
                        except Exception:
                            failure = training_dir(candidate, selected) / "failure.json"
                            if not failure.exists() or read_json(failure).get("failure_kind") != "candidate_error":
                                raise
                peek_round(config, candidates, reports, train_stage, llm_call)
                _summarize_tasks(config, number)
                previous_score = read_json(folder / "summary.json")["global_incumbent_score"]
                if stop_after_round is not None and number >= stop_after_round:
                    break
                continue
            jobs = [(candidate, selected) for candidate in candidates for selected in task_configs(config)]
            if any(training_dir(candidate, selected).exists() for candidate, selected in jobs):
                raise RuntimeError("Interrupted round; automatic resume is disabled")
            for candidate, selected in jobs:
                try:
                    run_candidate(selected, candidate, device)
                except Exception:
                    # Ordinary model failures are outcomes; infrastructure/provenance errors propagate.
                    failure = training_dir(candidate, selected) / "failure.json"
                    if not failure.exists() or read_json(failure).get("failure_kind") != "candidate_error":
                        raise
            _summarize_tasks(config, number)
            previous_score = read_json(folder / "summary.json")["global_incumbent_score"]
        if stop_after_round is not None and number >= stop_after_round:
            break
    _finalize_search(config, reports)
    return root


def _freeze_task_champion(root, ranking, config, reports):

    # One winning architecture, with an independently selected checkpoint per task.
    for selected in task_configs(config):
        task = selected["_task"]
        winner = dict(ranking[0])
        winner["best_score"] = winner["tasks"][task]["best_score"]
        _freeze_champion(root, [winner], selected, reports["tasks"][task],
                         champion_dir=root / "champion" / task)
    path = root / "champion" / "champion.json"
    expected = {
        "candidate_id": ranking[0]["candidate_id"], "round_index": ranking[0]["round_index"],
        "validation_score": ranking[0]["best_score"],
        "task_validation_scores": {task: row["best_score"] for task, row in ranking[0]["tasks"].items()},
        "selection_metric": "mean_task_best_validation_bacc",
    }
    if path.exists() and read_json(path) != expected:
        raise ValueError("Frozen champion differs from the shared task ranking")
    if not path.exists():
        write_json(path, expected)


def _final_test_tasks(config, device):
    from .data import CombinedData
    from .train import test_champion

    if search_status(config)["status"] != "search_complete":
        raise RuntimeError("Final test is allowed only after the search is complete")
    _finalize_search(config, task_reports(config))
    root = Path(config["output"]["root"]) / config["search"]["name"] / "champion"
    results = {}
    for selected in task_configs(config):
        task = selected["_task"]
        folder = root / task
        metadata = read_json(folder / "champion.json")
        for name, expected in metadata["files_sha256"].items():
            if hashlib.sha256((folder / name).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Frozen champion file changed: {task}/{name}")
        path = folder / "test.json"
        results[task] = read_json(path) if path.exists() else test_champion(
            CombinedData(selected), folder, folder / "best.pt", folder, device)
        metadata["test_evaluated"] = True
        write_json(folder / "champion.json", metadata)
    result = {"tasks": results, "mean_task_balanced_accuracy":
              sum(row["mean_dataset_balanced_accuracy"] for row in results.values()) / len(results)}
    write_json(root / "test.json", result)
    return result


# Search lifecycle and public entry points

def _candidate_id(candidate: Path) -> str:
    metadata = read_json(candidate.with_suffix(".json"))
    candidate_id = metadata["candidate_id"]
    if candidate_id != candidate.stem:
        raise ValueError(f"Candidate id does not match its filename: {candidate}")
    return str(candidate_id)


def _result_reaches(path: Path, target_epochs: int, full_epochs: int) -> bool:
    if not path.exists():
        return False
    result = read_json(path)
    completed = result.get("epochs_completed")
    history = result.get("history")
    expected_status = "complete" if target_epochs == full_epochs else "partial"
    return (
        result.get("status") in ({"complete"} if expected_status == "complete" else {"partial", "complete"})
        and isinstance(completed, int)
        and completed >= target_epochs
        and isinstance(history, list)
        and len(history) == completed
        and str(result.get("candidate_id", "")).removeprefix("r") == path.parent.name
        and (path.parent / "latest.pt").is_file()
        and (path.parent / "best.pt").is_file()
        and result.get("test_used_for_selection") is False
    )


def _default_components() -> tuple[Any, Callable[..., Any], Callable[..., Any], Callable[..., Any], Callable[..., Any]]:
    from .data import CombinedData
    from .AutoBCI_HAD import generate_round
    from .AutoBCI_PEEK import select_candidates
    from .train import run as train_candidate
    from .train import verify

    return CombinedData, generate_round, select_candidates, verify, train_candidate


def _prepare_data(data: Any, device: str) -> None:
    data.define_labels()
    data.load(splits=("train", "valid"), device=device)
    if "test" in getattr(data, "splits", {}):
        raise RuntimeError("Search data unexpectedly contains the held-out test split")


def _verify_candidate(candidate: Path, config: dict[str, Any], report: dict[str, Any],
                      device: str, verify_fn: Callable[..., dict[str, Any]], search_dir: Path) -> bool:
    output = training_dir(candidate)
    verification = output / "verification.json"
    try:
        result = verify_fn(candidate, config, report, output, device)
        if not verification.exists():
            write_json(verification, result)
    except GeneratedCodeChanged:
        raise
    except Exception as exc:
        write_json(verification, {"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        return False
    return read_json(verification).get("status") in {"passed", "complete"}


def _train_candidate(data: Any, candidate: Path, config: dict[str, Any], device: str,
                     target_epochs: int, train_fn: Callable[..., dict[str, Any]], search_dir: Path) -> None:
    output = training_dir(candidate)
    result_path = output / "result.json"
    full_epochs = int(config["training"]["epochs"])
    if _result_reaches(result_path, target_epochs, full_epochs):
        from .train import validate_saved_training

        validate_saved_training(output, candidate, config, data.report)
        return
    result = train_fn(data, candidate, output, device, target_epochs)
    if not result_path.exists():
        write_json(result_path, result)
    if not _result_reaches(result_path, target_epochs, full_epochs):
        raise RuntimeError(f"Training did not reach epoch {target_epochs}: {candidate}")
    (output / "failure.json").unlink(missing_ok=True)


def _round_summary(round_index: int, candidates: Sequence[Path], promoted: Sequence[Path],
                   previous_incumbent: float | None, config: dict[str, Any]) -> dict[str, Any]:
    promoted_ids = {_candidate_id(path) for path in promoted}
    rows = []
    actual_epochs = 0
    observed_seconds = 0.0
    parent_ids: set[str] = set()
    for candidate in candidates:
        metadata = read_json(candidate.with_suffix(".json"))
        candidate_id = _candidate_id(candidate)
        result_path = training_dir(candidate) / "result.json"
        failure_path = training_dir(candidate) / "failure.json"
        result = read_json(result_path) if result_path.exists() else {}
        epochs = int(result.get("epochs_completed", 0))
        actual_epochs += epochs
        observed_seconds += float(result.get("seconds", 0.0))
        parent = metadata.get("parent_id")
        if parent:
            parent_ids.add(str(parent))
        failure = read_json(failure_path) if failure_path.exists() else None
        rows.append({
            "candidate_id": candidate_id,
            "parent_id": parent,
            "promoted": candidate_id in promoted_ids,
            "status": "failed" if failure is not None else result.get("status", "failed"),
            "failure": failure,
            "epochs_completed": epochs,
            "best_score": float(result["best_score"]) if "best_score" in result else None,
            "best_epoch": int(result["best_epoch"]) if "best_epoch" in result else None,
            "parameter_count": int(result.get("parameter_count", 0)),
            "seconds": float(result.get("seconds", 0.0)),
            "test_used_for_selection": bool(result.get("test_used_for_selection", False)),
        })
    full_epochs = int(config["training"]["epochs"])
    full_rows = [
        row for row in rows
        if row["promoted"] and row["status"] == "complete" and row["epochs_completed"] == full_epochs
    ]
    full_rows.sort(key=lambda row: (-row["best_score"], row["candidate_id"]))
    if not full_rows:
        raise RuntimeError(f"Round {round_index} has no full-budget candidates")
    if any(row["test_used_for_selection"] for row in rows):
        raise RuntimeError("A search result reports test use during selection")

    round_best = full_rows[0]["best_score"]
    global_score = round_best if previous_incumbent is None else max(previous_incumbent, round_best)
    improvement = None if previous_incumbent is None else round_best - previous_incumbent
    nominal_epochs = len(candidates) * int(config["training"]["epochs"])
    return {
        "status": "complete",
        "round_index": round_index,
        "forecast_enabled": peek_settings(config)["enabled"],
        "parent_ids": sorted(parent_ids),
        "promoted_ids": sorted(promoted_ids),
        "candidates": rows,
        "full_budget_ranking": full_rows,
        "best_new_full_budget_candidate_id": full_rows[0]["candidate_id"],
        "best_new_full_budget_score": round_best,
        "previous_global_incumbent_score": previous_incumbent,
        "global_incumbent_score": global_score,
        "improved_global_incumbent": previous_incumbent is None or round_best > previous_incumbent,
        "improvement_over_previous_incumbent": improvement,
        "compute": {
            "actual_epochs_trained": actual_epochs,
            "nominal_full_budget_epochs": nominal_epochs,
            "observed_candidate_training_seconds": observed_seconds,
        },
        "test_used_for_selection": False,
    }


def _validate_completed_round(
    search_dir: Path,
    round_index: int,
    config: dict[str, Any],
    report: dict[str, Any],
    previous_incumbent: float | None,
) -> dict[str, Any]:
    """Validate a completed summary against its current source and training artifacts."""
    from .train import validate_saved_training
    if is_multitask(config):
        return validate_round(config, round_index, report, previous_incumbent)

    round_dir = search_dir / f"round_{round_index:02d}"
    summary_path = round_dir / "summary.json"
    summary = read_json(summary_path)
    check_generation_identity(round_dir, config)
    candidates = sorted(
        (path for path in round_dir.glob("candidates/*.py")), key=_candidate_id
    )
    if len(candidates) != int(config["search"]["new_candidates"]):
        raise ValueError(f"Completed round {round_index} does not have the configured candidates")
    for candidate in candidates:
        check_candidate_source(candidate)
        output = training_dir(candidate)
        if (output / "result.json").is_file():
            validate_saved_training(output, candidate, config, report)
    promoted_by_id = {path.stem: path for path in candidates}
    promoted_ids = summary.get("promoted_ids")
    if not isinstance(promoted_ids, list) or any(
        candidate_id not in promoted_by_id for candidate_id in promoted_ids
    ):
        raise ValueError(f"Completed round {round_index} has invalid promoted candidates")
    expected = _round_summary(
        round_index,
        candidates,
        [promoted_by_id[candidate_id] for candidate_id in promoted_ids],
        previous_incumbent,
        config,
    )
    if summary != expected:
        raise ValueError(f"Completed round {round_index} summary is stale or inconsistent")
    return summary


def _global_pool(search_dir: Path, round_count: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for round_index in range(1, round_count + 1):
        summary_path = search_dir / f"round_{round_index:02d}" / "summary.json"
        summary = read_json(summary_path)
        if summary.get("status") != "complete":
            continue
        for row in summary["full_budget_ranking"]:
            item = dict(row)
            item["round_index"] = int(summary["round_index"])
            item["candidate"] = str(
                search_dir / f"round_{item['round_index']:02d}" / "candidates" / f"{item['candidate_id']}.py"
            )
            rows.append(item)
    rows.sort(key=lambda row: (-row["best_score"], row["candidate_id"]))
    return rows


def _freeze_champion(
    search_dir: Path,
    ranking: Sequence[dict[str, Any]],
    config: dict[str, Any],
    report: dict[str, Any],
    *, champion_dir: Path | None = None,
) -> None:
    from .train import training_identity, training_identity_sha256, validate_saved_training

    if not ranking:
        raise RuntimeError("Cannot freeze a champion without full-budget results")
    champion_dir = champion_dir or search_dir / "champion"
    metadata_path = champion_dir / "champion.json"
    if metadata_path.exists():
        champion = read_json(metadata_path)
        expected = ranking[0]
        if (
            champion.get("candidate_id") != expected["candidate_id"]
            or champion.get("round_index") != expected["round_index"]
            or champion.get("validation_score") != expected["best_score"]
            or champion.get("training_identity") != training_identity(config, report)
            or champion.get("training_identity_sha256")
            != training_identity_sha256(config, report)
        ):
            raise ValueError("Frozen champion does not match the validated search ranking")
        return
    source = Path(ranking[0]["candidate"])
    check_candidate_source(source)
    validate_saved_training(training_dir(source, config), source, config, report)
    result = read_json(training_dir(source, config) / "result.json")
    if result.get("status") != "complete":
        raise RuntimeError("Champion source is not a complete full-budget result")
    champion_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model_file(source), champion_dir / "model.py")
    shutil.copy2(training_dir(source, config) / "best.pt", champion_dir / "best.pt")
    hashes = {}
    for name in ("model.py", "best.pt"):
        hashes[name] = hashlib.sha256((champion_dir / name).read_bytes()).hexdigest()
    write_json(metadata_path, {
        "candidate_id": ranking[0]["candidate_id"],
        "source_candidate": str(source),
        "validation_score": ranking[0]["best_score"],
        "round_index": ranking[0]["round_index"],
        "training_identity": training_identity(config, report),
        "training_identity_sha256": training_identity_sha256(config, report),
        "files_sha256": hashes,
        "test_evaluated": False,
    })


def run_search(config: dict[str, Any], *, llm_call: Callable[..., Any] | None = None,
               components: tuple[Any, Callable[..., Any], Callable[..., Any], Callable[..., Any], Callable[..., Any]] | None = None,
               stop_after_round: int | None = None, generate_only: bool = False,
               device: str = "cuda") -> Path:
    """Run or resume one search. ``components`` exists for cheap synthetic tests."""
    if stop_after_round is not None and not 1 <= stop_after_round <= int(config["search"]["rounds"]):
        raise ValueError("stop_after_round must be within the configured rounds, starting at 1")
    if is_multitask(config):
        return _search_tasks(config, llm_call=llm_call, stop_after_round=stop_after_round,
                      generate_only=generate_only, device=device)
    CombinedData, generate_round, select_candidates, verify_fn, train_fn = components or _default_components()
    search_name = str(config["search"]["name"])
    search_dir = Path(config["output"]["root"]) / search_name
    search_dir.mkdir(parents=True, exist_ok=True)

    data = CombinedData(config)
    data.define_labels()
    report = dict(data.report)
    # HAD only needs metadata. Keep its prompt identical when resuming to train.
    if not generate_only:
        _prepare_data(data, device)
    previous_incumbent: float | None = None

    for round_index in range(1, int(config["search"]["rounds"]) + 1):
        round_dir = search_dir / f"round_{round_index:02d}"
        summary_path = round_dir / "summary.json"
        if summary_path.exists() and read_json(summary_path).get("status") == "complete":
            summary = _validate_completed_round(
                search_dir, round_index, config, report, previous_incumbent
            )
            previous_incumbent = float(summary["global_incumbent_score"])
            if stop_after_round is not None and round_index >= stop_after_round:
                break
            continue

        generated = generate_round(search_dir, round_index, config, report, llm_call=llm_call)
        check_generation_identity(round_dir, config)
        candidates = sorted((Path(path) for path in generated), key=_candidate_id)
        expected = int(config["search"]["new_candidates"])
        if len(candidates) != expected or len({_candidate_id(path) for path in candidates}) != expected:
            raise RuntimeError(f"Round {round_index} must contain {expected} distinct new candidates")

        if generate_only:
            return search_dir

        full_epochs = int(config["training"]["epochs"])
        forecast_enabled = peek_settings(config)["enabled"]
        if not forecast_enabled:
            def complete_candidate(candidate: Path) -> bool:
                output = training_dir(candidate)
                try:
                    check_candidate_source(candidate)
                    if _result_reaches(output / "result.json", full_epochs, full_epochs):
                        return True
                    if not _verify_candidate(candidate, config, report, device, verify_fn, search_dir):
                        return False
                    _train_candidate(data, candidate, config, device, full_epochs, train_fn, search_dir)
                    if not _result_reaches(output / "result.json", full_epochs, full_epochs):
                        raise RuntimeError(f"Candidate did not finish full training: {candidate}")
                    return True
                except (GeneratedCodeChanged, TrainingIdentityChanged):
                    raise
                except Exception as exc:
                    write_json(output / "failure.json", {
                        "stage": "full_candidate",
                        "error": f"{type(exc).__name__}: {exc}",
                    })
                    return False

            completed = [complete_candidate(candidate) for candidate in candidates]
            promoted = [candidate for candidate, passed in zip(candidates, completed) if passed]
            if len(promoted) < int(config["search"]["parent_count"]):
                raise RuntimeError(f"Round {round_index} has too few full-budget candidates")
        else:
            verification = [
                _verify_candidate(candidate, config, report, device, verify_fn, search_dir)
                for candidate in candidates
            ]
            eligible = [candidate for candidate, passed in zip(candidates, verification) if passed]

            def train_early(candidate: Path) -> bool:
                try:
                    _train_candidate(data, candidate, config, device, int(peek_settings(config)["early_epochs"]), train_fn, search_dir)
                    return True
                except (GeneratedCodeChanged, TrainingIdentityChanged):
                    raise
                except Exception as exc:
                    write_json(training_dir(candidate) / "failure.json", {
                        "stage": "early_training",
                        "error": f"{type(exc).__name__}: {exc}",
                    })
                    return False
            early_results = [train_early(candidate) for candidate in eligible]
            early_survivors = [candidate for candidate, passed in zip(eligible, early_results) if passed]
            if len(early_survivors) < int(peek_settings(config)["promote_count"]):
                raise RuntimeError(f"Round {round_index} has too few candidates surviving early training")

            selected = select_candidates(round_dir, early_survivors, config, report, llm_call=llm_call)
            promoted = [Path(path) for path in selected]
            if len(promoted) != int(peek_settings(config)["promote_count"]):
                raise RuntimeError(f"PEEK must promote exactly {peek_settings(config)['promote_count']} candidates")
            if not set(promoted).issubset(set(early_survivors)):
                raise RuntimeError("PEEK promoted a candidate outside the current round")

            def train_full(candidate: Path) -> None:
                try:
                    _train_candidate(data, candidate, config, device, full_epochs, train_fn, search_dir)
                except (GeneratedCodeChanged, TrainingIdentityChanged):
                    raise
                except Exception as exc:
                    write_json(training_dir(candidate) / "failure.json", {
                        "stage": "full_training",
                        "error": f"{type(exc).__name__}: {exc}",
                    })
                    raise
            for candidate in promoted:
                train_full(candidate)

        summary = _round_summary(round_index, candidates, promoted, previous_incumbent, config)
        write_json(summary_path, summary)
        previous_incumbent = float(summary["global_incumbent_score"])
        if stop_after_round is not None and round_index >= stop_after_round:
            break

    _finalize_search(config, report)
    return search_dir


def search_status(config: dict[str, Any]) -> dict[str, Any]:
    """Infer saved progress from round artifacts; do not infer whether a process is running."""
    root = Path(config["output"]["root"]) / config["search"]["name"]
    status = {
        "status": "not_started", "completed_rounds": 0,
        "test_evaluated": (root / "champion" / "test.json").is_file(),
    }
    for number in range(1, int(config["search"]["rounds"]) + 1):
        folder = root / f"round_{number:02d}"
        summary = folder / "summary.json"
        if summary.is_file() and read_json(summary).get("status") == "complete":
            status["completed_rounds"] += 1
            continue
        status["round_index"] = number
        candidates = sorted(folder.glob("candidates/*.py"))
        status["generated_candidates"] = len(candidates)
        if any(training_dir(model, selected).exists()
               for model in candidates for selected in task_configs(config)):
            status["status"] = "round_incomplete"
        elif len(candidates) == int(config["search"]["new_candidates"]):
            status["status"] = "generated"
        elif folder.exists():
            status["status"] = "generation_incomplete"
        elif status["completed_rounds"]:
            status["status"] = "round_complete"
        return status
    status["status"] = "search_complete"
    champion = root / "champion" / "champion.json"
    if champion.is_file():
        status["champion_id"] = read_json(champion)["candidate_id"]
    return status


def _finalize_search(config: dict[str, Any], report: dict[str, Any] | None = None) -> None:
    if search_status(config)["status"] == "search_complete":
        if report is None:
            from .data import CombinedData

            report = task_reports(config) if is_multitask(config) else CombinedData(config).report
        root = Path(config["output"]["root"]) / config["search"]["name"]
        previous_incumbent = None
        for round_index in range(1, int(config["search"]["rounds"]) + 1):
            summary = _validate_completed_round(
                root, round_index, config, report, previous_incumbent
            )
            previous_incumbent = float(summary["global_incumbent_score"])
        if is_multitask(config):
            _freeze_task_champion(root, _global_pool(root, int(config["search"]["rounds"])), config, report)
            return
        _freeze_champion(
            root,
            _global_pool(root, int(config["search"]["rounds"])),
            config,
            report,
        )


def summarize_round(config: dict[str, Any], round_index: int) -> Path:
    """Record completed models and explicit failures without repairing candidates."""
    if is_multitask(config):
        return _summarize_tasks(config, round_index)
    root = Path(config["output"]["root"]) / config["search"]["name"]
    if not 1 <= round_index <= int(config["search"]["rounds"]):
        raise ValueError("Round is outside the configured search")
    previous_score = None
    for previous in range(1, round_index):
        summary = read_json(root / f"round_{previous:02d}" / "summary.json")
        if summary["status"] != "complete":
            raise ValueError("Complete earlier rounds first")
        previous_score = summary["global_incumbent_score"]
    round_dir = root / f"round_{round_index:02d}"
    check_generation_identity(round_dir, config)
    candidates = sorted(round_dir.glob("candidates/*.py"))
    full_epochs = int(config["training"]["epochs"])
    if len(candidates) != int(config["search"]["new_candidates"]):
        raise RuntimeError("Round is missing model candidates")
    successful = []
    from .data import CombinedData
    from .train import validate_saved_training

    data = CombinedData(config)
    data.define_labels()
    if peek_settings(config)["enabled"]:
        from .AutoBCI_PEEK import select_candidates
        from .train import candidate_outcome
        if not (round_dir / "peek/promotion.json").is_file():
            raise RuntimeError("Round is missing its PEEK promotion")
        promoted = select_candidates(round_dir, candidates, config, data.report)
        completed = []
        for candidate in candidates:
            target = full_epochs if candidate in promoted else int(peek_settings(config)["early_epochs"])
            outcome = candidate_outcome(candidate, {**config, "_target_epochs": target}, data.report)
            if candidate in promoted and outcome == "complete":
                completed.append(candidate)
        if len(completed) < int(config["search"]["parent_count"]):
            raise RuntimeError("Too few successful full-budget candidates to select HAD parents")
        write_json(round_dir / "summary.json", _round_summary(round_index, candidates, promoted, previous_score, config))
        _finalize_search(config, data.report)
        return round_dir
    for candidate in candidates:
        check_candidate_source(candidate)
        output = training_dir(candidate)
        source_hash = hashlib.sha256(model_file(candidate).read_bytes()).hexdigest()
        if not _result_reaches(output / "result.json", full_epochs, full_epochs):
            failure_path = output / "failure.json"
            if failure_path.exists():
                failure = read_json(failure_path)
                if (failure.get("stage") in {"verification", "training"}
                        and failure.get("candidate_sha256") == source_hash
                        and failure.get("repair_attempted") is False):
                    continue
            raise RuntimeError(f"Candidate did not complete full training: {candidate.stem}; see terminal output")
        validate_saved_training(output, candidate, config, data.report)
        successful.append(candidate)
    if len(successful) < int(config["search"]["parent_count"]):
        raise RuntimeError("Too few successful full-budget candidates to select HAD parents")
    write_json(round_dir / "summary.json", _round_summary(round_index, candidates, successful, previous_score, config))
    _finalize_search(config, data.report)
    return round_dir


def final_test(config: dict[str, Any], device: str = "cuda") -> dict[str, Any]:
    """Evaluate the frozen champion once, after every search round is complete."""
    if is_multitask(config):
        return _final_test_tasks(config, device)
    from .data import CombinedData
    from .train import test_champion
    from .train import training_identity, training_identity_sha256

    search_dir = Path(config["output"]["root"]) / str(config["search"]["name"])
    if search_status(config)["status"] != "search_complete":
        raise RuntimeError("Final test is allowed only after the search is complete")
    data = CombinedData(config)
    data.define_labels()
    _finalize_search(config, data.report)
    champion_dir = search_dir / "champion"
    result_path = champion_dir / "test.json"
    champion = read_json(champion_dir / "champion.json")
    if (
        champion.get("training_identity") != training_identity(config, data.report)
        or champion.get("training_identity_sha256")
        != training_identity_sha256(config, data.report)
    ):
        raise ValueError("Frozen champion does not match the active protocol and dataset")
    source = (
        search_dir
        / f"round_{int(champion['round_index']):02d}"
        / "candidates"
        / f"{champion['candidate_id']}.py"
    )
    if not source.is_file():
        raise ValueError("Frozen champion source is missing from the active search")
    for name, expected in champion["files_sha256"].items():
        actual = hashlib.sha256((champion_dir / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Frozen champion file changed: {name}")
    if result_path.exists():
        champion["test_evaluated"] = True
        write_json(champion_dir / "champion.json", champion)
        return read_json(result_path)
    result = test_champion(
        data,
        champion_dir,
        champion_dir / "best.pt",
        champion_dir,
        device,
    )
    if not result_path.exists():
        write_json(result_path, result)
    champion["test_evaluated"] = True
    write_json(champion_dir / "champion.json", champion)
    return read_json(result_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=("search", "generate", "summarize", "final-test", "status"), default="search")
    parser.add_argument("--config", default="config/experiments/gpt-5.6-sol.yaml")
    parser.add_argument("--search", help="independent search name (overrides search.name)")
    parser.add_argument("--device", help="operational device, for example cpu or cuda:0")
    parser.add_argument("--round", type=int, dest="round_index")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.search:
        config["search"]["name"] = args.search
    if args.command in {"search", "final-test"} and args.device is None:
        parser.error("search and final-test require --device, for example cuda:0")

    if args.command == "search":
        print(run_search(config, device=args.device))
    elif args.command == "generate":
        print(run_search(config, generate_only=True))
    elif args.command == "summarize":
        if args.round_index is None:
            parser.error("summarize requires --round")
        print(summarize_round(config, args.round_index))
    elif args.command == "final-test":
        print(json.dumps(final_test(config, device=args.device), indent=2, sort_keys=True))
    else:
        print(json.dumps(search_status(config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
