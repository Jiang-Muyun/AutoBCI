# License: see LICENSE.
"""
Forecast full-budget performance from early learning curves and promote architectures with PEEK.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Sequence
import ast
import json
import math

from .AutoBCI_HAD import _call_or_inject, _without_test_or_future
from .common import model_file, peek_settings, read_json, training_dir, validate_peek, write_json


HERE = Path(__file__).resolve().parent
PROMPT_PATH = HERE / "prompt" / "peek_prompt.md"


def _clean_code(source: str) -> str:
    """Drop comments and docstrings so model-authored prose cannot leak evidence."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.body and isinstance(node.body[0], ast.Expr):
                value = node.body[0].value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    node.body.pop(0)
                    if not node.body:
                        node.body.append(ast.Pass())
    code = ast.unparse(ast.fix_missing_locations(tree)) + "\n"
    compile(code, "<peek_architecture>", "exec")
    return code


def _candidate_id(path: Path) -> str:
    metadata = read_json(path.with_suffix(".json"))
    candidate_id = metadata.get("candidate_id")
    if candidate_id != path.stem:
        raise ValueError(f"candidate ID does not match its filename: {path}")
    return candidate_id


def _forecast_schema(candidate_ids: Sequence[str], promote_count: int) -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": ["predictions", "selected_ids"],
        "properties": {
            "predictions": {
                "type": "array",
                "minItems": len(candidate_ids),
                "maxItems": len(candidate_ids),
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["candidate_id", "predicted_best_score"],
                    "properties": {
                        "candidate_id": {"type": "string", "enum": list(candidate_ids)},
                        "predicted_best_score": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                },
            },
            "selected_ids": {
                "type": "array",
                "minItems": promote_count,
                "maxItems": promote_count,
                "items": {"type": "string", "enum": list(candidate_ids)},
            },
        },
    }


def _early_candidates(
    candidates: Sequence[Path], config: dict[str, Any]
) -> tuple[list[Path], list[dict[str, Any]], dict[str, float]]:
    from .run import is_multitask, task_configs

    if is_multitask(config):
        per_task = {}
        for selected in task_configs(config):
            _, rows, bounds = _early_candidates(candidates, selected)
            per_task[selected["_task"]] = ({row["candidate_id"]: row for row in rows}, bounds)
        eligible, payload, lower_bounds = [], [], {}
        for path in candidates:
            candidate_id = _candidate_id(path)
            if not all(candidate_id in rows for rows, _ in per_task.values()):
                continue
            eligible.append(path)
            payload.append({
                "candidate_id": candidate_id,
                "model_code": _clean_code(model_file(path).read_text()),
                "observed_tasks": {task: rows[candidate_id]["observed_epochs"]
                                   for task, (rows, _) in per_task.items()},
            })
            lower_bounds[candidate_id] = sum(bounds[candidate_id] for _, bounds in per_task.values()) / len(per_task)
        return eligible, payload, lower_bounds

    early_epochs = int(peek_settings(config)["early_epochs"])
    eligible_paths: list[Path] = []
    payload: list[dict[str, Any]] = []
    lower_bounds: dict[str, float] = {}
    seen: set[str] = set()
    for supplied in candidates:
        path = Path(supplied)
        candidate_id = _candidate_id(path)
        if candidate_id in seen:
            raise ValueError(f"duplicate candidate input: {candidate_id}")
        seen.add(candidate_id)
        result_path = training_dir(path, config) / "result.json"
        if not result_path.is_file() or (result_path.parent / "failure.json").exists():
            continue
        result = read_json(result_path)
        history = result.get("history")
        if (
            result.get("status") != "partial"
            or result.get("epochs_completed") != early_epochs
            or result.get("test_used_for_selection") is not False
            or not isinstance(history, list)
            or len(history) != early_epochs
        ):
            continue

        curve = []
        scores = []
        valid = True
        for expected_epoch, row in enumerate(history, start=1):
            if not isinstance(row, dict) or row.get("epoch") != expected_epoch:
                valid = False
                break
            loss = row.get("loss")
            validation = row.get("validation")
            score = validation.get("mean_dataset_balanced_accuracy") if isinstance(validation, dict) else None
            if (
                isinstance(loss, bool)
                or not isinstance(loss, (int, float))
                or not math.isfinite(float(loss))
                or isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(float(score))
                or not 0.0 <= float(score) <= 1.0
            ):
                valid = False
                break
            scores.append(float(score))
            curve.append(
                {
                    "epoch": expected_epoch,
                    "train_loss": float(loss),
                    "validation_mean_dataset_balanced_accuracy": float(score),
                }
            )
        if not valid:
            continue
        eligible_paths.append(path)
        payload.append(
            {
                "candidate_id": candidate_id,
                "model_code": _clean_code((model_file(path)).read_text()),
                "observed_epochs": curve,
            }
        )
        lower_bounds[candidate_id] = max(scores)
    return eligible_paths, payload, lower_bounds


def _validate_forecast(
    response: Any,
    candidate_ids: Sequence[str],
    lower_bounds: dict[str, float],
    promote_count: int,
) -> dict[str, Any]:
    if not isinstance(response, dict) or set(response) != {"predictions", "selected_ids"}:
        raise ValueError("PEEK response must contain only predictions and selected_ids")
    predictions = response["predictions"]
    selected = response["selected_ids"]
    if not isinstance(predictions, list) or len(predictions) != len(candidate_ids):
        raise ValueError("PEEK must predict every eligible candidate exactly once")
    expected = set(candidate_ids)
    scores: dict[str, float] = {}
    for index, prediction in enumerate(predictions):
        if not isinstance(prediction, dict) or set(prediction) != {
            "candidate_id", "predicted_best_score"
        }:
            raise ValueError(f"prediction {index} has unexpected or missing fields")
        candidate_id = prediction["candidate_id"]
        score = prediction["predicted_best_score"]
        if candidate_id not in expected or candidate_id in scores:
            raise ValueError("PEEK prediction IDs must be complete and nonduplicate")
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
            or not 0.0 <= float(score) <= 1.0
        ):
            raise ValueError(f"invalid forecast for {candidate_id}")
        score = float(score)
        if score < lower_bounds[candidate_id]:
            raise ValueError(
                f"forecast for {candidate_id} is below its observed early best score"
            )
        scores[candidate_id] = score
    if set(scores) != expected:
        raise ValueError("PEEK prediction IDs are incomplete")
    if (
        not isinstance(selected, list)
        or len(selected) != promote_count
        or len(set(selected)) != promote_count
        or any(candidate_id not in expected for candidate_id in selected)
    ):
        raise ValueError("PEEK selected_ids contain an invalid promotion")
    expected_top = sorted(candidate_ids, key=lambda candidate_id: (-scores[candidate_id], candidate_id))[
        :promote_count
    ]
    if set(selected) != set(expected_top):
        raise ValueError("PEEK must promote the candidates with the highest forecasts")
    return {
        "predictions": [
            {"candidate_id": row["candidate_id"], "predicted_best_score": float(row["predicted_best_score"])}
            for row in predictions
        ],
        "selected_ids": list(selected),
    }


def _resume_promotion(
    promotion: dict[str, Any], candidates: Sequence[Path], promote_count: int
) -> list[Path]:
    available: dict[str, Path] = {}
    for supplied in candidates:
        path = Path(supplied)
        candidate_id = _candidate_id(path)
        if candidate_id in available:
            raise ValueError(f"duplicate candidate input: {candidate_id}")
        available[candidate_id] = path
    candidate_ids = promotion.get("candidate_ids")
    selected = promotion.get("selected_ids")
    if (
        not isinstance(candidate_ids, list)
        or len(candidate_ids) != len(set(candidate_ids))
        or not set(candidate_ids).issubset(available)
        or not isinstance(selected, list)
        or len(selected) != promote_count
        or len(set(selected)) != promote_count
        or not set(selected).issubset(set(candidate_ids))
    ):
        raise ValueError("saved PEEK promotion is invalid for this round")
    predictions = promotion.get("predictions")
    if not isinstance(predictions, list) or len(predictions) != len(candidate_ids):
        raise ValueError("saved PEEK predictions are incomplete")
    scores: dict[str, float] = {}
    for row in predictions:
        if not isinstance(row, dict) or set(row) != {"candidate_id", "predicted_best_score"}:
            raise ValueError("saved PEEK prediction is malformed")
        candidate_id = row["candidate_id"]
        score = row["predicted_best_score"]
        if (
            candidate_id not in candidate_ids
            or candidate_id in scores
            or isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
            or not 0.0 <= float(score) <= 1.0
        ):
            raise ValueError("saved PEEK prediction is invalid")
        scores[candidate_id] = float(score)
    expected_top = sorted(candidate_ids, key=lambda item: (-scores[item], item))[:promote_count]
    if set(selected) != set(expected_top):
        raise ValueError("saved PEEK promotion is inconsistent with its forecasts")
    return [available[candidate_id] for candidate_id in selected]


def _render_prompt(
    payload: Sequence[dict[str, Any]], report: dict[str, Any], config: dict[str, Any]
) -> str:
    report_fields = ("input_shape", "labels", "dataset_order", "datasets", "source_classes", "counts")
    def describe(values):
        return {key: _without_test_or_future(values[key]) for key in report_fields if key in values}
    data_protocol = ({"tasks": {task: describe(values) for task, values in report["tasks"].items()}}
                     if "tasks" in report else describe(report))
    protocol = {
        "early_epochs": int(peek_settings(config)["early_epochs"]),
        "promote_count": int(peek_settings(config)["promote_count"]),
        "full_epochs": int(config["training"]["epochs"]),
        "max_parameters": int(config["search"]["max_parameters"]),
        "training": config["training"],
        "optimizer": "AdamW",
        "schedule": (
            f"CosineAnnealingLR, T_max={int(config['training']['epochs'])}, "
            "eta_min=0, stepped after each epoch"
        ),
        "normalization": "per-trial, per-channel temporal z-score; standard deviation clamped at 1e-6; center crop or pad to each catalog's input shape",
        "loss": "cross entropy over all output classes; no dataset ID is supplied to the model",
        "evaluation": "mask invalid dataset logits to negative infinity before argmax",
        "selection_metric": "best checkpoint's unweighted mean per-dataset validation BACC at one common epoch",
        "data": data_protocol,
    }
    if "tasks" in report:
        protocol["selection_metric"] = (
            "mean_task_best_validation_bacc: equal-weight mean of each task's best common-epoch "
            "mean per-dataset validation balanced accuracy; checkpoint epochs may differ between tasks"
        )
    return (
        PROMPT_PATH.read_text().rstrip()
        + "\n\nPROTOCOL\n"
        + json.dumps(protocol, indent=2, sort_keys=True)
        + "\n\nCANDIDATES\n"
        + json.dumps(payload, indent=2, sort_keys=True)
        + "\n"
    )


def select_candidates(
    round_dir: Path,
    candidates: Sequence[Path],
    config: dict[str, Any],
    report: dict[str, Any],
    llm_call: Callable[[str, dict[str, Any]], Any] | None = None,
) -> list[Path]:
    """Promote candidates using only source and the configured early epochs."""
    round_dir = Path(round_dir)
    settings = validate_peek(config)
    early_epochs = int(settings["early_epochs"])
    full_epochs = int(config["training"]["epochs"])
    if not 0 < early_epochs < full_epochs:
        raise ValueError("PEEK requires 0 < early_epochs < epochs")
    peek_dir = round_dir / "peek"
    peek_dir.mkdir(parents=True, exist_ok=True)
    promote_count = int(settings["promote_count"])
    request_path = peek_dir / "request.json"
    if request_path.exists():
        request = read_json(request_path)
        if (request["early_epochs"] != early_epochs or request["full_epochs"] != full_epochs
                or request.get("peek_settings", settings) != settings):
            raise ValueError("Saved PEEK request differs from the configured protocol")

    promotion_path = peek_dir / "promotion.json"
    if promotion_path.is_file():
        return _resume_promotion(read_json(promotion_path), candidates, promote_count)

    accepted_path = peek_dir / "response.json"
    request_path = peek_dir / "request.json"
    if accepted_path.is_file():
        request = read_json(request_path)
        candidate_ids = request["candidate_ids"]
        lower_bounds = request["observed_early_best"]
        response = _validate_forecast(
            read_json(accepted_path), candidate_ids, lower_bounds, promote_count
        )
        promotion = {
            "status": "selected_for_full_training",
            "candidate_ids": candidate_ids,
            **response,
            "test_used_for_selection": False,
        }
        write_json(promotion_path, promotion)
        return _resume_promotion(promotion, candidates, promote_count)

    eligible, payload, lower_bounds = _early_candidates(candidates, config)
    if len(eligible) < promote_count:
        raise ValueError(f"fewer than {promote_count} candidates have valid early-budget partial results")
    candidate_ids = [row["candidate_id"] for row in payload]
    schema = _forecast_schema(candidate_ids, promote_count)
    prompt = _render_prompt(payload, report, config)
    (peek_dir / "prompt.md").write_text(prompt)
    write_json(peek_dir / "schema.json", schema)
    write_json(
        request_path,
        {
            "status": "forecast_requested",
            "candidate_ids": candidate_ids,
            "observed_early_best": lower_bounds,
            "early_epochs": early_epochs,
            "full_epochs": int(config["training"]["epochs"]),
            "peek_settings": settings,
            "test_used_for_selection": False,
        },
    )
    forecast_config = deepcopy(config)
    for key in ("model", "harness", "harness_settings", "reasoning_effort", "generation_timeout_seconds"):
        if key in settings:
            forecast_config["search"][key] = settings[key]
    raw_response = _call_or_inject(llm_call, prompt, schema, peek_dir / "llm", forecast_config)
    response = _validate_forecast(raw_response, candidate_ids, lower_bounds, promote_count)
    write_json(accepted_path, response)
    promotion = {
        "status": "selected_for_full_training",
        "candidate_ids": candidate_ids,
        **response,
        "test_used_for_selection": False,
    }
    write_json(promotion_path, promotion)
    return _resume_promotion(promotion, candidates, promote_count)
