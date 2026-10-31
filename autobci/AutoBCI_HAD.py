"""
Generate and refine EEG architectures with Heuristic Architecture Discovery (HAD).

Copyright 2022 Centre for Brain Computing Research (CBCR), College of Computing and Data Science (CCDS), Nanyang Technological University (NTU);
licensed under the CBCR License 1.0 (see LICENSE).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence
import ast
import json
import math
import re

from .common import (
    check_candidate_source,
    original_response,
    peek_settings,
    read_json,
    resolve_config_path,
    training_dir,
    write_json,
)
from .harness import call_llm


HERE = Path(__file__).resolve().parent
PROMPT_PATH = HERE / "prompt" / "had_prompt.md"
NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{1,47}$")


def _round_number(path: Path) -> int:
    match = re.fullmatch(r"round_(\d+)", path.name)
    return int(match.group(1)) if match else -1


def _completed_parents(
    search_dir: Path, config: dict[str, Any], before_round: int | None = None
) -> list[dict[str, Any]]:
    from .run import is_multitask, completed_parents
    if is_multitask(config):
        return completed_parents(search_dir, config, before_round)
    full_epochs = int(config["training"]["epochs"])
    eligible: list[dict[str, Any]] = []
    for round_dir in sorted(Path(search_dir).glob("round_*")):
        round_index = _round_number(round_dir)
        if round_index < 1 or (before_round is not None and round_index >= before_round):
            continue
        summary_path = round_dir / "summary.json"
        if not summary_path.is_file():
            continue
        summary = read_json(summary_path)
        if summary.get("status") != "complete":
            continue
        ranking = summary.get("full_budget_ranking")
        if not isinstance(ranking, list):
            continue
        authorized_ids = {
            row.get("candidate_id") for row in ranking if isinstance(row, dict)
        }
        candidates_dir = round_dir / "candidates"
        if not candidates_dir.is_dir():
            continue
        for model_path in sorted(candidates_dir.glob("*.py")):
            result_path = training_dir(model_path) / "result.json"
            checkpoint_path = training_dir(model_path) / "best.pt"
            metadata_path = model_path.with_suffix(".json")
            if not all(path.is_file() for path in (result_path, checkpoint_path, metadata_path, model_path)):
                continue
            result = read_json(result_path)
            history = result.get("history")
            score = result.get("best_score")
            best_epoch = result.get("best_epoch")
            if (
                result.get("status") != "complete"
                or result.get("epochs_completed") != full_epochs
                or not isinstance(history, list)
                or len(history) != full_epochs
                or isinstance(best_epoch, bool)
                or not isinstance(best_epoch, int)
                or not 1 <= best_epoch <= full_epochs
                or result.get("test_used_for_selection") is not False
                or isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(float(score))
                or not 0.0 <= float(score) <= 1.0
            ):
                continue
            curve = []
            valid_history = True
            for expected_epoch, row in enumerate(history, start=1):
                validation = row.get("validation") if isinstance(row, dict) else None
                value = (
                    validation.get("mean_dataset_balanced_accuracy")
                    if isinstance(validation, dict)
                    else None
                )
                if (
                    not isinstance(row, dict)
                    or row.get("epoch") != expected_epoch
                    or isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or not 0.0 <= float(value) <= 1.0
                ):
                    valid_history = False
                    break
                curve.append(float(value))
            if (
                not valid_history
                or not math.isclose(float(score), max(curve), rel_tol=0.0, abs_tol=1e-12)
                or not math.isclose(float(score), curve[best_epoch - 1], rel_tol=0.0, abs_tol=1e-12)
            ):
                continue
            metadata = read_json(metadata_path)
            candidate_id = metadata.get("candidate_id")
            if candidate_id != model_path.stem or candidate_id not in authorized_ids:
                continue
            check_candidate_source(model_path)
            eligible.append(
                {
                    "candidate_path": model_path,
                    "candidate_id": candidate_id,
                    "round_index": round_index,
                    "best_score": float(score),
                    "best_epoch": best_epoch,
                    "model_code": model_path.read_text(),
                    "metadata": metadata,
                    "result": result,
                }
            )
    eligible.sort(key=lambda row: (-row["best_score"], row["candidate_id"]))
    return eligible


def select_parents(search_dir: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the global top two measured full-budget candidates in one search."""
    count = int(config["search"]["parent_count"])
    if count != 2:
        raise ValueError("this experiment requires exactly two HAD parents")
    return _completed_parents(Path(search_dir), config)[:count]


def _without_test_or_future(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_test_or_future(item)
            for key, item in value.items()
            if "test" not in str(key).lower() and "future" not in str(key).lower()
            and key not in {"sample_identity_sha256", "bytes"}
        }
    if isinstance(value, list):
        return [_without_test_or_future(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _candidate_schema(count: int, parent_ids: Sequence[str]) -> dict[str, Any]:
    parent_schema: dict[str, Any]
    if parent_ids:
        parent_schema = {"type": ["string", "null"]}
    else:
        parent_schema = {"type": "null"}
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": ["candidates"],
        "properties": {
            "candidates": {
                "type": "array",
                "minItems": count,
                "maxItems": count,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "name", "family_id", "role", "parent_id", "intended_change",
                        "hypothesis", "model_code",
                    ],
                    "properties": {
                        "name": {"type": "string", "pattern": "^[a-z][a-z0-9_]{1,47}$"},
                        "family_id": {"type": "string", "pattern": "^[a-z][a-z0-9_]{1,47}$"},
                        "role": {"type": "string", "enum": ["refinement", "exploration"]},
                        "parent_id": parent_schema,
                        "intended_change": {"type": "string", "minLength": 1},
                        "hypothesis": {"type": "string", "minLength": 1},
                        "model_code": {
                            "type": "string", "minLength": 1,
                            "description": "Complete executable Python source, including build_model; no placeholders or omitted code.",
                        },
                    },
                },
            }
        },
    }


def _validate_response(
    response: Any,
    *,
    round_index: int,
    parent_ids: Sequence[str],
    parent_codes: dict[str, str],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    if not isinstance(response, dict) or set(response) != {"candidates"}:
        raise ValueError("HAD response must contain only candidates")
    rows = response["candidates"]
    search = config["search"]
    initial_count = int(search["new_candidates"])
    refinements = int(search["refinements_per_parent"])
    explorations = int(search["explorations"])
    expected_count = initial_count if round_index == 1 else len(parent_ids) * refinements + explorations
    if not isinstance(rows, list) or len(rows) != expected_count:
        raise ValueError(f"expected exactly {expected_count} HAD candidates")

    names: set[str] = set()
    allocations = {parent_id: 0 for parent_id in parent_ids}
    exploration_count = 0
    sources: set[str] = set()
    required = {
        "name", "family_id", "role", "parent_id", "intended_change", "hypothesis", "model_code"
    }
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != required:
            raise ValueError(f"candidate {index} has unexpected or missing fields")
        if not isinstance(row["name"], str) or not NAME_PATTERN.fullmatch(row["name"]):
            raise ValueError(f"candidate {index} has an invalid name")
        if row["name"] in names:
            raise ValueError(f"duplicate candidate name {row['name']}")
        names.add(row["name"])
        if not isinstance(row["family_id"], str) or not NAME_PATTERN.fullmatch(row["family_id"]):
            raise ValueError(f"candidate {index} has an invalid family_id")
        for field in ("intended_change", "hypothesis", "model_code"):
            if not isinstance(row[field], str) or not row[field].strip():
                raise ValueError(f"candidate {index} has an invalid {field}")
        name = row["name"]
        try:
            tree = ast.parse(row["model_code"], filename=f"{name}/model.py")
            compile(tree, f"{name}/model.py", "exec")
        except SyntaxError as exc:
            raise ValueError(f"Candidate {name}: invalid Python at line {exc.lineno}: {exc.msg}") from exc
        if not any(isinstance(node, ast.FunctionDef) and node.name == "build_model" for node in tree.body):
            raise ValueError(f"Candidate {name}: incomplete model_code; missing top-level build_model()")
        normalized_source = row["model_code"].strip()
        if normalized_source in sources:
            raise ValueError("HAD proposed duplicate model sources")
        sources.add(normalized_source)

        if round_index == 1:
            if row["role"] != "exploration" or row["parent_id"] is not None:
                raise ValueError("round 01 candidates must be fresh explorations")
            exploration_count += 1
        elif row["role"] == "refinement":
            parent_id = row["parent_id"]
            if parent_id not in allocations:
                raise ValueError(f"candidate {index} has an invalid parent_id")
            if normalized_source == parent_codes[parent_id].strip():
                raise ValueError("a refinement must change its parent's model source")
            allocations[parent_id] += 1
        elif row["role"] == "exploration":
            if row["parent_id"] is not None:
                raise ValueError("exploration candidates cannot have a parent")
            exploration_count += 1
        else:
            raise ValueError(f"candidate {index} has an invalid role")

    if round_index == 1:
        if exploration_count != initial_count:
            raise ValueError("round 01 must contain exactly eight fresh explorations")
    elif any(count != refinements for count in allocations.values()) or exploration_count != explorations:
        raise ValueError("later rounds require three refinements per parent and two explorations")
    return rows


def _resolve_model(config: dict[str, Any]) -> str:
    return str(config["search"]["model"])


def _call_or_inject(
    llm_call: Callable[[str, dict[str, Any]], Any] | None,
    prompt: str,
    schema: dict[str, Any],
    output_dir: Path,
    config: dict[str, Any],
) -> Any:
    if llm_call is None:
        search = config["search"]
        settings = search.get("harness_settings")
        kwargs = {}
        if settings:
            kwargs["settings_path"] = resolve_config_path(settings, config.get("_config_path", ""))
        return call_llm(
            prompt,
            schema,
            output_dir,
            model=_resolve_model(config),
            harness=search.get("harness"),
            reasoning_effort=str(search["reasoning_effort"]),
            timeout_seconds=int(search["generation_timeout_seconds"]),
            **kwargs,
        )
    write_json(output_dir / "invocation.json", {"status": "injected", "model": None})
    response = llm_call(prompt, schema)
    if isinstance(response, str):
        (output_dir / "raw_response.json").write_text(response)
        return json.loads(response)
    (output_dir / "raw_response.json").write_text(json.dumps(response, indent=2) + "\n")
    return response


def _generation_protocol(round_index: int, report: dict, config: dict) -> dict:
    """Keep initial proposals unchanged; introduce task-specific feedback in later rounds."""
    search = config["search"]
    protocol = {
        "training": config["training"],
        "max_parameters": int(search["max_parameters"]),
    }
    if "peek" in config or peek_settings(config)["enabled"]:
        protocol["peek"] = peek_settings(config)
    if search.get("prompt_injection"):
        protocol["prompt_injection"] = search["prompt_injection"].strip()
    categories = []
    for values in report.get("tasks", {"combined": report}).values():
        for label in values.get("labels", []):
            family, name = label.split("::", 1)
            prefix = {"motor_imagery": "MI", "emotion": "Emotion", "sleep": "Sleep"}.get(family, family)
            category = f"{prefix}:{name}"
            if category not in categories:
                categories.append(category)
    if categories:
        protocol["class_categories"] = categories
    if round_index > 1:
        protocol["selection_metric"] = "best checkpoint's unweighted mean per-dataset validation BACC at one common epoch"
    if "tasks" in report and round_index > 1:
        protocol["selection_metric"] = "mean_task_best_validation_bacc"
        protocol["independent_training"] = (
            "Train identical architecture source from scratch separately for each task. "
            "Use each task's own n_classes, labels, optimizer and checkpoints. "
            "Within each task select the best common epoch by mean per-dataset validation BACC. "
            "Rank architectures by the equal-weight mean of all task best scores. "
            "All tasks must finish the full budget. No weight transfer between tasks."
        )
    return protocol


def _render_prompt(
    round_index: int,
    report: dict[str, Any],
    parents: Sequence[dict[str, Any]],
    config: dict[str, Any],
) -> str:
    search = config["search"]
    protocol = _generation_protocol(round_index, report, config)
    evidence = []
    for parent in parents:
        if "task_results" in parent:
            task_evidence = {}
            for task, result in parent["task_results"].items():
                best_row = result["history"][result["best_epoch"] - 1]
                task_evidence[task] = {
                    "best_score": result["best_score"], "best_epoch": result["best_epoch"],
                    "parameter_count": result["parameter_count"],
                    "mean_validation_bacc_curve": [
                        f"epoch {row['epoch']}: {row['validation']['mean_dataset_balanced_accuracy']:.4f}"
                        for row in result["history"]
                    ],
                    "per_dataset_bacc_at_best_epoch": {
                        f"dataset_{index:02d}": metrics["balanced_accuracy"]
                        for index, (_, metrics) in enumerate(
                            sorted(best_row["validation"]["per_dataset"].items()), start=1
                        )
                    },
                }
            evidence.append({"candidate_id": parent["candidate_id"], "best_score": parent["best_score"],
                             "tasks": task_evidence, "model_code": parent["model_code"]})
            continue
        result = parent["result"]
        mean_curve = [
            f"epoch {row['epoch']}: {row['validation']['mean_dataset_balanced_accuracy']:.4f}"
            for row in result["history"]
        ]
        best_row = next(
            row for row in result["history"] if row["epoch"] == result["best_epoch"]
        )
        per_dataset = best_row["validation"].get("per_dataset", {})
        per_dataset_bacc = {
            f"dataset_{index:02d}": (
                metrics.get("balanced_accuracy")
                if isinstance(metrics, dict)
                else metrics
            )
            for index, (_, metrics) in enumerate(sorted(per_dataset.items()), start=1)
        }
        evidence.append(
            {
                "candidate_id": parent["candidate_id"],
                "best_score": parent["best_score"],
                "best_epoch": parent["best_epoch"],
                "parameter_count": result.get("parameter_count"),
                "mean_validation_bacc_curve": mean_curve,
                "per_dataset_bacc_at_best_epoch": per_dataset_bacc,
                "model_code": parent["model_code"],
            }
        )
    allocation = (
        f"Generate exactly {int(search['new_candidates'])} fresh explorations with null parent_id."
        if round_index == 1
        else "Generate exactly three single-component refinements of each supplied parent "
        "and exactly two fresh explorations. Same-family refinements are allowed."
    )
    return (
        PROMPT_PATH.read_text().rstrip()
        + "\n\nROUND\n"
        + str(round_index)
        + "\n\nALLOCATION\n"
        + allocation
        + "\n\nPROTOCOL\n"
        + json.dumps(protocol, indent=2, sort_keys=True)
        + "\n\nMEASURED PARENT EVIDENCE\n"
        + json.dumps(evidence, indent=2, sort_keys=True)
        + "\n"
    )


def _candidate_id(round_index: int, row: dict[str, Any], parent_ids: Sequence[str], ordinal: int) -> str:
    if row["role"] == "exploration":
        same_role = ordinal
        prefix = f"r{round_index:02d}_x{same_role:02d}"
    else:
        parent_slot = parent_ids.index(row["parent_id"])
        prefix = f"r{round_index:02d}_p{parent_slot}_{ordinal:02d}"
    return f"{prefix}_{row['name']}"


def _materialize(
    round_dir: Path,
    rows: Sequence[dict[str, Any]],
    round_index: int,
    parent_ids: Sequence[str],
) -> list[Path]:
    root = round_dir / "candidates"
    root.mkdir(parents=True, exist_ok=True)
    role_ordinals: dict[tuple[str, str | None], int] = {}
    written: list[Path] = []
    for row in rows:
        key = (row["role"], row["parent_id"])
        ordinal = role_ordinals.get(key, 0)
        role_ordinals[key] = ordinal + 1
        candidate_id = _candidate_id(round_index, row, parent_ids, ordinal)
        model_path = root / f"{candidate_id}.py"
        metadata_path = model_path.with_suffix(".json")
        metadata = {
            "candidate_id": candidate_id,
            "round_index": round_index,
            "name": row["name"],
            "family_id": row["family_id"],
            "role": row["role"],
            "parent_id": row["parent_id"],
            "intended_change": row["intended_change"],
            "hypothesis": row["hypothesis"],
        }
        code = row["model_code"].rstrip() + "\n"
        if model_path.exists() or metadata_path.exists():
            if read_json(metadata_path) != metadata or model_path.read_text() != code:
                raise ValueError(f"existing candidate differs from accepted response: {model_path}")
        else:
            model_path.write_text(code)
            write_json(metadata_path, metadata)
        written.append(model_path)
    return written


def generate_round(
    search_dir: Path,
    round_index: int,
    config: dict[str, Any],
    report: dict[str, Any],
    llm_call: Callable[[str, dict[str, Any]], Any] | None = None,
) -> list[Path]:
    """Generate and materialize one deterministic eight-candidate HAD round."""
    if round_index < 1:
        raise ValueError("round numbering starts at 1")
    search_dir = Path(search_dir)
    search = config["search"]
    if (
        int(search["new_candidates"]) != 8
        or int(search["parent_count"]) != 2
        or int(search["refinements_per_parent"]) != 3
        or int(search["explorations"]) != 2
    ):
        raise ValueError("HAD requires 8 candidates: 3 refinements per each of 2 parents plus 2 explorations")
    round_dir = search_dir / f"round_{round_index:02d}"
    generation = round_dir / "generation"
    generation.mkdir(parents=True, exist_ok=True)

    if round_index == 1:
        parents: list[dict[str, Any]] = []
    else:
        parents = _completed_parents(search_dir, config, before_round=round_index)[:2]
        if len(parents) != 2:
            raise ValueError("a later HAD round requires two measured full-budget parents")
    parent_ids = [row["candidate_id"] for row in parents]
    count = int(config["search"]["new_candidates"])
    expected_count = (
        count
        if round_index == 1
        else len(parent_ids) * int(config["search"]["refinements_per_parent"])
        + int(config["search"]["explorations"])
    )
    if count != expected_count:
        raise ValueError("configured HAD allocations do not sum to new_candidates")
    schema = _candidate_schema(count, parent_ids)
    prompt = _render_prompt(round_index, report, parents, config)
    prompt_path = generation / "prompt.md"
    schema_path = generation / "schema.json"
    if prompt_path.exists() and prompt_path.read_text() != prompt:
        raise ValueError(f"existing HAD prompt differs: {prompt_path}")
    prompt_path.write_text(prompt)
    write_json(schema_path, schema)

    accepted_path = generation / "response.json"
    # A rejected response is still an experimental outcome, not permission to retry.
    if accepted_path.is_file() or (generation / "raw_response.json").is_file():
        response = original_response(generation)
    else:
        response = _call_or_inject(llm_call, prompt, schema, generation, config)
        if response != original_response(generation):
            raise ValueError("HAD response differs from the original base-model output")
    rows = _validate_response(
        response,
        round_index=round_index,
        parent_ids=parent_ids,
        parent_codes={row["candidate_id"]: row["model_code"] for row in parents},
        config=config,
    )
    if not accepted_path.exists():
        write_json(accepted_path, response)
    return _materialize(round_dir, rows, round_index, parent_ids)
