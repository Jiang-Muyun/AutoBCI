"""
Manage shared configuration, artifact paths, JSON storage, and generated-source provenance.

Copyright 2022 Centre for Brain Computing Research (CBCR), College of Computing and Data Science (CCDS), Nanyang Technological University (NTU);
licensed under the CBCR License 1.0 (see LICENSE).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import os

import yaml


# Configuration and durable JSON

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def peek_settings(config: dict) -> dict:
    """Read PEEK settings, accepting the original field names for saved configs."""
    search = config.get("search", {})
    settings = {
        "enabled": search.get("forecast_enabled", True),
        "early_epochs": config.get("training", {}).get("early_epochs", 10),
        "promote_count": search.get("promote_count", 2),
    }
    settings.update(config.get("peek", {}))
    return settings


def validate_peek(config: dict) -> dict:
    settings = peek_settings(config)
    if not isinstance(settings["enabled"], bool):
        raise ValueError("peek.enabled must be true or false")
    if settings["enabled"]:
        early, count = settings["early_epochs"], settings["promote_count"]
        if type(early) is not int or not 0 < early < config["training"]["epochs"]:
            raise ValueError("PEEK requires 0 < early_epochs < epochs")
        if type(count) is not int or not config["search"]["parent_count"] <= count <= config["search"]["new_candidates"]:
            raise ValueError("peek.promote_count must be between parent_count and new_candidates")
    return settings


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def write_json(path: str | Path, value: Any) -> None:
    """Atomically replace a JSON file so interrupted runs never leave half JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def resolve_config_path(value: str | Path, config_path: str | Path) -> Path:
    """Resolve config/... from the project root, other relative paths beside the YAML."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        base = PROJECT_ROOT if path.parts and path.parts[0] == "config" else Path(config_path).parent
        path = base / path
    return path.resolve()


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML and resolve its filesystem paths independently of the cwd."""
    config_path = Path(path).expanduser().resolve()
    if config_path.name.startswith(("exclude_", "_exclude_")):
        raise ValueError(f"Experiment is excluded from runs: {config_path.name}")
    config = yaml.safe_load(config_path.read_text())
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {config_path}")

    data = config.setdefault("data", {})
    catalogs = data.get("tasks", {})
    if catalogs:
        if not set(catalogs) <= {"emotion", "mi", "sleep"}:
            raise ValueError("data.tasks supports emotion, mi, and sleep catalogs")
    validate_peek(config)
    for name, value in list(catalogs.items()) if catalogs else [("catalog", data["catalog"])]:
        (catalogs if catalogs else data)[name] = str(resolve_config_path(value, config_path))

    output = config.setdefault("output", {})
    output_root = Path(output.get("root", "runs")).expanduser()
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    output["root"] = str(output_root.resolve())

    config["_config_path"] = str(config_path)
    config["_project_root"] = str(PROJECT_ROOT)
    return config


# Candidate and training paths

def training_dir(candidate: Path, config: dict | None = None) -> Path:
    """candidates/r01_x00_name.py -> training/01_x00_name/."""
    root = candidate.parent.parent / "training"
    if config and config.get("_task"):
        root = root / config["_task"]
    return root / candidate.stem.removeprefix("r")


def model_file(candidate: Path) -> Path:
    """Search candidates are .py files; standalone benchmarks/champions use model.py."""
    return candidate if candidate.suffix == ".py" else candidate / "model.py"


def task_label(candidate: Path, config: dict) -> str:
    return f"{candidate.stem}/{config['_task']}" if config.get("_task") else candidate.stem


# Generated-source provenance

class GeneratedCodeChanged(ValueError):
    """An edited proposal is not a valid result of the base-model experiment."""


def check_generation_identity(round_dir: Path, config: dict) -> None:
    """Keep candidates attributed to the model invocation that generated them."""
    invocation_path = Path(round_dir) / "generation" / "invocation.json"
    if not invocation_path.is_file():
        raise ValueError(f"Round is missing its generation invocation: {round_dir}")
    invocation = read_json(invocation_path)
    if invocation.get("status") == "injected":
        return
    search = config["search"]
    expected = {
        "model": str(search["model"]),
        "reasoning_effort": str(search["reasoning_effort"]),
    }
    if search.get("harness") is not None:
        expected["harness"] = search["harness"]
    mismatches = {
        name: (invocation.get(name), value)
        for name, value in expected.items()
        if invocation.get(name) != value
    }
    if mismatches:
        raise ValueError(f"Generation invocation does not match the active experiment: {mismatches}")


def original_response(generation: Path) -> dict:
    invocation = generation / "invocation.json"
    if invocation.exists() and read_json(invocation)["status"] not in ("complete", "injected"):
        raise GeneratedCodeChanged("The original harness invocation did not complete successfully")
    raw = read_json(generation / "raw_response.json")
    accepted = generation / "response.json"
    if accepted.exists() and read_json(accepted) != raw:
        raise GeneratedCodeChanged("Accepted response does not match the original base-model response; repairs are forbidden")
    return raw


def check_candidate_source(candidate: Path) -> None:
    candidate = Path(candidate)
    response = original_response(candidate.parent.parent / "generation")
    metadata = read_json(candidate.with_suffix(".json"))
    row = next(row for row in response["candidates"] if row["name"] == metadata["name"])
    expected = row["model_code"].rstrip() + "\n"
    if model_file(candidate).read_text() != expected:
        raise GeneratedCodeChanged(f"{candidate.stem} does not match its original base-model code; repairs are forbidden")
