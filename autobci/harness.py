"""
Run isolated model calls through Codex, Claude Code, OpenCode, and Gemini.

Copyright 2022 Centre for Brain Computing Research (CBCR), College of Computing and Data Science (CCDS), Nanyang Technological University (NTU);
licensed under the CBCR License 1.0 (see LICENSE).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit
import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time

import jsonschema
import yaml

from .common import PROJECT_ROOT, write_json


# Codex invocation and audit

_DISABLED_FEATURES = (
    "shell_tool",
    "unified_exec",
    "shell_snapshot",
    "apps",
    "plugins",
    "remote_plugin",
    "browser_use",
    "browser_use_external",
    "computer_use",
    "in_app_browser",
    "image_generation",
    "multi_agent",
    "memories",
    "skill_search",
    "code_mode",
    "code_mode_host",
)


def _audit_events(stdout: str) -> dict[str, Any]:
    """Reject a response if the supposedly isolated agent attempted a tool call."""
    usage = None
    turn_started = False
    for line in stdout.splitlines():
        event = json.loads(line)
        if event.get("type") == "turn.started":
            turn_started = True
        if event.get("type", "").startswith("item."):
            item = event.get("item", {})
            kind = item.get("type")
            startup_warning = kind == "error" and not turn_started and (
                item.get("message", "").startswith("`--dangerously-bypass-hook-trust` is enabled.")
                or item.get("message", "").startswith(
                    "Code Mode is unavailable because code-mode host is disabled."
                )
            )
            if kind not in ("agent_message", "reasoning") and not startup_warning:
                raise RuntimeError(f"unexpected Codex tool/event: {kind}")
        if event.get("type") == "turn.completed":
            usage = event.get("usage")
    if usage is None:
        raise RuntimeError("Codex did not record a completed turn")
    return usage


def _call_codex(
    prompt: str,
    schema: dict[str, Any],
    output_dir: Path,
    *,
    model: str,
    reasoning_effort: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    """Call Codex in an empty, read-only directory and preserve an audit trail."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    schema_path = output_dir / "schema.json"
    write_json(schema_path, schema)
    (output_dir / "prompt.md").write_text(prompt)

    started = time.monotonic()
    invocation: dict[str, Any] = {
        "status": "running",
        "provider": "openai",
        "harness": "codex",
        "model": model,
        "requested_model": model,
        "reported_models": [],
        "model_identity_sources": [],
        "model_identity_evidence": "request_only",
        "provider_endpoint": None,
        "reasoning_effort": reasoning_effort,
        "timeout_seconds": timeout_seconds,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output_dir / "invocation.json", invocation)

    try:
        with tempfile.TemporaryDirectory(prefix="autobci_comb_v2_") as temporary:
            work = Path(temporary)
            local_schema = work / "schema.json"
            local_schema.write_text(json.dumps(schema, indent=2) + "\n")
            instructions = work / "instructions.md"
            instructions.write_text(
                "Use only the supplied prompt. Do not use tools, inspect files, run "
                "commands, or retrieve external information. Return only JSON matching "
                "the supplied schema.\n"
            )
            response_path = work / "response.json"
            command = [
                "codex", "exec", "--model", model,
                "--sandbox", "read-only", "--ephemeral", "--ignore-user-config",
                "--ignore-rules", "--skip-git-repo-check", "--color", "never",
                "--json", "--output-schema", str(local_schema),
                "--output-last-message", str(response_path), "--cd", str(work),
                "-c", f'model_reasoning_effort={json.dumps(reasoning_effort)}',
                "-c", f'model_instructions_file={json.dumps(str(instructions))}',
                "-c", 'web_search="disabled"', "-c", "project_doc_max_bytes=0",
                "-c", "mcp_servers={}", "-c", "features.hooks=true",
                "-c", 'hooks.PreToolUse=[{matcher=".*",hooks=[{type="command",command="exit 2"}]}]',
                "--dangerously-bypass-hook-trust",
            ]
            for feature in _DISABLED_FEATURES:
                command.extend(("--disable", feature))
            command.append("-")
            invocation["command"] = command
            write_json(output_dir / "invocation.json", invocation)

            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=None,  # Inherit terminal stderr; do not create a log file.
                text=True,
                start_new_session=True,
            )
            try:
                stdout, _ = process.communicate(prompt, timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.communicate(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.communicate()
                if response_path.exists():
                    (output_dir / "raw_response.json").write_text(response_path.read_text())
                raise

            invocation["return_code"] = process.returncode
            if response_path.exists():
                (output_dir / "raw_response.json").write_text(response_path.read_text())
            if process.returncode:
                raise RuntimeError(f"Codex exited with {process.returncode}; see terminal output")
            usage = _audit_events(stdout)
            write_json(output_dir / "usage.json", usage)
            response = json.loads((output_dir / "raw_response.json").read_text())
            invocation["status"] = "complete"
    except Exception as exc:
        invocation.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        invocation["duration_seconds"] = time.monotonic() - started
        invocation["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(output_dir / "invocation.json", invocation)

    return response


# Provider selection and shared harness interface

SETTINGS_PATH = PROJECT_ROOT / "config/harnesses.yaml"
ALLOWED_HARNESSES = {
    "openai": ("codex",),
    "anthropic": ("claude-code",),
    "deepseek": ("claude-code", "opencode"),
    "qwen": ("claude-code", "opencode"),
    "google": ("gemini",),
}
INSTRUCTIONS = (
    "Use only the supplied prompt. Do not use tools, inspect files, run commands, "
    "or retrieve external information. Return only JSON matching the supplied "
    "schema, without Markdown fences.\n"
)


def select_harness(model: str, harness: str | None = None) -> tuple[str, str]:
    """Provider follows the model name; unsupported combinations fail explicitly."""
    if model.startswith(("gpt-", "o1", "o3", "o4")):
        provider = "openai"
    elif model.startswith("claude-"):
        provider = "anthropic"
    elif model.startswith("deepseek-"):
        provider = "deepseek"
    elif model.startswith(("qwen", "glm-")):
        # GLM is served through the same Qwen Token Plan endpoint and credential.
        provider = "qwen"
    elif model.startswith("gemini-"):
        provider = "google"
    else:
        raise ValueError(f"Unknown model provider: {model}")
    allowed = ALLOWED_HARNESSES[provider]
    harness = harness or allowed[0]
    if harness not in allowed:
        raise ValueError(f"{provider} models require one of {allowed}; got {harness}")
    return provider, harness


def _credential(settings: dict, settings_path: Path) -> str:
    key = os.environ.get(settings["api_key_env"], "").strip()
    if settings.get("api_key_file"):
        path = Path(settings["api_key_file"]).expanduser()
        if not path.is_absolute():
            path = settings_path.parent / path
        if path.is_file():
            key = path.read_text().strip()
    if key and len(key.split()) != 1:
        raise ValueError("The API key must be one opaque token")
    return key


def _sanitized_endpoint(url: str | None) -> str | None:
    """Record endpoint identity without credentials, queries, or fragments."""
    if not url:
        return None
    parsed = urlsplit(url)
    host = parsed.hostname or ""
    if parsed.port is not None:
        host += f":{parsed.port}"
    return urlunsplit((parsed.scheme, host, parsed.path, "", ""))


def _claude_command(work, model, effort, settings, key, max_tokens, system):
    env = os.environ.copy()
    # Avoid inheriting another provider or a different model's configuration.
    for name in list(env):
        if name.startswith(("ANTHROPIC_", "CLAUDE_CODE_")):
            env.pop(name)
    env.pop("MAX_THINKING_TOKENS", None)
    env.update(
        ANTHROPIC_BASE_URL=settings["claude_code_url"],
        ANTHROPIC_MODEL=model,
        ANTHROPIC_DEFAULT_OPUS_MODEL=model,
        ANTHROPIC_DEFAULT_SONNET_MODEL=model,
        ANTHROPIC_DEFAULT_HAIKU_MODEL=model,
        CLAUDE_CODE_SUBAGENT_MODEL=model,
        CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1",
        CLAUDE_CODE_MAX_OUTPUT_TOKENS=str(max_tokens),
    )
    command = ["claude", "--print", "--model", model, "--effort", effort,
               "--system-prompt", system, "--tools", "", "--strict-mcp-config",
               "--setting-sources", "", "--disable-slash-commands",
               "--no-session-persistence", "--output-format", "stream-json", "--verbose"]
    if key:
        env["ANTHROPIC_API_KEY"] = key
        env["CLAUDE_CONFIG_DIR"] = str(work / "claude-config")
        command.append("--bare")
    else:
        # Native Anthropic models may use an existing Claude Code login.
        if os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
            env["CLAUDE_CODE_OAUTH_TOKEN"] = os.environ["CLAUDE_CODE_OAUTH_TOKEN"]
        command.append("--safe-mode")
    return command, env


def _opencode_command(work, model, effort, settings, key, max_tokens, system):
    env = os.environ.copy()
    for name in list(env):
        if name.startswith("OPENCODE_"):
            env.pop(name)
    for kind in ("CONFIG", "DATA", "CACHE", "STATE"):
        env[f"XDG_{kind}_HOME"] = str(work / kind.lower())
    env["AUTOBCI_API_KEY"] = key
    config = {
        "model": f"autobci/{model}",
        "providers": {"autobci": {
            "env": ["AUTOBCI_API_KEY"],
            "package": "@opencode/ai/providers/openai-compatible",
            "settings": {"baseURL": settings["opencode_url"]},
            "models": {model: {
                "capabilities": {"tools": False, "input": ["text"], "output": ["text"]},
                "body": {"max_tokens": max_tokens, "reasoning_effort": effort},
            }},
        }},
        "agents": {"autobci": {
            "description": "One isolated JSON response for HAD or PEEK",
            "mode": "primary", "system": system,
            "permissions": [{"action": "*", "resource": "*", "effect": "deny"}],
        }},
        "snapshots": False,
        "compaction": {"auto": False},
        "experimental": {"policies": [
            {"action": "provider.use", "resource": "*", "effect": "deny"},
            {"action": "provider.use", "resource": "autobci", "effect": "allow"},
        ]},
    }
    # A private server reads this private global config. No credentials on disk.
    write_json(work / "config/opencode/opencode.json", config)
    command = ["opencode", "run", "--standalone", "--model", f"autobci/{model}",
               "--agent", "autobci", "--format", "json"]
    return command, env, config


def _gemini_command(work, model, effort, settings, key, max_tokens, system):
    """Use the existing Gemini login, with a private prompt and tools disabled."""
    env = os.environ.copy()
    (work / "system.md").write_text(system)
    env["GEMINI_SYSTEM_MD"] = str(work / "system.md")
    config = {
        "tools": {"core": []},
        "mcp": {"allowed": []},
        "hooksConfig": {"enabled": False},
        "skills": {"enabled": False},
        "context": {"fileName": "AUTOBCI_UNUSED_CONTEXT.md"},
        "model": {"maxSessionTurns": 1},
        "modelConfigs": {"customOverrides": [{
            "match": {"model": model},
            "modelConfig": {"generateContentConfig": {
                "maxOutputTokens": max_tokens,
                "thinkingConfig": {"thinkingLevel": effort.upper()},
            }},
        }]},
    }
    write_json(work / ".gemini/settings.json", config)
    # Deny any tools added by the CLI outside its core tool registry.
    (work / "policy.toml").write_text(
        '[[rule]]\ntoolName = "*"\ndecision = "deny"\npriority = 999\n'
    )
    command = ["gemini", "--model", model, "--prompt", "Follow the supplied prompt.",
               "--output-format", "stream-json", "--extensions", "none",
               "--skip-trust", "--policy", str(work / "policy.toml")]
    return command, env, config


def _run(command, prompt, work, env, timeout, key):
    """Capture response events in memory, print stderr, and redact credentials."""
    process = subprocess.Popen(command, cwd=work, env=env, stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, start_new_session=True)
    timed_out = False
    try:
        stdout, stderr = process.communicate(prompt, timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(process.pid, signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
    if key:
        stdout, stderr = stdout.replace(key, "[REDACTED]"), stderr.replace(key, "[REDACTED]")
    if stderr:
        print(stderr, end="", file=sys.stderr, flush=True)
    return process.returncode, timed_out, stdout


def _response(events, harness):
    """Extract the original text and reject tool activity or multiple model turns."""
    messages = set()
    text = []
    usage = {"reported": False}
    error = None
    completed = False
    reported_models = set()
    model_sources = set()
    for event in events:
        kind = event.get("type")
        if harness == "claude-code":
            if kind == "system" and event.get("subtype") == "init":
                if event.get("model"):
                    reported_models.add(event["model"])
                    model_sources.add("system.init")
            elif kind == "assistant":
                message = event["message"]
                messages.add(message["id"])
                if message.get("model"):
                    reported_models.add(message["model"])
                    model_sources.add("assistant")
                for block in message["content"]:
                    if block["type"] == "text":
                        text.append(block["text"])
                    elif block["type"] not in ("thinking", "redacted_thinking"):
                        error = f"unexpected Claude Code content: {block['type']}"
            elif kind == "result":
                completed = not event.get("is_error", True)
                usage = event.get("usage", usage)
                for reported, details in event.get("modelUsage", {}).items():
                    reported_models.add(reported)
                    if details.get("canonicalModel"):
                        reported_models.add(details["canonicalModel"])
                    model_sources.add("result.modelUsage")
                if event.get("permission_denials"):
                    error = "Claude Code attempted a denied tool call"
            elif kind not in ("system", "rate_limit_event"):
                error = f"unexpected Claude Code event: {kind}"
        elif harness == "gemini":
            if kind == "init" and event.get("model"):
                reported_models.add(event["model"])
                model_sources.add("init")
            elif kind == "message" and event.get("role") == "assistant":
                messages.add("one")  # The CLI is limited to one session turn.
                text.append(event["content"])
            elif kind == "result":
                completed = event.get("status") == "success"
                usage = event.get("stats", usage)
                reported_models.update(usage.get("models", {}))
                if usage.get("models"):
                    model_sources.add("result.stats.models")
                if usage.get("tool_calls", 0):
                    error = "Gemini attempted a tool call"
                elif not completed and not error:
                    error = event.get("error", {}).get("message")
            elif kind not in ("init", "message"):
                error = f"unexpected Gemini event: {kind}"
        else:
            if kind == "step_start":
                messages.add(event["part"]["messageID"])
            elif kind == "text":
                text.append(event["part"]["text"])
                completed = True  # OpenCode v2 emits final text; exit code is checked below.
            elif kind == "step_finish":
                usage = event["part"].get("tokens", usage)
            elif kind != "reasoning":
                error = f"unexpected OpenCode event: {kind}"
    if len(messages) != 1 or not completed:
        error = error or "Harness did not complete exactly one model turn"
    identity = {"reported_models": sorted(reported_models),
                "model_identity_sources": sorted(model_sources)}
    return "".join(text), usage, error, identity


def call_llm(prompt, schema, output_dir, *, model, reasoning_effort, timeout_seconds,
             harness=None, settings_path=SETTINGS_PATH):
    """Return structured output from the selected harness, with no fallback or repair."""
    provider, harness = select_harness(model, harness)
    output_dir = Path(output_dir)
    if any((output_dir / name).exists() for name in ("invocation.json", "raw_response.json", "events.jsonl")):
        raise FileExistsError("An LLM attempt already exists here; automatic regeneration is forbidden")
    if harness == "codex":
        return _call_codex(prompt, schema, output_dir, model=model,
                              reasoning_effort=reasoning_effort, timeout_seconds=timeout_seconds)

    settings_path = Path(settings_path).resolve()
    settings = yaml.safe_load(settings_path.read_text())
    provider_settings = settings["providers"][provider]
    key = _credential(provider_settings, settings_path)
    if provider not in ("anthropic", "google") and not key:
        raise ValueError(f"Set {provider_settings['api_key_env']} or its configured key file")
    system = INSTRUCTIONS + "\nJSON SCHEMA\n" + json.dumps(schema)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "prompt.md").write_text(prompt)
    write_json(output_dir / "schema.json", schema)
    started = time.monotonic()
    invocation = {"status": "running", "provider": provider, "harness": harness,
                  "model": model, "requested_model": model, "reported_models": [],
                  "model_identity_sources": [], "model_identity_evidence": "pending",
                  "provider_endpoint": _sanitized_endpoint(provider_settings.get(
                      "claude_code_url" if harness == "claude-code" else
                      "opencode_url" if harness == "opencode" else None)),
                  "reasoning_effort": reasoning_effort,
                  "timeout_seconds": timeout_seconds,
                  "started_at_utc": datetime.now(timezone.utc).isoformat()}
    try:
        with tempfile.TemporaryDirectory(prefix="autobci_harness_") as temporary:
            work = Path(temporary)
            args = (work, model, reasoning_effort, provider_settings, key,
                    settings["max_output_tokens"], system)
            if harness == "claude-code":
                command, env = _claude_command(*args)
            elif harness == "gemini":
                command, env, config = _gemini_command(*args)
                invocation["harness_config"] = config
            else:
                command, env, config = _opencode_command(*args)
                invocation["harness_config"] = config
            invocation["command"] = command
            write_json(output_dir / "invocation.json", invocation)
            code, timed_out, stdout = _run(command, prompt, work, env, timeout_seconds, key)
            invocation["return_code"] = code
            events = [json.loads(line) for line in stdout.splitlines() if line]
            raw, usage, audit_error, identity = _response(events, harness)
            # Preserve even rejected or incomplete output; never rewrite it into valid JSON.
            (output_dir / "raw_response.json").write_text(raw)
            write_json(output_dir / "usage.json", usage)
            invocation.update(identity)
            invocation["model_identity_evidence"] = (
                "provider_events" if identity["reported_models"] else "request_only"
            )
            reported = set(identity["reported_models"])
            if harness == "claude-code":
                required = {"system.init", "assistant", "result.modelUsage"}
                sources = set(identity["model_identity_sources"])
                if sources != required:
                    audit_error = audit_error or (
                        "Claude Code did not report model identity in system, assistant, and result events"
                    )
                elif reported != {model}:
                    audit_error = audit_error or (
                        f"Claude Code reported models {sorted(reported)!r}, expected only {model!r}"
                    )
            elif harness == "gemini" and reported != {model}:
                audit_error = audit_error or "Gemini did not use only the requested model"
            if timed_out:
                raise TimeoutError(f"{harness} exceeded {timeout_seconds} seconds")
            if code or audit_error:
                raise RuntimeError(audit_error or f"{harness} exited with {code}; see terminal output")
            response = json.loads(raw)
            jsonschema.validate(response, schema)
            invocation["status"] = "complete"
    except Exception as exc:
        invocation.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        invocation["duration_seconds"] = time.monotonic() - started
        invocation["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(output_dir / "invocation.json", invocation)
    return response


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--harness", choices=("codex", "claude-code", "opencode", "gemini"))
    parser.add_argument("--prompt", default="who are you")
    parser.add_argument("--effort", default="medium")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    schema = {"type": "object", "properties": {"answer": {"type": "string"}},
              "required": ["answer"], "additionalProperties": False}
    response = call_llm(args.prompt, schema, args.output, model=args.model,
                        harness=args.harness, reasoning_effort=args.effort,
                        timeout_seconds=args.timeout)
    print(response["answer"])


if __name__ == "__main__":
    main()
