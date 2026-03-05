from __future__ import annotations

import argparse
import fnmatch
import importlib.resources as resources
import json
import os
import re
import shlex
import uuid
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

from .config import (
    BACKEND_SPECIFIC_KEYS,
    Config,
    DEFAULT_BACKEND_NAME,
    DEFAULT_CONFIG_PATH,
    DEFAULT_CUDA_MINOR_VERSION,
    load_config,
)
from .slurm import (
    cancel_job,
    get_job_io_paths,
    list_jobs,
    run_health_checks,
    submit_job,
)
from .ssh import (
    SSHError,
    copy_to_remote,
    run_ssh,
)
from .manifest import update_manifest_metadata, write_run_manifest
from .runs import list_runs, record_submission, show_run, sync_statuses

DEFAULT_SNAPSHOT_EXCLUDES: list[str] = [
    ".git/",
    ".gitignore",
    ".venv/",
    ".venv-vllm/",
    "__pycache__/",
    "*.pyc",
    "*.pyo",
    "*.pyd",
    "*.log",
    "*.tmp",
    ".DS_Store",
    ".mypy_cache/",
    ".pytest_cache/",
    ".ruff_cache/",
    ".coverage",
    ".idea/",
    ".vscode/",
    ".claude/",
    "node_modules/",
]

SNAPSHOT_IGNORE_PATTERNS = [pattern.rstrip("/") for pattern in DEFAULT_SNAPSHOT_EXCLUDES]
SNAPSHOT_IGNORE_PATTERNS.extend([
    "run_metadata",
    "runs",
    "results",
])

DEFAULT_ENV_WATCH = [
    "scripts/setup_env.sh",
    "requirements.txt",
    "requirements.lock",
    "requirements-dev.txt",
    "pyproject.toml",
    "poetry.lock",
    "uv.lock",
    "environment.yml",
]


def _has_constraint_flag(args: list[str]) -> bool:
    for arg in args:
        if arg in {"--constraint", "-C"}:
            return True
        if arg.startswith("--constraint="):
            return True
        if arg.startswith("-C") and arg != "-C":
            return True
    return False


def _has_export_flag(args: list[str]) -> bool:
    for arg in args:
        if arg == "--export":
            return True
        if arg.startswith("--export="):
            return True
    return False


def _has_gres_flag(args: list[str]) -> bool:
    for arg in args:
        if arg == "--gres":
            return True
        if arg.startswith("--gres="):
            return True
    return False


def _sbatch_args_from_script(path: Path) -> list[str]:
    """Extract sbatch args declared via '#SBATCH' lines in the script."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return []

    collected: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("#SBATCH"):
            continue
        remainder = stripped[len("#SBATCH") :].strip()
        if not remainder:
            continue
        try:
            collected.extend(shlex.split(remainder))
        except ValueError:
            continue
    return collected


def _collect_export_envs(
    cli_env_flags: list[str],
    config_env_pass: list[str],
) -> tuple[list[str], list[str]]:
    """
    Build a list of KEY=VALUE entries to export into the submitted job.

    Returns a tuple of (assignments, missing_from_config) where the latter tracks config-provided
    variables that were skipped because they are not set locally.
    """
    exports: dict[str, str] = {}
    missing_from_config: list[str] = []

    def _handle_entry(entry: str, *, allow_missing: bool) -> None:
        raw = entry.strip()
        if not raw:
            return
        if "=" in raw:
            key, value = raw.split("=", 1)
            key = key.strip()
            if not key:
                raise ValueError(f"Invalid --env entry (missing key): {entry}")
            exports[key] = value
            return
        key = raw
        if key in os.environ:
            exports[key] = os.environ[key]
        elif allow_missing:
            missing_from_config.append(key)
        else:
            raise ValueError(
                f"--env {key} was requested but it is not set in your local environment."
            )

    for item in config_env_pass:
        _handle_entry(item, allow_missing=True)

    for item in cli_env_flags:
        _handle_entry(item, allow_missing=False)

    assignments = [f"{key}={value}" for key, value in exports.items()]
    return assignments, missing_from_config


def _load_template(name: str) -> str:
    """Load a text template bundled with the CLI."""
    try:
        return resources.files("koa_cli.templates").joinpath(name).read_text(encoding="utf-8")
    except AttributeError:  # pragma: no cover - fallback for Python <3.9
        return resources.read_text("koa_cli.templates", name)


def _prompt(value: Optional[str], question: str, *, default: Optional[str] = None, required: bool = False) -> str:
    if value is not None and str(value).strip():
        return str(value).strip()

    while True:
        suffix = f" [{default}]" if default else ""
        answer = input(f"{question}{suffix}: ").strip()
        if answer:
            return answer
        if default is not None and default != "":
            return default
        if not required:
            return ""
        print("Value required.", file=sys.stderr)


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to the KOA config file (defaults to koa-config.yaml in the repository or ~/.config/koa/config.yaml).",
    )
    parser.add_argument(
        "--backend",
        default=None,
        help="Slurm backend name to use (defaults to 'koa' unless overridden in the config).",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="koa",
        description="Utilities for running KOA HPC (Slurm) jobs from your local machine.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    setup_parser = subparsers.add_parser(
        "setup", help="Configure global KOA defaults (user, workspace roots, CUDA version)."
    )
    setup_parser.add_argument("--user", help="KOA username")
    setup_parser.add_argument("--host", help="KOA login host (default: koa.its.hawaii.edu)")
    setup_parser.add_argument(
        "--remote-root",
        help="Top-level remote workspace directory for KOA projects.",
    )
    setup_parser.add_argument(
        "--local-root",
        help="Top-level local workspace directory for KOA project mirrors.",
    )
    setup_parser.add_argument(
        "--default-account",
        help="Default Slurm account for submissions (optional).",
    )
    setup_parser.add_argument(
        "--default-partition",
        help="Default Slurm partition for submissions (e.g. kill-shared).",
    )
    setup_parser.add_argument(
        "--default-constraint",
        default=None,
        help='Default Slurm constraint to apply (e.g. hopper). Leave empty to disable.',
    )
    setup_parser.add_argument(
        "--default-gres",
        default=None,
        help='Default Slurm GRES string to apply (e.g. "gpu:a100:1"). Leave empty to disable.',
    )
    setup_parser.add_argument(
        "--backend",
        default=None,
        help="Name of the backend entry to configure (default: koa).",
    )
    setup_parser.add_argument(
        "--cuda-version",
        default=None,
        help="Default CUDA minor version to install for this backend (e.g. 12.4 or 12.8).",
    )
    setup_parser.add_argument(
        "--dashboard-base-url",
        default=None,
        help="Base URL for the cluster web file browser (e.g. https://koa.its.hawaii.edu/pun/sys/dashboard/files/fs).",
    )

    init_parser = subparsers.add_parser(
        "init",
        help="Initialise KOA project configuration in the current repository.",
    )
    init_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing koa-config.yaml or scripts if present.",
    )
    init_parser.add_argument(
        "--backend",
        default=None,
        help="Backend name to use for default templates (default: configured default backend).",
    )
    init_parser.add_argument(
        "--cuda-version",
        default=None,
        help="Override the CUDA minor version for this project (default: backend setting).",
    )

    check_parser = subparsers.add_parser(
        "check", help="Run KOA connectivity health checks."
    )
    _add_common_arguments(check_parser)

    jobs_parser = subparsers.add_parser(
        "jobs", help="List active KOA jobs for the configured user."
    )
    _add_common_arguments(jobs_parser)

    dashboard_parser = subparsers.add_parser(
        "dashboard", help="Launch the KOA Streamlit dashboard."
    )
    _add_common_arguments(dashboard_parser)
    dashboard_parser.add_argument(
        "--cache-ttl",
        type=int,
        default=30,
        help="Cache duration (seconds) for dashboard data queries (default: 30).",
    )

    cancel_parser = subparsers.add_parser("cancel", help="Cancel a KOA job by id.")
    _add_common_arguments(cancel_parser)
    cancel_parser.add_argument("job_id", help="Slurm job id to cancel.")

    submit_parser = subparsers.add_parser(
        "submit", help="Submit a job script via sbatch."
    )
    _add_common_arguments(submit_parser)
    submit_parser.add_argument(
        "job_script", type=Path, help="Path to the local job script."
    )
    submit_parser.add_argument("--remote-name", help="Override the filename on KOA.")
    submit_parser.add_argument(
        "--partition",
        help="Slurm partition (queue) to submit to. Defaults to kill-shared.",
    )
    submit_parser.add_argument(
        "--constraint",
        help="Slurm constraint to apply (e.g. hopper).",
    )
    submit_parser.add_argument("--time", help="Walltime request (e.g. 02:00:00).")
    submit_parser.add_argument("--gpus", type=int, help="Number of GPUs to request.")
    submit_parser.add_argument("--cpus", type=int, help="Number of CPUs to request.")
    submit_parser.add_argument("--memory", help="Memory request (e.g. 32G).")
    submit_parser.add_argument("--account", help="Slurm account if required.")
    submit_parser.add_argument("--qos", help="Quality of service if required.")
    submit_parser.add_argument(
        "--desc",
        help="Optional description appended to the timestamped run directory name.",
    )
    submit_parser.add_argument(
        "--env",
        action="append",
        default=[],
        help="Forward an environment variable into the job (NAME or NAME=value). Repeatable.",
    )
    submit_parser.add_argument(
        "--sbatch-arg",
        action="append",
        default=[],
        help="Additional raw sbatch arguments. Repeat for multiple flags.",
    )

    logs_parser = subparsers.add_parser(
        "logs", help="Stream or inspect a job's stdout/stderr log."
    )
    _add_common_arguments(logs_parser)
    logs_parser.add_argument("job_id", help="Job ID to inspect.")
    logs_parser.add_argument(
        "--stream",
        choices=["stdout", "stderr"],
        default="stdout",
        help="Select which stream to view (default: stdout).",
    )
    logs_parser.add_argument(
        "--lines",
        type=int,
        default=50,
        help="Number of lines to show when not following (default: 50).",
    )
    logs_parser.add_argument(
        "--follow",
        action="store_true",
        help="Follow log output in real time (tail -F).",
    )

    runs_parser = subparsers.add_parser(
        "runs", help="Manage and inspect recorded KOA job runs."
    )
    _add_common_arguments(runs_parser)
    runs_subparsers = runs_parser.add_subparsers(dest="runs_command", required=True)
    runs_list = runs_subparsers.add_parser("list", help="List recorded runs.")
    runs_list.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Number of runs to display (default: 20).",
    )
    runs_subparsers.add_parser(
        "sync", help="Sync job statuses from KOA (updates local catalog)."
    )
    runs_show = runs_subparsers.add_parser("show", help="Display details for a run.")
    runs_show.add_argument("job_id", help="Job ID to inspect.")

    return parser


def _load(args: argparse.Namespace) -> Config:
    backend = getattr(args, "backend", None)
    return load_config(args.config, backend_name=backend)


def _setup(args: argparse.Namespace) -> int:
    existing: dict = {}
    if DEFAULT_CONFIG_PATH.exists():
        try:
            existing = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")) or {}
        except Exception as exc:  # pragma: no cover - defensive
            print(f"Warning: failed to read existing config ({exc}); starting fresh.", file=sys.stderr)
            existing = {}

    backend_name = args.backend or existing.get("default_backend") or DEFAULT_BACKEND_NAME

    backends_existing = existing.get("backends")
    backend_defaults: dict = {}
    if isinstance(backends_existing, list):
        for entry in backends_existing:
            if isinstance(entry, dict) and entry.get("cluster_name") == backend_name:
                backend_defaults = dict(entry)
                break
    if not backend_defaults:
        backend_defaults = {
            key: existing.get(key)
            for key in BACKEND_SPECIFIC_KEYS
            if existing.get(key) is not None
        }
        if backend_defaults and "cluster_name" not in backend_defaults:
            backend_defaults["cluster_name"] = backend_name

    default_user = (
        args.user
        or backend_defaults.get("user")
        or existing.get("user")
        or os.getenv("KOA_USER")
        or os.getenv("USER")
        or ""
    )
    default_host = (
        args.host
        or backend_defaults.get("host")
        or existing.get("host")
        or os.getenv("KOA_HOST")
        or "koa.its.hawaii.edu"
    )

    user = _prompt(args.user, "KOA username", default=default_user, required=True)
    host = _prompt(args.host, "KOA login host", default=default_host, required=True)

    suggested_remote_root = (
        args.remote_root
        or backend_defaults.get("remote_root")
        or existing.get("remote_root")
        or (existing.get("remote", {}) or {}).get("root")
        or f"/mnt/lustre/koa/scratch/{user}/koa-cli"
    )
    remote_root = _prompt(
        args.remote_root,
        "Remote workspace root",
        default=suggested_remote_root,
        required=True,
    )

    suggested_local_root = (
        args.local_root
        or backend_defaults.get("local_root")
        or existing.get("local_root")
        or (existing.get("local", {}) or {}).get("root")
        or str(Path("~/koa-projects").expanduser())
    )
    local_root = _prompt(
        args.local_root,
        "Local workspace root",
        default=suggested_local_root,
        required=True,
    )

    suggested_default_account = (
        args.default_account
        or backend_defaults.get("default_account")
        or existing.get("default_account")
        or ""
    )
    default_account = _prompt(
        args.default_account,
        "Default Slurm account (optional)",
        default=suggested_default_account,
        required=False,
    )

    default_partition_value = (
        args.default_partition
        or backend_defaults.get("default_partition")
        or existing.get("default_partition")
        or "kill-shared"
    )
    default_partition = _prompt(
        args.default_partition,
        "Default Slurm partition",
        default=default_partition_value,
        required=False,
    )

    default_constraint_value = (
        args.default_constraint
        or backend_defaults.get("default_constraint")
        or existing.get("default_constraint")
        or ""
    )
    default_constraint = _prompt(
        args.default_constraint,
        "Default Slurm constraint (optional, e.g. hopper)",
        default=default_constraint_value,
        required=False,
    )

    suggested_default_gres = (
        args.default_gres
        or backend_defaults.get("default_gres")
        or existing.get("default_gres")
        or ""
    )
    default_gres = _prompt(
        args.default_gres,
        'Default Slurm GRES (optional, e.g. gpu:a100:1)',
        default=suggested_default_gres,
        required=False,
    )

    suggested_dashboard_base = (
        args.dashboard_base_url
        or backend_defaults.get("dashboard_base_url")
        or existing.get("dashboard_base_url")
    )
    dashboard_base_url = _prompt(
        args.dashboard_base_url,
        "Web dashboard base URL (optional, e.g. https://koa.its.hawaii.edu/pun/sys/dashboard/files/fs)",
        default=suggested_dashboard_base or "",
        required=False,
    ).strip()

    default_cuda_minor_value = (
        args.cuda_version
        or backend_defaults.get("cuda_minor_version")
        or existing.get("cuda_minor_version")
        or DEFAULT_CUDA_MINOR_VERSION
    )
    cuda_minor_version = _prompt(
        args.cuda_version,
        "Default CUDA minor version (e.g. 12.4 or 12.8)",
        default=str(default_cuda_minor_value),
        required=True,
    )
    cuda_minor_version = cuda_minor_version.strip() or DEFAULT_CUDA_MINOR_VERSION

    config_data = dict(existing)
    normalized_backends: list[dict] = []
    if isinstance(backends_existing, list):
        for entry in backends_existing:
            if isinstance(entry, dict):
                normalized_backends.append(dict(entry))
    if not normalized_backends:
        legacy_backend = {
            key: existing.get(key)
            for key in BACKEND_SPECIFIC_KEYS
            if existing.get(key) is not None
        }
        if legacy_backend:
            legacy_backend.setdefault("cluster_name", backend_name)
            normalized_backends.append(dict(legacy_backend))

    config_data["backends"] = normalized_backends
    for key in BACKEND_SPECIFIC_KEYS:
        config_data.pop(key, None)
    for legacy_key in ("python_module", "cuda_module", "modules"):
        config_data.pop(legacy_key, None)

    backend_entry = None
    for entry in normalized_backends:
        if entry.get("cluster_name") == backend_name:
            backend_entry = entry
            break
    if backend_entry is None:
        backend_entry = {"cluster_name": backend_name}
        normalized_backends.append(backend_entry)

    backend_entry["user"] = user
    backend_entry["host"] = host
    backend_entry["remote_root"] = remote_root
    backend_entry["local_root"] = local_root
    backend_entry["cuda_minor_version"] = cuda_minor_version
    if default_account:
        backend_entry["default_account"] = default_account
    else:
        backend_entry.pop("default_account", None)
    if default_partition:
        backend_entry["default_partition"] = default_partition
    else:
        backend_entry.pop("default_partition", None)
    if default_constraint:
        backend_entry["default_constraint"] = default_constraint
    else:
        backend_entry.pop("default_constraint", None)
    if default_gres:
        backend_entry["default_gres"] = default_gres
    else:
        backend_entry.pop("default_gres", None)
    if dashboard_base_url:
        backend_entry["dashboard_base_url"] = dashboard_base_url
    else:
        backend_entry.pop("dashboard_base_url", None)


    if not config_data.get("default_backend") or args.backend is None:
        config_data["default_backend"] = backend_name

    # Ensure the config directory exists
    DEFAULT_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Ensure the local root exists
    try:
        Path(local_root).expanduser().mkdir(parents=True, exist_ok=True)
    except Exception as exc:  # pragma: no cover - filesystem issues
        print(f"Warning: unable to create local workspace root ({exc})", file=sys.stderr)

    DEFAULT_CONFIG_PATH.write_text(
        yaml.safe_dump(config_data, sort_keys=False),
        encoding="utf-8",
    )

    print("Updated KOA global configuration:")
    print(f"  File: {DEFAULT_CONFIG_PATH}")
    print(f"  Backend: {backend_name}")
    print(f"  User: {user}@{host}")
    print(f"  Remote workspace: {remote_root}")
    print(f"  Local workspace: {local_root}")
    if default_account:
        print(f"  Default account: {default_account}")
    if default_partition:
        print(f"  Default partition: {default_partition}")
    if default_constraint:
        print(f"  Default constraint: {default_constraint}")
    if default_gres:
        print(f"  Default GRES: {default_gres}")
    if dashboard_base_url:
        print(f"  Web dashboard: {dashboard_base_url}")
    print(f"  CUDA minor version: {cuda_minor_version}")

    return 0


def _load_global_config_data() -> dict:
    if not DEFAULT_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Global config not found at {DEFAULT_CONFIG_PATH}. Run `koa setup` first."
        )
    return yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")) or {}


def _render_setup_env_script(cuda_minor_version: Optional[str]) -> str:
    template = _load_template("setup_env.sh.tmpl")
    version = cuda_minor_version or DEFAULT_CUDA_MINOR_VERSION
    return template.replace("__CUDA_MINOR_VERSION__", version)


def _render_basic_job_template(
    project_name: str,
    default_partition: str,
    default_constraint: Optional[str],
) -> str:
    template = _load_template("basic_job.slurm.tmpl")
    constraint_line = (
        f'#SBATCH --constraint="{default_constraint}"' if default_constraint else ""
    )
    return (
        template.replace("__JOB_NAME__", project_name)
        .replace("__DEFAULT_PARTITION__", default_partition)
        .replace("__DEFAULT_CONSTRAINT__", constraint_line)
    )


def _write_file(path: Path, content: str, *, overwrite: bool = False, executable: bool = False) -> None:
    if path.exists() and not overwrite:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    if executable:
        path.chmod(path.stat().st_mode | 0o111)


def _init_project(args: argparse.Namespace) -> int:
    data = _load_global_config_data()
    backend_name = args.backend or data.get("default_backend") or DEFAULT_BACKEND_NAME
    cwd = Path.cwd()
    project_name = cwd.name

    backend_entry: dict = {}
    backends = data.get("backends")
    if isinstance(backends, list):
        for entry in backends:
            if isinstance(entry, dict) and entry.get("cluster_name") == backend_name:
                backend_entry = entry
                break
    if not backend_entry:
        backend_entry = {
            key: data.get(key)
            for key in BACKEND_SPECIFIC_KEYS
            if data.get(key) is not None
        }

    user = backend_entry.get("user") or data.get("user")
    host = backend_entry.get("host") or data.get("host")
    if not user or not host:
        raise ValueError("Global config missing user/host; run `koa setup` to fix.")

    remote_root_value = (
        backend_entry.get("remote_root")
        or data.get("remote_root")
        or f"/mnt/lustre/koa/scratch/{user}/koa-cli"
    )
    remote_root = Path(remote_root_value).expanduser()

    local_root_value = backend_entry.get("local_root") or data.get("local_root")
    if local_root_value:
        local_root = Path(local_root_value).expanduser()
    else:
        local_root = Path("~/koa-projects").expanduser()

    default_partition = (
        backend_entry.get("default_partition")
        or data.get("default_partition")
        or "kill-shared"
    )
    default_constraint = (
        backend_entry.get("default_constraint")
        or data.get("default_constraint")
        or ""
    )

    override_cuda_minor = args.cuda_version.strip() if args.cuda_version else None
    backend_cuda_minor = (
        override_cuda_minor
        or backend_entry.get("cuda_minor_version")
        or data.get("cuda_minor_version")
        or DEFAULT_CUDA_MINOR_VERSION
    )
    project_cuda_minor = str(backend_cuda_minor).strip() or DEFAULT_CUDA_MINOR_VERSION

    config_path = cwd / "koa-config.yaml"

    env_watch_lines = "\n".join(f"  - {item}" for item in DEFAULT_ENV_WATCH)

    config_template = _load_template("koa-config.yaml.tmpl")
    config_rendered = (
        config_template.replace("__PROJECT_NAME__", project_name)
        .replace("__DEFAULT_BACKEND__", backend_name)
        .replace("__CUDA_MINOR_VERSION__", project_cuda_minor)
        .replace("__DEFAULT_PARTITION__", default_partition)
        .replace("__ENV_WATCH__", env_watch_lines)
    )
    _write_file(config_path, config_rendered, overwrite=args.force)

    scripts_dir = cwd / "scripts"
    _write_file(
        scripts_dir / "setup_env.sh",
        _render_setup_env_script(project_cuda_minor),
        overwrite=args.force,
        executable=True,
    )

    _write_file(
        scripts_dir / "basic_job.slurm",
        _render_basic_job_template(project_name, default_partition, default_constraint),
        overwrite=args.force,
        executable=True,
    )

    # Keep remote paths lexical; resolve() on macOS rewrites /home/... to /System/Volumes/Data/home/...
    remote_project_root = remote_root / "projects" / project_name
    local_project_root = (local_root / "projects" / project_name).expanduser().resolve()
    local_jobs_root = local_project_root / "jobs"
    local_jobs_root.mkdir(parents=True, exist_ok=True)

    print(f"Initialised KOA project '{project_name}'")
    print(f"  Backend: {backend_name}")
    print(f"  Config: {config_path}")
    print(f"  CUDA minor version: {project_cuda_minor}")
    print(f"  Remote project root: {remote_project_root}")
    print(f"  Local project root: {local_project_root}")
    return 0


def _snapshot_ignore(
    root: Path,
    patterns: list[str],
):
    normalized_patterns: list[tuple[str, bool]] = []
    for pattern in patterns:
        if not pattern:
            continue
        raw = pattern.strip().replace("\\", "/")
        path_only = raw.startswith("./") or raw.startswith("/")
        cleaned = raw.removeprefix("./").lstrip("/")
        cleaned = cleaned.rstrip("/")
        if not cleaned:
            continue
        normalized_patterns.append((cleaned, path_only or "/" in cleaned))

    def _ignore(current_dir: str, names: list[str]) -> set[str]:
        ignored: set[str] = set()
        try:
            rel_root = Path(current_dir).resolve().relative_to(root)
        except Exception:
            rel_root = Path(".")
        rel_root_posix = rel_root.as_posix().lstrip("./")
        for name in names:
            rel_path = name
            if rel_root_posix and rel_root_posix != ".":
                rel_path = f"{rel_root_posix}/{name}"
            for cleaned, has_sep in normalized_patterns:
                if has_sep:
                    if fnmatch.fnmatch(rel_path, cleaned):
                        ignored.add(name)
                        break
                else:
                    if fnmatch.fnmatch(name, cleaned):
                        ignored.add(name)
                        break
        return ignored

    return _ignore


def _create_repo_snapshot(source: Path, destination: Path, extra_excludes: Optional[list[str]] = None) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    patterns = list(SNAPSHOT_IGNORE_PATTERNS)
    if extra_excludes:
        patterns.extend(extra_excludes)
    shutil.copytree(source, destination, ignore=_snapshot_ignore(source, patterns))

def _submit(args: argparse.Namespace, config: Config) -> int:
    script_sbatch_args = _sbatch_args_from_script(Path(args.job_script))

    sbatch_args: list[str] = []
    if args.partition:
        sbatch_args.extend(["--partition", args.partition])
    if args.constraint:
        sbatch_args.extend(["--constraint", args.constraint])
    if args.time:
        sbatch_args.extend(["--time", args.time])
    if args.gpus:
        sbatch_args.append(f"--gres=gpu:{args.gpus}")
    if args.cpus:
        sbatch_args.extend(["--cpus-per-task", str(args.cpus)])
    if args.memory:
        sbatch_args.extend(["--mem", args.memory])
    if args.account:
        sbatch_args.extend(["--account", args.account])
    elif config.default_account:
        sbatch_args.extend(["--account", config.default_account])
    if args.qos:
        sbatch_args.extend(["--qos", args.qos])
    sbatch_args.extend(args.sbatch_arg or [])

    if (
        (not args.constraint)
        and config.default_constraint
        and not _has_constraint_flag(sbatch_args)
        and not _has_constraint_flag(script_sbatch_args)
    ):
        sbatch_args.extend(["--constraint", config.default_constraint])
    if (
        config.default_gres
        and not _has_gres_flag(sbatch_args)
        and not _has_gres_flag(script_sbatch_args)
    ):
        sbatch_args.append(f"--gres={config.default_gres}")

    try:
        export_envs, missing_config_envs = _collect_export_envs(
            args.env or [],
            config.env_pass or [],
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if export_envs:
        if _has_export_flag(sbatch_args):
            print(
                "Cannot combine --env/env_pass with a manually provided --export sbatch argument.",
                file=sys.stderr,
            )
            return 1
        sbatch_args.append("--export=" + ",".join(["ALL", *export_envs]))
    if missing_config_envs:
        missing_list = ", ".join(missing_config_envs)
        print(
            f"Warning: skipped env_pass variables not set locally: {missing_list}",
            file=sys.stderr,
        )

    with tempfile.TemporaryDirectory(prefix="koa-submit-") as tmpdir:
        tmp_path = Path(tmpdir)
        manifest_path = tmp_path / "run_metadata"
        env_watch = config.env_watch_files or DEFAULT_ENV_WATCH
        write_run_manifest(manifest_path, env_watch=env_watch)
        update_manifest_metadata(
            manifest_path,
            sbatch_args=sbatch_args,
            job_script=str(args.job_script),
        )

        repo_snapshot_path = tmp_path / "repo"
        _create_repo_snapshot(Path.cwd(), repo_snapshot_path, config.snapshot_excludes)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        desc = args.desc or ""
        if desc:
            desc = re.sub(r"[^A-Za-z0-9_-]+", "_", desc).strip("_-")
        job_folder = timestamp if not desc else f"{timestamp}_{desc}"

        remote_job_dir: Optional[Path] = None
        if config.remote_results_dir:
            remote_job_dir = config.remote_results_dir / job_folder
            try:
                run_ssh(config, ["mkdir", "-p", str(remote_job_dir)])
                run_ssh(config, ["mkdir", "-p", str(remote_job_dir / "results")])
                copy_to_remote(
                    config,
                    manifest_path,
                    remote_job_dir / "run_metadata",
                    recursive=True,
                )
                copy_to_remote(
                    config,
                    repo_snapshot_path,
                    remote_job_dir / "repo",
                    recursive=True,
                )
            except SSHError as exc:
                print(f"Warning: failed to stage files on KOA: {exc}", file=sys.stderr)
                remote_job_dir = None

        local_job_dir: Optional[Path] = None
        if config.local_results_dir:
            local_job_dir = (config.local_results_dir / job_folder).expanduser()
            if local_job_dir.exists():
                shutil.rmtree(local_job_dir)
            local_job_dir.mkdir(parents=True, exist_ok=True)
            (local_job_dir / "results").mkdir(exist_ok=True)
            shutil.copytree(manifest_path, local_job_dir / "run_metadata")
            shutil.copytree(repo_snapshot_path, local_job_dir / "repo")

    job_id = submit_job(
        config,
        args.job_script,
        sbatch_args=sbatch_args,
        script_sbatch_args=script_sbatch_args,
        remote_name=args.remote_name,
        run_dir=remote_job_dir,
        job_desc=args.desc,
    )

    update_manifest_metadata(
        manifest_path,
        job_id=job_id,
        remote_code_dir=str(config.remote_code_dir),
        remote_results_dir=str(config.remote_results_dir) if config.remote_results_dir else None,
    )

    manifest_data = json.loads((manifest_path / "manifest.json").read_text(encoding="utf-8"))

    record_submission(
        config,
        job_id=job_id,
        sbatch_args=sbatch_args,
        manifest=manifest_data,
        local_job_dir=local_job_dir,
        remote_job_dir=remote_job_dir,
        description=args.desc,
    )

    print(f"Submitted KOA job {job_id}")
    return 0


def _cancel(args: argparse.Namespace, config: Config) -> int:
    cancel_job(config, args.job_id)
    print(f"Cancelled KOA job {args.job_id}")
    return 0


def _jobs(_: argparse.Namespace, config: Config) -> int:
    print(list_jobs(config), end="")
    return 0


def _check(_: argparse.Namespace, config: Config) -> int:
    print(run_health_checks(config), end="")
    return 0


def _dashboard(args: argparse.Namespace, config: Config) -> int:
    try:
        import streamlit  # type: ignore  # noqa: F401
    except ImportError:
        print(
            "Streamlit is not installed. Install it with `pip install koa-cli[dashboard]` or `pip install streamlit`.",
            file=sys.stderr,
        )
        return 1

    script = resources.files("koa_cli").joinpath("dashboard_app.py")
    script_path = str(script)
    if not os.path.exists(script_path):
        print("Unable to locate the dashboard app script.", file=sys.stderr)
        return 1

    cmd = [sys.executable, "-m", "streamlit", "run", script_path]
    extra_args: list[str] = []
    if args.config:
        extra_args.extend(["--config", str(args.config)])
    backend_name = args.backend or config.cluster_name
    if backend_name:
        extra_args.extend(["--backend", backend_name])
    cache_ttl = max(10, args.cache_ttl or 0)
    extra_args.extend(["--cache-ttl", str(cache_ttl)])

    if extra_args:
        cmd.append("--")
        cmd.extend(extra_args)

    print("Launching koa dashboard via Streamlit. Press Ctrl+C to stop.")
    result = subprocess.run(cmd)
    return result.returncode




def _logs(args: argparse.Namespace, config: Config) -> int:
    try:
        io_paths = get_job_io_paths(config, args.job_id)
    except SSHError as exc:
        print(f"Error querying job {args.job_id}: {exc}", file=sys.stderr)
        return 1

    target_path = io_paths.stdout if args.stream == "stdout" else io_paths.stderr

    if not target_path or target_path in {"UNKNOWN", "UNDEFINED"}:
        print(
            f"No {args.stream} log path reported for job {args.job_id}.",
            file=sys.stderr,
        )
        return 1

    quoted = shlex.quote(target_path)
    tail_flags = "-F" if args.follow else ""
    lines = max(0, args.lines)
    command = f"tail {tail_flags} -n {lines} {quoted}" if tail_flags else f"tail -n {lines} {quoted}"

    print(f"Streaming {args.stream} log: {target_path}")
    result = run_ssh(
        config,
        ["bash", "-lc", command],
        check=False,
    )
    return result.returncode


def _runs_list(args: argparse.Namespace, config: Config) -> int:
    if not config.local_results_dir:
        print("No local results directory configured; run `koa init` before recording runs.")
        return 0
    runs = list_runs(config)
    if not runs:
        print("No recorded runs yet. Submit a job with `koa submit` to create one.")
        return 0

    limit = max(1, args.limit)
    print(f"Showing {min(limit, len(runs))} of {len(runs)} run(s):")
    for entry in runs[:limit]:
        job_id = entry.get("job_id")
        status = entry.get("status") or "UNKNOWN"
        submitted = entry.get("submitted_at") or "---"
        remote_dir = entry.get("remote_job_dir") or "<remote>"
        print(f"- {job_id}: {status} @ {submitted} -> {remote_dir}")
    return 0


def _runs_sync(_: argparse.Namespace, config: Config) -> int:
    if not config.local_results_dir:
        print("No local results directory configured; run `koa init` first.")
        return 0
    updates = sync_statuses(config)
    if updates:
        print(f"Updated statuses for {updates} run(s).")
    else:
        print("No status changes detected.")
    return 0


def _runs_show(args: argparse.Namespace, config: Config) -> int:
    if not config.local_results_dir:
        print("No local results directory configured; run `koa init` first.")
        return 0
    entry = show_run(config, args.job_id)
    if not entry:
        print(f"No run recorded with job ID {args.job_id}.", file=sys.stderr)
        return 1
    print(json.dumps(entry, indent=2, sort_keys=True))
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "setup":
        return _setup(args)
    if args.command == "init":
        return _init_project(args)

    try:
        config = _load(args)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    try:
        if args.command == "submit":
            return _submit(args, config)
        if args.command == "cancel":
            return _cancel(args, config)
        if args.command == "jobs":
            return _jobs(args, config)
        if args.command == "dashboard":
            return _dashboard(args, config)
        if args.command == "check":
            return _check(args, config)
        if args.command == "logs":
            return _logs(args, config)
        if args.command == "runs":
            if args.runs_command == "list":
                return _runs_list(args, config)
            if args.runs_command == "sync":
                return _runs_sync(args, config)
            if args.runs_command == "show":
                return _runs_show(args, config)
    except (SSHError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    parser.error(f"Unhandled command {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
