from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import yaml

DEFAULT_CONFIG_PATH = Path("~/.config/koa/config.yaml").expanduser()
PROJECT_CONFIG_FILENAMES: tuple[str, ...] = ("koa-config.yaml", ".koa-config.yaml")
DEFAULT_BACKEND_NAME = "koa"
DEFAULT_CUDA_MINOR_VERSION = "12.8"
BACKEND_SPECIFIC_KEYS: set[str] = {
    "cluster_name",
    "user",
    "host",
    "identity_file",
    "proxy_command",
    "project_name",
    "remote_root",
    "local_root",
    "default_account",
    "default_partition",
    "default_constraint",
    "default_gres",
    "cuda_minor_version",
    "dashboard_base_url",
    "env_pass",
}


@dataclass
class Config:
    cluster_name: str
    user: str
    host: str
    identity_file: Optional[Path] = None
    proxy_command: Optional[str] = None
    project_name: str = ""
    remote_root: Path = Path("~/koa-cli")
    local_root: Path = Path("./results")
    remote_project_root: Path = Path("~/koa-cli/projects/default")
    local_project_root: Path = Path("./results/projects/default")
    remote_code_dir: Path = Path("~/koa-cli/projects/default/jobs")
    remote_results_dir: Path = Path("~/koa-cli/projects/default/jobs")
    local_results_dir: Path = Path("./results/projects/default/jobs")
    shared_env_dir: Path = Path("~/koa-cli/projects/default/envs/uv")
    default_account: Optional[str] = None
    default_partition: Optional[str] = None
    default_constraint: Optional[str] = None
    default_gres: Optional[str] = None
    cuda_minor_version: str = DEFAULT_CUDA_MINOR_VERSION
    env_watch_files: List[str] = field(default_factory=list)
    snapshot_excludes: List[str] = field(default_factory=list)
    dashboard_base_url: Optional[str] = None
    env_pass: List[str] = field(default_factory=list)

    @property
    def login(self) -> str:
        return f"{self.user}@{self.host}"

    @property
    def remote_workdir(self) -> Path:
        """Backward compatible alias for the legacy field name."""
        return self.remote_code_dir


PathLikeOrStr = Union[os.PathLike[str], str]


def discover_config_path(start: Optional[PathLikeOrStr] = None) -> Path:
    """
    Locate the project configuration file by walking parent directories from `start`.

    Falls back to ~/.config/koa/config.yaml and to the legacy ~/.config/koa-ml/config.yaml location when a project-level
    config is not available.
    """

    if start is None:
        current = Path.cwd().resolve()
    else:
        current = Path(start).expanduser().resolve()

    for directory in [current, *current.parents]:
        for candidate_name in PROJECT_CONFIG_FILENAMES:
            candidate = directory / candidate_name
            if candidate.exists():
                return candidate

    if DEFAULT_CONFIG_PATH.exists():
        return DEFAULT_CONFIG_PATH

    searched_locations = [
        str(Path.cwd().resolve() / name) for name in PROJECT_CONFIG_FILENAMES
    ]
    raise FileNotFoundError(
        "Unable to locate koa-config.yaml in this project. "
        f"Searched: {', '.join(searched_locations)}, {DEFAULT_CONFIG_PATH}, "
        + ". Create one with `cp koa-config.example.yaml koa-config.yaml`."
    )
def load_config(
    config_path: Optional[PathLikeOrStr] = None,
    *,
    backend_name: Optional[str] = None,
) -> Config:
    """
    Load configuration from disk. When no path is provided we search for
    koa-config.yaml in the current project and fall back to ~/.config/koa/config.yaml
    or, for backwards compatibility, ~/.config/koa-ml/config.yaml.
    """

    if not DEFAULT_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Global configuration file not found at {DEFAULT_CONFIG_PATH}. "
            "Run `koa setup` to create one."
        )

    with DEFAULT_CONFIG_PATH.open("r", encoding="utf-8") as fh:
        global_data = yaml.safe_load(fh) or {}

    project_config_path: Optional[Path] = None
    if config_path:
        project_config_path = Path(config_path).expanduser()
    else:
        current = Path.cwd().resolve()
        for directory in [current, *current.parents]:
            candidate = directory / "koa-config.yaml"
            if candidate.exists() and candidate != DEFAULT_CONFIG_PATH:
                project_config_path = candidate
                break

    project_data: dict = {}
    if project_config_path and project_config_path.exists() and project_config_path != DEFAULT_CONFIG_PATH:
        with project_config_path.open("r", encoding="utf-8") as fh:
            project_data = yaml.safe_load(fh) or {}

    def _find_backend_entry(source: dict, name: str) -> dict:
        entries = source.get("backends")
        if isinstance(entries, list):
            for entry in entries:
                if entry.get("cluster_name") == name:
                    return dict(entry)
        return {}

    resolved_backend_name = (
        backend_name
        or os.getenv("KOA_BACKEND")
        or project_data.get("default_backend")
        or global_data.get("default_backend")
        or DEFAULT_BACKEND_NAME
    )

    backend_candidate = _find_backend_entry(global_data, resolved_backend_name)
    if not backend_candidate:
        backend_candidate = {
            key: global_data.get(key)
            for key in BACKEND_SPECIFIC_KEYS
            if global_data.get(key) is not None
        }
        if backend_candidate and "cluster_name" not in backend_candidate:
            backend_candidate["cluster_name"] = resolved_backend_name

    project_backend_override = _find_backend_entry(project_data, resolved_backend_name)

    merged_backend: dict = dict(backend_candidate)
    merged_backend.setdefault("cluster_name", resolved_backend_name)
    merged_backend.setdefault("cuda_minor_version", DEFAULT_CUDA_MINOR_VERSION)
    merged_backend.update(project_backend_override)
    for key in BACKEND_SPECIFIC_KEYS:
        if key == "cluster_name":
            continue
        if key in project_data and project_data[key] is not None:
            merged_backend[key] = project_data[key]

    env_overrides = {
        "user": os.getenv("KOA_USER"),
        "host": os.getenv("KOA_HOST"),
        "identity_file": os.getenv("KOA_IDENTITY_FILE"),
        "remote_root": os.getenv("KOA_REMOTE_ROOT"),
        "local_root": os.getenv("KOA_LOCAL_ROOT"),
        "default_account": os.getenv("KOA_ACCOUNT"),
        "default_partition": os.getenv("KOA_DEFAULT_PARTITION"),
        "default_constraint": os.getenv("KOA_DEFAULT_CONSTRAINT"),
        "default_gres": os.getenv("KOA_DEFAULT_GRES"),
        "cuda_minor_version": os.getenv("KOA_CUDA_VERSION")
        or os.getenv("KOA_CUDA_MINOR_VERSION"),
        "env_watch": os.getenv("KOA_ENV_WATCH"),
        "snapshot_excludes": os.getenv("KOA_SNAPSHOT_EXCLUDES"),
        "proxy_command": os.getenv("KOA_PROXY_COMMAND"),
        "dashboard_base_url": os.getenv("KOA_DASHBOARD_BASE_URL"),
        "env_pass": os.getenv("KOA_ENV_PASS"),
    }

    for key, value in env_overrides.items():
        if value is None:
            continue
        if key in {"env_watch", "snapshot_excludes", "env_pass"}:
            project_data[key] = [item.strip() for item in value.split(",") if item.strip()]
        else:
            merged_backend[key] = value

    missing = [key for key in ("user", "host") if not merged_backend.get(key)]
    if missing:
        raise ValueError(f"Missing required config keys: {', '.join(missing)}")

    identity_file = merged_backend.get("identity_file") or None
    identity_path: Optional[Path] = None
    if identity_file:
        identity_path = Path(identity_file).expanduser()
        if not identity_path.exists():
            raise FileNotFoundError(
                f"Configured identity_file not found: {identity_path}. "
                "Update the path or remove the identity_file setting to rely on your SSH defaults."
            )

    config_dir = project_config_path.parent if project_config_path else Path.cwd()

    project_name = (
        merged_backend.get("project_name")
        or project_data.get("project")
        or global_data.get("project")
    )
    if not project_name:
        if project_config_path and project_config_path.parent != DEFAULT_CONFIG_PATH.parent:
            project_name = project_config_path.parent.name
        else:
            project_name = "default"

    remote_root_value = merged_backend.get("remote_root") or (
        (project_data.get("remote") or {}).get("root")
        or (global_data.get("remote") or {}).get("root")
    )
    if not remote_root_value:
        raise ValueError("remote_root is not configured. Run `koa setup` or set remote_root in your global config.")
    remote_root = Path(remote_root_value).expanduser()

    local_root_value = merged_backend.get("local_root") or (
        (project_data.get("local") or {}).get("root")
        or (global_data.get("local") or {}).get("root")
    )
    if local_root_value:
        local_root = Path(local_root_value).expanduser()
        if not local_root.is_absolute():
            local_root = (config_dir / local_root).resolve()
    else:
        local_root = Path("./runs").resolve()

    # Keep remote paths lexical; resolve() on macOS rewrites /home/... to /System/Volumes/Data/home/...
    remote_project_root = remote_root / "projects" / project_name
    remote_jobs_root = remote_project_root / "jobs"
    remote_env_dir = remote_project_root / ".venv"
    local_project_root = (local_root / "projects" / project_name).resolve()
    local_jobs_root = local_project_root / "jobs"

    env_watch_raw = (
        project_data.get("env_watch")
        or project_data.get("env_watch_files")
        or global_data.get("env_watch")
        or global_data.get("env_watch_files")
        or []
    )
    if isinstance(env_watch_raw, str):
        env_watch_files = [env_watch_raw]
    else:
        env_watch_files = list(env_watch_raw)

    snapshot_excludes_raw = project_data.get("snapshot_excludes") or global_data.get("snapshot_excludes") or []
    if isinstance(snapshot_excludes_raw, str):
        snapshot_excludes = [snapshot_excludes_raw]
    else:
        snapshot_excludes = list(snapshot_excludes_raw)

    env_pass_raw = (
        merged_backend.get("env_pass")
        or project_data.get("env_pass")
        or global_data.get("env_pass")
        or []
    )
    if isinstance(env_pass_raw, str):
        env_pass = [item.strip() for item in env_pass_raw.split(",") if item.strip()]
    else:
        env_pass = [str(item).strip() for item in env_pass_raw if str(item).strip()]

    return Config(
        cluster_name=merged_backend.get("cluster_name", resolved_backend_name),
        user=merged_backend["user"],
        host=merged_backend["host"],
        identity_file=identity_path,
        proxy_command=merged_backend.get("proxy_command") or None,
        project_name=project_name,
        remote_root=remote_root,
        local_root=local_root,
        remote_project_root=remote_project_root,
        local_project_root=local_project_root,
        remote_code_dir=remote_jobs_root,
        remote_results_dir=remote_jobs_root,
        local_results_dir=local_jobs_root,
        shared_env_dir=remote_env_dir,
        default_account=merged_backend.get("default_account"),
        default_partition=merged_backend.get("default_partition"),
        default_constraint=merged_backend.get("default_constraint"),
        default_gres=merged_backend.get("default_gres"),
        cuda_minor_version=merged_backend.get("cuda_minor_version")
        or DEFAULT_CUDA_MINOR_VERSION,
        env_watch_files=env_watch_files,
        snapshot_excludes=snapshot_excludes,
        dashboard_base_url=merged_backend.get("dashboard_base_url"),
        env_pass=env_pass,
    )
