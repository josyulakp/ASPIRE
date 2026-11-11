"""Typed discovery for the PR-chopsticks insertion scenes (insertion_skill ``envs/<task>/``)."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any

DEFAULT_HORIZON = 1200
"""Control ticks per episode (60 s at 20 Hz): enough for a blind lateral search; the
force-guided baseline in insertion_skill seats in under 3 s."""


class InsertionRegistryError(ValueError):
    """Raised when an insertion task reference is invalid."""


def default_envs_root() -> Path:
    """``INSERTION_ENVS``, else ``<insertion_skill repo>/envs`` next to the ASPIRE checkout."""
    configured = os.environ.get("INSERTION_ENVS")
    if configured:
        return Path(configured).expanduser().resolve()
    try:
        from insertion_skill.env import ENVS

        return Path(ENVS).resolve()
    except ImportError:
        return Path(__file__).resolve().parents[4] / "envs"


@dataclass(frozen=True, slots=True)
class InsertionTaskMetadata:
    task_name: str
    language: str
    scene_dir: str
    mount: str
    task_id: int = 0
    family: str = "insertion"
    horizon: int | None = DEFAULT_HORIZON
    camera_names: tuple[str, ...] = ("closeup", "overview")

    @property
    def suite_name(self) -> str:
        return self.task_name

    @property
    def task_ref(self) -> str:
        return f"{self.task_name}:{self.task_id}"

    @property
    def arms(self) -> tuple[str, ...]:
        return ("primary",)

    def to_manifest_record(self) -> dict[str, Any]:
        return {
            "arms": list(self.arms),
            "camera_names": list(self.camera_names),
            "family": self.family,
            "language": self.language,
            "mount": self.mount,
            "scene_dir": self.scene_dir,
            "suite_name": self.suite_name,
            "task_id": self.task_id,
            "task_name": self.task_name,
            "task_ref": self.task_ref,
        }


def discover_tasks(envs_root: Path | None = None) -> tuple[InsertionTaskMetadata, ...]:
    """Every ``envs/<task>/`` that has both a scene and a manifest, sorted by name."""
    root = Path(envs_root) if envs_root is not None else default_envs_root()
    tasks = []
    for manifest_path in sorted(root.glob("*/manifest.json")):
        scene_dir = manifest_path.parent
        if not (scene_dir / "scene.xml").is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        tasks.append(
            InsertionTaskMetadata(
                task_name=str(manifest.get("task", scene_dir.name)),
                language=str(manifest.get("description", scene_dir.name.replace("_", " "))),
                scene_dir=str(scene_dir),
                mount=str(manifest.get("mount", "unknown")),
            )
        )
    return tuple(tasks)


class InsertionTaskRegistry:
    """Resolve the generated insertion scenes; each task has the single task_id 0."""

    def __init__(
        self,
        tasks: tuple[InsertionTaskMetadata, ...] | None = None,
        *,
        envs_root: Path | None = None,
    ) -> None:
        resolved = tasks if tasks is not None else discover_tasks(envs_root)
        self._tasks = {task.task_name: task for task in resolved}
        if len(self._tasks) != len(resolved):
            raise InsertionRegistryError("insertion task names must be unique")

    @property
    def available_tasks(self) -> tuple[str, ...]:
        return tuple(self._tasks)

    def enumerate_tasks(self) -> tuple[InsertionTaskMetadata, ...]:
        return tuple(self._tasks.values())

    def resolve(
        self,
        task_ref: str | tuple[str, int] | InsertionTaskMetadata,
        task_id: int | None = None,
    ) -> InsertionTaskMetadata:
        if isinstance(task_ref, InsertionTaskMetadata):
            return task_ref
        if isinstance(task_ref, tuple):
            if len(task_ref) != 2:
                raise InsertionRegistryError("task tuple must contain (task_name, task_id)")
            task_name, tuple_id = task_ref
            if task_id is not None and task_id != tuple_id:
                raise InsertionRegistryError("conflicting insertion task ids")
            task_id = int(tuple_id)
        else:
            task_name = task_ref
            if ":" in task_name:
                task_name, encoded_id = task_name.rsplit(":", 1)
                if task_id is not None and task_id != int(encoded_id):
                    raise InsertionRegistryError("conflicting insertion task ids")
                task_id = int(encoded_id)
        if task_id is None:
            task_id = 0
        if task_id != 0:
            raise InsertionRegistryError("each insertion task currently has task_id 0")
        try:
            return self._tasks[str(task_name)]
        except KeyError as exc:
            raise InsertionRegistryError(
                f"unknown insertion task {task_name!r}; available: "
                f"{', '.join(self.available_tasks) or '(none found)'}"
            ) from exc


__all__ = [
    "DEFAULT_HORIZON",
    "InsertionRegistryError",
    "InsertionTaskMetadata",
    "InsertionTaskRegistry",
    "default_envs_root",
    "discover_tasks",
]
