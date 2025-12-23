"""MLflow logging utilities."""

from pathlib import Path
from typing import Any

import mlflow


def setup_mlflow(
    experiment_name: str,
    tracking_uri: str | None = None,
    run_name: str | None = None,
    tags: dict[str, str] | None = None,
) -> str:
    """Setup MLflow experiment and start run. Returns run_id."""
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(experiment_name)

    if mlflow.active_run() is None:
        mlflow.start_run(run_name=run_name, tags=tags)

    return mlflow.active_run().info.run_id


def log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
    mlflow.log_metrics(metrics, step=step)


def log_artifacts(artifact_paths: list[str | Path]) -> None:
    for path in artifact_paths:
        path = Path(path)
        if path.exists():
            if path.is_dir():
                mlflow.log_artifacts(str(path))
            else:
                mlflow.log_artifact(str(path))


def _flatten_dict(
    d: dict[str, Any], parent_key: str = "", sep: str = "."
) -> dict[str, Any]:
    """Flatten nested dict for MLflow params (truncates values >250 chars)."""
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            str_val = str(v)
            items.append(
                (new_key, str_val[:247] + "..." if len(str_val) > 250 else str_val)
            )
    return dict(items)
