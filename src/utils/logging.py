"""MLflow logging utilities."""

import os
import subprocess
from pathlib import Path
from typing import Any

import mlflow


def setup_mlflow(
    experiment_name: str,
    tracking_uri: str | None = None,
    run_name: str | None = None,
    tags: dict[str, str] | None = None,
) -> str:
    """Setup MLflow tracking.

    Args:
        experiment_name: Name of the MLflow experiment.
        tracking_uri: MLflow tracking URI (default: local ./mlruns).
        run_name: Optional run name.
        tags: Optional run tags.

    Returns:
        Run ID.
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(experiment_name)

    # Start run if not already active
    if mlflow.active_run() is None:
        mlflow.start_run(run_name=run_name, tags=tags)

    return mlflow.active_run().info.run_id


def log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
    """Log metrics to MLflow.

    Args:
        metrics: Dict of metric name -> value.
        step: Optional step number.
    """
    mlflow.log_metrics(metrics, step=step)


def log_artifacts(artifact_paths: list[str | Path]) -> None:
    """Log artifacts to MLflow.

    Args:
        artifact_paths: List of paths to log.
    """
    for path in artifact_paths:
        path = Path(path)
        if path.exists():
            if path.is_dir():
                mlflow.log_artifacts(str(path))
            else:
                mlflow.log_artifact(str(path))

def _flatten_dict(
    d: dict[str, Any],
    parent_key: str = "",
    sep: str = ".",
) -> dict[str, Any]:
    """Flatten nested dict for MLflow params.

    Args:
        d: Dict to flatten.
        parent_key: Prefix for keys.
        sep: Separator between nested keys.

    Returns:
        Flattened dict.
    """
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            # Truncate long values for MLflow param limit
            str_val = str(v)
            if len(str_val) > 250:
                str_val = str_val[:247] + "..."
            items.append((new_key, str_val))
    return dict(items)




