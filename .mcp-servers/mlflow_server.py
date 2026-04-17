#!/usr/bin/env python3
"""
MLflow MCP Server - Extended for Traditional ML Experiment Tracking

This script creates a Model Context Protocol (MCP) server that connects to MLflow,
exposing comprehensive ML tracking functionality through standardized tools
that AI assistants can use.

Environment variables:
    MLFLOW_TRACKING_URI: URI of the MLflow tracking server (default: file-based local)
    LOG_LEVEL: Logging level (default: INFO)

Usage:
    python mlflow_server.py
"""

import json
import logging
import os
import sys
from typing import Any, Optional
from datetime import datetime
from statistics import mean, median, stdev, variance

import mlflow
from mlflow import MlflowClient

# Set up logging
log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("mlflow-mcp-server")

# Set MLflow tracking URI from environment variable
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "")
if TRACKING_URI:
    mlflow.set_tracking_uri(uri=TRACKING_URI)
logger.info(f"Using MLflow tracking URI: {mlflow.get_tracking_uri()}")

# Get MLflow client
client = MlflowClient()

try:
    from mcp.server.fastmcp import FastMCP

    # Initialize FastMCP server with focused instructions
    mcp = FastMCP(
        name="mlflow-experiments",
        instructions="""
        I help you interact with MLflow to manage machine learning experiments.
        
        I can:
        - List and search experiments and runs
        - Get detailed metrics, parameters, and artifacts
        - Compare runs and find best performers
        - Calculate statistics on metrics across runs
        - Analyze experiment results for insights
        """
    )
except ImportError:
    logger.error("Failed to import MCP. Please install with: pip install mcp fastmcp")
    sys.exit(1)


class MLflowTools:
    """Collection of helper utilities for MLflow interactions."""

    @staticmethod
    def _format_timestamp(timestamp_ms: Optional[int]) -> str:
        """Convert a millisecond timestamp to a human-readable string."""
        if not timestamp_ms:
            return "N/A"
        dt = datetime.fromtimestamp(timestamp_ms / 1000.0)
        return dt.strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def _format_duration(start_ms: Optional[int], end_ms: Optional[int]) -> str:
        """Calculate duration between timestamps."""
        if not start_ms or not end_ms:
            return "N/A"
        duration_s = (end_ms - start_ms) / 1000.0
        if duration_s < 60:
            return f"{duration_s:.1f}s"
        elif duration_s < 3600:
            return f"{duration_s / 60:.1f}m"
        else:
            return f"{duration_s / 3600:.1f}h"

    @staticmethod
    def _safe_float(value: Any) -> float:
        """Safely convert value to float."""
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("nan")

    @staticmethod
    def calculate_statistics(values: list[float]) -> dict[str, float]:
        """Calculate comprehensive statistics for a list of values."""
        # Filter out NaN values
        clean_values = [v for v in values if v == v]  # NaN != NaN
        
        if not clean_values:
            return {"error": "No valid values to calculate statistics"}
        
        stats = {
            "count": len(clean_values),
            "min": min(clean_values),
            "max": max(clean_values),
            "mean": mean(clean_values),
            "median": median(clean_values),
            "range": max(clean_values) - min(clean_values),
        }
        
        if len(clean_values) >= 2:
            stats["std"] = stdev(clean_values)
            stats["variance"] = variance(clean_values)
        
        # Round for readability
        for key in stats:
            if isinstance(stats[key], float):
                stats[key] = round(stats[key], 6)
        
        return stats


# =============================================================================
# EXPERIMENT TOOLS
# =============================================================================

@mcp.tool()
def list_experiments(name_contains: str = "", max_results: int = 100) -> str:
    """
    List all experiments in MLflow with optional name filtering.
    
    Args:
        name_contains: Filter experiments whose names contain this string (case-insensitive)
        max_results: Maximum number of experiments to return (default: 100)
    
    Returns:
        JSON with experiment list including: id, name, run_count, lifecycle_stage, creation_time
    """
    logger.info(f"Listing experiments (filter: '{name_contains}', max: {max_results})")
    
    try:
        experiments = client.search_experiments()
        
        if name_contains:
            experiments = [
                exp for exp in experiments
                if name_contains.lower() in exp.name.lower()
            ]
        
        experiments = experiments[:max_results]
        
        experiments_info = []
        for exp in experiments:
            # Get run count
            try:
                runs = client.search_runs(experiment_ids=[exp.experiment_id])
                run_count = len(runs)
            except Exception:
                run_count = "Error"
            
            exp_info = {
                "experiment_id": exp.experiment_id,
                "name": exp.name,
                "run_count": run_count,
                "lifecycle_stage": exp.lifecycle_stage,
                "artifact_location": exp.artifact_location,
                "creation_time": MLflowTools._format_timestamp(
                    getattr(exp, "creation_time", None)
                ),
            }
            experiments_info.append(exp_info)
        
        return json.dumps({
            "total_experiments": len(experiments_info),
            "experiments": experiments_info
        }, indent=2)
    
    except Exception as e:
        logger.error(f"Error listing experiments: {e}", exc_info=True)
        return json.dumps({"error": str(e)})


# =============================================================================
# RUN TOOLS
# =============================================================================

@mcp.tool()
def list_runs(
    experiment_id: str,
    max_results: int = 50,
    order_by: str = "start_time DESC",
    filter_string: str = ""
) -> str:
    """
    List runs in an experiment with their key metrics and parameters.
    
    Args:
        experiment_id: The experiment ID to list runs from
        max_results: Maximum runs to return (default: 50)
        order_by: Sort order (default: "start_time DESC"). Options: start_time, end_time, or metric name
        filter_string: MLflow filter string (e.g., "metrics.accuracy > 0.9")
    
    Returns:
        JSON with runs including: run_id, status, duration, metrics, params summary
    """
    logger.info(f"Listing runs for experiment {experiment_id}")
    
    try:
        runs = client.search_runs(
            experiment_ids=[experiment_id],
            max_results=max_results,
            order_by=[order_by] if order_by else None,
            filter_string=filter_string if filter_string else None
        )
        
        runs_info = []
        for run in runs:
            # Get top metrics (limit to avoid huge output)
            metrics = {}
            for k, v in list(run.data.metrics.items())[:15]:
                metrics[k] = round(v, 6) if isinstance(v, float) else v
            
            # Get top params
            params = dict(list(run.data.params.items())[:10])
            
            run_info = {
                "run_id": run.info.run_id,
                "run_name": run.info.run_name or run.info.run_id[:8],
                "status": run.info.status,
                "start_time": MLflowTools._format_timestamp(run.info.start_time),
                "duration": MLflowTools._format_duration(
                    run.info.start_time, run.info.end_time
                ),
                "metrics": metrics,
                "params": params,
            }
            runs_info.append(run_info)
        
        return json.dumps({
            "experiment_id": experiment_id,
            "total_runs": len(runs_info),
            "runs": runs_info
        }, indent=2)
    
    except Exception as e:
        logger.error(f"Error listing runs: {e}", exc_info=True)
        return json.dumps({"error": str(e)})


@mcp.tool()
def get_run_details(run_id: str) -> str:
    """
    Get comprehensive details for a specific run.
    
    Args:
        run_id: The run ID to get details for
    
    Returns:
        JSON with full run info: all metrics, params, tags, artifacts, and timing
    """
    logger.info(f"Getting details for run {run_id}")
    
    try:
        run = client.get_run(run_id)
        
        # All metrics
        metrics = {}
        for k, v in run.data.metrics.items():
            metrics[k] = round(v, 6) if isinstance(v, float) else v
        
        # All params
        params = dict(run.data.params)
        
        # Tags (filter out internal mlflow tags for readability)
        tags = {
            k: v for k, v in run.data.tags.items()
            if not k.startswith("mlflow.")
        }
        
        # Artifacts
        try:
            artifacts = client.list_artifacts(run_id)
            artifact_list = [
                {"path": a.path, "size": getattr(a, "file_size", "N/A")}
                for a in artifacts
            ]
        except Exception:
            artifact_list = []
        
        run_info = {
            "run_id": run.info.run_id,
            "run_name": run.info.run_name,
            "experiment_id": run.info.experiment_id,
            "status": run.info.status,
            "start_time": MLflowTools._format_timestamp(run.info.start_time),
            "end_time": MLflowTools._format_timestamp(run.info.end_time),
            "duration": MLflowTools._format_duration(
                run.info.start_time, run.info.end_time
            ),
            "artifact_uri": run.info.artifact_uri,
            "metrics": metrics,
            "params": params,
            "tags": tags,
            "artifacts": artifact_list,
        }
        
        return json.dumps(run_info, indent=2)
    
    except Exception as e:
        logger.error(f"Error getting run details: {e}", exc_info=True)
        return json.dumps({"error": str(e)})


@mcp.tool()
def get_best_run(
    experiment_id: str,
    metric_name: str,
    maximize: bool = True,
    top_n: int = 1
) -> str:
    """
    Find the best run(s) in an experiment based on a metric.
    
    Args:
        experiment_id: The experiment ID to search
        metric_name: The metric to optimize (e.g., "val/miou", "accuracy")
        maximize: If True, find highest value; if False, find lowest (default: True)
        top_n: Number of top runs to return (default: 1)
    
    Returns:
        JSON with the best run(s) details including the optimized metric value
    """
    logger.info(f"Finding best run by {metric_name} in experiment {experiment_id}")
    
    try:
        order = "DESC" if maximize else "ASC"
        runs = client.search_runs(
            experiment_ids=[experiment_id],
            order_by=[f"metrics.{metric_name} {order}"],
            max_results=top_n
        )
        
        if not runs:
            return json.dumps({"error": f"No runs found with metric '{metric_name}'"})
        
        best_runs = []
        for i, run in enumerate(runs):
            metric_value = run.data.metrics.get(metric_name)
            
            run_info = {
                "rank": i + 1,
                "run_id": run.info.run_id,
                "run_name": run.info.run_name or run.info.run_id[:8],
                f"{metric_name}": round(metric_value, 6) if metric_value else None,
                "status": run.info.status,
                "start_time": MLflowTools._format_timestamp(run.info.start_time),
                "params": dict(run.data.params),
            }
            best_runs.append(run_info)
        
        return json.dumps({
            "experiment_id": experiment_id,
            "metric": metric_name,
            "optimization": "maximize" if maximize else "minimize",
            "best_runs": best_runs
        }, indent=2)
    
    except Exception as e:
        logger.error(f"Error finding best run: {e}", exc_info=True)
        return json.dumps({"error": str(e)})


@mcp.tool()
def compare_runs(run_ids: str, metrics: str = "") -> str:
    """
    Compare multiple runs side-by-side on specified metrics.
    
    Args:
        run_ids: Comma-separated list of run IDs to compare (e.g., "run1,run2,run3")
        metrics: Comma-separated list of metrics to compare (empty = all common metrics)
    
    Returns:
        JSON with comparison table showing each run's values for each metric
    """
    run_id_list = [rid.strip() for rid in run_ids.split(",")]
    metric_list = [m.strip() for m in metrics.split(",") if m.strip()] if metrics else []
    
    logger.info(f"Comparing runs: {run_id_list}")
    
    try:
        runs_data = []
        all_metrics = set()
        all_params = set()
        
        for run_id in run_id_list:
            run = client.get_run(run_id)
            runs_data.append(run)
            all_metrics.update(run.data.metrics.keys())
            all_params.update(run.data.params.keys())
        
        # Filter metrics if specified
        if metric_list:
            compare_metrics = [m for m in metric_list if m in all_metrics]
        else:
            compare_metrics = sorted(all_metrics)
        
        # Build comparison
        comparison = {
            "runs": [],
            "metrics_comparison": {},
            "params_comparison": {},
        }
        
        for run in runs_data:
            comparison["runs"].append({
                "run_id": run.info.run_id,
                "run_name": run.info.run_name or run.info.run_id[:8],
                "status": run.info.status,
            })
        
        # Compare metrics
        for metric in compare_metrics:
            values = []
            for run in runs_data:
                val = run.data.metrics.get(metric)
                values.append(round(val, 6) if val is not None else None)
            
            comparison["metrics_comparison"][metric] = {
                "values": values,
                "best_idx": values.index(max(v for v in values if v is not None)) 
                           if any(v is not None for v in values) else None
            }
        
        # Compare params (show only differing ones)
        for param in sorted(all_params):
            values = [run.data.params.get(param, "N/A") for run in runs_data]
            if len(set(values)) > 1:  # Only show if different
                comparison["params_comparison"][param] = values
        
        return json.dumps(comparison, indent=2)
    
    except Exception as e:
        logger.error(f"Error comparing runs: {e}", exc_info=True)
        return json.dumps({"error": str(e)})


@mcp.tool()
def search_runs(
    experiment_id: str,
    filter_string: str,
    max_results: int = 20
) -> str:
    """
    Search runs using MLflow filter syntax.
    
    Args:
        experiment_id: The experiment ID to search in
        filter_string: MLflow filter (e.g., "metrics.`val/miou` > 0.7 AND params.model = 'segformer'")
        max_results: Maximum runs to return (default: 20)
    
    Returns:
        JSON with matching runs
    
    Filter examples:
        - "metrics.accuracy > 0.9"
        - "params.learning_rate = '0.001'"
        - "metrics.`val/loss` < 0.5"
        - "status = 'FINISHED'"
        - "tags.mlflow.runName LIKE '%baseline%'"
    """
    logger.info(f"Searching runs: {filter_string}")
    
    try:
        runs = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string=filter_string,
            max_results=max_results
        )
        
        runs_info = []
        for run in runs:
            metrics = {k: round(v, 6) for k, v in list(run.data.metrics.items())[:10]}
            
            runs_info.append({
                "run_id": run.info.run_id,
                "run_name": run.info.run_name or run.info.run_id[:8],
                "status": run.info.status,
                "metrics": metrics,
                "params": dict(list(run.data.params.items())[:8]),
            })
        
        return json.dumps({
            "filter": filter_string,
            "matches": len(runs_info),
            "runs": runs_info
        }, indent=2)
    
    except Exception as e:
        logger.error(f"Error searching runs: {e}", exc_info=True)
        return json.dumps({"error": str(e)})


# =============================================================================
# METRICS & STATISTICS TOOLS
# =============================================================================

@mcp.tool()
def get_metric_statistics(experiment_id: str, metric_name: str) -> str:
    """
    Calculate comprehensive statistics for a metric across all runs in an experiment.
    
    Args:
        experiment_id: The experiment ID to analyze
        metric_name: The metric to calculate statistics for (e.g., "val/miou")
    
    Returns:
        JSON with statistics: count, min, max, mean, median, std, variance, range
        Plus the run_ids of best and worst performers
    """
    logger.info(f"Calculating statistics for {metric_name} in experiment {experiment_id}")
    
    try:
        runs = client.search_runs(experiment_ids=[experiment_id])
        
        values = []
        run_values = []
        
        for run in runs:
            val = run.data.metrics.get(metric_name)
            if val is not None:
                values.append(val)
                run_values.append({
                    "run_id": run.info.run_id,
                    "run_name": run.info.run_name or run.info.run_id[:8],
                    "value": round(val, 6)
                })
        
        if not values:
            return json.dumps({
                "error": f"No runs found with metric '{metric_name}'"
            })
        
        # Sort by value
        run_values.sort(key=lambda x: x["value"], reverse=True)
        
        stats = MLflowTools.calculate_statistics(values)
        
        result = {
            "experiment_id": experiment_id,
            "metric": metric_name,
            "statistics": stats,
            "best_run": run_values[0] if run_values else None,
            "worst_run": run_values[-1] if run_values else None,
            "all_values": run_values
        }
        
        return json.dumps(result, indent=2)
    
    except Exception as e:
        logger.error(f"Error calculating statistics: {e}", exc_info=True)
        return json.dumps({"error": str(e)})


@mcp.tool()
def get_metric_history(run_id: str, metric_name: str) -> str:
    """
    Get the full history of a metric for a run (all logged values across steps/epochs).
    
    Args:
        run_id: The run ID
        metric_name: The metric name (e.g., "train/loss", "val/miou")
    
    Returns:
        JSON with metric history: list of (step, value, timestamp) tuples and summary stats
    """
    logger.info(f"Getting history for {metric_name} in run {run_id}")
    
    try:
        history = client.get_metric_history(run_id, metric_name)
        
        if not history:
            return json.dumps({
                "error": f"No history found for metric '{metric_name}'"
            })
        
        values = [h.value for h in history]
        
        history_data = [
            {
                "step": h.step,
                "value": round(h.value, 6),
                "timestamp": MLflowTools._format_timestamp(h.timestamp)
            }
            for h in history
        ]
        
        # Calculate trend
        if len(values) >= 2:
            trend = "increasing" if values[-1] > values[0] else "decreasing"
            improvement = round(values[-1] - values[0], 6)
        else:
            trend = "N/A"
            improvement = 0
        
        result = {
            "run_id": run_id,
            "metric": metric_name,
            "total_steps": len(history_data),
            "first_value": round(values[0], 6),
            "last_value": round(values[-1], 6),
            "trend": trend,
            "total_change": improvement,
            "statistics": MLflowTools.calculate_statistics(values),
            "history": history_data
        }
        
        return json.dumps(result, indent=2)
    
    except Exception as e:
        logger.error(f"Error getting metric history: {e}", exc_info=True)
        return json.dumps({"error": str(e)})


@mcp.tool()
def compare_metrics_across_experiments(
    experiment_ids: str,
    metric_name: str
) -> str:
    """
    Compare a metric's performance across multiple experiments.
    
    Args:
        experiment_ids: Comma-separated experiment IDs (e.g., "123,456,789")
        metric_name: The metric to compare
    
    Returns:
        JSON with per-experiment statistics and cross-experiment comparison
    """
    exp_id_list = [eid.strip() for eid in experiment_ids.split(",")]
    logger.info(f"Comparing {metric_name} across experiments: {exp_id_list}")
    
    try:
        experiment_stats = []
        
        for exp_id in exp_id_list:
            try:
                exp = client.get_experiment(exp_id)
                exp_name = exp.name
            except Exception:
                exp_name = exp_id
            
            runs = client.search_runs(experiment_ids=[exp_id])
            values = []
            
            for run in runs:
                val = run.data.metrics.get(metric_name)
                if val is not None:
                    values.append(val)
            
            if values:
                stats = MLflowTools.calculate_statistics(values)
                experiment_stats.append({
                    "experiment_id": exp_id,
                    "experiment_name": exp_name,
                    "run_count": len(values),
                    "statistics": stats
                })
        
        # Rank experiments by mean
        experiment_stats.sort(
            key=lambda x: x["statistics"].get("mean", 0),
            reverse=True
        )
        
        for i, exp in enumerate(experiment_stats):
            exp["rank"] = i + 1
        
        return json.dumps({
            "metric": metric_name,
            "experiments_compared": len(experiment_stats),
            "comparison": experiment_stats
        }, indent=2)
    
    except Exception as e:
        logger.error(f"Error comparing experiments: {e}", exc_info=True)
        return json.dumps({"error": str(e)})


# =============================================================================
# MODEL REGISTRY TOOLS
# =============================================================================

@mcp.tool()
def list_models(name_contains: str = "", max_results: int = 100) -> str:
    """
    List all registered models in the MLflow model registry.
    
    Args:
        name_contains: Filter models whose names contain this string
        max_results: Maximum models to return (default: 100)
    
    Returns:
        JSON with registered models and their latest versions
    """
    logger.info(f"Listing registered models (filter: '{name_contains}')")
    
    try:
        models = client.search_registered_models(max_results=max_results)
        
        if name_contains:
            models = [
                m for m in models
                if name_contains.lower() in m.name.lower()
            ]
        
        models_info = []
        for model in models:
            model_info = {
                "name": model.name,
                "creation_time": MLflowTools._format_timestamp(model.creation_timestamp),
                "last_updated": MLflowTools._format_timestamp(model.last_updated_timestamp),
                "description": model.description or "",
                "latest_versions": []
            }
            
            if model.latest_versions:
                for version in model.latest_versions:
                    model_info["latest_versions"].append({
                        "version": version.version,
                        "stage": version.current_stage,
                        "status": version.status,
                        "run_id": version.run_id
                    })
            
            models_info.append(model_info)
        
        return json.dumps({
            "total_models": len(models_info),
            "models": models_info
        }, indent=2)
    
    except Exception as e:
        logger.error(f"Error listing models: {e}", exc_info=True)
        return json.dumps({"error": str(e)})


@mcp.tool()
def get_model_details(model_name: str) -> str:
    """
    Get detailed information about a registered model including all versions.
    
    Args:
        model_name: The name of the registered model
    
    Returns:
        JSON with model details including all versions and their source runs
    """
    logger.info(f"Getting details for model: {model_name}")
    
    try:
        model = client.get_registered_model(model_name)
        
        model_info = {
            "name": model.name,
            "creation_time": MLflowTools._format_timestamp(model.creation_timestamp),
            "last_updated": MLflowTools._format_timestamp(model.last_updated_timestamp),
            "description": model.description or "",
            "versions": []
        }
        
        versions = client.search_model_versions(f"name='{model_name}'")
        
        for version in versions:
            version_info = {
                "version": version.version,
                "stage": version.current_stage,
                "status": version.status,
                "creation_time": MLflowTools._format_timestamp(version.creation_timestamp),
                "run_id": version.run_id,
                "source": version.source
            }
            
            # Get run metrics if available
            if version.run_id:
                try:
                    run = client.get_run(version.run_id)
                    version_info["run_metrics"] = {
                        k: round(v, 6) for k, v in list(run.data.metrics.items())[:10]
                    }
                except Exception:
                    pass
            
            model_info["versions"].append(version_info)
        
        return json.dumps(model_info, indent=2)
    
    except Exception as e:
        logger.error(f"Error getting model details: {e}", exc_info=True)
        return json.dumps({"error": str(e)})


# =============================================================================
# SYSTEM TOOLS
# =============================================================================

@mcp.tool()
def get_system_info() -> str:
    """
    Get information about the MLflow tracking server and overall statistics.
    
    Returns:
        JSON with MLflow version, tracking URI, experiment count, model count, etc.
    """
    logger.info("Getting MLflow system information")
    
    try:
        info = {
            "mlflow_version": mlflow.__version__,
            "tracking_uri": mlflow.get_tracking_uri(),
            "registry_uri": mlflow.get_registry_uri(),
            "server_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        
        # Get experiment count
        try:
            experiments = client.search_experiments()
            info["experiment_count"] = len(experiments)
            
            # Count total runs
            total_runs = 0
            for exp in experiments:
                runs = client.search_runs(experiment_ids=[exp.experiment_id])
                total_runs += len(runs)
            info["total_runs"] = total_runs
        except Exception as e:
            info["experiment_count"] = f"Error: {e}"
        
        # Get model count
        try:
            models = client.search_registered_models()
            info["model_count"] = len(models)
        except Exception:
            info["model_count"] = 0
        
        return json.dumps(info, indent=2)
    
    except Exception as e:
        logger.error(f"Error getting system info: {e}", exc_info=True)
        return json.dumps({"error": str(e)})


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    try:
        logger.info(f"Starting MLflow MCP server with tracking URI: {mlflow.get_tracking_uri()}")
        mcp.run(transport="stdio")
    except Exception as e:
        logger.error(f"Error running MCP server: {e}", exc_info=True)
        sys.exit(1)
