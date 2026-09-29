"""Shared iteration paths and completed-adapter discovery (no model loading)."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def iteration_data(iteration):
    if iteration not in (1, 2, 3):
        raise ValueError("Iteration must be 1, 2, or 3.")
    return ROOT / "data" / f"iteration_{iteration}"


def iteration_results(iteration):
    iteration_data(iteration)  # Validate the iteration.
    return ROOT / "results" / f"iteration_{iteration}"


def adapter_directory(iteration, run_id):
    iteration_data(iteration)  # Validate the iteration.
    if not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("Run ID must be a folder name, not a path.")
    return ROOT / "models" / "adapters" / f"iteration_{iteration}" / run_id


def select_completed_adapter(iteration=1, run_id=None):
    """Use actual directory locations; historical metadata paths stay untouched."""
    parent = adapter_directory(iteration, run_id or "unused").parent
    candidates = []
    for folder in sorted(parent.iterdir()) if parent.exists() else []:
        if run_id is not None and folder.name != run_id:
            continue
        log_dir = iteration_results(iteration) / "training" / folder.name
        record = log_dir / "metadata.json"
        if (folder / "adapter_model.safetensors").is_file() and record.is_file():
            metadata = json.loads(record.read_text())
            if metadata.get("status") == "complete" and metadata.get("iteration", 1) == iteration:
                candidates.append((folder, log_dir, metadata))
    if len(candidates) != 1:
        names = ", ".join(item[0].name for item in candidates) or "none"
        raise ValueError(f"Select one completed iteration {iteration} adapter with --run RUN_ID. Matches: {names}")
    return candidates[0]
