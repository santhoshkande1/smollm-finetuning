"""Inspect saved LoRA files and recorded weight changes, without training.

With one completed adapter, run without arguments. If more exist, select one
with --run RUN_ID. The full base model is not loaded into memory.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file
from experiment_paths import select_completed_adapter


ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = ROOT / "models" / "base" / "SmolLM2-135M-Instruct"


def select_run():
    """Select a completed training run, avoiding guesses when several exist."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iteration", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--run", help="Training run's folder name.")
    args = parser.parse_args()
    try:
        return select_completed_adapter(args.iteration, args.run)
    except ValueError as error:
        parser.error(str(error))


def describe_weight_file(path):
    """Read shapes and storage types without loading the large base tensors."""
    with safe_open(path, framework="pt", device="cpu") as handle:
        slices = [handle.get_slice(name) for name in handle.keys()]
        return {
            "bytes": path.stat().st_size,
            "parameters": sum(math.prod(item.get_shape()) for item in slices),
            "tensors": len(slices),
            "dtypes": ", ".join(sorted({item.get_dtype() for item in slices})),
        }


def check_adapter_tensors(weights, config, base_config, metadata):
    """Check all expected A/B tables for the q_proj/v_proj adapter we trained."""
    if sorted(config["target_modules"]) != ["q_proj", "v_proj"]:
        raise ValueError("This inspection expects the q_proj/v_proj experiment.")
    hidden = base_config["hidden_size"]
    value_size = hidden // base_config["num_attention_heads"] * base_config["num_key_value_heads"]
    rank = config["r"]
    expected = {}
    for layer in range(base_config["num_hidden_layers"]):
        for projection, output_size in [("q_proj", hidden), ("v_proj", value_size)]:
            prefix = f"base_model.model.model.layers.{layer}.self_attn.{projection}"
            expected[f"{prefix}.lora_A.weight"] = (rank, hidden)
            expected[f"{prefix}.lora_B.weight"] = (output_size, rank)
    if set(weights) != set(expected):
        raise ValueError("Saved adapter does not contain exactly the expected LoRA tables.")
    for name, shape in expected.items():
        if tuple(weights[name].shape) != shape or not torch.isfinite(weights[name]).all().item():
            raise ValueError(f"Invalid adapter tensor: {name}")
    count = sum(weight.numel() for weight in weights.values())
    if count != metadata["trainable_parameters"]:
        raise ValueError("Saved parameter count differs from the training record.")


def show_weight_changes(weights, verification):
    """Compare actual saved values with before/after samples in the training log."""
    print("\nFirst-layer adapter weights: first five values")
    for training_name, before in verification["adapter_samples_before"].items():
        # PEFT omits the in-memory adapter name 'default' from saved tensor names.
        saved_name = training_name.replace(".lora_A.default.", ".lora_A.").replace(
            ".lora_B.default.", ".lora_B.",
        )
        tensor = weights[saved_name]
        after = tensor.flatten()[:5].tolist()
        if after != verification["adapter_samples_after"][training_name]:
            raise ValueError(f"Saved values differ from the training log: {saved_name}")
        label = saved_name.split("self_attn.", 1)[1]
        print(f"\n  {label}: {tuple(tensor.shape)}")
        print(f"    Before: {before}")
        print(f"    After:  {after}")

    b_tables = [weight for name, weight in weights.items() if ".lora_B." in name]
    nonzero = sum(torch.count_nonzero(weight).item() for weight in b_tables)
    total = sum(weight.numel() for weight in b_tables)
    print(f"\nB-table entries now nonzero: {nonzero:,} / {total:,}")
    print("These B tables started entirely at zero in our standard LoRA initialization.")


def main():
    # 1. Read the saved configuration and training records.
    adapter_dir, log_dir, metadata = select_run()
    config = json.loads((adapter_dir / "adapter_config.json").read_text())
    base_config = json.loads((BASE_DIR / "config.json").read_text())
    verification = json.loads((log_dir / "weight_verification.json").read_text())
    adapter_file = adapter_dir / "adapter_model.safetensors"
    base_file = BASE_DIR / "model.safetensors"

    # 2. Compare file sizes and inspect only the small adapter's actual weights.
    base_info = describe_weight_file(base_file)
    adapter_info = describe_weight_file(adapter_file)
    weights = load_file(adapter_file, device="cpu")
    check_adapter_tensors(weights, config, base_config, metadata)
    print(f"Run: {adapter_dir.name}")
    print("\nSaved files:")
    for path in sorted(adapter_dir.iterdir()):
        if path.is_file():
            print(f"  {path.name}: {path.stat().st_size:,} bytes")
    print(f"\nBase weight file:    {base_info['bytes'] / 1024**2:.2f} MiB ({base_info['dtypes']})")
    print(f"Adapter weight file: {adapter_info['bytes'] / 1024**2:.2f} MiB ({adapter_info['dtypes']})")
    print(f"Adapter file is {100 * adapter_info['bytes'] / base_info['bytes']:.2f}% of the base weight file's size.")
    print("File-size ratios differ from parameter ratios because the storage types differ.")
    print(f"\nBase parameters: {base_info['parameters']:,}")
    print(f"Adapter parameters: {adapter_info['parameters']:,} in {adapter_info['tensors']} tensors")
    print(f"Targets: {', '.join(sorted(config['target_modules']))} | rank: {config['r']}")

    # 3. Show learned values alongside their recorded initialization values.
    show_weight_changes(weights, verification)

    # 4. Distinguish recorded training checks from the file check performed now.
    with base_file.open("rb") as handle:
        current_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    if current_hash != metadata["base_weights_sha256"]:
        raise ValueError("The base model file differs from its pre-training fingerprint.")
    if not verification["all_original_tensors_unchanged"] or not verification["base_file_unchanged"]:
        raise ValueError("The recorded base-weight verification did not pass.")
    print(f"\nTraining record: {verification['original_tensors_checked']} original tensors stayed unchanged.")
    print(f"Training record: {verification['adapter_tensors_changed']}/{verification['adapter_tensors_total']} adapter tensors changed.")
    print("PASS now: saved adapter shapes, parameter count, and logged samples match.")
    print("PASS now: original model file still matches its pre-training fingerprint.")
    print("Read-only inspection complete. No training, answer generation, or file changes.")


if __name__ == "__main__":
    main()
