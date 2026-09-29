"""Train a fresh LoRA adapter using a selected iteration's training examples.

Run normally to train and save an adapter. Use --dry-run to check one batch
and its gradients without updating weights or creating output files.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import peft
import torch
import transformers
from transformers import AutoConfig, AutoTokenizer
from experiment_paths import adapter_directory, iteration_data, iteration_results
from practice_checks import prepare_practice, save_practice_checkpoint

from inspect_lora import (
    attach_lora,
    load_model_and_tokenizer,
    record_original_weights,
    weight_hash,
)
from prepare_training_data import (
    IGNORE_LABEL,
    MODEL_PATH,
    ROOT,
    load_examples,
    make_batch,
    prepare_example,
)


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iteration", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--checkpoint-every", type=int, default=5,
                        help="Iterations 2/3: save and check every N epochs, plus the last epoch.")
    parser.add_argument("--epochs", type=int, default=30, help="Passes through the training examples.")
    parser.add_argument("--batch-size", type=int, default=4, help="Examples per weight update.")
    parser.add_argument("--learning-rate", type=float, default=0.0005, help="Size of learning steps.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Check gradients without weight updates or saves.")
    mode.add_argument("--check-data", action="store_true", help="Check tokenized training/practice data without model weights or writes.")
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or args.checkpoint_every < 1:
        parser.error("epochs, batch-size and checkpoint-every must be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("learning-rate must be finite and positive")
    return args


def file_hash(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def batches(examples, batch_size, pad_id, shuffle=False):
    """Keep examples independent while padding each batch to equal lengths."""
    order = torch.randperm(len(examples)).tolist() if shuffle else list(range(len(examples)))
    for start in range(0, len(order), batch_size):
        selected = [examples[index] for index in order[start:start + batch_size]]
        yield make_batch(selected, pad_id)


def checked_loss(model, batch):
    loss = model(**batch, use_cache=False).loss
    if not torch.isfinite(loss).item():
        raise RuntimeError("Training loss is not finite; stopping.")
    return loss


def measure_training_loss(model, examples, batch_size, pad_id):
    """Measure answer-token loss on training data, not on test questions."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.inference_mode():
        for batch in batches(examples, batch_size, pad_id):
            # The model shifts labels internally: position 0 is not a target.
            tokens = (batch["labels"][:, 1:] != IGNORE_LABEL).sum().item()
            total_loss += checked_loss(model, batch).item() * tokens
            total_tokens += tokens
    return total_loss / total_tokens


def verify_base_weights(parameters, original_hashes):
    """Check every original tensor, not just a few sample values."""
    for name, parameter in parameters.items():
        if parameter.requires_grad or parameter.grad is not None:
            raise RuntimeError(f"Original parameter was not kept frozen: {name}")
        if weight_hash(parameter) != original_hashes[name]:
            raise RuntimeError(f"Original parameter changed: {name}")


def check_one_batch(model, examples, args, pad_id, base_parameters, base_hashes, adapter):
    """Exercise forward/backward calculations without taking an optimizer step."""
    before = {name: weight_hash(parameter) for name, parameter in adapter.items()}
    model.train()
    loss = checked_loss(model, next(batches(examples, args.batch_size, pad_id)))
    loss.backward()
    gradients = [p.grad for p in adapter.values() if p.grad is not None]
    if not gradients or not all(torch.isfinite(g).all().item() for g in gradients):
        raise RuntimeError("Adapter gradients are missing or not finite.")
    if not any(torch.count_nonzero(g).item() for g in gradients):
        raise RuntimeError("All adapter gradients are zero.")
    verify_base_weights(base_parameters, base_hashes)
    if any(weight_hash(p) != before[name] for name, p in adapter.items()):
        raise RuntimeError("A weight changed during the dry run.")
    model.zero_grad(set_to_none=True)
    print(f"Dry-run answer loss: {loss.item():.4f}")
    print("PASS: adapter gradients are finite and nonzero; base weights have no gradients.")
    print("PASS: all base and adapter weights are unchanged. No files saved.")


def train_epochs(model, examples, args, pad_id, adapter, history_path, on_checkpoint=None):
    """Repeat the examples, updating only the explicitly selected LoRA weights."""
    optimizer = torch.optim.AdamW(adapter.values(), lr=args.learning_rate, weight_decay=0.0)
    step = 0
    with history_path.open("x") as history:
        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss = 0.0
            total_tokens = 0
            for batch in batches(examples, args.batch_size, pad_id, shuffle=True):
                optimizer.zero_grad(set_to_none=True)
                loss = checked_loss(model, batch)
                loss.backward()  # Calculate how the adapter should change.
                torch.nn.utils.clip_grad_norm_(
                    adapter.values(), max_norm=1.0, error_if_nonfinite=True,
                )
                optimizer.step()  # This is the actual weight update.
                tokens = (batch["labels"][:, 1:] != IGNORE_LABEL).sum().item()
                total_loss += loss.item() * tokens
                total_tokens += tokens
                step += 1

            epoch_loss = total_loss / total_tokens
            history.write(json.dumps({"epoch": epoch, "step": step, "training_loss": epoch_loss}) + "\n")
            history.flush()
            print(f"Epoch {epoch:2}/{args.epochs} | step {step:3} | answer loss: {epoch_loss:.4f}", flush=True)
            if on_checkpoint and (epoch % args.checkpoint_every == 0 or epoch == args.epochs):
                on_checkpoint(epoch, step)
    return step


def train_and_save(model, examples, args, pad_id, base_parameters, base_hashes, adapter,
                   tokenizer, practice, training_path):
    """Save a separate adapter only after the unchanged-base checks pass."""
    run_id = datetime.now(timezone.utc).strftime(f"iteration-{args.iteration}-ramayana-lora-%Y%m%dT%H%M%S-%fZ")
    log_dir = iteration_results(args.iteration) / "training" / run_id
    adapter_dir = adapter_directory(args.iteration, run_id)
    if adapter_dir.exists():
        raise FileExistsError(adapter_dir)
    log_dir.mkdir(parents=True, exist_ok=False)
    metadata_path = log_dir / "metadata.json"
    initial_adapter = {name: weight_hash(p) for name, p in adapter.items()}
    sample_before = {name: p.detach().flatten()[:5].tolist()
                     for name, p in adapter.items() if ".layers.0." in name}
    base_file_hash = file_hash(MODEL_PATH / "model.safetensors")
    config = model.peft_config["default"]
    metadata = {
        "status": "preparing", "run_id": run_id, "iteration": args.iteration,
        "model_path": str(MODEL_PATH), "base_weights_sha256": base_file_hash,
        "training_file": str(training_path), "training_sha256": file_hash(training_path),
        "training_examples": len(examples), "epochs": args.epochs,
        "batch_size": args.batch_size, "learning_rate": args.learning_rate,
        "optimizer": "AdamW", "weight_decay": 0.0, "max_grad_norm": 1.0,
        "seed": 0, "device": "cpu", "dtype": "float32", "cpu_threads": 4,
        "lora": {"rank": config.r, "alpha": config.lora_alpha,
                 "dropout": config.lora_dropout, "targets": sorted(config.target_modules)},
        "trainable_parameters": sum(p.numel() for p in adapter.values()),
        "frozen_parameters": sum(p.numel() for p in base_parameters.values()),
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                     "peft": peft.__version__},
        "test_questions_used": False, "adapter_path": str(adapter_dir),
        "final_test_used": False, "practice_used": practice is not None,
        "practice_source_sha256": practice["source_sha256"] if practice else {},
        "checkpoint_every": args.checkpoint_every if practice else None,
        "selection_rule": "lowest practice answer-token loss; ties keep earlier checkpoint" if practice else "last epoch",
        "selection_is_accuracy_score": False,
        "script_source_sha256": {name: file_hash(Path(__file__).parent / name) for name in
                                 ["train_lora.py", "practice_checks.py", "prepare_training_data.py", "inspect_lora.py"]},
    }
    write_json(metadata_path, metadata)
    print(f"Training logs: {log_dir}", flush=True)

    best = {}

    def checkpoint(epoch, step):
        verify_base_weights(base_parameters, base_hashes)
        if not all(torch.isfinite(p).all().item() for p in adapter.values()):
            raise RuntimeError("Nonfinite adapter weights at checkpoint.")
        loss = measure_training_loss(model, practice["prepared"], args.batch_size, pad_id)
        name = f"epoch_{epoch:03}"
        record = save_practice_checkpoint(
            model, tokenizer, practice, adapter_dir / "checkpoints" / name,
            log_dir / "practice" / name, epoch, step, loss,
        )
        with (log_dir / "practice_metrics.jsonl").open("a") as history:
            history.write(json.dumps(record) + "\n")
        if not best or loss < best["loss"]:
            best.update(loss=loss, epoch=epoch, path=record["checkpoint_path"],
                        weights={name: p.detach().clone() for name, p in adapter.items()})
        metadata.update(last_checkpoint_epoch=epoch, nominated_epoch=best["epoch"],
                        nominated_practice_loss=best["loss"])
        write_json(metadata_path, metadata)
        print(f"Practice answer loss: {loss:.4f}; nominated epoch: {best['epoch']} (not an accuracy score).", flush=True)

    try:
        initial_loss = measure_training_loss(model, examples, args.batch_size, pad_id)
        print(f"Training-data loss before updates: {initial_loss:.4f}", flush=True)
        metadata.update(status="training", initial_training_loss=initial_loss)
        if practice:
            metadata["initial_practice_loss"] = measure_training_loss(
                model, practice["prepared"], args.batch_size, pad_id)
        write_json(metadata_path, metadata)
        steps = train_epochs(model, examples, args, pad_id, adapter, log_dir / "loss.jsonl",
                             checkpoint if practice else None)

        if practice:
            metadata["last_epoch_training_loss"] = measure_training_loss(model, examples, args.batch_size, pad_id)
            with torch.no_grad():
                for name, parameter in adapter.items():
                    parameter.copy_(best["weights"][name])
            metadata.update(selected_epoch=best["epoch"], selected_checkpoint=best["path"],
                            selected_practice_loss=best["loss"],
                            selected_improves_initial_practice_loss=best["loss"] < metadata["initial_practice_loss"],
                            human_review_pending=True)

        final_loss = measure_training_loss(model, examples, args.batch_size, pad_id)
        verify_base_weights(base_parameters, base_hashes)
        if file_hash(MODEL_PATH / "model.safetensors") != base_file_hash:
            raise RuntimeError("The original model file changed.")
        source_hashes = {str(training_path): metadata["training_sha256"],
                         **metadata["practice_source_sha256"]}
        if any(file_hash(Path(path)) != digest for path, digest in source_hashes.items()):
            raise RuntimeError("A training or practice data file changed during the run.")
        if not all(torch.isfinite(p).all().item() for p in adapter.values()):
            raise RuntimeError("An adapter weight is not finite.")
        changed = [name for name, p in adapter.items() if weight_hash(p) != initial_adapter[name]]
        if not changed:
            raise RuntimeError("No adapter tensors changed during training.")

        verification = {
            "all_original_tensors_unchanged": True, "base_file_unchanged": True,
            "original_tensors_checked": len(base_parameters),
            "adapter_tensors_changed": len(changed), "adapter_tensors_total": len(adapter),
            "changed_adapter_names": changed,
            "adapter_samples_before": sample_before,
            "adapter_samples_after": {name: adapter[name].detach().flatten()[:5].tolist()
                                      for name in sample_before},
        }
        write_json(log_dir / "weight_verification.json", verification)
        adapter_dir.mkdir(parents=True, exist_ok=practice is not None)
        if (adapter_dir / "adapter_model.safetensors").exists():
            raise FileExistsError("Refusing to overwrite saved run weights.")
        model.save_pretrained(adapter_dir, safe_serialization=True, save_embedding_layers=False)
        metadata.update(status="complete", steps=steps, final_training_loss=final_loss,
                        completed_at_utc=datetime.now(timezone.utc).isoformat())
        write_json(metadata_path, metadata)

        print(f"\nTraining-data loss: {initial_loss:.4f} -> {final_loss:.4f}")
        print("PASS: every original weight and the original model file stayed unchanged.")
        print(f"PASS: {len(changed)}/{len(adapter)} adapter tensors changed.")
        print(f"Saved adapter: {adapter_dir}")
        if practice:
            print(f"Nominated checkpoint: epoch {best['epoch']} by practice loss. Review practice answers before final testing.")
        print("Final-test questions have not been read or run. The Streamlit app still uses the base model.")
    except (Exception, KeyboardInterrupt) as error:
        metadata.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                        error=str(error), ended_at_utc=datetime.now(timezone.utc).isoformat())
        write_json(metadata_path, metadata)
        raise


def main():
    args = parse_arguments()
    training_path = iteration_data(args.iteration) / "training" / "examples.jsonl"
    rows = load_examples(training_path)
    if args.check_data:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
        config = AutoConfig.from_pretrained(MODEL_PATH, local_files_only=True)
        prepared = [prepare_example(tokenizer, row["messages"], config.max_position_embeddings) for row in rows]
        practice = prepare_practice(tokenizer, config.max_position_embeddings, rows, args.iteration) if args.iteration >= 2 else None
        print(f"Iteration {args.iteration}: {len(prepared)} training examples; {len(practice['prepared']) if practice else 0} practice examples.")
        print(f"Training lengths: {min(len(r['input_ids']) for r in prepared)}–{max(len(r['input_ids']) for r in prepared)} tokens.")
        print("PASS: answer boundaries and context limits. No model weights loaded, files saved, or final tests read.")
        return

    # 1. Load the original model and record every original weight.
    model, tokenizer = load_model_and_tokenizer()
    base_parameters, base_hashes = record_original_weights(model)

    # 2. Attach exactly the LoRA configuration we inspected earlier.
    model = attach_lora(model)
    adapter = {name: p for name, p in model.named_parameters() if p.requires_grad}
    if not adapter or not all("lora_" in name for name in adapter):
        raise RuntimeError("Only LoRA parameters should be trainable.")
    verify_base_weights(base_parameters, base_hashes)

    # 3. Prepare only the training file using the answer-only labels we inspected.
    examples = [prepare_example(tokenizer, row["messages"], model.config.max_position_embeddings)
                for row in rows]
    practice = prepare_practice(tokenizer, model.config.max_position_embeddings, rows, args.iteration) if args.iteration >= 2 else None
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    updates = args.epochs * math.ceil(len(examples) / args.batch_size)
    print(f"\nIteration: {args.iteration} | examples: {len(examples)} | epochs: {args.epochs} | batch size: {args.batch_size}")
    if practice:
        print(f"Practice: {len(practice['questions'])} questions every {args.checkpoint_every} epochs and at the final epoch; final test stays unused.")
    print(f"Planned updates: {updates} | learning rate: {args.learning_rate}")
    print(f"Frozen base parameters: {sum(p.numel() for p in base_parameters.values()):,}")
    print(f"Trainable LoRA parameters: {sum(p.numel() for p in adapter.values()):,}")

    # 4. Either check without updates, or train, verify, and save the adapter.
    if args.dry_run:
        check_one_batch(model, examples, args, pad_id, base_parameters, base_hashes, adapter)
    else:
        train_and_save(model, examples, args, pad_id, base_parameters, base_hashes, adapter,
                       tokenizer, practice, training_path)


if __name__ == "__main__":
    main()
