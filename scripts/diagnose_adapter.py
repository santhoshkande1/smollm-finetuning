"""Generate answers to exact training prompts from saved checkpoints.

This measures reproduction of training examples, not generalization.
Never reads practice or final-test questions and never updates weights.
"""

import argparse
from datetime import datetime, timezone
import gc
import json

import torch
from peft import PeftModel
from transformers import GenerationConfig

from experiment_paths import iteration_data, iteration_results, select_completed_adapter
from inspect_lora import load_model_and_tokenizer
from prepare_training_data import load_examples
from run_baseline import MODEL_PATH, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iteration", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--run")
    parser.add_argument("--epochs", type=int, nargs="+", default=[10, 30])
    args = parser.parse_args()
    if any(epoch < 1 for epoch in args.epochs) or len(set(args.epochs)) != len(args.epochs):
        parser.error("epochs must be unique positive integers")
    adapter_dir, _, record = select_completed_adapter(args.iteration, args.run)
    checkpoints = [adapter_dir / "checkpoints" / f"epoch_{epoch:03}" for epoch in args.epochs]
    for folder in checkpoints:
        if not (folder / "adapter_model.safetensors").is_file():
            parser.error(f"Missing checkpoint: {folder}")
    path = iteration_data(args.iteration) / "training" / "examples.jsonl"
    if sha256(path) != record["training_sha256"]:
        raise ValueError("Training data differs from this run's recorded data.")
    if sha256(MODEL_PATH / "model.safetensors") != record["base_weights_sha256"]:
        raise ValueError("Base model differs from this run's recorded model.")
    rows = load_examples(path)
    destination = iteration_results(args.iteration) / "diagnostics" / datetime.now(timezone.utc).strftime("training-recall-%Y%m%dT%H%M%S-%fZ")
    destination.mkdir(parents=True, exist_ok=False)
    metadata = {"status": "running", "adapter_run": adapter_dir.name,
                "source_training_file": str(path), "source_training_sha256": sha256(path),
                "epochs": args.epochs, "examples_per_epoch": len(rows),
                "final_test_used": False, "training_performed": False,
                "note": "Exact-match counts are wording-sensitive diagnostics, not semantic correctness scores.",
                "checkpoints": {}}
    metadata_path = destination / "metadata.json"

    def save_metadata():
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    save_metadata()
    print(f"Diagnostic results: {destination}", flush=True)
    try:
        for epoch, folder in zip(args.epochs, checkpoints, strict=True):
            weight_hash = sha256(folder / "adapter_model.safetensors")
            model, tokenizer = load_model_and_tokenizer()
            model = PeftModel.from_pretrained(model, folder, is_trainable=False, local_files_only=True).eval()
            settings = GenerationConfig(max_new_tokens=96, do_sample=False, num_beams=1, use_cache=True,
                                        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id)
            exact = 0
            with (destination / f"epoch_{epoch:03}.jsonl").open("x") as output:
                for number, row in enumerate(rows, 1):
                    messages = row["messages"][:-1]
                    expected = row["messages"][-1]["content"]
                    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
                    with torch.inference_mode():
                        generated = model.generate(**inputs, generation_config=settings)
                    ids = generated[0, inputs["input_ids"].shape[1]:].tolist()
                    answer = tokenizer.decode(ids, skip_special_tokens=True).strip()
                    match = answer == expected.strip()
                    exact += match
                    output.write(json.dumps({"id": f"T{number:02}", "question": messages[-1]["content"],
                                             "messages": messages, "expected_answer": expected,
                                             "response": answer, "generated_token_ids": ids,
                                             "exact_match": match,
                                             "hit_token_limit": len(ids) >= 96 and ids[-1] != tokenizer.eos_token_id}, ensure_ascii=False) + "\n")
                    output.flush()
                    print(f"Epoch {epoch} T{number:02}: {answer}", flush=True)
            if sha256(folder / "adapter_model.safetensors") != weight_hash:
                raise RuntimeError("Checkpoint file changed during inference.")
            metadata["checkpoints"][str(epoch)] = {"path": str(folder), "sha256": weight_hash,
                                                    "exact_matches": exact, "total": len(rows)}
            save_metadata()
            del model
            gc.collect()
        if sha256(MODEL_PATH / "model.safetensors") != record["base_weights_sha256"]:
            raise RuntimeError("Base model file changed during inference.")
        metadata["status"] = "complete"
        save_metadata()
        print(json.dumps(metadata["checkpoints"], indent=2))
    except (Exception, KeyboardInterrupt) as error:
        metadata.update(status="failed", error=str(error))
        save_metadata()
        raise


if __name__ == "__main__":
    main()
