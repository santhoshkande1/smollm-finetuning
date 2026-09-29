"""Compare base and adapter answers on an explicitly selected question set.

Use --dry-run to check one baseline question without saving results.
Normal runs save fresh results and preserve all earlier experiments.
Defaults to the 30 iteration 1 development questions. Final testing requires
both --question-set final and --confirm-final-test, after checkpoint selection.
"""

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import json
from pathlib import Path
import re

import peft
from peft import PeftModel
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from run_baseline import INSTRUCTION, MODEL_PATH, ROOT, sha256
from experiment_paths import select_completed_adapter, iteration_results, iteration_data


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iteration", type=int, choices=(1, 2, 3), default=1,
                        help="Adapter iteration. Questions remain iteration 1 development checks.")
    parser.add_argument("--run", help="Adapter run folder; optional when only one completed run exists.")
    parser.add_argument("--checkpoint", type=int, help="Use a saved epoch checkpoint instead of the run-root loss nomination.")
    parser.add_argument("--question-set", choices=("development", "practice", "final"), default="development")
    parser.add_argument("--evaluation-iteration", type=int, choices=(2, 3), default=2,
                        help="Source of practice/final questions; iteration 3 reuses iteration 2's reserved final test.")
    parser.add_argument("--confirm-final-test", action="store_true",
                        help="Explicitly confirm checkpoint selection is finished before opening the final questions.")
    parser.add_argument("--dry-run", action="store_true", help="Generate one answer pair without saving files.")
    args = parser.parse_args(argv)
    if args.checkpoint is not None and args.checkpoint < 1:
        parser.error("--checkpoint must be positive")
    if args.question_set == "final" and not args.confirm_final_test:
        parser.error("Final testing requires --confirm-final-test after checkpoint selection")
    return args


def select_adapter(run_id, iteration=1):
    """Select a completed adapter, never an unfinished training run."""
    folder, _, metadata = select_completed_adapter(iteration, run_id)
    return folder, metadata


def load_questions(question_set="development", evaluation_iteration=2):
    """Read baseline question text and test questions, never their answer keys."""
    if question_set != "development":
        if question_set not in ("practice", "final"):
            raise ValueError("Unknown question set")
        stem = "practice" if question_set == "practice" else "final_test"
        path = iteration_data(evaluation_iteration) / "evaluation" / f"{stem}_questions.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if not rows or len({r["id"] for r in rows}) != len(rows):
            raise ValueError("Question IDs must be unique and nonempty")
        if any(set(r) != {"id", "category", "question"} for r in rows):
            raise ValueError("Question files must not include answer keys or extra prompt fields")
        return [{**r, "set": stem} for r in rows], {str(path.relative_to(ROOT)): sha256(path)}
    baseline_path = ROOT / "data" / "iteration_1" / "evaluation" / "baseline_questions.md"
    test_path = ROOT / "data" / "iteration_1" / "evaluation" / "test_questions.jsonl"
    section = baseline_path.read_text().split("## Questions\n", 1)[1].split("\n## ", 1)[0]
    baseline = re.findall(r"^(\d+)\. (.+)$", section, re.MULTILINE)
    if [int(number) for number, _ in baseline] != list(range(1, 16)):
        raise ValueError("Expected baseline questions 1 through 15.")
    questions = [
        {"id": f"B{int(number):02}", "set": "baseline", "question": question}
        for number, question in baseline
    ]
    tests = [json.loads(line) for line in test_path.read_text().splitlines() if line.strip()]
    if len(tests) != 15:
        raise ValueError("Expected 15 test questions.")
    questions.extend({"id": row["id"], "set": "test", "category": row["category"],
                      "question": row["question"]} for row in tests)
    if len({q["id"] for q in questions}) != len(questions):
        raise ValueError("Question IDs must be unique.")
    return questions, {str(path.relative_to(ROOT)): sha256(path)
                       for path in [baseline_path, test_path]}


def prepare_prompt(tokenizer, question):
    messages = [{"role": "system", "content": INSTRUCTION},
                {"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    return messages, prompt, inputs


def load_models(adapter_dir, check_question):
    """Load one base model and verify that disabling LoRA restores its scores."""
    torch.set_num_threads(4)
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, local_files_only=True, dtype=torch.float32,
    ).to("cpu").eval()
    _, _, inputs = prepare_prompt(tokenizer, check_question)
    with torch.inference_mode():
        original_scores = base(**inputs, use_cache=False).logits[:, -1, :].clone()
    model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=False, local_files_only=True)
    model.eval()
    if any(p.requires_grad for p in model.parameters()):
        raise RuntimeError("Comparison should have no trainable weights.")
    with model.disable_adapter(), torch.inference_mode():
        disabled_scores = model(**inputs, use_cache=False).logits[:, -1, :]
    torch.testing.assert_close(original_scores, disabled_scores, rtol=0, atol=0)
    print("PASS: disabling the adapter reproduces original-model scores on the check prompt.")
    return model, tokenizer


def generate_answer(model, tokenizer, inputs, settings, adapter_enabled):
    """Generate fresh tokens with no conversation history or reused generation state."""
    context = nullcontext() if adapter_enabled else model.disable_adapter()
    with context, torch.inference_mode():
        output = model.generate(**inputs, generation_config=settings)
    token_ids = output[0, inputs["input_ids"].shape[1]:].tolist()
    ended = bool(token_ids and token_ids[-1] == tokenizer.eos_token_id)
    return {"response": tokenizer.decode(token_ids, skip_special_tokens=True),
            "generated_token_ids": token_ids, "ended_with_eos": ended,
            "hit_token_limit": len(token_ids) >= settings.max_new_tokens and not ended}


def compare_question(model, tokenizer, question, settings):
    messages, prompt, inputs = prepare_prompt(tokenizer, question["question"])
    if inputs["input_ids"].shape[1] + settings.max_new_tokens > model.config.max_position_embeddings:
        raise ValueError(f"Question {question['id']} exceeds the context limit.")
    # Exactly the same prompt and settings go to both configurations.
    base = generate_answer(model, tokenizer, inputs, settings, adapter_enabled=False)
    adapted = generate_answer(model, tokenizer, inputs, settings, adapter_enabled=True)
    common = {**question, "messages": messages, "rendered_prompt": prompt}
    print(f"\n{question['id']}: {question['question']}", flush=True)
    print(f"Base: {base['response']}", flush=True)
    print(f"LoRA: {adapted['response']}", flush=True)
    for label, answer in [("Base", base), ("LoRA", adapted)]:
        if answer["hit_token_limit"]:
            print(f"[{label} answer reached the 96-token limit.]", flush=True)
    return {**common, **base}, {**common, **adapted}


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def save_comparison(model, tokenizer, questions, settings, metadata):
    """Keep base and adapter answers separate, plus one readable comparison."""
    run_id = datetime.now(timezone.utc).strftime("comparison-%Y%m%dT%H%M%S-%fZ")
    result_root = iteration_results(metadata["adapter_iteration"])
    base_dir = result_root / "before_finetuning" / run_id
    adapted_dir = result_root / "after_finetuning" / run_id
    base_dir.mkdir(parents=True, exist_ok=False)
    adapted_dir.mkdir(parents=True, exist_ok=False)
    metadata.update(comparison_id=run_id, status="running", base_results=str(base_dir),
                    adapter_results=str(adapted_dir), question_count=len(questions))

    def save_metadata():
        write_json(base_dir / "metadata.json", {**metadata, "adapter_enabled": False})
        write_json(adapted_dir / "metadata.json", {**metadata, "adapter_enabled": True})

    save_metadata()
    print(f"\nBase results: {base_dir}\nLoRA results: {adapted_dir}", flush=True)
    try:
        with (base_dir / "responses.jsonl").open("x") as base_file, \
             (adapted_dir / "responses.jsonl").open("x") as adapted_file, \
             (adapted_dir / "comparison.md").open("x") as report:
            report.write("# Original model vs trained LoRA\n\n")
            report.write(f"Adapter: `{metadata['adapter_run']}`\n\n")
            report.write("Same independent questions and generation settings. Raw answers are not graded.\n")
            report.write("The base-only answers were generated during this comparison; the original baseline files remain separate.\n")
            for question in questions:
                base, adapted = compare_question(model, tokenizer, question, settings)
                base_file.write(json.dumps(base, ensure_ascii=False) + "\n")
                adapted_file.write(json.dumps(adapted, ensure_ascii=False) + "\n")
                base_file.flush()
                adapted_file.flush()
                report.write(f"\n## {question['id']} ({question['set']}): {question['question']}\n")
                for label, answer in [("Original model", base), ("Base + LoRA", adapted)]:
                    report.write(f"\n**{label}**\n\n```text\n{answer['response']}\n```\n")
                    if answer["hit_token_limit"]:
                        report.write("\nAnswer reached the 96-token limit.\n")
                report.flush()

        # Inference should not write to either set of weights.
        for path, digest in metadata["weight_file_sha256"].items():
            if sha256(Path(path)) != digest:
                raise RuntimeError(f"A weight file changed: {path}")
        metadata.update(status="complete", completed_at_utc=datetime.now(timezone.utc).isoformat())
        save_metadata()
        print(f"\nComparison complete: {len(questions)} questions, {2 * len(questions)} answers.")
        print(f"Read the comparison: {adapted_dir / 'comparison.md'}")
        print("Base and adapter files stayed unchanged. No training or automatic grading performed.")
    except (Exception, KeyboardInterrupt) as error:
        metadata.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                        error=str(error))
        save_metadata()
        raise


def main():
    args = parse_arguments()
    adapter_dir, training_record = select_adapter(args.run, args.iteration)
    if args.checkpoint is not None:
        adapter_dir = adapter_dir / "checkpoints" / f"epoch_{args.checkpoint:03}"
        if not (adapter_dir / "adapter_model.safetensors").is_file():
            raise ValueError(f"Missing checkpoint: {adapter_dir}")
    print(f"Question set: {args.question_set}; adapter iteration {args.iteration}, checkpoint {args.checkpoint or 'run root'}.")
    base_file = MODEL_PATH / "model.safetensors"
    if sha256(base_file) != training_record["base_weights_sha256"]:
        raise ValueError("The original model differs from the training run's base model.")
    questions, question_hashes = load_questions(args.question_set, args.evaluation_iteration)
    model, tokenizer = load_models(adapter_dir, questions[0]["question"])
    settings = GenerationConfig(
        max_new_tokens=96, do_sample=False, num_beams=1, use_cache=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
    )
    print(f"Adapter: {adapter_dir.name}")
    if args.dry_run:
        compare_question(model, tokenizer, questions[0], settings)
        print("\nDry run complete: one baseline question checked. No result files saved or weights changed.")
        return

    metadata = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "adapter_run": training_record["run_id"], "model_path": str(MODEL_PATH),
        "adapter_iteration": args.iteration, "adapter_checkpoint": args.checkpoint,
        "question_set": args.question_set,
        "evaluation_iteration": 1 if args.question_set == "development" else args.evaluation_iteration,
        "final_test_confirmed": args.confirm_final_test,
        "adapter_path": str(adapter_dir), "system_instruction": INSTRUCTION,
        "chat_template": tokenizer.chat_template, "generation_config": settings.to_dict(),
        "device": "cpu", "dtype": "float32", "seed": 0, "cpu_threads": 4,
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                     "peft": peft.__version__},
        "question_file_sha256": question_hashes,
        "weight_file_sha256": {str(path): sha256(path) for path in
                               [base_file, adapter_dir / "adapter_model.safetensors"]},
        "config_file_sha256": {str(path): sha256(path) for path in
                               [MODEL_PATH / "config.json", MODEL_PATH / "tokenizer.json",
                                MODEL_PATH / "tokenizer_config.json", adapter_dir / "adapter_config.json"]},
        "answer_keys_provided_to_model": False,
        "note": "before_finetuning denotes base-only behavior; both sides are generated during this comparison.",
    }
    save_comparison(model, tokenizer, questions, settings, metadata)


if __name__ == "__main__":
    main()
