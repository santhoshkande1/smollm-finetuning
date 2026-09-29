"""Development checks for the selected iteration. Never reads final-test files."""

import json
import re

import torch
from transformers import GenerationConfig

from experiment_paths import iteration_data
from prepare_training_data import prepare_example
from run_baseline import INSTRUCTION, sha256


def prepare_practice(tokenizer, context_limit, training_rows, iteration=2):
    """Prepare answer-only loss targets separately from generation prompts."""
    folder = iteration_data(iteration) / "evaluation"
    paths = [folder / "practice_questions.jsonl", folder / "practice_answers.jsonl"]
    questions, answers = [
        [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        for path in paths
    ]
    question_ids = [row["id"] for row in questions]
    answer_ids = [row["id"] for row in answers]
    if (not questions or len(set(question_ids)) != len(question_ids)
            or len(set(answer_ids)) != len(answer_ids) or set(question_ids) != set(answer_ids)):
        raise ValueError("Practice questions and answers need unique, matching IDs.")
    normalize = lambda text: re.sub(r"[^a-z0-9]", "", text.lower())
    training_questions = {normalize(row["messages"][1]["content"]) for row in training_rows}
    if len({normalize(q["question"]) for q in questions}) != len(questions):
        raise ValueError("Duplicate practice questions.")
    answer_map = {row["id"]: row["expected_answer"] for row in answers}
    prepared, prompts = [], []
    for question in questions:
        if set(question) != {"id", "category", "question"}:
            raise ValueError("Question records must not contain answers or extra prompt fields.")
        if normalize(question["question"]) in training_questions:
            raise ValueError("A practice question is an exact copy of a training question.")
        answer = answer_map[question["id"]]
        if not question["question"].strip() or not isinstance(answer, str) or not answer.strip():
            raise ValueError("Practice text must be nonempty.")
        messages = [{"role": "system", "content": INSTRUCTION},
                    {"role": "user", "content": question["question"]}]
        prepared.append(prepare_example(tokenizer, messages + [
            {"role": "assistant", "content": answer}], context_limit))
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        if inputs["input_ids"].shape[1] + 96 > context_limit:
            raise ValueError("Practice generation would exceed the context limit.")
        prompts.append((messages, prompt, inputs))
    return {"questions": questions, "answers": answer_map, "prepared": prepared,
            "prompts": prompts, "source_sha256": {str(p): sha256(p) for p in paths}}


def save_practice_checkpoint(model, tokenizer, practice, checkpoint_dir, report_dir,
                             epoch, step, loss):
    """Save adapter-only snapshots and ungraded answers with no gradients."""
    model.eval()
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    report_dir.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(checkpoint_dir, safe_serialization=True, save_embedding_layers=False)
    settings = GenerationConfig(
        max_new_tokens=96, do_sample=False, num_beams=1, use_cache=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
    )
    with (report_dir / "responses.jsonl").open("x") as raw, \
         (report_dir / "responses.md").open("x") as report:
        report.write(f"# Practice answers — epoch {epoch}\n\n")
        report.write("Raw model answers, not correctness scores. Expected answers were not in generation prompts.\n")
        # Preserve the training random stream even if generation internals use randomness.
        with torch.random.fork_rng(devices=[]), torch.inference_mode():
            for question, (messages, prompt, inputs) in zip(practice["questions"], practice["prompts"], strict=True):
                output = model.generate(**inputs, generation_config=settings)
                ids = output[0, inputs["input_ids"].shape[1]:].tolist()
                response = tokenizer.decode(ids, skip_special_tokens=True)
                ended = bool(ids and ids[-1] == tokenizer.eos_token_id)
                truncated = len(ids) >= settings.max_new_tokens and not ended
                row = {**question, "messages": messages, "rendered_prompt": prompt,
                       "response": response, "generated_token_ids": ids,
                       "ended_with_eos": ended, "hit_token_limit": truncated}
                raw.write(json.dumps(row, ensure_ascii=False) + "\n")
                raw.flush()
                report.write(f"\n## {question['id']}: {question['question']}\n\n")
                report.write(f"Expected: {practice['answers'][question['id']]}\n\n")
                report.write(f"Model:\n\n```text\n{response}\n```\n")
                if truncated:
                    report.write("\nAnswer reached the 96-token limit.\n")
                print(f"  Practice {question['id']}: {response}", flush=True)
    record = {"status": "complete", "epoch": epoch, "step": step,
              "practice_answer_loss": loss, "checkpoint_path": str(checkpoint_dir),
              "adapter_sha256": sha256(checkpoint_dir / "adapter_model.safetensors"),
              "question_count": len(practice["questions"]),
              "source_sha256": practice["source_sha256"],
              "generation_config": settings.to_dict(),
              "answer_keys_in_generation_prompts": False,
              "answer_keys_used_for_no_gradient_loss": True,
              "final_test_used": False, "optimizer_state_saved": False}
    (report_dir / "metadata.json").write_text(json.dumps(record, indent=2) + "\n")
    return record
