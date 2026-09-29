"""Prepare answer-only training inputs and inspect one example.

Only the tokenizer and configuration are loaded, not the model weights.
Prepared examples stay in memory; no training or file writes happen here.
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer
from experiment_paths import iteration_data


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "base" / "SmolLM2-135M-Instruct"
TRAINING_PATH = ROOT / "data" / "iteration_1" / "training" / "examples.jsonl"
IGNORE_LABEL = -100  # Transformers ignores these positions when calculating loss.


def load_examples(path=TRAINING_PATH):
    """Read only the chosen training file, never the evaluation directory."""
    examples = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        messages = row["messages"]
        if [message["role"] for message in messages] != ["system", "user", "assistant"]:
            raise ValueError(f"Line {line_number}: expected system, user, assistant messages.")
        if not all(isinstance(m["content"], str) and m["content"].strip() for m in messages):
            raise ValueError(f"Line {line_number}: messages must contain nonempty text.")
        examples.append(row)
    if not examples:
        raise ValueError("The training file is empty.")
    return examples


def prepare_example(tokenizer, messages, context_limit):
    """Use the real chat template, then mark only the answer and its end token."""
    # This is everything the model sees before it starts answering.
    prompt_text = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True,
    )
    # The full conversation also contains the correct answer.
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False,
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    input_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

    # Check the boundary rather than assuming separate tokenizations line up.
    answer_start = len(prompt_ids)
    if input_ids[:answer_start] != prompt_ids:
        raise ValueError("Tokenization changed at the answer boundary; inspect the template.")
    if len(input_ids) > context_limit:
        raise ValueError("Example exceeds the context limit; it has not been truncated.")
    if tokenizer.eos_token_id is None:
        raise ValueError("The tokenizer needs an end-of-answer token.")

    # Include the end token so the model also learns when to stop answering.
    answer_ids = input_ids[answer_start:]
    if tokenizer.eos_token_id not in answer_ids:
        raise ValueError("The assistant answer has no end token.")
    answer_end = answer_start + answer_ids.index(tokenizer.eos_token_id) + 1
    if answer_end <= answer_start + 1:
        raise ValueError("The answer contains no content tokens.")

    labels = [IGNORE_LABEL] * len(input_ids)
    labels[answer_start:answer_end] = input_ids[answer_start:answer_end]
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


def make_batch(examples, pad_token_id):
    """Pad sequences equally; padding must never become an answer target."""
    longest = max(len(example["input_ids"]) for example in examples)
    batch = {"input_ids": [], "attention_mask": [], "labels": []}
    for example in examples:
        padding = longest - len(example["input_ids"])
        batch["input_ids"].append(example["input_ids"] + [pad_token_id] * padding)
        batch["attention_mask"].append(example["attention_mask"] + [0] * padding)
        batch["labels"].append(example["labels"] + [IGNORE_LABEL] * padding)
    return {name: torch.tensor(values, dtype=torch.long) for name, values in batch.items()}


def show_example(tokenizer, source, prepared):
    """Show the question as context and the answer tokens as learning targets."""
    messages = source["messages"]
    print("\nFirst example")
    print(f"Instruction (context only): {messages[0]['content']}")
    print(f"Question (context only): {messages[1]['content']}")
    print(f"Correct answer: {messages[2]['content']}")
    print("\nAnswer tokens that will contribute to the training loss:")
    print(f"{'Position':>8} {'Token ID':>9}  Token text")
    for position, label in enumerate(prepared["labels"]):
        if label != IGNORE_LABEL:
            piece = tokenizer.decode([label], skip_special_tokens=False)
            print(f"{position:>8} {label:>9}  {piece!r}")
    print("\nInstruction, question, and formatting outside the answer have label -100.")
    print("They provide context but are not scored as answer targets.")
    print("Labels align with input tokens; the causal model shifts them internally to predict the next token.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iteration", type=int, choices=(1, 2, 3), default=1)
    args = parser.parse_args()
    # 1. Read the training examples and load the local tokenizer.
    examples = load_examples(iteration_data(args.iteration) / "training" / "examples.jsonl")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    config = AutoConfig.from_pretrained(MODEL_PATH, local_files_only=True)

    # 2. Tokenize the conversations and select the answer tokens for learning.
    prepared = [
        prepare_example(tokenizer, example["messages"], config.max_position_embeddings)
        for example in examples
    ]

    # 3. Check padding on a demonstration batch. This does not choose a training batch size.
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    batch = make_batch(prepared, pad_id)
    assert batch["input_ids"].shape == batch["labels"].shape == batch["attention_mask"].shape
    assert torch.all(batch["labels"][batch["attention_mask"] == 0] == IGNORE_LABEL)
    assert torch.all((batch["labels"] != IGNORE_LABEL).sum(dim=1) > 0)
    # PAD and EOS share an ID here: mask by position, not by token ID.
    for example in prepared:
        target_ids = [label for label in example["labels"] if label != IGNORE_LABEL]
        assert target_ids[-1] == tokenizer.eos_token_id

    # 4. Show exactly what one example contributes to training.
    lengths = [len(example["input_ids"]) for example in prepared]
    print(f"Prepared {len(prepared)} training examples in memory.")
    print(f"Sequence lengths: {min(lengths)}–{max(lengths)} tokens.")
    show_example(tokenizer, examples[0], prepared[0])
    print("\nPASS: answer targets, end tokens, and padding checks.")
    print("No model weights loaded or changed. No training or prepared files saved.")


if __name__ == "__main__":
    main()
