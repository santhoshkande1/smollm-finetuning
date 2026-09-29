"""Ask the local model the baseline questions and preserve its raw answers."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import re

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "base" / "SmolLM2-135M-Instruct"
RESULTS_DIR = ROOT / "results" / "iteration_1" / "before_finetuning"
INSTRUCTION = (
    "Answer the question about the Ramayana in one or two sentences. "
    'If you do not know, say "I don\'t know."'
)


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    question_file = ROOT / "data" / "iteration_1" / "evaluation" / "baseline_questions.md"
    question_section = question_file.read_text().split("## Questions\n", 1)[1]
    question_section = question_section.split("\n## ", 1)[0]
    questions = re.findall(r"^(\d+)\. (.+)$", question_section, flags=re.MULTILINE)
    if [int(number) for number, _ in questions] != list(range(1, 16)):
        raise ValueError("Expected baseline questions numbered 1 through 15.")

    # Preserve exactly which local checkpoint and tokenizer this run used.
    filenames = [
        "model.safetensors", "config.json", "tokenizer.json",
        "tokenizer_config.json", "special_tokens_map.json",
    ]
    hashes = {name: sha256(MODEL_PATH / name) for name in filenames}
    download_metadata = MODEL_PATH / ".cache/huggingface/download/model.safetensors.metadata"
    provenance = {}
    if download_metadata.exists():
        lines = download_metadata.read_text().splitlines()
        if len(lines) >= 2:
            provenance = {"recorded_revision": lines[0], "recorded_etag": lines[1]}
            if re.fullmatch(r"[0-9a-f]{64}", lines[1]):
                matches = hashes["model.safetensors"] == lines[1]
                provenance["weights_match_download_metadata"] = matches
                if not matches:
                    raise ValueError("Model weights differ from the local download metadata.")

    # CPU keeps this small experiment portable; eval() disables training behavior.
    torch.set_num_threads(4)
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, local_files_only=True, dtype=torch.float32,
    ).to("cpu").eval()
    generation = GenerationConfig(
        max_new_tokens=96,
        do_sample=False,
        num_beams=1,
        use_cache=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )

    started = datetime.now(timezone.utc)
    output_dir = RESULTS_DIR / started.strftime("baseline-%Y%m%dT%H%M%S-%fZ")
    output_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        "started_at_utc": started.isoformat(),
        "model_path": str(MODEL_PATH),
        "repository": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "file_sha256": hashes,
        "question_file_sha256": sha256(question_file),
        "local_download_metadata": provenance,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "device": "cpu", "dtype": "float32", "seed": 0, "cpu_threads": 4,
        "system_instruction": INSTRUCTION,
        "chat_template": tokenizer.chat_template,
        "generation_config": generation.to_dict(),
        "question_count": len(questions),
        "status": "running",
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saving answers to {output_dir}", flush=True)

    with (output_dir / "responses.jsonl").open("x") as raw, (
        output_dir / "responses.md"
    ).open("x") as readable:
        readable.write("# Base model answers — before LoRA\n\nRaw generated answers; not corrected or graded.\n")
        for number, question in questions:
            # Each question gets a fresh prompt. No answer key or prior answers.
            messages = [
                {"role": "system", "content": INSTRUCTION},
                {"role": "user", "content": question},
            ]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            with torch.inference_mode():
                output = model.generate(**inputs, generation_config=generation)
            generated_ids = output[0, inputs["input_ids"].shape[1]:].tolist()
            response = tokenizer.decode(generated_ids, skip_special_tokens=True)
            record = {
                "id": int(number), "question": question, "messages": messages,
                "rendered_prompt": prompt, "response": response,
                "generated_token_ids": generated_ids,
                "ended_with_eos": bool(generated_ids and generated_ids[-1] == tokenizer.eos_token_id),
            }
            raw.write(json.dumps(record, ensure_ascii=False) + "\n")
            raw.flush()
            readable.write(f"\n## {number}. {question}\n\n```text\n{response}\n```\n")
            readable.flush()
            print(f"[{number}/15] {question}\n{response}\n", flush=True)

    metadata["status"] = "complete"
    metadata["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print("Baseline complete. Model weights were not modified.", flush=True)


if __name__ == "__main__":
    main()
