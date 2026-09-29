"""Offline regression tests using tiny stand-ins, not the real model or final test.

Run: python -m unittest discover -s scripts -p 'test_*.py'
Temporary test artifacts are automatically removed.
"""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import load_file, save_file

import experiment_paths
import compare_models
import practice_checks
import train_lora
from build_training_variations import build_examples


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = torch.nn.Parameter(torch.tensor([2.0]), requires_grad=False)
        self.lora_weight = torch.nn.Parameter(torch.tensor([0.0]))
        self.peft_config = {"default": SimpleNamespace(
            r=1, lora_alpha=1, lora_dropout=0.0, target_modules={"q_proj", "v_proj"})}
        self.seen_inputs = []

    def forward(self, **kwargs):
        return SimpleNamespace(loss=(self.lora_weight - 1).square().mean())

    def generate(self, input_ids, **kwargs):
        self.seen_inputs.append(input_ids.clone())
        return torch.cat([input_ids, torch.tensor([[7, 2]])], dim=1)

    def save_pretrained(self, folder, **kwargs):
        save_file({"lora_weight": self.lora_weight.detach().clone()}, str(folder / "adapter_model.safetensors"))
        (folder / "adapter_config.json").write_text('{}\n')


class WorkflowTests(unittest.TestCase):
    def test_iteration_three_dataset_is_reproducible(self):
        rows, counts = build_examples()
        self.assertEqual(build_examples(), (rows, counts))
        self.assertEqual(len(rows), 238)
        self.assertEqual(sum(counts.values()), len(rows))
        self.assertEqual(len({r['messages'][1]['content'] for r in rows}), len(rows))
        self.assertEqual(train_lora.parse_arguments(['--iteration', '3']).iteration, 3)

    def test_comparison_results_follow_adapter_iteration(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(experiment_paths, "ROOT", Path(tmp)):
            question = {"id": "B01", "set": "baseline", "question": "Fixture question?"}
            answers = ({"response": "base fixture", "hit_token_limit": False},
                       {"response": "adapter fixture", "hit_token_limit": False})
            for iteration in (1, 2):
                metadata = {"adapter_iteration": iteration, "adapter_run": "fixture",
                            "weight_file_sha256": {}}
                with patch.object(compare_models, "compare_question", return_value=answers), redirect_stdout(io.StringIO()):
                    compare_models.save_comparison(None, None, [question], None, metadata)
                parent = Path(tmp) / "results" / f"iteration_{iteration}"
                for kind in ("before_finetuning", "after_finetuning"):
                    run = next((parent / kind).iterdir())
                    self.assertTrue((run / "responses.jsonl").is_file())
                    self.assertEqual(json.loads((run / "metadata.json").read_text())["status"], "complete")
            self.assertFalse((Path(tmp) / "results" / "before_finetuning").exists())

    def test_practice_loader_keeps_keys_out_of_prompts(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(experiment_paths, "ROOT", Path(tmp)):
            folder = Path(tmp) / "data" / "iteration_2" / "evaluation"
            folder.mkdir(parents=True)
            questions = [{"id": "P01", "category": "direct_facts", "question": "Practice question?"}]
            answers = [{"id": "P01", "expected_answer": "SECRET answer"}]
            (folder / "practice_questions.jsonl").write_text(json.dumps(questions[0]) + '\n')
            (folder / "practice_answers.jsonl").write_text(json.dumps(answers[0]) + '\n')
            # Final-test files deliberately do not exist in this fixture.
            class Tokenizer:
                def apply_chat_template(self, messages, **kwargs):
                    return json.dumps(messages)

                def __call__(self, prompt, **kwargs):
                    return {"input_ids": torch.tensor([[3, 4]])}

            rows = [{"messages": [{"role": "system", "content": "instruction"},
                                    {"role": "user", "content": "Training question?"},
                                    {"role": "assistant", "content": "Training answer"}]}]
            with patch.object(practice_checks, "prepare_example", return_value={}) as prepare:
                result = practice_checks.prepare_practice(Tokenizer(), 512, rows)
                self.assertNotIn("SECRET", result["prompts"][0][1])
                self.assertEqual(prepare.call_args.args[1][-1]["content"], "SECRET answer")
                (folder / "practice_answers.jsonl").write_text(json.dumps({"id": "P02", "expected_answer": "wrong ID"}) + '\n')
                with self.assertRaises(ValueError):
                    practice_checks.prepare_practice(Tokenizer(), 512, rows)

    def test_arguments(self):
        self.assertEqual(train_lora.parse_arguments([]).iteration, 1)
        args = train_lora.parse_arguments(["--iteration", "2", "--check-data"])
        self.assertTrue(args.check_data)
        for argv in (["--iteration", "4"], ["--checkpoint-every", "0"],
                     ["--learning-rate", "nan"], ["--dry-run", "--check-data"]):
            with redirect_stdout(io.StringIO()), patch('sys.stderr', io.StringIO()), self.assertRaises(SystemExit):
                train_lora.parse_arguments(argv)
        with patch('sys.stderr', io.StringIO()), self.assertRaises(SystemExit):
            compare_models.parse_arguments(['--question-set', 'final'])
        args = compare_models.parse_arguments(['--question-set', 'final', '--confirm-final-test', '--checkpoint', '8'])
        self.assertTrue(args.confirm_final_test)
        self.assertEqual(args.checkpoint, 8)

    def test_discovery(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(experiment_paths, "ROOT", Path(tmp)):
            for iteration, name, status in [(1, "old", "complete"), (2, "new", "complete"), (2, "bad", "failed")]:
                adapter = experiment_paths.adapter_directory(iteration, name)
                adapter.mkdir(parents=True)
                (adapter / "adapter_model.safetensors").write_bytes(b'test')
                log = Path(tmp) / "results" / f"iteration_{iteration}" / "training" / name
                log.mkdir(parents=True)
                record = {"status": status}
                if iteration == 2:
                    record["iteration"] = iteration
                (log / "metadata.json").write_text(json.dumps(record))
            self.assertEqual(experiment_paths.select_completed_adapter()[0].name, "old")
            self.assertEqual(experiment_paths.select_completed_adapter(2)[0].name, "new")
            with self.assertRaises(ValueError):
                experiment_paths.select_completed_adapter(2, "bad")
            with self.assertRaises(ValueError):
                experiment_paths.select_completed_adapter(2, "../old")

    def test_checkpoint_selection_and_no_answer_leak(self):
        """Full save flow: choose epoch 2 over epoch 3; root weights match epoch 2."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_dir = root / "base"
            base_dir.mkdir()
            (base_dir / "model.safetensors").write_bytes(b'original base sentinel')
            training_path = root / "training.jsonl"
            training_path.write_text('synthetic fixture\n')
            model = TinyModel()
            base = {"base": model.base}
            hashes = {name: train_lora.weight_hash(p) for name, p in base.items()}
            adapter = {"lora_weight": model.lora_weight}
            examples = [{"input_ids": [3, 7, 2], "attention_mask": [1, 1, 1], "labels": [-100, 7, 2]}]
            practice_prepared = [dict(examples[0])]
            prompt_ids = torch.tensor([[3, 4]])
            practice = {"prepared": practice_prepared, "questions": [{"id": "P01", "category": "direct_facts", "question": "fixture?"}],
                        "answers": {"P01": "SECRET expected answer"}, "source_sha256": {},
                        "prompts": [([{"role": "user", "content": "fixture?"}], "fixture prompt", {"input_ids": prompt_ids})]}
            tokenizer = SimpleNamespace(eos_token_id=2, pad_token_id=2,
                                        decode=lambda *args, **kwargs: "generated fixture")
            args = train_lora.parse_arguments(["--iteration", "2", "--epochs", "3", "--checkpoint-every", "2"])
            losses = iter([2.0, 0.5, 0.8])
            original_measure = train_lora.measure_training_loss

            def measure(m, items, batch_size, pad_id):
                if items is practice_prepared:
                    return next(losses)
                return original_measure(m, items, batch_size, pad_id)

            with patch.object(train_lora, "ROOT", root), patch.object(train_lora, "MODEL_PATH", base_dir), \
                 patch.object(experiment_paths, "ROOT", root), \
                 patch.object(train_lora, "measure_training_loss", side_effect=measure), \
                 redirect_stdout(io.StringIO()):
                train_lora.train_and_save(model, examples, args, 2, base, hashes, adapter,
                                          tokenizer, practice, training_path)
            log = next((root / "results" / "iteration_2" / "training").iterdir())
            metadata = json.loads((log / "metadata.json").read_text())
            self.assertEqual(metadata["status"], "complete")
            self.assertEqual(metadata["selected_epoch"], 2)
            self.assertFalse(metadata["final_test_used"])
            saved = Path(metadata["adapter_path"])
            chosen = load_file(str(saved / "checkpoints" / "epoch_002" / "adapter_model.safetensors"))
            final = load_file(str(saved / "adapter_model.safetensors"))
            torch.testing.assert_close(final["lora_weight"], chosen["lora_weight"], rtol=0, atol=0)
            last = load_file(str(saved / "checkpoints" / "epoch_003" / "adapter_model.safetensors"))
            self.assertFalse(torch.equal(final["lora_weight"], last["lora_weight"]))
            self.assertEqual(len(model.seen_inputs), 2)
            for inputs in model.seen_inputs:
                torch.testing.assert_close(inputs, prompt_ids)
            raw = (log / "practice" / "epoch_002" / "responses.jsonl").read_text()
            self.assertNotIn("SECRET", raw)
            self.assertIn("SECRET", (log / "practice" / "epoch_002" / "responses.md").read_text())
            self.assertEqual(model.base.item(), 2.0)
            self.assertEqual((base_dir / "model.safetensors").read_bytes(), b'original base sentinel')

    def test_iteration_one_save_remains_supported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_dir = root / "base"
            base_dir.mkdir()
            (base_dir / "model.safetensors").write_bytes(b'base')
            training_path = root / "training.jsonl"
            training_path.write_text('fixture\n')
            model = TinyModel()
            examples = [{"input_ids": [3, 7, 2], "attention_mask": [1, 1, 1], "labels": [-100, 7, 2]}]
            args = train_lora.parse_arguments(["--epochs", "1"])
            with patch.object(train_lora, "ROOT", root), patch.object(train_lora, "MODEL_PATH", base_dir), \
                 patch.object(experiment_paths, "ROOT", root), redirect_stdout(io.StringIO()):
                train_lora.train_and_save(model, examples, args, 2, {"base": model.base},
                    {"base": train_lora.weight_hash(model.base)}, {"lora_weight": model.lora_weight},
                    None, None, training_path)
                folder, _, metadata = experiment_paths.select_completed_adapter(1)
                self.assertEqual(folder.parent.name, "iteration_1")
                self.assertFalse(metadata["practice_used"])
                self.assertFalse((folder / "checkpoints").exists())


if __name__ == "__main__":
    unittest.main()
