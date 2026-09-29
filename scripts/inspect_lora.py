"""Inspect an untrained LoRA adapter.

Read main() at the bottom for the experiment's steps.
This script runs in memory: it does not train or save model weights.
"""

import hashlib
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "models" / "base" / "SmolLM2-135M-Instruct"
)
CHECK_QUESTION = "Who is Rama's father?"


def load_model_and_tokenizer():
    """Load the original model on CPU with consistent settings."""
    torch.set_num_threads(4)
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, local_files_only=True, dtype=torch.float32,
    ).to("cpu").eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    return model, tokenizer


def weight_hash(parameter):
    """Fingerprint a CPU float32 tensor to detect changes to its values."""
    values = parameter.detach().contiguous().numpy().tobytes()
    return hashlib.sha256(values).hexdigest()


def record_original_weights(model):
    """Keep original parameter references and fingerprints taken before LoRA."""
    parameters = dict(model.named_parameters())
    fingerprints = {
        name: weight_hash(parameter)
        for name, parameter in parameters.items()
    }
    return parameters, fingerprints


def prepare_check_inputs(tokenizer):
    """Convert one check question into model inputs; this is not training data."""
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": CHECK_QUESTION}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return tokenizer(prompt, return_tensors="pt", add_special_tokens=False)


def get_next_token_scores(model, inputs):
    """Read prediction scores without calculating training gradients."""
    with torch.inference_mode():
        return model(**inputs, use_cache=False).logits[:, -1, :].clone()


def attach_lora(model):
    """Choose where LoRA is added and attach its untrained weight tables."""
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "v_proj"],  # Attach to these attention tables.
        r=8,                                # Width of the small adapter tables.
        lora_alpha=16,                      # Scales the adapter's contribution.
        lora_dropout=0.0,
        bias="none",                        # Do not train original bias weights.
        init_lora_weights=True,             # Standard initialization: B is zero.
    )
    model = get_peft_model(model, config)
    model.eval()
    return model


def verify_attachment(model, original_parameters, original_hashes, inputs, before_scores):
    """Check that only LoRA can learn and attachment changed no base weights."""
    trainable_names = [
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    assert trainable_names, "No trainable adapter parameters found."
    assert all("lora_" in name for name in trainable_names), (
        "An unexpected non-LoRA parameter is trainable."
    )

    for name, parameter in original_parameters.items():
        assert not parameter.requires_grad, f"Original weight is trainable: {name}"
        assert weight_hash(parameter) == original_hashes[name], (
            f"Original weight changed: {name}"
        )

    original_count = sum(p.numel() for p in original_parameters.values())
    frozen_count = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    assert frozen_count == original_count, "Frozen parameter count changed."

    after_scores = get_next_token_scores(model, inputs)
    torch.testing.assert_close(before_scores, after_scores, rtol=0, atol=0)


def print_adapter_report(model):
    """Show parameter counts and sample adapter tensors after checks pass."""
    trainable = [
        (name, parameter) for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    trainable_count = sum(parameter.numel() for _, parameter in trainable)
    frozen_count = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total_count = frozen_count + trainable_count
    trainable_percent = 100 * trainable_count / total_count
    config = model.peft_config["default"]

    print("\nUntrained LoRA adapter attached in memory.")
    print(
        f"Targets: {', '.join(sorted(config.target_modules))} "
        f"in all {model.config.num_hidden_layers} layers; rank: {config.r}"
    )
    print(f"Original parameters (frozen): {frozen_count:,}")
    print(f"Added adapter parameters (trainable): {trainable_count:,}")
    print(f"Combined parameters: {total_count:,}")
    print(f"Trainable share of combined model: {trainable_percent:.2f}%")

    print("\nAdapter tensors in the first transformer layer:")
    for name, parameter in trainable:
        if ".layers.0." in name:
            values = parameter.detach().flatten()[:5].tolist()
            print(f"  {name}: {tuple(parameter.shape)}")
            print(f"    First 5 values: {values}")

    print("\nPASS: every original parameter is frozen and unchanged.")
    print("PASS: next-token scores for the check prompt are identical before and after attachment.")
    print("No training performed. No model or adapter files saved.")


def main():
    # 1. Load the original model.
    model, tokenizer = load_model_and_tokenizer()

    # 2. Record original weights and prediction scores for comparison.
    original_parameters, original_hashes = record_original_weights(model)
    inputs = prepare_check_inputs(tokenizer)
    before_scores = get_next_token_scores(model, inputs)

    # 3. Attach the untrained adapter.
    model = attach_lora(model)

    # 4. Verify that attachment preserves the original weights and check scores.
    verify_attachment(model, original_parameters, original_hashes, inputs, before_scores)

    # 5. Show which weights are frozen and which are trainable.
    print_adapter_report(model)


if __name__ == "__main__":
    main()
