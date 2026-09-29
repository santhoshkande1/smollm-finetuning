"""Load the local SmolLM2 model and print a few of its learned weights."""

from pathlib import Path

from transformers import AutoModelForCausalLM


def main():
    model_path = Path(__file__).resolve().parents[1] / "models" / "base" / "SmolLM2-135M-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
    )
    parameters = dict(model.named_parameters())

    names = [
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
    ]

    for name in names:
        values = parameters[name].detach().flatten()[:10].tolist()
        print(f"\n{name}")
        print(values)


if __name__ == "__main__":
    main()
