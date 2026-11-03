import argparse

import torch

from config import ModelConfig
from model.gpt import GPTLanguageModel
from tokenizer.loader import load_tokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate text using a trained GPT model."
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the trained model checkpoint.",
    )

    parser.add_argument(
        "--prompt",
        default="Once upon a time",
        help="Starting text prompt.",
    )

    parser.add_argument(
        "--tokens",
        type=int,
        default=200,
        help="Number of new tokens to generate.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=40,
        help="Top-k sampling. Use 0 to disable.",
    )

    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling threshold.",
    )

    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.15,
        help="Repetition penalty. Use 1.0 to disable.",
    )

    parser.add_argument(
        "--no-repeat-ngram-size",
        type=int,
        default=3,
        help="Prevent repeated n-grams. Use 0 to disable.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible generation.",
    )

    parser.add_argument(
        "--tokenizer-dir",
        default=None,
        help="Optional tokenizer directory override.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.tokens < 0:
        raise ValueError("--tokens cannot be negative.")

    if args.temperature <= 0:
        raise ValueError("--temperature must be greater than zero.")

    if args.top_k < 0:
        raise ValueError("--top-k cannot be negative.")

    if not 0.0 < args.top_p <= 1.0:
        raise ValueError("--top-p must be in the range (0, 1].")

    if args.repetition_penalty < 1.0:
        raise ValueError("--repetition-penalty must be >= 1.0.")

    if args.no_repeat_ngram_size < 0:
        raise ValueError("--no-repeat-ngram-size cannot be negative.")

    # Set reproducible random seeds.
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Load checkpoint.
    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    model_config = ModelConfig.from_dict(checkpoint["model_config"])

    # Resolve tokenizer directory.
    tokenizer_dir = (
        args.tokenizer_dir
        or checkpoint.get("tokenizer_dir")
        or checkpoint.get("train_config", {}).get(
            "tokenizer_dir",
            "data/processed/tokenizer",
        )
    )

    tokenizer_type = checkpoint.get("tokenizer_type")

    tokenizer = load_tokenizer(
        tokenizer_dir,
        tokenizer_type=tokenizer_type,
    )

    # Validate tokenizer and model vocabulary.
    if tokenizer.vocab_size != model_config.vocab_size:
        raise ValueError(
            f"Vocabulary mismatch: "
            f"tokenizer={tokenizer.vocab_size}, "
            f"model={model_config.vocab_size}"
        )

    # Load model.
    model = GPTLanguageModel(model_config)

    model.load_state_dict(checkpoint["model"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    # Encode prompt.
    input_ids = tokenizer.encode(args.prompt)

    if not input_ids:
        input_ids = [0]

    if min(input_ids) < 0 or max(input_ids) >= model_config.vocab_size:
        raise ValueError("Prompt produced a token ID outside the model vocabulary.")

    input_tensor = torch.tensor(
        [input_ids],
        dtype=torch.long,
        device=device,
    )

    print("\nGeneration settings:")
    print(f"Device: {device}")
    print(f"Tokenizer: {tokenizer_type}")
    print(f"Temperature: {args.temperature}")
    print(f"Top-k: {args.top_k}")
    print(f"Top-p: {args.top_p}")
    print(f"Repetition penalty: {args.repetition_penalty}")
    print(f"No-repeat n-gram size: {args.no_repeat_ngram_size}")
    print(f"New tokens: {args.tokens}")
    print(f"Seed: {args.seed}")

    # Generate text.
    with torch.inference_mode():
        output = model.generate(
            input_tensor,
            max_new_tokens=args.tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
        )

    generated_text = tokenizer.decode(output[0].tolist())

    print("\nGenerated text:\n")
    print(generated_text)


if __name__ == "__main__":
    main()
