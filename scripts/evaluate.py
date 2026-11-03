import argparse

import torch

from config import ModelConfig, TrainConfig
from model.gpt import GPTLanguageModel
from training.data import TokenDataset
from training.engine import choose_device, estimate_loss


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained GPT checkpoint.")

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the model checkpoint.",
    )

    parser.add_argument(
        "--batches",
        type=int,
        default=None,
        help="Number of fixed evaluation batches per split. "
        "Defaults to the checkpoint's eval_batches setting.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Evaluation seed. Defaults to the checkpoint's training seed.",
    )

    args = parser.parse_args()

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    train_config = {
        key: value
        for key, value in checkpoint["train_config"].items()
        if key in TrainConfig.__dataclass_fields__
    }

    config = TrainConfig(**train_config)

    if args.batches is not None:
        if args.batches <= 0:
            parser.error("--batches must be greater than zero.")

        config.eval_batches = args.batches

    if config.eval_batches <= 0:
        parser.error("Checkpoint eval_batches must be greater than zero.")

    model_config = ModelConfig.from_dict(checkpoint["model_config"])

    device = choose_device(config.device)

    model = GPTLanguageModel(model_config).to(device)

    model.load_state_dict(checkpoint["model"])

    dataset = TokenDataset(
        config.data_dir,
        model_config.block_size,
    )

    evaluation = estimate_loss(
        model=model,
        dataset=dataset,
        config=config,
        device=device,
        eval_seed=args.seed,
    )

    print(
        f"Checkpoint: {args.checkpoint}\n"
        f"Device: {device}\n"
        f"Evaluation batches per split: {config.eval_batches}\n"
        f"Evaluation seed: "
        f"{args.seed if args.seed is not None else config.seed}\n"
        f"Results: {evaluation}"
    )


if __name__ == "__main__":
    main()
