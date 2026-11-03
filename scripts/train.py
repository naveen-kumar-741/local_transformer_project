import os

# Must be configured before importing PyTorch or initializing CUDA.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse

from config import TrainConfig
from training.engine import train


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None)

    args = parser.parse_args()

    config = TrainConfig.load(args.config)

    print(f"Deterministic mode: {config.deterministic}")

    checkpoint_path = train(config, args.resume)

    print("Saved checkpoint:", checkpoint_path)


if __name__ == "__main__":
    main()
