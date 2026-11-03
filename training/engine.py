import csv
import json
import math
import os
import random
import time

from pathlib import Path

import numpy as np
import torch

from torch.utils.tensorboard import SummaryWriter

from config import ModelConfig
from model.gpt import GPTLanguageModel, count_parameters
from tokenizer.loader import load_tokenizer
from training.data import TokenDataset


def choose_device(name):
    if name != "auto":
        return torch.device(name)

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_manifest(data_dir):
    manifest_path = Path(data_dir) / "manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Dataset manifest not found: {manifest_path}")

    return json.loads(manifest_path.read_text(encoding="utf-8"))


def validate_tokenizer(config, manifest):
    tokenizer_type = manifest.get("tokenizer")
    expected_vocab = int(manifest["vocab_size"])

    tokenizer = load_tokenizer(config.tokenizer_dir, tokenizer_type=tokenizer_type)

    if tokenizer.vocab_size != expected_vocab:
        raise ValueError(
            f"Tokenizer vocabulary mismatch: "
            f"manifest={expected_vocab}, "
            f"tokenizer={tokenizer.vocab_size}"
        )

    return tokenizer, tokenizer_type, expected_vocab


def estimate_loss(model, dataset, config, device, amp_dtype=None):
    model.eval()

    result = {}

    with torch.no_grad():
        for split in ("train", "val"):
            losses = []

            for _ in range(config.eval_batches):
                x, y = dataset.get_batch(split, config.batch_size, device)

                with torch.autocast(
                    device_type=device.type,
                    dtype=amp_dtype,
                    enabled=(amp_dtype is not None and device.type == "cuda"),
                ):
                    _, loss = model(x, y)

                losses.append(float(loss.item()))

            result[split] = sum(losses) / max(1, len(losses))

    model.train()

    return result


def load_checkpoint(path, model, optimizer=None, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model"])

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint


def train(config, resume=None):
    seed_everything(config.seed)

    device = choose_device(config.device)

    # Read the prepared dataset metadata.
    manifest = read_manifest(config.data_dir)

    tokenizer, tokenizer_type, vocab_size = validate_tokenizer(config, manifest)

    # The actual vocabulary always comes from the tokenizer.
    model_dict = dict(config.model or {})
    configured_vocab = model_dict.get("vocab_size")

    if configured_vocab != vocab_size:
        print(f"Overriding model vocab_size: " f"{configured_vocab} -> {vocab_size}")

    model_dict["vocab_size"] = vocab_size

    mc = ModelConfig.from_dict(model_dict)

    if mc.block_size <= 0:
        raise ValueError("block_size must be positive.")

    if mc.n_embd % mc.n_head != 0:
        raise ValueError("n_embd must be divisible by n_head.")

    dataset = TokenDataset(config.data_dir, mc.block_size)

    model = GPTLanguageModel(mc).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
    )

    amp_dtype = None

    if device.type == "cuda":
        if config.precision.lower() == "fp16":
            amp_dtype = torch.float16

        elif config.precision.lower() == "bf16":
            amp_dtype = torch.bfloat16

    scaler = torch.amp.GradScaler(
        "cuda", enabled=(amp_dtype == torch.float16 and device.type == "cuda")
    )

    start_step = 0

    if resume:
        checkpoint = torch.load(resume, map_location=device, weights_only=False)

        if checkpoint.get("tokenizer_type", tokenizer_type) != tokenizer_type:
            raise ValueError("Cannot resume: tokenizer type does not match checkpoint.")

        if checkpoint.get("tokenizer_vocab_size", vocab_size) != vocab_size:
            raise ValueError(
                "Cannot resume: tokenizer vocabulary does not match checkpoint."
            )

        if checkpoint["model_config"]["vocab_size"] != vocab_size:
            raise ValueError("Cannot resume: model vocabulary does not match dataset.")

        model.load_state_dict(checkpoint["model"])

        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])

        start_step = int(checkpoint.get("step", 0))

        if "rng" in checkpoint:
            rng = checkpoint["rng"]

            # Restore CPU PyTorch RNG state
            torch_rng = rng.get("torch")

            if torch_rng is not None:
                if not isinstance(torch_rng, torch.Tensor):
                    torch_rng = torch.tensor(torch_rng, dtype=torch.uint8)
                else:
                    torch_rng = torch_rng.cpu().to(dtype=torch.uint8)

                torch.set_rng_state(torch_rng)

            # Restore CUDA RNG states when available.
            cuda_rng = rng.get("cuda")

            if cuda_rng is not None and torch.cuda.is_available():
                current_device_count = torch.cuda.device_count()

                for device_index, state in enumerate(cuda_rng):
                    if device_index >= current_device_count:
                        break

                    # CUDA RNG restoration requires a CPU ByteTensor.
                    if not isinstance(state, torch.Tensor):
                        state = torch.as_tensor(state, dtype=torch.uint8)
                    else:
                        state = state.detach().cpu().to(dtype=torch.uint8)

                    torch.cuda.set_rng_state(state, device=device_index)

            # Restore NumPy RNG state
            numpy_rng = rng.get("numpy")

            if numpy_rng is not None:
                np.random.set_state(numpy_rng)

            # Restore Python RNG state
            python_rng = rng.get("python")

            if python_rng is not None:
                random.setstate(python_rng)

    output_dir = Path(config.out_dir) / config.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(str(Path("runs") / config.run_name))

    csv_path = output_dir / "metrics.csv"

    fresh_csv = not csv_path.exists() or start_step == 0

    csv_file = csv_path.open("w" if fresh_csv else "a", newline="", encoding="utf-8")

    csv_writer = csv.writer(csv_file)

    if fresh_csv:
        csv_writer.writerow(
            [
                "step",
                "train_loss",
                "val_loss",
                "lr",
                "tokens_per_sec",
                "elapsed_sec",
                "peak_vram_mb",
                "parameters",
            ]
        )

    parameters = count_parameters(model)

    print(
        f"device={device}; "
        f"parameters={parameters:,}; "
        f"vocab_size={vocab_size}; "
        f"tokenizer={tokenizer_type}; "
        f"block_size={mc.block_size}; "
        f"precision={config.precision}"
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    start_time = time.time()

    try:
        model.train()

        for step in range(start_step, config.max_steps):

            # Learning rate schedule.
            if step < config.warmup_steps:
                lr = config.learning_rate * (step + 1) / max(1, config.warmup_steps)
            else:
                progress = (step - config.warmup_steps) / max(
                    1, config.max_steps - config.warmup_steps
                )

                lr = config.min_learning_rate + 0.5 * (
                    config.learning_rate - config.min_learning_rate
                ) * (1 + math.cos(math.pi * min(1, progress)))

            for group in optimizer.param_groups:
                group["lr"] = lr

            optimizer.zero_grad(set_to_none=True)

            loss_total = 0.0

            for _ in range(config.gradient_accumulation_steps):

                x, y = dataset.get_batch("train", config.batch_size, device)

                with torch.autocast(
                    device_type=device.type,
                    dtype=amp_dtype,
                    enabled=(amp_dtype is not None and device.type == "cuda"),
                ):
                    _, loss = model(x, y)

                    scaled_loss = loss / config.gradient_accumulation_steps

                scaler.scale(scaled_loss).backward()

                loss_total += float(loss.item())

            # Average loss across microbatches.
            average_loss = loss_total / config.gradient_accumulation_steps

            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            global_step = step + 1

            if global_step % config.log_interval == 0:
                elapsed = time.time() - start_time

                tokens_per_sec = (
                    global_step
                    * config.batch_size
                    * config.gradient_accumulation_steps
                    * mc.block_size
                    / max(elapsed, 1e-6)
                )

                peak_vram = (
                    torch.cuda.max_memory_allocated(device) / (1024**2)
                    if device.type == "cuda"
                    else 0
                )

                print(
                    f"step {global_step}/{config.max_steps} "
                    f"loss={average_loss:.4f} "
                    f"lr={lr:.2e} "
                    f"tok/s~{tokens_per_sec:.1f} "
                    f"peak_vram={peak_vram:.0f}MB"
                )

                writer.add_scalar("train/loss_step", average_loss, global_step)

            if global_step % config.eval_interval == 0 or global_step == 1:
                evaluation = estimate_loss(model, dataset, config, device, amp_dtype)

                elapsed = time.time() - start_time

                tokens_per_sec = (
                    global_step
                    * config.batch_size
                    * config.gradient_accumulation_steps
                    * mc.block_size
                    / max(elapsed, 1e-6)
                )

                peak_vram = (
                    torch.cuda.max_memory_allocated(device) / (1024**2)
                    if device.type == "cuda"
                    else 0
                )

                csv_writer.writerow(
                    [
                        global_step,
                        evaluation["train"],
                        evaluation["val"],
                        lr,
                        tokens_per_sec,
                        elapsed,
                        peak_vram,
                        parameters,
                    ]
                )

                csv_file.flush()

                writer.add_scalar("loss/train_eval", evaluation["train"], global_step)

                writer.add_scalar("loss/val", evaluation["val"], global_step)

                print(
                    f"eval step={global_step} "
                    f"train={evaluation['train']:.4f} "
                    f"val={evaluation['val']:.4f}"
                )

            if (
                global_step % config.save_interval == 0
                or global_step == config.max_steps
            ):
                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": global_step,
                    "model_config": mc.__dict__,
                    "train_config": config.to_dict(),
                    "parameters": parameters,
                    "tokenizer_type": tokenizer_type,
                    "tokenizer_vocab_size": vocab_size,
                    "tokenizer_dir": config.tokenizer_dir,
                    "dataset_manifest": manifest,
                    "rng": {
                        "torch": torch.get_rng_state().cpu(),
                        "cuda": (
                            torch.cuda.get_rng_state_all()
                            if torch.cuda.is_available()
                            else None
                        ),
                        "numpy": np.random.get_state(),
                        "python": random.getstate(),
                    },
                }

                temp_path = output_dir / "latest.tmp.pt"

                torch.save(checkpoint, temp_path)

                os.replace(temp_path, output_dir / "latest.pt")

                torch.save(checkpoint, output_dir / f"step_{global_step}.pt")

                print(f"Saved checkpoint: " f"{output_dir / 'latest.pt'}")

    finally:
        csv_file.close()
        writer.close()

    return str(output_dir / "latest.pt")
