import csv
import hashlib
import json
import math
import os
import random
import signal
import time
import warnings

from pathlib import Path

import numpy as np
import torch

from torch.utils.tensorboard import SummaryWriter

from config import ModelConfig
from model.gpt import GPTLanguageModel, count_parameters
from tokenizer.loader import load_tokenizer
from training.data import TokenDataset

CHECKPOINT_FORMAT_VERSION = 2

METRICS_HEADER = [
    "step",
    "train_loss",
    "val_loss",
    "lr",
    "interval_tokens_per_sec",
    "run_average_tokens_per_sec",
    "interval_training_sec",
    "run_training_sec",
    "peak_allocated_vram_mb",
    "peak_reserved_vram_mb",
    "parameters",
]

# These settings affect the optimizer updates or the training trajectory.
# They must remain unchanged when resuming a checkpoint.
RESUME_CRITICAL_CONFIG_KEYS = [
    "batch_size",
    "gradient_accumulation_steps",
    "learning_rate",
    "min_learning_rate",
    "warmup_steps",
    "weight_decay",
    "grad_clip",
    "seed",
    "precision",
    "deterministic",
]

# These settings affect logging/evaluation/checkpoint frequency only.
# They can safely change when resuming.
RESUME_SAFE_CONFIG_KEYS = [
    "eval_interval",
    "eval_batches",
    "save_interval",
    "log_interval",
]


# =============================================================
# Determinism and device
# =============================================================


def configure_determinism(enabled):
    """
    Configure deterministic execution.

    This must run before CUDA is initialized.
    Strict mode may reduce training performance.
    """

    if not enabled:
        return

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    torch.use_deterministic_algorithms(True)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)


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


# =============================================================
# Fingerprints
# =============================================================


def sha256_file(path):
    """
    Calculate a file's SHA-256 without loading the entire file
    into memory.
    """

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"File not found for fingerprint: {path}")

    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def sha256_json(data):
    """
    Hash JSON data using a stable serialization.
    """

    serialized = json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")

    return hashlib.sha256(serialized).hexdigest()


def sha256_directory(directory):
    """
    Fingerprint every file inside a directory.

    Relative paths are included so that adding, removing,
    renaming, or changing a tokenizer file changes the hash.
    """

    directory = Path(directory)

    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found for fingerprint: {directory}")

    files = sorted(path for path in directory.rglob("*") if path.is_file())

    if not files:
        raise ValueError(f"No files found in fingerprint directory: {directory}")

    entries = []

    for path in files:
        relative_path = path.relative_to(directory).as_posix()

        entries.append(
            {
                "path": relative_path,
                "sha256": sha256_file(path),
            }
        )

    return sha256_json(entries)


def read_manifest(data_dir):
    manifest_path = Path(data_dir) / "manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Dataset manifest not found: {manifest_path}")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid dataset manifest: {manifest_path}") from error

    if not isinstance(manifest, dict):
        raise ValueError("Dataset manifest must contain a JSON object.")

    return manifest


def create_dataset_fingerprint(data_dir, manifest):
    """
    Fingerprint the manifest and actual processed token files.

    This detects:
    - Changes to raw dataset hashes recorded in the manifest.
    - Changes to train/validation split metadata.
    - Changes to processed train.bin or val.bin, even if
      manifest.json was not updated.
    """

    data_dir = Path(data_dir)

    train_path = data_dir / "train.bin"
    val_path = data_dir / "val.bin"

    dataset_files = {
        "train.bin": sha256_file(train_path),
        "val.bin": sha256_file(val_path),
    }

    manifest_hash = sha256_json(manifest)

    fingerprint_data = {
        "manifest_sha256": manifest_hash,
        "processed_files": dataset_files,
    }

    return {
        "manifest_sha256": manifest_hash,
        "processed_files": dataset_files,
        "sha256": sha256_json(fingerprint_data),
    }


def create_tokenizer_fingerprint(tokenizer_dir):
    """
    Fingerprint all tokenizer artifacts.
    """

    return sha256_directory(tokenizer_dir)


# =============================================================
# Tokenizer and dataset validation
# =============================================================


def validate_tokenizer(config, manifest):
    tokenizer_type = manifest.get("tokenizer")

    if not tokenizer_type:
        raise ValueError("Dataset manifest does not contain a tokenizer type.")

    if "vocab_size" not in manifest:
        raise ValueError("Dataset manifest does not contain vocab_size.")

    expected_vocab = int(manifest["vocab_size"])

    tokenizer = load_tokenizer(
        config.tokenizer_dir,
        tokenizer_type=tokenizer_type,
    )

    if tokenizer.vocab_size != expected_vocab:
        raise ValueError(
            "Tokenizer vocabulary mismatch: "
            f"manifest={expected_vocab}, "
            f"tokenizer={tokenizer.vocab_size}"
        )

    return tokenizer, tokenizer_type, expected_vocab


def validate_model_config(model_config):
    if model_config.block_size <= 0:
        raise ValueError("block_size must be positive.")

    if model_config.n_layer <= 0:
        raise ValueError("n_layer must be positive.")

    if model_config.n_head <= 0:
        raise ValueError("n_head must be positive.")

    if model_config.n_embd <= 0:
        raise ValueError("n_embd must be positive.")

    if model_config.vocab_size <= 0:
        raise ValueError("vocab_size must be positive.")

    if model_config.n_embd % model_config.n_head != 0:
        raise ValueError("n_embd must be divisible by n_head.")


# =============================================================
# Checkpoint compatibility validation
# =============================================================


def compare_config_values(saved, current, keys):
    """
    Return a dictionary containing mismatched settings.
    """

    differences = {}

    for key in keys:
        if key not in saved or key not in current:
            continue

        saved_value = saved[key]
        current_value = current[key]

        if saved_value != current_value:
            differences[key] = {
                "checkpoint": saved_value,
                "current": current_value,
            }

    return differences


def format_differences(differences):
    return "\n".join(
        f"  - {key}: checkpoint={values['checkpoint']!r}, "
        f"current={values['current']!r}"
        for key, values in differences.items()
    )


def validate_checkpoint_structure(checkpoint):
    """
    Validate the basic structure before accessing checkpoint data.
    """

    if not isinstance(checkpoint, dict):
        raise ValueError("Invalid checkpoint: expected a dictionary.")

    required_keys = [
        "model",
        "model_config",
        "step",
    ]

    missing = [key for key in required_keys if key not in checkpoint]

    if missing:
        raise ValueError(
            "Invalid checkpoint. Missing required fields: " + ", ".join(missing)
        )

    if not isinstance(checkpoint["model"], dict):
        raise ValueError("Invalid checkpoint: model state must be a dictionary.")

    if not isinstance(checkpoint["model_config"], dict):
        raise ValueError("Invalid checkpoint: model_config must be a dictionary.")

    step = checkpoint["step"]

    if not isinstance(step, int) or step < 0:
        raise ValueError(f"Invalid checkpoint step: {step!r}")

    version = int(checkpoint.get("checkpoint_format_version", 0))

    if version > CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            "Checkpoint was created by a newer format version. "
            f"Checkpoint version={version}, "
            f"supported version={CHECKPOINT_FORMAT_VERSION}."
        )

    if version < 0:
        raise ValueError(f"Invalid checkpoint format version: {version}")

    return version


def validate_checkpoint_model_config(checkpoint, current_model_config):
    saved_config = checkpoint["model_config"]
    current_config = current_model_config.__dict__

    differences = compare_config_values(
        saved_config,
        current_config,
        list(current_config.keys()),
    )

    if differences:
        raise ValueError(
            "Cannot resume: model architecture/configuration mismatch.\n"
            + format_differences(differences)
        )


def validate_checkpoint_tokenizer(
    checkpoint,
    tokenizer_type,
    vocab_size,
    tokenizer_fingerprint,
):
    saved_type = checkpoint.get("tokenizer_type")
    saved_vocab = checkpoint.get("tokenizer_vocab_size")

    if saved_type is not None and saved_type != tokenizer_type:
        raise ValueError(
            "Cannot resume: tokenizer type mismatch. "
            f"Checkpoint={saved_type!r}, current={tokenizer_type!r}."
        )

    if saved_vocab is not None and int(saved_vocab) != vocab_size:
        raise ValueError(
            "Cannot resume: tokenizer vocabulary mismatch. "
            f"Checkpoint={saved_vocab}, current={vocab_size}."
        )

    saved_fingerprint = checkpoint.get("tokenizer_fingerprint")

    if saved_fingerprint is not None:
        if saved_fingerprint != tokenizer_fingerprint:
            raise ValueError(
                "Cannot resume: tokenizer files have changed. "
                "The current tokenizer fingerprint does not match "
                "the checkpoint."
            )

    elif saved_type is not None:
        warnings.warn(
            "Legacy checkpoint does not contain a tokenizer "
            "fingerprint. Tokenizer type and vocabulary were "
            "validated, but exact tokenizer-file identity cannot "
            "be verified.",
            RuntimeWarning,
        )


def validate_checkpoint_dataset(
    checkpoint,
    current_manifest,
    current_fingerprint,
):
    saved_fingerprint = checkpoint.get("dataset_fingerprint")

    if saved_fingerprint is not None:
        if saved_fingerprint.get("sha256") != current_fingerprint["sha256"]:
            raise ValueError(
                "Cannot resume: dataset fingerprint mismatch.\n"
                "The dataset manifest or processed train/validation "
                "files have changed since this checkpoint was saved."
            )

        return

    # Legacy checkpoint: compare the manifest that was saved
    # before dataset fingerprints were introduced.
    saved_manifest = checkpoint.get("dataset_manifest")

    if saved_manifest is None:
        warnings.warn(
            "Legacy checkpoint does not contain a dataset manifest "
            "or dataset fingerprint. Dataset identity cannot be "
            "verified for this checkpoint.",
            RuntimeWarning,
        )
        return

    if sha256_json(saved_manifest) != sha256_json(current_manifest):
        raise ValueError(
            "Cannot resume: dataset manifest differs from the "
            "legacy checkpoint.\n"
            "The raw dataset metadata or split configuration changed."
        )

    warnings.warn(
        "Legacy checkpoint: dataset manifest matches, but the "
        "processed train.bin and val.bin files cannot be compared "
        "against historical fingerprints because they were not "
        "saved in this checkpoint.",
        RuntimeWarning,
    )


def validate_checkpoint_training_config(checkpoint, config):
    saved_config = checkpoint.get("train_config")

    if not isinstance(saved_config, dict):
        warnings.warn(
            "Checkpoint does not contain a usable train_config. "
            "Training configuration compatibility cannot be fully "
            "verified.",
            RuntimeWarning,
        )
        return

    current_config = config.to_dict()

    differences = compare_config_values(
        saved_config,
        current_config,
        RESUME_CRITICAL_CONFIG_KEYS,
    )

    if differences:
        raise ValueError(
            "Cannot resume: training settings that affect "
            "the optimization trajectory have changed.\n"
            + format_differences(differences)
        )

    saved_max_steps = saved_config.get("max_steps")
    current_max_steps = config.max_steps

    if saved_max_steps is not None and saved_max_steps != current_max_steps:
        if current_max_steps < checkpoint["step"]:
            raise ValueError(
                "Cannot resume: max_steps is smaller than the "
                f"checkpoint step. checkpoint step={checkpoint['step']}, "
                f"current max_steps={current_max_steps}."
            )

        warnings.warn(
            "max_steps changed from "
            f"{saved_max_steps} to {current_max_steps}. "
            "This is allowed, but the future cosine learning-rate "
            "schedule will change. This is an intentional training "
            "extension, not an exact continuation of the original "
            "learning-rate schedule.",
            RuntimeWarning,
        )

    safe_differences = compare_config_values(
        saved_config,
        current_config,
        RESUME_SAFE_CONFIG_KEYS,
    )

    if safe_differences:
        print("\nResume configuration changes (safe):")

        print(format_differences(safe_differences))


def validate_resume_checkpoint(
    checkpoint,
    config,
    model_config,
    tokenizer_type,
    vocab_size,
    tokenizer_fingerprint,
    manifest,
    dataset_fingerprint,
):
    """
    Run every compatibility check before loading model/optimizer/RNG.
    """

    version = validate_checkpoint_structure(checkpoint)

    print(f"Checkpoint format version: " f"{version if version else 'legacy (0)'}")

    if version == 0:
        warnings.warn(
            "Resuming a legacy checkpoint created before "
            "checkpoint compatibility metadata was introduced.",
            RuntimeWarning,
        )

    validate_checkpoint_model_config(
        checkpoint,
        model_config,
    )

    validate_checkpoint_tokenizer(
        checkpoint,
        tokenizer_type,
        vocab_size,
        tokenizer_fingerprint,
    )

    validate_checkpoint_dataset(
        checkpoint,
        manifest,
        dataset_fingerprint,
    )

    validate_checkpoint_training_config(
        checkpoint,
        config,
    )

    saved_deterministic = checkpoint.get(
        "train_config",
        {},
    ).get("deterministic")

    if saved_deterministic is not None and bool(saved_deterministic) != bool(
        config.deterministic
    ):
        raise ValueError(
            "Cannot resume with a different deterministic setting. "
            f"Checkpoint={saved_deterministic}, "
            f"current config={config.deterministic}."
        )

    if checkpoint["step"] > config.max_steps:
        raise ValueError(
            "Checkpoint step is greater than current max_steps. "
            f"Checkpoint={checkpoint['step']}, "
            f"current max_steps={config.max_steps}."
        )


# =============================================================
# Deterministic evaluation
# =============================================================


def estimate_loss(
    model,
    dataset,
    config,
    device,
    amp_dtype=None,
    eval_seed=None,
):
    """
    Evaluate train and validation loss on fixed batches.

    Local generators ensure evaluation does not modify
    the global training RNG state.
    """

    was_training = model.training
    model.eval()

    result = {}

    base_seed = (
        int(eval_seed) if eval_seed is not None else int(getattr(config, "seed", 1337))
    )

    split_seeds = {
        "train": base_seed,
        "val": base_seed + 1_000_003,
    }

    try:
        with torch.no_grad():
            for split in ("train", "val"):
                losses = []

                generator = torch.Generator(device="cpu")
                generator.manual_seed(split_seeds[split] % (2**63 - 1))

                for _ in range(config.eval_batches):
                    x, y = dataset.get_batch(
                        split,
                        config.batch_size,
                        device,
                        generator=generator,
                    )

                    with torch.autocast(
                        device_type=device.type,
                        dtype=amp_dtype,
                        enabled=(amp_dtype is not None and device.type == "cuda"),
                    ):
                        _, loss = model(x, y)

                    losses.append(float(loss.item()))

                result[split] = sum(losses) / max(1, len(losses))

    finally:
        model.train(was_training)

    return result


# =============================================================
# Performance instrumentation
# =============================================================


def synchronize_device(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def get_peak_vram(device):
    """
    Return PyTorch CUDA allocator peak memory in MB.
    """

    if device.type != "cuda":
        return {
            "allocated": 0.0,
            "reserved": 0.0,
        }

    return {
        "allocated": (torch.cuda.max_memory_allocated(device) / (1024**2)),
        "reserved": (torch.cuda.max_memory_reserved(device) / (1024**2)),
    }


def get_metrics_csv_path(output_dir, start_step):
    """
    Preserve old metrics.csv files when their schema differs.
    """

    metrics_path = output_dir / "metrics.csv"

    if start_step == 0 or not metrics_path.exists():
        return metrics_path

    with metrics_path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        existing_header = next(csv.reader(file), [])

    if existing_header == METRICS_HEADER:
        return metrics_path

    version = 2

    while True:
        candidate = output_dir / f"metrics_v{version}.csv"

        if not candidate.exists():
            return candidate

        with candidate.open(
            "r",
            newline="",
            encoding="utf-8",
        ) as file:
            candidate_header = next(csv.reader(file), [])

        if candidate_header == METRICS_HEADER:
            return candidate

        version += 1


# =============================================================
# Checkpoint loading and RNG restoration
# =============================================================


def load_checkpoint(path, model, optimizer=None, device="cpu"):
    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    validate_checkpoint_structure(checkpoint)

    model.load_state_dict(checkpoint["model"])

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint


def restore_rng_state(rng):
    """
    Restore all saved RNG states.

    Call after model, optimizer and GradScaler initialization.
    """

    torch_rng = rng.get("torch")

    if torch_rng is not None:
        if not isinstance(torch_rng, torch.Tensor):
            torch_rng = torch.tensor(
                torch_rng,
                dtype=torch.uint8,
            )
        else:
            torch_rng = torch_rng.detach().cpu().to(dtype=torch.uint8)

        torch.set_rng_state(torch_rng)

    cuda_rng = rng.get("cuda")

    if cuda_rng is not None and torch.cuda.is_available():
        current_device_count = torch.cuda.device_count()

        if len(cuda_rng) != current_device_count:
            raise RuntimeError(
                "CUDA device count differs from the saved checkpoint: "
                f"checkpoint={len(cuda_rng)}, "
                f"current={current_device_count}. "
                "Exact RNG restoration is not possible."
            )

        for device_index, state in enumerate(cuda_rng):
            if not isinstance(state, torch.Tensor):
                state = torch.as_tensor(
                    state,
                    dtype=torch.uint8,
                )
            else:
                state = state.detach().cpu().to(dtype=torch.uint8)

            torch.cuda.set_rng_state(
                state,
                device=device_index,
            )

    numpy_rng = rng.get("numpy")

    if numpy_rng is not None:
        np.random.set_state(numpy_rng)

    python_rng = rng.get("python")

    if python_rng is not None:
        random.setstate(python_rng)


# =============================================================
# Training
# =============================================================


def train(config, resume=None):

    # Must happen before CUDA initialization.
    configure_determinism(bool(getattr(config, "deterministic", False)))

    seed_everything(config.seed)

    device = choose_device(config.device)

    # ---------------------------------------------------------
    # Dataset and tokenizer preparation
    # ---------------------------------------------------------

    manifest = read_manifest(config.data_dir)

    tokenizer, tokenizer_type, vocab_size = validate_tokenizer(
        config,
        manifest,
    )

    tokenizer_fingerprint = create_tokenizer_fingerprint(config.tokenizer_dir)

    dataset_fingerprint = create_dataset_fingerprint(
        config.data_dir,
        manifest,
    )

    model_dict = dict(config.model or {})
    configured_vocab = model_dict.get("vocab_size")

    if configured_vocab != vocab_size:
        print(f"Overriding model vocab_size: " f"{configured_vocab} -> {vocab_size}")

    model_dict["vocab_size"] = vocab_size

    mc = ModelConfig.from_dict(model_dict)

    validate_model_config(mc)

    dataset = TokenDataset(
        config.data_dir,
        mc.block_size,
    )

    # ---------------------------------------------------------
    # Model and optimizer initialization
    # ---------------------------------------------------------

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
        "cuda",
        enabled=(amp_dtype == torch.float16 and device.type == "cuda"),
    )

    start_step = 0

    # ---------------------------------------------------------
    # Resume
    # ---------------------------------------------------------

    if resume:
        resume_path = Path(resume)

        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

        checkpoint = torch.load(
            resume_path,
            map_location=device,
            weights_only=False,
        )

        # Validate everything before loading any saved state.
        validate_resume_checkpoint(
            checkpoint=checkpoint,
            config=config,
            model_config=mc,
            tokenizer_type=tokenizer_type,
            vocab_size=vocab_size,
            tokenizer_fingerprint=tokenizer_fingerprint,
            manifest=manifest,
            dataset_fingerprint=dataset_fingerprint,
        )

        # -----------------------------------------------------
        # Restore states after all compatibility checks pass.
        # -----------------------------------------------------

        model.load_state_dict(checkpoint["model"])

        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        else:
            warnings.warn(
                "Checkpoint has no optimizer state. "
                "This resume will not preserve optimizer continuity.",
                RuntimeWarning,
            )

        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])

        start_step = int(checkpoint["step"])

        if "rng" in checkpoint:
            restore_rng_state(checkpoint["rng"])
        else:
            warnings.warn(
                "Checkpoint has no RNG state. "
                "Exact reproducibility cannot be guaranteed.",
                RuntimeWarning,
            )

        print(f"Checkpoint compatibility validation passed: " f"{resume_path}")

    # ---------------------------------------------------------
    # Output setup
    # ---------------------------------------------------------

    output_dir = Path(config.out_dir) / config.run_name

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    writer = SummaryWriter(str(Path("runs") / config.run_name))

    csv_path = get_metrics_csv_path(
        output_dir,
        start_step,
    )

    fresh_csv = not csv_path.exists() or start_step == 0

    csv_file = csv_path.open(
        "w" if fresh_csv else "a",
        newline="",
        encoding="utf-8",
    )

    csv_writer = csv.writer(csv_file)

    if fresh_csv:
        csv_writer.writerow(METRICS_HEADER)
        csv_file.flush()

    parameters = count_parameters(model)

    print(
        f"device={device}; "
        f"parameters={parameters:,}; "
        f"vocab_size={vocab_size}; "
        f"tokenizer={tokenizer_type}; "
        f"block_size={mc.block_size}; "
        f"precision={config.precision}; "
        f"deterministic={config.deterministic}"
    )

    print(f"Starting step: {start_step}; " f"Target step: {config.max_steps}")

    # ---------------------------------------------------------
    # Graceful interruption state
    # ---------------------------------------------------------

    interrupt_state = {
        "requested": False,
    }

    previous_sigint_handler = signal.getsignal(signal.SIGINT)

    def handle_sigint(signum, frame):
        """
        Defer Ctrl+C until the current optimizer step finishes.

        Do not perform checkpoint operations inside this handler.
        """

        if not interrupt_state["requested"]:
            interrupt_state["requested"] = True

            print(
                "\nCtrl+C received. "
                "Finishing the current optimizer step "
                "before saving..."
            )

        else:
            print(
                "\nInterruption already requested. "
                "Please wait for checkpoint saving."
            )

    signal.signal(
        signal.SIGINT,
        handle_sigint,
    )

    # ---------------------------------------------------------
    # Training instrumentation
    # ---------------------------------------------------------

    synchronize_device(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    run_tokens = 0
    interval_tokens = 0

    run_training_elapsed = 0.0
    interval_elapsed_accum = 0.0

    interval_start = time.perf_counter()

    tokens_per_optimizer_step = (
        config.batch_size * config.gradient_accumulation_steps * mc.block_size
    )

    def capture_training_time():
        nonlocal interval_start
        nonlocal interval_elapsed_accum

        synchronize_device(device)

        now = time.perf_counter()

        interval_elapsed_accum += max(
            0.0,
            now - interval_start,
        )

        interval_start = now

        return interval_elapsed_accum

    def reset_timer_after_non_training_work():
        nonlocal interval_start

        synchronize_device(device)

        interval_start = time.perf_counter()

    # ---------------------------------------------------------
    # Checkpoint helper
    # ---------------------------------------------------------

    def save_checkpoint(global_step):
        """
        Save the completed optimizer step.

        latest.pt is replaced atomically.
        """

        synchronize_device(device)

        checkpoint = {
            "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "step": global_step,
            "model_config": mc.__dict__,
            "train_config": config.to_dict(),
            "parameters": parameters,
            "tokenizer_type": tokenizer_type,
            "tokenizer_vocab_size": vocab_size,
            "tokenizer_dir": config.tokenizer_dir,
            "tokenizer_fingerprint": tokenizer_fingerprint,
            "dataset_manifest": manifest,
            "dataset_fingerprint": dataset_fingerprint,
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
        latest_path = output_dir / "latest.pt"

        torch.save(
            checkpoint,
            temp_path,
        )

        os.replace(
            temp_path,
            latest_path,
        )

        step_path = output_dir / f"step_{global_step}.pt"

        torch.save(
            checkpoint,
            step_path,
        )

        print(f"Saved checkpoint: {latest_path} " f"(step {global_step})")

        reset_timer_after_non_training_work()

        return latest_path

    # ---------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------

    try:
        model.train()

        for step in range(
            start_step,
            config.max_steps,
        ):

            # Learning rate schedule.
            if step < config.warmup_steps:
                lr = config.learning_rate * (step + 1) / max(1, config.warmup_steps)

            else:
                progress = (step - config.warmup_steps) / max(
                    1,
                    config.max_steps - config.warmup_steps,
                )

                lr = config.min_learning_rate + 0.5 * (
                    config.learning_rate - config.min_learning_rate
                ) * (1 + math.cos(math.pi * min(1, progress)))

            for group in optimizer.param_groups:
                group["lr"] = lr

            optimizer.zero_grad(set_to_none=True)

            loss_total = 0.0

            # -------------------------------------------------
            # Gradient accumulation
            # -------------------------------------------------

            for _ in range(config.gradient_accumulation_steps):

                x, y = dataset.get_batch(
                    "train",
                    config.batch_size,
                    device,
                )

                with torch.autocast(
                    device_type=device.type,
                    dtype=amp_dtype,
                    enabled=(amp_dtype is not None and device.type == "cuda"),
                ):
                    _, loss = model(x, y)

                    scaled_loss = loss / config.gradient_accumulation_steps

                scaler.scale(scaled_loss).backward()

                loss_total += float(loss.item())

            average_loss = loss_total / config.gradient_accumulation_steps

            # -------------------------------------------------
            # Complete optimizer step
            # -------------------------------------------------

            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.grad_clip,
            )

            scaler.step(optimizer)
            scaler.update()

            # The completed optimizer step is now safe to save.
            global_step = step + 1

            run_tokens += tokens_per_optimizer_step
            interval_tokens += tokens_per_optimizer_step

            # -------------------------------------------------
            # Throughput logging
            # -------------------------------------------------

            should_log = (
                global_step % config.log_interval == 0
                or global_step == config.max_steps
            )

            if should_log:
                interval_training_elapsed = capture_training_time()

                run_training_elapsed += interval_training_elapsed

                interval_tokens_per_sec = interval_tokens / max(
                    interval_training_elapsed, 1e-9
                )

                run_average_tokens_per_sec = run_tokens / max(
                    run_training_elapsed, 1e-9
                )

                vram = get_peak_vram(device)

                print(
                    f"step {global_step}/"
                    f"{config.max_steps} "
                    f"loss={average_loss:.4f} "
                    f"lr={lr:.2e} "
                    f"tok/s(interval)="
                    f"{interval_tokens_per_sec:.1f} "
                    f"tok/s(run)="
                    f"{run_average_tokens_per_sec:.1f} "
                    f"peak_allocated="
                    f"{vram['allocated']:.0f}MB "
                    f"peak_reserved="
                    f"{vram['reserved']:.0f}MB"
                )

                writer.add_scalar(
                    "train/loss_step",
                    average_loss,
                    global_step,
                )

                writer.add_scalar(
                    "performance/interval_tokens_per_sec",
                    interval_tokens_per_sec,
                    global_step,
                )

                writer.add_scalar(
                    "performance/run_average_tokens_per_sec",
                    run_average_tokens_per_sec,
                    global_step,
                )

                writer.add_scalar(
                    "memory/peak_allocated_mb",
                    vram["allocated"],
                    global_step,
                )

                writer.add_scalar(
                    "memory/peak_reserved_mb",
                    vram["reserved"],
                    global_step,
                )

                interval_elapsed_accum = 0.0
                interval_tokens = 0

            # -------------------------------------------------
            # Evaluation
            # Skip if interruption was requested.
            # -------------------------------------------------

            should_evaluate = (
                global_step % config.eval_interval == 0 or global_step == 1
            ) and not interrupt_state["requested"]

            if should_evaluate:
                capture_training_time()

                evaluation = estimate_loss(
                    model,
                    dataset,
                    config,
                    device,
                    amp_dtype,
                )

                reset_timer_after_non_training_work()

                pending_training_elapsed = interval_elapsed_accum

                current_run_training_elapsed = (
                    run_training_elapsed + pending_training_elapsed
                )

                interval_tokens_per_sec = interval_tokens / max(
                    pending_training_elapsed, 1e-9
                )

                run_average_tokens_per_sec = run_tokens / max(
                    current_run_training_elapsed, 1e-9
                )

                vram = get_peak_vram(device)

                csv_writer.writerow(
                    [
                        global_step,
                        evaluation["train"],
                        evaluation["val"],
                        lr,
                        interval_tokens_per_sec,
                        run_average_tokens_per_sec,
                        pending_training_elapsed,
                        current_run_training_elapsed,
                        vram["allocated"],
                        vram["reserved"],
                        parameters,
                    ]
                )

                csv_file.flush()

                writer.add_scalar(
                    "loss/train_eval",
                    evaluation["train"],
                    global_step,
                )

                writer.add_scalar(
                    "loss/val",
                    evaluation["val"],
                    global_step,
                )

                print(
                    f"eval step={global_step} "
                    f"train={evaluation['train']:.4f} "
                    f"val={evaluation['val']:.4f}"
                )

            # -------------------------------------------------
            # Checkpoint saving
            # -------------------------------------------------

            should_save = (
                global_step % config.save_interval == 0
                or global_step == config.max_steps
                or interrupt_state["requested"]
            )

            if should_save:
                capture_training_time()

                save_checkpoint(global_step)

            # -------------------------------------------------
            # Graceful exit
            # -------------------------------------------------

            if interrupt_state["requested"]:
                print("\nTraining interrupted gracefully.")

                print(f"Last completed optimizer step: " f"{global_step}")

                print(f"Resume checkpoint: " f"{output_dir / 'latest.pt'}")

                break

        else:
            print("\nTraining completed successfully.")

    finally:
        # Restore the handler that was active before training.
        signal.signal(
            signal.SIGINT,
            previous_sigint_handler,
        )

        csv_file.close()
        writer.close()

    return str(output_dir / "latest.pt")
