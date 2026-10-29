from dataclasses import dataclass, asdict
from typing import Any
import json
from pathlib import Path

@dataclass
class ModelConfig:
    vocab_size: int = 256
    block_size: int = 256
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.1
    bias: bool = True
    gradient_checkpointing: bool = False

    @classmethod
    def from_dict(cls, d: dict[str, Any]):
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in d.items() if k in allowed})

@dataclass
class TrainConfig:
    run_name: str = "demo"
    data_dir: str = "data/processed"
    tokenizer_dir: str = "data/processed/tokenizer"
    out_dir: str = "checkpoints"
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_steps: int = 500
    eval_interval: int = 50
    eval_batches: int = 10
    save_interval: int = 100
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_steps: int = 20
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    seed: int = 1337
    precision: str = "fp16"
    num_workers: int = 0
    log_interval: int = 10
    device: str = "auto"
    model: dict[str, Any] | None = None

    @classmethod
    def load(cls, path: str):
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        model = d.pop("model", {})
        obj = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        obj.model = model
        return obj

    def to_dict(self):
        return asdict(self)
