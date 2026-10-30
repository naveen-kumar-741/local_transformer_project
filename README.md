# Local Transformer Training Project (Phases 0–15)

A modular PyTorch GPT-style language model project designed to understand and build Transformer models from scratch.

The project targets Windows/Linux laptops, including systems with an NVIDIA RTX 3050-class GPU and 4 GB VRAM. It covers character-level tokenization, BPE, GPT architecture, training, evaluation, reproducible checkpoints, model-size experiments, optimization, optional fine-tuning, RAG, and speech integration.

The primary goal is to understand how GPT-style language models work internally, from raw text to tokenization, training, and text generation.

## Project Status

<text weight="semibold">Current milestone: Character-level GPT baseline successfully trained.</text>

| Component                  | Current status                     |
| -------------------------- | ---------------------------------- |
| Environment and CUDA setup | Completed                          |
| Character-level tokenizer  | Implemented                        |
| GPT architecture           | Implemented                        |
| Dataset preparation        | Working                            |
| Training pipeline          | Working                            |
| CUDA FP16 training         | Working                            |
| Checkpoint saving          | Working                            |
| Model evaluation           | Working                            |
| Text generation            | Working, quality needs improvement |
| BPE tokenizer              | Next milestone                     |
| 10M–100M model experiments | Planned                            |
| Fine-tuning                | Planned                            |
| RAG integration            | Planned                            |
| STT/TTS integration        | Planned                            |

## Current Training Baseline

The first GPT training experiment uses a character-level tokenizer with a vocabulary of 53 characters.

| Metric                |             Result |
| --------------------- | -----------------: |
| Configuration         | `configs/10m.json` |
| Actual parameters     |          5,273,088 |
| Tokenizer             |    Character-level |
| Vocabulary size       |                 53 |
| Context length        |                256 |
| Training steps        |              1,000 |
| Device                |               CUDA |
| Precision             |               FP16 |
| Final training loss   |              ~1.89 |
| Final validation loss |              ~1.84 |

**Important:** The configuration is named `10m.json`, but the current architecture contains approximately 5.27 million parameters. It is not yet a true 10M-parameter model.

### Current generation example

Prompt:

```text
The World of Knowledge
```

Generated text:

```text
The World of Knowledge. The wat they licker sthe apled ank he woors...
```

The model successfully generates text and has learned some character and word patterns, but the output is not yet coherent.

This is expected for an early character-level experiment with limited training data and compute.

### Next improvement

The next milestone is to implement BPE tokenization, prepare a larger and more diverse dataset, and retrain the model. Generation quality and validation loss will be evaluated again.

## Important Hardware Reality

- The 10M/30M/70M/100M names are target model sizes, not guaranteed exact parameter counts.
- Run the parameter-count script to inspect the actual architecture size:

```powershell
python -m scripts.count_parameters --config configs/10m.json
```

- A 4 GB GPU may require small microbatches and shorter context lengths.
- The 70M/100M configurations are experiments and stress tests, not promises of practical training speed.
- Gradient accumulation reduces the microbatch activation requirement but does not eliminate model or optimizer memory costs.
- Start with the demo configuration before attempting larger models.
- Loss reduction alone does not prove that a model has learned useful language ability.
- Training from scratch is not equivalent to using a pretrained assistant. Results depend heavily on dataset size, data quality, tokenization, model architecture, and compute.

## Hardware and Development Environment

| Component        | Configuration               |
| ---------------- | --------------------------- |
| Operating System | Windows 10                  |
| Python           | 3.10.11                     |
| Framework        | PyTorch                     |
| GPU              | NVIDIA RTX 3050A, 4 GB VRAM |
| CPU              | Intel Core i5 H-series      |
| RAM              | 16 GB                       |
| CUDA             | CUDA-enabled PyTorch        |
| Development      | VS Code                     |

## Setup (Windows PowerShell)

### 1. Create a virtual environment

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
```

### 2. Install PyTorch

Install a PyTorch build compatible with your NVIDIA driver and CUDA runtime.

Official installation selector:

https://pytorch.org/get-started/locally/

Example CUDA 12.6 installation:

```powershell
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

### 3. Install project dependencies

```powershell
python -m pip install -r requirements.txt
```

If optional speech or RAG dependencies fail on Windows, install the core dependencies first:

```powershell
python -m pip install numpy tokenizers tensorboard tqdm
```

Install optional packages separately when needed.

### 4. Verify the environment

```powershell
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Not available')"
```

## Quick Start

### 1. Prepare the dataset

Place UTF-8 `.txt` or `.md` files inside:

```text
data/
└── raw/
    └── input.txt
```

Prepare the character-level dataset:

```powershell
python -m scripts.prepare_data --input data/raw --output data/processed --tokenizer char
```

### 2. Train a demo model

```powershell
python -m scripts.train --config configs/demo.json
```

### 3. Train the current model configuration

```powershell
python -m scripts.train --config configs/10m.json
```

### 4. Evaluate the checkpoint

```powershell
python -m scripts.evaluate --checkpoint checkpoints/10m/latest.pt
```

### 5. Generate text

```powershell
python -m scripts.generate --checkpoint checkpoints/10m/latest.pt --prompt "The World of Knowledge"
```

### 6. Inspect model parameters

```powershell
python -m scripts.count_parameters --config configs/10m.json
```

## Tokenization

The project explores two tokenization approaches.

### Character-level tokenizer

- Each character is represented by a token.
- Simple to understand and implement.
- Small vocabulary.
- Requires more tokens to represent words and sentences.
- Useful for learning the fundamentals of language modelling.

### BPE tokenizer

- Learns frequently occurring character sequences as tokens.
- Represents common words and subwords more efficiently.
- Supports larger text corpora with a more compact token sequence.
- Planned as the next tokenizer milestone.

Example BPE dataset preparation command, when supported by the current preparation script:

```powershell
python -m scripts.prepare_data --input data/raw --output data/processed --tokenizer bpe --vocab-size 8000
```

**Important:** Use the same tokenizer and tokenizer files when preparing data, training, evaluating, and generating text. A checkpoint must be loaded with its matching tokenizer.

## Dataset and Licensing

Only use datasets and documents that you are legally allowed to use.

Recommended dataset considerations:

- Prefer public-domain or appropriately licensed datasets.
- Keep the original dataset source and license information.
- Record dataset version, document count, preprocessing details, and split information.
- Keep validation documents separate before tokenization to avoid data leakage.

The processed dataset should contain:

```text
data/
├── raw/
└── processed/
    ├── train.bin
    ├── val.bin
    ├── manifest.json
    └── tokenizer files
```

For serious experiments, split data at the document or source level before packing tokens. This helps prevent overlapping training and validation content.

## Training Controls

Training configurations are stored in JSON files under `configs/`.

Important configuration parameters:

| Parameter                     | Purpose                                      |
| ----------------------------- | -------------------------------------------- |
| `batch_size`                  | Number of sequences per microbatch           |
| `gradient_accumulation_steps` | Accumulate gradients across microbatches     |
| `block_size`                  | Context length                               |
| `max_steps`                   | Maximum training steps                       |
| `eval_interval`               | Evaluation frequency                         |
| `save_interval`               | Checkpoint saving frequency                  |
| `learning_rate`               | Optimizer learning rate                      |
| `weight_decay`                | Weight regularization                        |
| `precision`                   | Training precision                           |
| `gradient_checkpointing`      | Reduce activation memory                     |
| Architecture settings         | Model dimensions and attention configuration |

The training pipeline is intended to support:

- Checkpoint saving and resuming.
- Model and optimizer state persistence.
- Configuration and RNG-state persistence.
- Training and validation loss tracking.
- CSV and TensorBoard logging.

Resume training when supported by the configuration and checkpoint:

```powershell
python -m scripts.train --config configs/10m.json --resume
```

## Main Commands

```powershell
# Environment
python -m scripts.check_environment

# Dataset preparation
python -m scripts.prepare_data --input data/raw --output data/processed --tokenizer char

# Parameter count
python -m scripts.count_parameters --config configs/10m.json

# Training
python -m scripts.train --config configs/demo.json
python -m scripts.train --config configs/10m.json

# Evaluation
python -m scripts.evaluate --checkpoint checkpoints/10m/latest.pt

# Generation
python -m scripts.generate --checkpoint checkpoints/10m/latest.pt --prompt "The future of science"

# Fine-tuning (planned/optional)
python -m scripts.finetune --checkpoint checkpoints/10m/latest.pt --data data/processed

# RAG (planned
```
