import argparse
import hashlib
import json
import shutil

from pathlib import Path

import numpy as np

from tokenizer.char_tokenizer import CharTokenizer
from tokenizer.bpe_tokenizer import BPETokenizer


def read_documents(input_dir: Path):
    files = sorted(
        p
        for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in (".txt", ".md")
    )

    if not files:
        raise SystemExit(f"No .txt or .md files found under {input_dir}")

    documents = []

    for file in files:
        documents.append(file.read_text(encoding="utf-8", errors="replace"))

    return files, documents


def split_documents(documents, val_ratio):
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1.")

    if len(documents) == 1:
        text = documents[0]

        cut = int(len(text) * (1 - val_ratio))

        train_text = text[:cut]
        val_text = text[cut:]

    else:
        cut = round(len(documents) * (1 - val_ratio))
        cut = max(1, min(len(documents) - 1, cut))

        train_text = "\n\n".join(documents[:cut])
        val_text = "\n\n".join(documents[cut:])

    if not train_text.strip() or not val_text.strip():
        raise ValueError(
            "Training or validation data is empty. "
            "Add more documents or adjust val_ratio."
        )

    return train_text, val_text


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", default="data/raw")

    parser.add_argument("--output", default="data/processed")

    parser.add_argument("--tokenizer", choices=["char", "bpe"], default="char")

    parser.add_argument("--vocab-size", type=int, default=8000)

    parser.add_argument("--val-ratio", type=float, default=0.05)

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    tokenizer_dir = output_dir / "tokenizer"

    files, documents = read_documents(input_dir)

    train_text, val_text = split_documents(documents, args.val_ratio)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Remove old tokenizer files to prevent mixed tokenizers.
    if tokenizer_dir.exists():
        shutil.rmtree(tokenizer_dir)

    tokenizer_dir.mkdir(parents=True, exist_ok=True)

    print(f"Tokenizer: {args.tokenizer}")
    print(f"Documents: {len(files)}")
    print(f"Training text: {len(train_text):,} characters")
    print(f"Validation text: {len(val_text):,} characters")

    if args.tokenizer == "char":
        tokenizer = CharTokenizer().fit(train_text)

    else:
        tokenizer = BPETokenizer.train(train_text, vocab_size=args.vocab_size)

    tokenizer.save(tokenizer_dir)

    train_tokens = np.asarray(tokenizer.encode(train_text), dtype=np.uint32)

    val_tokens = np.asarray(tokenizer.encode(val_text), dtype=np.uint32)

    if len(train_tokens) < 300 or len(val_tokens) < 100:
        raise ValueError(
            "Dataset is too small for reliable training. "
            "Increase the dataset or adjust the validation split."
        )

    train_tokens.tofile(output_dir / "train.bin")
    val_tokens.tofile(output_dir / "val.bin")

    manifest = {
        "tokenizer": args.tokenizer,
        "vocab_size": tokenizer.vocab_size,
        "requested_vocab_size": (args.vocab_size if args.tokenizer == "bpe" else None),
        "files": [
            {"path": str(file), "sha256": hashlib.sha256(file.read_bytes()).hexdigest()}
            for file in files
        ],
        "train_tokens": len(train_tokens),
        "val_tokens": len(val_tokens),
        "train_characters": len(train_text),
        "val_characters": len(val_text),
        "val_ratio": args.val_ratio,
        "note": (
            "Tokenizer trained only on training text. "
            "For rigorous experiments, split at source/document level."
        ),
    }

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print("\nDataset preparation completed.")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
