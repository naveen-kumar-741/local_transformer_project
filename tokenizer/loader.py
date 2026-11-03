from pathlib import Path

from tokenizer.char_tokenizer import CharTokenizer
from tokenizer.bpe_tokenizer import BPETokenizer


def load_tokenizer(directory, tokenizer_type=None):
    path = Path(directory)

    if not path.exists():
        raise FileNotFoundError(f"Tokenizer directory does not exist: {path}")

    char_file = path / "char_tokenizer.json"
    bpe_file = path / "bpe_tokenizer.json"

    if tokenizer_type == "char":
        if not char_file.exists():
            raise FileNotFoundError(f"Character tokenizer not found: {char_file}")
        return CharTokenizer.load(path)

    if tokenizer_type == "bpe":
        if not bpe_file.exists():
            raise FileNotFoundError(f"BPE tokenizer not found: {bpe_file}")
        return BPETokenizer.load(path)

    available = []

    if char_file.exists():
        available.append("char")

    if bpe_file.exists():
        available.append("bpe")

    if len(available) == 1:
        return load_tokenizer(path, available[0])

    if len(available) > 1:
        raise ValueError(
            "Multiple tokenizer files found. "
            "Specify tokenizer_type='char' or tokenizer_type='bpe'."
        )

    raise FileNotFoundError(f"No supported tokenizer found in {path}")
