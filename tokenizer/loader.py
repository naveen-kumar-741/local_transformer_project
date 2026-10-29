from pathlib import Path
from tokenizer.char_tokenizer import CharTokenizer
from tokenizer.bpe_tokenizer import BPETokenizer

def load_tokenizer(directory):
    p = Path(directory)
    if (p / "char_tokenizer.json").exists():
        return CharTokenizer.load(p)
    if (p / "bpe_tokenizer.json").exists():
        return BPETokenizer.load(p)
    raise FileNotFoundError(f"No tokenizer file found in {p}")
