from pathlib import Path

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders


class BPETokenizer:
    SPECIAL_TOKENS = ["[UNK]", "[BOS]", "[EOS]"]

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    @property
    def unk_id(self) -> int:
        return self.tokenizer.token_to_id("[UNK]") or 0

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text).ids

    def decode(self, ids) -> str:
        return self.tokenizer.decode([int(token_id) for token_id in ids])

    def save(self, directory):
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(str(path / "bpe_tokenizer.json"))

    @classmethod
    def train(cls, text: str, vocab_size: int = 8000):
        if not text.strip():
            raise ValueError("Cannot train BPE tokenizer on empty text.")

        tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

        tokenizer.decoder = decoders.ByteLevel()

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=2,
            special_tokens=cls.SPECIAL_TOKENS,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )

        tokenizer.train_from_iterator([text], trainer=trainer)

        return cls(tokenizer)

    @classmethod
    def load(cls, directory):
        path = Path(directory) / "bpe_tokenizer.json"

        if not path.exists():
            raise FileNotFoundError(f"BPE tokenizer not found: {path}")

        return cls(Tokenizer.from_file(str(path)))
