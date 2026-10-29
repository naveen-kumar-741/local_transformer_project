from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

class BPETokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @property
    def vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def encode(self, text):
        return self.tokenizer.encode(text).ids

    def decode(self, ids):
        return self.tokenizer.decode([int(x) for x in ids])

    def save(self, directory):
        p = Path(directory); p.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(str(p / "bpe_tokenizer.json"))

    @classmethod
    def train(cls, files, vocab_size=8000):
        tok = Tokenizer(models.BPE(unk_token="[UNK]"))
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["[UNK]", "[BOS]", "[EOS]"])
        tok.train(files, trainer)
        return cls(tok)

    @classmethod
    def load(cls, directory):
        return cls(Tokenizer.from_file(str(Path(directory) / "bpe_tokenizer.json")))
