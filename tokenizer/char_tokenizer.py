import json
from pathlib import Path

class CharTokenizer:
    def __init__(self, vocab=None):
        self.vocab = vocab or []
        self.stoi = {ch: i for i, ch in enumerate(self.vocab)}
        self.unk_id = self.stoi.get("�", 0)

    def fit(self, text: str):
        self.vocab = sorted(set(text))
        if "�" not in self.vocab:
            self.vocab.insert(0, "�")
        self.stoi = {ch: i for i, ch in enumerate(self.vocab)}
        self.unk_id = self.stoi["�"]
        return self

    @property
    def vocab_size(self):
        return len(self.vocab)

    def encode(self, text: str):
        return [self.stoi.get(ch, self.unk_id) for ch in text]

    def decode(self, ids):
        return "".join(self.vocab[int(i)] if 0 <= int(i) < len(self.vocab) else "�" for i in ids)

    def save(self, directory):
        p = Path(directory); p.mkdir(parents=True, exist_ok=True)
        (p / "char_tokenizer.json").write_text(json.dumps({"type":"char","vocab":self.vocab}, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, directory):
        d = json.loads((Path(directory) / "char_tokenizer.json").read_text(encoding="utf-8"))
        return cls(d["vocab"])
