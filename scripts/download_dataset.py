from datasets import load_dataset
from pathlib import Path
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
output_dir = PROJECT_ROOT / "data" / "raw"
output_dir.mkdir(parents=True, exist_ok=True)

print("Downloading TinyStories...")

train = load_dataset("roneneldan/TinyStories", split="train[:20000]")

output_file = output_dir / "input.txt"

with output_file.open("w", encoding="utf-8") as file:
    for item in train:
        text = item.get("text", "").strip()

        if text:
            file.write(text + "\n\n")

print(f"Saved {len(train)} stories to {output_file}")
