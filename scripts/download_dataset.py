from pathlib import Path
from datasets import load_dataset

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw"

# Dataset settings
TARGET_SIZE_MB = 100
TARGET_BYTES = TARGET_SIZE_MB * 1024 * 1024
CHUNK_BYTES = 1 * 1024 * 1024

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Remove files created by previous runs of this downloader
for old_file in OUTPUT_DIR.glob("tinystories_*.txt"):
    old_file.unlink()

print("Loading TinyStories dataset...")

dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

total_bytes = 0
file_number = 0
buffer = bytearray()


def save_chunk(data: bytearray, number: int) -> None:
    output_file = OUTPUT_DIR / f"tinystories_{number:04d}.txt"
    output_file.write_bytes(data)
    print(f"Saved: {output_file.name} ({len(data) / (1024 * 1024):.2f} MB)")


for row in dataset:
    story = row.get("text", "")

    if not isinstance(story, str) or not story.strip():
        continue

    story_bytes = (story.strip() + "\n\n").encode("utf-8")

    remaining = TARGET_BYTES - total_bytes

    if remaining <= 0:
        break

    # Do not exceed the requested dataset size.
    story_bytes = story_bytes[:remaining]

    # Avoid ending with an incomplete UTF-8 character.
    story_bytes = story_bytes.decode("utf-8", errors="ignore").encode("utf-8")

    if not story_bytes:
        break

    buffer.extend(story_bytes)
    total_bytes += len(story_bytes)

    if len(buffer) >= CHUNK_BYTES:
        file_number += 1
        save_chunk(buffer, file_number)
        buffer = bytearray()

    if total_bytes >= TARGET_BYTES:
        break

if buffer:
    file_number += 1
    save_chunk(buffer, file_number)

print("\nDataset download completed.")
print(f"Total size: {total_bytes / (1024 * 1024):.2f} MB")
print(f"Total files: {file_number}")
print(f"Output directory: {OUTPUT_DIR}")
