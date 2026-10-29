# Minimal Phase 1 baseline. Trains a character-level next-token model using the shared GPT implementation.
import argparse
from pathlib import Path
from scripts.prepare_data import main as prepare_main
from scripts.train import main as train_main
if __name__=="__main__":
    print("Use `python -m scripts.prepare_data --tokenizer char` then `python -m scripts.train --config configs/demo.json`.")
