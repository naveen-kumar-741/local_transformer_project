# Supervised next-token fine-tuning from an existing checkpoint on a prepared text dataset.
import argparse, torch
from config import TrainConfig, ModelConfig
from model.gpt import GPTLanguageModel
from training.engine import train
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",required=True); p.add_argument("--data",default="data/processed"); p.add_argument("--steps",type=int,default=200); p.add_argument("--run-name",default="finetuned"); a=p.parse_args()
    ck=torch.load(a.checkpoint,map_location="cpu",weights_only=False)
    tc=TrainConfig(**{k:v for k,v in ck["train_config"].items() if k in TrainConfig.__dataclass_fields__})
    tc.data_dir=a.data; tc.run_name=a.run_name; tc.max_steps=a.steps; tc.learning_rate=min(tc.learning_rate,1e-4); tc.out_dir="checkpoints"
    # train() initializes from random weights; load the base weights into a temporary-compatible training path.
    # To avoid silently pretending base weights were loaded, require explicit resume-compatible checkpoint.
    raise SystemExit("For true fine-tuning, use the base checkpoint as --resume with a compatible run config: python -m scripts.train --config <finetune-config.json> --resume <checkpoint>. This wrapper is intentionally guarded to prevent accidental training from random initialization.")
if __name__=="__main__": main()
