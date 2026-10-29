import argparse, torch
from config import ModelConfig, TrainConfig
from model.gpt import GPTLanguageModel
from training.data import TokenDataset
from training.engine import choose_device, estimate_loss
def main():
    p=argparse.ArgumentParser(); p.add_argument("--checkpoint",required=True); p.add_argument("--batches",type=int,default=20); a=p.parse_args()
    ck=torch.load(a.checkpoint,map_location="cpu",weights_only=False); tc=TrainConfig(**{k:v for k,v in ck["train_config"].items() if k in TrainConfig.__dataclass_fields__})
    tc.eval_batches=a.batches; mc=ModelConfig.from_dict(ck["model_config"]); d=choose_device(tc.device); m=GPTLanguageModel(mc).to(d); m.load_state_dict(ck["model"])
    ds=TokenDataset(tc.data_dir,mc.block_size); print(estimate_loss(m,ds,tc,d))
if __name__=="__main__": main()
