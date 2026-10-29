import argparse
from config import TrainConfig
from training.engine import train
def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",required=True); p.add_argument("--resume",default=None); a=p.parse_args()
    cfg=TrainConfig.load(a.config); print("Saved checkpoint:",train(cfg,a.resume))
if __name__=="__main__": main()
