import argparse, json
from config import ModelConfig
from model.gpt import GPTLanguageModel, count_parameters
def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",required=True); a=p.parse_args()
    d=json.load(open(a.config,encoding="utf-8")); mc=ModelConfig.from_dict(d.get("model",{}))
    m=GPTLanguageModel(mc); print(f"Parameters: {count_parameters(m):,} ({count_parameters(m)/1e6:.3f}M)")
if __name__=="__main__": main()
