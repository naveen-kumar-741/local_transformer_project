import argparse, torch
from config import ModelConfig
from model.gpt import GPTLanguageModel
from tokenizer.loader import load_tokenizer
def main():
    p=argparse.ArgumentParser(); p.add_argument("--checkpoint",required=True); p.add_argument("--prompt",default="Once upon a time"); p.add_argument("--tokens",type=int,default=200); p.add_argument("--temperature",type=float,default=.8); p.add_argument("--top-k",type=int,default=40); p.add_argument("--tokenizer-dir",default=None); a=p.parse_args()
    ck=torch.load(a.checkpoint,map_location="cpu",weights_only=False)
    mc=ModelConfig.from_dict(ck["model_config"]); model=GPTLanguageModel(mc); model.load_state_dict(ck["model"]); model.eval()
    tokdir=a.tokenizer_dir or ck.get("train_config",{}).get("tokenizer_dir","data/processed/tokenizer")
    tok=load_tokenizer(tokdir); device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model.to(device)
    ids=tok.encode(a.prompt)
    if not ids: ids=[0]
    out=model.generate(torch.tensor([ids],device=device),a.tokens,a.temperature,a.top_k)[0].tolist()
    print(tok.decode(out))
if __name__=="__main__": main()
