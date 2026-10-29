import argparse, json
from pathlib import Path
import faiss
from sentence_transformers import SentenceTransformer
import torch
from config import ModelConfig
from model.gpt import GPTLanguageModel
from tokenizer.loader import load_tokenizer

def main():
    p=argparse.ArgumentParser(); p.add_argument("--checkpoint",required=True); p.add_argument("--index-dir",default="data/rag_index"); p.add_argument("--tokenizer-dir",default=None); p.add_argument("--question",default=None); p.add_argument("--top-k",type=int,default=3); a=p.parse_args()
    d=Path(a.index_dir); index=faiss.read_index(str(d/"index.faiss")); records=json.loads((d/"chunks.json").read_text(encoding="utf-8")); meta=json.loads((d/"metadata.json").read_text(encoding="utf-8"))
    embedder=SentenceTransformer(meta["embedding_model"]); ck=torch.load(a.checkpoint,map_location="cpu",weights_only=False); mc=ModelConfig.from_dict(ck["model_config"]); model=GPTLanguageModel(mc); model.load_state_dict(ck["model"]); model.eval()
    tokdir=a.tokenizer_dir or ck.get("train_config",{}).get("tokenizer_dir","data/processed/tokenizer"); tok=load_tokenizer(tokdir); device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model.to(device)
    while True:
        q=a.question if a.question is not None else input("Question (or exit): ")
        if q.strip().lower() in ("exit","quit"): break
        vec=embedder.encode([q],normalize_embeddings=True,convert_to_numpy=True).astype("float32"); scores, ids=index.search(vec,a.top_k)
        context="\n".join(f"[{records[i]['source']}] {records[i]['text']}" for i in ids[0] if i>=0)
        prompt=f"Context:\n{context}\n\nQuestion: {q}\nAnswer:"
        ids_in=tok.encode(prompt)[-mc.block_size:]
        out=model.generate(torch.tensor([ids_in],device=device),max_new_tokens=120,temperature=.7,top_k=30)[0].tolist()
        print(tok.decode(out)[len(tok.decode(ids_in)):].strip())
        if a.question is not None: break
if __name__=="__main__": main()
