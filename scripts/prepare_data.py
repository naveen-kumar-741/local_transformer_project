import argparse, json, hashlib
from pathlib import Path
import numpy as np
from tokenizer.char_tokenizer import CharTokenizer
from tokenizer.bpe_tokenizer import BPETokenizer

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",default="data/raw"); ap.add_argument("--output",default="data/processed")
    ap.add_argument("--tokenizer",choices=["char","bpe"],default="char"); ap.add_argument("--vocab-size",type=int,default=8000)
    ap.add_argument("--val-ratio",type=float,default=0.05)
    a=ap.parse_args(); inp=Path(a.input); out=Path(a.output); tokdir=out/"tokenizer"
    files=sorted([p for p in inp.rglob("*") if p.is_file() and p.suffix.lower() in (".txt",".md")])
    if not files: raise SystemExit(f"No .txt/.md files found under {inp}. Add UTF-8 text files first.")
    texts=[p.read_text(encoding="utf-8",errors="replace") for p in files]
    if len(texts)==1:
        text=texts[0]; cut=max(1,int(len(text)*(1-a.val_ratio))); train_text=text[:cut]; val_text=text[cut:]
        if not val_text: val_text=text[-max(1,len(text)//20):]
    else:
        # Deterministic document-level split; keep at least one document in each partition.
        cut=max(1,min(len(texts)-1,round(len(texts)*(1-a.val_ratio))))
        train_text="\n\n".join(texts[:cut]); val_text="\n\n".join(texts[cut:])
    out.mkdir(parents=True,exist_ok=True); tokdir.mkdir(parents=True,exist_ok=True)
    if a.tokenizer=="char":
        tok=CharTokenizer().fit(train_text); tok.save(tokdir)
    else:
        tok=BPETokenizer.train([str(p) for p in files],a.vocab_size); tok.save(tokdir)
    tr=np.asarray(tok.encode(train_text),dtype=np.uint32); va=np.asarray(tok.encode(val_text),dtype=np.uint32)
    if len(tr)<300 or len(va)<100: print("WARNING: very small dataset; training may overfit or fail for larger block sizes.")
    tr.tofile(out/"train.bin"); va.tofile(out/"val.bin")
    manifest={"tokenizer":a.tokenizer,"vocab_size":tok.vocab_size,"files":[{"path":str(p),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for p in files],"train_tokens":len(tr),"val_tokens":len(va),"val_ratio":a.val_ratio,"note":"Verify document-level split and licensing for serious experiments."}
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(json.dumps(manifest,indent=2))
if __name__=="__main__": main()
