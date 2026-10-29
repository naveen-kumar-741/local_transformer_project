import argparse, json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

def chunks(text, size=700, overlap=100):
    text=" ".join(text.split())
    if not text: return []
    out=[]; start=0
    while start<len(text):
        out.append(text[start:start+size])
        if start+size>=len(text): break
        start+=size-overlap
    return out

def main():
    p=argparse.ArgumentParser(); p.add_argument("--docs",default="data/knowledge"); p.add_argument("--index-dir",default="data/rag_index"); p.add_argument("--model",default="sentence-transformers/all-MiniLM-L6-v2"); a=p.parse_args()
    docs=Path(a.docs); files=sorted(x for x in docs.rglob("*") if x.is_file() and x.suffix.lower() in (".txt",".md"))
    records=[]
    for f in files:
        for i,c in enumerate(chunks(f.read_text(encoding="utf-8",errors="replace"))): records.append({"source":str(f),"chunk":i,"text":c})
    if not records: raise SystemExit(f"No text documents found in {docs}")
    model=SentenceTransformer(a.model); emb=model.encode([r["text"] for r in records],normalize_embeddings=True,convert_to_numpy=True).astype("float32")
    index=faiss.IndexFlatIP(emb.shape[1]); index.add(emb)
    out=Path(a.index_dir); out.mkdir(parents=True,exist_ok=True); faiss.write_index(index,str(out/"index.faiss")); np.save(out/"embeddings.npy",emb)
    (out/"chunks.json").write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding="utf-8")
    (out/"metadata.json").write_text(json.dumps({"embedding_model":a.model,"count":len(records),"dimension":emb.shape[1]},indent=2),encoding="utf-8")
    print(f"Indexed {len(records)} chunks")
if __name__=="__main__": main()
