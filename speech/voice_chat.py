import argparse, tempfile
from pathlib import Path
import torch
import sounddevice as sd
import soundfile as sf
import pyttsx3
from faster_whisper import WhisperModel
from config import ModelConfig
from model.gpt import GPTLanguageModel
from tokenizer.loader import load_tokenizer

def main():
    p=argparse.ArgumentParser(); p.add_argument("--checkpoint",required=True); p.add_argument("--tokenizer-dir",default=None); p.add_argument("--seconds",type=int,default=6); p.add_argument("--stt-model",default="base"); a=p.parse_args()
    ck=torch.load(a.checkpoint,map_location="cpu",weights_only=False); mc=ModelConfig.from_dict(ck["model_config"]); model=GPTLanguageModel(mc); model.load_state_dict(ck["model"]); model.eval()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model.to(device)
    tok=load_tokenizer(a.tokenizer_dir or ck.get("train_config",{}).get("tokenizer_dir","data/processed/tokenizer"))
    print("Loading Whisper model (first run may download model files)...")
    stt=WhisperModel(a.stt_model,device="cuda" if torch.cuda.is_available() else "cpu",compute_type="float16" if torch.cuda.is_available() else "int8")
    tts=pyttsx3.init()
    while True:
        input(f"Press Enter to record {a.seconds}s, or Ctrl+C to quit...")
        audio=sd.rec(int(a.seconds*16000),samplerate=16000,channels=1,dtype="float32"); sd.wait()
        with tempfile.NamedTemporaryFile(suffix=".wav",delete=False) as f: path=f.name
        sf.write(path,audio,16000)
        segments,_=stt.transcribe(path); question=" ".join(s.text.strip() for s in segments).strip(); Path(path).unlink(missing_ok=True)
        if not question: print("No speech recognized; try again."); continue
        print("You:",question)
        prompt=f"User: {question}\nAssistant:"
        encoded=tok.encode(prompt)[-mc.block_size:]
        output=model.generate(torch.tensor([encoded],device=device),max_new_tokens=100,temperature=.8,top_k=30)[0].tolist()
        answer=tok.decode(output)[len(tok.decode(encoded)):].strip() or "I could not generate a response."
        print("Model:",answer); tts.say(answer); tts.runAndWait()
if __name__=="__main__": main()
