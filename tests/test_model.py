import torch
from config import ModelConfig
from model.gpt import GPTLanguageModel, count_parameters

def test_forward_and_loss():
    cfg=ModelConfig(vocab_size=32,block_size=16,n_layer=2,n_head=2,n_embd=32,dropout=0)
    model=GPTLanguageModel(cfg)
    x=torch.randint(0,32,(2,12)); y=torch.randint(0,32,(2,12))
    logits,loss=model(x,y)
    assert logits.shape==(2,12,32)
    assert loss.ndim==0 and torch.isfinite(loss)
    loss.backward()
    assert count_parameters(model)>0

def test_generation_length():
    cfg=ModelConfig(vocab_size=32,block_size=16,n_layer=1,n_head=2,n_embd=16,dropout=0)
    model=GPTLanguageModel(cfg)
    out=model.generate(torch.zeros((1,4),dtype=torch.long),max_new_tokens=3)
    assert out.shape==(1,7)
