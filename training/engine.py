import csv, math, os, random, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from config import ModelConfig
from model.gpt import GPTLanguageModel, count_parameters
from training.data import TokenDataset

def choose_device(name):
    if name != "auto": return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def estimate_loss(model, ds, cfg, device, amp_dtype=None):
    model.eval(); result = {}
    with torch.no_grad():
        for split in ("train", "val"):
            losses = []
            for _ in range(cfg.eval_batches):
                x, y = ds.get_batch(split, cfg.batch_size, device)
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None and device.type == "cuda"):
                    _, loss = model(x, y)
                losses.append(float(loss.item()))
            result[split] = sum(losses) / max(1, len(losses))
    model.train()
    return result

def load_checkpoint(path, model, optimizer=None, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt: optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt

def train(config, resume=None):
    seed_everything(config.seed)
    device = choose_device(config.device)
    mc = ModelConfig.from_dict(config.model or {})
    mc.gradient_checkpointing = bool(mc.gradient_checkpointing)
    ds = TokenDataset(config.data_dir, mc.block_size)
    model = GPTLanguageModel(mc).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay, betas=(0.9,0.95))
    amp_dtype = torch.float16 if config.precision.lower() == "fp16" and device.type == "cuda" else (torch.bfloat16 if config.precision.lower() == "bf16" and device.type == "cuda" else None)
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16 and device.type == "cuda"))
    start_step = 0
    if resume:
        ckpt = load_checkpoint(resume, model, optimizer, device)
        start_step = int(ckpt.get("step", 0))
        if "rng" in ckpt:
            torch.set_rng_state(ckpt["rng"]["torch"])
            np.random.set_state(ckpt["rng"]["numpy"])
            random.setstate(ckpt["rng"]["python"])
    out = Path(config.out_dir) / config.run_name
    out.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(Path("runs") / config.run_name))
    csv_path = out / "metrics.csv"
    fresh_csv = not csv_path.exists() or start_step == 0
    csv_file = csv_path.open("w" if fresh_csv else "a", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    if fresh_csv: csv_writer.writerow(["step","train_loss","val_loss","lr","tokens_per_sec","elapsed_sec","peak_vram_mb","parameters"])
    params = count_parameters(model)
    print(f"device={device}; parameters={params:,}; block_size={mc.block_size}; precision={config.precision}")
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
    start_time = time.time()
    try:
        model.train()
        for step in range(start_step, config.max_steps):
            if step < config.warmup_steps:
                lr = config.learning_rate * (step + 1) / max(1, config.warmup_steps)
            else:
                progress = (step - config.warmup_steps) / max(1, config.max_steps - config.warmup_steps)
                lr = config.min_learning_rate + 0.5 * (config.learning_rate-config.min_learning_rate) * (1 + math.cos(math.pi * min(1, progress)))
            for group in optimizer.param_groups: group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            loss_total = 0.0
            for _ in range(config.gradient_accumulation_steps):
                x, y = ds.get_batch("train", config.batch_size, device)
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None and device.type == "cuda"):
                    _, loss = model(x, y)
                    loss = loss / config.gradient_accumulation_steps
                scaler.scale(loss).backward()
                loss_total += float(loss.item()) * config.gradient_accumulation_steps
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer); scaler.update()
            global_step = step + 1
            if global_step % config.log_interval == 0:
                elapsed = time.time() - start_time
                tps = global_step * config.batch_size * config.gradient_accumulation_steps * mc.block_size / max(elapsed, 1e-6)
                peak = torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else 0
                print(f"step {global_step}/{config.max_steps} loss={loss_total:.4f} lr={lr:.2e} tok/s~{tps:.1f} peak_vram={peak:.0f}MB")
                writer.add_scalar("train/loss_step", loss_total, global_step)
            if global_step % config.eval_interval == 0 or global_step == 1:
                ev = estimate_loss(model, ds, config, device, amp_dtype)
                elapsed = time.time() - start_time
                tps = global_step * config.batch_size * config.gradient_accumulation_steps * mc.block_size / max(elapsed, 1e-6)
                peak = torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else 0
                csv_writer.writerow([global_step,ev["train"],ev["val"],lr,tps,elapsed,peak,params]); csv_file.flush()
                writer.add_scalar("loss/train_eval", ev["train"], global_step); writer.add_scalar("loss/val", ev["val"], global_step)
                print(f"eval step={global_step} train={ev['train']:.4f} val={ev['val']:.4f}")
            if global_step % config.save_interval == 0 or global_step == config.max_steps:
                ckpt = {"model":model.state_dict(),"optimizer":optimizer.state_dict(),"step":global_step,"model_config":mc.__dict__,"train_config":config.to_dict(),"parameters":params,"rng":{"torch":torch.get_rng_state(),"numpy":np.random.get_state(),"python":random.getstate()}}
                tmp = out / "latest.tmp.pt"; torch.save(ckpt,tmp); os.replace(tmp,out/"latest.pt")
                torch.save(ckpt,out/f"step_{global_step}.pt")
    finally:
        csv_file.close(); writer.close()
    return str(out / "latest.pt")
