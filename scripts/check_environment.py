import sys, platform
import torch
def main():
    print("Python:", sys.version)
    print("Platform:", platform.platform())
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA runtime:", torch.version.cuda)
        print("GPU:", torch.cuda.get_device_name(0))
        print("VRAM GiB:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))
        x = torch.randn(1024, 1024, device="cuda"); y = x @ x
        torch.cuda.synchronize()
        print("GPU tensor operation:", tuple(y.shape), "OK")
    else:
        print("Running CPU-only. Install a CUDA-enabled PyTorch build if you expect GPU training.")
if __name__ == "__main__": main()
