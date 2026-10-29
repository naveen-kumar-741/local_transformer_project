from pathlib import Path
import numpy as np
import torch

class TokenDataset:
    def __init__(self, data_dir, block_size):
        p = Path(data_dir)
        self.train = np.memmap(p / "train.bin", dtype=np.uint32, mode="r")
        self.val = np.memmap(p / "val.bin", dtype=np.uint32, mode="r")
        self.block_size = block_size
        if len(self.train) <= block_size + 1 or len(self.val) <= block_size + 1:
            raise ValueError("train.bin and val.bin must each contain more tokens than block_size + 1")

    def get_batch(self, split, batch_size, device):
        data = self.train if split == "train" else self.val
        ix = torch.randint(len(data) - self.block_size - 1, (batch_size,))
        x = torch.stack([torch.from_numpy(np.array(data[i:i+self.block_size], dtype=np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(np.array(data[i+1:i+self.block_size+1], dtype=np.int64)) for i in ix])
        return x.to(device, non_blocking=True), y.to(device, non_blocking=True)
