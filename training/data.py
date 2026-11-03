from pathlib import Path

import numpy as np
import torch


class TokenDataset:
    def __init__(self, data_dir, block_size):
        p = Path(data_dir)

        self.train = np.memmap(
            p / "train.bin",
            dtype=np.uint32,
            mode="r",
        )

        self.val = np.memmap(
            p / "val.bin",
            dtype=np.uint32,
            mode="r",
        )

        self.block_size = block_size

        if len(self.train) <= block_size or len(self.val) <= block_size:
            raise ValueError(
                "train.bin and val.bin must each contain at least "
                "block_size + 1 tokens."
            )

    def get_batch(self, split, batch_size, device, generator=None):
        if split == "train":
            data = self.train
        elif split == "val":
            data = self.val
        else:
            raise ValueError(f"Invalid split: {split!r}. Expected 'train' or 'val'.")

        # A sequence of block_size input tokens needs one additional
        # token for its shifted target sequence.
        max_start_exclusive = len(data) - self.block_size

        ix = torch.randint(
            low=0,
            high=max_start_exclusive,
            size=(batch_size,),
            generator=generator,
        )

        x = torch.stack(
            [
                torch.from_numpy(
                    np.array(
                        data[i : i + self.block_size],
                        dtype=np.int64,
                    )
                )
                for i in ix.tolist()
            ]
        )

        y = torch.stack(
            [
                torch.from_numpy(
                    np.array(
                        data[i + 1 : i + self.block_size + 1],
                        dtype=np.int64,
                    )
                )
                for i in ix.tolist()
            ]
        )

        return (
            x.to(device, non_blocking=True),
            y.to(device, non_blocking=True),
        )
