"""Modeling for the dataset used for training/validating"""

import torch as _torch
from torch.utils.data import Dataset as _Dataset, DataLoader as _DataLoader
from torchvision.datasets import ImageFolder as _ImageFolder
from torchvision.transforms import v2 as _transforms

_compose = _transforms.Compose(
    [
        _transforms.ToImage(),
        _transforms.Resize([32, 32]),
        _transforms.ToDtype(dtype=_torch.float32, scale=True),
    ]
)


class AslDataset(_Dataset):
    """Subclass Dataset for ASL dataset"""

    def __init__(self, folder: str, device=None):
        super().__init__()
        self.device = device
        self.dataset = _ImageFolder(root=folder)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset.__getitem__(index)

    def __getitems__(self, indexes):
        return self.dataset.__getitems__(indexes)

    @staticmethod
    def collate_fn(batch):
        features, targets = zip(*batch)
        return (_torch.stack(list(map(_compose, features))), _torch.tensor(targets))

    def dataloader(self, batch_size: int = None, shuffle: bool = False):
        return _DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.collate_fn,
        )
