"""Modeling for models training loop"""

import torch as _torch
from torch import nn as _nn
from torch.nn import functional as _F
from torch.utils.data import DataLoader as _DataLoader
from torch.optim import Optimizer as _Optimizer
from torch.optim.lr_scheduler import LRScheduler as _LRScheduler
from torchmetrics.classification import MulticlassF1Score as _MulticlassF1Score
from tqdm import tqdm as _tqdm


class Trainer:
    """Trainer class for automated training loop"""

    def __init__(
        self,
        model: _nn.Module,
        dataloader: _DataLoader,
        optimizer: _Optimizer,
        lr_scheduler: _LRScheduler,
        loss_fn,
        epochs: int,
        device=None,
    ):
        self.device = device
        self.model = model.to(self.device)
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_fn = loss_fn
        self.epochs = epochs

    def train(self):
        dataloader_len = len(self.dataloader)
        print(f"training on device={self.device}")

        f1_meter = _MulticlassF1Score(self.model.num_classes)

        for epoch in range(self.epochs):
            batches_bar = _tqdm(self.dataloader, desc=f"epoch={epoch}/{self.epochs}")
            epoch_loss = 0
            epoch_f1 = 0
            for batch in batches_bar:
                self.optimizer.zero_grad()

                features, targets = batch
                features = _torch.as_tensor(features, device=self.device)
                targets = _torch.as_tensor(targets, device=self.device)

                logits = self.model(features)
                loss = self.loss_fn(logits, targets)
                loss.backward()

                self.optimizer.step()
                self.lr_scheduler.step()

                f1_score = f1_meter(_F.softmax(logits, dim=1).argmax(dim=1), targets)

                epoch_loss += loss.item()
                epoch_f1 += f1_score

                batches_bar.set_postfix(
                    {"loss": f"{loss.item():.4f}", "f1": f"{f1_score:.4f}"}
                )

            epoch_loss /= dataloader_len
            epoch_f1 /= dataloader_len

            print(f"\navg_loss={epoch_loss:.4f} avg_f1={epoch_f1:.4f}")

    def save_model(self, path: str):
        with open(path, "w") as f:
            _torch.save(self.model.state_dict(), f)
