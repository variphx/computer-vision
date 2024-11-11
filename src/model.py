import torch
import torchmetrics
import lightning as L
from torch import nn
from torch.nn import functional as F
from lightning.pytorch import callbacks
from torchvision.transforms import v2 as transforms


class AslTranslator(L.LightningModule):
    def __init__(
        self,
        input_dim: tuple[int, int, int],
        output_dim: int,
        lr=5e-5,
        optimizer_cls: type[torch.optim.Optimizer] = torch.optim.SGD,
        lr_scheduler_cls: type[
            torch.optim.lr_scheduler.LRScheduler
        ] = torch.optim.lr_scheduler.CosineAnnealingLR,
        lr_scheduler_kwargs: dict[str, any] = {},
    ):
        super().__init__()
        self.save_hyperparameters()

        channels, _, _ = input_dim
        decoy_tensor = torch.zeros(input_dim)

        conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels * 2,
                kernel_size=7,
            ),
            nn.AvgPool2d(kernel_size=7),
            nn.GELU(),
            nn.Dropout(),
        )

        conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=channels * 2,
                out_channels=channels * 4,
                kernel_size=7,
            ),
            nn.AvgPool2d(kernel_size=7),
            nn.GELU(),
            nn.Dropout(),
        )

        conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=channels * 4,
                out_channels=channels * 8,
                kernel_size=7,
            ),
            nn.AvgPool2d(kernel_size=7),
            nn.GELU(),
            nn.Dropout(),
        )

        self.conv = nn.Sequential(conv1, conv2, conv3)
        decoy_tensor: torch.Tensor = self.conv(decoy_tensor)
        decoy_tensor = torch.flatten(decoy_tensor, start_dim=1)
        hidden_size = decoy_tensor.size(1) // 2

        linear1 = nn.Sequential(
            nn.Linear(in_features=decoy_tensor.size(1), out_features=hidden_size),
            nn.GELU(),
            nn.Dropout(),
        )

        linear2 = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.GELU(),
            nn.Dropout(),
        )

        linear3 = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.GELU(),
            nn.Dropout(),
        )

        self.linear = nn.Sequential(linear1, linear2, linear3)
        decoy_tensor: torch.Tensor = self.linear(decoy_tensor)

        self.classifier = nn.Linear(
            in_features=decoy_tensor.size(1), out_features=output_dim
        )

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.linear(x)
        logits = self.classifier(x)
        return logits

    def training_step(self, examples, _):
        features: torch.Tensor = examples["images"]
        targets: torch.Tensor = examples["labels"]

        logits: torch.Tensor = self(features)

        loss = F.cross_entropy(logits, targets)

        preds = logits.argmax(1)
        f1 = torchmetrics.functional.f1_score(preds, targets, task="multiclass")
        acc = torchmetrics.functional.accuracy(preds, targets, task="multiclass")
        recall = torchmetrics.functional.recall(preds, targets, task="multiclass")

        self.log_dict(
            {
                "train_loss": loss.item(),
                "train_f1": f1.item(),
                "train_acc": acc.item(),
                "train_recall": recall.item(),
            },
            prog_bar=True,
            batch_size=targets.size(0),
        )

        return loss

    def validation_step(self, examples, _):
        features: torch.Tensor = examples["images"]
        targets: torch.Tensor = examples["labels"]
        logits: torch.Tensor = self(features)

        loss = F.cross_entropy(logits, targets)

        preds = logits.argmax(1)
        f1 = torchmetrics.functional.f1_score(preds, targets, task="multiclass")
        acc = torchmetrics.functional.accuracy(preds, targets, task="multiclass")
        recall = torchmetrics.functional.recall(preds, targets, task="multiclass")

        self.log_dict(
            {
                "val_loss": loss.item(),
                "val_f1": f1.item(),
                "val_acc": acc.item(),
                "val_recall": recall.item(),
            },
            prog_bar=True,
            batch_size=targets.size(0),
        )

        return loss

    def predict_step(self, examples, index):
        raise NotImplementedError()

    def configure_optimizers(self):
        optimizer_cls: type[torch.optim.Optimizer] = self.hparams.get("optimizer_cls")
        lr_scheduler_cls: type[torch.optim.lr_scheduler.LRScheduler] = self.hparams.get(
            "lr_scheduler_cls"
        )

        lr: float = self.hparams.get("lr")
        lr_scheduler_kwargs: dict[str, any] = self.hparams.get("lr_scheduler_kwargs")

        optimizer = optimizer_cls(self.parameters(), lr=lr)
        lr_scheduler = lr_scheduler_cls(optimizer, **lr_scheduler_kwargs)

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def configure_callbacks(self):
        swa = callbacks.StochasticWeightAveraging(swa_lrs=1e-2)
        checkpoint = callbacks.ModelCheckpoint(
            monitor="val_loss", mode="min", save_top_k=5
        )
        return [swa, checkpoint]
