"""Modeling of ML model to be used in training/inferencing"""

from torch import nn as _nn
from torch.nn import functional as _F


class AslClassifier(_nn.Module):
    """Convolutional neural network for classifying ASL signatures"""

    def __init__(
        self,
        input_dim: tuple[int, int, int],
        num_classes: int,
        kernel_size: int,
        device=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.device = device
        self.num_classes = num_classes

        self.layer_norm = _nn.LayerNorm(
            normalized_shape=input_dim,
            device=device,
        )

        self.conv = _nn.Conv2d(
            input_dim[0],
            16,
            kernel_size=kernel_size,
            device=device,
        )

        self.conv1 = _nn.Conv2d(
            16,
            32,
            kernel_size=kernel_size,
            device=device,
        )

        self.conv2 = _nn.Conv2d(
            32,
            64,
            kernel_size=kernel_size,
            device=device,
        )

        self.avg_pool = _nn.AdaptiveAvgPool2d([32, 32])
        self.flatten = _nn.Flatten()

        hidden_size = self.conv2.out_channels * 32 * 32
        self.linear = _nn.Linear(hidden_size, 1024, device=device)
        self.linear1 = _nn.Linear(1024, 256, device=device)
        self.linear2 = _nn.Linear(256, 64, device=device)
        self.linear3 = _nn.Linear(64, num_classes, device=device)

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.conv(x)
        x = _F.relu(x)
        x = self.conv1(x)
        x = _F.relu(x)
        x = self.conv2(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = _F.relu(x)
        x = self.linear(x)
        x = _F.relu(x)
        x = self.linear1(x)
        x = _F.relu(x)
        x = self.linear2(x)
        x = _F.relu(x)
        x = self.linear3(x)
        return x
