"""A small CNN. No BatchNorm on purpose: BN statistics behave badly under non-IID data
(and are awkward to encrypt/aggregate in Week 2)."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class SmallCNN(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 10, image_size: int = 28):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        side = image_size // 4
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * side * side, 128), nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def get_weights(model: nn.Module) -> dict:
    return {k: v.detach().cpu().numpy().copy() for k, v in model.state_dict().items()}


def set_weights(model: nn.Module, weights: dict) -> None:
    state = {k: torch.from_numpy(np.array(v, copy=True)) for k, v in weights.items()}
    model.load_state_dict(state)
