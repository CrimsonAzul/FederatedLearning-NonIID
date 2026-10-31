"""The model every run uses: a small CNN (~0.67 M parameters).

There is deliberately no BatchNorm. BatchNorm keeps running statistics of the
data it sees; under non-IID data each client's statistics describe a different
label mix, and averaging them produces statistics that match nobody. That
would be a second, confounding source of non-IID damage (it is what FedBN
addresses), so the comparison stays cleaner without it.
"""
import torch.nn as nn


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                                   # 32x32 -> 16x16
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                                   # 16x16 -> 8x8
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                                   # 8x8 -> 4x4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(),
            nn.Linear(256, num_classes),
        )
        # He initialisation keeps activations from shrinking layer after layer,
        # which matters without normalisation layers.
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
        # A near-zero last layer starts every class at equal probability
        # (loss = ln 10) instead of with large random logits.
        nn.init.normal_(self.classifier[-1].weight, std=0.01)

    def forward(self, x):
        return self.classifier(self.features(x))


def num_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
