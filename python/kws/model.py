"""DS-CNN-S from "Hello Edge" (Zhang et al. 2017): ~23k params, ~11M MACs per inference.

Only conv / depthwise conv / pointwise conv / BN / ReLU / global average pool / FC,
so every layer is straightforward to implement in RTL. BN folds into the conv
weights before export.
"""

from torch import nn

from .config import CLASSES


def conv_bn(c_in, c_out, k, stride=1, padding=0, groups=1):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, k, stride, padding, groups=groups, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


class DSCNN(nn.Module):
    def __init__(self, channels=64, n_blocks=4, n_classes=len(CLASSES)):
        super().__init__()
        # Input (1, 49, 40) -> (64, 25, 20)
        self.stem = conv_bn(1, channels, (10, 4), stride=2, padding=(5, 1))
        self.blocks = nn.Sequential(*[
            nn.Sequential(
                conv_bn(channels, channels, 3, padding=1, groups=channels),  # depthwise
                conv_bn(channels, channels, 1),                              # pointwise
            )
            for _ in range(n_blocks)
        ])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels, n_classes)

    def forward(self, x):
        x = self.blocks(self.stem(x))
        return self.fc(self.pool(x).flatten(1))
