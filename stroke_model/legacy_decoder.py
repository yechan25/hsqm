"""Original FPN decoder, retained for existing checkpoint inference."""
from typing import Sequence, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class FPNDecoder(nn.Module):
    def __init__(self, in_channels: Sequence[int], out_channels: int, fpn_ch: int = 256):
        super().__init__()
        c1, c2, c3, c4 = in_channels
        self.lateral4 = nn.Conv2d(c4, fpn_ch, 1)
        self.lateral3 = nn.Conv2d(c3, fpn_ch, 1)
        self.lateral2 = nn.Conv2d(c2, fpn_ch, 1)
        self.lateral1 = nn.Conv2d(c1, fpn_ch, 1)

        self.smooth3 = nn.Sequential(nn.Conv2d(fpn_ch, fpn_ch, 3, padding=1), nn.BatchNorm2d(fpn_ch), nn.ReLU(inplace=True))
        self.smooth2 = nn.Sequential(nn.Conv2d(fpn_ch, fpn_ch, 3, padding=1), nn.BatchNorm2d(fpn_ch), nn.ReLU(inplace=True))
        self.smooth1 = nn.Sequential(nn.Conv2d(fpn_ch, fpn_ch, 3, padding=1), nn.BatchNorm2d(fpn_ch), nn.ReLU(inplace=True))

        self.head = nn.Sequential(
            nn.Conv2d(fpn_ch, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_channels, 1),
        )

    def forward(self, feats: Sequence[torch.Tensor], out_size: Tuple[int, int]) -> torch.Tensor:
        f1, f2, f3, f4 = feats

        p4 = self.lateral4(f4)
        p3 = self.lateral3(f3) + F.interpolate(p4, size=f3.shape[-2:], mode="bilinear", align_corners=False)
        p3 = self.smooth3(p3)

        p2 = self.lateral2(f2) + F.interpolate(p3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        p2 = self.smooth2(p2)

        p1 = self.lateral1(f1) + F.interpolate(p2, size=f1.shape[-2:], mode="bilinear", align_corners=False)
        p1 = self.smooth1(p1)

        p = F.interpolate(p1, size=out_size, mode="bilinear", align_corners=False)
        return self.head(p)
