"""
SwinWarpHintStrokeModel 정의.

이번 개정에서 바뀐 부분은 decoder뿐입니다:
  FPNDecoder -> BiFPNDecoder (BiFPN 가중 특징 융합 + Coordinate Attention)
Warp Hint / Swin backbone / Cross-Attention / 나머지 구조는 기존과 동일합니다.
"""
from typing import Any, Dict, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# ---------------------------------------------------------------------------
# Warp Hint (기존과 동일)
# ---------------------------------------------------------------------------

class LightweightWarpHintGenerator(nn.Module):
    """[I_g || I_prep] -> pixel flow -> warp(I_G_strokes). 마지막 conv zero-init."""

    def __init__(self, max_flow_px: float = 12.0):
        super().__init__()
        self.max_flow_px = max_flow_px
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )
        self.head1 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(inplace=True))
        self.head2 = nn.Conv2d(64, 2, 3, padding=1)
        nn.init.zeros_(self.head2.weight)
        nn.init.zeros_(self.head2.bias)

    def forward(self, I_g: torch.Tensor, I_prep: torch.Tensor) -> torch.Tensor:
        x = torch.cat([I_g, I_prep], dim=1)
        feat = self.encoder(x)
        flow = self.head2(self.head1(feat))
        flow = F.interpolate(flow, size=I_prep.shape[-2:], mode="bilinear", align_corners=False)
        return torch.tanh(flow) * self.max_flow_px


def make_base_grid(B: int, H: int, W: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    ys, xs = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device, dtype=dtype),
        torch.linspace(-1, 1, W, device=device, dtype=dtype),
        indexing="ij",
    )
    grid = torch.stack([xs, ys], dim=-1)
    return grid.unsqueeze(0).repeat(B, 1, 1, 1)


def warp_by_pixel_flow(x: torch.Tensor, flow_px: torch.Tensor) -> torch.Tensor:
    B, C, H, W = x.shape
    grid = make_base_grid(B, H, W, x.device, x.dtype)
    flow_x = flow_px[:, 0] * (2.0 / max(W - 1, 1))
    flow_y = flow_px[:, 1] * (2.0 / max(H - 1, 1))
    flow = torch.stack([flow_x, flow_y], dim=-1)
    sample_grid = grid + flow
    return F.grid_sample(x, sample_grid, mode="bilinear", padding_mode="zeros", align_corners=True)


# ---------------------------------------------------------------------------
# Reference / Cross-Attention (기존과 동일)
# ---------------------------------------------------------------------------

class ReferenceEncoder(nn.Module):
    def __init__(self, out_ch: int = 768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 3, stride=2, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, out_ch, 3, stride=2, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BottleneckCrossAttention(nn.Module):
    def __init__(self, channels: int, heads: int = 8):
        super().__init__()
        self.norm_q = nn.LayerNorm(channels)
        self.norm_kv = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.gamma = nn.Parameter(torch.tensor(0.0))

    def forward(self, q_feat: torch.Tensor, ref_feat: torch.Tensor) -> torch.Tensor:
        B, C, H, W = q_feat.shape
        q = q_feat.flatten(2).transpose(1, 2)
        kv = ref_feat.flatten(2).transpose(1, 2)
        out, _ = self.attn(self.norm_q(q), self.norm_kv(kv), self.norm_kv(kv), need_weights=False)
        out = out.transpose(1, 2).reshape(B, C, H, W)
        return q_feat + self.gamma * out


def ensure_nchw(feat: torch.Tensor) -> torch.Tensor:
    """timm Swin features_only가 버전에 따라 NHWC를 줄 수 있어 NCHW로 통일."""
    if feat.ndim != 4:
        raise ValueError(f"Expected 4D feature, got {feat.shape}")
    if feat.shape[1] <= 64 and feat.shape[-1] > feat.shape[1]:
        return feat.permute(0, 3, 1, 2).contiguous()
    return feat.contiguous()


# ---------------------------------------------------------------------------
# NEW: Coordinate Attention
# (Hou et al., "Coordinate Attention for Efficient Mobile Network Design", CVPR 2021)
# 얇은 획의 위치 정보를 채널 attention에 명시적으로 반영하기 위해 도입.
# ---------------------------------------------------------------------------

class CoordAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 32):
        super().__init__()
        mip = max(8, channels // reduction)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1 = nn.Conv2d(channels, mip, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish(inplace=True)
        self.conv_h = nn.Conv2d(mip, channels, kernel_size=1)
        self.conv_w = nn.Conv2d(mip, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        n, c, h, w = x.shape
        x_h = self.pool_h(x)                       # (n, c, h, 1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)    # (n, c, w, 1)
        y = torch.cat([x_h, x_w], dim=2)            # (n, c, h+w, 1)
        y = self.act(self.bn1(self.conv1(y)))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))
        return identity * a_h * a_w


# ---------------------------------------------------------------------------
# NEW: BiFPN (가중 양방향 특징 융합), EfficientDet 스타일.
# 작은 데이터셋에서 과적합 위험을 줄이기 위해 레이어 반복 횟수(num_layers)를
# 기본 2로 제한. 필요하면 config에서 조절.
# ---------------------------------------------------------------------------

def _resize_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)


class _FastFusion(nn.Module):
    """ReLU 기반 fast-normalized weighted fusion (EfficientDet식)."""

    def __init__(self, n_inputs: int, channels: int, eps: float = 1e-4):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(n_inputs))
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),  # depthwise
            nn.Conv2d(channels, channels, 1),                              # pointwise
            nn.BatchNorm2d(channels),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, *feats: torch.Tensor) -> torch.Tensor:
        w = F.relu(self.weight)
        w = w / (w.sum() + self.eps)
        out = sum(w[i] * feats[i] for i in range(len(feats)))
        return self.conv(self.act(out))


class BiFPNLayer(nn.Module):
    """4개 레벨(f1=최고해상도 ... f4=최저해상도)에 대한 top-down + bottom-up 1회 패스."""

    def __init__(self, channels: int):
        super().__init__()
        self.td3 = _FastFusion(2, channels)  # P3_td = fuse(P3, resize(P4))
        self.td2 = _FastFusion(2, channels)  # P2_td = fuse(P2, resize(P3_td))
        self.td1 = _FastFusion(2, channels)  # P1_out = fuse(P1, resize(P2_td))

        self.bu2 = _FastFusion(3, channels)  # P2_out = fuse(P2, P2_td, resize(P1_out))
        self.bu3 = _FastFusion(3, channels)  # P3_out = fuse(P3, P3_td, resize(P2_out))
        self.bu4 = _FastFusion(2, channels)  # P4_out = fuse(P4, resize(P3_out))

    def forward(self, feats: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        p1, p2, p3, p4 = feats

        p3_td = self.td3(p3, _resize_like(p4, p3))
        p2_td = self.td2(p2, _resize_like(p3_td, p2))
        p1_out = self.td1(p1, _resize_like(p2_td, p1))

        p2_out = self.bu2(p2, p2_td, _resize_like(p1_out, p2))
        p3_out = self.bu3(p3, p3_td, _resize_like(p2_out, p3))
        p4_out = self.bu4(p4, _resize_like(p3_out, p4))

        return [p1_out, p2_out, p3_out, p4_out]


class BiFPNDecoder(nn.Module):
    """기존 FPNDecoder를 대체. lateral conv -> BiFPN N회 -> CoordAttention -> head."""

    def __init__(self, in_channels: Sequence[int], out_channels: int,
                 fpn_ch: int = 256, num_layers: int = 2):
        super().__init__()
        self.laterals = nn.ModuleList([nn.Conv2d(c, fpn_ch, 1) for c in in_channels])
        self.bifpn_layers = nn.ModuleList([BiFPNLayer(fpn_ch) for _ in range(num_layers)])
        self.coord_attn = CoordAttention(fpn_ch)
        self.head = nn.Sequential(
            nn.Conv2d(fpn_ch, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_channels, 1),
        )

    def forward(self, feats: Sequence[torch.Tensor], out_size: Tuple[int, int]) -> torch.Tensor:
        feats = [lat(f) for lat, f in zip(self.laterals, feats)]
        for layer in self.bifpn_layers:
            feats = layer(feats)

        p1 = self.coord_attn(feats[0])  # 최고해상도 레벨에 좌표 attention 적용
        p = F.interpolate(p1, size=out_size, mode="bilinear", align_corners=False)
        return self.head(p)


# ---------------------------------------------------------------------------
# 최종 모델 (decoder 교체 외 전부 동일)
# ---------------------------------------------------------------------------

class SwinWarpHintStrokeModel(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        max_strokes = int(config["max_strokes"])
        self.max_strokes = max_strokes
        in_chans = 1 + max_strokes

        self.hint_generator = LightweightWarpHintGenerator(max_flow_px=12.0)

        self.swin = timm.create_model(
            config["swin_name"],
            pretrained=config["swin_pretrained"],
            features_only=True,
            out_indices=(0, 1, 2, 3),
            in_chans=in_chans,
            img_size=config["image_size"],
            drop_rate=config["drop_rate"],
            drop_path_rate=config["drop_path_rate"],
        )

        info = self.swin.feature_info
        channels = list(info.channels())
        print(f"[Model] Swin channels: {channels}")

        bottleneck_ch = channels[-1]
        self.ref_encoder = ReferenceEncoder(out_ch=bottleneck_ch)
        self.cross_attn = BottleneckCrossAttention(bottleneck_ch, heads=8)
        self.decoder = BiFPNDecoder(
            channels, out_channels=max_strokes,
            fpn_ch=config.get("fpn_channels", 256),
            num_layers=config.get("bifpn_layers", 2),
        )

    def forward(self, I_prep: torch.Tensor, I_g: torch.Tensor, I_G_strokes: torch.Tensor) -> Dict[str, torch.Tensor]:
        flow = self.hint_generator(I_g, I_prep)
        H = warp_by_pixel_flow(I_G_strokes, flow)

        x = torch.cat([I_prep, H], dim=1)
        feats = [ensure_nchw(f) for f in self.swin(x)]

        g_feat = self.ref_encoder(I_g)
        g_feat = F.interpolate(g_feat, size=feats[-1].shape[-2:], mode="bilinear", align_corners=False)
        feats[-1] = self.cross_attn(feats[-1], g_feat)

        logits = self.decoder(feats, out_size=I_prep.shape[-2:])
        return {"logits": logits, "flow": flow, "H": H}

    def set_backbone_trainable(self, trainable: bool):
        for p in self.swin.parameters():
            p.requires_grad = trainable
