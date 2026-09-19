"""Reference-conditioned V2. Inputs: white ink on black, Bx1xHxW in [0,1].

No thresholding, CPU conversion or detached image features in the forward path.
Old checkpoints are deliberately not compatible with this new architecture.
"""
import torch
from torch import nn
from torch.nn import functional as F


def block(cin, cout, stride=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride=stride, padding=1),
                         nn.GroupNorm(4, cout), nn.GELU())


def resize(x, size):
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


def warp(strokes, flow):
    b, _, h, w = strokes.shape
    ys = (torch.arange(h, device=strokes.device, dtype=strokes.dtype) + .5) * 2 / h - 1
    xs = (torch.arange(w, device=strokes.device, dtype=strokes.dtype) + .5) * 2 / w - 1
    y, x = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack((x, y), -1)[None].expand(b, -1, -1, -1)
    offset = torch.stack((flow[:, 0] * 2 / w, flow[:, 1] * 2 / h), -1)
    return F.grid_sample(strokes, grid + offset, align_corners=False)


class ReferenceStrokeModel(nn.Module):
    def __init__(self, image_size=128, width=64, pretrained=True,
                 backbone=None, channels=None, channels_last=False):
        super().__init__()
        if width % 4:
            raise ValueError("width must be divisible by 4")
        mean, std = (.485, .456, .406), (.229, .224, .225)
        if backbone is None:
            import timm
            backbone = timm.create_model(
                "swin_tiny_patch4_window7_224", pretrained=pretrained,
                features_only=True, out_indices=(0, 1, 2, 3),
                in_chans=3, img_size=image_size, drop_path_rate=0.)
            channels = backbone.feature_info.channels()
            channels_last = True  # timm Swin features are NHWC.
            mean = backbone.pretrained_cfg.get("mean", mean)
            std = backbone.pretrained_cfg.get("std", std)
        if not channels:
            raise ValueError("Injected backbone requires its feature channels")
        self.backbone = backbone
        self.channels_last = channels_last
        self.backbone_frozen = True
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        self.register_buffer("rgb_mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("rgb_std", torch.tensor(std).view(1, 3, 1, 1))
        self.lateral = nn.ModuleList(nn.Conv2d(c, width, 1) for c in channels)
        self.smooth = nn.ModuleList(block(width, width) for _ in channels)
        self.detail = block(1, width)  # Full-resolution branch for thin strokes.
        self.fuse = block(width * 2, width)
        self.flow_net = nn.Sequential(block(2, 16, 2), block(16, 32, 2),
                                      nn.Conv2d(32, 2, 3, padding=1))
        nn.init.zeros_(self.flow_net[-1].weight)
        nn.init.zeros_(self.flow_net[-1].bias)
        # Shared across stroke IDs; full glyph + selected stroke + xy context.
        self.reference = nn.Sequential(block(4, 16, 2), block(16, width, 2))
        self.head = nn.Sequential(block(width * 2 + 4, width),
                                  nn.Conv2d(width, 1, 1))

    def train(self, mode=True):
        super().train(mode)
        if self.backbone_frozen:
            self.backbone.eval()
        return self

    def forward(self, image, reference, strokes, active=None):
        if image.ndim != 4 or image.shape[1] != 1 or reference.shape != image.shape:
            raise ValueError("image/reference must have matching Bx1xHxW shapes")
        b, _, h, w = image.shape
        if strokes.ndim != 4 or strokes.shape[0] != b or strokes.shape[-2:] != (h, w):
            raise ValueError("strokes must have shape BxKxHxW")
        k = strokes.shape[1]
        if active is None:
            active = strokes.flatten(2).sum(-1) > 0  # Fixed reference metadata only.
        if active.shape != (b, k) or active.dtype != torch.bool or not active.any(1).all():
            raise ValueError("Each sample needs at least one active reference stroke")
        rgb = (1 - image).expand(-1, 3, -1, -1)
        feats = self.backbone((rgb - self.rgb_mean) / self.rgb_std)
        if self.channels_last:
            feats = [f.permute(0, 3, 1, 2) for f in feats]
        pyramid = None
        for i in reversed(range(len(feats))):
            lateral = self.lateral[i](feats[i])
            pyramid = lateral if pyramid is None else lateral + resize(pyramid, lateral.shape[-2:])
            pyramid = self.smooth[i](pyramid)
        features = self.fuse(torch.cat((resize(pyramid, (h, w)), self.detail(image)), 1))
        flow = 12 * resize(self.flow_net(torch.cat((reference, image), 1)), (h, w)).tanh()
        hints = warp(strokes, flow)
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, h, device=image.device, dtype=image.dtype),
                                torch.linspace(-1, 1, w, device=image.device, dtype=image.dtype), indexing="ij")
        xy = torch.stack((xx, yy))[None].expand(b, -1, -1, -1)
        scores = []
        for j in range(k):
            # Spatial reference features retain more information than one pooled prototype.
            r = self.reference(torch.cat((reference, strokes[:, j:j+1], xy), 1))
            r = warp(resize(r, (h, w)), flow)
            x = torch.cat((features, r, hints[:, j:j+1], image, xy), 1)
            scores.append(self.head(x))
        logits = torch.cat(scores, 1).masked_fill(~active[:, :, None, None], -torch.inf)
        probabilities = logits.softmax(1) * active[:, :, None, None]
        probabilities = probabilities / probabilities.sum(1, keepdim=True)
        return {"logits": logits, "probabilities": probabilities,
                "masks": image * probabilities, "flow": flow, "hints": hints,
                "active": active}


def exclusive_masks(output, foreground):
    """Export only: non-differentiable exact partition of the supplied foreground."""
    expected = output['probabilities'][:, :1].shape
    if foreground.shape != expected:
        raise ValueError('foreground must have shape Bx1xHxW')
    labels = output["probabilities"].argmax(1)
    masks = F.one_hot(labels, output["probabilities"].shape[1]).permute(0, 3, 1, 2)
    return masks.bool() & foreground.bool()


def stroke_moments(masks, eps=1e-6):
    """Area fraction, centroid x/y and xx/xy/yy covariance in normalized coordinates.

    Near-empty strokes have unreliable geometry: callers must mask/inspect them.
    """
    h, w = masks.shape[-2:]
    y, x = torch.meshgrid(torch.linspace(-1, 1, h, device=masks.device, dtype=masks.dtype),
                          torch.linspace(-1, 1, w, device=masks.device, dtype=masks.dtype), indexing="ij")
    mass = masks.sum((-2, -1))
    weights = masks / mass.clamp_min(eps)[..., None, None]
    cx, cy = (weights * x).sum((-2, -1)), (weights * y).sum((-2, -1))
    dx, dy = x - cx[..., None, None], y - cy[..., None, None]
    return torch.stack((mass / (h * w), cx, cy,
                        (weights * dx.square()).sum((-2, -1)),
                        (weights * dx * dy).sum((-2, -1)),
                        (weights * dy.square()).sum((-2, -1))), -1)


def partition_loss(output, labels):
    """labels: BxHxW, 0..K-1 for owned pixels; -1 background/unknown.

    Unknown crossing pixels are excluded, not assigned pseudo-certainty.
    Soft Dice averages only annotated, nonempty target strokes.
    """
    logits = output["logits"]
    valid = labels >= 0
    if labels.shape != logits.shape[:1] + logits.shape[2:]:
        raise ValueError("labels must have shape BxHxW")
    if not valid.any() or (labels < -1).any() or (labels >= logits.shape[1]).any():
        raise ValueError("Need annotated pixels with labels in -1..K-1")
    target = F.one_hot(labels.clamp_min(0).long(), logits.shape[1]).permute(0, 3, 1, 2)
    target = target * valid[:, None]
    if (target.bool() & ~output["active"][:, :, None, None]).any():
        raise ValueError("Target refers to an inactive reference stroke")
    ce = F.cross_entropy(logits, labels.long(), ignore_index=-1)
    pred = output["probabilities"] * valid[:, None]
    mass = target.sum((-2, -1))
    dice = (2 * (pred * target).sum((-2, -1)) + 1e-6) / (pred.sum((-2, -1)) + mass + 1e-6)
    dice_loss = (1 - dice)[mass > 0].mean()
    flow = output["flow"]
    smooth = (flow[..., 1:, :] - flow[..., :-1, :]).square().mean()
    smooth = smooth + (flow[..., :, 1:] - flow[..., :, :-1]).square().mean()
    return {"loss": ce + dice_loss + .001 * smooth, "ce": ce,
            "dice_loss": dice_loss, "flow_smoothness": smooth}
