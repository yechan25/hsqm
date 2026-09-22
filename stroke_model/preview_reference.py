"""Visual check of converted labels and trained validation predictions."""
import torch
import numpy as np
from pathlib import Path
from .train_reference import read_manifest, StrokeDataset


def soft_diagnostics(output):
    """Detached visualization/export only. Do not use this helper in guidance."""
    probabilities = output['probabilities'][0].detach().float().cpu()
    masks = output['masks'][0].detach().float().cpu()
    active = output['active'][0].detach().cpu().bool()
    entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(0)
    if active.sum() > 1:
        entropy = entropy / active.sum().float().log()
    else:
        entropy = torch.zeros_like(entropy)
    return {'probabilities': probabilities.numpy(), 'soft_masks': masks.numpy(),
            'entropy': entropy.numpy(), 'active': active.numpy()}


def show_soft_strokes(image, strokes, output, prefix=None):
    """Plot true fixed-scale probabilities and soft masks, without argmax/threshold."""
    import matplotlib.pyplot as plt
    payload = soft_diagnostics(output)
    ids = np.flatnonzero(payload['active'])
    foreground = image[0].numpy() > .5  # Display mask only, not used on soft masks.
    for page, offset in enumerate(range(0, len(ids), 6)):
        selected = ids[offset:offset+6]
        figure, axes = plt.subplots(3, len(selected), squeeze=False, figsize=(2.6*len(selected), 7))
        for col, k in enumerate(selected):
            axes[0, col].imshow(strokes[k], cmap='gray_r', vmin=0, vmax=1)
            axes[0, col].set_title(f'Reference stroke {k+1}')
            p = np.ma.masked_where(~foreground, payload['probabilities'][k])
            axes[1, col].imshow(p, cmap='viridis', vmin=0, vmax=1)
            axes[1, col].set_title(f'p{ k+1 }: ink display only')
            artist = axes[2, col].imshow(payload['soft_masks'][k], cmap='viridis', vmin=0, vmax=1)
            axes[2, col].set_title(f'Soft mask {k+1}: image * p')
            for ax in axes[:, col]:
                ax.axis('off')
        figure.suptitle('Before exclusive: fixed scale 0..1 (no per-stroke normalization)')
        figure.subplots_adjust(right=.9, top=.9, hspace=.25)
        figure.colorbar(artist, cax=figure.add_axes([.93, .15, .015, .65]))
        if prefix is not None:
            figure.savefig(str(prefix) + f'_soft_{page+1}.png', dpi=140)
        plt.show()
        plt.close(figure)
    figure, axis = plt.subplots(figsize=(4, 4))
    artist = axis.imshow(np.ma.masked_where(~foreground, payload['entropy']), cmap='magma', vmin=0, vmax=1)
    axis.set_title('Assignment entropy: 0=certain, 1=ambiguous')
    axis.axis('off')
    figure.colorbar(artist, ax=axis)
    if prefix is not None:
        figure.savefig(str(prefix) + '_entropy.png', dpi=140, bbox_inches='tight')
        np.savez_compressed(str(prefix) + '_soft.npz', **payload, image=image.numpy(), reference_strokes=strokes.numpy())
    plt.show()
    plt.close(figure)


def preview_reference(manifest, checkpoint=None, count=3, split='val', start=0, save_dir=None, show_soft=True):
    import matplotlib.pyplot as plt
    from .reference_model import ReferenceStrokeModel
    rows = [r for r in read_manifest(manifest) if r['split'] == split][start:start+count]
    size = 128
    model = None
    if checkpoint is not None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        size = state['image_size']
        model = ReferenceStrokeModel(image_size=state['image_size'], width=state['width'], pretrained=False).to(device)
        model.load_state_dict(state['model'])
        model.eval()
    dataset = StrokeDataset(rows, size=size)
    for i in range(len(dataset)):
        image, reference, strokes, labels = dataset[i]
        palette = plt.get_cmap('tab20')(np.arange(len(strokes)) % 20)[:, :3]
        def colored(ids, mask):
            rgb = np.ones((*ids.shape, 3))
            rgb[mask] = palette[ids[mask]]
            return rgb
        known = labels.numpy() >= 0
        gt = colored(labels.numpy(), known)
        unknown = (image[0].numpy() > .5) & ~known
        gt[unknown] = [.65, .65, .65]
        panels = [(image[0], 'Input'), (reference[0], 'Reference'), (gt, 'Labels: gray = unknown')]
        if model is not None:
            with torch.no_grad():
                out = model(image[None].to(device), reference[None].to(device), strokes[None].to(device))
            prediction = out['probabilities'].argmax(1)[0].cpu().numpy()
            panels.append((colored(prediction, image[0].numpy() > .5), 'Prediction: exclusive'))
        figure, axes = plt.subplots(1, len(panels), figsize=(3 * len(panels), 3))
        for ax, (content, title) in zip(axes, panels):
            ax.imshow(content, cmap='gray_r', vmin=0, vmax=1)
            ax.set_title(title)
            ax.axis('off')
        plt.tight_layout()
        if save_dir is not None:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            figure.savefig(Path(save_dir) / f'{split}_{start+i:04d}.png', dpi=140)
        plt.show()
        plt.close(figure)
        if model is not None and show_soft:
            prefix = Path(save_dir) / f'{split}_{start+i:04d}' if save_dir is not None else None
            show_soft_strokes(image, strokes, out, prefix=prefix)
