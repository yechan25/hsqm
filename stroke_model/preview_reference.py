"""Visual check of converted labels and trained validation predictions."""
import torch
import numpy as np
from .train_reference import read_manifest, StrokeDataset


def preview_reference(manifest, checkpoint=None, count=3):
    import matplotlib.pyplot as plt
    from .reference_model import ReferenceStrokeModel
    rows = [r for r in read_manifest(manifest) if r['split'] == 'val'][:count]
    dataset = StrokeDataset(rows)
    model = None
    if checkpoint is not None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        model = ReferenceStrokeModel(image_size=state['image_size'], width=state['width'], pretrained=False).to(device)
        model.load_state_dict(state['model'])
        model.eval()
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
        plt.show()
