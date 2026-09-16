"""Synthetic illustration, not a benchmark on real model predictions.

From hsqm: python -m tests.demo_graph_postprocess is not required;
use: python tests/demo_graph_postprocess.py
"""
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from stroke_model.graph_postprocess import refine_strokes_graph
from stroke_model.postprocess import postprocess_sample_seeded_partition


def run():
    output = Path(__file__).resolve().parents[1] / "diagnostics"
    output.mkdir(exist_ok=True)
    cases = {}
    cross = np.zeros((2, 48, 48), np.float32)
    cross[0, 22:26, 6:42] = 1
    cross[1, 6:42, 22:26] = 1
    cases["Crossing strokes"] = cross
    short = np.zeros_like(cross)
    short[0, 22:26, 6:42] = 1
    short[1, 18:26, 22:26] = 1
    cases["Short attached stroke"] = short
    colors = np.array([[0.95, 0.25, 0.20], [0.05, 0.75, 0.80]])
    figure, axes = plt.subplots(len(cases), 3, figsize=(10, 6.8))
    records = []
    for row, (name, target) in enumerate(cases.items()):
        ink = target.max(0)
        start = time.perf_counter()
        legacy = postprocess_sample_seeded_partition(target, ink, target_hint_b=target)
        legacy_time = time.perf_counter() - start
        start = time.perf_counter()
        graph = refine_strokes_graph(target, ink, reference_strokes=target) > 0.25
        graph_time = time.perf_counter() - start
        # Both methods receive identical ideal predictions. The reference
        # serves as oracle input here solely for this controlled illustration.
        for col, (label, result, seconds) in enumerate([
            ("Ideal input / GT", target > 0, 0),
            ("Legacy exclusive", legacy > 0.5, legacy_time),
            ("Graph overlap", graph, graph_time),
        ]):
            dice = float((2 * (result * target).sum((1, 2)) / (result.sum((1, 2)) + target.sum((1, 2)))).mean())
            shared = int((result.sum(0) > 1).sum())
            rgb = np.clip(np.einsum("khw,kc->hwc", result.astype(float), colors), 0, 1)
            axes[row, col].imshow(rgb)
            axes[row, col].set_title(f"{label}\nDice={dice:.3f}, shared={shared}")
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            if col == 0:
                axes[row, col].set_ylabel(name)
            records.append({"case": name, "method": label, "active_dice": dice,
                            "shared_pixels": shared, "seconds": seconds})
    figure.suptitle("Synthetic sanity check: ideal predictions (not real model accuracy)\nRed / cyan: separate strokes. White: shared crossing pixels.")
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    figure.savefig(output / "graph_postprocess_synthetic.png", dpi=160)
    plt.close(figure)
    (output / "graph_postprocess_synthetic.json").write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    run()
