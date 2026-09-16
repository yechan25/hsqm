from .utils import set_seed, get_device, ensure_dir, load_config


def __getattr__(name):
    # Postprocessing can run on CPU without importing the timm backbone.
    if name == "SwinWarpHintStrokeModel":
        from .model import SwinWarpHintStrokeModel
        return SwinWarpHintStrokeModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
