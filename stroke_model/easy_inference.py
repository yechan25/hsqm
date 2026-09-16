"""Three-cell Colab entry point: find an existing checkpoint and show results."""
from pathlib import Path


def _choose(paths, title):
    paths = sorted(set(Path(p) for p in paths if Path(p).is_file()))
    if len(paths) == 1:
        return paths[0]
    if paths:
        print(title)
        for i, path in enumerate(paths, 1):
            print(f"  {i}. {path}")
        while True:
            answer = input("사용할 파일 번호를 입력하세요: ").strip()
            if answer.isdigit() and 1 <= int(answer) <= len(paths):
                return paths[int(answer) - 1]
            print("목록에 있는 번호를 입력해주세요.")
    while True:
        path = Path(input(f"{title}\n파일을 찾지 못했습니다. Drive에서 파일 경로를 복사해 붙여넣으세요: ").strip().strip('"').strip("'"))
        if path.is_file():
            return path
        print("해당 파일이 없습니다. 경로를 다시 확인해주세요.")


def _find_checkpoints(root):
    # Search the research project, never the entire mounted Drive by default.
    if not root.is_dir():
        return []
    return list(root.glob("outputs/**/best.pt")) + list(root.glob("output/**/best.pt")) + list(root.glob("checkpoints/best.pt"))


def load_session(checkpoint_path=""):
    import torch
    from .utils import load_config, safe_torch_load
    from .data import StrokeDataset, set_ignore_dirnames
    from .model import SwinWarpHintStrokeModel

    defaults = load_config(Path(__file__).resolve().parents[1] / "configs/base.yaml")
    roots = [Path("/content/drive/Shareddrives/2026 자율연구/HSQM"),
             Path("/content/drive/MyDrive/HSQM")]
    if checkpoint_path.strip():
        chosen = Path(checkpoint_path.strip())
        if not chosen.is_file():
            raise FileNotFoundError(f"체크포인트 파일을 찾지 못했습니다: {chosen}")
    else:
        candidates = [p for root in roots for p in _find_checkpoints(root)]
        chosen = _choose(candidates, "학습된 모델(best.pt) 선택")
    print("모델 파일:", chosen)
    checkpoint = safe_torch_load(chosen, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError("HSQM 학습에서 저장한 best.pt 파일을 선택해주세요.")
    config = {**defaults, **checkpoint.get("config", {})}
    config["swin_pretrained"] = False  # Already contained in the checkpoint.
    state = checkpoint["model"]
    if "decoder.lateral1.weight" in state:
        architecture = "FPN (기존 모델)"
        config["fpn_channels"] = state["decoder.lateral1.weight"].shape[0]
    elif "decoder.laterals.0.weight" in state:
        architecture = "BiFPN"
        config["fpn_channels"] = state["decoder.laterals.0.weight"].shape[0]
        layers = {int(k.split('.')[2]) for k in state if k.startswith("decoder.bifpn_layers.")}
        config["bifpn_layers"] = max(layers) + 1 if layers else 0
    else:
        raise ValueError("지원하는 FPN/BiFPN 체크포인트가 아닙니다.")
    config["max_strokes"] = state["decoder.head.3.weight"].shape[0]
    model = SwinWarpHintStrokeModel(config)
    if architecture.startswith("FPN"):
        from .legacy_decoder import FPNDecoder
        model.decoder = FPNDecoder(model.swin.feature_info.channels(), config["max_strokes"], config["fpn_channels"])
    model.load_state_dict(state, strict=True)
    del checkpoint, state
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    csv = Path(config["test_csv_path"])
    if not csv.is_file():
        project_roots = roots + [p for p in chosen.parents if p.name == "HSQM"]
        csv = _choose([root / "dataset/test_dataset/paths.csv" for root in project_roots], "테스트 데이터 paths.csv 선택")
    configured_root = Path(config["test_dataset_root"])
    # If the CSV moved, resolve its relative image paths from its new directory.
    root = configured_root if csv == Path(config["test_csv_path"]) and configured_root.is_dir() else csv.parent
    set_ignore_dirnames(config.get("ignore_dirnames", []))
    dataset = StrokeDataset(str(csv), str(root), train=False, config=config)
    if not len(dataset):
        raise ValueError("선택한 테스트 데이터 CSV가 비어 있습니다.")
    print(f"준비 완료: {architecture}, {device}, 테스트 이미지 {len(dataset)}장")
    print("이제 3번 셀을 실행하세요.")
    return {"model": model, "dataset": dataset, "device": device, "config": config,
            "checkpoint_path": str(chosen)}


def show_result(session, sample_index=0):
    import torch
    import matplotlib.pyplot as plt
    from .graph_postprocess import postprocess_batch_predictions_graph

    dataset = session["dataset"]
    if not 0 <= sample_index < len(dataset):
        raise ValueError(f"이미지 번호는 0부터 {len(dataset)-1} 사이로 입력하세요.")
    item = dataset[sample_index]
    device = session["device"]
    prep = item["I_prep"][None].to(device)
    refs = item["I_G_strokes"][None].to(device)
    with torch.no_grad():
        out = session["model"](prep, item["I_g"][None].to(device), refs)
        raw = torch.sigmoid(out["logits"])
        post = postprocess_batch_predictions_graph(
            raw, prep, refs, warped_hints=out["H"], mode="overlap",
            threshold=session["config"].get("hard_threshold", 0.25),
        )
    active = torch.where(refs[0].sum((1, 2)) > 0)[0].tolist()
    fig, axes = plt.subplots(3, len(active) + 1, squeeze=False,
                             figsize=(2.3 * (len(active) + 1), 7))
    axes[0, 0].imshow(item["I_prep"][0], cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title("Input")
    axes[1, 0].imshow(post[0].sum(0).cpu(), cmap="viridis", vmin=0, vmax=max(2, len(active)))
    axes[1, 0].set_title("Shared stroke count")
    axes[2, 0].imshow(item["I_g"][0], cmap="gray", vmin=0, vmax=1)
    axes[2, 0].set_title("Reference")
    for j, k in enumerate(active, 1):
        for row, (label, image) in enumerate([
            ("Before", raw[0, k].cpu()), ("After", post[0, k].cpu()),
            ("Target (soft)", item["target_strokes"][k]),
        ]):
            axes[row, j].imshow(image, cmap="gray", vmin=0, vmax=1)
            axes[row, j].set_title(f"{label} #{k+1}")
    for ax in axes.flat:
        ax.axis("off")
    plt.tight_layout()
    plt.show()
    print(f"이미지 {sample_index}: {item['char_id']}")
    print("위: 모델 예측 / 가운데: 새 후처리 / 아래: 정답")
    print("다른 이미지는 3번 셀의 이미지_번호만 바꿔 다시 실행하세요.")
    return post.cpu()
