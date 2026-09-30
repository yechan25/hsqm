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


def load_session(checkpoint_path="", expected_architecture=None):
    import torch
    from .utils import load_config, safe_torch_load
    from .data import StrokeDataset, set_ignore_dirnames
    from .model import SwinWarpHintStrokeModel
    from .convert_reference_csv import existing_path
    def resolve_existing(path):
        try:
            return existing_path(path)
        except FileNotFoundError:
            return Path(path)

    defaults = load_config(Path(__file__).resolve().parents[1] / "configs/base.yaml")
    roots = [Path("/content/drive/Shareddrives/2026 자율연구/HSQM"),
             Path("/content/drive/MyDrive/HSQM")]
    roots = [resolve_existing(p) for p in roots]
    if checkpoint_path.strip():
        chosen = resolve_existing(checkpoint_path.strip())
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
    if expected_architecture and architecture != expected_architecture:
        raise ValueError(f'{expected_architecture} 체크포인트가 필요하지만 선택한 파일은 {architecture}입니다. 2번 셀에서 다른 파일을 선택하세요.')
    config["max_strokes"] = state["decoder.head.3.weight"].shape[0]
    model = SwinWarpHintStrokeModel(config)
    if architecture.startswith("FPN"):
        from .legacy_decoder import FPNDecoder
        model.decoder = FPNDecoder(model.swin.feature_info.channels(), config["max_strokes"], config["fpn_channels"])
    model.load_state_dict(state, strict=True)
    del checkpoint, state
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    csv = resolve_existing(config["test_csv_path"])
    if not csv.is_file():
        project_roots = roots + [p for p in chosen.parents if p.name == "HSQM"]
        csv = _choose([root / "dataset/test_dataset/paths.csv" for root in project_roots], "테스트 데이터 paths.csv 선택")
    configured_root = resolve_existing(config["test_dataset_root"])
    # If the CSV moved, resolve its relative image paths from its new directory.
    root = configured_root if csv == resolve_existing(config["test_csv_path"]) and configured_root.is_dir() else csv.parent
    set_ignore_dirnames(config.get("ignore_dirnames", []))
    dataset = StrokeDataset(str(csv), str(root), train=False, config=config)
    if not len(dataset):
        raise ValueError("선택한 테스트 데이터 CSV가 비어 있습니다.")
    print(f"준비 완료: {architecture}, {device}, 테스트 이미지 {len(dataset)}장")
    print("이제 3번 셀을 실행하세요.")
    return {"model": model, "dataset": dataset, "device": device, "config": config,
            "checkpoint_path": str(chosen), "architecture": architecture}


def show_bifpn_baseline(session, sample_index=0, mode='raw'):
    """Raw sigmoid is never altered. Optional legacy result is a separate row."""
    import torch
    import matplotlib.pyplot as plt
    if session.get('architecture') != 'BiFPN':
        raise ValueError('BiFPN 체크포인트를 먼저 불러오세요. V2/FPN 결과와 혼동하지 않도록 제한합니다.')
    if mode not in ('raw', 'legacy'):
        raise ValueError('mode must be raw or legacy')
    if not 0 <= sample_index < len(session['dataset']):
        raise ValueError(f"이미지 번호: 0..{len(session['dataset'])-1}")
    item = session['dataset'][sample_index]
    prep, reference, strokes = [item[key][None].to(session['device']) for key in ('I_prep', 'I_g', 'I_G_strokes')]
    with torch.no_grad():
        output = session['model'](prep, reference, strokes)
        raw = output['logits'].sigmoid().cpu()
    legacy = None
    if mode == 'legacy':
        from .postprocess import postprocess_batch_predictions_seeded_partition
        legacy = postprocess_batch_predictions_seeded_partition(raw, prep.cpu(), target_hints=strokes.cpu())
    active = (strokes[0].sum((1, 2)) > 0).nonzero().flatten().tolist()
    if not active:
        raise ValueError('유효한 기준 획이 없습니다')
    for offset in range(0, len(active), 5):
        ids = active[offset:offset+5]
        rows = 4 if legacy is not None else 3
        fig, axes = plt.subplots(rows, len(ids)+1, figsize=(2.5*(len(ids)+1), 2.5*rows), squeeze=False)
        axes[0, 0].imshow(item['I_prep'][0].numpy(), cmap='gray_r', vmin=0, vmax=1)
        axes[0, 0].set_title('Input')
        axes[1, 0].imshow(item['I_g'][0].numpy(), cmap='gray_r', vmin=0, vmax=1)
        axes[1, 0].set_title('Reference')
        for col, k in enumerate(ids, 1):
            panels = [('Raw sigmoid', raw[0, k]), ('Reference stroke', strokes[0, k].cpu()),
                      ('Training target (soft)', item['target_strokes'][k])]
            if legacy is not None:
                panels.append(('Legacy postprocess', legacy[0, k]))
            for row, (title, values) in enumerate(panels):
                axes[row, col].imshow(values.numpy(), cmap='gray_r', vmin=0, vmax=1)
                axes[row, col].set_title(f'{title} #{k+1}')
        for ax in axes.flat:
            ax.axis('off')
        fig.suptitle('BiFPN baseline: raw scale 0..1, no graph refinement')
        fig.tight_layout()
        plt.show()
        plt.close(fig)
    print('체크포인트:', session['checkpoint_path'])
    print(f"이미지 {sample_index}: {item.get('char_id', '')}")
    print('Raw sigmoid: 전경 곱하기, argmax, threshold, 그래프 보정, 획별 최대값 정규화 없음.')
    if legacy is not None:
        print('마지막 줄만 기존 bigstroke_protected 후처리를 적용했습니다. 그래프 보강은 사용하지 않습니다.')
    return {'raw_masks': raw, 'legacy_masks': legacy}


def show_result(session, sample_index=0, mode="exclusive"):
    import torch
    import matplotlib.pyplot as plt
    from .graph_postprocess import postprocess_batch_predictions_graph
    from .postprocess import check_partition_integrity

    if mode not in {"exclusive", "overlap"}:
        raise ValueError("mode는 exclusive 또는 overlap으로 선택하세요.")

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
            raw, prep, refs, warped_hints=out["H"], mode=mode,
            threshold=session["config"].get("hard_threshold", 0.25),
        )
    active = torch.where(refs[0].sum((1, 2)) > 0)[0].tolist()
    fig, axes = plt.subplots(3, len(active) + 1, squeeze=False,
                             figsize=(2.3 * (len(active) + 1), 7))
    axes[0, 0].imshow(item["I_prep"][0], cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title("Input")
    axes[1, 0].imshow(post[0].sum(0).cpu(), cmap="viridis", vmin=0, vmax=max(2, len(active)))
    axes[1, 0].set_title(f"Assignment count ({mode})")
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
    report = check_partition_integrity(post[0], prep[0])
    print(f"모드: {mode} | 글자 픽셀: {int(report['input_ink_pixels'])} | "
          f"미배정: {int(report['missing_pixels'])} | "
          f"중복 배정(초과 횟수): {int(report['overlap_excess_pixels'])} | "
          f"배경 침범: {int(report['extra_pixels'])}")
    print("검사는 입력에서 글자로 판별된 영역 기준입니다.")
    print("이미지_번호 또는 mode를 바꾼 뒤 3번 셀만 다시 실행하세요.")
    return post.cpu()
