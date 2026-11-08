# 5번 셀 전체 교체: 기존 점수 + 추가 잉크 감점 위치 확인
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm

# 같은 생성 폴더의 점수 결과가 있으면 재사용합니다.
RECOMPUTE = False
same_run = False
if 'result' in globals():
    entries = result.get('extraction', {}).get('results', [])
    if entries:
        same_run = Path(entries[0]['npz']).resolve().parents[1] == Path(OUTPUT_DIR).resolve()
if RECOMPUTE or not same_run:
    result, scores = score_pilot(OUTPUT_DIR/'generation_report.json', CHECKPOINT,
        score_config=SCORE_CONFIG, source_config=SOURCE_CONFIG, show=True)
display(scores)

for sample, measured in zip(result['extraction']['results'], result['indicator']['results']):
    with np.load(measured['measurement_npz'], allow_pickle=False) as data:
        ink, extra, support = [data[key].copy() for key in ('image', 'extra_map', 'support')]
    fig, axes = plt.subplots(1, 4, figsize=(19, 4))
    axes[0].imshow(ink, cmap='gray_r', vmin=0, vmax=1)
    if support.min() < .5 < support.max():
        axes[0].contour(support, levels=[.5], colors=['cyan'], linewidths=.8)
    axes[0].set_title('Scored input + allowed region')
    heat = axes[1].imshow(extra, cmap='magma', norm=PowerNorm(gamma=.35, vmin=0, vmax=1))
    fig.colorbar(heat, ax=axes[1], fraction=.045)
    axes[1].set_title('Extra ink: enhanced faint values')
    axes[2].imshow(ink, cmap='gray_r', vmin=0, vmax=1)
    axes[2].imshow(np.ones((*ink.shape, 3))*[1, 0, 0], alpha=np.clip(extra/.08, 0, .85))
    axes[2].set_title('Extra overlay: display amplified')
    bins = [0, .02, .1, .5, 1.000001]
    mass = np.array([extra[(ink >= a) & (ink < b)].sum() for a, b in zip(bins[:-1], bins[1:])])
    shares = 100*mass/max(float(extra.sum()), 1e-12)
    axes[3].bar(['0-.02', '.02-.10', '.10-.50', '.50-1'], shares)
    axes[3].set(xlabel='Input ink intensity', ylabel='Share of extra-ink mass (%)', ylim=(0, 100))
    for ax in axes[:3]:
        ax.axis('off')
    fig.suptitle(f"{sample.get('label', '')} | score={measured['structural_score']:.1f} | "
                 f"before extra={measured['score_before_extra']:.1f} | extra fraction={100*measured['extra']:.2f}%")
    fig.tight_layout()
    path = Path(result['save_dir'])/f"extra_diagnosis_{measured['sample']:04d}.png"
    fig.savefig(path, dpi=140)
    plt.show()
    plt.close(fig)

print('빨강은 추가 잉크로 계산된 위치입니다. 옅은 성분을 보려고 표시 대비만 높였으며 점수는 바꾸지 않았습니다.')
print('막대의 왼쪽에 몰리면 옅은 픽셀의 영향, 오른쪽에 몰리면 진한 잉크의 영향을 확인하세요.')
print('밝기 구간만으로 배경과 실제 추가 획을 자동 판정하지는 않습니다.')
print('진단 그림 저장:', result['save_dir'])
from stroke_model.source_quality_v5 import compare_quality_v5
QUALITY_CONFIG = {}  # 실험 기본값. 원래 점수는 보존하고 새 비교 열을 추가합니다.
quality_report, quality_scores = compare_quality_v5(result, config=QUALITY_CONFIG, show=True)
display(quality_scores[['label','raw_source_score','foreground_structural_score',
    'contour_excess','contour_score_cap','experimental_score']])
print('새 점수는 실험용입니다. 정상 글꼴 차이와 실제 돌출·패임을 구분해 그림으로 확인하세요.')
print('윤곽 항은 현재 진단용이며, 역확산 guidance에 바로 미분해서 쓸 수 있는 구현은 아닙니다.')
from google.colab import files
download = widgets.Button(description='기존 진단 ZIP 다운로드')
debug_path = result['debug_zip']
download.on_click(lambda _, path=debug_path: files.download(path))
display(download)
