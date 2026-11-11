"""Build a self-contained L4 notebook consuming an existing fit-sweep ZIP."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build():
    cells = []
    def add(kind, source):
        if kind == 'code':
            ast.parse(source)
        cell = dict(cell_type=kind, metadata={}, source=source.strip().splitlines(keepends=True))
        if kind == 'code':
            cell.update(outputs=[], execution_count=None)
        cells.append(cell)
    add('markdown', '''# 획 추출 기반 후보 → 실제 추가 역확산 (L4)

1 → 2 → 3 순서로 실행합니다. 기존 `sweep_...zip` 하나만 있으면 됩니다.
Drive 마운트와 best.pt 재탐색, 획 추출 재실행이 필요 없습니다.
2번의 `ZIP_PATH`에 이미 올린 파일 경로를 쓰거나, 비워두고 ZIP을 업로드하세요.

기본 사례는 strength=.65 / seed=100입니다. 원래 생성물을 다시 노이즈화하고,
같은 초기 latent로 **추가 역확산만** 한 결과와 **획 추출 기반 후보로 유도한 역확산**을 비교합니다.
기하학적 후보는 가이드에 사용하지 않습니다. 후보는 정답 이미지가 아니며,
후보와의 일치도 향상 자체가 실제 오류 감소나 HSQM 성능을 보장하지는 않습니다.
최종 PNG는 VAE를 거친 실제 역확산 출력이며 후보 픽셀을 붙여 넣지 않습니다.
''')
    helper = (ROOT/'diffusion_model/fitted_candidate_diffusion.py').read_text(encoding='utf-8')
    add('code', '''import subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'install',
    'diffusers==0.37.0', 'transformers==4.57.6', 'accelerate',
    'safetensors', 'numpy', 'pillow', 'matplotlib'], check=True)
from pathlib import Path
import importlib.util
''' + 'HELPER_SOURCE = '+repr(helper)+'''
helper_path=Path('/content/fitted_candidate_diffusion_runtime.py')
helper_path.write_text(HELPER_SOURCE,encoding='utf-8')
spec=importlib.util.spec_from_file_location('fitted_candidate_diffusion_runtime',helper_path)
repair=importlib.util.module_from_spec(spec);spec.loader.exec_module(repair)
import torch
if not torch.cuda.is_available():raise RuntimeError('런타임 → 런타임 유형 변경 → L4 GPU를 선택하세요.')
print('GPU:',torch.cuda.get_device_name(0))
''')
    add('code', '''import json, zipfile
from google.colab import files
from IPython.display import display

ZIP_PATH=''  # 예: '/content/sweep_20261111_062740_406072.zip'
CASE_ID='case_003_strength_0p65_seed_100'
if not ZIP_PATH:
    uploaded=files.upload()
    names=[name for name in uploaded if name.lower().endswith('.zip')]
    if len(names)!=1:raise ValueError('sweep ZIP 하나를 선택하세요.')
    ZIP_PATH=names[0]
with zipfile.ZipFile(ZIP_PATH) as z:
    available=json.loads(z.read('case_manifest.json'))
print('선택 가능한 사례:',[c['case_id'] for c in available])
source,candidate,metadata=repair.load_case(ZIP_PATH,CASE_ID)
print('선택:',CASE_ID,'| 왼쪽: 오류 입력 / 오른쪽: 획 추출 기반 후보')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,2,figsize=(10,5))
for ax,im,title in zip(axes,[source,candidate],['Erroneous input','Extractor-fitted guide']):
    ax.imshow(im);ax.set_title(title);ax.axis('off')
plt.show();plt.close(fig)
''')
    add('code', '''from datetime import datetime
import shutil
from PIL import Image
from huggingface_hub import model_info

# .35는 이번 추가 역확산의 강도입니다. 원래 오류 생성 강도 .65와 별개입니다.
STRENGTH=.35
STEPS=60  # 실제 추가 역확산은 int(STEPS*STRENGTH)=21단계
INNER_STEPS=4
STEP_RATIO=.02
SEED=2026
MODEL_ID='stable-diffusion-v1-5/stable-diffusion-v1-5'
REVISION=model_info(MODEL_ID).sha
OUTPUT=Path('/content')/('fitted_reverse_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
def progress(method,step,total,record):
    if step==1 or step%4==0 or step==total:
        print(f"{method}: {step}/{total} | accepted={record['accepted_inner_steps']} | "
              f"loss={record.get('loss_before')} → {record.get('loss_after')}",flush=True)
report=repair.run_pair(source,candidate,metadata,OUTPUT,steps=STEPS,strength=STRENGTH,
    seed=SEED,inner_steps=INNER_STEPS,step_ratio=STEP_RATIO,
    model_id=MODEL_ID,revision=REVISION,progress=progress)
panels=[('Input',source),('Re-denoise only',Image.open(OUTPUT/'redenoise_only.png')),
        ('Candidate-guided reverse diffusion',Image.open(OUTPUT/'candidate_guided.png')),
        ('Guide ONLY',candidate)]
fig,axes=plt.subplots(1,4,figsize=(20,5))
for ax,(title,im) in zip(axes,panels):
    ax.imshow(im);ax.set_title(title);ax.axis('off')
fig.tight_layout();fig.savefig(OUTPUT/'comparison.png',dpi=140);plt.show();plt.close(fig)
print('実際の追加ステップ数 / 실제 추가 단계:',report['actual_steps'])
print('후보에 대한 손실(정확도 아님):',json.dumps(report['metrics'],indent=2))
for method in ('redenoise_only','candidate_guided'):
    snapshots=sorted(OUTPUT.glob(method+'_step_*.png'))
    fig,axes=plt.subplots(1,len(snapshots),figsize=(4*len(snapshots),4),squeeze=False)
    for ax,path in zip(axes.flat,snapshots):
        ax.imshow(Image.open(path));ax.set_title(path.stem);ax.axis('off')
    fig.tight_layout();fig.savefig(OUTPUT/(method+'_trajectory.png'),dpi=120);plt.show();plt.close(fig)
(OUTPUT/'runtime_source.py').write_text(HELPER_SOURCE,encoding='utf-8')
archive=shutil.make_archive(str(OUTPUT),'zip',OUTPUT)
print('결과:',archive)
files.download(archive)
'''.replace('実際の追加ステップ数 / ', ''))
    nb = dict(cells=cells, metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
              language_info=dict(name='python'), accelerator='GPU', colab=dict(provenance=[])),
              nbformat=4, nbformat_minor=5)
    path = Path(__file__).with_name('fitted_candidate_diffusion.ipynb')
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
    return path


if __name__ == '__main__':
    print(build())
