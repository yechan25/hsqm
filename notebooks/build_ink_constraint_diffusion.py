"""Build the sparse erase/restore L4 experiment from the shared paired runner."""
import ast
import json
from pathlib import Path
from build_fitted_candidate_diffusion import build as build_pixel_notebook


def build():
    nb = json.loads(build_pixel_notebook().read_text(encoding='utf-8'))
    def put(index, source):
        if nb['cells'][index]['cell_type'] == 'code':
            ast.parse(source)
        nb['cells'][index]['source'] = source.strip().splitlines(keepends=True)
    put(0, '''# 국소 삭제·추가 압력으로 역확산 (L4)

1 → 2 → 3 실행. `sweep_...zip` 또는 이전 `fitted_reverse_...zip`을 업로드합니다.
Drive와 best.pt가 필요 없습니다. 이전 결과 ZIP에서는 **보정 전 원래 입력**을 다시 사용합니다.
candidate RGB 일치 손실은 사용하지 않습니다. 후보에서는 삭제 영역과 누락 중심선만 얻습니다.
첫 시험은 삭제만 활성화합니다. 2번에서 마스크를 확인하고 3번을 실행하세요.

빨강=삭제 / 파랑=추가. 정상적인 경계 차이는 허용하고, 나머지 영역에는 입력 보존 손실을 약하게 적용합니다.
확산 모델은 전체 이미지를 예측하며 가이드는 국소 latent에만 적용됩니다.
자동 마스크도 틀릴 수 있으므로 빨강이 정상 획을 넓게 덮지 않는지 확인하세요.
원하면 2번의 ENABLE_RESTORE=True로 중심선 추가도 켤 수 있습니다.
이번 실험은 기준 획 정보가 있는 조건이며, 정답 없이 보정 성능이 입증됐다는 뜻은 아닙니다.
''')
    setup = ''.join(nb['cells'][1]['source'])
    setup = setup.replace("'safetensors', 'numpy', 'pillow', 'matplotlib'", "'safetensors', 'numpy', 'pillow', 'matplotlib', 'scipy', 'scikit-image'")
    put(1, setup)
    put(2, '''import json, zipfile
from google.colab import files
import matplotlib.pyplot as plt
import numpy as np

ZIP_PATH=''
CASE_ID='case_003_strength_0p65_seed_100'
ENABLE_RESTORE=False  # 첫 실행: 삭제만. 누락 중심선 추가 시험은 True.
TOLERANCE=5.          # 현재 576px 이미지 기준. 정상 윤곽 차이를 허용하는 거리.
CONSTRAINT_CONFIG=dict(enable_restore=ENABLE_RESTORE,tolerance=TOLERANCE,
                       halo=2,preserve_weight=.5,background_ink=.015,required_ink=.7)
if not ZIP_PATH:
    uploaded=files.upload()
    names=[name for name in uploaded if name.lower().endswith('.zip')]
    if len(names)!=1:raise ValueError('결과 ZIP 하나를 선택하세요.')
    ZIP_PATH=names[0]
with zipfile.ZipFile(ZIP_PATH) as z:
    available=(json.loads(z.read('case_manifest.json')) if 'case_manifest.json' in z.namelist()
               else [json.loads(z.read('report.json'))['metadata']['case']])
print('사용 가능한 사례:',[c['case_id'] for c in available])
source,candidate,metadata=repair.load_case(ZIP_PATH,CASE_ID)
preview=repair.InkConstraintObjective(repair.image_tensor(source,'cpu'),
    repair.image_tensor(candidate,'cpu'),**CONSTRAINT_CONFIG)
erase=preview.erase[0,0].numpy();restore=preview.restore[0,0].numpy()
overlay=np.asarray(source,dtype=np.float32)/255
alpha=.7*np.maximum(erase,restore)[...,None]
color=np.stack([erase,np.zeros_like(erase),restore],-1)
color=color/np.maximum(color.max(-1,keepdims=True),1e-8)
overlay=overlay*(1-alpha)+color*alpha
fig,axes=plt.subplots(1,4,figsize=(20,5))
for ax,im,title in zip(axes,[source,overlay,erase,restore],
        ['Original erroneous input','Red=erase / blue=restore','Erase pressure','Restore centerline']):
    ax.imshow(im,cmap='magma',vmin=0,vmax=1);ax.set_title(title);ax.axis('off')
plt.show();plt.close(fig)
print('制約 / 제약:',preview.metadata)
print('자동 마스크가 잘못됐으면 3번을 실행하지 말고 TOLERANCE를 조정하세요.')
'''.replace('制約 / ', ''))
    run = ''.join(nb['cells'][3]['source'])
    run = run.replace('STRENGTH=.35', 'STRENGTH=.25').replace('=21단계', '=15단계')
    run = run.replace('STEP_RATIO=.02', 'STEP_RATIO=.012')
    run = run.replace("'fitted_reverse_'", "'ink_constraint_reverse_'")
    run = run.replace('progress=progress)', "progress=progress,guidance_mode='ink_constraints',constraint_config=CONSTRAINT_CONFIG)")
    run = run.replace('Candidate-guided reverse diffusion', 'Sparse ink-guided reverse diffusion')
    run = run.replace('candidate_guided', 'ink_guided')
    run = run.replace('Guide ONLY', 'Candidate (not an RGB target)')
    run = run.replace('후보에 대한 손실(정확도 아님)', '잉크 제약과 원본 보존 손실(정확도 아님)')
    put(3, run)
    path = Path(__file__).with_name('ink_constraint_diffusion.ipynb')
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
    return path


if __name__ == '__main__':
    print(build())
