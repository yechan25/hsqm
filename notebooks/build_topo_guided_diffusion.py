"""Build the paired four-glyph reverse-diffusion guidance Colab."""
import ast
import json
from pathlib import Path
from build_multiglyph_diffusion_pilot import build_multiglyph

ROOT=Path(__file__).resolve().parents[1]


def build_guided():
    notebook=json.loads(build_multiglyph().read_text(encoding='utf-8'))
    def source(index,text):
        text=text.strip()
        if notebook['cells'][index]['cell_type']=='code':ast.parse(text)
        notebook['cells'][index]['source']=text.splitlines(keepends=True)
    source(0,'''# 4글자 공동 생성 — Topo 추출 획으로 역확산 유도

Stable Diffusion 1.5의 **잡음 예측 U-Net + DDIM**, strength **0.65**, seed **100**.
학습 범위의 기준 글자 4개를 한 장에 놓고 공동 생성합니다. 기본 경로와 guidance 경로는
VAE 샘플링과 잡음 추가까지 완료한 **동일한 초기 잠재값을 복사해** 시작합니다.

각 guidance 단계에서 예상 완성 잠재값 → fp32 VAE → 4칸 torch crop/resize → 동결 Topo
→ 실제 예측 soft 획의 누락/분리/정션 손실을 계산합니다. 이 손실을 예상 완성 잠재값에
미분해 잡음 예측을 보정합니다. U-Net을 통한 역전파와 모델 가중치 학습은 하지 않습니다.
원본 점수가 주 손실을 대신하지 않습니다. 주변 잉크에는 작은 별도 보조 항만 적용합니다.

보정량은 예상 완성 잠재값 RMS의 1%에서 시작하며, 같은 시점의 손실이 내려가고
4글자의 평균 추출 획 점수가 낮아지지 않아야 채택합니다.
연결 성분 분리·이진 윤곽 점수는 guidance에 사용하지 않습니다. 모델 체크포인트는 읽기만 합니다.

1. 준비·Drive → 2. 기준 자료만 준비(학습 NPZ 일괄 변환 없음)
3. 정확히 4개 기준 선택 → 4. 기본/유도 공동 생성과 원본 비교 → 5. **추출 획 기반 점수** 비교

Colab GPU에서 실행하세요. 역전파가 추가되어 기존 생성보다 느리고 메모리를 더 사용합니다.
검사는 실제 DDIM과 작은 대체 네트워크를 사용합니다. 사용자의 SD/Topo GPU 결과는
실행 후 확인해야 하며, 손실 하락은 최종 사람 평가 품질 개선을 보장하지 않습니다.''')
    setup=''.join(notebook['cells'][2]['source'])
    setup=setup.replace('GLYPH_COUNTS = [1, 4, 9]','GLYPH_COUNTS = [4]')
    setup=setup.replace('STRENGTHS = [.65, .85]','STRENGTHS = [.65]')
    setup+='''
from diffusion_model.run_topo_guidance import generate_guided_pair, evaluate_topo_pair
GUIDANCE = dict(clean_step_ratio=.01,start_fraction=.30,end_fraction=1.,
                every=1,backtracks=2,snapshot_every=4)
STRUCTURE_CONFIG = {}  # 기존 StrokeStructureLoss 기본 측정. 임의의 반복 상한으로 자르지 않습니다.
EXTRA_AUX_WEIGHT = .05
'''
    source(2,setup)
    preview=''.join(notebook['cells'][3]['source'])
    preview=preview.replace("print(f'최소 {needed}개를 선택하세요. 각 개수 조건은 선택 목록의 앞부분을 사용합니다.')",
                            "print('정확히 4개를 선택하세요. 같은 4개로 기본/유도 경로를 비교합니다.')")
    source(3,preview)
    source(4,'''
from datetime import datetime
from huggingface_hub import model_info
MODEL_REVISION = model_info(MODEL_ID).sha
OUTPUT_DIR = HSQM/'topo_guided_diffusion'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
generation = generate_guided_pair(TRAIN_MANIFEST,list(reference_select.value),CHECKPOINT,OUTPUT_DIR,
    model_id=MODEL_ID,revision=MODEL_REVISION,steps=STEPS,canvas_size=CANVAS_SIZE,
    guidance=GUIDANCE,structure_config=STRUCTURE_CONFIG,extra_weight=EXTRA_AUX_WEIGHT)
generation['data_preparation']=preparation
(OUTPUT_DIR/'generation_report.json').write_text(json.dumps(generation,ensure_ascii=False,indent=2),encoding='utf-8')
(OUTPUT_DIR/'git_commit.txt').write_text(commit,encoding='utf-8')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,3,figsize=(15,5))
for ax,page in zip(axes,generation['pages']):
    with Image.open(page['input']) as image:ax.imshow(image.convert('RGB'))
    ax.set_title(page['method']);ax.axis('off')
plt.tight_layout();plt.show()
print('채택된 guidance 단계:',generation['guided_accepted_steps'])
print('동일한 시작 잠재값:',generation['initial_latent_sha256'])
print('원본·예상 완성 이미지·보정 전후 손실:',OUTPUT_DIR)
from diffusion_model.guidance_diagnostics import diagnose_pair
diagnostic_summary,pixel_differences=diagnose_pair(OUTPUT_DIR,show=True)
display(pixel_differences)
''')
    source(5,'''
import pandas as pd
from diffusion_model.guidance_diagnostics import diagnose_pair
# 기존 실행 결과도 재생성 없이 PNG와 로그만 읽어 진단합니다.
diagnostic_summary,pixel_differences=diagnose_pair(OUTPUT_DIR,show=True)
display(pixel_differences)
topo_scores,topo_extraction,debug_zip=evaluate_topo_pair(
    OUTPUT_DIR/'generation_report.json',CHECKPOINT,show=True)
display(topo_scores[['method','slot','extracted_stroke_score_no_extra',
    'stroke_loss','missing','split','junction','extra_auxiliary']])
paired=topo_scores[topo_scores['method']!='reference'].pivot(
    index='slot',columns='method',values='extracted_stroke_score_no_extra')
paired['guided_minus_baseline']=paired['guided']-paired['baseline']
display(paired)
print('주 점수는 Topo가 실제 추출한 soft 획에서 계산합니다. extra는 주 점수에서 제외하고 보조 항으로 별도 기록합니다.')
print('점수 상승과 원본 글자의 실제 개선을 함께 확인하세요. 채널 재배정만 좋아진 경우도 있을 수 있습니다.')
RUN_INDEPENDENT_SOURCE_AUDIT = False  # 원본 직접 측정까지 추가 검증하려면 True
if RUN_INDEPENDENT_SOURCE_AUDIT:
    from stroke_model.topology_scoring import score_topology_results
    audit=score_topology_results(topo_extraction,show=False)
    print('아래는 독립 원본 검사이며 Topo 주 점수와 다릅니다.')
    display(pd.DataFrame(audit['indicator']['results'])[['sample','structural_score']])
from google.colab import files
button=widgets.Button(description='역확산 비교 ZIP 다운로드')
button.on_click(lambda _,path=debug_zip:files.download(path))
display(button)
''')
    notebook['metadata']['colab']['name']='Topo_추출획_역확산_guidance.ipynb'
    path=ROOT/'notebooks/topo_guided_diffusion.ipynb'
    path.write_text(json.dumps(notebook,ensure_ascii=False,indent=2),encoding='utf-8')
    print(path)
    return path


if __name__=='__main__':build_guided()
