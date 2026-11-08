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
학습 범위의 기준 글자 4개를 한 장에 놓고 공동 생성합니다. 보정 없음/획 구조 보정/윤곽 보조 보정은
VAE 샘플링과 잡음 추가까지 완료한 **동일한 초기 잠재값을 복사해** 시작합니다.

각 guidance 단계에서 예상 완성 잠재값 → fp32 VAE → 4칸 torch crop/resize → 동결 Topo
→ 실제 예측 soft 획의 누락/분리/정션 손실을 계산합니다. 이 손실을 예상 완성 잠재값에
미분해 잡음 예측을 보정합니다. U-Net을 통한 역전파와 모델 가중치 학습은 하지 않습니다.
기존 추출 획 손실을 유지하면서, soft 획의 기준 대비 윤곽·국소 잉크량 오차를 보조 손실로 추가합니다.
이는 기준 형태·글씨체에 가까워지는 유도입니다. 물리적 획 두께 측정이나 새 구조 점수는 아닙니다.

보정량은 예상 완성 잠재값 RMS의 1%에서 시작하며, 같은 시점의 손실이 내려가고
4글자의 평균 추출 획 점수가 낮아지지 않아야 채택합니다.
보정한 완성 잠재값을 다음 상태에 넣을 때 U-Net의 잡음 방향을 유지해 보정량 상쇄를 줄입니다.
윤곽 기준은 정상 기준 이미지에 대한 **같은 Topo의 추출 결과**로 한 번 고정합니다.
정상 기준의 윤곽 오차는 0인지 확인하고, 기준 마스크 배정 차이를 결함으로 보정하지 않습니다.
실제 soft 획은 그대로 평가하되, 미분에서는 획 배정을 고정하고 잉크량만 수정합니다.
원본 해상도에서 새 국소 잉크가 늘어나는 후보를 검사하고, 최종 이미지도 별도로 확인합니다.
획별 집중은 구조 결함과 윤곽 결함 중 큰 쪽을 사용해 최대 3배 비중을 주고, 후보 비교 중 고정합니다.
90점 이상인 획이 같은 시점에서 1점 넘게 악화되면 보정을 거부합니다.
연결 성분 분리·이진 윤곽 점수는 guidance에 사용하지 않습니다. 모델 체크포인트는 읽기만 합니다.

1. 준비·Drive → 2. 기준 자료만 준비(학습 NPZ 일괄 변환 없음)
3. 정확히 4개 기준 선택 → 4. 세 경로 공동 생성과 원본 비교 → 5. **획별 구조 점수·윤곽 오차·수정 방향** 비교

Colab GPU에서 실행하세요. 역전파가 추가되어 기존 생성보다 느리고 메모리를 더 사용합니다.
검사는 실제 DDIM과 작은 대체 네트워크를 사용합니다. 사용자의 SD/Topo GPU 결과는
실행 후 확인해야 하며, 손실 하락은 최종 사람 평가 품질 개선을 보장하지 않습니다.''')
    setup=''.join(notebook['cells'][2]['source'])
    setup=setup.replace('GLYPH_COUNTS = [1, 4, 9]','GLYPH_COUNTS = [4]')
    setup=setup.replace('STRENGTHS = [.65, .85]','STRENGTHS = [.65]')
    setup+='''
from diffusion_model.run_topo_guidance import generate_guided_pair, evaluate_topo_pair
GUIDANCE = dict(clean_step_ratio=.01,start_fraction=.30,end_fraction=1.,
                every=1,backtracks=5,snapshot_every=4,state_update='clean_residual',
                healthy_stroke_score=90.,max_healthy_stroke_drop=1.,max_new_local_ink=.002)
MASK_GRADIENT = 'ink_only'  # Topo 배정을 바꾸려고 입력에 점을 찍는 미분 경로 제한
DIRECT_REFERENCE = True
SOURCE_SPILL_WEIGHT = .5  # 기준의 1픽셀 여유 밖 원본 잉크에 별도 보조 손실
STROKE_FOCUS = dict(focus_strength=2.,max_stroke_weight=3.)
COMPARE_UNIFORM = False  # 이전 균등 보정까지 추가 비교하려면 True (생성 경로 1개 추가)
COMPARE_STRUCTURE = True  # 동일 예산·새 잉크 검사로 구조 보정과 윤곽 추가를 비교
APPEARANCE_WEIGHT = .5  # 0이면 윤곽 보조 손실만 끔
APPEARANCE_TARGET = 'extractor_reference'  # 정상 기준의 실제 Topo 반응으로 보조 손실 기준 고정
APPEARANCE_CONFIG = dict(mass_radii=(1,2),mass_allowance=.03,
                         contour_allowance=.05,contour_weight=.5,priority_scale=.1)
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
import inspect
from diffusion_model.run_topo_guidance import generate_guided_pair, evaluate_topo_pair
if 'direct_reference' not in inspect.signature(generate_guided_pair).parameters:
    raise RuntimeError('실행 코드가 이전 버전입니다. 최신 노트북에서 1 → 2 → 3 → 4번 셀을 다시 실행하세요.')
from huggingface_hub import model_info
MODEL_REVISION = model_info(MODEL_ID).sha
OUTPUT_DIR = HSQM/'topo_guided_diffusion'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
generation = generate_guided_pair(TRAIN_MANIFEST,list(reference_select.value),CHECKPOINT,OUTPUT_DIR,
    model_id=MODEL_ID,revision=MODEL_REVISION,steps=STEPS,canvas_size=CANVAS_SIZE,
    guidance=GUIDANCE,structure_config=STRUCTURE_CONFIG,extra_weight=EXTRA_AUX_WEIGHT,
    stroke_focus=STROKE_FOCUS,compare_uniform=COMPARE_UNIFORM,
    appearance_weight=APPEARANCE_WEIGHT,appearance_config=APPEARANCE_CONFIG,
    compare_structure=COMPARE_STRUCTURE,appearance_target=APPEARANCE_TARGET,
    mask_gradient=MASK_GRADIENT,source_spill_weight=SOURCE_SPILL_WEIGHT,direct_reference=DIRECT_REFERENCE)
generation['data_preparation']=preparation
(OUTPUT_DIR/'generation_report.json').write_text(json.dumps(generation,ensure_ascii=False,indent=2),encoding='utf-8')
(OUTPUT_DIR/'git_commit.txt').write_text(commit,encoding='utf-8')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,len(generation['pages']),figsize=(5*len(generation['pages']),5))
for ax,page in zip(axes,generation['pages']):
    with Image.open(page['input']) as image:ax.imshow(image.convert('RGB'))
    ax.set_title(page['method']);ax.axis('off')
plt.tight_layout();plt.show()
print('채택된 guidance 단계:',generation['guided_accepted_steps'])
print('baseline=보정 없음 / structure_guided=획 구조 보정 / guided=윤곽·두께 보조 손실 추가')
print('보정 전달 방식:',generation['guidance_config']['state_update'])
print('정상 기준의 윤곽 오차 검증:',generation['reference_calibration'])
print('최종 원본 잉크 검사:',generation['source_ink_audit'])
if generation.get('quality_review_required'):
    print('최종 원본에서 새 국소 잉크 증가가 감지됐습니다. 이 결과는 점수 상승만으로 개선으로 판정하면 안 됩니다.')
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
if (OUTPUT_DIR/'stroke_guidance_trace.csv').exists():
    stroke_trace=pd.read_csv(OUTPUT_DIR/'stroke_guidance_trace.csv')
    print('실제 역확산 단계에서 사용한 획별 가중치와 전후 점수:')
    display(stroke_trace.tail(40))
topo_scores,topo_extraction,debug_zip=evaluate_topo_pair(
    OUTPUT_DIR/'generation_report.json',CHECKPOINT,show=True)
display(topo_scores[['method','slot','extracted_stroke_score_no_extra',
    'stroke_loss','appearance_loss','contour','local_mass','missing','split','junction','extra_auxiliary']])
paired=topo_scores[topo_scores['method']!='reference'].pivot(
    index='slot',columns='method',values='extracted_stroke_score_no_extra')
paired['guided_minus_baseline']=paired['guided']-paired['baseline']
if 'uniform_guided' in paired:
    paired['guided_minus_uniform']=paired['guided']-paired['uniform_guided']
if 'structure_guided' in paired:
    paired['guided_minus_structure']=paired['guided']-paired['structure_guided']
display(paired)
evaluation_dir=Path(debug_zip).parent
stroke_scores=pd.read_csv(evaluation_dir/'stroke_scores.csv')
display(stroke_scores[['method','slot','stroke','score','weight','weight_share','missing','split','junction',
                       'appearance_loss','contour','local_mass']])
display(pd.read_csv(evaluation_dir/'stroke_comparison.csv'))
print('윤곽·국소 잉크량 오차는 낮을수록 기준과 가깝습니다. *_minus_guided가 양수이면 오차 감소입니다.')
display(pd.read_csv(evaluation_dir/'appearance_comparison.csv'))
print('결함 지도와 획별 RGB 손실 기울기: 빨강은 어둡게, 파랑은 밝게 하는 방향입니다. 각 그림의 색은 따로 확대 표시합니다.')
print('RGB 기울기는 입력 이미지에 대한 방향 진단이며, 실제 잠재값 보정 결과와 동일한 그림은 아닙니다.')
print('최종 그림의 가중치는 해당 이미지에서 재계산한 진단값이며, 실제 단계별 가중치는 stroke_guidance_trace.csv에 있습니다.')
print('주 점수는 Topo가 실제 추출한 soft 획에서 계산합니다. extra는 주 점수에서 제외하고 보조 항으로 별도 기록합니다.')
print('구조 점수 공식은 이전과 같습니다. 윤곽 보조 손실과 원본 이미지를 함께 비교하세요.')
print('새 국소 잉크 지도: new_ink_local_difference.png. 원본 후처리로 점을 지우지 않고 최종 결과를 검사합니다.')
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
