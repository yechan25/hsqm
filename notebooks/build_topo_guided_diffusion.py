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
    source(0,'''# 원본 기반 국소 역확산 편집 (실험)

기존 Topo는 사후 획 평가에 사용합니다. 보정은 기준과 비교한 삭제/복원/보호 지도에서 직접 계산합니다.
기본 STEPS=100으로 실제 역확산을 65단계 수행합니다(strength=0.65). 이전 설정은 39단계입니다.
후반부 첫 편집 단계에서 지도를 고정하고, 최대 23개 보정 단계마다 48회 잠재값을 수정합니다.
삭제 영역 주변 2픽셀 중 기준의 빈 배경만 추가로 정리하고, 마지막 단계는 최대 96회 보정합니다.
삭제 손실은 MSE + 0.5×MAE + 0.5×상위 10% 절대오차로 남은 잔상에 집중합니다.
정상 획 보호 한도와 한 번의 보정량은 기존대로이며, 실행 시간은 늘어납니다.
보호 영역의 평균/국소 변화와 잠재값 이동 예산을 제한합니다. 전체 추출 점수 상승은 채택 조건이 아닙니다.
기준과 생성물의 위치가 맞는 조건을 전제로 하며, 폰트·정렬 차이도 편집 대상으로 잡힐 수 있습니다.
실제 SD 생성 개선은 실행 후 확인해야 합니다. 새 체크포인트를 학습한 모델은 아닙니다.

1→2→3→4→5 순서로 실행합니다. 4번에서 baseline과 국소 편집을 동일 초기 잠재값으로 비교합니다.
삭제·복원·보호 지도는 local_erase.png / local_fill.png / local_protect.png로 저장됩니다.
''')
    setup=''.join(notebook['cells'][2]['source'])
    setup=setup.replace('STEPS = 30','STEPS = 100  # strength=.65에서 실제 역확산 65단계')
    setup=setup.replace('GLYPH_COUNTS = [1, 4, 9]','GLYPH_COUNTS = [4]')
    setup=setup.replace('STRENGTHS = [.65, .85]','STRENGTHS = [.65]')
    setup+='''
from diffusion_model.run_topo_guidance import generate_guided_pair, evaluate_topo_pair
GUIDANCE = dict(local_edit=True,local_iterations=48,local_halo_radius=2,local_final_iterations=96,local_erase_l1_weight=.5,local_erase_tail_weight=.5,clean_step_ratio=.01,start_fraction=.65,end_fraction=1.,
                every=1,backtracks=5,snapshot_every=4,state_update='clean_residual',
                healthy_stroke_score=90.,max_healthy_stroke_drop=1.,max_new_local_ink=.002)
MASK_GRADIENT = 'ink_only'  # Topo 배정을 바꾸려고 입력에 점을 찍는 미분 경로 제한
DIRECT_REFERENCE = True
SOURCE_SPILL_WEIGHT = .5  # 기준의 1픽셀 여유 밖 원본 잉크에 별도 보조 손실
STROKE_FOCUS = dict(focus_strength=2.,max_stroke_weight=3.)
COMPARE_UNIFORM = False  # 이전 균등 보정까지 추가 비교하려면 True (생성 경로 1개 추가)
COMPARE_STRUCTURE = False  # 동일 예산·새 잉크 검사로 구조 보정과 윤곽 추가를 비교
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
print('실제 역확산 단계:',generation['effective_steps'])
trace_path=OUTPUT_DIR/'trace_guided.jsonl'
active_records=[json.loads(line) for line in trace_path.read_text().splitlines() if json.loads(line).get('active')]
if active_records:
    last=active_records[-1]
    print('마지막 보정 종료 이유:',last.get('stop_reason'))
    print('삭제 영역 상위 10% 오차 전/후:',last.get('erase_tail_before'),last.get('erase_tail_after'))
    print('삭제 영역 잔여 잉크 평균/최대:',last.get('residual_mean_after'),last.get('residual_peak_after'))
    print('잠재값 이동 예산 사용 비율:',last.get('latent_budget_fraction'))
print('채택된 guidance 단계:',generation['guided_accepted_steps'])
print('baseline=보정 없음 / guided=국소 삭제·복원 반복 보정')
for name in ('erase','fill','protect'):
    path=OUTPUT_DIR/f'local_{name}.png'
    if path.exists():
        print(name);display(Image.open(path))
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
