"""Add controlled img2img generation and apply extraction/fitting to every output."""
import ast
import json
from pathlib import Path
from textwrap import indent
from build_stroke_reference_fit import build as build_single

ROOT=Path(__file__).resolve().parents[1]


def build():
    old=json.loads(build_single().read_text(encoding='utf-8'))
    nb=dict(old,cells=[])
    def add(kind,source):
        if kind=='code':ast.parse(source)
        cell=dict(cell_type=kind,metadata={},source=source.strip().splitlines(keepends=True))
        if kind=='code':cell.update(outputs=[],execution_count=None)
        nb['cells'].append(cell)
    add('markdown','''# 네 기준 글자 → 여러 img2img 생성물 → 획 추출·기준 획 변형

L4에서 1→2→3→4번을 실행합니다. Drive 연결과 기존 best.pt SHA256 검사를 유지합니다.
이번 실험은 정확한 네 글자 기준 이미지를 초기 입력으로 주는 img2img 통제 실험입니다. 프롬프트만으로 글자를 생성하는 시험이 아닙니다.
2번에서 같은 기준 이미지로 강도×seed 조합마다 독립적으로 생성합니다. 오류 없는 결과, 심한 오류, 안전 필터 결과까지 모두 저장합니다.
3·4번은 모든 생성물과 정상 입력 대조군에 획 추출/기준 획 변형을 각각 적용합니다. 자동으로 성공 사례만 고르지 않습니다.
강도를 높여도 특정 종류의 오류나 누락이 반드시 나오지는 않습니다. 심하게 무너진 입력의 후보도 실패/경고와 함께 남깁니다.
마지막 결과는 변형된 정상 후보입니다. 그 후보로 원래 생성물을 보정하는 추가 역확산은 아직 수행하지 않습니다.
''')
    setup=''.join(old['cells'][1]['source'])
    start=setup.index('runtime_helpers=');end=setup.index('for name,code in runtime_helpers.items():',start)
    helpers={p:(ROOT/p).read_text(encoding='utf-8') for p in (
        'diffusion_model/saved_repair_demo.py','stroke_model/stroke_reference_fit.py',
        'diffusion_model/multiglyph_pilot.py','diffusion_model/reference_fit_cases.py')}
    setup=setup[:start]+'runtime_helpers='+repr(helpers)+'\n'+setup[end:]
    setup=setup.replace("'transformers==4.57.6',","'diffusers==0.37.0','transformers==4.57.6',",1)
    setup=setup.replace("Path('/content/stroke_reference_fit_runs')","Path('/content/stroke_reference_fit_sweeps')")
    setup=setup.replace("print('GPU:',torch.cuda.get_device_name(0),'| 실제 입력:',demo['source'])",
        "print('GPU:',torch.cuda.get_device_name(0),'| 기준 획 자료:',demo['manifest'])")
    setup=setup.replace("display(Image.open(demo['source']))", "print('1번은 기준 획과 모델만 준비했습니다. 2번에서 새로운 생성물을 만듭니다.')")
    add('code',setup)
    add('code','''from diffusion_model.multiglyph_pilot import generate_multiglyph_pilot,compose_sheet
from diffusion_model.score_pilot import training_catalog
from diffusion_model.reference_fit_cases import cases_from_generation
from huggingface_hub import model_info

INPUT_MODE='generate'  # 'saved'이면 기존 오류 이미지 한 장으로 실행
STRENGTHS=(.45,.65,.85)
SEEDS=(100,101)
GENERATION_STEPS=100
MODEL_ID='stable-diffusion-v1-5/stable-diffusion-v1-5'
SHOW_DETAILS_FOR=(3,)  # 표시만 제한. 처리/저장은 전 케이스 수행. (3,)=.65/100
PROCESS_INPUT_CONTROL=True
CASE_ROOT=RUN_DIR/('sweep_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
CASE_ROOT.mkdir()
shutil.copytree(Path(demo['manifest']).parent,CASE_ROOT/'source_conditions')
if INPUT_MODE=='generate':
    MODEL_REVISION=model_info(MODEL_ID).sha
    generated=generate_multiglyph_pilot(demo['manifest'],demo['rows'],CASE_ROOT/'generation',
        glyph_counts=(4,),canvas_size=demo['size'],strengths=STRENGTHS,seeds=SEEDS,
        steps=GENERATION_STEPS,model_id=MODEL_ID,revision=MODEL_REVISION)
    CASES=cases_from_generation(generated,include_control=PROCESS_INPUT_CONTROL)
    # Release the temporary generation pipeline before loading the extractor.
    gc.collect();torch.cuda.empty_cache()
elif INPUT_MODE=='saved':
    CASES=[dict(case_id='case_saved',page_id=0,input=demo['source'],control=False,
        strength=None,seed=None,safety_filtered=None,label='Existing saved erroneous image',source_type='saved_baseline')]
else:raise ValueError("INPUT_MODE must be 'generate' or 'saved'")
(CASE_ROOT/'case_manifest.json').write_text(json.dumps(CASES,ensure_ascii=False,indent=2))
(CASE_ROOT/'experiment_info.json').write_text(json.dumps(dict(input_mode=INPUT_MODE,
    free_text_generation=False,reference_initialized_generation=INPUT_MODE=='generate',
    old_error_image_used_as_generation_input=False,case_count=len(CASES),
    show_details_for=SHOW_DETAILS_FOR,reference_manifest=demo['manifest'],checkpoint_sha256=actual_sha),indent=2))
fig,axes=plt.subplots((len(CASES)+2)//3,3,figsize=(15,5*((len(CASES)+2)//3)),squeeze=False)
for ax in axes.flat:ax.axis('off')
for ax,case in zip(axes.flat,CASES):
    with Image.open(case['input']) as image:ax.imshow(image.convert('RGB'))
    ax.set_title(f"{case['page_id']}: {case['label']}")
fig.tight_layout();fig.savefig(CASE_ROOT/'generation_gallery.png',dpi=120);plt.show();plt.close(fig)
print('전체 처리 케이스:',len(CASES),'| 상세 표시:',SHOW_DETAILS_FOR)
print('다음 3번과 4번은 CASES 전체를 처리합니다.')
''')
    extraction=''.join(old['cells'][2]['source'])
    begin=extraction.index("with Image.open(demo['source'])")
    end=extraction.index('extractor.cpu();')
    prefix=extraction[:begin]
    body=extraction[begin:end]
    body=body.replace("demo['source']","case['input']").replace('RUN_DIR/', 'case_dir/')
    ps=body.index('    fig,axes=plt.subplots');pe=body.index(';plt.show()',ps)+len(';plt.show()')
    plot=body[ps:pe].replace(';plt.show()',';plt.show();plt.close(fig)')
    body=body[:ps]+'    if show_details:\n'+indent(plot,'    ')+body[pe:]
    code=prefix+'''CASE_EXTRACTIONS=[]
def extract_one(case,case_dir,show_details):
'''+indent(body,'    ')+'''\n    return dict(case=case,case_dir=case_dir,source_array=source_array,extractions=extractions)
for case in CASES:
    case_dir=CASE_ROOT/case['case_id'];case_dir.mkdir(exist_ok=True)
    print('Extracting:',case['case_id'],flush=True)
    (case_dir/'case.json').write_text(json.dumps(case,ensure_ascii=False,indent=2))
    CASE_EXTRACTIONS.append(extract_one(case,case_dir,case['page_id'] in SHOW_DETAILS_FOR))
extractor.cpu();gc.collect();torch.cuda.empty_cache()
print('추출 완료:',len(CASE_EXTRACTIONS),'개. 4번에서 각 케이스의 기준 획을 변형합니다.')
'''
    add('code',code)
    fit=''.join(old['cells'][3]['source'])
    a=fit.index('results=[]');b=fit.index("(RUN_DIR/'runtime_helpers.json')")
    prefix=fit[:a]
    body=fit[a:b].replace('plt.show()','show_or_close(fig,show_details)')
    body=body.replace("display(pd.DataFrame(summary)[['slot','stroke','method','warnings']])",
        "if show_details:display(pd.DataFrame(summary)[['slot','stroke','method','warnings']])")
    body=body.replace("print('이 단계는 실제 역확산을 실행하지 않습니다.')",'')
    tail='''
    return dict(case_id=case['case_id'],page_id=case['page_id'],control=case['control'],
        seed=case['seed'],strength=case['strength'],safety_filtered=case['safety_filtered'],
        input=case['input'],candidate=str(RUN_DIR/'normal_candidate.png'),
        geometry_candidate=str(RUN_DIR/'geometry_candidate.png') if RUN_GEOMETRY_CONTROL else None,
        warning_strokes=len(warnings),fallback_strokes=sum('fallback' in r['method'] for r in summary),
        status='candidate_with_warnings' if warnings else 'no_fit_warnings_not_quality_verified')
'''
    code=prefix+'''def show_or_close(fig,show_details):
    if show_details:display(fig)
    plt.close(fig)

def fit_one(record):
    case=record['case'];RUN_DIR=record['case_dir']
    demo=dict(base_demo,source=case['input'])
    source_array=record['source_array'];extractions=record['extractions']
    show_details=case['page_id'] in SHOW_DETAILS_FOR
'''+indent(body,'    ')+tail+'''
base_demo=demo.copy();CASE_SUMMARIES=[]
for record in CASE_EXTRACTIONS:
    print('Fitting:',record['case']['case_id'],flush=True)
    CASE_SUMMARIES.append(fit_one(record))
frame=pd.DataFrame(CASE_SUMMARIES);display(frame)
frame.to_csv(CASE_ROOT/'all_cases_summary.csv',index=False)
fig,axes=plt.subplots(len(CASE_SUMMARIES),3,figsize=(15,5*len(CASE_SUMMARIES)),squeeze=False)
for row,result in enumerate(CASE_SUMMARIES):
    for ax,(key,title) in zip(axes[row],[('input','Generated input'),('candidate','Fitted candidate'),('geometry_candidate','No learned extraction')]):
        path=result[key]
        if path:
            with Image.open(path) as image:ax.imshow(image.convert('RGB'))
        ax.set_title(f"{result['case_id']}\\n{title}");ax.axis('off')
fig.tight_layout();fig.savefig(CASE_ROOT/'all_cases_comparison.png',dpi=100);plt.show();plt.close(fig)
(CASE_ROOT/'runtime_helpers.json').write_text(json.dumps(runtime_helpers,ensure_ascii=False))
(CASE_ROOT/'git_commit.txt').write_text(commit)
archive=CASE_ROOT.with_suffix('.zip')
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for path in CASE_ROOT.rglob('*'):
        if path.is_file():z.write(path,path.relative_to(CASE_ROOT))
drive_folder=HSQM/'stroke_reference_fit_sweep_results';drive_folder.mkdir(exist_ok=True)
shutil.copy2(archive,drive_folder/archive.name)
print('Drive 전체 결과:',drive_folder/archive.name)
print('누락/붕괴/경고 케이스도 모두 포함했습니다. 후보가 정상이라는 뜻은 아닙니다.')
files.download(str(archive))
'''
    add('code',code)
    path=Path(__file__).with_name('stroke_reference_fit_sweep.ipynb')
    path.write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf-8')
    return path


if __name__=='__main__':print(build())
