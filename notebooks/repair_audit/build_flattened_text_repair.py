"""Build a Colab that accepts only a flattened image and four reference glyphs."""
import ast
import hashlib
import json
from pathlib import Path

from build_selectable_target_repair import build as build_base

ROOT = Path(__file__).resolve().parents[2]


def build():
    base = build_base(ink_only=True,write=False)
    codes = [''.join(cell['source']) for cell in base['cells'] if cell['cell_type']=='code']
    setup = codes[0]
    helpers = next(ast.literal_eval(node.value) for node in ast.parse(setup).body
        if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name)
        and node.targets[0].id=='RUNTIME_HELPERS')
    name = 'repair_audit/flattened_text_repair.py'
    helpers[name] = (ROOT/name).read_text(encoding='utf-8')
    hashes = {name:hashlib.sha256(code.encode()).hexdigest() for name,code in helpers.items()}
    setup = '\n'.join('RUNTIME_HELPERS='+repr(helpers) if line.startswith('RUNTIME_HELPERS=') else
        'SOURCE_HASHES='+repr(hashes) if line.startswith('SOURCE_HASHES=') else line for line in setup.splitlines())
    setup += '''\nfrom io import BytesIO
from PIL import ImageOps
from repair_audit.flattened_text_repair import prepare_flattened_plan, inpaint_deleted_ink, FlattenedTextGuide
from diffusion_model.stroke_projected_diffusion import run_projected_pair
'''
    cells = []

    def add(kind, source):
        source = source.strip()+'\n'
        if kind=='code':ast.parse(source)
        cells.append(dict(cell_type=kind,metadata={},source=source.splitlines(keepends=True),
            **(dict(execution_count=None,outputs=[]) if kind=='code' else {})))

    add('markdown','''# 배경이 합쳐진 글자 이미지 보정

정사각형 2×2 입력 한 장과 칸별 정답 기준 획을 사용합니다. 입력은 576×576으로 맞춥니다. 짙은 글자와 상대적으로 밝은 배경에서 시험하세요.
**4번에서 기준 글자 순서 지정 → 5번에서 완성 이미지 업로드 → 6번 모델 로딩 → 7번 삭제·추가 마스크 확인 → 8번 실행 → 9번 ZIP.**
7번에서 어둡고 채도가 낮은 잉크를 흰 배경에 남겨 획을 추출합니다. 빨강은 원본에서 인페인팅할 곳, 파랑은 잉크를 추가할 곳입니다.
가려진 배경은 주변을 보고 추정합니다. 글자 위치는 2×2의 각 칸으로 지정합니다.
''')
    add('code',setup)
    config = '\n'.join(line for line in codes[1].splitlines()
        if not line.startswith(('GENERATION_','PLAN_CONFIG','DRIVE_RESULT_DIR','PROTOTYPE_BANK')))
    add('code',config)
    add('code',codes[2].split('bank_index =',1)[0]+"print('기준 목록 준비 완료. 4번에서 목표 글자를 고르세요.')")
    add('code','''# 4. 각 칸의 목표 글자와 기준 획 선택
entries_by_label = {f"{e.get('resolved_character') or '?'} | row {e['manifest_row']} | {(e.get('reference') or '').rsplit('/',1)[-1]}":e for e in CATALOG}
labels = list(entries_by_label)
choices = [widgets.Combobox(options=labels,value=labels[min(i,len(labels)-1)],
    ensure_option=True,description=name,continuous_update=False,layout=widgets.Layout(width='90%'))
    for i,name in enumerate(['왼쪽 위','오른쪽 위','왼쪽 아래','오른쪽 아래'])]
preview = widgets.Output()
def show_reference(change=None):
    with preview:
        preview.clear_output(wait=True)
        if any(choice.value not in entries_by_label for choice in choices):return
        entries = [entries_by_label[choice.value] for choice in choices]
        display(canonical_sheet([dict(reference=load_reference_image(entry)) for entry in entries],256))
for choice in choices:choice.observe(show_reference,names='value')
display(widgets.VBox([*choices,preview]));show_reference()
''')
    add('code','''# 5. 배경과 글자가 합쳐진 이미지 업로드
source = None
uploaded = files.upload()
if len(uploaded)!=1:raise ValueError('정사각형 2×2 입력 이미지 한 장을 올리세요.')
source_name, contents = next(iter(uploaded.items()))
with Image.open(BytesIO(contents)) as image:
    image = ImageOps.exif_transpose(image).convert('RGB')
    if image.width != image.height:raise ValueError('2×2 시트는 정사각형이어야 합니다.')
    source = image.resize((CANVAS_SIZE,CANVAS_SIZE),Image.Resampling.LANCZOS)
display(source.resize((320,320)))
''')
    add('code',codes[5].replace("print('로딩 완료. 7번 실행.')", "print('로딩 완료. 7번에서 마스크를 확인하세요.')"))
    add('code','''# 7. 획 추출 및 삭제·추가 마스크 확인
plan = None
if source is None:raise ValueError('5번에서 입력을 올리세요.')
entries = [entries_by_label[choice.value] for choice in choices]
planned_choices = [choice.value for choice in choices]
conditions = [dict(zip(('reference','strokes'),load_catalog_reference(entry))) for entry in entries]
if any(c['reference'].shape[-2:]!=(EXTRACTOR_SIZE,EXTRACTOR_SIZE) for c in conditions):
    raise ValueError('기준 이미지 해상도가 추출기와 다릅니다.')
try:
    extractor.to('cuda')
    plan = prepare_flattened_plan(source,conditions,extractor)
finally:
    extractor.to('cpu');torch.cuda.empty_cache()
overlay = np.asarray(source).copy()
overlay[plan.erase] = [255,64,64]
overlay[plan.add] = [75,155,255]
panel = Image.new('RGB',(960,320))
for index,image in enumerate([source,plan.proxy,Image.fromarray(overlay)]):
    panel.paste(image.resize((320,320)),(index*320,0))
print('왼쪽 원본 / 가운데 획 추출용 흰 배경 / 오른쪽 삭제(빨강)·추가(파랑)')
display(panel)
''')
    add('code','''# 8. 잘못된 획 자리 인페인팅 → 목표 획 가이드 역확산
from diffusers import AutoPipelineForInpainting
if plan is None:raise ValueError('7번에서 마스크를 먼저 확인하세요.')
if source is None or planned_choices!=[choice.value for choice in choices] or not np.array_equal(np.asarray(plan.source),np.asarray(source)):
    raise ValueError('입력이나 기준이 바뀌었습니다. 7번에서 마스크를 다시 만드세요.')
OUTPUT = Path('/content')/('flattened_text_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
OUTPUT.mkdir()
source.save(OUTPUT/'input.png');plan.proxy.save(OUTPUT/'ink_proxy.png')
Image.fromarray(plan.erase.astype(np.uint8)*255).save(OUTPUT/'erase_mask.png')
Image.fromarray(plan.add.astype(np.uint8)*255).save(OUTPUT/'add_mask.png')
json_write(OUTPUT/'run.json',dict(plan.report,source_name=source_name,reference_rows=[e['manifest_row'] for e in entries],
    checkpoint_sha256=file_hash(CHECKPOINT),source_hashes=SOURCE_HASHES,versions=VERSIONS,model_id=MODEL_ID,revision=REVISION))
try:
    pipe.vae.to(dtype=pipe.unet.dtype)
    inpaint = AutoPipelineForInpainting.from_pipe(pipe)
    inpaint.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    filled = inpaint_deleted_ink(plan,inpaint,steps=25)
finally:
    pipe.vae.to(dtype=torch.float32)
filled.save(OUTPUT/'inpainted.png')
guide = FlattenedTextGuide(plan,filled)
target = Image.fromarray(np.rint(guide.target*255).astype(np.uint8))
target.save(OUTPUT/'target.png')
report = run_projected_pair(pipe,source,guide,OUTPUT/'reverse',steps=60,strength=.25,
    prompt='dark Korean typography in its original scene',guided_only=True)
with Image.open(OUTPUT/'reverse/stroke_projected.png') as result:
    panel = Image.new('RGB',(960,240))
    for index,image in enumerate([source,filled,target,result]):
        panel.paste(image.resize((240,240)),(index*240,0))
    panel.save(OUTPUT/'comparison.png');display(panel)
print('왼쪽부터 입력 / 삭제 자리 메움 / 획 추가 목표 / 최종 보정')
''')
    add('code',"# 9. 결과 ZIP 다운로드\nfiles.download(shutil.make_archive(str(OUTPUT),'zip',OUTPUT))")
    for i,cell in enumerate(cells):cell['id']=f'flattened-text-{i}'
    filename = 'flattened_text_repair.ipynb'
    base['metadata']['colab']['name'] = filename
    path = ROOT/'notebooks/repair_audit'/filename
    path.write_text(json.dumps(dict(base,cells=cells),ensure_ascii=False,indent=1),encoding='utf-8')
    return path


if __name__=='__main__':print(build())
