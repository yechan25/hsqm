"""Build the Colab that maps all strokes and denoises their composite once."""
import ast
import hashlib
import json
from pathlib import Path
from build_selectable_target_repair import build as build_base

ROOT=Path(__file__).resolve().parents[2]


def build():
    base=json.loads(build_base().read_text(encoding='utf-8'))
    codes=[''.join(c['source']) for c in base['cells'] if c['cell_type']=='code']
    setup=codes[0]
    helpers=next(ast.literal_eval(n.value) for n in ast.parse(setup).body if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='RUNTIME_HELPERS')
    for name in ['stroke_model/stroke_reference_fit.py','stroke_model/attribute_guidance.py']:
        helpers[name]=(ROOT/name).read_text(encoding='utf-8')
    hashes={name:hashlib.sha256(code.encode()).hexdigest() for name,code in helpers.items()}
    setup='\n'.join('RUNTIME_HELPERS='+repr(helpers) if line.startswith('RUNTIME_HELPERS=') else
        'SOURCE_HASHES='+repr(hashes) if line.startswith('SOURCE_HASHES=') else line for line in setup.splitlines())
    setup+='\nfrom stroke_model.attribute_guidance import mapped_sheet_plan\nfrom diffusion_model.stroke_projected_diffusion import run_projected_pair\n'
    cells=[]
    def add(kind,source):
        source=source.strip()+'\n'
        if kind=='code':ast.parse(source)
        cells.append(dict(cell_type=kind,metadata={},source=source.splitlines(keepends=True),
            **(dict(execution_count=None,outputs=[]) if kind=='code' else {})))
    add('markdown','''# 획별 매핑 → 목표 한 장 → 재역확산

GPU 코랩에서 **1→6** 실행. 기존 모델·가중치 유지. 추가 학습 없음.
**2번 글자 선택 → 3번 생성·획 추출 → 4번 합친 목표 확인 → 5번 재역확산 → 6번 ZIP 다운로드.**
각 기준 획을 해당 관측 획에 따로 맞추고 대표 두께로 그립니다. 모든 획을 합친 목표 한 장을 사용합니다.
흰색/단순 배경용 실험입니다. 목표에서 획끼리 만나는 부분을 확인하세요.
''')
    add('code',setup)
    config=codes[1].replace('GENERATION_SEEDS = [100, 101]','GENERATION_SEEDS = [100]')
    config=config[:config.index("print('기준 목록:")]
    config=config.replace("PROTOTYPE_BANK = '/content/drive/MyDrive/HSQM_fixed_character_prototypes_v1'",
                          "PROTOTYPE_BANK = '/content/attribute_reference_bank'")
    config='\n'.join(line for line in config.splitlines() if not line.startswith(
        ('GENERATION_','SCORES_JSON','REPAIR_STEPS','PLAN_CONFIG')))+'\n'
    config+='''
BASELINE_FONT = 'NanumGothic'  # 모든 입력 폰트에 공유할 기준 폰트 한 종류
CATALOG = prepare_selection_cache(SOURCE_CATALOG,'/content/hsqm_picker_cache',source_fingerprint=file_hash(TRAIN_MANIFEST))
SELECTION_THUMBNAILS = selection_thumbnails(CATALOG)
def reference_font(entry):
    value=entry.get('reference','') or entry.get('source_entry',{}).get('reference','')
    stem=Path(value.replace('\\\\','/')).stem
    return stem.rsplit('_',1)[0] if '_' in stem else '(폰트명 없음)'
BASELINES={e['resolved_character']:e for e in CATALOG
    if e.get('resolved_character') and reference_font(e)==BASELINE_FONT}
if not BASELINES: raise ValueError('기준 폰트 '+BASELINE_FONT+'가 없습니다. 데이터 경로를 확인하세요.')
AVAILABLE={f"{e['resolved_character']} | {reference_font(e)} | {e['manifest_row']}":e
    for e in CATALOG if e.get('resolved_character') in BASELINES}
preferred=[k for k,e in AVAILABLE.items() if reference_font(e)!=BASELINE_FONT] or list(AVAILABLE)
glyph_choices=[widgets.Combobox(options=list(AVAILABLE),value=preferred[min(i,len(preferred)-1)],
    ensure_option=True,description=f'칸 {i+1}',continuous_update=False) for i in range(4)]
glyph_preview=widgets.Output()
def chosen_entries():
    if any(w.value not in AVAILABLE for w in glyph_choices): raise ValueError('목록에서 글자를 선택하세요.')
    return [AVAILABLE[w.value] for w in glyph_choices]
def preview_glyphs(change=None):
    with glyph_preview:
        glyph_preview.clear_output(wait=True)
        try: display(canonical_sheet([dict(reference=load_reference_image(e)) for e in chosen_entries()],320))
        except ValueError as exc: print(str(exc))
for w in glyph_choices:w.observe(preview_glyphs,names='value')
display(widgets.VBox([widgets.HTML('글자 이름/폰트로 선택. 기준은 '+BASELINE_FONT+' 한 종류를 자동 사용합니다.'),
    *glyph_choices,glyph_preview]))
preview_glyphs()
'''
    add('code',config)
    load=codes[4].replace('# 5.','# 3.')
    load=load[:load.index("print('로딩 완료.")]
    load+='''
GUIDE=None
entries=chosen_entries();keys=[e['resolved_character'] for e in entries]
for key in set(keys): save_prototype(CATALOG,BASELINES[key]['manifest_row'],key,PROTOTYPE_BANK)
OUTPUT=Path('/content')/('attribute_straightness_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
OUTPUT.mkdir()
controls=prepare_selected_sheet(CATALOG,[e['manifest_row'] for e in entries],OUTPUT/'reference',
    prototype_keys=keys,prototype_bank=PROTOTYPE_BANK,size=CANVAS_SIZE)
def save_snapshot():
    archive=shutil.make_archive(str(OUTPUT)+'_checkpoint','zip',OUTPUT)
    destination=Path(DRIVE_RESULT_DIR);destination.mkdir(parents=True,exist_ok=True)
    shutil.copy2(archive,destination/Path(archive).name)
    return archive
try:
    pipe.vae.to(dtype=torch.float16)
    CASES=generate_fresh_cases(pipe,controls,OUTPUT/'inputs',strengths=[.65],seeds=[100],steps=60,
        on_saved=lambda *args:save_snapshot())
finally: pipe.vae.to(dtype=torch.float32)
case=CASES[-1]
root=Path(case['directory']);source=Image.open(root/'input.png').convert('RGB')
conditions=load_conditions(root)
try:
    extractor.to('cuda')
    masks=extract_sheet_masks(source,conditions,extractor)
    for i,mask in enumerate(masks,1):np.savez_compressed(root/f'extracted_{i}.npz',masks=mask)
finally:extractor.to('cpu');torch.cuda.empty_cache();save_snapshot()
json_write(OUTPUT/'runtime_manifest.json',dict(source_hashes=SOURCE_HASHES,
    checkpoint_sha256=file_hash(CHECKPOINT),model_id=MODEL_ID,revision=REVISION,
    baseline_font=BASELINE_FONT,selected_rows=[e['manifest_row'] for e in entries],characters=keys,
    settings=dict(generation_steps=60,generation_strength=.65,generation_seed=100,
        repair_steps=120,repair_strength=.25,repair_seed=2026)))
json_write(OUTPUT/'versions.json',VERSIONS)
save_snapshot()
print('생성·추출 완료. 4번에서 합친 목표를 확인하세요.')
'''
    add('code',load)
    add('code','''# 4. 모든 획을 따로 매핑하고 목표 한 장으로 합치기
GUIDE=None
GUIDE=mapped_sheet_plan(source,masks,conditions)
target=Image.fromarray(np.uint8(np.rint(GUIDE.target*255)))
target.save(OUTPUT/'target.png')
json_write(OUTPUT/'mapping.json',GUIDE.report)
canvas=Image.new('RGB',(576,312),'white');draw=ImageDraw.Draw(canvas)
draw.text((5,5),'input',fill='black');draw.text((293,5),'combined target',fill='black')
canvas.paste(source.resize((288,288)),(0,24));canvas.paste(target.resize((288,288)),(288,24))
display(canvas)
fallbacks=[(r['glyph'],r['stroke'],r['mode']) for r in GUIDE.report['strokes'] if r['mode']!='endpoints']
if fallbacks:print('끝점 대신 중심·크기 또는 기준 위치를 쓴 획 (칸, 획, 방식)',fallbacks)
save_snapshot()
''')
    add('code','''# 5. 합친 목표로 재역확산 한 번 (실제 30스텝)
if GUIDE is None:raise ValueError('4번에서 합친 목표를 먼저 만드세요.')
run=OUTPUT/('repair_'+datetime.now().strftime('%H%M%S_%f'));run.mkdir()
Image.fromarray(np.uint8(np.rint(GUIDE.target*255))).save(run/'target.png')
json_write(run/'attribute_plan.json',GUIDE.report)
def progress(arm,step,total,row):
    if step==1 or step%5==0 or step==total:print(arm,step,'/',total,flush=True)
try:
    report=run_projected_pair(pipe,source,GUIDE,run/'reverse',steps=120,strength=.25,seed=2026,
        prompt=case['prompt'],negative_prompt=case['negative_prompt'],
        start_from_noise=False,save_final_projection=False,progress=progress,guided_only=True)
    panels=[source,Image.open(run/'target.png'),Image.open(run/'reverse/stroke_projected.png')]
    canvas=Image.new('RGB',(864,312),'white');draw=ImageDraw.Draw(canvas)
    for i,(label,im) in enumerate(zip(['input','combined target','guided result'],panels)):
        draw.text((i*288+5,5),label,fill='black');canvas.paste(im.convert('RGB').resize((288,288)),(i*288,24))
    canvas.save(run/'comparison.png');display(canvas)
finally:save_snapshot()
''')
    add('code','''# 6. 결과 다운로드
files.download(save_snapshot())
''')
    nb=dict(base,cells=cells)
    nb['metadata']['colab']['name']='per_stroke_composite.ipynb'
    for i,c in enumerate(cells):c['id']=f'attribute-{i}'
    path=ROOT/'notebooks/repair_audit/per_stroke_composite.ipynb'
    path.write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf-8')
    return path


if __name__=='__main__':print(build())
