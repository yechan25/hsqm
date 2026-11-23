"""Build a standalone arm-level resumer; no mid-arm state is fabricated."""
import ast
import hashlib
import json
from pathlib import Path
from build_stage4 import build as build_stage4
from build_stage3 import embedded

ROOT=Path(__file__).resolve().parents[2]


def build():
    nb=json.loads(build_stage4().read_text(encoding='utf-8'))
    for cell in nb['cells']:
        source=''.join(cell['source'])
        if cell['cell_type']=='code' and 'RUNTIME_HELPERS=' in source:
            helpers=embedded(source)
            hashes={k:hashlib.sha256(v.encode()).hexdigest() for k,v in helpers.items()}
            source=source.replace('hsqm_stage4_audit_runtime','hsqm_stage4_resume_runtime')
            source+='\nfrom repair_audit.resume import prepare_resume, completed_case, complete_arm, ARMS\n'
        elif cell['cell_type']=='code' and 'ARCHIVE = ' in source:
            source='''# 2. 중단한 4단계 ZIP 업로드. 설정은 ZIP에서 복구합니다.
ARCHIVE = ''
CHECKPOINT = '/content/drive/Shareddrives/2026 자율연구/HSQM/v2_topology_runs/20261104_214705_208224/best.pt'
DRIVE_RESULT_DIR = '/content/drive/MyDrive/HSQM_stage4_resume_results'
if not Path('/content/drive/MyDrive').is_dir():
    drive.mount('/content/drive')
if not ARCHIVE:
    uploaded=files.upload()
    if len(uploaded)!=1:
        raise ValueError('중단한 결과 ZIP 하나만 선택하세요.')
    ARCHIVE=str(Path(next(iter(uploaded))).resolve())
    del uploaded
OUTPUT=Path('/content')/('repair_stage4_resume_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
CASES, EXPERIMENT, PARENT = prepare_resume(ARCHIVE, OUTPUT, SOURCE_HASHES)
if file_hash(CHECKPOINT)!=PARENT['checkpoint_sha256']:
    raise ValueError('기존 실행과 다른 best.pt입니다.')
MODEL_ID, REVISION = PARENT['model_id'], PARENT['revision']
SETTINGS = PARENT['settings']
REPAIR_STEPS, REPAIR_STRENGTH, REPAIR_SEED = [SETTINGS[k] for k in ('steps','strength','seed')]
BOUNDARY_CONFIG, WIDTH_CONFIG = SETTINGS['boundary_config'], SETTINGS['width_config']
json_write(OUTPUT/'runtime_manifest.json', dict(source_hashes=SOURCE_HASHES,
    model_id=MODEL_ID, revision=REVISION, checkpoint_sha256=PARENT['checkpoint_sha256'],
    torch_version=torch.__version__, gpu_type=torch.cuda.get_device_name(0),
    settings=SETTINGS, resume_archive=str(ARCHIVE)))
METRICS, FAILURES = [], []
for case in CASES:
    cached=completed_case(case, SETTINGS)
    if cached is not None:
        METRICS.extend(cached)
    audit=Path(case['directory'])/'audit'
    done=[a for a in ARMS if complete_arm(audit,a,list(range(225,-1,-16)))] if REPAIR_STEPS==60 and REPAIR_STRENGTH==.25 else []
    print(case['case_id'], '사례 전체 재사용' if cached is not None else '완료 경로: '+str(done))

# Own timestamped ZIP only; each completed arm refreshes this recovery snapshot.
def save_snapshot():
    write_stage3_tables(OUTPUT, CASES, METRICS, FAILURES)
    archive=shutil.make_archive(str(OUTPUT)+'_checkpoint','zip',OUTPUT)
    if DRIVE_RESULT_DIR:
        destination=Path(DRIVE_RESULT_DIR)
        destination.mkdir(parents=True,exist_ok=True)
        shutil.copy2(archive, destination/Path(archive).name)
    return archive
def arm_saved(case_id, arm):
    archive=save_snapshot()
    print('경로 완료·Drive 백업:',case_id,arm,Path(archive).name,flush=True)
save_snapshot()
print('기존 결과 복구 및 Drive 백업 완료. 3번 모델 로딩 → 4번 실행.')
'''
        elif cell['cell_type']=='code' and 'METRICS, FAILURES' in source:
            source='''# 4. 완료 경로 재사용, 부분 경로는 처음부터 재계산
def progress(case_id, arm, step, total, row):
    if row['phase']=='cached' or step==1 or step%3==0 or step==total:
        print(case_id,arm,f'{step}/{total}',row['phase'],
              'accepted=',row['accepted_inner_steps'],flush=True)
for case in CASES:
    try:
        cached=completed_case(case, SETTINGS)
        if cached is None:
            rows=run_case(pipe,extractor,case,steps=REPAIR_STEPS,strength=REPAIR_STRENGTH,
                seed=REPAIR_SEED,boundary_guard=True,boundary_config=BOUNDARY_CONFIG,
                observed_width=True,width_config=WIDTH_CONFIG,resume=True,
                progress=progress,on_arm_complete=arm_saved)
        else:
            rows=cached
            print('SKIP 완료 사례:',case['case_id'],flush=True)
        METRICS[:]=[r for r in METRICS if r['case_id']!=case['case_id']]
        METRICS.extend(rows)
        FAILURES[:]=[f for f in FAILURES if f.get('case_id')!=case['case_id']]
        shape_comparison(case,arms=['input','vae_roundtrip','redenoise_only','cyclic_guided','boundary_guarded','observed_width'])
        display(Image.open(Path(case['directory'])/'audit/shape_comparison.png'))
    except KeyboardInterrupt:
        print('중단 요청: 현재 부분 경로는 다음에 처음부터 계산합니다. 백업 후 종료합니다.',flush=True)
        break
    except Exception as exc:
        failure=dict(case_id=case['case_id'],type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
        FAILURES.append(failure)
        json_write(Path(case['directory'])/'resume_failure.json',failure)
        print('FAILED:',case['case_id'],str(exc),flush=True)
    finally:
        save_snapshot()
        gc.collect(); torch.cuda.empty_cache()
print('완료 또는 중단 상태를 Drive에 저장했습니다. 5번에서 ZIP 다운로드.')
'''
        elif cell['cell_type']=='code' and '# 5.' in source:
            source='''# 5. 현재 결과 ZIP 다운로드 (완료/중단 모두 가능)
archive=save_snapshot()
print('다운로드:',archive)
files.download(archive)
'''
        elif cell['cell_type']=='markdown' and source.startswith('# 4단계:'):
            source='''# 4단계 재개: 완료 경로를 건너뛰고 나머지만 실행

새 T4 또는 L4 GPU 세션에서 **코드 셀 1→5**를 실행합니다.
2번에 **repair_stage4_20261123_132010_265667.zip**을 업로드하세요.
기존 best.pt만 추가로 필요하며 학습·새 이미지 생성은 없습니다.

이 ZIP에는 첫 사례의 네 경로와 두 번째 사례의 일반 재확산이 완료돼 있습니다.
이 다섯 경로는 다시 계산하지 않습니다. 두 번째 사례의 미완료 보정부터 진행합니다.
중단된 경로의 중간 latent/controller가 저장되지 않았으므로 해당 경로는 1/15부터 재시작합니다.

각 경로 완료 시 전체 상태를 Drive에 자동 백업합니다. 4번 셀을 중지하면 중단 지점까지 백업하고,
5번에서 ZIP을 받을 수 있습니다. 연결 강제 종료 시에는 마지막 완료 경로의 Drive 백업을 사용하세요.
다음 재개에서도 이 노트북에 최신 checkpoint ZIP을 넣으면 됩니다.

옛 ZIP의 부분 사례에는 초기 latent hash가 없습니다. 과거 경로와 새 경로의 비트 단위 초기 상태
동일성을 소급해 증명할 수 없어 보고서에 legacy_initial_unrecorded=true로 명시합니다.
새로 계산하는 경로들은 같은 초기 latent를 공유하고 이를 저장해 이후 재개에서 재사용합니다.
GPU 종류나 패키지가 바뀐 이력도 부모 manifest와 versions.json에 남습니다.
'''
        elif cell['cell_type']=='markdown' and source.startswith('## 2번 셀'):
            source='''## 2번: 중단한 결과 ZIP 선택
- 3단계 입력 ZIP 대신 **중단한 4단계 결과 ZIP** 또는 이 재개 코랩의 checkpoint ZIP을 선택합니다.
- 보정 설정·모델 revision은 ZIP에서 복구하며 모델/손실 소스와 체크포인트 해시를 검사합니다.
- 완료 여부는 최종 PNG와 전체 timestep trace를 함께 확인합니다. 중간 PNG만으로 완료 처리하지 않습니다.
- partial 경로의 예전 프레임은 interrupted_attempts/로 보관합니다.
- Drive 자동 백업 위치: MyDrive/HSQM_stage4_resume_results.
'''
        if cell['cell_type']=='code': ast.parse(source)
        cell['source']=source.strip().splitlines(keepends=True)
    path=ROOT/'notebooks/repair_audit/stage4_resume.ipynb'
    path.write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf-8')
    return path


if __name__=='__main__':print(build())
