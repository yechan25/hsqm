"""Stable notebook: static foundation, consensus proposals, matched stroke guards."""
import ast
import json
from pathlib import Path
from build_adaptive_ink_diffusion import build as build_adaptive

ROOT=Path(__file__).resolve().parents[1]


def build():
    nb=json.loads(build_adaptive().read_text(encoding='utf-8'))
    def put(i, source):
        if nb['cells'][i]['cell_type']=='code':ast.parse(source)
        nb['cells'][i]['source']=source.strip().splitlines(keepends=True)
    put(0, '''# HSQM 국소 역확산 안정화판 — 오류 삭제와 정상 획 보호

L4에서 1 → 2 → 3 → 4 실행. 기본 설정을 고정한 단일 실행 흐름입니다.
입력: 원래 sweep ZIP 또는 획 묶음이 포함된 adaptive_ink_reverse 결과 ZIP.
2번에서 기존 획 추출 모델을 Drive에서 찾아 SHA256을 검증합니다. 직접 업로드도 가능합니다.
최신 adaptive ZIP에서도 보정 전 원래 입력부터 비교하며, 손상된 이전 출력을 이어 쓰지 않습니다.

성공했던 고정 마스크를 기본으로 쓰고, 재추출 제안은 2회 연속 확인된 경우만 반영합니다.
획별 정상 영역에 대해 몸통 평균 오차, 국소 오차, 잉크 감소 한도를 검사합니다.
한도를 넘는 latent 수정은 거절합니다. 전체 출력도 재검사하여 실패하면 원래 입력을 보존합니다.
최종 선택본은 selected.png, 원시 역확산 결과와 거절 기록도 모두 저장합니다.
이는 내부 보호 조건이며 시각적 정상성/완전 복원의 보증이 아닙니다.
candidate RGB를 복제하거나 결과 픽셀을 지우는 후처리는 하지 않습니다.
''')
    setup=''.join(nb['cells'][1]['source'])
    start=setup.index('RUNTIME_HELPERS=');end=setup.index("\nROOT=Path",start)
    paths=('diffusion_model/fitted_candidate_diffusion.py','stroke_model/adaptive_ink_constraints.py',
           'stroke_model/guarded_ink_constraints.py','stroke_model/reference_model.py','stroke_model/stroke_reference_fit.py')
    helpers={p:(ROOT/p).read_text(encoding='utf-8') for p in paths}
    setup=setup[:start]+'RUNTIME_HELPERS='+repr(helpers)+setup[end:]
    setup+='\nfrom stroke_model.guarded_ink_constraints import GuardedInkConstraints\n'
    put(1,setup)
    load=''.join(nb['cells'][2]['source'])
    load=load.replace("# 기존 sweep ZIP. 이전 ink_constraint/fitted_reverse 결과 ZIP 아님.",
                      '# sweep ZIP 또는 stroke_bundle이 들어 있는 adaptive 결과 ZIP.')
    a=load.index("    local=Path('/content/sweep_")
    b=load.index('    if matches:ZIP_PATH=',a)
    load=load[:a]+'''    names=('adaptive_ink_reverse_20261111_075257_135265.zip','sweep_20261111_062740_406072.zip')
    matches=[Path('/content')/name for name in names if (Path('/content')/name).is_file()]
    if not matches and USE_DRIVE and drive_root.is_dir():
        print('Drive에서 기존 결과 ZIP을 찾습니다...',flush=True)
        for name in names:
            matches=list(drive_root.rglob(name))
            if matches:break
'''+load[b:]
    load=load.replace('원래 sweep ZIP을 업로드하세요.', '원래 sweep 또는 stroke_bundle이 들어 있는 adaptive 결과 ZIP을 업로드하세요.')
    load=load.replace('sweep ZIP 하나를 선택하세요.', '실험 ZIP 하나를 선택하세요.')
    put(2,load)
    constructor="GuardedInkConstraints(source_rgb,image_tensor(candidate,'cuda'),extractor,records,refresh_every=REFRESH_EVERY,enable_restore=ENABLE_RESTORE)"
    preview=''.join(nb['cells'][3]['source'])
    a=preview.index('objective=AdaptiveInkConstraints(');b=preview.index("\nprint('획별",a)
    preview=preview[:a]+'objective='+constructor+preview[b:]
    preview=preview.replace('MARGIN_RATIO=.20  # 정상 몸통 전체 굵기의 20%를 허용 여유로 사용\n','')
    preview=preview.replace("print('정상 획을 넓게 덮는다면 MARGIN_RATIO를 높여 재생성하세요.')",
        "print('기본 삭제만 먼저 적용하며 새 제안은 반복 확인과 정상 획 보호 검사를 통과해야 합니다.')")
    put(3,preview)
    run=''.join(nb['cells'][4]['source'])
    a=run.index('objective=AdaptiveInkConstraints(');b=run.index('\ndef progress(',a)
    run=run[:a]+'objective='+constructor+run[b:]
    run=run.replace("'adaptive_ink_reverse_'", "'guarded_ink_release_'")
    run=run.replace('Adaptive stroke-guided reverse', 'Raw guarded reverse')
    run=run.replace("('Raw guarded reverse',Image.open(OUTPUT/'adaptive_guided.png'))]",
        "('Raw guarded reverse',Image.open(OUTPUT/'adaptive_guided.png')),('Selected output',Image.open(OUTPUT/'selected.png'))]")
    run=run.replace('plt.subplots(1,3,figsize=(18,6))','plt.subplots(1,4,figsize=(24,6))')
    run=run.replace("print('공통 초기 마스크", "print('최종 선택:',json.dumps(report['selection'],ensure_ascii=False,indent=2))\nprint('공통 초기 마스크")
    run=run.replace("f\"accepted={row['accepted_inner_steps']} | ",
        "f\"accepted={row['accepted_inner_steps']} | guard_rejected={row.get('rejected_guard_attempts',0)} | ")
    # Bundle can be loaded from an earlier release without nesting stroke_bundle.
    run=run.replace("        if name.startswith('source_conditions/') or name.startswith(CASE_ID+'/warped_slot_'):\n            path=OUTPUT/'stroke_bundle'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(z.read(name))",
        "        clean=name.removeprefix('stroke_bundle/')\n        if clean.startswith('source_conditions/') or clean.startswith(CASE_ID+'/warped_slot_'):\n            path=OUTPUT/'stroke_bundle'/clean;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(z.read(name))")
    put(4,run)
    path=Path(__file__).with_name('guarded_ink_release.ipynb')
    path.write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf-8')
    return path


if __name__=='__main__':print(build())
