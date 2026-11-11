"""Keep glyph selection/single generation; replace deletion-only with cyclic repair."""
import ast
import json
from pathlib import Path
from build_selectable_adaptive_sweep import build as build_selectable

ROOT=Path(__file__).resolve().parents[1]


def build():
    nb=json.loads(build_selectable().read_text(encoding='utf-8'))
    def put(i,source):
        if nb['cells'][i]['cell_type']=='code':ast.parse(source)
        nb['cells'][i]['source']=source.strip().splitlines(keepends=True)
    put(0,'''# 네 글자 선택 → 누락 획 추가 → 자연스럽게 역확산 → 삭제·재검사

L4에서 1→2→3→4→5 실행. 네 글자 선택과 생성 조건(seed 100 / strength .65)은 그대로입니다.
각 조합당 한 장만 생성·보정합니다. 다른 seed는 4번부터 새로 생성합니다.
이번 보정은 삭제 전용 adaptive와 다릅니다. 복원 → 2단계 정리 → 재추출·삭제로 진행합니다.

누락 경로는 살아 있는 획의 정합 정보를 이용해 기준 획을 배치합니다.
깨끗한 예측 이미지의 누락 경로에 최소 잉크를 명시적으로 추가하고 VAE로 latent에 반영합니다.
VAE 왕복 후 획별 경로 커버리지가 기준을 넘은 경우에만 삭제로 진행합니다.
삭제 중 획이 빠지면 복원으로 복귀하며 최대 2회 복원 구간을 허용합니다.
실패한 복원은 성공으로 표시하지 않습니다. 원시 결과·coverage·단계별 이미지를 모두 저장합니다.
candidate 전체 RGB를 따라 그리거나 최종 출력에 기준 글자를 덮지 않습니다.
다만 누락 구조를 넣는 명시적 투영이 역확산 내부에 들어가는 기준 획 보조 실험입니다.
전체 누락의 위치·정합이 잘못되면 오복원될 수 있습니다. 실제 GPU 결과 확인이 필요합니다.
''')
    setup=''.join(nb['cells'][1]['source'])
    nodes=[n for n in ast.parse(setup).body if isinstance(n,ast.Assign)]
    helpers=next(ast.literal_eval(n.value) for n in nodes
        if any(isinstance(t,ast.Name) and t.id=='RUNTIME_HELPERS' for t in n.targets))
    for path in ('stroke_model/restore_settle_erase.py','diffusion_model/cyclic_stroke_repair.py'):
        helpers[path]=(ROOT/path).read_text(encoding='utf-8')
    a=setup.index('RUNTIME_HELPERS=');b=setup.index("\nROOT=Path",a)
    setup=setup[:a]+'RUNTIME_HELPERS='+repr(helpers)+setup[b:]
    setup+= '\nfrom diffusion_model.cyclic_stroke_repair import repair_cycle_cases\n'
    put(1,setup)
    generation=''.join(nb['cells'][4]['source']).replace("'selectable_adaptive_'","'cyclic_stroke_repair_'")
    generation=generation.replace('adaptive_commit=ADAPTIVE_COMMIT,','adaptive_base_commit=ADAPTIVE_COMMIT,repair_mode="restore_settle_erase",')
    put(4,generation)
    run=''.join(nb['cells'][5]['source'])
    run=run.replace("if row['status']!='completed':", "if row['status']=='failed':")
    run=run.replace('Previous adaptive','Restore / settle / erase')
    run=run.replace('SUMMARIES=repair_cases(', 'SUMMARIES=repair_cycle_cases(')
    run=run.replace("if step==1 or step%5==0 or step==total:print(method,step,'/',total,'accepted=',row['accepted_inner_steps'],flush=True)",
        "if step==1 or step%3==0 or step==total:print(method,step,'/',total,'phase=',row['phase'],'cycle=',row.get('cycle'),'verified=',row.get('restoration_verified'),'accepted=',row['accepted_inner_steps'],flush=True)")
    needle="fig.tight_layout();fig.savefig(Path(row['output'])/'comparison.png',dpi=120);plt.show();plt.close(fig)"
    run=run.replace(needle,needle+'''
    print('획별 경로 커버리지:',json.dumps(row['metrics'],ensure_ascii=False,indent=2))
    paths=sorted((Path(row['output'])/'cycle').glob('cyclic_guided_step_*.png'))
    chosen=[];last_phase=None
    for path in paths:
        phase=path.stem.split('_')[-1]
        if phase!=last_phase:chosen.append(path)
        last_phase=phase
    if paths and paths[-1] not in chosen:chosen.append(paths[-1])
    if chosen:
        fig,axes=plt.subplots(1,len(chosen),figsize=(4*len(chosen),4),squeeze=False)
        for ax,path in zip(axes.flat,chosen):
            ax.imshow(Image.open(path));ax.set_title(path.stem);ax.axis('off')
        fig.tight_layout();fig.savefig(Path(row['output'])/'cycle_stages.png',dpi=120);plt.show();plt.close(fig)
''')
    put(5,run)
    path=Path(__file__).with_name('cyclic_stroke_repair.ipynb')
    path.write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf-8')
    return path


if __name__=='__main__':print(build())
