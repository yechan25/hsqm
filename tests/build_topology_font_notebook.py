"""Build cross-font Topo test Colab."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    cells=[]
    def add(kind,source):
        if kind=='code':compile(source,'<colab>','exec')
        cell=dict(cell_type=kind,metadata={},source=source.splitlines(keepends=True))
        if kind=='code':cell.update(execution_count=None,outputs=[])
        cells.append(cell)
    add('markdown','''# Topo — 다른 컴퓨터 폰트로 획 추출 테스트
같은 글자를 고딕/명조와 굵기가 다른 폰트로 렌더링해서 현재 Topo 모델에 넣습니다.
기준 이미지와 기준 획은 CSV의 것을 그대로 유지합니다. 손글씨를 그릴 필요는 없습니다.

1. 준비·Drive 연결·폰트 설치
2. 체크포인트·CSV 설정
3. 기준 이미지, 입력할 글자, 테스트 폰트 선택
4. 추출 실행. 다른 글자를 보려면 3번의 선택을 바꾸고 4번만 재실행하세요.

처음의 Reference control은 기준 이미지 자체를 입력한 대조군입니다.
이어지는 결과는 다른 폰트를 입력한 결과입니다. 폰트별 입력/획 비교표와 Soft·Exclusive 전체를 저장합니다.
훈련·체크포인트 변경·출력 팽창/직선 보정은 하지 않습니다. 점수는 기본 계산하지 않습니다.
새 폰트의 획별 정답이 없으므로 정확도나 확산 모델의 성능을 측정하는 실험은 아닙니다.
100% ink coverage는 softmax 분할의 성질이며 올바른 분리를 뜻하지 않습니다.
학습 폰트 목록을 별도로 대조하지 않으므로 미학습 폰트임을 보장하지 않습니다.

기본 배치는 글자 전체의 비율을 유지해 기준 잉크 박스 안에 맞춥니다.
개별 자모 위치/획 굵기를 기준에 맞춰 변형하지 않습니다. 배치 영향을 보려면 PLACEMENT='centered'.
''')
    add('code','''from pathlib import Path
import subprocess,sys,tempfile
root=Path(tempfile.mkdtemp(prefix='hsqm_fonts_',dir='/content'))
subprocess.run(['git','clone','--depth','1','--branch','main','https://github.com/yechan25/hsqm.git',str(root)],check=True)
subprocess.run([sys.executable,'-m','pip','install','-q','-r',str(root/'requirements.txt'),'fonttools','ipywidgets'],check=True)
subprocess.run(['apt-get','update','-qq'],check=True)
subprocess.run(['apt-get','install','-y','-qq','fonts-nanum'],check=True)
sys.path.insert(0,str(root))
for key in list(sys.modules):
    if key in ('stroke_model','font_test') or key.startswith(('stroke_model.','font_test.')):del sys.modules[key]
commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
print('코드 버전:',commit)
from google.colab import drive,output
drive.mount('/content/drive')
output.enable_custom_widget_manager()
''')
    add('code','''from stroke_model.convert_reference_csv import existing_path
from stroke_model.font_topology import reference_catalog,run_topology_fonts
HSQM=existing_path('/content/drive/Shareddrives/2026 자율연구/HSQM')
CHECKPOINT=None  # 특정 Before/After best.pt 경로를 직접 지정할 수 있습니다.
DATA_ROOT=HSQM/'dataset/test_dataset'
CSV_PATH=None
DATASET_ROOT=None
EXTRA_FONT_DIR=None  # 추가 .ttf/.otf/.ttc가 있는 Drive 폴더. 컬렉션은 첫 face 사용.
PLACEMENT='reference_bbox'  # 또는 'centered'
if CHECKPOINT is None:
    candidates=sorted((HSQM/'v2_topology_runs').glob('*/best.pt'),key=lambda p:p.parent.name)
    if not candidates:raise FileNotFoundError('CHECKPOINT에 Topo best.pt 경로를 지정하세요.')
    CHECKPOINT=candidates[-1]
if CSV_PATH is None:
    data_root=existing_path(DATA_ROOT)
    candidates=[p for p in data_root.iterdir() if p.is_file() and p.name.casefold() in ('path.csv','paths.csv','path (1).csv')]
    if len(candidates)!=1:raise ValueError(f'CSV_PATH를 지정하세요. 후보: {candidates}')
    CSV_PATH=candidates[0]
import torch
from stroke_model.reference_v2_topology import ARCHITECTURE
state=torch.load(CHECKPOINT,map_location='cpu',weights_only=True)
if state.get('architecture')!=ARCHITECTURE:raise ValueError('V2 기반 Topo 모델을 지정하세요.')
print('사용 모델:',CHECKPOINT,'/ epoch:',state.get('epoch'))
del state
FONT_PATHS=sorted(p for p in Path('/usr/share/fonts/truetype/nanum').rglob('*') if p.suffix.lower() in ('.ttf','.otf','.ttc'))
if EXTRA_FONT_DIR:
    FONT_PATHS+=sorted(p for p in existing_path(EXTRA_FONT_DIR).rglob('*') if p.suffix.lower() in ('.ttf','.otf','.ttc'))
if not FONT_PATHS:raise FileNotFoundError('사용할 폰트가 없습니다.')
print('CSV:',CSV_PATH,'/ 폰트 수:',len(FONT_PATHS))
''')
    add('code','''import io
import numpy as np
import ipywidgets as widgets
from PIL import Image
from IPython.display import display
from stroke_model.convert_reference_csv import Resolver,load_ink
catalog=reference_catalog(CSV_PATH)
if not catalog:raise ValueError('선택 가능한 기준 글자가 없습니다.')
by_row={item['csv_row']:item for item in catalog}
default=next((item['csv_row'] for item in catalog if item['glyph']=='아'),catalog[0]['csv_row'])
reference_select=widgets.Dropdown(options=[(item['label'],item['csv_row']) for item in catalog],value=default,description='기준 글자:',layout=widgets.Layout(width='95%'))
glyph_text=widgets.Text(description='입력 문자:',placeholder='위 기준과 같은 글자 1개')
reference_preview=widgets.Image(format='png',width=180,height=180)
resolve=Resolver(Path(CSV_PATH),existing_path(DATASET_ROOT or DATA_ROOT))
def update_reference(change=None):
    item=by_row[reference_select.value]
    glyph_text.value=item['glyph'] or ''
    image=load_ink(resolve(item['reference']),180)
    stream=io.BytesIO();Image.fromarray(np.uint8((1-image[0])*255)).save(stream,format='PNG')
    reference_preview.value=stream.getvalue()
reference_select.observe(update_reference,names='value');update_reference()
preferred=('nanumgothic.ttf','nanumgothicbold.ttf','nanummyeongjo.ttf','nanummyeongjobold.ttf')
defaults=[str(p) for p in FONT_PATHS if p.name.casefold() in preferred]
font_select=widgets.SelectMultiple(options=[(p.name+' | '+str(p.parent),str(p)) for p in FONT_PATHS],value=tuple(defaults or [str(FONT_PATHS[0])]),description='폰트:',rows=min(10,len(FONT_PATHS)),layout=widgets.Layout(width='95%'))
display(widgets.VBox([reference_select,reference_preview,glyph_text,font_select]))
print('기준 그림과 같은 문자를 입력하세요. 파일명이 숫자이면 문자를 자동으로 추측하지 않습니다.')
print('폰트는 Ctrl/Cmd를 눌러 여러 개 선택할 수 있습니다. 선택 후 다음 셀 실행.')
''')
    add('code','''font_report=run_topology_fonts(
    CHECKPOINT,CSV_PATH,csv_row=int(reference_select.value),glyph=glyph_text.value,
    font_paths=list(font_select.value),dataset_root=DATASET_ROOT or DATA_ROOT,
    placement=PLACEMENT,include_reference_control=True)
report_dir=Path(font_report['results'][0]['npz']).parent
(report_dir/'git_commit.txt').write_text(commit)
print('기준 입력 대조군 + 다른 폰트 결과 저장:',report_dir)
''')
    notebook=dict(cells=cells,metadata=dict(accelerator='GPU',kernelspec=dict(display_name='Python 3',language='python',name='python3')),nbformat=4,nbformat_minor=5)
    (ROOT/'notebooks/test_topology_fonts.ipynb').write_text(json.dumps(notebook,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':main()
