# 7. 재역확산 단계별 보기
from io import BytesIO

if 'OUTPUT' not in globals():
    raise ValueError('먼저 6번 셀을 실행하세요.')

available_runs = {}
for report_path in sorted(OUTPUT.glob('fresh_inputs/*/target_repair/reverse/report.json')):
    report = json.loads(report_path.read_text(encoding='utf-8'))
    if report.get('step_images'):
        available_runs[report_path.parents[2].name] = (report_path.parent, report)
if not available_runs:
    raise ValueError('6번 셀에서 단계 이미지가 저장된 사례가 없습니다.')

case_picker = widgets.Dropdown(options=list(available_runs), description='사례')
step_slider = widgets.IntSlider(description='단계', min=0, continuous_update=False,
                                layout=widgets.Layout(width='95%'))
frame_preview = widgets.Image(format='png')
frame_label = widgets.HTML()


def display_frame(change=None):
    reverse, report = available_runs[case_picker.value]
    frames = report['step_images']
    index = step_slider.value
    if index == 0:
        name, caption = 'input.png', '입력'
    elif index > len(frames):
        name, caption = 'stroke_projected.png', '최종 가이드 결과'
    else:
        frame = frames[index-1]
        name = frame['path']
        caption = f"스텝 {frame['step']}/{report['actual_steps']} · timestep {frame['timestep']} · 가이드 적용 후 깨끗한 이미지 예측"
    frame_label.value = caption
    preview = Image.new('RGB',(768,384),'white')
    for column,path in enumerate([reverse/name, reverse.parent/'target_plan/aligned_target.png']):
        with Image.open(path) as panel:
            preview.paste(panel.convert('RGB').resize((384,384)),(column*384,0))
    buffer = BytesIO()
    preview.save(buffer,format='PNG')
    frame_preview.value = buffer.getvalue()


def change_case(change=None):
    step_slider.value = 0
    step_slider.max = len(available_runs[case_picker.value][1]['step_images'])+1
    display_frame()


case_picker.observe(change_case,names='value')
step_slider.observe(display_frame,names='value')
change_case()
display(widgets.VBox([case_picker,step_slider,frame_label,
                      widgets.HTML('왼쪽 · 선택 단계 / 오른쪽 · 정합 목표'),frame_preview]))
