# 7. 재역확산 단계별 보기
from io import BytesIO
from functools import lru_cache

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
raw_view = widgets.Checkbox(value=False, description='잉크 색 정리·배경 복원 전 보기')


@lru_cache(maxsize=3)
def presentation_data(reverse):
    with np.load(reverse.parent/'target_plan/guide.npz',allow_pickle=False) as guide:
        background=guide['background'] if 'background' in guide.files else None
        return guide['source'],guide['weights']>0,background


def display_frame(change=None):
    reverse, report = available_runs[case_picker.value]
    frames = report['step_images']
    index = step_slider.value
    ink_mode = report.get('guide',{}).get('kind')=='glyph_ink_projection'
    raw_view.disabled = not ink_mode
    if index == 0:
        name, caption = 'input.png', '입력'
    elif index > len(frames):
        if ink_mode and raw_view.value:
            name, caption = 'stroke_projected_raw.png', '최종 역확산 원본'
        else:
            name, caption = 'stroke_projected.png', '최종 가이드 결과'
    else:
        frame = frames[index-1]
        name = frame['path']
        caption = f"스텝 {frame['step']}/{report['actual_steps']} · timestep {frame['timestep']} · 가이드 적용 후 깨끗한 이미지 예측"
        if ink_mode and not raw_view.value:
            caption += ' · 최종과 같은 표시 처리'
    frame_label.value = caption
    preview = Image.new('RGB',(768,384),'white')
    with Image.open(reverse/name) as panel:
        shown=panel.convert('RGB')
    if ink_mode and not raw_view.value and 0<index<=len(frames):
        source,mask,background=presentation_data(reverse)
        pixels=np.rint(source*255).astype(np.uint8)
        if background is None:
            gray=np.rint(np.asarray(shown)[mask].mean(axis=1)).astype(np.uint8)
            pixels[mask]=gray[:,None]
        else:
            from stroke_model.ink_target_guide import finish_ink_rgb
            def tensor(array):return torch.from_numpy(array.copy()).permute(2,0,1)[None]
            finished=finish_ink_rgb(tensor(np.asarray(shown,np.float32)/255),tensor(source),
                torch.from_numpy(mask)[None,None],tensor(background))
            pixels=np.rint(finished[0].permute(1,2,0).numpy()*255).astype(np.uint8)
        shown=Image.fromarray(pixels)
    preview.paste(shown.resize((384,384)),(0,0))
    with Image.open(reverse.parent/'target_plan/aligned_target.png') as target:
        preview.paste(target.convert('RGB').resize((384,384)),(384,0))
    buffer = BytesIO()
    preview.save(buffer,format='PNG')
    frame_preview.value = buffer.getvalue()


def change_case(change=None):
    step_slider.value = 0
    step_slider.max = len(available_runs[case_picker.value][1]['step_images'])+1
    display_frame()


case_picker.observe(change_case,names='value')
step_slider.observe(display_frame,names='value')
raw_view.observe(display_frame,names='value')
change_case()
display(widgets.VBox([case_picker,step_slider,raw_view,frame_label,
                      widgets.HTML('왼쪽 · 선택 단계 / 오른쪽 · 정합 목표'),frame_preview]))
