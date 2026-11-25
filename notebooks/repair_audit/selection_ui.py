# 3. 그림 바로 아래의 선택 버튼으로 입력/기준을 지정합니다. 모델 코드는 변경하지 않습니다.
import io
import html

LABEL_TO_ENTRY = {f"row {e['manifest_row']} | {e.get('char_id') or '(이름 없음)'}": e for e in CATALOG}
LABELS = list(LABEL_TO_ENTRY)
ROW_TO_LABEL = {e['manifest_row']: label for label, e in LABEL_TO_ENTRY.items()}
bank_index = Path(PROTOTYPE_BANK)/'index.json'
LOCKED_KEYS = list(json.loads(bank_index.read_text(encoding='utf-8'))) if bank_index.is_file() else []
# Preserve the existing selection contract consumed by cell 5.
reference_selectors = [widgets.Combobox(options=LABELS, value=LABELS[min(i,len(LABELS)-1)],
    ensure_option=True, description='입력 row', layout=widgets.Layout(width='285px')) for i in range(4)]
character_keys = [widgets.Combobox(options=LOCKED_KEYS, value='', description='글자 이름',
    layout=widgets.Layout(width='285px')) for _ in range(4)]
prototype_selectors = [widgets.Combobox(options=LABELS, value='', ensure_option=True, description='기준 row',
    layout=widgets.Layout(width='285px')) for _ in range(4)]
identity_checks = [widgets.Checkbox(value=False, description='입력과 기준이 같은 글자임을 확인',
    indent=False, layout=widgets.Layout(width='295px')) for _ in range(4)]
pin_buttons = [widgets.Button(description='선택한 기준 저장', layout=widgets.Layout(width='180px')) for _ in range(4)]
slot_picker = widgets.ToggleButtons(options=[('왼쪽 위',0),('오른쪽 위',1),('왼쪽 아래',2),('오른쪽 아래',3)],
    value=0, style={'button_width':'71px'}, layout=widgets.Layout(width='305px'))
pick_mode = widgets.ToggleButtons(options=[('입력 폰트 선택','input'),('고정 기준 선택','prototype')],
    value='input', style={'button_width':'145px'}, layout=widgets.Layout(width='305px'))
search = widgets.Text(placeholder='이름 또는 row 번호 검색', layout=widgets.Layout(width='260px'))
previous_page = widgets.Button(description='이전', layout=widgets.Layout(width='65px'))
next_page = widgets.Button(description='다음', layout=widgets.Layout(width='65px'))
page_label = widgets.HTML()
gallery = widgets.GridBox(layout=widgets.Layout(grid_template_columns='repeat(4, 106px)', grid_gap='6px'))
active_form = widgets.VBox(layout=widgets.Layout(width='305px'))
status = widgets.HTML()
summary = widgets.HBox(layout=widgets.Layout(flex_flow='row wrap'))
_page = 0
_thumbnail_cache = {}


def selected_rows():
    values = [w.value for w in reference_selectors]
    if any(v not in LABEL_TO_ENTRY for v in values): raise ValueError('각 칸의 입력 이미지를 선택하세요.')
    return [LABEL_TO_ENTRY[v]['manifest_row'] for v in values]


def selected_keys():
    keys = [w.value.strip() for w in character_keys]
    for key, check in zip(keys, identity_checks):
        if not key or not check.value: raise ValueError('각 칸의 글자 이름과 동일 글자 확인이 필요합니다.')
        load_prototype(PROTOTYPE_BANK,key)
    return keys


def _png(reference, size=92):
    array = np.uint8(np.clip(1-reference[0],0,1)*255)
    image = Image.fromarray(array).resize((size,size))
    buffer = io.BytesIO(); image.save(buffer,format='PNG')
    return buffer.getvalue()


def _thumb(entry):
    row = entry['manifest_row']
    if row not in _thumbnail_cache: _thumbnail_cache[row] = _png(load_reference_image(entry))
    return _thumbnail_cache[row]


def _message(text): status.value = '<span>'+html.escape(text)+'</span>'


def refresh_selection(change=None):
    index = slot_picker.value
    entry = LABEL_TO_ENTRY.get(reference_selectors[index].value)
    pictures = []
    if entry:
        pictures.append(widgets.VBox([widgets.HTML('입력 · row '+str(entry['manifest_row'])),
            widgets.Image(value=_thumb(entry),format='png',width=92,height=92)]))
    key = character_keys[index].value.strip()
    fixed = None; caption = '기준 미선택'
    if key:
        try:
            fixed, record = load_prototype(PROTOTYPE_BANK,key)
            caption = '고정됨 · row '+str(record['manifest_row'])
        except (FileNotFoundError,ValueError,KeyError): pass
    candidate = LABEL_TO_ENTRY.get(prototype_selectors[index].value)
    if fixed is None and candidate:
        reference, strokes = load_catalog_reference(candidate)
        fixed = dict(reference=reference,strokes=strokes)
        caption = '기준 후보 · row '+str(candidate['manifest_row'])
    if fixed:
        pictures.append(widgets.VBox([widgets.HTML(caption),
            widgets.Image(value=_png(fixed['reference']),format='png',width=92,height=92)]))
    else: pictures.append(widgets.HTML('오른쪽 그림에서<br>고정 기준을 선택하세요.'))
    advanced = widgets.Accordion(children=[widgets.VBox([reference_selectors[index],prototype_selectors[index]])],
        selected_index=None,layout=widgets.Layout(width='300px'))
    advanced.set_title(0,'row 직접 입력 (선택 사항)')
    active_form.children = (widgets.HBox(pictures),character_keys[index],pin_buttons[index],
                            identity_checks[index],advanced)
    cards = []
    for i,w in enumerate(reference_selectors):
        e = LABEL_TO_ENTRY.get(w.value)
        if not e: continue
        label = ['왼쪽 위','오른쪽 위','왼쪽 아래','오른쪽 아래'][i]
        button = widgets.Button(description=label,layout=widgets.Layout(width='94px'))
        button.on_click(lambda b,i=i:setattr(slot_picker,'value',i))
        cards.append(widgets.VBox([button,widgets.Image(value=_thumb(e),format='png',width=64,height=64)],
            layout=widgets.Layout(border='2px solid #4387ee' if i==index else '1px solid #ccc',padding='3px',margin='0 8px 0 0')))
    summary.children = tuple(cards)


def choose_thumbnail(row):
    index = slot_picker.value
    label = ROW_TO_LABEL[row]
    if pick_mode.value=='input':
        reference_selectors[index].value = label
        _message(f'현재 칸의 입력을 row {row}로 선택했습니다.')
    else:
        key = character_keys[index].value.strip()
        if key:
            try:
                _,record = load_prototype(PROTOTYPE_BANK,key)
                _message(f'{key}의 기준은 이미 row {record["manifest_row"]}로 고정돼 있습니다.')
                return
            except (FileNotFoundError,ValueError,KeyError): pass
        prototype_selectors[index].value = label
        _message(f'기준 후보 row {row}. 글자 이름을 넣고 기준 저장을 누르세요.')
    refresh_selection()


def pin(index):
    try:
        label = prototype_selectors[index].value
        if label not in LABEL_TO_ENTRY: raise ValueError('오른쪽 그림에서 고정 기준을 먼저 선택하세요.')
        record = lock_prototype(CATALOG,LABEL_TO_ENTRY[label]['manifest_row'],character_keys[index].value,PROTOTYPE_BANK)
        keys = list(json.loads(bank_index.read_text(encoding='utf-8')))
        for widget in character_keys: widget.options = keys
        _message(f'{record["character"]}: 기준 row {record["manifest_row"]} 저장 완료.')
    except (ValueError,FileNotFoundError) as exc: _message(str(exc))
    refresh_selection()


def render_gallery(change=None):
    global _page
    query = search.value.strip().casefold()
    entries = [e for label,e in LABEL_TO_ENTRY.items() if query in label.casefold()]
    pages = max(1,(len(entries)+7)//8); _page = min(_page,pages-1)
    children = []
    for entry in entries[_page*8:(_page+1)*8]:
        row = entry['manifest_row']
        button = widgets.Button(description=f'선택 {row}',layout=widgets.Layout(width='100px',height='28px'))
        button.on_click(lambda b,row=row:choose_thumbnail(row))
        children.append(widgets.VBox([widgets.Image(value=_thumb(entry),format='png',width=92,height=92),button],
            layout=widgets.Layout(border='1px solid #ccc',padding='3px',align_items='center')))
    gallery.children = tuple(children)
    page_label.value = f'{_page+1} / {pages} · {len(entries)}개'
    previous_page.disabled = _page==0; next_page.disabled = _page==pages-1


def move_page(delta):
    global _page
    _page = max(0,_page+delta); render_gallery()


def changed(index):
    identity_checks[index].value = False
    refresh_selection()


def filter_changed(change):
    global _page
    _page = 0; render_gallery()


for i,widget in enumerate(reference_selectors):widget.observe(lambda change,i=i:changed(i),names='value')
for i,widget in enumerate(character_keys):widget.observe(lambda change,i=i:changed(i),names='value')
for widget in prototype_selectors:widget.observe(refresh_selection,names='value')
for i,button in enumerate(pin_buttons):button.on_click(lambda b,i=i:pin(i))
slot_picker.observe(refresh_selection,names='value')
search.observe(filter_changed,names='value')
previous_page.on_click(lambda b:move_page(-1)); next_page.on_click(lambda b:move_page(1))
display(widgets.VBox([
    widgets.HTML('① 칸 선택 → ② 입력/기준 선택 → ③ 그림 아래 선택 버튼 클릭. 이름·기준 저장은 글자별 최초 한 번.'),
    summary,
    widgets.HBox([
        widgets.VBox([slot_picker,pick_mode,active_form],layout=widgets.Layout(width='310px',margin='0 12px 0 0')),
        widgets.VBox([widgets.HBox([search,previous_page,next_page]),page_label,gallery])
    ],layout=widgets.Layout(flex_flow='row wrap',align_items='flex-start')),
    status
]))
refresh_selection(); render_gallery()
print('처음에는 선택 후 4번→5번. 다른 글자는 이 화면에서 바꾸고 5번 재실행.')
