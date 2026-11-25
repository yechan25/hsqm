# 4. 전처리된 그림으로 선택. 원본 파일 읽기와 썸네일 생성은 3번에서 완료합니다.
import io
import html

LABEL_TO_ENTRY = {f"row {e['manifest_row']} | {e.get('char_id') or '(이름 없음)'}": e for e in CATALOG}
LABELS = list(LABEL_TO_ENTRY)
ROW_TO_LABEL = {e['manifest_row']: label for label, e in LABEL_TO_ENTRY.items()}
_prototype_records = dict(PROTOTYPE_RECORDS)
_prototype_images = dict(PROTOTYPE_PREVIEWS)
# Preserve the existing selection contract consumed by cell 5.
reference_selectors = [widgets.Combobox(options=LABELS, value=LABELS[min(i,len(LABELS)-1)],
    ensure_option=True, description='입력 row', layout=widgets.Layout(width='285px')) for i in range(4)]
character_keys = [widgets.Combobox(options=list(_prototype_records),
    value=CATALOG[min(i,len(CATALOG)-1)].get('resolved_character') or '',description='글자',continuous_update=False,
    layout=widgets.Layout(width='285px')) for i in range(4)]
prototype_selectors = [widgets.Combobox(options=LABELS, value='', ensure_option=True, description='기준 row',
    layout=widgets.Layout(width='285px')) for _ in range(4)]
identity_checks = [widgets.Checkbox(value=False, description='입력과 기준이 같은 글자임을 확인',
    indent=False, layout=widgets.Layout(width='295px')) for _ in range(4)]
pin_buttons = [widgets.Button(description='기준 적용 / 변경', layout=widgets.Layout(width='180px')) for _ in range(4)]
slot_picker = widgets.ToggleButtons(options=[('왼쪽 위',0),('오른쪽 위',1),('왼쪽 아래',2),('오른쪽 아래',3)],
    value=0, style={'button_width':'71px'}, layout=widgets.Layout(width='305px'))
pick_mode = widgets.ToggleButtons(options=[('입력 폰트 선택','input'),('공통 기준 선택','prototype')],
    value='input', style={'button_width':'145px'}, layout=widgets.Layout(width='305px'))
search = widgets.Text(placeholder='이름 또는 row 번호',continuous_update=False,layout=widgets.Layout(width='180px'))
search_button = widgets.Button(description='검색',layout=widgets.Layout(width='60px'))
previous_page = widgets.Button(description='이전', layout=widgets.Layout(width='65px'))
next_page = widgets.Button(description='다음', layout=widgets.Layout(width='65px'))
page_label = widgets.HTML()
gallery = widgets.GridBox(layout=widgets.Layout(grid_template_columns='repeat(4, 106px)', grid_gap='6px'))
active_form = widgets.VBox(layout=widgets.Layout(width='305px'))
status = widgets.HTML()
summary = widgets.HBox(layout=widgets.Layout(flex_flow='row wrap'))
_page = 0
_thumbnail_cache = SELECTION_THUMBNAILS


def selected_rows():
    values = [w.value for w in reference_selectors]
    if any(v not in LABEL_TO_ENTRY for v in values): raise ValueError('각 칸의 입력 이미지를 선택하세요.')
    return [LABEL_TO_ENTRY[v]['manifest_row'] for v in values]


def selected_keys():
    keys = [w.value.strip() for w in character_keys]
    for i,(key, check) in enumerate(zip(keys, identity_checks)):
        if not key or not check.value: raise ValueError('각 칸의 글자 이름과 동일 글자 확인이 필요합니다.')
        _,record=load_prototype(PROTOTYPE_BANK,key)
        candidate=LABEL_TO_ENTRY.get(prototype_selectors[i].value)
        if candidate and candidate['manifest_row']!=record['manifest_row']:
            raise ValueError('새로 고른 기준의 적용 버튼을 먼저 누르세요.')
    return keys


def _thumb(entry):
    return _thumbnail_cache[entry['manifest_row']]


def _message(text): status.value = '<span>'+html.escape(text)+'</span>'


def refresh_selection(change=None):
    index = slot_picker.value
    entry = LABEL_TO_ENTRY.get(reference_selectors[index].value)
    pictures = []
    if entry:
        pictures.append(widgets.VBox([widgets.HTML('입력 · row '+str(entry['manifest_row'])),
            widgets.Image(value=_thumb(entry),format='png',width=92,height=92)]))
    key = character_keys[index].value.strip()
    known_character = entry.get('resolved_character') if entry else None
    character_keys[index].disabled = bool(known_character)
    character_keys[index].description = '글자(자동)' if known_character else '글자 이름'
    preview = None; caption = '기준 미선택'
    candidate = LABEL_TO_ENTRY.get(prototype_selectors[index].value)
    if candidate:
        preview = _thumb(candidate)
        caption = '선택한 기준 · row '+str(candidate['manifest_row'])
    elif key in _prototype_images:
        preview = _prototype_images[key]
        caption = '사용할 기준 · row '+str(_prototype_records[key]['manifest_row'])
    if preview is not None:
        pictures.append(widgets.VBox([widgets.HTML(caption),
            widgets.Image(value=preview,format='png',width=92,height=92)]))
    else: pictures.append(widgets.HTML('오른쪽 그림에서<br>공통 기준을 선택하세요.'))
    prototype_entry = candidate or _prototype_records.get(key,{}).get('reference',{})
    prototype_character = reference_identity(prototype_entry)['resolved_character']
    if known_character and prototype_character:
        identity_checks[index].value = known_character==prototype_character==key
        identity_checks[index].disabled = True
        identity_checks[index].description = '동일 글자 자동 확인' if identity_checks[index].value else '입력과 기준의 글자가 다릅니다'
    else:
        identity_checks[index].disabled = False
        identity_checks[index].description = '입력과 기준이 같은 글자임을 확인'
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
        prototype_selectors[index].value = label
        _message(f'기준 row {row} 선택. 이름을 확인하고 기준 적용을 누르세요. 실행 전에는 다시 바꿀 수 있습니다.')
    refresh_selection()


def pin(index):
    try:
        label = prototype_selectors[index].value
        if label not in LABEL_TO_ENTRY: raise ValueError('오른쪽 그림에서 공통 기준을 먼저 선택하세요.')
        key = character_keys[index].value.strip()
        record = save_prototype(CATALOG,LABEL_TO_ENTRY[label]['manifest_row'],key,PROTOTYPE_BANK)
        _prototype_records[key] = record
        _prototype_images[key] = _thumb(LABEL_TO_ENTRY[label])
        # Same character uses ONE shared choice in all slots in this experiment.
        for i,key_widget in enumerate(character_keys):
            if key_widget.value.strip()==key:
                prototype_selectors[i].value=label
                identity_checks[i].value=False
        keys = list(_prototype_records)
        for widget in character_keys: widget.options = keys
        _message(f'{record["character"]}: 기준 row {record["manifest_row"]} 적용. 필요하면 다시 선택해 변경하세요.')
    except (ValueError,FileNotFoundError) as exc: _message(str(exc))
    refresh_selection()


def render_gallery(change=None):
    global _page
    query = _applied_query
    entries = [e for label,e in LABEL_TO_ENTRY.items() if query in label.casefold()]
    source_entry = LABEL_TO_ENTRY.get(reference_selectors[slot_picker.value].value,{})
    character = source_entry.get('resolved_character')
    if pick_mode.value=='prototype' and character:
        entries = [e for e in entries if e.get('resolved_character')==character]
    pages = max(1,(len(entries)+7)//8); _page = min(_page,pages-1)
    children = []
    for entry in entries[_page*8:(_page+1)*8]:
        row = entry['manifest_row']
        button = widgets.Button(description=f'선택 {row}',layout=widgets.Layout(width='100px',height='28px'))
        button.on_click(lambda b,row=row:choose_thumbnail(row))
        children.append(widgets.VBox([widgets.Image(value=_thumb(entry),format='png',width=92,height=92),button],
            layout=widgets.Layout(border='1px solid #ccc',padding='3px',align_items='center')))
    gallery.children = tuple(children)
    suffix = f' · {character}의 기준만 표시' if pick_mode.value=='prototype' and character else ''
    page_label.value = f'{_page+1} / {pages} · {len(entries)}개'+suffix
    previous_page.disabled = _page==0; next_page.disabled = _page==pages-1


def move_page(delta):
    global _page
    _page = max(0,_page+delta); render_gallery()


def changed(index, key_changed=False):
    identity_checks[index].value = False
    if key_changed: prototype_selectors[index].value=''
    else:
        entry=LABEL_TO_ENTRY.get(reference_selectors[index].value,{})
        character=entry.get('resolved_character')
        if character and character_keys[index].value!=character:character_keys[index].value=character
    refresh_selection()
    if pick_mode.value=='prototype':render_gallery()


def filter_changed(change):
    global _page, _applied_query
    _page = 0; _applied_query = search.value.strip().casefold(); render_gallery()


def mode_changed(change):
    global _page, _applied_query
    _page=0;_applied_query='';search.value='';render_gallery()


for i,widget in enumerate(reference_selectors):widget.observe(lambda change,i=i:changed(i),names='value')
for i,widget in enumerate(character_keys):widget.observe(lambda change,i=i:changed(i,True),names='value')
for widget in prototype_selectors:widget.observe(refresh_selection,names='value')
for i,button in enumerate(pin_buttons):button.on_click(lambda b,i=i:pin(i))
slot_picker.observe(refresh_selection,names='value')
slot_picker.observe(render_gallery,names='value')
pick_mode.observe(mode_changed,names='value')
_applied_query = ''
search_button.on_click(filter_changed)
previous_page.on_click(lambda b:move_page(-1)); next_page.on_click(lambda b:move_page(1))
display(widgets.VBox([
    widgets.HTML('① 칸 선택 → ② 입력/공통 기준 선택 → ③ 그림 아래 버튼 클릭. 글자 이름은 파일 정보에서 자동 연결합니다. 기준은 실행 전에 변경 가능합니다.'),
    summary,
    widgets.HBox([
        widgets.VBox([slot_picker,pick_mode,active_form],layout=widgets.Layout(width='310px',margin='0 12px 0 0')),
        widgets.VBox([widgets.HBox([search,search_button,previous_page,next_page]),page_label,gallery])
    ],layout=widgets.Layout(flex_flow='row wrap',align_items='flex-start')),
    status
]))
refresh_selection(); render_gallery()
print('선택 후 5번 모델 로딩 → 6번 실행. 다음 실험은 선택을 바꾸고 6번만 실행.')
