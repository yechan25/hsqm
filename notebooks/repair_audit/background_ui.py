# 5. 배경 설정
from io import BytesIO
from PIL import ImageOps

background_mode = widgets.ToggleButtons(options=[('기본 배경','default'),('단색','solid'),('그림','image')])
background_color = widgets.ColorPicker(description='색상',value='#e8dcc8')
background_upload = widgets.FileUpload(accept='image/*',multiple=False,description='그림 선택')
background_options = widgets.VBox()


def refresh_background_options(change=None):
    options = {'default':(), 'solid':(background_color,), 'image':(background_upload,)}
    background_options.children = options[background_mode.value]


def selected_background(size):
    mode = background_mode.value
    if mode=='default':
        return None,dict(mode=mode)
    if mode=='solid':
        return Image.new('RGB',(size,size),background_color.value),dict(mode=mode,color=background_color.value)
    uploaded = background_upload.value
    if not uploaded:
        raise ValueError('5번 셀에서 배경 그림을 선택하세요.')
    if isinstance(uploaded,dict):
        name, item = next(iter(uploaded.items()))  # ipywidgets 7
    else:
        item = uploaded[0]
        name = item['name']
    with Image.open(BytesIO(bytes(item['content']))) as image:
        rgba=ImageOps.exif_transpose(image).convert('RGBA')
        opaque=Image.alpha_composite(Image.new('RGBA',rgba.size,'white'),rgba).convert('RGB')
        fitted=ImageOps.fit(opaque,(size,size),method=Image.Resampling.LANCZOS)
    return fitted,dict(mode=mode,filename=name,placement='center_crop_to_sheet')


background_mode.observe(refresh_background_options,names='value')
refresh_background_options()
display(widgets.VBox([background_mode,background_options,
    widgets.HTML('그림은 2×2 시트 크기에 맞춰 가운데를 잘라 씁니다. 글자는 어두운 잉크로 올라갑니다.')]))
