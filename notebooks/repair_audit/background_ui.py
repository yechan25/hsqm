# @title 5. 배경 설정 — 옵션을 고른 뒤 이 셀을 실행하세요
from io import BytesIO
from PIL import ImageOps, UnidentifiedImageError

BACKGROUND_MODE = '기본 배경' # @param ["기본 배경", "단색", "그림"]
BACKGROUND_COLOR = '#e8dcc8' # @param {type:"string"}

_prepared_background = None
_background_details = None

if BACKGROUND_MODE == '단색':
    try:
        _prepared_background = Image.new('RGB',(CANVAS_SIZE,CANVAS_SIZE),BACKGROUND_COLOR)
    except ValueError as exc:
        raise ValueError('색상 코드를 #e8dcc8 같은 형식으로 입력하세요.') from exc
    _background_details = {'mode':'solid','color':BACKGROUND_COLOR}
elif BACKGROUND_MODE == '그림':
    print('배경으로 사용할 이미지 파일을 하나 선택하세요.')
    uploaded = files.upload()
    if len(uploaded)!=1:
        raise ValueError('배경 이미지를 한 장 선택하고 5번 셀을 다시 실행하세요.')
    name, contents = next(iter(uploaded.items()))
    try:
        with Image.open(BytesIO(contents)) as image:
            rgba = ImageOps.exif_transpose(image).convert('RGBA')
            opaque = Image.alpha_composite(Image.new('RGBA',rgba.size,'white'),rgba).convert('RGB')
            _prepared_background = ImageOps.fit(opaque,(CANVAS_SIZE,CANVAS_SIZE),method=Image.Resampling.LANCZOS)
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError('이 파일을 이미지로 열 수 없습니다. PNG나 JPG를 골라 주세요.') from exc
    _background_details = {'mode':'image','filename':name,'placement':'center_crop_to_sheet'}
    print(f'배경 적용 완료 · {name}')
elif BACKGROUND_MODE == '기본 배경':
    _background_details = {'mode':'default'}
else:
    raise ValueError('배경은 기본 배경, 단색, 그림 중 하나를 고르세요.')


def selected_background(size):
    if _background_details is None:
        raise ValueError('5번 셀에서 배경 설정을 완료하세요.')
    if _prepared_background is None:
        return None, _background_details.copy()
    image = _prepared_background
    if image.size != (size,size):
        image = image.resize((size,size),Image.Resampling.LANCZOS)
    return image.copy(), _background_details.copy()


if _prepared_background is None:
    print('기본 배경 사용. 6번 셀로 진행하세요.')
else:
    display(_prepared_background.resize((256,256)))
    print('위 배경이 준비됐습니다. 6번 셀로 진행하세요.')
