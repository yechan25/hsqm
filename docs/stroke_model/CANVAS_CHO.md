# Colab 캔버스에서 초 쓰고 평가

기존 [점수 Colab](https://colab.research.google.com/github/yechan25/hsqm/blob/main/notebooks/structural_indicator.ipynb)의 설정·준비·점수 모듈 로드 셀을 먼저 실행한다. [캔버스 셀](../../notebooks/canvas_cho_cell.py)을 새 셀에 붙이거나 다음 코드로 불러온다.

```python
from urllib.request import urlopen
code = urlopen('https://raw.githubusercontent.com/yechan25/hsqm/main/notebooks/canvas_cho_cell.py').read().decode('utf-8-sig')
exec(compile(code, 'canvas_cho_cell.py', 'exec'))
```

CSV의 `I_0_11.png` 행이 가리키는 정답 초 이미지와 정답 획들을 읽는다. 해당 입력 사진이나 체크포인트는 평가에 사용하지 않는다. 먼저 기준과 획을 보여 주고, 캔버스 점선으로 기준의 위치·크기를 안내한다. 마우스나 펜으로 쓰고 평가를 누르면 저장한 PNG를 128 해상도로 읽어 source 방식의 사진·점수·획별 오류를 출력한다. 한 획 취소, 지우기, 펜 굵기 조절을 제공한다. 다시 쓰려면 셀을 다시 실행한다.

기본 캔버스는 384, 펜 굵기는 18이다. 코드를 직접 붙인 경우 위쪽 REFERENCE_INPUT_FILE, CANVAS_SIZE, PEN_WIDTH에서 설정한다. CSV_PATH와 점수/원본 측정 설정은 기존 런타임 것을 사용한다. 원본 PNG, 평가 배열, 보고서는 `/content/canvas_scores/시간`에 저장한다.

이 기능은 기존 구조 평가기에 직접 쓴 PNG를 넣는 입력 UI다. 자유 손글씨 전반에서 점수가 사람 판단과 일치한다고 검증한 모델은 아니다. 브라우저의 실제 그리기는 Colab에서 확인해야 하며, 로컬 검사는 PNG·기준 읽기·점수·파일 저장의 연결을 검사한다.
