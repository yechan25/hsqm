# Font tests

폰트 관련 실험 코드와 설정을 저장하는 폴더입니다.

`render.py`는 다른 폰트의 글자를 비율을 유지해 렌더링하는 도우미입니다.
Topo 테스트는 `notebooks/test_topology_fonts.ipynb`, 실행 모듈은
`stroke_model/font_topology.py`, 설명은 `docs/stroke_model/TOPOLOGY_FONTS.md`에 있습니다.

- 재사용할 코드와 작은 설정 파일은 이 폴더에서 버전 관리합니다.
- 실험 목적, 실행 방법과 결과를 코드와 함께 문서화합니다.
- 원본 폰트·데이터는 `data/`, 생성 결과는 `outputs/`에 두며 두 폴더는 Git에서 제외합니다.
- 결과 기록에는 사용한 폰트 식별 정보, 설정 및 Git 커밋을 남깁니다.
