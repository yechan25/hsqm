# Drive ZIP 기반 획 학습

## 기존 평가 노트북 재사용

`notebooks/test_reference_v2.ipynb`의 기본 MODE="zip_val"은 최신 stroke_zip_runs/*/best.pt를 선택한다.
checkpoint에 경로를 지정하면 특정 실행을 평가한다. 학습 당시 저장한 selected_paths.csv와 run_config의 seed/크기로
로컬 데이터를 복원하고 val만 평가한다. 학습 당시 ZIP 크기/수정 시각과 다르면 복원을 중단한다.
기존 evaluate/preview_reference를 그대로 사용하여 val Dice, 정답 커버리지, exclusive 예측, 획별 확률/soft mask/entropy를 저장한다.
결과는 checkpoint 옆 val_results_시간에 저장한다. 이는 모델 선택에 사용한 validation이며 독립 test 평가가 아니다.
MODE="test_csv"는 기존 별도 test CSV 전체 평가를 유지한다.

[Colab에서 실행](https://colab.research.google.com/github/yechan25/hsqm/blob/main/notebooks/train_stroke_zips.ipynb)하고 GPU 런타임에서 위부터 실행한다.
첫 셀은 GitHub main 최신 코드를 clone하고 전체 커밋 SHA를 출력한다. 실행 커밋을 결과 폴더의 git_commit.txt에도 저장한다.
모델 구현은 기존 ReferenceStrokeModel이며 기본은 Swin 사전학습 + 새 분할 head 학습이다.
호환되는 reference_v2, image_size=128 best.pt가 있으면 INIT_CHECKPOINT로 가중치를 초기화할 수 있다.
optimizer/epoch 복원은 하지 않는다. 자모 모델 가중치가 획 분리 성능을 높인다는 보장은 없다.

기본 폴더는 `/content/drive/Shareddrives/2026 자율연구/HSQM/dataset/automated_train_dataset`.
input_stroke.zip, input.zip, target_stroke.zip, target.zip을 런타임 로컬로 한 번 복사하고 해제한다.
CSV는 데이터 폴더 바로 아래의 path.csv/paths.csv를 우선 검색하고, 없으면 해제된 ZIP 내부를 찾는다.
여러 후보면 중단하며 CSV_PATH로 선택한다. CSV가 나중에 올라오면 준비 셀만 재실행한다.
데이터 폴더 밖의 이전 train/test CSV는 검색하지 않는다.

CSV의 입력은 input, 기준 글자는 target, 기준 획은 target_stroke, 정답 획은 input_stroke에 해당하는 기존 CSV 규약을 사용한다.
실제 컬럼 별칭은 convert_reference_csv.py에 정의되어 있다. 네 ZIP만으로 획 순서/의미 대응을 추측하지 않는다.
기준/정답 획 목록은 같은 순서여야 하며, preview와 선택된 컬럼을 학습 전에 확인한다.
CSV의 오래된 Drive 절대 경로는 로컬 파일 경로의 접미사로 대응한다. 모호한 동명 파일은 중단한다.
로컬 전용 Resolver를 사용하므로 CSV의 원래 Drive 이미지가 남아 있어도 개별 파일을 Drive에서 읽지 않는다.

변환 NPZ와 manifest는 /content에 보관하고 배치마다 로컬에서 읽는다. 추가 증강은 기본으로 하지 않는다.
Drive에는 HSQM/stroke_zip_runs/실행시간 아래 best.pt, history.json, run_config.json, CSV/변환 보고서와 소스 식별 정보를 저장한다.
변환 보고서는 교차/누락 픽셀을 제외하는 정답 커버리지를 포함한다. val Dice는 주석된 전경 픽셀에 한정된다.
writer/group ID가 있으면 그룹 분할하며 없으면 동일 입력 이미지 해시 단위로 80/20 분리한다.
필자 ID가 없을 때 필자 누수를 방지했다고 주장할 수 없다. 런타임 삭제 후에는 ZIP 복사·변환이 다시 필요하다.
ZIP 캐시는 경로/크기/수정 시각을 사용하므로 원본을 변경한 경우 새 업로드가 완료된 후 실행한다.

`python tests/build_stroke_zip_notebook.py`로 소스 변경 후 노트북을 다시 생성한다.
`python -m unittest discover -s tests -p test_stroke_zips.py -v`로 합성 ZIP→CSV→NPZ와 캐시/경로 보호를 검증한다.
실제 Drive ZIP/CSV와 GPU 학습 성능 검증은 Colab 실행에서 수행해야 한다.
