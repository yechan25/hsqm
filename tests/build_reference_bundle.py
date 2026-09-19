"""Package V2 sources and a five-cell Colab entry point, without dataset/weights."""
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    cells = []
    def markdown(text):
        cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': text.splitlines(keepends=True)})
    def code(text):
        compile(text, '<colab>', 'exec')
        cells.append({'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [],
                      'source': text.splitlines(keepends=True)})
    markdown('# 기준 획 모델 V2\nGPU 런타임을 선택하세요. 학습된 모델이 아니라 새 모델의 학습 노트북입니다.\n'
             '첫 셀이 실제 GitHub 저장소에서 코드를 자동으로 받습니다. ZIP 업로드는 필요 없습니다. '
             '두 번째 셀이 기존 paths.csv를 자동 변환합니다. 기준/정답 획은 CSV에서 같은 순서여야 합니다. '
             '데이터 형식은 REFERENCE_V2.md를 참조하세요. 기존 best.pt와는 호환되지 않습니다.\n')
    code('''from pathlib import Path
import sys, subprocess, tempfile
root = Path(tempfile.mkdtemp(prefix="reference_v2_", dir="/content"))
subprocess.run(["git", "clone", "--depth", "1", "--branch", "main",
                "https://github.com/yechan25/hsqm.git", str(root)], check=True)
if not (root / "requirements.txt").is_file():
    raise FileNotFoundError("저장소에서 requirements.txt를 찾지 못했습니다.")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(root / "requirements.txt")], check=True)
sys.path.insert(0, str(root))
for key in list(sys.modules):
    if key == "stroke_model" or key.startswith("stroke_model."):
        del sys.modules[key]
commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "--short", "HEAD"], text=True).strip()
print("모델 코드 준비 완료 / GitHub 버전:", commit)
''')
    code('''from google.colab import drive
from datetime import datetime
drive.mount("/content/drive")
from stroke_model.convert_reference_csv import existing_path
from stroke_model.reference_cache import prepare_cached
from stroke_model.preview_reference import preview_reference
csv_path = existing_path("/content/drive/Shareddrives/2026 자율연구/HSQM/dataset/train_dataset/paths.csv")
# 원본 CSV/이미지는 변경하지 않습니다. train CSV에서만 80/20으로 나눕니다.
manifest = prepare_cached(csv_path, csv_path.parents[2] / "reference_v2_data", rebuild=False)
prepared = manifest.parent
preview_reference(manifest)
print("데이터 준비 완료:", manifest)
''')
    code('''# 학습 데이터만 증강합니다. 0으로 바꾸면 증강 없이 비교할 수 있습니다.
variants = 20
from stroke_model.reference_cache import ensure_augmented
training_manifest = ensure_augmented(manifest, variants=variants, seed=42)
print("학습 manifest:", training_manifest)
''')
    code('''from datetime import datetime
epochs = 30
output = manifest.parent / ("reference_v2_run_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
subprocess.run([sys.executable, "-m", "stroke_model.train_reference",
                "--manifest", str(training_manifest), "--epochs", str(epochs),
                "--output", str(output)], cwd=root, check=True)
print("체크포인트:", output / "best.pt")
''')
    code('''# 학습이 끝난 후 가장 좋은 모델의 검증 이미지 결과를 확인합니다.
from stroke_model.preview_reference import preview_reference
preview_reference(manifest, checkpoint=output / "best.pt", count=5)
''')
    notebook = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                                           'colab': {'name': 'train_reference_v2.ipynb'}, 'accelerator': 'GPU'},
                'nbformat': 4, 'nbformat_minor': 5}
    path = ROOT / 'notebooks/train_reference_v2.ipynb'
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding='utf-8')
    dest = ROOT / 'outputs/reference_v2_source.zip'
    dest.parent.mkdir(exist_ok=True)
    files = ['requirements.txt', 'REFERENCE_V2.md', 'stroke_model/__init__.py', 'stroke_model/utils.py',
             'stroke_model/reference_model.py', 'stroke_model/train_reference.py', 'stroke_model/augment_reference.py',
             'stroke_model/convert_reference_csv.py', 'stroke_model/preview_reference.py',
             'stroke_model/reference_cache.py',
             'tests/test_reference_model.py', 'tests/test_reference_data.py']
    with zipfile.ZipFile(dest, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in files:
            archive.write(ROOT / name, name)
    print(path)
    print(dest)


if __name__ == '__main__':
    main()
