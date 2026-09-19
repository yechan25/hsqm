"""Package the V2 sources and a four-cell Colab entry point, without dataset/weights."""
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
             '학습에는 검수된 NPZ 데이터/manifest.json이 필요하며 기존 paths.csv는 직접 지원하지 않습니다. '
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
drive.mount("/content/drive")
manifest = Path(input("Drive의 manifest.json 전체 경로: ").strip())
if manifest.suffix.lower() == ".csv":
    raise ValueError("기존 paths.csv를 입력하셨습니다. V2는 아직 CSV 직접 학습을 지원하지 않습니다. 획 대응을 확인한 데이터 변환이 필요합니다.")
if not manifest.is_file():
    raise FileNotFoundError(manifest)
from stroke_model.train_reference import read_manifest, StrokeDataset
rows = read_manifest(manifest)
for split in ("train", "val", "test"):
    subset = [r for r in rows if r["split"] == split]
    for i in range(len(subset)):
        StrokeDataset(subset)[i]
    print(split, len(subset))
print("데이터 형식과 그룹 분할 검사 완료")
''')
    code('''# 학습 데이터만 증강합니다. 0으로 바꾸면 증강 없이 비교할 수 있습니다.
variants = 20
training_manifest = manifest
if variants > 0:
    subprocess.run([sys.executable, "-m", "stroke_model.augment_reference",
                    "--manifest", str(manifest), "--variants", str(variants)], cwd=root, check=True)
    training_manifest = manifest.parent / "augmented_seed42" / "manifest.json"
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
    notebook = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                                           'colab': {'name': 'train_reference_v2.ipynb'}, 'accelerator': 'GPU'},
                'nbformat': 4, 'nbformat_minor': 5}
    path = ROOT / 'notebooks/train_reference_v2.ipynb'
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding='utf-8')
    dest = ROOT / 'outputs/reference_v2_source.zip'
    dest.parent.mkdir(exist_ok=True)
    files = ['requirements.txt', 'REFERENCE_V2.md', 'stroke_model/__init__.py', 'stroke_model/utils.py',
             'stroke_model/reference_model.py', 'stroke_model/train_reference.py', 'stroke_model/augment_reference.py',
             'tests/test_reference_model.py', 'tests/test_reference_data.py']
    with zipfile.ZipFile(dest, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in files:
            archive.write(ROOT / name, name)
    print(path)
    print(dest)


if __name__ == '__main__':
    main()
