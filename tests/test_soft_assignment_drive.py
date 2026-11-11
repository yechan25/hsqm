import ast
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch
import numpy as np
from PIL import Image
from diffusion_model.saved_repair_demo import pack_saved_generation,prepare_saved_demo


ROOT=Path(__file__).resolve().parents[1]


class DriveRepairTests(unittest.TestCase):
    def mount_prefix(self):
        data=json.loads((ROOT/'notebooks/soft_assignment_repair.ipynb').read_text(encoding='utf-8'))
        code=ast.parse(''.join(data['cells'][1]['source']))
        stop=next(i for i,node in enumerate(code.body) if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='root' for t in node.targets))
        return compile(ast.Module(body=code.body[:stop],type_ignores=[]),'mount_prefix','exec')

    def test_mount_precedes_downloads_and_uses_official_api(self):
        mount=Mock()
        with patch.dict('sys.modules',{'google.colab':SimpleNamespace(drive=SimpleNamespace(mount=mount))}),patch('os.path.ismount',return_value=False):
            exec(self.mount_prefix(),{})
        mount.assert_called_once_with('/content/drive',force_remount=False,timeout_ms=180000)

    def test_existing_mount_is_reused(self):
        mount=Mock()
        with patch.dict('sys.modules',{'google.colab':SimpleNamespace(drive=SimpleNamespace(mount=mount))}),patch('os.path.ismount',return_value=True):
            exec(self.mount_prefix(),{})
        mount.assert_not_called()

    def test_mount_error_stops_without_hiding_original_cause(self):
        failure=ValueError('auth timeout');mount=Mock(side_effect=failure)
        with patch.dict('sys.modules',{'google.colab':SimpleNamespace(drive=SimpleNamespace(mount=mount))}),patch('os.path.ismount',return_value=False):
            with self.assertRaisesRegex(RuntimeError,'Drive 인증/마운트') as context:exec(self.mount_prefix(),{})
        self.assertIs(context.exception.__cause__,failure)

    def test_drive_report_can_restore_input_without_uploading_dataset(self):
        workspace=ROOT/'outputs';workspace.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=workspace) as directory:
            folder=Path(directory)
            image=folder/'page_001_baseline.png';Image.new('RGB',(64,64),'white').save(image)
            rows=[]
            for i in range(4):
                filename=f'sample_{i:04d}.npz'
                np.savez(folder/filename,reference=np.ones((1,16,16),np.float32),reference_strokes=np.ones((1,16,16),np.float32))
                rows.append(dict(method='reference',slot=i,npz='/old/'+filename,
                    crop_bbox=[(i%2)*32,(i//2)*32,(i%2+1)*32,(i//2+1)*32]))
            report=dict(checkpoint='/old/best.pt',checkpoint_sha256='abc',results=rows,
                pages=[dict(method='baseline',input='/old/'+image.name)])
            report_path=folder/'generation_report.json';report_path.write_text(json.dumps(report))
            archive=pack_saved_generation(report_path,folder/'source.zip')
            prepared=prepare_saved_demo(archive,folder/'prepared')
            self.assertEqual(Path(prepared['source']).read_bytes(),image.read_bytes())
            self.assertEqual(prepared['checkpoint_sha256'],'abc')


if __name__=='__main__':unittest.main()
