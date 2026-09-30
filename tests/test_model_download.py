import hashlib
import io
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from scripts.download_qwen_weights import download_file, manifest


class Response(io.BytesIO):
    def __init__(self, content, status=200, headers=None):
        super().__init__(content)
        self.status, self.headers = status, headers or {}


class ModelDownload(unittest.TestCase):
    @unittest.skipUnless(os.name == 'posix' and shutil.which('bash'), 'Shell syntax checked on Linux CI')
    def test_smoke_launcher_shell_syntax(self):
        path = Path(__file__).resolve().parents[1] / 'scripts/smoke_qwen_offline.sh'
        subprocess.run(['bash', '-n', str(path)], check=True, capture_output=True)

    def item(self, content):
        return {'name': 'fixture.bin', 'size': len(content), 'sha256': hashlib.sha256(content).hexdigest()}

    def test_resume_checks_range_and_verifies_final_hash(self):
        content = b'unit-test-download-content'
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'fixture.bin.part').write_bytes(content[:5])
            def opener(request, **kwargs):
                self.assertEqual(request.headers['Range'], 'bytes=5-')
                return Response(content[5:], 206, {'Content-Range': f'bytes 5-{len(content)-1}/{len(content)}'})
            result = download_file(self.item(content), root, opener=opener)
            self.assertEqual(result['status'], 'downloaded_and_verified')
            self.assertEqual((root / 'fixture.bin').read_bytes(), content)
            self.assertFalse((root / 'fixture.bin.part').exists())

    def test_ignored_range_restarts_instead_of_appending(self):
        content = b'unit-test-download-content'
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'fixture.bin.part').write_bytes(content[:5])
            download_file(self.item(content), root, opener=lambda *a, **k: Response(content))
            self.assertEqual((root / 'fixture.bin').read_bytes(), content)

    def test_wrong_hash_retains_partial_and_never_publishes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
                download_file(self.item(b'right'), root, opener=lambda *a, **k: Response(b'wrong'))
            self.assertFalse((root / 'fixture.bin').exists())
            self.assertEqual((root / 'fixture.bin.part').read_bytes(), b'wrong')

    def test_verify_only_never_downloads_and_rejects_corrupt_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            def forbidden(*args, **kwargs):
                self.fail('Network used in verification-only mode')
            with self.assertRaises(FileNotFoundError):
                download_file(self.item(b'right'), root, verify_only=True, opener=forbidden)
            (root / 'fixture.bin').write_bytes(b'wrong')
            with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
                download_file(self.item(b'right'), root, verify_only=True, opener=forbidden)

    def test_pinned_manifest_has_three_weight_shards(self):
        spec = manifest()
        self.assertEqual(len([r for r in spec['files'] if r['name'].endswith('.safetensors')]), 3)


if __name__ == '__main__':
    unittest.main()
