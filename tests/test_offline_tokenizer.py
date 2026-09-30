import tempfile
from pathlib import Path
import unittest

from factorbridge.io import write_json, write_jsonl, sha256
from factorbridge.pretraining import verify_tokenizer_bundle
from scripts.finish_pretraining_offline import resumable, resume


class OfflineTokenizer(unittest.TestCase):
    def test_local_bundle_rejects_wrong_revision_modified_and_unlisted_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            files = ['tokenizer_config.json', 'tokenizer.json', 'chat_template.jinja', 'vocab.json', 'merges.txt']
            for name in files:
                (root / name).write_text('unit-test-file', encoding='utf-8')
            c = {'model_name': 'unit-test-model', 'model_revision': 'a' * 40}
            proof = dict(model_name=c['model_name'], revision=c['model_revision'], files={n: sha256(root / n) for n in files})
            write_json(root / 'offline_manifest.json', proof)
            self.assertEqual(verify_tokenizer_bundle(c, root)[0], root.resolve())
            with self.assertRaisesRegex(ValueError, 'model/revision'):
                verify_tokenizer_bundle(dict(c, model_revision='b' * 40), root)
            (root / 'unlisted.py').write_text('unexpected')
            with self.assertRaisesRegex(ValueError, 'Unlisted'):
                verify_tokenizer_bundle(c, root)
            (root / 'unlisted.py').unlink()
            (root / 'tokenizer.json').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                verify_tokenizer_bundle(c, root)

    def test_completed_run_is_not_resumable(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_jsonl(root / 'pretraining_commands.jsonl', [{'step': 'tokenizer_length_and_completion_mask_check', 'status': 'failed'}])
            self.assertTrue(resumable(root))
            write_json(root / 'STOP_BEFORE_TRAINING.json', {'pretraining_checks': 'passed'})
            self.assertFalse(resumable(root))

    def test_frozen_run_cannot_be_resumed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_jsonl(root / 'pretraining_commands.jsonl', [{'step': 'tokenizer_length_and_completion_mask_check', 'status': 'failed'}])
            write_json(root / 'config.json', {'run_dir': str(root), 'stage': 'noise_recovery', 'defer_public_test': True})
            write_json(root / 'frozen_protocol.json', {})
            with self.assertRaisesRegex(ValueError, 'beyond pretraining'):
                resume(root, root / 'unused-tokenizer')


if __name__ == '__main__':
    unittest.main()
