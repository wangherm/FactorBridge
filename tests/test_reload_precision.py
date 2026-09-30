from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from factorbridge.io import read_json, write_json
from factorbridge.llm import load_model
from scripts.archive_failed_smoke import archive_failed_smoke


class Parameter:
    def __init__(self, dtype, requires_grad=False):
        self.dtype, self.requires_grad = dtype, requires_grad


class Base:
    def __init__(self):
        self.config = SimpleNamespace(use_cache=True)
        self.weights = [('embed_tokens.weight', Parameter('bf16')), ('norm.weight', Parameter('bf16'))]
        self.gradient_checkpointing_enable = Mock()
    def named_parameters(self):
        return self.weights


class Wrapped:
    def __init__(self, base, trainable):
        self.base = base
        self.weights = base.weights + [('lora_A.default.weight', Parameter('fp32', trainable))]
    def named_parameters(self):
        return self.weights


class ReloadPrecision(unittest.TestCase):
    def config(self):
        c = read_json(Path(__file__).resolve().parents[1] / 'configs/stage1.json')
        c['model_revision'] = 'a' * 40
        return c

    def dependencies(self):
        # Unit doubles isolate the loader's precision-policy branch. They do not
        # stand in for a real Qwen/GPU save-reload test or claim numerical success.
        torch = SimpleNamespace(bfloat16='bf16', float16='fp16', cuda=SimpleNamespace(is_bf16_supported=lambda: True))
        loader = Mock(side_effect=lambda *a, **k: Base())
        transformers = SimpleNamespace(AutoModelForCausalLM=SimpleNamespace(from_pretrained=loader),
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: SimpleNamespace(pad_token_id=0)),
            BitsAndBytesConfig=lambda **k: k)
        def prepare(base, **kwargs):
            for _, p in base.named_parameters():
                p.dtype = 'fp32'
            return base
        prepare_mock = Mock(side_effect=prepare)
        peft = SimpleNamespace(LoraConfig=lambda **k: k,
            get_peft_model=lambda b, c: Wrapped(b, True), prepare_model_for_kbit_training=prepare_mock,
            PeftModel=SimpleNamespace(from_pretrained=lambda b, p, is_trainable: Wrapped(b, is_trainable)))
        return torch, transformers, peft, prepare_mock, loader

    def test_qlora_train_reload_and_frozen_baseline_share_fp32_base_policy(self):
        torch, transformers, peft, prepare_mock, loader = self.dependencies()
        with patch('factorbridge.llm.require_gpu', return_value=torch), patch.dict('sys.modules', {'transformers': transformers, 'peft': peft}):
            training, _, _ = load_model(self.config())
            reloaded, _, _ = load_model(self.config(), adapter='saved-adapter', trainable=False)
            frozen, _, _ = load_model(self.config(), trainable=False)
        self.assertEqual([p.dtype for _, p in training.base.named_parameters()], ['fp32', 'fp32'])
        self.assertEqual([p.dtype for _, p in reloaded.base.named_parameters()], ['fp32', 'fp32'])
        self.assertEqual([p.dtype for _, p in frozen.named_parameters()], ['fp32', 'fp32'])
        self.assertEqual([c.kwargs['use_gradient_checkpointing'] for c in prepare_mock.call_args_list], [True, False, False])
        self.assertTrue(all(c.kwargs['gradient_checkpointing_kwargs'] == {'use_reentrant': False} for c in prepare_mock.call_args_list))
        self.assertTrue(all(c.kwargs['dtype'] == 'bf16' and 'torch_dtype' not in c.kwargs for c in loader.call_args_list))

    def test_nonquantized_lora_preserves_dtype_and_uses_explicit_checkpoint_mode(self):
        torch, transformers, peft, prepare_mock, _ = self.dependencies()
        c = dict(self.config(), load_in_4bit=False)
        with patch('factorbridge.llm.require_gpu', return_value=torch), patch.dict('sys.modules', {'transformers': transformers, 'peft': peft}):
            model, _, _ = load_model(c)
        prepare_mock.assert_not_called()
        model.base.gradient_checkpointing_enable.assert_called_once_with(gradient_checkpointing_kwargs={'use_reentrant': False})
        self.assertTrue(all(p.dtype == 'bf16' for _, p in model.base.named_parameters()))

    def test_archive_preserves_failed_adapter_and_original_config(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = root / 'config.json'
            write_json(config, {'run_dir': str(root)})
            before = config.read_bytes()
            write_json(root / 'smoke/status.json', {'status': 'failed', 'error': 'Save/reload logits differ'})
            (root / 'smoke/adapter').mkdir()
            (root / 'smoke/adapter/evidence.txt').write_text('failed-attempt evidence')
            target = archive_failed_smoke(config)
            self.assertFalse((root / 'smoke').exists())
            self.assertEqual((target / 'adapter/evidence.txt').read_text(), 'failed-attempt evidence')
            self.assertEqual(read_json(target / 'status.json')['status'], 'failed')
            self.assertEqual(config.read_bytes(), before)
            self.assertTrue((target / 'archive_receipt.json').exists())

    def test_running_and_passed_smokes_are_never_moved(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = root / 'config.json'
            write_json(config, {'run_dir': str(root)})
            for status in ('running', 'passed'):
                write_json(root / 'smoke/status.json', {'status': status})
                with self.assertRaisesRegex(ValueError, 'Only status=failed'):
                    archive_failed_smoke(config)
                self.assertTrue((root / 'smoke/status.json').exists())


if __name__ == '__main__':
    unittest.main()
