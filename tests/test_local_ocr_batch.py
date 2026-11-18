
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import ast
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class Mask(list):
    def __invert__(self):
        return Mask(not x for x in self)

class Ids(list):
    def isin(self, values):
        return Mask(x in values for x in self)

class Frame:
    def __init__(self, rows):
        self.rows = rows
        self.id = Ids(r['id'] for r in rows)
    def __len__(self): return len(self.rows)
    def __getitem__(self, mask): return Frame([r for r, keep in zip(self.rows, mask) if keep])
    def to_dict(self, mode): return self.rows

class Bar:
    def __init__(self, **kwargs): pass
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def update(self, *args): pass
    def set_postfix(self, **kwargs): pass


class BatchTests(unittest.TestCase):
    def test_resume_oom_image_limit_and_interrupt(self):
        class OOM(Exception): pass
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True,
                    empty_cache=lambda: None, OutOfMemoryError=OOM))
        engine = SimpleNamespace(load_ocr=lambda: (object(), object()))
        with patch.dict(sys.modules, {'torch': torch, 'ocr_engine': engine,
                                     'tqdm.auto': SimpleNamespace(tqdm=Bar)}):
            spec = importlib.util.spec_from_file_location('batch_runner_test', ROOT / 'src/ocr_local_runner.py')
            runner = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(runner)
        df = Frame([dict(id=str(i), path=f'{i}.jpg') for i in range(7)])
        calls = []
        def infer(model, proc, rows, data_dir, cap):
            calls.append([r['id'] for r in rows])
            if len(rows) > 2: raise OOM()
            return [dict(r, ocr_raw='text', ocr_tokens=1, ocr_truncated=False,
                         max_new_tokens=cap, prompt='Spotting:') for r in rows]
        runner.infer_rows = infer
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / 'out.jsonl'
            runner.run_local(df, Path(d), out, 1024, 'Spotting:', batch_size=4, max_images=3)
            self.assertEqual(calls, [['0', '1', '2'], ['0'], ['1'], ['2']])
            self.assertEqual(len(out.read_text().splitlines()), 3)
            runner.infer_rows = lambda *args: (_ for _ in ()).throw(KeyboardInterrupt())
            runner.run_local(df, Path(d), out, 1024, 'Spotting:', batch_size=2)
            self.assertEqual(len(out.read_text().splitlines()), 3)
            runner.infer_rows = infer
            runner.run_local(df, Path(d), out, 1024, 'Spotting:', batch_size=2)
            rows = [json.loads(s) for s in out.read_text().splitlines()]
            self.assertEqual([r['id'] for r in rows], list(map(str, range(7))))

    def test_batch_left_padding_and_eos(self):
        class Inputs(dict):
            def to(self, device): return self
        class Output:
            def __getitem__(self, key):
                assert key[1].start == 9
                return self
            def tolist(self): return [[11, 2, 0, 0], [12, 13, 14, 15]]
        class Proc:
            tokenizer = SimpleNamespace(padding_side='right')
            def apply_chat_template(self, messages, processor_kwargs=None, **kwargs):
                assert processor_kwargs == {'padding': True}
                return Inputs(input_ids=SimpleNamespace(shape=(2, 9)))
            def decode(self, tokens, **kwargs): return str(tokens)
        model = SimpleNamespace(device='cuda', generation_config=SimpleNamespace(eos_token_id=2),
                                generate=lambda **kwargs: Output())
        scope = dict(torch=SimpleNamespace(inference_mode=lambda: lambda f: f), SPOTTING_PROMPT='Spotting:')
        node = next(n for n in ast.parse((ROOT / 'src/ocr_engine.py').read_text(encoding='utf-8')).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'run_spotting_batch')
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<batch>', 'exec'), scope)
        proc = Proc()
        outputs = scope['run_spotting_batch'](model, proc, [object(), object()], 4)
        self.assertEqual(proc.tokenizer.padding_side, 'left')
        self.assertEqual(outputs, [('[11, 2]', False, 2), ('[12, 13, 14, 15]', True, 4)])


if __name__ == '__main__': unittest.main()
