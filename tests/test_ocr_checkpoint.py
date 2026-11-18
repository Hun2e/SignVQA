
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import json
import tempfile
import unittest
from pathlib import Path

from ocr_checkpoint import atomic_json, durable_append, resume_ids, validate_metadata


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'cache.jsonl'
        self.row = dict(id='a', path='a.jpg', ocr_raw='한글', ocr_tokens=2,
                        ocr_truncated=False, max_new_tokens=1024, prompt='Spotting:')

    def resume(self):
        return resume_ids(self.path, 1024, 'Spotting:')

    def test_saved_record_survives_reopen(self):
        with self.path.open('a', encoding='utf-8') as log:
            durable_append(log, self.row)
        self.assertEqual(self.resume(), {'a'})
        self.assertEqual(self.resume(), {'a'})

    def test_torn_utf8_tail_backed_up_then_reprocessed(self):
        first = (json.dumps(self.row) + '\n').encode()
        original = first + b'{"id":"b","ocr_raw":"\xed\x95'
        self.path.write_bytes(original)
        self.assertEqual(self.resume(), {'a'})
        self.assertEqual(self.path.read_bytes(), first)
        self.assertEqual(next(self.path.parent.glob('*.bak')).read_bytes(), original)
        with self.path.open('a', encoding='utf-8') as log:
            durable_append(log, dict(self.row, id='b'))
        self.assertEqual(self.resume(), {'a', 'b'})

    def test_valid_record_without_newline(self):
        self.path.write_text(json.dumps(self.row), encoding='utf-8')
        self.assertEqual(self.resume(), {'a'})
        with self.path.open('a', encoding='utf-8') as log:
            durable_append(log, dict(self.row, id='b'))
        self.assertEqual(self.resume(), {'a', 'b'})

    def test_invalid_records_do_not_modify_file(self):
        valid = json.dumps(self.row) + '\n'
        cases = [valid + json.dumps(dict(self.row, ocr_raw='different')) + '\n', '{bad}\n' + valid,
                 json.dumps(dict(self.row, prompt='OCR:')) + '\n',
                 json.dumps(dict(self.row, max_new_tokens=512)) + '\n']
        for content in cases:
            with self.subTest(content=content):
                self.path.write_text(content, encoding='utf-8')
                before = self.path.read_bytes()
                with self.assertRaises(ValueError):
                    self.resume()
                self.assertEqual(self.path.read_bytes(), before)

    def test_duplicate_with_different_timing_is_backed_up_and_deduplicated(self):
        original = (json.dumps(dict(self.row, seconds=12)) + '\n' +
                    json.dumps(dict(self.row, seconds=7)) + '\n').encode()
        self.path.write_bytes(original)
        self.assertEqual(self.resume(), {'a'})
        self.assertEqual(len(self.path.read_text().splitlines()), 1)
        self.assertEqual(next(self.path.parent.glob('*.bak')).read_bytes(), original)
        self.assertEqual(json.loads(self.path.read_text())['seconds'], 12)

    def test_later_conflict_prevents_partial_deduplication(self):
        original = (json.dumps(self.row) + '\n') * 2
        original += json.dumps(dict(self.row, prompt='OCR:')) + '\n'
        self.path.write_text(original, encoding='utf-8')
        with self.assertRaises(ValueError):
            self.resume()
        self.assertEqual(self.path.read_text(encoding='utf-8'), original)

    def test_metadata_guards_and_atomic_progress(self):
        meta = self.path.with_suffix('.meta.json')
        settings = dict(rows=20, model='downloads/models/PaddleOCR-VL-1.6', max_pixels=1605632)
        atomic_json(meta, settings)
        validate_metadata(meta, dict(settings, rows=6714))
        with self.assertRaises(ValueError):
            validate_metadata(meta, dict(settings, max_pixels=1))
        atomic_json(meta, dict(settings, rows=6714))
        self.assertEqual(json.loads(meta.read_text())['rows'], 6714)


if __name__ == '__main__':
    unittest.main()
