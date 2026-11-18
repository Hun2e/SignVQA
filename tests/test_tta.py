from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import math
import tempfile
import unittest

import pandas as pd

import common_eval as ce
import tta

# 원래 선지 a~d 기준의 "진짜" 확률. 네 순서로 보여 준 결과를 되돌려 평균하면 이 값이 나와야 합니다.
TRUE_PROBS = [0.1, 0.6, 0.2, 0.1]


def fake_results(true_probs, ids=("x",), flip_order=None):
    """순서마다 화면 자리 기준 로그확률을 만들어 ve.read 결과처럼 돌려줍니다."""
    results = {}
    for perm in tta.ORDERS:
        slots = [true_probs[ce.CHOICES.index(orig)] for orig in perm]
        if perm == flip_order:
            slots = list(reversed(slots))
        lp = {ch: math.log(p) for ch, p in zip(ce.CHOICES, slots)}
        results[perm] = pd.DataFrame({"id": list(ids), "letter_logprobs": [lp] * len(ids)})
    return results


class Order4Tests(unittest.TestCase):
    def test_every_option_visits_every_slot_once(self):
        for slot in range(len(ce.CHOICES)):
            self.assertEqual(sorted(perm[slot] for perm in tta.ORDERS), ce.CHOICES)

    def test_restored_average_recovers_original_probs(self):
        row = tta.order4(fake_results(TRUE_PROBS)).iloc[0]
        for col, want in zip(tta.PROB_COLS, TRUE_PROBS):
            self.assertAlmostEqual(row[col], want)
        self.assertEqual(row.pred, "b")
        self.assertFalse(row.disagree)

    def test_one_order_changing_its_mind_is_disagreement(self):
        row = tta.order4(fake_results(TRUE_PROBS, flip_order="cdab")).iloc[0]
        self.assertTrue(row.disagree)
        self.assertTrue(tta.gate(pd.DataFrame([row]))[0])

    def test_missing_order_rows_raise(self):
        results = fake_results(TRUE_PROBS, ids=("x", "y"))
        results["dabc"] = results["dabc"].iloc[:1]
        with self.assertRaises(ValueError):
            tta.order4(results)


class GateBlendTests(unittest.TestCase):
    def test_gate_thresholds_are_team_values(self):
        self.assertEqual((tta.GATE_MAX_PROB, tta.GATE_MARGIN, tta.BLEND_WEIGHT), (0.85, 0.25, 0.5))

    def test_blend_only_touches_reviewed_ids(self):
        base = pd.DataFrame([dict(id=i, **tta.summarize([0.45, 0.40, 0.1, 0.05])) for i in ("x", "y")])
        review = pd.DataFrame([dict(id="x", **tta.summarize([0.2, 0.7, 0.05, 0.05]))])
        out = tta.blend(base, review).set_index("id")
        self.assertEqual((out.loc["x", "pred"], out.loc["y", "pred"]), ("b", "a"))
        self.assertEqual(list(out.reviewed), [True, False])

    def test_review_outside_base_raises(self):
        base = pd.DataFrame([dict(id="x", **tta.summarize(TRUE_PROBS))])
        review = pd.DataFrame([dict(id="z", **tta.summarize(TRUE_PROBS))])
        with self.assertRaises(ValueError):
            tta.blend(base, review)


class ZoomImageTests(unittest.TestCase):
    def setUp(self):
        from PIL import Image
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.old_dir, ce.DATA_DIR = ce.DATA_DIR, Path(self.temp.name)
        self.addCleanup(setattr, ce, "DATA_DIR", self.old_dir)
        Image.new("RGB", (720, 960)).save(Path(self.temp.name) / "small.jpg")
        Image.new("RGB", (4000, 3000)).save(Path(self.temp.name) / "big.jpg")

    def area(self, name, fn):
        im = fn({"path": name})
        return im.width * im.height

    def test_small_image_grows_past_native_input(self):
        zoom = self.area("small.jpg", tta.zoom_image)
        self.assertGreater(zoom, self.area("small.jpg", ce.open_image))
        self.assertGreaterEqual(zoom, tta.ZOOM_MIN_PIXELS * 0.99)

    def test_big_image_is_capped(self):
        self.assertLessEqual(self.area("big.jpg", tta.zoom_image), tta.ZOOM_MAX_PIXELS * 1.01)


if __name__ == "__main__":
    unittest.main()
