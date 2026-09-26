"""Model-routing tests — every Gemini call site must hit the config tier
designed for its role (config.MODEL ROUTING):
  main solve / corrections / diagram recovery  -> GEMINI_SOLVE_MODEL_ID
  point-coordinate vision cross-check          -> GEMINI_VISION_MODEL_ID
  figure bounding-box location                 -> GEMINI_VISION_MODEL_ID
  heading + chapter-title page scans           -> GEMINI_SCAN_MODEL_ID
"""
import unittest
from unittest import mock

import config


def _fake_client(captured):
    class _Models:
        def generate_content(self, **kwargs):
            captured.append(kwargs.get("model"))
            resp = mock.Mock()
            resp.text = "{}"
            return resp
    client = mock.Mock()
    client.models = _Models()
    return client


class TestSolverRouting(unittest.TestCase):
    def test_main_solve_uses_solve_tier(self):
        import solver
        captured = []
        with mock.patch.object(solver, "get_client",
                               return_value=_fake_client(captured)):
            solver._call_gemini(b"%PDF", "prompt")
        self.assertEqual(captured, [config.GEMINI_SOLVE_MODEL_ID])

    def test_point_coordinate_check_uses_vision_tier(self):
        import solver
        captured = []
        with mock.patch.object(solver, "get_client",
                               return_value=_fake_client(captured)):
            solver._call_gemini_vision_point(b"png", "prompt")
        self.assertEqual(captured, [config.GEMINI_VISION_MODEL_ID])


class TestCorrectionRouting(unittest.TestCase):
    def test_correction_uses_solve_tier(self):
        import correction_engine
        captured = []
        with mock.patch.object(correction_engine.solver, "get_client",
                               return_value=_fake_client(captured)):
            correction_engine._call_gemini_correction("prompt")
        self.assertEqual(captured, [config.GEMINI_SOLVE_MODEL_ID])


class TestSafetyNetRouting(unittest.TestCase):
    def test_diagram_recovery_uses_solve_tier(self):
        import diagram_safety_net
        captured = []
        with mock.patch("solver.get_client", return_value=_fake_client(captured)):
            diagram_safety_net._call_gemini_recovery("prompt")
        self.assertEqual(captured, [config.GEMINI_SOLVE_MODEL_ID])


class TestVisionOcrRouting(unittest.TestCase):
    def setUp(self):
        from vision_ocr import _call_vision_batch
        self._call = _call_vision_batch

    def test_explicit_scan_model_is_used_when_passed(self):
        captured = []
        self._call(_fake_client(captured), [], "p", model=config.GEMINI_SCAN_MODEL_ID)
        self.assertEqual(captured, [config.GEMINI_SCAN_MODEL_ID])

    def test_default_is_vision_tier_not_solve_or_scan(self):
        # Figure-location callers omit the model: must land on the VISION
        # tier (accuracy-leaning default), never on the cheap scan tier.
        captured = []
        self._call(_fake_client(captured), [], "p")
        self.assertEqual(captured, [config.GEMINI_VISION_MODEL_ID])

    def test_tiers_are_distinct_and_configured(self):
        # The whole point of routing: three distinct models configured.
        self.assertTrue(config.GEMINI_SOLVE_MODEL_ID)
        self.assertTrue(config.GEMINI_VISION_MODEL_ID)
        self.assertTrue(config.GEMINI_SCAN_MODEL_ID)
        self.assertNotEqual(config.GEMINI_SOLVE_MODEL_ID,
                            config.GEMINI_SCAN_MODEL_ID)


if __name__ == "__main__":
    unittest.main()
