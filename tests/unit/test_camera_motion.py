"""Camera pan/zoom estimation on synthetic textured frames with known shifts."""

import unittest

import cv2
import numpy as np

from src.preprocessing.camera_motion import (
    IDENTITY, CameraMotionEstimator, apply_affine, compose, invert,
)


def texture(seed=0):
    noise = np.random.default_rng(seed).integers(0, 255, (540, 960), np.uint8)
    return cv2.cvtColor(cv2.GaussianBlur(noise, (5, 5), 0), cv2.COLOR_GRAY2BGR)


def shifted(image, dx, dy):
    """Content moves by (dx, dy) pixels, as when the camera pans the other way."""
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]),
                          borderMode=cv2.BORDER_REFLECT)


class AffineTests(unittest.TestCase):
    def test_compose_and_invert_round_trip(self):
        matrix = (1.1, 0.05, 0.02, -0.05, 1.1, -0.03)
        for value, expected in zip(compose(matrix, invert(matrix)), IDENTITY):
            self.assertAlmostEqual(value, expected, places=9)
        point = apply_affine(compose((1, 0, .1, 0, 1, 0), (2, 0, 0, 0, 2, 0)), (.2, .3))
        self.assertAlmostEqual(point[0], .5)
        self.assertAlmostEqual(point[1], .6)


class EstimatorTests(unittest.TestCase):
    def test_pan_maps_current_frame_back_to_shot_reference(self):
        estimator = CameraMotionEstimator()
        base = texture()
        first = estimator.update(base, [])
        self.assertIsNone(first.step_reliable)
        motion = estimator.update(shifted(base, 12, -6), [])
        self.assertTrue(motion.step_reliable)
        self.assertEqual(motion.shot, 0)
        self.assertAlmostEqual(motion.step_shift, np.hypot(12, 6) / 960, places=3)
        x, y = apply_affine(motion.to_shot_reference, (.5 + 12 / 960, .5 - 6 / 540))
        self.assertAlmostEqual(x, .5, places=3)
        self.assertAlmostEqual(y, .5, places=3)

    def test_still_camera_stays_identity_and_cut_starts_new_shot(self):
        estimator = CameraMotionEstimator()
        base = texture()
        estimator.update(base, [])
        self.assertEqual(estimator.update(base.copy(), [(.4, .2, .6, .9)]).to_shot_reference, IDENTITY)
        cut = estimator.update(texture(seed=1), [])
        self.assertFalse(cut.step_reliable)
        self.assertEqual((cut.shot, cut.to_shot_reference), (1, IDENTITY))


if __name__ == "__main__":
    unittest.main()
