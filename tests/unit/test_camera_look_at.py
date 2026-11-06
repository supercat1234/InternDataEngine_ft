"""Check USD camera axes without requiring an Isaac application."""
import ast
from pathlib import Path
import unittest

import numpy as np
from scipy.spatial.transform import Rotation


path = Path(__file__).resolve().parents[2] / "workflows/simbox/core/utils/camera_utils.py"
functions = [n for n in ast.parse(path.read_text()).body
             if isinstance(n, ast.FunctionDef) and n.name in ["look_at_usd_orientation", "sample_camera_pose"]]
scope = {"np": np}
exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), scope)
look_at = scope["look_at_usd_orientation"]
sample_pose = scope["sample_camera_pose"]


class CameraLookAtTests(unittest.TestCase):
    def test_mast_camera_looks_at_table_with_positive_x_to_image_right(self):
        position, target = np.array([0., -.43, 1.44]), np.array([0., 0., .75])
        q = look_at(position, target)
        rotation = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
        expected = target - position
        expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(rotation @ [0, 0, -1], expected, atol=1e-8)
        np.testing.assert_allclose(rotation @ [1, 0, 0], [1, 0, 0], atol=1e-8)
        camera_target = rotation.T @ (target - position)
        np.testing.assert_allclose(camera_target[:2], 0, atol=1e-8)
        self.assertLess(camera_target[2], 0)

    def test_degenerate_look_directions_are_rejected(self):
        with self.assertRaises(ValueError):
            look_at([0, 0, 0], [0, 0, 0])
        with self.assertRaises(ValueError):
            look_at([0, 0, 1], [0, 0, 0])

    def test_jitter_respects_xyz_and_total_angle_bounds_about_look_at(self):
        cfg = dict(translation=[0, -.48, 1.44], orientation=[1, 0, 0, 0],
                   look_at=[0, 0, .75], camera_axes='usd')
        nominal = Rotation.from_quat(look_at(cfg['translation'], cfg['look_at'])[[1, 2, 3, 0]])
        rng = np.random.default_rng(42)
        translations = []
        for _ in range(400):
            state = sample_pose(cfg, [.01, .01, .01], 2., rng)
            delta = np.array(state['translation']) - cfg['translation']
            self.assertTrue(np.all(np.abs(delta) <= .01 + 1e-12))
            q = np.asarray(state['orientation_wxyz'])
            self.assertAlmostEqual(np.linalg.norm(q), 1.)
            actual = Rotation.from_quat(q[[1, 2, 3, 0]])
            self.assertLessEqual(np.rad2deg((actual * nominal.inv()).magnitude()), 2.000001)
            translations.append(delta)
        translations = np.asarray(translations)
        self.assertTrue(np.all(translations.min(axis=0) < -.009))
        self.assertTrue(np.all(translations.max(axis=0) > .009))

    def test_zero_noise_uses_look_at_and_seeded_resets_do_not_accumulate(self):
        cfg = dict(translation=[0, -.48, 1.44], orientation=[1, 0, 0, 0],
                   look_at=[0, 0, .75], camera_axes='usd')
        zero = sample_pose(cfg, [0, 0, 0], 0, np.random.default_rng(42))
        np.testing.assert_allclose(zero['translation'], cfg['translation'])
        np.testing.assert_allclose(zero['orientation_wxyz'], look_at(cfg['translation'], cfg['look_at']))
        first = sample_pose(cfg, [.01] * 3, 2, np.random.default_rng(42))
        cfg['mem_pose'] = first
        again = sample_pose(cfg, [.01] * 3, 2, np.random.default_rng(42))
        self.assertEqual(first, again)
        self.assertEqual(cfg['translation'], [0, -.48, 1.44])

    def test_scalar_translation_noise_retains_sphere_bound_and_bad_limits_fail(self):
        cfg = dict(translation=[0, 0, 0], orientation=[1, 0, 0, 0], camera_axes='usd')
        state = sample_pose(cfg, .01, 0, np.random.default_rng(42))
        self.assertLessEqual(np.linalg.norm(state['translation']), .01)
        for noise in [-.01, [.01, -.01, .01], [0, 0], [0, np.nan, 0]]:
            with self.assertRaises(ValueError):
                sample_pose(cfg, noise, 2)
        with self.assertRaises(ValueError):
            sample_pose(cfg, 0, -2)


if __name__ == "__main__":
    unittest.main()
