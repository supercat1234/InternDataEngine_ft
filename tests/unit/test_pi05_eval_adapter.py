import importlib.util
from pathlib import Path
import unittest
import numpy as np

PATH = Path(__file__).resolve().parents[2] / 'scripts/eval/cup_plate_adapter.py'
spec = importlib.util.spec_from_file_location('adapter', PATH)
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


class AdapterTests(unittest.TestCase):
    def test_action_mapping(self):
        action = np.zeros(14)
        action[[6, 13]] = [1, .4]
        result = adapter.targets(action, np.tile([-4, 4], (14, 1)))
        np.testing.assert_allclose(result[[6, 13]], [.05, .02])
        self.assertEqual(adapter.ACTION_NAMES[7], 'right_joint_1.pos')

    def test_invalid_targets(self):
        limits = np.tile([-1, 1], (14, 1))
        with self.assertRaises(ValueError):
            adapter.targets(np.full(14, np.nan), limits)
        action = np.zeros(14)
        action[0] = 2
        with self.assertRaises(ValueError):
            adapter.targets(action, limits)

    def test_boundary_roundoff(self):
        action = np.zeros(14)
        action[0] = 1.01
        limited = adapter.targets(action, np.tile([-1, 1], (14, 1)))
        self.assertEqual(limited[0], 1)

    def test_observation_and_rgb(self):
        robot = {f'states.{side}_{part}.position': np.full(6 if part == 'joint' else 1, value)
                 for side in ('left', 'right') for part, value in [('joint', .2), ('gripper', .04)]}
        image = np.zeros((480, 640, 3), np.uint8)
        image[..., 0] = 255
        obs = adapter.observation({'robots': {'split_aloha': robot},
                                   'cameras': {name: {'color_image': image} for name in adapter.CAMERAS.values()}})
        np.testing.assert_allclose(obs['observation.state'][[6, 13]], [.4, .4])
        frame = obs['observation.images.TOP_CAM']
        self.assertEqual(frame.shape, (256, 256, 3))
        np.testing.assert_array_equal(frame[128,128], [255,0,0])
        self.assertFalse(frame[:32].any())

if __name__ == '__main__':
    unittest.main()
