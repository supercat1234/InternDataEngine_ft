"""Check future-feedback alignment, arm order, gripper units and image layout."""

import importlib.util
from pathlib import Path
import pickle
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/datasets"))
spec = importlib.util.spec_from_file_location("robotcooking_export", ROOT / "scripts/datasets/export_cup_plate_robotcooking.py")
ex = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ex)


class RobotCookingExportTests(unittest.TestCase):
    def source(self, count=6):
        values = {}
        for arm, sign in [("left", 1), ("right", -1)]:
            values[f"states.{arm}_joint.position"] = sign * np.arange(count * 6).reshape(count, 6) * .01
            values[f"states.{arm}_gripper.position"] = np.linspace(0, .1, count)[:, None]
            values[f"master_actions.{arm}_joint.position"] = np.full((count, 6), 99.)
        meta = {"num_steps": count, "image_valid_step_ids": {f"images.rgb.{cam}": list(range(count)) for cam in ex.CAMERAS.values()}}
        class Txn:
            def get(self, key):
                value = values.get(key.decode())
                return pickle.dumps(value) if value is not None else None
        return Txn(), meta, values

    def test_action_uses_two_future_measured_frames_and_preserves_arm_order(self):
        txn, meta, _ = self.source()
        state, action = ex.prepare_joint_episode(txn, meta)
        self.assertEqual(state.shape, (4, 14))
        np.testing.assert_allclose(state[0, :6], np.arange(6) * .01)
        np.testing.assert_allclose(action[0, :6], np.arange(12, 18) * .01)
        np.testing.assert_allclose(action[0, 7:13], -np.arange(12, 18) * .01)
        np.testing.assert_allclose(action[0, [6, 13]], [.4, .4])
        self.assertEqual(state.dtype, np.float32)

    def test_last_label_reaches_last_source_state_without_crossing_episode(self):
        txn, meta, _ = self.source()
        state, action = ex.prepare_joint_episode(txn, meta)
        np.testing.assert_allclose(action[-1, :6], np.arange(30, 36) * .01)
        np.testing.assert_allclose(action[-1, [6, 13]], [1, 1])

    def test_missing_image_steps_rejected(self):
        txn, meta, _ = self.source()
        meta["image_valid_step_ids"]["images.rgb.external"].pop()
        with self.assertRaisesRegex(ValueError, "image step"):
            ex.prepare_joint_episode(txn, meta)

    def test_nonfinite_joints_and_invalid_gripper_rejected(self):
        txn, meta, values = self.source()
        values["states.right_joint.position"][0, 0] = np.nan
        with self.assertRaises(ValueError):
            ex.prepare_joint_episode(txn, meta)
        txn, meta, values = self.source()
        values["states.left_gripper.position"][0, 0] = .2
        with self.assertRaisesRegex(ValueError, "gripper width"):
            ex.prepare_joint_episode(txn, meta)

    def test_short_episode_and_zero_offset_rejected(self):
        txn, meta, _ = self.source(2)
        with self.assertRaises(ValueError):
            ex.prepare_joint_episode(txn, meta)
        txn, meta, _ = self.source()
        with self.assertRaises(ValueError):
            ex.prepare_joint_episode(txn, meta, future_offset=0)

    def test_letterbox_preserves_aspect_ratio_channels_and_black_padding(self):
        image = np.full((4, 8, 3), [255, 10, 0], dtype=np.uint8)
        output = ex.letterbox(image, 8, 8)
        np.testing.assert_array_equal(output[2:6], image)
        self.assertFalse(output[:2].any())
        self.assertFalse(output[6:].any())


if __name__ == "__main__":
    unittest.main()
