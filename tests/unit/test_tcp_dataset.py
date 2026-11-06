"""Check frame conventions, tool offsets and command/observation alignment."""

import importlib.util
from pathlib import Path
import pickle
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("tcp_export", ROOT / "scripts/datasets/export_cup_plate_lerobot.py")
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)


class TcpDatasetTests(unittest.TestCase):
    def test_translation_uses_current_tcp_axes(self):
        current = np.eye(4)
        current[:3, :3] = Rotation.from_euler("z", 90, degrees=True).as_matrix()
        target = current.copy()
        target[1, 3] = .02  # Base +Y is TCP +X.
        action = export.relative_tcp_actions(current[None], target[None], [1])[0]
        np.testing.assert_allclose(action, [.02, 0, 0, 0, 0, 0, 1], atol=1e-7)

    def test_rotation_composes_on_the_right(self):
        current = np.eye(4)
        current[:3, :3] = Rotation.from_euler("z", 90, degrees=True).as_matrix()
        target = current.copy()
        target[:3, :3] = current[:3, :3] @ Rotation.from_euler("x", 30, degrees=True).as_matrix()
        action = export.relative_tcp_actions(current[None], target[None], [0])[0]
        np.testing.assert_allclose(action[3:6], [np.pi / 6, 0, 0], atol=1e-7)
        np.testing.assert_allclose(export.apply_tcp_action(current, action), target, atol=1e-7)

    def test_rotating_flange_moves_offset_tcp(self):
        tool = np.eye(4)
        tool[2, 3] = .12
        target_flange = np.eye(4)
        target_flange[:3, :3] = Rotation.from_euler("y", 90, degrees=True).as_matrix()
        action = export.relative_tcp_actions(tool[None], (target_flange @ tool)[None], [1])[0]
        np.testing.assert_allclose(action[:3], [.12, 0, -.12], atol=1e-7)

    def test_angle_wrap_is_small_rotation(self):
        poses = np.repeat(np.eye(4)[None], 2, axis=0)
        poses[:, :3, :3] = Rotation.from_euler("z", [[179], [-179]], degrees=True).as_matrix()
        action = export.relative_tcp_actions(poses[:1], poses[1:], [0])[0]
        np.testing.assert_allclose(action[3:6], [0, 0, np.deg2rad(2)], atol=1e-7)

    def test_invalid_rotation_and_gripper_rejected(self):
        pose = np.eye(4)[None]
        invalid = pose.copy()
        invalid[0, 0, 0] = -1
        with self.assertRaises(ValueError):
            export.relative_tcp_actions(invalid, pose, [0])
        with self.assertRaises(ValueError):
            export.relative_tcp_actions(pose, pose, [2])

    def test_urdf_fixed_origin_precedes_joint_rotation(self):
        urdf = '''<robot name="test"><link name="base"/><link name="tool"/>
        <joint name="j" type="revolute"><parent link="base"/><child link="tool"/>
        <origin xyz="1 0 0" rpy="0 0 1.5707963267948966"/><axis xyz="1 0 0"/>
        </joint></robot>'''
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.urdf"
            path.write_text(urdf)
            fk = export.UrdfFK(path, "base", "tool", ["j"])
            actual = fk([[np.pi / 2]])[0]
        np.testing.assert_allclose(actual[:3, 3], [1, 0, 0], atol=1e-7)
        np.testing.assert_allclose(actual[:3, :3], [[0, 0, 1], [1, 0, 0], [0, 1, 0]], atol=1e-7)

    def episode(self):
        cfg = {"cameras": ["external"], "T_link6_tcp": np.eye(4), "gripper_max_width_m": .1,
               "max_fk_position_error_m": .003, "max_fk_rotation_error_rad": .005}
        current = np.repeat(np.eye(4)[None], 2, axis=0)
        command_q = np.zeros((2, 6))
        command_q[:, 0] = [.01, .03]
        values = {"states.left_joint.position": np.zeros((2, 6)),
                  "master_actions.left_joint.position": command_q, "T_base_ee_fl": current,
                  "states.left_gripper.position": [[.05], [.1]],
                  "master_actions.left_gripper.openness": [0, 1],
                  "images.rgb.external/0000": 0, "images.rgb.external/0001": 0}
        class Txn:
            def get(self, key):
                value = values.get(key.decode())
                return pickle.dumps(value) if value is not None else None
        def fk(q):
            transforms = np.repeat(np.eye(4)[None], len(q), axis=0)
            transforms[:, 0, 3] = q[:, 0]
            return transforms
        meta = {"num_steps": 2, "image_valid_step_ids": {"images.rgb.external": [0, 1]}}
        return Txn(), meta, cfg, fk

    def test_action_is_current_command_not_next_observation(self):
        frames, report = export.prepare_episode(*self.episode())
        # Observations are static but commands differ, including the last frame.
        np.testing.assert_allclose(frames["action"][:, 0], [.01, .03])
        np.testing.assert_allclose(frames["action"][:, -1], [0, 1])
        np.testing.assert_allclose(frames["observation.state"][:, -1], [.5, 1])
        self.assertLess(report["max_reconstruction_error"], 1e-7)

    def test_image_time_gaps_are_rejected(self):
        txn, meta, cfg, fk = self.episode()
        meta["image_valid_step_ids"]["images.rgb.external"] = [0, 2]
        with self.assertRaisesRegex(ValueError, "resampling"):
            export.prepare_episode(txn, meta, cfg, fk)

    def test_urdf_usd_frame_mismatch_is_rejected(self):
        txn, meta, cfg, fk = self.episode()
        def wrong_fk(q):
            poses = fk(q)
            poses[:, 2, 3] += .1
            return poses
        with self.assertRaisesRegex(ValueError, "USD/URDF"):
            export.prepare_episode(txn, meta, cfg, wrong_fk)


if __name__ == "__main__":
    unittest.main()
