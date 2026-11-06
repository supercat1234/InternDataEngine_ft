"""Shared platform sampling, live drive stabilization and saved-scene restoration."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import random
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import yaml
from pxr import Usd, UsdPhysics
from test_fixture_height_randomization import method_from_file

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('robot_height', ROOT / 'workflows/simbox/core/utils/robot_height_randomization.py')
height = importlib.util.module_from_spec(spec)
spec.loader.exec_module(height)


class RobotHeightTests(unittest.TestCase):
    def setUp(self):
        self.cfg = yaml.safe_load((ROOT / 'local_configs/cup_plate_desk/task.yaml').read_text())['tasks'][0]['robots'][0]

    def test_active_setup_keeps_table_fixed_and_uses_shared_platform(self):
        arena = yaml.safe_load((ROOT / 'local_configs/cup_plate_desk/arena.yaml').read_text())
        table = next(c for c in arena['fixtures'] if c['name'] == 'table')
        self.assertFalse(table['height_randomization']['enabled'])
        self.assertEqual(table['scale'], [.001, .001, .001])
        self.assertEqual(self.cfg['translation'], [0, -.68813, 0])
        self.assertEqual(self.cfg['base_height_randomization']['joint_name'], 'lifting_joint')
        self.assertEqual(self.cfg['base_height_randomization']['nominal_offset'], .08)
        self.assertEqual(self.cfg['base_height_randomization']['offset_jitter_range'], [-.02, .02])

    def test_sampling_is_bounded_reproducible_and_does_not_accumulate(self):
        rng = random.Random(42)
        samples = [height.sample_robot_base_height(self.cfg, rng) for _ in range(200)]
        self.assertEqual(samples[0], height.sample_robot_base_height(self.cfg, random.Random(42)))
        self.assertGreater(len({s['height_offset_m'] for s in samples}), 190)
        for sample in samples:
            self.assertTrue(.06 <= sample['height_offset_m'] <= .10)
            self.assertTrue(-.5262 <= sample['joint_position_m'] <= -.4862)
        self.assertEqual(self.cfg['initial_joint_positions']['lifting_joint'], -.5862)

    def test_replay_reuses_sample_and_disabling_keeps_eighty_mm_increase(self):
        initial = height.episode_initial_joint_positions(self.cfg)
        with patch.object(height.random, 'uniform', side_effect=AssertionError('Replay resampled')):
            self.assertEqual(height.episode_initial_joint_positions(self.cfg, use_mem=True), initial)
        self.cfg['base_height_randomization']['enabled'] = False
        fixed = height.episode_initial_joint_positions(self.cfg, use_mem=True)
        self.assertAlmostEqual(fixed['lifting_joint'], -.5062)

    def test_invalid_bounds_and_drive_gains_are_rejected(self):
        for change in ({'offset_jitter_range': [.02, -.02]}, {'offset_jitter_range': [float('nan'), .02]},
                       {'drive_damping': 0}, {'drive_stiffness': -1}):
            cfg = deepcopy(self.cfg)
            cfg['base_height_randomization'].update(change)
            with self.assertRaises(ValueError):
                height.sample_robot_base_height(cfg)

    def test_repeated_initialization_holds_sample_and_sets_only_platform_gains(self):
        initialize = method_from_file(ROOT / 'workflows/simbox/core/robots/split_aloha.py',
                                     'SplitAloha', '_set_initial_positions', {'np': np})
        initial = height.episode_initial_joint_positions(self.cfg)
        kp, kd = np.array([[1e5]]), np.array([[0.]])
        view = SimpleNamespace(set_joint_positions=Mock(), set_joint_position_targets=Mock(),
                               get_gains=Mock(return_value=(kp, kd)), set_gains=Mock())
        robot = SimpleNamespace(cfg=self.cfg, _episode_initial_joint_positions=initial,
            _episode_base_height_cfg=self.cfg['base_height_randomization'],
            _articulation_view=view, get_dof_index=lambda name: {'lifting_joint': 4,
                'mobile_translate_x': 0, 'mobile_translate_y': 1, 'mobile_rotate': 2}[name],
            left_joint_home=[], right_joint_home=[], left_gripper_home=[], right_gripper_home=[],
            left_joint_indices=[], right_joint_indices=[], left_gripper_indices=[], right_gripper_indices=[])
        for _ in range(3):
            initialize(robot)
        for call in view.set_joint_positions.call_args_list:
            self.assertEqual(call.args[0], list(initial.values()))
            self.assertEqual(call.kwargs['joint_indices'], [4, 0, 1, 2])
        self.assertEqual(view.set_joint_positions.call_args_list, view.set_joint_position_targets.call_args_list)
        self.assertEqual(view.get_gains.call_count, 1)
        self.assertEqual(view.set_gains.call_count, 3)
        self.assertEqual(view.set_gains.call_args.kwargs['joint_indices'], [4])
        np.testing.assert_array_equal(view.set_gains.call_args.kwargs['kds'], [[10000.]])
        robot._episode_base_height_cfg = None
        initialize(robot)
        np.testing.assert_array_equal(view.set_gains.call_args.kwargs['kps'], kp)
        np.testing.assert_array_equal(view.set_gains.call_args.kwargs['kds'], kd)

    def test_only_held_platform_limits_change_and_legacy_restores_them(self):
        stage = Usd.Stage.CreateInMemory()
        stage.DefinePrim('/Robot', 'Xform').GetReferences().AddReference(
            str(ROOT / 'workflows/simbox/assets' / self.cfg['path']))
        robot = SimpleNamespace(robot_prim_path='/Robot')
        joints = {p.GetName(): p for p in stage.Traverse() if p.IsA(UsdPhysics.Joint)}
        before = {str(p.GetPath()): (p.GetAttribute('physics:lowerLimit').Get(),
                                  p.GetAttribute('physics:upperLimit').Get())
                  for p in joints.values() if p.GetAttribute('physics:lowerLimit')}
        initial = height.episode_initial_joint_positions(self.cfg)
        height.set_robot_height_limits(stage, robot, self.cfg, initial)
        for name, position in initial.items():
            factor = 180/np.pi if joints[name].IsA(UsdPhysics.RevoluteJoint) else 1.
            self.assertAlmostEqual(joints[name].GetAttribute('physics:lowerLimit').Get(), (position-.0001)*factor, places=6)
        for name, joint in joints.items():
            path = str(joint.GetPath())
            if name not in initial and path in before:
                self.assertEqual((joint.GetAttribute('physics:lowerLimit').Get(),
                                  joint.GetAttribute('physics:upperLimit').Get()), before[path])
        height.set_robot_height_limits(stage, robot, {}, {})
        for joint in joints.values():
            path = str(joint.GetPath())
            if path in before:
                self.assertEqual((joint.GetAttribute('physics:lowerLimit').Get(),
                                  joint.GetAttribute('physics:upperLimit').Get()), before[path])

    def test_replay_and_legacy_override_live_robot_initial_height(self):
        prepare = method_from_file(ROOT / 'workflows/simbox/core/tasks/banana.py',
                                   'BananaBaseTask', '_set_robot_base_heights')
        original = height.episode_initial_joint_positions(self.cfg)
        task = SimpleNamespace(cfg={'robots': [self.cfg]}, robots={'split_aloha': SimpleNamespace()}, stage=None)
        with patch.dict('sys.modules', {'core.utils.robot_height_randomization': height}), patch.object(height, 'set_robot_height_limits'):
            prepare(task, use_mem=True)
            self.assertEqual(task.robots['split_aloha']._episode_initial_joint_positions, original)
            legacy = deepcopy(self.cfg)
            del legacy['base_height_randomization']
            legacy['initial_joint_positions']['lifting_joint'] = -.6
            task.cfg = {'robots': [legacy]}
            prepare(task, use_mem=True)
            self.assertEqual(task.robots['split_aloha']._episode_initial_joint_positions, {'lifting_joint': -.6})
            self.assertIsNone(task.robots['split_aloha']._episode_base_height_cfg)


if __name__ == '__main__':
    unittest.main()
