"""Geometry, IK ranking, stage ordering and cached execution regressions."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import ast
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("cup_chain", ROOT / "workflows/simbox/core/utils/cup_plate_planning.py")
chain = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chain)


class CupChainTests(unittest.TestCase):
    def test_budget_expires_across_stages_without_resetting_each_segment(self):
        now = [100.]
        budget = chain.PlanningBudget(45, clock=lambda: now[0])
        now[0] += 25
        budget.check('pregrasp')
        self.assertEqual(budget.remaining(), 20)
        now[0] += 21
        with self.assertRaisesRegex(chain.PlanningBudgetExceeded, 'place'):
            budget.check('place')
        for seconds in [0, -1, np.nan, np.inf]:
            with self.assertRaises(ValueError):
                chain.PlanningBudget(seconds)

    def test_placement_cache_reuses_branches_but_rescores_and_invalidates(self):
        planner = chain.CupPlateChainPlanner.__new__(chain.CupPlateChainPlanner)
        planner._place_endpoint_cache = {}; planner._attachment_generation = 0
        planner.stats = __import__('collections').Counter()
        planner.lower, planner.upper = np.full(6, -2.), np.full(6, 2.)
        planner.preferred_margin = np.deg2rad(5)
        planner.ctrl = SimpleNamespace(world_cfg=object())
        poses = np.repeat(np.eye(4)[None], 2, axis=0)
        solutions = np.stack([np.full((2, 6), -.4), np.full((2, 6), .4)], axis=1)
        planner.ik_solutions = Mock(return_value=(solutions, np.ones((2, 2), dtype=bool)))
        negative, _ = planner.placement_ik(poses, np.full(6, -.4))
        positive, _ = planner.placement_ik(poses, np.full(6, .4))
        np.testing.assert_allclose(negative, -.4); np.testing.assert_allclose(positive, .4)
        self.assertEqual(planner.ik_solutions.call_count, 1)
        self.assertEqual(planner.stats['endpoint_cache_hits'], 1)
        planner.ctrl.world_cfg = object()
        planner.placement_ik(poses, np.zeros(6))
        planner._attachment_generation += 1
        planner.placement_ik(poses, np.zeros(6))
        poses[0, 0, 3] += .01
        planner.placement_ik(poses, np.zeros(6))
        self.assertEqual(planner.ik_solutions.call_count, 4)

    def test_budget_failure_is_normal_pick_failure_and_releases_virtual_cup(self):
        planner = SimpleNamespace(stats={}, choose_chain=Mock(side_effect=chain.PlanningBudgetExceeded('test deadline')),
                                  clear_virtual=Mock())
        pick = SimpleNamespace(controller=SimpleNamespace(), manip_list=['old'])
        with patch.object(chain, 'CupPlateChainPlanner', return_value=planner):
            chain.prepare_pick(pick)
        self.assertEqual(pick.manip_list, [])
        self.assertIsNone(pick.controller._cup_plate_chain)
        planner.clear_virtual.assert_called_once()

    def test_colliding_actual_carry_start_stops_before_repeated_goal_search(self):
        planner = chain.CupPlateChainPlanner.__new__(chain.CupPlateChainPlanner)
        planner.stats = __import__('collections').Counter()
        planner.state = lambda q: q
        model = Mock()
        model.check_start_state.return_value = (False, 'INVALID_START_STATE_WORLD_COLLISION')
        planner.ctrl = SimpleNamespace(motion_gen=model)
        planner.ik = Mock()
        self.assertIsNone(planner.place_chain(np.zeros(6), np.eye(4), world_object=np.eye(4)))
        planner.ik.assert_not_called()
        model.rollout_fn.robot_self_collision_constraint.enable_cost.assert_called_once()

    def test_ik_world_only_refreshes_after_world_config_changes(self):
        planner = chain.CupPlateChainPlanner.__new__(chain.CupPlateChainPlanner)
        planner.ctrl = SimpleNamespace(world_cfg=object(), ik_solver=Mock())
        planner.sync_ik_world()
        planner.sync_ik_world()
        planner.ctrl.ik_solver.update_world.assert_called_once_with(planner.ctrl.world_cfg)
        planner.ctrl.world_cfg = object()
        planner.sync_ik_world()
        self.assertEqual(planner.ctrl.ik_solver.update_world.call_count, 2)

    def test_failed_fallback_restores_self_collision_checks(self):
        planner = chain.CupPlateChainPlanner.__new__(chain.CupPlateChainPlanner)
        planner.cfg = {}; planner.stats = __import__('collections').Counter()
        planner.guided_path = Mock(return_value=None)
        planner.state = lambda q: q
        model = Mock()
        model.plan_single_js.return_value = SimpleNamespace(success=np.array([False]), status='WORLD_COLLISION')
        planner.ctrl = SimpleNamespace(plan_config=Mock(), motion_gen=model)
        self.assertIsNone(planner.plan_segment(np.zeros(6), np.ones(6), 'preplace'))
        model.rollout_fn.robot_self_collision_constraint.enable_cost.assert_called_once()
        model.rollout_fn.robot_self_collision_constraint.enable_cost.reset_mock()
        model.plan_single_js.side_effect = RuntimeError('planner error')
        with self.assertRaisesRegex(RuntimeError, 'planner error'):
            planner.plan_segment(np.zeros(6), np.ones(6), 'preplace')
        model.rollout_fn.robot_self_collision_constraint.enable_cost.assert_called_once()

    def test_only_expected_mesh_cache_reuse_warning_is_filtered(self):
        import logging
        spec = importlib.util.spec_from_file_location('planning_logging', ROOT/'workflows/simbox/core/utils/planning_logging.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        filt = module.MeshCacheHitFilter()
        def record(msg):
            return logging.LogRecord('curobo', logging.WARNING, '', 0, msg, (), None)
        self.assertFalse(filt.filter(record('Object already in warp cache, using existing instance for: /table')))
        for msg in ['Mesh cache capacity exceeded', 'World collision', 'IK failed', 'Mesh has invalid geometry']:
            self.assertTrue(filt.filter(record(msg)))

    def test_pregrasp_rotation_pivots_at_tcp_and_final_entry_is_separate(self):
        grasp = np.eye(4)
        grasp[:3, :3] = Rotation.from_euler('x', 170, degrees=True).as_matrix()
        grasp[:3, 3] = [.2, .1, .3]
        poses, ids, heights = chain.flexible_pregrasp_variants(grasp[None], [.04, .07],
            np.array([0, 0, 1]), .12, [0, -10, 10])
        tcp = poses[:, :3, 3]+poses[:, :3, 2]*.12
        expected = grasp[:3, 3]+grasp[:3, 2]*.12
        np.testing.assert_allclose(tcp[:, :2], np.tile(expected[:2], (6, 1)))
        np.testing.assert_allclose(tcp[:, 2], expected[2]+heights)
        entry, _, _ = chain.pregrasp_variants(grasp[None], [.012], up=[0, 0, 1])
        np.testing.assert_allclose(entry[0, :3, :3], grasp[:3, :3])
        self.assertTrue(chain.vertical_entry_clear(np.stack([entry[0], grasp]), [0, 0, 1], .12))

    def test_coupled_lift_moves_object_base_and_bounds_tilt(self):
        obj = np.eye(4); obj[:3, 3] = [.3, .1, .75]
        grasp = obj.copy(); grasp[:3, 3] += [.03, 0, .19]
        relative = np.linalg.inv(obj) @ grasp
        poses = chain.lift_pose_variants(grasp, obj, np.array([0, 0, 1]), .04, [0, .02], [0, -10, 10])
        objects = poses @ np.linalg.inv(relative)
        np.testing.assert_allclose(objects[:, 2, 3], .79)
        np.testing.assert_allclose(np.rad2deg(np.arccos(objects[:, 2, 2])), [0, 10, 10, 0, 10, 10], atol=1e-6)
        self.assertLess(np.linalg.norm(objects[-1, :2, 3]), np.linalg.norm(obj[:2, 3]))

    def test_tilted_place_compensates_lowest_point_and_rejects_excessive_tilt(self):
        cfg = dict(chain_place_tilts_deg=[0, 3, 4], max_tilt_deg=5, support_surface_z=.005,
                   place_z_offset=.003, pre_place_z_offset=.04, post_place_vector=[0, 0, .04])
        targets, _ = chain.upright_place_targets(np.eye(4), np.eye(4), np.eye(4), np.eye(4), cfg)
        obj = targets[:, 1]
        sin_tilt = np.linalg.norm(obj[:, :2, 2], axis=1)
        np.testing.assert_allclose(obj[:, 2, 3]-.0265*sin_tilt, .008)
        self.assertLessEqual(np.rad2deg(np.arccos(obj[:, 2, 2])).max(), 4.000001)
        cfg['chain_place_tilts_deg'] = [6]
        with self.assertRaises(ValueError):
            chain.upright_place_targets(np.eye(4), np.eye(4), np.eye(4), np.eye(4), cfg)
        cfg['max_tilt_deg'] = 8
        cfg['chain_place_tilts_deg'] = [0, 6, 7]
        poses, _ = chain.upright_place_targets(np.eye(4), np.eye(4), np.eye(4), np.eye(4), cfg)
        self.assertAlmostEqual(np.rad2deg(np.arccos(poses[:, 1, 2, 2])).max(), 7)
        cfg['chain_place_tilts_deg'] = [9]
        with self.assertRaises(ValueError):
            chain.upright_place_targets(np.eye(4), np.eye(4), np.eye(4), np.eye(4), cfg)

    def test_flexible_place_only_locks_attitude_for_last_short_descent(self):
        cfg = dict(chain_flexible_placement=True, chain_final_place_distance=.012,
                   support_surface_z=.005, place_z_offset=.003,
                   pre_place_z_offset=.04, post_place_vector=[0, 0, .04])
        poses, _ = chain.upright_place_targets(np.eye(4), np.eye(4), np.eye(4), np.eye(4), cfg)
        np.testing.assert_allclose(poses[:, 0, 2, 3]-poses[:, 1, 2, 3], .012)
        np.testing.assert_allclose(poses[:, 2, 2, 3]-poses[:, 1, 2, 3], .04)

    def test_carry_path_cannot_exceed_cup_tilt_bound(self):
        planner = chain.CupPlateChainPlanner.__new__(chain.CupPlateChainPlanner)
        planner.cfg = {'chain_carry_max_tilt_deg': 10}
        planner.stats = __import__('collections').Counter()
        planner.lower, planner.upper, planner.minimum_margin = np.full(6, -2), np.full(6, 2), .01
        planner.up = np.array([0, 0, 1]); planner.carry_object_from_ee = np.eye(4)
        planner.JointState = SimpleNamespace(from_position=lambda q, **kw: q)
        planner.ctrl = SimpleNamespace(tensor_args=SimpleNamespace(to_device=np.asarray),
            motion_gen=SimpleNamespace(joint_names=list(range(6)), check_constraints=lambda q: SimpleNamespace(feasible=np.ones(len(q), dtype=bool))))
        poses = np.repeat(np.eye(4)[None], 3, axis=0)
        poses[1, :3, :3] = Rotation.from_euler('x', 12, degrees=True).as_matrix()
        planner.fk_path = lambda q: poses
        self.assertFalse(planner.valid_path(np.zeros((3, 6)), 'preplace'))
        poses[1, :3, :3] = Rotation.from_euler('x', 5, degrees=True).as_matrix()
        self.assertTrue(planner.valid_path(np.zeros((3, 6)), 'preplace'))
        # A relaxed transport bound must never leak into the final descent.
        planner.cfg['chain_carry_max_tilt_deg'] = 60
        planner.cfg['chain_lift_max_tilt_deg'] = 10
        planner.place_cfg = {'max_tilt_deg': 5}
        poses[1, :3, :3] = Rotation.from_euler('x', 30, degrees=True).as_matrix()
        for label in ['preplace', 'preplace_refresh']:
            self.assertTrue(planner.valid_path(np.zeros((3, 6)), label))
        for label in ['lift', 'lift_refresh', 'place', 'place_refresh']:
            self.assertFalse(planner.valid_path(np.zeros((3, 6)), label))
        poses[1, :3, :3] = Rotation.from_euler('x', 61, degrees=True).as_matrix()
        self.assertFalse(planner.valid_path(np.zeros((3, 6)), 'preplace'))
        poses[1, :3, :3] = Rotation.from_euler('x', 6, degrees=True).as_matrix()
        self.assertTrue(planner.valid_path(np.zeros((3, 6)), 'preplace'))
        self.assertFalse(planner.valid_path(np.zeros((3, 6)), 'place'))
        poses[1, :3, :3] = Rotation.from_euler('x', 4, degrees=True).as_matrix()
        self.assertTrue(planner.valid_path(np.zeros((3, 6)), 'place'))
        planner.place_cfg['max_tilt_deg'] = 8
        poses[1, :3, :3] = Rotation.from_euler('x', 7, degrees=True).as_matrix()
        self.assertTrue(planner.valid_path(np.zeros((3, 6)), 'place'))
        poses[1, :3, :3] = Rotation.from_euler('x', 9, degrees=True).as_matrix()
        self.assertFalse(planner.valid_path(np.zeros((3, 6)), 'place'))

    def test_home_return_reloads_cup_and_checks_path_before_execution(self):
        planner = SimpleNamespace(cfg={'chain_max_joint_step': .025}, up=np.array([0, 0, 1]),
            current_joints=lambda: np.full(6, .4), begin_budget=Mock(), check_budget=Mock(),
            valid_path=Mock(return_value=True), plan_segment=Mock(), fk=lambda q: np.eye(4),
            fk_path=lambda q: np.repeat(np.eye(4)[None], len(q), axis=0), open_gripper_state=.24)
        controller = SimpleNamespace(_cup_plate_chain={'planner': planner, 'initial_joints': np.zeros(6)},
            _cup_plate_place_succeeded=True, ignore_substring=['material'], reference_prim_path='/Base',
            update_specific=Mock())
        skill = SimpleNamespace(controller=controller, skill_cfg={})
        chain.prepare_return_home(skill)
        self.assertFalse(skill.failed)
        controller.update_specific.assert_called_once_with(['material'], '/Base')
        segment = skill.manip_list[0][3]['segment']
        np.testing.assert_allclose(segment['path'][0], .4)
        np.testing.assert_allclose(segment['path'][-1], 0)
        self.assertTrue(np.all((segment['path'] >= 0) & (segment['path'] <= .4)))
        planner.plan_segment.assert_not_called()
        planner.valid_path.return_value = False
        planner.plan_segment.return_value = None
        chain.prepare_return_home(skill)
        self.assertTrue(skill.failed); self.assertEqual(skill.manip_list, [])
        controller._cup_plate_place_succeeded = False
        planner.plan_segment.reset_mock()
        chain.prepare_return_home(skill)
        planner.plan_segment.assert_not_called()

    def test_random_grasp_order_retains_only_feasible_candidates_and_ranked_fallbacks(self):
        scores = np.array([4., 1., np.inf, 3., 2., np.nan])
        np.testing.assert_array_equal(chain.grasp_candidate_order(scores), [1, 4, 3, 0])
        order = chain.grasp_candidate_order(scores, 'random_top_k', 3, np.random.default_rng(42))
        self.assertEqual(set(order[:3]), {1, 4, 3})
        np.testing.assert_array_equal(order[3:], [0])
        again = chain.grasp_candidate_order(scores, 'random_top_k', 3, np.random.default_rng(42))
        np.testing.assert_array_equal(order, again)
        first = {int(chain.grasp_candidate_order(scores, 'random_top_k', 3, np.random.default_rng(seed))[0])
                 for seed in range(20)}
        self.assertEqual(first, {1, 4, 3})

    def test_random_grasp_order_rejects_bad_mode_and_handles_no_ik(self):
        self.assertEqual(len(chain.grasp_candidate_order([np.inf], 'random_top_k', 8)), 0)
        for mode, count in [('unknown', 8), ('random_top_k', 0)]:
            with self.assertRaises(ValueError):
                chain.grasp_candidate_order([1.], mode, count)

    def test_short_curve_preserves_endpoints_and_penalizes_up_down_loop(self):
        start, goal = np.eye(4), np.eye(4)
        goal[:3, 3] = [.2, .0, .0]
        poses = chain.curve_poses(start, goal, np.array([0, 0, 1]), bow=.01)
        np.testing.assert_allclose(poses[[0, -1]], [start, goal])
        self.assertAlmostEqual(poses[:, 2, 3].max(), .01)
        natural = chain.trajectory_quality(poses, np.array([0, 0, 1]))
        poses[12, 2, 3] = .15
        self.assertGreater(chain.trajectory_quality(poses, np.array([0, 0, 1]))["cost"], natural["cost"])

    def test_joint_resampling_bounds_steps_and_does_not_overshoot(self):
        knots = np.array([[0., 0.], [.1, -.03], [.2, -.05]])
        path = chain.resample_joints(knots, max_step=.01)
        np.testing.assert_allclose(path[[0, -1]], knots[[0, -1]])
        self.assertLessEqual(np.abs(np.diff(path, axis=0)).max(), .01001)
        self.assertTrue(np.all(path >= knots.min(axis=0) - 1e-9))
        self.assertTrue(np.all(path <= knots.max(axis=0) + 1e-9))

    def test_joint_resampling_handles_localized_motion_that_exhausted_four_refinements(self):
        # The previous length-based sampler raises "Could not bound joint
        # increments" on this valid path: one moving interval among 24.
        knots = np.zeros((25, 6))
        knots[14:, 0] = .1
        path = chain.resample_joints(knots, max_step=.025)
        np.testing.assert_allclose(path[[0, -1]], knots[[0, -1]])
        self.assertLessEqual(np.abs(np.diff(path, axis=0)).max(), .025000001)
        self.assertTrue(np.all(path >= -1e-12))
        self.assertTrue(np.all(path[:, 0] <= .1 + 1e-12))

    def test_joint_resampling_stress_with_sparse_and_reversing_ik_knots(self):
        rng = np.random.default_rng(42)
        for _ in range(150):
            increments = rng.uniform(-.2, .2, (24, 6))
            increments[rng.uniform(size=24) < .8] = 0
            knots = np.vstack([np.zeros(6), np.cumsum(increments, axis=0)])
            path = chain.resample_joints(knots, max_step=.025)
            np.testing.assert_allclose(path[[0, -1]], knots[[0, -1]])
            self.assertLessEqual(np.abs(np.diff(path, axis=0)).max(), .025000001)
            self.assertTrue(np.all(path >= knots.min(axis=0) - 1e-10))
            self.assertTrue(np.all(path <= knots.max(axis=0) + 1e-10))

    def test_joint_resampling_rejects_nonfinite_or_excessive_paths_without_unbounded_allocation(self):
        with self.assertRaises(chain.TrajectoryResamplingError):
            chain.resample_joints([[0.], [np.nan]])
        with self.assertRaises(chain.TrajectoryResamplingError):
            chain.resample_joints([[0.], [1.]], max_step=1e-12)
        for step in [0., -1., np.nan]:
            with self.assertRaises(ValueError):
                chain.resample_joints([[0.], [1.]], max_step=step)

    def test_bad_resampling_rejects_candidate_while_unexpected_errors_still_propagate(self):
        from collections import Counter
        planner = chain.CupPlateChainPlanner.__new__(chain.CupPlateChainPlanner)
        planner.cfg = {'chain_max_joint_step': .025}
        planner.stats = Counter()
        planner.up = np.array([0, 0, 1])
        planner.lower, planner.upper = np.full(6, -2.), np.full(6, 2.)
        planner.preferred_margin = np.deg2rad(5.)
        planner.fk = lambda q: np.eye(4)
        planner.valid_path = Mock(return_value=True)
        planner.ik_solutions = lambda poses, references: (references[:, None, :],
                                                         np.ones((len(poses), 1), dtype=bool))
        args = (np.zeros(6), np.full(6, .1), 'lift', 0.)
        with patch.object(chain, 'resample_joints', side_effect=chain.TrajectoryResamplingError('test bound failure')):
            self.assertIsNone(planner.guided_path(*args))
        self.assertEqual(planner.stats['lift:resample_rejected'], 1)
        planner.valid_path.assert_not_called()
        with patch.object(chain, 'resample_joints', side_effect=RuntimeError('unexpected runtime failure')):
            with self.assertRaisesRegex(RuntimeError, 'unexpected runtime failure'):
                planner.guided_path(*args)

    def test_palm_guard_rejects_intrusion_but_allows_finger_region(self):
        points = chain.cylinder_points(.0375, .08)
        bounds = [[-.0829, -.0334, -.0001], [.0828, .0921, .0733]]
        ee = np.eye(4)
        cup = np.eye(4)
        cup[2, 3] = .06
        self.assertFalse(chain.palm_clear(ee[None], cup, points, bounds))
        cup[2, 3] = .09
        self.assertTrue(chain.palm_clear(ee[None], cup, points, bounds))

    def test_grasp_waits_for_measured_joint_motion_to_settle(self):
        positions = np.array([.2] * 6)
        controller = SimpleNamespace(arm_indices=list(range(6)),
            world=SimpleNamespace(get_physics_dt=lambda: 1 / 30.),
            robot=SimpleNamespace(get_joints_state=lambda: SimpleNamespace(positions=positions)))
        segment = dict(label="grasp", complete=True,
                       planner=SimpleNamespace(cfg={"chain_grasp_settle_steps": 3}))
        self.assertFalse(chain.segment_finished(controller, segment, True))
        positions[:] = 0
        self.assertFalse(chain.segment_finished(controller, segment, True))
        self.assertFalse(chain.segment_finished(controller, segment, True))
        self.assertFalse(chain.segment_finished(controller, segment, True))
        self.assertTrue(chain.segment_finished(controller, segment, True))

    def test_preclose_cup_motion_rejects_episode(self):
        planner = SimpleNamespace(cfg={}, current_joints=lambda: np.zeros(6), world_object=np.eye(4),
            pick=SimpleNamespace(pick_obj=SimpleNamespace(get_local_pose=lambda: ([.01, 0, 0], None))))
        controller = SimpleNamespace(dummy_forward=lambda target, grip: target)
        segment = dict(planner=planner, label="grasp")
        chain.follow_validated_segment(controller, segment, 1.)
        self.assertEqual(controller.num_plan_failed, 1000)

    def test_curve_collision_checks_use_joint_batch_without_time_axis(self):
        planner = chain.CupPlateChainPlanner.__new__(chain.CupPlateChainPlanner)
        planner.lower, planner.upper, planner.minimum_margin = np.full(6, -2.), np.full(6, 2.), .01
        planner.JointState = SimpleNamespace(from_position=lambda q, **kw: SimpleNamespace(position=q))
        def check(state):
            self.assertEqual(state.position.shape, (7, 6))
            return SimpleNamespace(feasible=np.array([True] * 7))
        planner.ctrl = SimpleNamespace(tensor_args=SimpleNamespace(to_device=np.asarray),
            motion_gen=SimpleNamespace(joint_names=list(range(6)), check_constraints=check))
        self.assertTrue(planner.valid_path(np.zeros((7, 6)), "lift"))

    def test_pregrasp_is_above_and_outside_for_every_allowed_tilt(self):
        poses = np.repeat(np.eye(4)[None], 4, axis=0)
        alpha = np.deg2rad([1, 15, 25, 30])
        poses[:, :3, :3] = Rotation.from_euler("y", 90 + np.array([1, 15, 25, 30]), degrees=True).as_matrix()
        poses[:, 2, 3] = .05
        pre, indices, distances = chain.pregrasp_variants(poses, [.03, .045, .06])
        self.assertTrue(chain.downward_mask(poses, np.array([0, 0, 1])).all())
        self.assertTrue(np.all(pre[:, 2, 3] > .05))
        np.testing.assert_allclose(pre[:, 2, 3] - .05, distances * np.sin(alpha[indices]))
        np.testing.assert_allclose(np.linalg.norm(pre[:, :3, 3] - poses[indices, :3, 3], axis=1), distances)
        poses[0, :3, :3] = Rotation.from_euler("y", 80, degrees=True).as_matrix()
        self.assertFalse(chain.downward_mask(poses[:1], np.array([0, 0, 1]))[0])

    def test_rim_entry_rejects_lateral_swing_and_height_reversal(self):
        poses = np.repeat(np.eye(4)[None], 15, axis=0)
        poses[:, :3, :3] = Rotation.from_euler("x", 180, degrees=True).as_matrix()
        poses[:, 2, 3] = np.linspace(.23, .18, len(poses))
        self.assertTrue(chain.vertical_entry_clear(poses, [0, 0, 1], .12))
        poses[7, 0, 3] = .008
        self.assertFalse(chain.vertical_entry_clear(poses, [0, 0, 1], .12))
        poses[7, 0, 3] = 0
        poses[7, 2, 3] = .24
        self.assertFalse(chain.vertical_entry_clear(poses, [0, 0, 1], .12))

    def test_rim_pregrasp_moves_vertically_even_when_the_gripper_is_inclined(self):
        grasp = np.eye(4)[None]
        grasp[0, :3, :3] = Rotation.from_euler("y", 160, degrees=True).as_matrix()
        grasp[0, :3, 3] = [.3, .1, .9]
        pre, ids, distances = chain.pregrasp_variants(grasp, [.04, .055], up=[0, 0, 1])
        np.testing.assert_allclose(pre[:, :2, 3], np.tile(grasp[0, :2, 3], (2, 1)))
        np.testing.assert_allclose(pre[:, 2, 3]-grasp[0, 2, 3], distances)

    def test_wrist_margin_wins_over_shorter_joint_motion(self):
        lower, upper = np.full(6, -2.), np.full(6, 2.)
        q = np.zeros((2, 6))
        q[:, 4] = [1.999, 1.8]
        scores = chain.branch_scores(q, q[0], lower, upper, np.deg2rad(5))
        self.assertLess(scores[1], scores[0])

    def test_upright_yaw_search_preserves_tilted_grasp_relation(self):
        grasp = np.eye(4)
        grasp[:3, :3] = Rotation.from_euler("xyz", [20, -30, 40], degrees=True).as_matrix()
        grasp[:3, 3] = [.04, .02, -.03]
        plate = np.eye(4)
        plate[:3, 3] = [.4, .1, .75]
        cfg = dict(support_surface_z=.005, place_z_offset=.003, pre_place_z_offset=.09,
                   post_place_vector=[0, 0, .12], chain_place_yaw_step_deg=15)
        poses, yaws = chain.upright_place_targets(np.eye(4), plate, grasp, np.eye(4), cfg)
        self.assertEqual(len(yaws), 24)
        self.assertEqual(yaws[0], 0)
        self.assertIn(-180, yaws)
        for pre, place, retreat in poses:
            obj = place @ np.linalg.inv(grasp)
            np.testing.assert_allclose(obj[:3, 2], [0, 0, 1], atol=1e-7)
            np.testing.assert_allclose(obj[:3, 3], [.4, .1, .758], atol=1e-7)
            np.testing.assert_allclose(pre[:3, 3] - place[:3, 3], [0, 0, .09], atol=1e-7)
            np.testing.assert_allclose(retreat[:3, 3] - place[:3, 3], [0, 0, .12], atol=1e-7)

    def test_cached_segment_cannot_finish_before_all_waypoints(self):
        segment = {"complete": False}
        self.assertFalse(chain.segment_finished(None, segment, True))
        segment["complete"] = True
        self.assertFalse(chain.segment_finished(None, segment, False))
        self.assertTrue(chain.segment_finished(None, segment, True))

    def test_replay_holds_terminal_target_and_updates_pose_cache(self):
        planner = SimpleNamespace(cfg={}, current_joints=lambda: np.zeros(6), plan_segment=Mock())
        segment = dict(planner=planner, path=np.array([np.zeros(6), np.full(6, .01)]),
                       start=np.zeros(6), goal=np.eye(4), label="grasp", complete=False)
        controller = SimpleNamespace(tensor_args=SimpleNamespace(to_device=np.asarray),
                                     dummy_forward=lambda target, gripper: (target.copy(), gripper))
        first = chain.follow_validated_segment(controller, segment, -1.)
        self.assertFalse(segment["complete"])
        second = chain.follow_validated_segment(controller, segment, -1.)
        third = chain.follow_validated_segment(controller, segment, -1.)
        np.testing.assert_array_equal(first[0], np.zeros(6))
        np.testing.assert_array_equal(second[0], third[0])
        self.assertTrue(segment["complete"])
        self.assertEqual(controller.num_last_cmd, 2)
        planner.plan_segment.assert_not_called()
        np.testing.assert_array_equal(controller._ee_ori, [1, 0, 0, 0])

    def test_changed_start_is_replanned_before_replay(self):
        actual = np.full(6, .2)
        planner = SimpleNamespace(cfg={}, current_joints=lambda: actual,
                                 plan_segment=Mock(return_value=None))
        segment = dict(planner=planner, path=np.zeros((2, 6)), start=np.zeros(6), goal=np.eye(4), label="lift")
        controller = SimpleNamespace(dummy_forward=lambda target, gripper: target)
        chain.follow_validated_segment(controller, segment, -1.)
        np.testing.assert_array_equal(planner.plan_segment.call_args.args[0], actual)
        self.assertEqual(controller.num_plan_failed, 1000)

    def test_invalid_pregrasp_offsets_rejected(self):
        for offsets in [[], [0], [-.03]]:
            with self.assertRaises(ValueError):
                chain.pregrasp_variants(np.eye(4)[None], offsets)

    def place_planner(self, fail_retreat=False):
        planner = chain.CupPlateChainPlanner.__new__(chain.CupPlateChainPlanner)
        planner.cfg = {"chain_place_candidates": 1}
        planner.place_cfg = dict(support_surface_z=.005, place_z_offset=.003,
                                pre_place_z_offset=.09, post_place_vector=[0, 0, .12])
        planner.base_from_world = planner.world_plate = planner.world_object = np.eye(4)
        q = np.repeat(np.array([np.full(6, 1.), np.full(6, 2.), np.full(6, 3.)])[None], 24, axis=0)
        planner.ik = lambda poses, reference: (q.reshape(-1, 6), np.zeros(72))
        spheres = Mock()
        saved_spheres = object()
        spheres.clone.return_value = saved_spheres
        model = Mock()
        model.kinematics.kinematics_config.get_link_spheres.return_value = spheres
        planner.ctrl = SimpleNamespace(motion_gen=model)
        calls = []
        def plan(start, goal, label):
            calls.append((start.copy(), goal.copy(), label))
            if fail_retreat and label == "retreat":
                raise RuntimeError("test retreat error")
            return {"path": np.array([start, goal + .001])}
        planner.plan_segment = plan
        return planner, calls, model, saved_spheres

    def test_place_segments_start_at_previous_actual_plan_endpoint(self):
        planner, calls, model, spheres = self.place_planner()
        result = planner.place_chain(np.zeros(6), np.eye(4))
        self.assertIsNotNone(result)
        self.assertEqual([c[2] for c in calls], ["preplace", "place", "retreat"])
        np.testing.assert_allclose(calls[1][0], np.full(6, 1.001))
        np.testing.assert_allclose(calls[2][0], np.full(6, 2.001))
        model.attach_spheres_to_robot.assert_called_once_with(sphere_tensor=spheres)

    def test_empty_wrist_retreat_can_rescue_invalid_fixed_attitude_retreat(self):
        planner, calls, model, spheres = self.place_planner()
        planner.place_cfg.update(chain_flexible_placement=True,
            chain_retreat_rotations_deg=[0, -10, 10])
        planner.ctrl.robot = SimpleNamespace(tcp_offset=.12)
        def ik(poses, reference):
            if np.asarray(reference).ndim == 1:
                qs = np.repeat(np.array([np.full(6, 1.), np.full(6, 2.)])[None], 24, axis=0)
                scores = np.zeros((24, 2))
                return qs.reshape(-1, 6), scores.reshape(-1)
            scores = np.full((24, 3), np.inf); scores[:, 1] = 0
            return np.full((len(poses), 6), 4.), scores.reshape(-1)
        planner.ik = ik
        result = planner.place_chain(np.zeros(6), np.eye(4))
        self.assertIsNotNone(result)
        np.testing.assert_array_equal(calls[-1][1], np.full(6, 4.))
        model.detach_object_from_robot.assert_called_once()
        model.attach_spheres_to_robot.assert_called_once_with(sphere_tensor=spheres)

    def test_infeasible_held_cup_endpoints_skip_all_retreat_queries(self):
        planner, calls, model, _ = self.place_planner()
        planner.place_cfg['chain_flexible_placement'] = True
        planner.ik = Mock(return_value=(np.zeros((48, 6)), np.full(48, np.inf)))
        self.assertIsNone(planner.place_chain(np.zeros(6), np.eye(4)))
        planner.ik.assert_called_once()
        self.assertEqual(calls, [])
        model.detach_object_from_robot.assert_not_called()

    def test_carry_geometry_restored_when_retreat_validation_raises(self):
        planner, calls, model, spheres = self.place_planner(fail_retreat=True)
        with self.assertRaisesRegex(RuntimeError, "retreat"):
            planner.place_chain(np.zeros(6), np.eye(4))
        model.attach_spheres_to_robot.assert_called_once_with(sphere_tensor=spheres)

    def test_attachment_exclusion_adds_only_explicit_contact_links(self):
        path = ROOT / "workflows/simbox/core/controllers/template_controller.py"
        tree = ast.parse(path.read_text())
        method = next(n for c in tree.body if isinstance(c, ast.ClassDef)
                      for n in c.body if isinstance(n, ast.FunctionDef) and n.name == "_load_robot")
        cfg = {"kinematics": {"self_collision_ignore": {"link7": ["link8", "attached_object"]}}}
        scope = {"load_yaml": lambda unused: {"robot_cfg": cfg}}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(path), "exec"), scope)
        controller = SimpleNamespace(robot=SimpleNamespace(cfg={"attached_object_contact_links": ["link7", "link8"]}))
        scope["_load_robot"](controller, "unused")
        self.assertEqual(cfg["kinematics"]["self_collision_ignore"],
                         {"link7": ["link8", "attached_object"], "link8": ["attached_object"]})


if __name__ == "__main__":
    unittest.main()
