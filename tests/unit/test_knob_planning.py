import importlib.util
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

path = Path(__file__).resolve().parents[2] / "workflows/simbox/core/utils/knob_planning.py"
spec = importlib.util.spec_from_file_location("knob_planning", path)
planning = importlib.util.module_from_spec(spec)
spec.loader.exec_module(planning)

skill_path = path.parent.parent / "skills/turn_knob.py"
skill_tree = ast.parse(skill_path.read_text())
done_method = next(n for c in skill_tree.body if isinstance(c, ast.ClassDef)
                   for n in c.body if isinstance(n, ast.FunctionDef) and n.name == "is_done")
scope = {"np": np}
exec(compile(ast.fix_missing_locations(ast.Module(body=[done_method], type_ignores=[])), str(skill_path), "exec"), scope)


def candidate(height):
    pose = np.eye(4)
    pose[2, 3] = height
    return {"contact": pose, "roll_deg": height}


class KnobPlanningTests(unittest.TestCase):
    def test_does_not_advance_an_unaligned_knob_command(self):
        controller = SimpleNamespace(get_ee_pose=lambda: (np.array([.01, 0., 0.]), np.array([1., 0., 0., 0.])),
                                     num_last_cmd=0)
        skill = SimpleNamespace(controller=controller, skill_cfg={},
                                manip_list=[(np.zeros(3), np.array([1., 0., 0., 0.]))] * 2)
        self.assertFalse(scope["is_done"](skill))
        self.assertEqual(len(skill.manip_list), 2)
        controller.get_ee_pose = lambda: (np.zeros(3), Rotation.from_euler('x', 3., degrees=True).as_quat(scalar_first=True))
        self.assertFalse(scope["is_done"](skill))
        self.assertEqual(len(skill.manip_list), 2)

    def test_advances_a_precisely_aligned_knob_command(self):
        controller = SimpleNamespace(get_ee_pose=lambda: (np.array([.002, 0., 0.]),
                                      Rotation.from_euler('x', 1., degrees=True).as_quat(scalar_first=True)), num_last_cmd=0)
        skill = SimpleNamespace(controller=controller, skill_cfg={},
                                manip_list=[(np.zeros(3), np.array([1., 0., 0., 0.]))])
        self.assertTrue(scope["is_done"](skill))
        self.assertEqual(skill.manip_list, [])

    def test_tilted_turn_keeps_fingertip_at_the_spindle(self):
        pivot, axis = np.array([.3, -.2, .8]), np.array([0., 1., 0.])
        orientation = np.column_stack([axis, [-1., 0., 0.], [0., 0., 1.]])
        contact = planning.knob_contact_pose(pivot, orientation, .156, 30., -5.)
        for angle in np.linspace(0., np.pi / 2, 9):
            pose = planning.rotate_about_knob(contact, pivot, axis, angle)
            np.testing.assert_allclose(pose[:3, 3] + .156 * pose[:3, 0], pivot, atol=1e-12)
            np.testing.assert_allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-12)

    def test_selects_margin_over_first_feasible_path(self):
        def plan(pose, start):
            goal = np.array([.99 if pose[2, 3] == 0. else .25])
            return np.stack([start, goal])

        best, report = planning.select_knob_path(
            [candidate(0.), candidate(1.)], np.zeros(3), np.array([1., 0., 0.]),
            [.03], [.02], 0., 2, [.999], [-1.], [1.], plan)
        self.assertEqual(best["roll_deg"], 1.)
        self.assertAlmostEqual(best["keypose_min_margin_fraction"], .375)
        self.assertAlmostEqual(best["trajectory_min_margin_fraction"], .0005)
        self.assertEqual(report["connected_paths"], 2)

    def test_rejects_disconnected_intermediate_turn(self):
        def plan(pose, start):
            angle = Rotation.from_matrix(pose[:3, :3]).magnitude()
            if pose[2, 3] == 0. and np.isclose(angle, np.pi / 4):
                return None
            return np.array([start, [.1]])

        best, report = planning.select_knob_path(
            [candidate(0.), candidate(1.)], np.zeros(3), np.array([0., 0., 1.]),
            [.03], [.02], np.pi / 2, 2, [.0], [-1.], [1.], plan)
        self.assertEqual(best["roll_deg"], 1.)
        self.assertEqual(report["connected_paths"], 1)

    def test_search_expands_when_budgeted_prefix_fails(self):
        def plan(pose, start):
            return None if pose[2, 3] == 0. else np.array([start, [.1]])

        best, report = planning.select_knob_path(
            [candidate(0.), candidate(1.), candidate(2.)], np.zeros(3), np.array([1., 0., 0.]),
            [.03], [.02], 0., 2, [.0], [-1.], [1.], plan, max_motion_candidates=1)
        self.assertIsNotNone(best)
        self.assertEqual(report["motion_candidates_evaluated"], 2)

    def test_skips_distances_rejected_by_candidate_ik(self):
        checked = []

        def plan(pose, start):
            checked.append(pose[0, 3])
            return np.array([start, [.1]])

        grasp = {**candidate(0.), "approach_candidates_m": [.015], "retreat_candidates_m": [.01]}
        best, _ = planning.select_knob_path(
            [grasp], np.zeros(3), np.array([1., 0., 0.]),
            [.1], [.2], 0., 2, [.0], [-1.], [1.], plan)
        self.assertEqual(best["approach_m"], .015)
        self.assertEqual(best["retreat_m"], .01)
        self.assertNotIn(-.1, checked)
        self.assertNotIn(-.2, checked)

    def test_collision_geometry_matches_each_motion_phase(self):
        current_width = None

        def prepare(phase):
            nonlocal current_width
            current_width = {"approach": .088, "contact": .088, "turn": .035, "retreat": .041}[phase]

        def plan(pose, start):
            angle = Rotation.from_matrix(pose[:3, :3]).magnitude()
            if pose[2, 3] < 0.:
                expected = .041 if angle > 0. else .088
            else:
                expected = .035 if angle > 0. else .088
            self.assertEqual(current_width, expected)
            return np.array([start, [.1]])

        best, _ = planning.select_knob_path(
            [candidate(0.)], np.zeros(3), np.array([0., 0., 1.]),
            [.03], [.02], np.pi / 2, 2, [.0], [-1.], [1.], plan,
            prepare_phase=prepare)
        self.assertIsNotNone(best)


if __name__ == "__main__":
    unittest.main()
