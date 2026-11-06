from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/eval'))
from placement_evaluation import DEFAULT_CRITERIA, evaluate_placement, tcp_from_ee_world


class PlacementTests(unittest.TestCase):
    def evaluate(self, xy=.019, tilt=0., speed=0., tcp=(0,0,.3), height_error=0.):
        theta = np.deg2rad(tilt)
        cup = np.eye(4)
        cup[:3,:3] = [[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]]
        cup[:3,3] = [xy,0,.005+.0265*np.sin(theta)+height_error]
        return evaluate_placement(cup,np.eye(4),([speed,0,0],[0,0,0]),
                                  ([0,0,0],[0,0,0]),tcp,DEFAULT_CRITERIA)

    def test_two_success_levels_have_different_position_checks(self):
        self.assertTrue(self.evaluate()['candidate'])
        self.assertTrue(self.evaluate()['strict_candidate'])
        relaxed = self.evaluate(xy=.06)
        self.assertTrue(relaxed['candidate'])
        self.assertFalse(relaxed['strict_candidate'])
        self.assertEqual(relaxed['strict_failed_checks'], ['footprint'])
        bad = self.evaluate(xy=.078)
        self.assertEqual(bad['failed_checks'], ['center_in_plate'])
        self.assertFalse(bad['strict_candidate'])

    def test_small_tilt_checks_lowest_edge_not_bottom_center_height(self):
        good = self.evaluate(tilt=7.)
        self.assertTrue(good['candidate'])
        self.assertGreater(good['metrics']['bottom_center_z_m'], .008)
        self.assertAlmostEqual(good['metrics']['lowest_bottom_z_m'], .005)
        self.assertIn('cup_tilt', self.evaluate(tilt=8.1)['strict_failed_checks'])
        self.assertIn('height', self.evaluate(height_error=.005)['failed_checks'])

    def test_standing_cup_at_eight_point_seven_degrees_is_ordinary_success(self):
        result = self.evaluate(xy=.02985,tilt=8.70)
        self.assertTrue(result['candidate'])
        self.assertEqual(result['failed_checks'],[])
        self.assertFalse(result['strict_candidate'])
        self.assertIn('cup_tilt',result['strict_failed_checks'])
        self.assertIn('footprint',result['strict_failed_checks'])
        self.assertIn('cup_tilt',self.evaluate(tilt=15.1)['failed_checks'])

    def test_held_or_moving_cup_does_not_pass(self):
        held = self.evaluate(tcp=(.019,0,.08))
        self.assertTrue(held['placed'])
        self.assertFalse(held['candidate'])
        self.assertIn('tcp_clearance',held['failed_checks'])
        self.assertIn('cup_linear_speed',self.evaluate(speed=.02)['failed_checks'])

    def test_nonfinite_pose_is_rejected_with_diagnostic(self):
        self.assertEqual(self.evaluate(xy=float('nan'))['failed_checks'],['nonfinite'])
        self.assertFalse(self.evaluate(xy=float('nan'))['strict_candidate'])

    def test_tcp_offset_uses_actual_world_rotation_and_translation(self):
        world = np.eye(4)
        world[:3, :3] = [[0,0,1],[0,1,0],[-1,0,0]]
        world[:3, 3] = [-.2, .1, .8]
        np.testing.assert_allclose(tcp_from_ee_world(world,'z',.12),[-.08,.1,.8])


if __name__ == '__main__':
    unittest.main()
