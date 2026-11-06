"""Shared-footprint bounds, separation, directional diversity and seed replay."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('shared_sampling', ROOT / 'workflows/simbox/core/utils/shared_region_sampling.py')
sampling = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sampling)


class SharedRegionTests(unittest.TestCase):
    def setUp(self):
        self.cfg = yaml.safe_load((ROOT / 'local_configs/cup_plate_desk/task.yaml').read_text())['tasks'][0]['shared_region_sampling']
        self.bounds = np.array(self.cfg['xy_bounds'])
        self.radii = np.array(list(self.cfg['object_radii'].values()))

    def test_active_region_full_footprints_separation_and_all_relative_directions(self):
        rng = np.random.RandomState(42)
        layouts = np.array([sampling.sample_separated_disks(self.bounds, self.radii, self.cfg['clearance'], rng=rng)[0]
                            for _ in range(3000)])
        self.assertTrue(np.all(layouts-self.radii[None, :, None] >= self.bounds[0]-1e-12))
        self.assertTrue(np.all(layouts+self.radii[None, :, None] <= self.bounds[1]+1e-12))
        relative = layouts[:, 0]-layouts[:, 1]
        self.assertTrue(np.all(np.linalg.norm(relative, axis=1) >= self.radii.sum()+self.cfg['clearance']))
        self.assertEqual(len(set(map(tuple, np.sign(relative)))), 4)
        for axis in (0, 1):
            self.assertAlmostEqual(float(np.mean(relative[:, axis] > 0)), .5, delta=.04)

    def test_seed_reproduction_and_zero_clearance(self):
        first = sampling.sample_separated_disks(self.bounds, self.radii, rng=np.random.RandomState(42))
        second = sampling.sample_separated_disks(self.bounds, self.radii, rng=np.random.RandomState(42))
        np.testing.assert_array_equal(first[0], second[0])
        self.assertEqual(first[1], second[1])
        positions, _ = sampling.sample_separated_disks(self.bounds, self.radii, clearance=0, rng=np.random.RandomState(42))
        self.assertGreaterEqual(np.linalg.norm(positions[0]-positions[1]), self.radii.sum())

    def test_invalid_and_impossible_regions_fail_explicitly(self):
        for bounds in ([[0, 0], [-1, 1]], [[0, 0], [.05, .05]], [[0, 0], [np.nan, 1]]):
            with self.assertRaises(ValueError):
                sampling.sample_separated_disks(bounds, self.radii)
        with self.assertRaisesRegex(ValueError, 'Cannot place'):
            sampling.sample_separated_disks([[0, 0], [.16, .16]], [.077, .077], max_attempts=5)
        for clearance in (-.1, np.inf):
            with self.assertRaises(ValueError):
                sampling.sample_separated_disks(self.bounds, self.radii, clearance=clearance)


if __name__ == '__main__':
    unittest.main()
