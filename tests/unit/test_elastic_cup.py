"""Physical mesh invariants and reference-frame math, without starting Isaac."""
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("elastic_cup_math", ROOT/"workflows/simbox/core/utils/elastic_cup.py")
cup = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = cup
spec.loader.exec_module(cup)


class ElasticCupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mesh = cup.make_elastic_cup()

    def test_connected_watertight_material_volume_and_fem_tet_quality(self):
        d = self.mesh
        points, tets = d["points"], d["tets"]
        volumes = cup.signed_tet_volumes(points.astype(np.float32), tets)
        self.assertGreater(volumes.min()*6, 1e-9)
        edges = tets[:, [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]].reshape(-1, 2)
        graph = coo_matrix((np.ones(len(edges)), edges.T), shape=(len(points), len(points)))
        self.assertEqual(connected_components(graph, directed=False)[0], 1)
        faces = d["skin_faces"]
        skin_edges = np.sort(faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
        self.assertTrue(np.all(np.unique(skin_edges, axis=0, return_counts=True)[1] == 2))
        # Surface orientation and volume must agree with the actual material cells.
        triangles = points[d["skin_ids"]][faces]
        surface_volume = np.einsum("ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])).sum()/6
        self.assertAlmostEqual(surface_volume, volumes.sum(), places=11)

    def test_cavity_is_empty_and_existing_cup_dimensions_are_preserved(self):
        d = self.mesh
        q = d["points"][d["tets"]]
        # Cells above the bottom disk all lie close to the wall, not in the cavity.
        wall = q[q[:, :, 2].min(1) >= 0.0012-1e-10]
        self.assertGreater(np.linalg.norm(wall.mean(1)[:, :2], axis=1).min(), 0.024)
        np.testing.assert_allclose(d["points"].min(0), [-0.0375, -0.0375, 0], atol=1e-10)
        np.testing.assert_allclose(d["points"].max(0), [0.0375, 0.0375, 0.08], atol=1e-10)
        self.assertLess(d["material_volume_m3"], np.pi*0.0375**2*0.08*0.1)

    def test_pose_and_deformation_do_not_confuse_rigid_motion_with_squeezing(self):
        points = self.mesh["points"]
        rotation = Rotation.from_euler("xyz", [18, -23, 79], degrees=True)
        translation = np.array([0.3, -0.2, 0.75])
        moved = rotation.apply(points)+translation
        p, q = cup.fit_cup_frame(points, moved)
        np.testing.assert_allclose(p, translation, atol=1e-12)
        np.testing.assert_allclose(Rotation.from_quat(q, scalar_first=True).as_matrix(), rotation.as_matrix(), atol=1e-12)
        self.assertLess(cup.cup_deformation(points, moved)["max_displacement_m"], 1e-12)
        deformed = points.copy()
        deformed[:, 0] *= 0.95
        self.assertGreater(cup.cup_deformation(points, rotation.apply(deformed)+translation)["rms_displacement_m"], 0.0005)

    def test_bad_parameters_and_nonfinite_states_are_rejected(self):
        for kwargs in ({"wall_thickness": -1}, {"height": float("nan")}, {"sectors": True}, {"wall_segments": 1}):
            with self.assertRaises(ValueError):
                cup.make_elastic_cup(cup.ElasticCupSpec(**kwargs))
        invalid = self.mesh["points"].copy()
        invalid[0, 0] = np.nan
        with self.assertRaises(ValueError):
            cup.fit_cup_frame(self.mesh["points"], invalid)


if __name__ == "__main__":
    unittest.main()
