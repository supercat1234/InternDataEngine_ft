"""Table geometry, supported object placement, replay and collision cache checks."""
import ast
from copy import deepcopy
import importlib.util
from pathlib import Path
import random
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from omegaconf import OmegaConf
from pxr import Gf, Usd, UsdGeom
from scipy.spatial.transform import Rotation
import yaml

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "fixture_height", ROOT / "workflows/simbox/core/utils/fixture_randomization.py")
height = importlib.util.module_from_spec(spec)
spec.loader.exec_module(height)


def method_from_file(path, class_name, method_name, namespace=None):
    cls = next(n for n in ast.parse(path.read_text()).body
               if isinstance(n, ast.ClassDef) and n.name == class_name)
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name)
    scope = namespace or {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(path), "exec"), scope)
    return scope[method_name]


def bbox(prim):
    return UsdGeom.Imageable(prim).ComputeWorldBound(
        Usd.TimeCode.Default(), UsdGeom.Tokens.default_).ComputeAlignedBox()


class FixtureHeightTests(unittest.TestCase):
    def setUp(self):
        self.cfg = yaml.safe_load((ROOT / "local_configs/cup_plate_desk/arena.yaml").read_text())["fixtures"][0]
        self.cfg["height_randomization"]["enabled"] = True  # Exercise the opt-in utility; task keeps table fixed.

    def test_bounds_seed_and_no_accumulation(self):
        rng = random.Random(42)
        states = [height.sample_fixture_height(self.cfg, rng) for _ in range(200)]
        self.assertEqual(states[0], height.sample_fixture_height(self.cfg, random.Random(42)))
        self.assertGreater(len({s["height_m"] for s in states}), 190)
        for state in states:
            self.assertTrue(.73 <= state["height_m"] <= .77)
            np.testing.assert_allclose(state["scale"][:2], [.001, .001])
            self.assertAlmostEqual(state["scale"][2] * 750, state["height_m"])
        np.testing.assert_array_equal(self.cfg["scale"], [.001] * 3)

    def test_invalid_height_bounds_fail(self):
        for changes in ({"nominal_height": 0}, {"height_range": [.77, .73]},
                        {"height_range": [0, .77]}, {"height_range": [.73, float("inf")]}):
            cfg = deepcopy(self.cfg)
            cfg["height_randomization"].update(changes)
            with self.assertRaises(ValueError):
                height.sample_fixture_height(cfg)

    def test_real_usd_floor_contact_support_placement_and_replay(self):
        stage = Usd.Stage.CreateInMemory()
        prim = stage.DefinePrim("/Table", "Xform")
        prim.GetReferences().AddReference(str(ROOT / "workflows/simbox/assets" / self.cfg["path"]))
        scale_op = UsdGeom.Xformable(prim).AddScaleOp()
        fixture = SimpleNamespace(set_local_scale=lambda scale: scale_op.Set(Gf.Vec3f(*scale)))
        points_before = prim.GetChild("Desktop").GetAttribute("points").Get()
        sample = method_from_file(ROOT / "workflows/simbox/core/utils/region_sampler.py",
                                  "RandomRegionSampler", "A_on_B_region_sampler",
                                  {"np": np, "R": Rotation, "compute_bbox": bbox})
        assets = ROOT / "workflows/simbox/assets/custom/objects"
        objects = []
        for name, asset in (("Cup", "paper_cup_001"), ("Plate", "ceramic_plate_001")):
            body = stage.DefinePrim("/" + name, "Xform")
            body.GetReferences().AddReference(str(assets / asset / "Aligned_obj.usd"))
            UsdGeom.Xformable(body).AddScaleOp().Set(Gf.Vec3f(.001))
            objects.append(SimpleNamespace(prim=body, get_local_pose=lambda: (np.zeros(3), [1, 0, 0, 0])))
        # Use OmegaConf as production does; cached lists must remain serializable/replayable.
        cfg = OmegaConf.create(self.cfg)
        for target in (.73, .77, .73):
            cfg.height_randomization.height_range = [target, target]
            state = height.apply_fixture_height(fixture, cfg)
            bounds = bbox(prim)
            np.testing.assert_allclose(bounds.min, [-.7, -.3, 0], atol=1e-6)
            np.testing.assert_allclose(bounds.max, [.7, .3, target], atol=1e-6)
            for obj in objects:
                pos, _ = sample(obj, SimpleNamespace(prim=prim), [[0, 0, 0], [0, 0, 0]], [0, 0])
                # Their lowest vertex starts exactly 1 mm above the resized support.
                self.assertAlmostEqual(pos[2] + bbox(obj.prim).min[2], target + .001, places=6)
            scale_op.Set(Gf.Vec3f(.001))
            with patch.object(height.random, "uniform", side_effect=AssertionError("Replay resampled height")):
                restored = height.apply_fixture_height(fixture, cfg, use_mem=True)
            self.assertEqual(dict(restored), state)
            self.assertAlmostEqual(bbox(prim).max[2], target, places=6)
        self.assertEqual(prim.GetChild("Desktop").GetAttribute("points").Get(), points_before)
        cfg.height_randomization.enabled = False
        height.apply_fixture_height(fixture, cfg, use_mem=True)
        self.assertAlmostEqual(bbox(prim).max[2], .75, places=6)

    def test_fixture_height_precedes_physics_reset_and_regions(self):
        for reset_name, method_name, restore in (
                ("individual_reset", "individual_randomize", False),
                ("individual_reset_from_mem", "individual_randomize_from_mem", True)):
            reset = method_from_file(ROOT / "workflows/simbox/core/tasks/banana.py", "BananaBaseTask", reset_name,
                {"update_rigid_objs": lambda cfg: cfg, "update_articulated_objs": lambda cfg: cfg,
                 "optimize_2d_manip_layout": lambda *args: None})
            apply = method_from_file(ROOT / "workflows/simbox/core/tasks/banana.py",
                                     "BananaBaseTask", method_name)
            events = []
            task = SimpleNamespace(cfg={"arena": {"fixtures": []}, "objects": [], "regions": []},
                current_id=0, objects={}, _set_pickcontact_view=Mock(), _set_artcontact_view=Mock(),
                language_instruction=None, detailed_language_instruction=None,
                _set_fixture_heights=lambda **kwargs: events.append(("height", kwargs.get("use_mem", False))),
                _set_robot_base_heights=Mock(),
                _set_regions=lambda **kwargs: events.append(("regions", None)),
                _set_envmap=Mock(), _set_fixture_textures=Mock(), _set_camera_poses=Mock())
            apply.__globals__["update_language"] = lambda cfg: ("", "")
            reset(task)
            events.append(("world_reset", None))
            apply(task)
            self.assertEqual(events, [("height", restore), ("world_reset", None), ("regions", None)])

    def test_resizing_invalidates_both_planning_mesh_caches_once(self):
        reset = method_from_file(ROOT / "workflows/simbox/core/controllers/template_controller.py",
                                 "TemplateController", "_reset_fixture_geometry_cache")
        scale = [.001, .001, .001]
        ctrl = SimpleNamespace(task=SimpleNamespace(cfg={"arena": {"fixtures": [self.cfg]}},
            fixtures={"table": SimpleNamespace(get_local_scale=lambda: scale)}),
            motion_gen=SimpleNamespace(clear_world_cache=Mock()),
            ik_solver=SimpleNamespace(world_coll_checker=SimpleNamespace(clear_cache=Mock())))
        self.assertTrue(reset(ctrl))
        self.assertFalse(reset(ctrl))
        scale[2] *= .77 / .75
        self.assertTrue(reset(ctrl))
        self.assertEqual(ctrl.motion_gen.clear_world_cache.call_count, 2)
        self.assertEqual(ctrl.ik_solver.world_coll_checker.clear_cache.call_count, 2)
        ctrl.task.cfg = {}
        # Restoring an older scene still needs to drop the resized vertices.
        scale[2] = .001
        self.assertTrue(reset(ctrl))
        self.assertFalse(reset(ctrl))

    def test_loading_legacy_config_restores_original_table_scale(self):
        apply = method_from_file(ROOT / "workflows/simbox/core/tasks/banana.py",
                                 "BananaBaseTask", "_set_fixture_heights")
        fixture = SimpleNamespace(set_local_scale=Mock())
        task = SimpleNamespace(cfg={"arena": {"fixtures": [self.cfg]}}, fixtures={"table": fixture})
        with patch.dict("sys.modules", {"core.utils.fixture_randomization": height}):
            apply(task)
            del self.cfg["height_randomization"]
            apply(task, use_mem=True)
        fixture.set_local_scale.assert_called_with([.001, .001, .001])

    def test_objects_are_placed_before_physics_starts_with_resized_table(self):
        reset = method_from_file(ROOT / "workflows/simbox_dual_workflow.py",
                                 "SimBoxDualWorkFlow", "_reset_layout_world")
        for managed in (True, False):
            for restore in (True, False):
                events = []
                task = SimpleNamespace(_height_randomized_fixtures={"table"} if managed else set(),
                    individual_randomize=lambda: events.append("new_objects"),
                    individual_randomize_from_mem=lambda: events.append("restored_objects"),
                    post_reset=lambda: events.append("post_reset"))
                wf = SimpleNamespace(task=task, world=SimpleNamespace(
                    stop=lambda: events.append("stop"), reset=lambda: events.append("reset")))
                reset(wf, use_mem=restore)
                objects = "restored_objects" if restore else "new_objects"
                expected = ["stop", objects, "reset", "post_reset"] if managed else ["reset", objects, "post_reset"]
                self.assertEqual(events, expected)


if __name__ == "__main__":
    unittest.main()
