"""Task regressions runnable with USD/numpy, without starting Isaac Sim or CUDA."""

import ast
import importlib.util
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import yaml
from pxr import Usd, UsdGeom
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]


def load_file(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


validation = load_file("placement_validation", "workflows/simbox/core/utils/placement_validation.py")
geometry = load_file("rigid_geometry", "workflows/simbox/core/utils/rigid_geometry.py")
grasps = load_file("cup_grasps", "scripts/assets/make_cup_grasp_labels.py")
hollow = load_file("hollow_cup", "workflows/simbox/core/utils/hollow_cup_collision.py")


def controller_methods():
    path = ROOT / "workflows/simbox/core/controllers/template_controller.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TemplateController")
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef)
               and n.name in {"attach_obj", "detach_obj", "update_specific"}]
    class JointStateStub:
        def __init__(self, **kw):
            self.__dict__.update(kw)
        def get_ordered_joint_state(self, names):
            indices = [self.joint_names.index(name) for name in names]
            return SimpleNamespace(position=self.position[indices], joint_names=list(names))
    scope = {"JointState": JointStateStub,
             "SphereFitType": SimpleNamespace(VOXEL_VOLUME_SAMPLE_SURFACE=1),
             "Pose": SimpleNamespace(from_list=lambda *args: None)}
    exec(compile(ast.fix_missing_locations(ast.Module(body=methods, type_ignores=[])), str(path), "exec"), scope)
    return type("ControllerMethods", (), {n.name: scope[n.name] for n in methods})


class CupPlateTaskTests(unittest.TestCase):
    def test_config_uses_eight_degree_cup_limit_and_verifies_after_home(self):
        cfg = yaml.safe_load((ROOT/'local_configs/cup_plate_desk/task.yaml').read_text())['tasks'][0]
        skills = cfg['skills'][0]['split_aloha'][0]['left']
        self.assertEqual([s['name'] for s in skills], ['pick', 'place', 'return_home', 'verify_placement'])
        self.assertEqual(cfg['robots'][0]['left_gripper_home'], [0.])
        self.assertEqual(cfg['robots'][0]['planner_gripper_open_width'], .050)
        self.assertEqual(skills[0]['pre_grasp_width'], .050)
        self.assertEqual(skills[3]['gripper_state'], -1.)
        for skill in [skills[1], skills[3]]:
            self.assertEqual(skill['max_tilt_deg'], 8)
            self.assertEqual(skill['max_support_tilt_deg'], 5)

    def test_cup_tilt_eight_does_not_relax_support_tilt(self):
        cfg = dict(support_radius=.048, object_radius=.0375, support_surface_z=.005,
                   max_tilt_deg=8, max_support_tilt_deg=5)
        cup = np.eye(4); cup[2, 3] = .005
        cup[:3, :3] = Rotation.from_euler('x', 7, degrees=True).as_matrix()
        still = (np.zeros(3), np.zeros(3))
        self.assertTrue(validation.placement_is_stable(cup, np.eye(4), still, still, cfg))
        cup[:3, :3] = Rotation.from_euler('x', 9, degrees=True).as_matrix()
        self.assertFalse(validation.placement_is_stable(cup, np.eye(4), still, still, cfg))
        plate = np.eye(4); plate[:3, :3] = Rotation.from_euler('x', 6, degrees=True).as_matrix()
        cup = plate.copy(); cup[:3, 3] += plate[:3, 2]*.005
        self.assertFalse(validation.placement_is_stable(cup, plate, still, still, cfg))

    def test_return_home_requires_full_path_and_consecutive_joint_accuracy(self):
        path = ROOT/'workflows/simbox/core/skills/return_home.py'
        cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef))
        cls.decorator_list = []
        scope = {'np': np, 'BaseSkill': object, 'prepare_return_home': Mock()}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), str(path), 'exec'), scope)
        q = np.zeros(6)
        skill = scope['ReturnHome'](SimpleNamespace(get_joint_positions=lambda: q),
            SimpleNamespace(arm_indices=np.arange(6), num_plan_failed=0), None, {})
        skill.target_joints = np.zeros(6)
        segment = {'complete': False}
        skill.manip_list = [(None, None, None, {'segment': segment})]
        self.assertFalse(skill.is_done())
        segment['complete'] = True
        q[0] = np.deg2rad(1)
        self.assertFalse(skill.is_done())
        q[0] = 0
        self.assertFalse(skill.is_done()); self.assertFalse(skill.is_done())
        self.assertTrue(skill.is_done()); self.assertTrue(skill.is_success())

    def setUp(self):
        self.cfg = dict(support_radius=.048, object_radius=.0375, support_surface_z=.005,
                        height_tolerance=.004, max_tilt_deg=5., max_linear_speed=.015,
                        max_angular_speed=.15)
        self.plate = np.eye(4)
        self.cup = np.eye(4)
        self.cup[2, 3] = .005
        self.still = (np.zeros(3), np.zeros(3))

    def valid(self, cup=None, velocity=None):
        return validation.placement_is_stable(self.cup if cup is None else cup, self.plate,
                                             self.still if velocity is None else velocity,
                                             self.still, self.cfg)

    def test_upright_supported_cup_is_valid(self):
        self.assertTrue(self.valid())

    def test_airborne_and_penetrating_cup_are_rejected(self):
        for z in [.02, -.005]:
            self.cup[2, 3] = z
            self.assertFalse(self.valid())

    def test_plate_bbox_corner_and_partial_footprint_are_rejected(self):
        for xy in [(.06, .06), (.02, 0)]:
            self.cup[:2, 3] = xy
            self.assertFalse(self.valid())

    def test_tipped_or_moving_cup_is_rejected(self):
        self.cup[:3, :3] = np.diag([1., -1., -1.])
        self.assertFalse(self.valid())
        self.cup[:3, :3] = np.eye(3)
        self.assertFalse(self.valid(velocity=(np.array([.02, 0, 0]), np.zeros(3))))
        self.assertFalse(self.valid(velocity=(np.zeros(3), np.array([0, 0, .2]))))

    def test_validation_follows_plate_translation_and_yaw(self):
        yaw = .7
        self.plate[:3, :3] = [[np.cos(yaw), -np.sin(yaw), 0],
                              [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
        self.plate[:3, 3] = [.4, -.2, .75]
        self.cup = self.plate @ self.cup
        self.assertTrue(self.valid())

    def test_generated_and_saved_candidates_are_exterior_side_only(self):
        candidates = grasps.make_side_grasps()
        np.testing.assert_array_equal(candidates, np.load(ROOT / "local_configs/cup_plate_desk/grasps/paper_cup_side.npy"))
        rotations = candidates[:, 4:13].reshape(-1, 3, 3)
        tilt = np.rad2deg(np.arcsin(-rotations[:, 2, 0]))
        self.assertEqual(len(candidates), 288)
        np.testing.assert_array_equal(candidates[:, 3], 0.)
        np.testing.assert_allclose(np.unique(np.round(tilt, 3)), [1, 15, 25, 30], atol=1e-4)
        np.testing.assert_allclose(rotations[:, 2, 1], 0, atol=1e-7)
        np.testing.assert_allclose(np.linalg.det(rotations), 1, atol=1e-6)
        self.assertTrue(np.all((candidates[:, 15] >= .044) & (candidates[:, 15] <= .056)))
        # Approach slopes down; jaw closure stays horizontal across exterior walls.
        convention = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
        ee_rotations = rotations @ convention.T
        np.testing.assert_allclose(ee_rotations[:, 2, 2], -np.sin(np.deg2rad(tilt)), atol=1e-7)
        np.testing.assert_allclose(ee_rotations[:, 2, 1], np.cos(np.deg2rad(tilt)), atol=1e-7)

    def test_active_rim_grasps_pinch_one_wall_with_vertical_entry(self):
        cfg = yaml.safe_load((ROOT/"local_configs/cup_plate_desk/task.yaml").read_text())["tasks"][0]
        pick = cfg["skills"][0]["split_aloha"][0]["left"][0]
        self.assertEqual(cfg["objects"][0]["target_class"], "RigidObject")
        candidates = grasps.make_rim_grasps()
        np.testing.assert_array_equal(candidates, np.load(ROOT/pick["grasp_pose_file"]))
        np.testing.assert_allclose(candidates[:, 1], pick['pre_grasp_width'])
        self.assertEqual(len(candidates), 768)
        np.testing.assert_allclose(np.unique(candidates[:, 15]), [.065, .068, .071, .074], atol=1e-7)
        rotations = candidates[:, 4:13].reshape(-1, 3, 3)
        inclinations = np.rad2deg(np.arccos(-rotations[:, 2, 0]))
        np.testing.assert_allclose(np.unique(np.round(inclinations, 3)), [0, 2, 5, 10, 15], atol=1e-4)
        radial = rotations[:, :, 1]
        tangent = np.column_stack([-radial[:, 1], radial[:, 0], np.zeros(len(radial))])
        signed = np.rad2deg(np.arctan2(np.einsum('ni,ni->n', rotations[:, :, 0], tangent), -rotations[:, 2, 0]))
        np.testing.assert_allclose(np.unique(np.round(signed, 3)), [-10, -5, -2, 0, 2, 5, 10, 15], atol=1e-4)
        downward_lo, downward_hi = pick["downward_tilt_range_deg"]
        self.assertTrue(np.all((90-inclinations >= downward_lo-.01) & (90-inclinations <= downward_hi+.01)))
        np.testing.assert_allclose(rotations[:, 2, 1], 0, atol=1e-7)
        np.testing.assert_allclose(np.linalg.det(rotations), 1, atol=1e-6)
        radius = np.linalg.norm(candidates[:, 13:15], axis=1)
        np.testing.assert_allclose(radius, .0265+(.0362-.0265)*np.minimum(candidates[:, 15], .07875)/.07875-.0006, atol=1e-7)
        # Open inner jaw is inside the lip; open outer jaw is outside it.
        inner = candidates[:, 13:16]-rotations[:, :, 1]*pick["pre_grasp_width"]/2
        outer = candidates[:, 13:16]+rotations[:, :, 1]*pick["pre_grasp_width"]/2
        self.assertTrue(np.all(np.linalg.norm(inner[:, :2], axis=1) < .030))
        self.assertTrue(np.all(np.linalg.norm(outer[:, :2], axis=1) > .040))
        self.assertEqual(pick["grasp_standoff"], 0)

    def test_rim_tcp_at_lip_is_allowed_but_not_above_cup(self):
        candidates = grasps.make_rim_grasps(grasp_heights=[.07875, .080])
        # The conical sidewall ends below the lip top; never extrapolate its radius.
        np.testing.assert_allclose(np.linalg.norm(candidates[:, 13:15], axis=1), .0356, atol=1e-7)
        for invalid_height in [.059, .081, np.nan]:
            with self.assertRaises(ValueError):
                grasps.make_rim_grasps(grasp_heights=[invalid_height])

    def test_fractional_gripper_command_does_not_reopen_to_full_width(self):
        path = ROOT/"workflows/simbox/core/controllers/template_controller.py"
        cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef))
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "dummy_forward")
        scope = {"np": np}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(path), "exec"), scope)
        ctrl = SimpleNamespace(arm_indices=np.arange(6), gripper_indices=np.array([6]), lr_name="left",
                               open_gripper=Mock(), close_gripper=Mock())
        ctrl.get_gripper_action = lambda: np.array([ctrl._gripper_state*.05])
        action = scope["dummy_forward"](ctrl, np.zeros(6), .24)
        np.testing.assert_allclose(action["gripper_action"], [.012])
        ctrl.open_gripper.assert_not_called()

    def test_planner_gripper_opening_preserves_opposed_piper_joint_signs(self):
        path = ROOT/"workflows/simbox/core/controllers/template_controller.py"
        cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef))
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_load_robot")
        config = {"kinematics": {"lock_joints": {"joint7": .05, "joint8": -.05}}}
        scope = {"np": np, "load_yaml": lambda unused: {"robot_cfg": config}}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(path), "exec"), scope)
        ctrl = SimpleNamespace(robot=SimpleNamespace(cfg={"planner_gripper_open_width": .024, "gripper_max_width": .1}))
        scope["_load_robot"](ctrl, "unused")
        self.assertEqual(config["kinematics"]["lock_joints"], {"joint7": .012, "joint8": -.012})

    def test_upright_placement_preserves_real_grasp_transform(self):
        path = ROOT / "workflows/simbox/core/skills/place.py"
        tree = ast.parse(path.read_text())
        method = next(n for c in tree.body if isinstance(c, ast.ClassDef)
                      for n in c.body if isinstance(n, ast.FunctionDef)
                      if n.name == "generate_constrained_rotation_batch")
        scope = {"np": np, "R": Rotation, "CUROBO_BATCH_SIZE": 20}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(path), "exec"), scope)
        grasp = np.eye(4)
        grasp[:3, :3] = Rotation.from_euler("xyz", [10, -25, 70], degrees=True).as_matrix()
        support = np.eye(4)
        support[:3, :3] = Rotation.from_euler("z", 30, degrees=True).as_matrix()
        skill = SimpleNamespace(skill_cfg={"object_rotation_mode": "upright"},
                                T_base_world=np.eye(4), T_world_container=support, T_obj_ee=grasp)
        ee_rotations = scope["generate_constrained_rotation_batch"](skill)
        object_rotations = ee_rotations @ grasp[:3, :3].T
        np.testing.assert_allclose(object_rotations[:, :, 2], np.tile([0, 0, 1], (20, 1)), atol=1e-7)
        skill.T_world_obj = np.eye(4)
        skill.T_world_obj[:3, :3] = Rotation.from_euler("z", 65, degrees=True).as_matrix()
        skill.skill_cfg["place_yaw_range"] = [-30, 30]
        anchored = scope["generate_constrained_rotation_batch"](skill) @ grasp[:3, :3].T
        np.testing.assert_allclose(anchored[0], skill.T_world_obj[:3, :3], atol=1e-7)

    def test_looks_scope_does_not_hide_any_cup_mesh(self):
        stage = Usd.Stage.Open(str(ROOT / "workflows/simbox/assets/custom/objects/paper_cup_001/Aligned_obj.usd"))
        prim = stage.GetPrimAtPath("/World/Aligned")
        self.assertEqual(prim.GetChildren()[0].GetName(), "Looks")
        paths = geometry.collect_mesh_paths(prim)
        self.assertEqual(len(paths), 6)
        self.assertTrue(all(stage.GetPrimAtPath(p).IsA(UsdGeom.Mesh) for p in paths))
        self.assertTrue(any(p.endswith("BottomInside") for p in paths))

    def test_hollow_collision_preserves_open_cavity_and_original_rigid_asset(self):
        from scipy.spatial import ConvexHull
        from pxr import UsdPhysics
        import hashlib
        source = ROOT/"workflows/simbox/assets/custom/objects/paper_cup_001/Aligned_obj.usd"
        before = hashlib.sha256(source.read_bytes()).hexdigest()
        stage = Usd.Stage.CreateInMemory()
        stage.DefinePrim("/Cup", "Xform").GetReferences().AddReference(str(source))
        body = stage.GetPrimAtPath("/Cup/Aligned")
        visuals = geometry.collect_mesh_paths(body)
        mass = UsdPhysics.MassAPI(body).GetMassAttr().Get()
        points_before = [np.asarray(UsdGeom.Mesh(stage.GetPrimAtPath(p)).GetPointsAttr().Get()).copy() for p in visuals]
        hollow.build_hollow_cup(stage, body, visuals)
        self.assertAlmostEqual(UsdPhysics.MassAPI(body).GetMassAttr().Get(), mass)
        for path, points in zip(visuals, points_before):
            np.testing.assert_array_equal(UsdGeom.Mesh(stage.GetPrimAtPath(path)).GetPointsAttr().Get(), points)
            self.assertFalse(UsdPhysics.CollisionAPI(stage.GetPrimAtPath(path)).GetCollisionEnabledAttr().Get())
        # No proxy seals the opening or fills the usable interior above the base.
        pieces = hollow.cup_convex_pieces()
        self.assertEqual(len(pieces), 97)
        for name, points in pieces:
            hull = ConvexHull(points)
            self.assertGreater(hull.volume, 0)
            for point in ([0, 0, 40], [20, 0, 40]):
                self.assertFalse(np.all(hull.equations[:, :3]@point+hull.equations[:, 3] <= 1e-8))
        bottom = ConvexHull(pieces[-1][1])
        self.assertTrue(np.all(bottom.equations[:, :3]@[0, 0, .6]+bottom.equations[:, 3] <= 1e-8))
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), before)

    def test_all_attached_meshes_stay_excluded_until_release(self):
        controller = controller_methods()()
        controller.arm_indices = np.array([0]); controller.raw_js_names = ['joint1']
        paths = ["/Cup/Wall", "/Cup/Bottom", "/Cup/Rim"]
        controller.robot = SimpleNamespace(cfg={}, dof_names=["joint1"],
            get_joints_state=lambda: SimpleNamespace(positions=np.zeros(1), velocities=np.zeros(1)))
        controller.tensor_args = SimpleNamespace(to_device=lambda x: x)
        controller.motion_gen = Mock()
        controller.motion_gen.joint_names = ['joint1']
        controller.motion_gen.world_model.get_obstacle.return_value = object()
        controller.motion_gen.attach_objects_to_robot.return_value = True
        controller.usd_help = Mock()
        controller.attach_obj(paths)
        self.assertEqual(controller.motion_gen.attach_objects_to_robot.call_args.args[1], paths)
        controller.update_specific(["robot"], "/Base")
        self.assertEqual(controller.usd_help.get_obstacles_from_stage.call_args.kwargs["ignore_paths"], paths)
        controller.detach_obj()
        controller.update_specific(["robot"], "/Base")
        self.assertEqual(controller.usd_help.get_obstacles_from_stage.call_args.kwargs["ignore_paths"], [])

    def test_attachment_fk_uses_active_arm_order_not_full_robot_positions(self):
        controller = controller_methods()()
        controller.arm_indices = np.array([2, 3]); controller.raw_js_names = ['joint2', 'joint1']
        controller.robot = SimpleNamespace(cfg={},
            dof_names=['lifting', 'fr_joint1', 'fl_joint2', 'fl_joint1', 'gripper'],
            get_joints_state=lambda: SimpleNamespace(positions=np.array([-.6, 9, 2, 1, .012]), velocities=np.zeros(5)))
        controller.tensor_args = SimpleNamespace(to_device=lambda x: x)
        controller.motion_gen = Mock()
        controller.motion_gen.joint_names = ['joint1', 'joint2']
        controller.motion_gen.world_model.get_obstacle.return_value = object()
        controller.motion_gen.attach_objects_to_robot.return_value = True
        controller.attach_obj(['/Cup/Wall'])
        state = controller.motion_gen.attach_objects_to_robot.call_args.args[0]
        np.testing.assert_array_equal(state.position, [1, 2])
        self.assertEqual(state.joint_names, ['joint1', 'joint2'])

    def test_final_verification_requires_consecutive_stable_frames(self):
        path = ROOT / "workflows/simbox/core/skills/verify_placement.py"
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        cls.decorator_list = []
        scope = {"np": np, "BaseSkill": object,
                 "placement_is_stable": validation.placement_is_stable,
                 "tf_matrix_from_pose": lambda pose, unused: pose}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), str(path), "exec"), scope)
        cup = SimpleNamespace(get_local_pose=lambda: (self.cup, None),
                              get_linear_velocity=lambda: np.zeros(3), get_angular_velocity=lambda: np.zeros(3))
        plate = SimpleNamespace(get_local_pose=lambda: (self.plate, None),
                                get_linear_velocity=lambda: np.zeros(3), get_angular_velocity=lambda: np.zeros(3))
        skill = scope["VerifyPlacement"](None, None, SimpleNamespace(objects={"cup": cup, "plate": plate}),
            dict(self.cfg, objects=["cup", "plate"], wait_steps=6, stable_steps=3))
        skill.robot = SimpleNamespace(get_joint_positions=lambda: np.zeros(6))
        skill.controller = SimpleNamespace(arm_indices=np.arange(6),
            get_ee_pose=lambda: (np.zeros(3), np.array([1, 0, 0, 0])))
        skill.skill_cfg['gripper_state'] = -1.
        skill.simple_generate_manip_cmds()
        self.assertEqual(skill.manip_list[0][3]['gripper_state'], -1.)
        skill.manip_list = [None] * 6
        for _ in range(3):
            skill.update(); skill.is_done()
        self.assertEqual(skill.stable_count, 3)
        self.assertFalse(skill.is_success())  # The final wait has not completed.
        self.cup[2, 3] = .05
        skill.update(); skill.is_done()
        self.assertEqual(skill.stable_count, 0)
        self.cup[2, 3] = .005
        for _ in range(2):
            skill.update(); skill.is_done()
        self.assertFalse(skill.is_success())  # Only two consecutive stable frames after the bounce.
        skill.manip_list = [None] * 3
        skill.stable_count = 0
        for _ in range(3):
            skill.update(); skill.is_done()
        self.assertTrue(skill.is_success())

    def test_three_cameras_record_images_and_calibration(self):
        cfg = yaml.safe_load((ROOT / "local_configs/cup_plate_desk/task.yaml").read_text())["tasks"][0]
        external = next(c for c in cfg["cameras"] if c["name"].endswith("external"))
        self.assertEqual(external["parent"], "")  # World mount is independent of logging ownership.
        path = ROOT / "workflows/simbox_dual_workflow.py"
        tree = ast.parse(path.read_text())
        method = next(n for c in tree.body if isinstance(c, ast.ClassDef)
                      for n in c.body if isinstance(n, ast.FunctionDef) and n.name == "_record_rgb_depth")
        scope = {"np": np}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(path), "exec"), scope)
        obs = dict(color_image=np.zeros((2, 2, 3), dtype=np.uint8), camera2env_pose=np.eye(4),
                   camera_params=[1, 1, 1, 1], depth_image=np.array([[np.nan, 1.]]),
                   semantic_mask=np.zeros((2, 2), dtype=np.uint16))
        cameras = {c["name"]: SimpleNamespace(get_observations=lambda: obs) for c in cfg["cameras"]}
        logger = Mock()
        workflow = SimpleNamespace(task=SimpleNamespace(cameras=cameras, robots={"split_aloha": object()}),
                                   logger=logger, log_motion_vectors=False)
        scope["_record_rgb_depth"](workflow, 0)
        self.assertEqual(logger.add_color_image.call_count, 3)
        self.assertEqual(logger.add_json_data.call_count, 3)
        self.assertEqual(logger.add_seg_image.call_count, 3)
        self.assertEqual(logger.add_depth_image.call_count, 3)
        for call in logger.add_depth_image.call_args_list:
            self.assertTrue(np.isfinite(call.args[2]).all())


if __name__ == "__main__":
    unittest.main()
