import glob
import os
import random
from copy import deepcopy
from typing import Dict

import numpy as np
import yaml
from core.cameras import CustomCamera
from core.objects import get_object_cls
from core.robots import get_robot_cls
from core.tasks.base_task import register_task
from core.utils.dr import update_articulated_objs, update_rigid_objs, update_scenes
from core.utils.language import update_language
from core.utils.layout import optimize_2d_manip_layout
from core.utils.region_sampler import RandomRegionSampler
from core.utils.scene_utils import deactivate_selected_prims
from core.utils.transformation_utils import get_orientation
from core.utils.visual_distractor import set_distractors
from omegaconf import DictConfig
from omni.isaac.core.materials import PreviewSurface
from omni.isaac.core.prims import RigidContactView, XFormPrim
from omni.isaac.core.scenes.scene import Scene
from omni.isaac.core.tasks import BaseTask
from omni.isaac.core.utils.prims import (
    delete_prim,
    get_prim_at_path,
    is_prim_path_valid,
)
from omni.isaac.core.utils.stage import get_current_stage
from omni.physx.scripts import particleUtils
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdShade, Vt
from scipy.spatial.transform import Rotation as R


@register_task
class BananaBaseTask(BaseTask):
    def __init__(
        self,
        cfg: DictConfig,
    ):
        super().__init__(name=cfg["name"], offset=cfg["offset"])
        self.cfg = cfg
        self._render = cfg.get("render", True)
        self.asset_root = os.path.abspath(self.cfg["asset_root"])
        self.root_prim_path = os.path.join("/World", f"task_{cfg['task_id']}")
        self.robots = {}
        self.cameras = {}
        self.cameras_info = {}
        self.objects = {}
        self.distractors = {}
        self.fixtures = {}
        self.visuals = {}
        self.stage = get_current_stage()
        self.random_region_list = self.cfg.get("random_region_list", [])
        self.current_id = 0

        self.first_set_fluid = True
        self.particleSystemPath = None
        self.particlesPath = None
        self.particlesPbdMaterialPath = None
        self.particlesVisualMaterialPath = None
        self._defaultFluidPath = Sdf.Path("/World/task_0/fulid")

    def set_up_scene(self, scene: Scene) -> None:
        super().set_up_scene(scene)
        self._set_envmap()
        self.cfg = update_scenes(self.cfg)
        for cfg in self.cfg["arena"]["fixtures"]:
            self.fixtures[cfg["name"]] = self._load_obj(cfg)
            if cfg["target_class"] == "ConveyorObject":
                self.conveyor_velocity = cfg["linear_velocity"][0]

        self._set_fixture_heights()
        self._set_fixture_appearance()

        self.cfg = update_rigid_objs(self.cfg)
        self.cfg = update_articulated_objs(self.cfg)

        for cfg in self.cfg["objects"]:
            self.objects[cfg["name"]] = self._load_obj(cfg)

        for cfg in self.cfg["robots"]:
            self._load_robot(cfg)
        self._set_robot_base_heights()
        for cfg in self.cfg["cameras"]:
            self._load_camera(cfg)
            if cfg.get("apply_randomization", False):
                self._perturb_camera(
                    self.cameras[cfg["name"]],
                    cfg,
                    max_translation_noise=cfg.get("max_translation_noise", 0.05),
                    max_orientation_noise=cfg.get("max_orientation_noise", 10.0),
                )

        self._task_objects |= self.fixtures | self.objects | self.robots | self.cameras

        # Initialize object regions according to region configs
        self._set_regions()

        optimize_2d_manip_layout(self.cfg["objects"], self.cfg["regions"], self.objects)

        # Object collision filtering (mainly for dynamicpick)
        self.ignore_objects = [obj["name"] for obj in self.cfg["objects"]]
        self.pickcontact_views = self._set_pickcontact_view(self.cfg)
        self.artcontact_views = self._set_artcontact_view(self.cfg)

        # Set up visual distrator (if exists)
        if self.cfg.get("distractors", None):
            cfgs = self._create_distractor_cfg()
            for cfg in cfgs:
                self.distractors[cfg["name"]] = self._load_obj(cfg)

            self.cfg["mem_distractors"] = cfgs
            set_distractors(
                self.objects,
                self.distractors,
                self._task_objects[self.cfg["distractors"]["target"]],
                self.cfg["distractors"],
                cfgs,
            )

        # Update language
        self.language_instruction, self.detailed_language_instruction = update_language(self.cfg)

    def individual_reset(self):
        self.current_id += 1
        # Update table and scene pair
        scene_update_freq = self.cfg["arena"].get("update_freq", 10)
        if self.current_id % scene_update_freq == 0:
            self.cfg = update_scenes(self.cfg)
            for cfg in self.cfg["arena"]["fixtures"]:
                if cfg.get("apply_randomization", False):
                    delete_prim(self.fixtures[cfg["name"]].prim_path)
                    self.fixtures[cfg["name"]] = self._load_obj(cfg)
                    self._task_objects[cfg["name"]] = self.fixtures[cfg["name"]]

        # Update objects
        self.cfg = update_rigid_objs(self.cfg)
        self.cfg = update_articulated_objs(self.cfg)
        for cfg in self.cfg["objects"]:
            if cfg.get("apply_randomization", False):
                delete_prim(os.path.dirname(self.objects[cfg["name"]].prim_path))
                self.objects[cfg["name"]] = self._load_obj(cfg)
                self._task_objects[cfg["name"]] = self.objects[cfg["name"]]

        optimize_2d_manip_layout(self.cfg["objects"], self.cfg["regions"], self.objects)
        self.pickcontact_views = self._set_pickcontact_view(self.cfg)
        self.artcontact_views = self._set_artcontact_view(self.cfg)
        # PhysX must see resized static collision meshes before World.reset().
        self._set_fixture_heights()
        self._set_robot_base_heights()

    def individual_reset_from_mem(self):
        for cfg in self.cfg["arena"]["fixtures"]:
            if cfg.get("apply_randomization", False):
                delete_prim(self.fixtures[cfg["name"]].prim_path)
                self.fixtures[cfg["name"]] = self._load_obj(cfg)
                self._task_objects[cfg["name"]] = self.fixtures[cfg["name"]]

        # Update objects
        for cfg in self.cfg["objects"]:
            if cfg.get("apply_randomization", False):
                delete_prim(os.path.dirname(self.objects[cfg["name"]].prim_path))
                self.objects[cfg["name"]] = self._load_obj(cfg)
                self._task_objects[cfg["name"]] = self.objects[cfg["name"]]

        optimize_2d_manip_layout(self.cfg["objects"], self.cfg["regions"], self.objects)
        self.pickcontact_views = self._set_pickcontact_view(self.cfg)
        self.artcontact_views = self._set_artcontact_view(self.cfg)
        self._set_fixture_heights(use_mem=True)
        self._set_robot_base_heights(use_mem=True)

    def individual_randomize(self):
        # Randomize objects in regions
        self._set_regions()

        # Update envmap, fixture textures, and camera poses
        self._set_envmap()
        self._set_fixture_textures()
        self._set_camera_poses()

        # Set up visual distractor (if exists, resample and update mem)
        if self.cfg.get("distractors", None):
            for obj in self.distractors.values():
                delete_prim(os.path.dirname(obj.prim_path))

            self.distractors = {}
            cfgs = self._create_distractor_cfg()
            self.cfg["mem_distractors"] = cfgs
            # use_mem=False → also calls set_distractors for new random placement
            self._rebuild_distractors(cfgs, use_mem=False)

        # Update language
        self.language_instruction, self.detailed_language_instruction = update_language(self.cfg)

    def individual_randomize_from_mem(self):
        # Randomize objects in regions (re-sample placements)
        self._set_regions(use_mem=True)

        # Update envmap, fixture textures, and camera poses
        self._set_envmap(use_mem=True)
        self._set_fixture_textures(use_mem=True)
        self._set_camera_poses(use_mem=True)

        # Rebuild visual distractors from mem (no new random placement)
        if self.cfg.get("mem_distractors", None) and self.cfg.get("distractors", None):
            for obj in self.distractors.values():
                delete_prim(os.path.dirname(obj.prim_path))

            self.distractors = {}
            cfgs = self.cfg["mem_distractors"]
            # use_mem=True → only rebuild objects, do not call set_distractors
            self._rebuild_distractors(cfgs, use_mem=True)

        # Update language
        self.language_instruction, self.detailed_language_instruction = update_language(self.cfg)

    def post_reset(self):
        for _, robot in self.robots.items():
            robot.initialize()
        for cfg in self.cfg["objects"]:
            if cfg["target_class"] == "ArticulatedObject":
                self.objects[cfg["name"]].initialize()

        all_views = [
            contact_view
            for views_dict in (self.pickcontact_views, self.artcontact_views)
            for lr_views in views_dict.values()
            for contact_views in lr_views.values()
            for contact_view in contact_views.values()
        ]

        for view in all_views:
            view.initialize()

    def apply_action(self, action: Dict[str, Dict[str, np.ndarray]]):
        for name in action.keys():
            self.robots[name].apply_action(**action[name])

    def get_observations(self):
        obs = {
            "robots": {},
            "objects": {},
            "cameras": {},
        }
        for name, robot in self.robots.items():
            obs["robots"][name] = robot.get_observations()
        for name, obj in self.objects.items():
            obs["objects"][name] = obj.get_observations()
        if self._render:
            for name, camera in self.cameras.items():
                obs["cameras"][name] = camera.get_observations()
        return obs

    # Load robot, camera and objects
    def _load_robot(self, cfg):
        robot = get_robot_cls(cfg["target_class"])(
            self.asset_root,
            self.root_prim_path,
            cfg,
        )
        orientation = get_orientation(cfg.get("euler"), cfg.get("quaternion"))
        robot.set_local_pose(
            translation=cfg.get("translation", [0.0, 0.0, 0.0]),
            orientation=orientation,
        )
        robot.set_local_scale(cfg.get("scale", [1.0, 1.0, 1.0]))
        self.robots[cfg["name"]] = robot

    def _load_camera(self, cfg):
        cameras_root = os.path.join(self.root_prim_path, "cameras")
        camera_prim_path = os.path.join(cameras_root, cfg["name"], "camera")
        camera_reference_path = os.path.join(self.root_prim_path, cfg["parent"]) if cfg["parent"] else ""

        if not is_prim_path_valid(os.path.join(cameras_root, cfg["name"])):
            for path in [cameras_root, os.path.join(cameras_root, cfg["name"])]:
                xform = XFormPrim(prim_path=path)
                xform.set_local_pose(translation=[0.0, 0.0, 0.0], orientation=[1.0, 0.0, 0.0, 0.0])

        # Load camera params from external file if camera_file is specified
        camera_file_path = cfg["camera_file"]
        with open(camera_file_path, "r", encoding="utf-8") as f:
            camera_params = yaml.safe_load(f)
        cfg = dict(cfg)
        cfg["params"] = camera_params

        # Use a single generic camera implementation.
        camera = CustomCamera(
            cfg=cfg,
            prim_path=camera_prim_path,
            root_prim_path=cameras_root,
            reference_path=camera_reference_path,
            name=cfg["name"],
        )

        orientation = cfg["orientation"]
        if cfg.get("look_at") is not None:
            from core.utils.camera_utils import look_at_usd_orientation
            if cfg["camera_axes"] != "usd":
                raise ValueError("look_at currently requires camera_axes: usd")
            orientation = look_at_usd_orientation(cfg["translation"], cfg["look_at"],
                                                  cfg.get("look_at_up", [0., 0., 1.]))
        camera.set_local_pose(
            translation=cfg["translation"],
            orientation=orientation,
            camera_axes=cfg["camera_axes"],
        )

        self.cameras[cfg["name"]] = camera
        self.cameras_info[cfg["name"]] = {
            "translation": deepcopy(camera.get_local_pose()[0]),
            "orientation": deepcopy(camera.get_local_pose()[1]),
        }

    def _load_obj(self, cfg: DictConfig):
        """Create and initialize any object based on cfg['target_class']."""
        target_class = cfg.get("target_class") or cfg["target_class"]
        obj_cls = get_object_cls(target_class)

        # Decide root prim and constructor args
        root_prim_path = self.root_prim_path
        ctor_args = [self.asset_root]

        if target_class == "XFormObject" and cfg.get("parent_obj", None):
            root_prim_path = self.objects[cfg["parent_obj"]].prim_path

        ctor_args.append(root_prim_path)

        if target_class == "ConveyorObject":
            ctor_args.append(self.stage)

        ctor_args.append(cfg)
        obj = obj_cls(*ctor_args)

        # Optional texture (for non-shape objects)
        if cfg.get("texture") and target_class not in ("ShapeObject",):
            obj.apply_texture(self.asset_root, cfg.get("texture"))

        orientation = get_orientation(cfg.get("euler"), cfg.get("quaternion"))
        obj.set_local_pose(translation=cfg.get("translation"), orientation=orientation)
        obj.set_local_scale(cfg.get("scale", [1.0, 1.0, 1.0]))
        obj.set_visibility(cfg.get("visible", True))

        # Extra behavior per type
        if target_class == "ArticulatedObject":
            obj.get_joint_position(self.stage)
            if cfg.get("collision_profile") == "hollow_microwave_0013":
                from core.utils.microwave_collision import build_hollow_microwave

                build_hollow_microwave(self.stage, obj)
        elif target_class == "ShapeObject":
            material = PreviewSurface(
                prim_path="/World/Materials/Red",
                color=np.array(cfg.get("color", np.array([1, 0, 0]))),
            )
            obj.apply_visual_material(material)

        # Special handling for scene object (only for general rigid/geometry)
        if target_class in ("RigidObject", "GeometryObject") and obj.name == "scene":
            deactivate_selected_prims(
                obj.prim, ["pan", "hearth", "ceiling", "__default_setting", "other", "microwave"], ["light"]
            )

        return obj

    # Set
    def _set_artcontact_view(self, cfg):
        artcontact_views = {}
        for cfg_skill_dict in cfg["skills"]:
            for robot_name, robot_skill_list in cfg_skill_dict.items():
                for lr_skill_dict in robot_skill_list:
                    for lr_name, lr_skill_list in lr_skill_dict.items():
                        for lr_skill in lr_skill_list:
                            if lr_skill.get("name") == "turn_knob":
                                robot = self.robots[robot_name]
                                object_name = lr_skill["objects"][0]
                                body = os.path.dirname(self._task_objects[object_name].object_base_path) + "/contact_link_0001"
                                filters = robot.fl_filter_paths_expr if lr_name == "left" else robot.fr_filter_paths_expr
                                views = artcontact_views.setdefault(robot_name, {}).setdefault(lr_name, {})
                                views[object_name + "_knob_fingers"] = RigidContactView(
                                    prim_paths_expr=body, filter_paths_expr=filters)
                            if lr_skill.get("name") == "open" or lr_skill.get("name") == "close":
                                if robot_name not in artcontact_views:
                                    artcontact_views[robot_name] = {}
                                if lr_name not in artcontact_views[robot_name]:
                                    artcontact_views[robot_name][lr_name] = {}

                                object_name = lr_skill["objects"][0]
                                robot = self.robots[robot_name]
                                filter_paths_expr = (
                                    robot.fl_filter_paths_expr if lr_name == "left" else robot.fr_filter_paths_expr
                                )
                                forbid_collision_paths = (
                                    robot.fl_forbid_collision_paths
                                    if lr_name == "left"
                                    else robot.fr_forbid_collision_paths
                                )
                                if (object_name + "_fingers_link") not in artcontact_views[robot_name][lr_name]:
                                    artcontact_views[robot_name][lr_name][
                                        object_name + "_fingers_link"
                                    ] = RigidContactView(
                                        prim_paths_expr=self._task_objects[object_name].object_link_path,
                                        filter_paths_expr=filter_paths_expr,
                                    )
                                if (object_name + "_fingers_base") not in artcontact_views[robot_name][lr_name]:
                                    artcontact_views[robot_name][lr_name][
                                        object_name + "_fingers_base"
                                    ] = RigidContactView(
                                        prim_paths_expr=self._task_objects[object_name].object_base_path,
                                        filter_paths_expr=filter_paths_expr,
                                    )

                                if (object_name + "_forbid_collision") not in artcontact_views[robot_name][lr_name]:
                                    artcontact_views[robot_name][lr_name][
                                        object_name + "_forbid_collision"
                                    ] = RigidContactView(
                                        prim_paths_expr=self._task_objects[object_name].object_prim_path
                                        + "/instance/*",
                                        filter_paths_expr=forbid_collision_paths,
                                    )

        return artcontact_views

    def _set_pickcontact_view(self, cfg):
        pickcontact_views = {}
        for cfg_skill_dict in cfg["skills"]:
            for robot_name, robot_skill_list in cfg_skill_dict.items():
                for lr_skill_dict in robot_skill_list:
                    for lr_name, lr_skill_list in lr_skill_dict.items():
                        for lr_skill in lr_skill_list:
                            if "pick" in lr_skill.get("name") or lr_skill.get("name") == "pinch_food":
                                if robot_name not in pickcontact_views:
                                    pickcontact_views[robot_name] = {}
                                if lr_name not in pickcontact_views[robot_name]:
                                    pickcontact_views[robot_name][lr_name] = {}
                                object_name = lr_skill["objects"][0]
                                prim_paths_expr = self.objects[object_name].prim_path
                                robot = self.robots[robot_name]
                                filter_paths_expr = (
                                    robot.fl_filter_paths_expr if lr_name == "left" else robot.fr_filter_paths_expr
                                )
                                if object_name not in pickcontact_views[robot_name][lr_name]:
                                    pickcontact_views[robot_name][lr_name][object_name] = RigidContactView(
                                        prim_paths_expr=prim_paths_expr, filter_paths_expr=filter_paths_expr
                                    )
        return pickcontact_views

    def _set_shared_regions(self, use_mem=False):
        from core.utils.shared_region_sampling import sample_separated_disks
        spec = self.cfg.get("shared_region_sampling", {})
        if not spec.get("enabled", False):
            return set()
        names = list(spec["object_radii"])
        regions = {cfg["object"]: cfg for cfg in self.cfg["regions"]}
        if any(regions[name]["target"] != spec["target"] or
               regions[name]["random_type"] != "A_on_B_region_sampler" for name in names):
            raise ValueError('Shared objects must use A_on_B_region_sampler on the same target')
        target = self._task_objects[spec["target"]]
        from core.utils.usd_geom_utils import compute_bbox
        bounds = np.asarray(spec["xy_bounds"], dtype=float)
        box = compute_bbox(target.prim)
        center = (np.asarray(box.min)[:2]+np.asarray(box.max)[:2])/2
        if bounds.shape != (2, 2) or np.any(bounds[0]+center < np.asarray(box.min)[:2]-1e-6) or np.any(bounds[1]+center > np.asarray(box.max)[:2]+1e-6):
            raise ValueError('Shared XY region must fit inside the support tabletop')
        state = self.cfg.get("mem_shared_layout") if use_mem else None
        if state is not None:
            for name in names:
                self._task_objects[name].set_local_pose(**state["poses"][name])
            return set(names)
        radii = [float(spec["object_radii"][name]) for name in names]
        positions, attempts = sample_separated_disks(bounds, radii,
            float(spec.get("clearance", .01)), spec.get("max_attempts", 10000))
        object_configs = {cfg["name"]: cfg for cfg in self.cfg["objects"]}
        poses = {}
        for name, xy in zip(names, positions):
            obj = self._task_objects[name]
            cfg = object_configs[name]
            # Restore upright model attitude before measuring height and sampling yaw.
            obj.set_local_pose(orientation=get_orientation(cfg.get("euler"), cfg.get("quaternion")))
            fixed = [float(xy[0]), float(xy[1]), 0.]
            position, orientation = RandomRegionSampler.A_on_B_region_sampler(obj, target,
                [fixed, fixed], regions[name]["random_config"]["yaw_rotation"])
            obj.set_local_pose(position, orientation)
            poses[name] = {"translation": np.asarray(position).tolist(), "orientation": np.asarray(orientation).tolist()}
        self.cfg["mem_shared_layout"] = {"xy_bounds": bounds.tolist(), "object_radii": dict(spec["object_radii"]),
            "clearance": float(spec.get("clearance", .01)), "sampling_attempts": attempts, "poses": poses}
        print("Shared object layout:", dict(self.cfg["mem_shared_layout"]), flush=True)
        return set(names)

    def _set_regions(self, use_mem=False):
        """Randomize object poses according to region configs."""
        shared_names = self._set_shared_regions(use_mem=use_mem)
        random_region_list = deepcopy(self.random_region_list)
        for cfg in self.cfg["regions"]:
            if cfg["object"] in shared_names:
                continue
            obj = self._task_objects[cfg["object"]]
            tgt = self._task_objects[cfg["target"]]
            if "sub_tgt_prim" in cfg:
                tgt = XFormPrim(prim_path=tgt.prim_path + cfg["sub_tgt_prim"])
            if "priority" in cfg:
                if cfg["priority"]:
                    idx = random.choice(cfg["priority"])
                else:
                    idx = random.randint(0, len(random_region_list) - 1)
                random_config = (random_region_list.pop(idx))["random_config"]
                sampler_fn = getattr(RandomRegionSampler, cfg["random_type"])
                pose = sampler_fn(obj, tgt, **random_config)
                obj.set_local_pose(*pose)
            elif "container" in cfg:
                container = self._task_objects[cfg["container"]]
                obj_trans = container.get_local_pose()[0]
                x_bias = random.choice(container.gap) if container.gap else 0
                obj_trans[0] += x_bias
                obj_trans[2] += cfg["z_init"]
                obj_ori = obj.get_local_pose()[1]
                obj.set_local_pose(obj_trans, obj_ori)
            elif "target2" in cfg:
                tgt2 = self._task_objects[cfg["target2"]]
                sampler_fn = getattr(RandomRegionSampler, cfg["random_type"])
                pose = sampler_fn(obj, tgt, tgt2, **cfg["random_config"])
                obj.set_local_pose(*pose)
            else:
                sampler_fn = getattr(RandomRegionSampler, cfg["random_type"])
                pose = sampler_fn(obj, tgt, **cfg["random_config"])
                obj.set_local_pose(*pose)

    def _set_robot_base_heights(self, use_mem=False):
        from core.utils.robot_height_randomization import episode_initial_joint_positions, set_robot_height_limits
        for cfg in self.cfg["robots"]:
            robot = self.robots[cfg["name"]]
            robot._episode_initial_joint_positions = episode_initial_joint_positions(cfg, use_mem)
            set_robot_height_limits(self.stage, robot, cfg, robot._episode_initial_joint_positions)
            robot._episode_base_height_cfg = cfg.get("base_height_randomization")
            if robot._episode_base_height_cfg:
                print(f"Robot base height {cfg['name']}: {dict(robot._episode_base_height_cfg['mem_height'])}", flush=True)

    def _set_fixture_heights(self, use_mem=False):
        from core.utils.fixture_randomization import apply_fixture_height
        previous = getattr(self, "_height_randomized_fixtures", set())
        current = set()
        for cfg in self.cfg["arena"]["fixtures"]:
            if cfg.get("height_randomization"):
                state = apply_fixture_height(self.fixtures[cfg["name"]], cfg, use_mem=use_mem)
                current.add(cfg["name"])
                print(f"Fixture height {cfg['name']}: {state}", flush=True)
            elif cfg["name"] in previous:
                # Old saved scenes predate this option; restore their baseline scale.
                self.fixtures[cfg["name"]].set_local_scale(cfg.get("scale", [1., 1., 1.]))
        self._height_randomized_fixtures = previous | current

    def _set_fixture_appearance(self, use_mem=False):
        from core.utils.appearance_randomization import apply_fixture_appearance
        for cfg in self.cfg["arena"]["fixtures"]:
            if cfg.get("appearance_randomization"):
                state = apply_fixture_appearance(self.stage, self.fixtures[cfg["name"]].prim_path,
                    cfg["appearance_randomization"], self.asset_root, use_mem=use_mem)
                print(f"Fixture appearance {cfg['name']}: {state}", flush=True)

    def _set_fixture_textures(self, use_mem=False):
        """Apply or randomize textures for arena fixtures (table, floor, background)."""
        for cfg in self.cfg["arena"]["fixtures"]:
            if cfg.get("texture"):
                obj = self.fixtures[cfg["name"]]
                if cfg["target_class"] == "PlaneObject":
                    obj.apply_texture(self.asset_root, cfg["texture"], use_mem=use_mem)
                else:
                    obj.apply_texture(self.asset_root, cfg["texture"])
        self._set_fixture_appearance(use_mem=use_mem)

    def _set_camera_poses(self, use_mem=False):
        """Randomize camera poses according to camera configs."""
        for cfg in self.cfg["cameras"]:
            if cfg.get("apply_randomization", False):
                self._perturb_camera(
                    self.cameras[cfg["name"]],
                    cfg,
                    max_translation_noise=cfg.get("max_translation_noise", 0.05),
                    max_orientation_noise=cfg.get("max_orientation_noise", 10.0),
                    use_mem=use_mem,
                )

    def _set_envmap(self, use_mem=False):
        """Randomize or reset the environment map (HDR dome light)."""
        cfg = self.cfg["env_map"]
        if cfg.get("light_type", "DomeLight") == "DomeLight":
            envmap_hdr_path_list = glob.glob(os.path.join(self.asset_root, cfg["envmap_lib"], "*.hdr"))
            envmap_hdr_path_list.sort()
            if not envmap_hdr_path_list:
                raise ValueError(f"No HDR environment maps in {cfg['envmap_lib']}")
            if cfg.get("apply_randomization", False):
                envmap_id = random.randint(0, len(envmap_hdr_path_list) - 1)
                intensity = random.uniform(cfg["intensity_range"][0], cfg["intensity_range"][1])
                rotation = [random.uniform(cfg["rotation_range"][0], cfg["rotation_range"][1]) for _ in range(3)]
            else:
                envmap_id = 0
                intensity = 1000.0
                rotation = [0.0, 0.0, 0.0]
            photometric = cfg.get("photometric_randomization", {})
            light_state = None
            if photometric.get("enabled", False):
                from core.utils.appearance_randomization import sample_lighting
                light_state = cfg.get("mem_light") if use_mem else None
                light_state = light_state or sample_lighting(photometric)
                intensity, rotation = light_state["intensity"], light_state["rotation_deg"]
                envmap_id = int(light_state.get("envmap_id", envmap_id))
                light_state["envmap_id"] = envmap_id
                cfg["mem_light"] = light_state
            dome_prim_path = f"{self.root_prim_path}/DomeLight"
            envmap_hdr_path = envmap_hdr_path_list[envmap_id]

            if not is_prim_path_valid(dome_prim_path):
                self.dome_light_prim = UsdLux.DomeLight.Define(self.stage, dome_prim_path)
                UsdGeom.Xformable(self.dome_light_prim).AddRotateXYZOp().Set((rotation[0], rotation[1], rotation[2]))
            else:
                self.dome_light_prim.GetOrderedXformOps()[0].Set((rotation[0], rotation[1], rotation[2]))
            self.dome_light_prim.GetIntensityAttr().Set(intensity)
            self.dome_light_prim.GetTextureFileAttr().Set(envmap_hdr_path)
            if light_state is not None:
                light_api = UsdLux.LightAPI(self.dome_light_prim.GetPrim())
                light_api.CreateEnableColorTemperatureAttr().Set(True)
                light_api.CreateColorTemperatureAttr().Set(light_state["temperature_kelvin"])
                light_api.CreateColorAttr().Set(Gf.Vec3f(1, 1, 1))
                print(f"Episode lighting: {light_state}", flush=True)

    def _set_fluid(self):
        # Particle params
        self.particleContactOffset = self.cfg["fluid"].get("particleContactOffset", 0.005)
        self.particleSpacing = self.particleContactOffset * self.cfg["fluid"].get("spacing_scale", 1.2)

        offset = self._get_container_center()
        numParticlesX = self.cfg["fluid"].get("numParticlesX", 7)
        numParticlesY = self.cfg["fluid"].get("numParticlesY", 7)
        numParticlesZ = self.cfg["fluid"].get("numParticlesZ", 450)
        lower_x = (numParticlesX - 1) * self.particleSpacing * -0.5 + offset[0].item()
        lower_y = (numParticlesY - 1) * self.particleSpacing * -0.5 + offset[1].item()
        lower_z = (
            (numParticlesZ - 1) * self.particleSpacing * -0.5 + offset[2].item()
            if self.cfg["fluid"].get("center_z", False)
            else offset[2].item()
        )
        z_offset = self.cfg["fluid"].get("z_offset", 0.0)
        lower_z += z_offset
        lower = Gf.Vec3f(lower_x, lower_y, lower_z)

        positions, velocities = particleUtils.create_particles_grid(
            lower, self.particleSpacing, numParticlesX, numParticlesY, numParticlesZ
        )
        widths = [self.particleSpacing] * len(positions)

        positions = Vt.Vec3fArray(positions)
        velocities = Vt.Vec3fArray(velocities)
        widths = Vt.FloatArray(widths)

        if self.first_set_fluid:
            # Particle system
            self.particleSystemPath = self._defaultFluidPath.AppendChild("particleSystem0")

            self.particle_system = particleUtils.add_physx_particle_system(
                stage=self.stage,
                particle_system_path=self.particleSystemPath,
                particle_system_enabled=True,
                simulation_owner=None,
                # contact_offset=self.particleContactOffset,
                # rest_offset=self.particleContactOffset * 0.99,
                particle_contact_offset=self.particleContactOffset,
                # solid_rest_offset=self.particleContactOffset * 0.99,
                # fluid_rest_offset=self.particleContactOffset * 0.99 * 0.6,
                enable_ccd=True,
                solver_position_iterations=16,
                max_depenetration_velocity=None,
                wind=None,
                max_neighborhood=96,
                neighborhood_scale=1.01,
                max_velocity=self.cfg["fluid"].get("max_velocity", 0.8),
                global_self_collision_enabled=True,
                non_particle_collision_enabled=None,
            )
            particleUtils.add_physx_particle_isosurface(self.stage, self.particleSystemPath)

            smoothingAPI = PhysxSchema.PhysxParticleSmoothingAPI.Apply(self.particle_system.GetPrim())
            smoothingAPI.CreateParticleSmoothingEnabledAttr().Set(True)
            smoothingAPI.CreateStrengthAttr().Set(50.0)

            self.particlesPath = self._defaultFluidPath.AppendChild("particles")

            self.stage.SetInterpolationType(Usd.InterpolationTypeLinear)

            self.particles = particleUtils.add_physx_particleset_points(
                stage=self.stage,
                path=self.particlesPath,
                positions_list=positions,
                velocities_list=velocities,
                widths_list=widths,
                particle_system_path=self.particleSystemPath,
                self_collision=True,
                fluid=True,
                particle_group=0,
                particle_mass=self.cfg["fluid"].get("mass", 0.000000),
                density=self.cfg["fluid"].get("density", 0.000000),
            )

            self.particlesPbdMaterialPath = self._defaultFluidPath.AppendChild("pdbMaterial")

            self.particlesVisualMaterialPath = self._defaultFluidPath.AppendChild("visualMaterial")

            particleUtils.add_pbd_particle_material(
                stage=self.stage,
                path=self.particlesPbdMaterialPath,
                cohesion=0.01,
                drag=0,
                lift=0,
                damping=0,
                friction=0.1,
                surface_tension=0.0074,
                viscosity=0.0000017,
                vorticity_confinement=0,
            )
            particlesPbdMaterial_prim = get_prim_at_path(self.particlesPbdMaterialPath)
            material = UsdShade.Material(particlesPbdMaterial_prim)

            particleSystem_prim = get_prim_at_path(self.particleSystemPath)
            binding_api = UsdShade.MaterialBindingAPI.Apply(particleSystem_prim)
            binding_api.Bind(material)

            material = self._create_colored_material(
                self.stage,
                self.particlesVisualMaterialPath,
                color=self.cfg["fluid"].get("color", [1.0, 1.0, 1.0]),
                emissiveColor=self.cfg["fluid"].get("emissiveColor", [0.0, 0.0, 0.0]),
                opacity=self.cfg["fluid"].get("opacity", 1),
            )
            binding_api.Bind(material)

            self.first_set_fluid = False

        else:
            self.particles.GetPointsAttr().Set(positions)
            self.particles.GetVelocitiesAttr().Set(velocities)
            self.particles.GetWidthsAttr().Set(widths)

        particles_prim = self.stage.GetPrimAtPath(self.particlesPath)
        if particles_prim:
            purpose_attr = particles_prim.CreateAttribute("purpose", Sdf.ValueTypeNames.Token)
            purpose_attr.Set("proxy")

        return self.particles

    # Utilities
    def _perturb_camera(self, camera, cfg, max_translation_noise=0.05, max_orientation_noise=10.0, use_mem=False):
        from core.utils.camera_utils import sample_camera_pose
        state = cfg.get("mem_pose") if use_mem else None
        state = state or sample_camera_pose(cfg, max_translation_noise, max_orientation_noise)
        camera.set_local_pose(
            translation=state["translation"],
            orientation=state["orientation_wxyz"],
            camera_axes=cfg["camera_axes"],
        )
        cfg["mem_pose"] = state
        print(f"Camera pose {cfg['name']}: {state}", flush=True)

    def _create_distractor_cfg(self):
        distractors_cfg = self.cfg["distractors"]

        # Collect all available distractor asset paths
        distractor_paths = glob.glob(
            os.path.join(self.asset_root, distractors_cfg["path"], "*", "*", "*.usd")  # category  # subcategory
        )
        distractor_paths.sort()

        # Categories already used by main objects in the scene
        current_categories = {obj_cfg["path"].split("/")[-3] for obj_cfg in self.cfg["objects"]}

        # Optional: categories to be excluded from distractors via config
        # Example in config:
        # distractors:
        #   exclude_categories: ["omniobject3d-shoe", "omniobject3d-book"]
        #   exclude_keywords: ["shoe", "book"]
        excluded_categories = set(distractors_cfg.get("exclude_categories", []))
        exclude_keywords = [k.lower() for k in distractors_cfg.get("exclude_keywords", [])]

        filtered_distractors = []
        for path in distractor_paths:
            category = path.split("/")[-3]
            category_lower = category.lower()

            # Skip if category is already used by main objects
            if category in current_categories:
                continue

            # Skip if category is explicitly excluded
            if category in excluded_categories:
                continue

            # Skip if any keyword appears in the category name (case-insensitive)
            if any(kw in category_lower for kw in exclude_keywords):
                continue

            filtered_distractors.append(path)

        num_samples = random.randint(
            distractors_cfg["min_num"], min(distractors_cfg["max_num"], len(filtered_distractors))
        )
        filtered_distractors = random.sample(filtered_distractors, num_samples)

        cfgs = []
        for path in filtered_distractors:
            tmp_cfg = {}
            tmp_cfg["name"] = "distractors" + "/" + path.split("/")[-2]
            tmp_cfg["name"] = (tmp_cfg["name"]).replace("-", "_")
            tmp_cfg["path"] = path.replace(self.asset_root, "")
            tmp_cfg["target_class"] = distractors_cfg.get("target_class", "RigidObject")  # "RigidObject"
            tmp_cfg["prim_path_child"] = distractors_cfg.get("prim_path_child", "Aligned")  # "Aligned"
            tmp_cfg["translation"] = distractors_cfg.get("translation", [0.0, 0.0, 0.0])
            tmp_cfg["scale"] = distractors_cfg.get("scale", [1.0, 1.0, 1.0])
            tmp_category = path.split("/")[-3]
            tmp_cfg["category"] = tmp_category
            tmp_cfg = DictConfig(tmp_cfg)
            cfgs.append(tmp_cfg)

        return cfgs

    def _rebuild_distractors(self, cfgs, use_mem: bool):
        """Rebuild distractor objects from a list of configs.

        - If use_mem is True, only rebuild objects (no new random placement via set_distractors).
        - If use_mem is False, also call set_distractors to (re)sample placements.
        """
        for cfg in cfgs:
            if cfg["target_class"] == "RigidObject":
                self.distractors[cfg["name"]] = self._load_obj(cfg)
            else:
                raise NotImplementedError

        if (not use_mem) and self.cfg.get("distractors", None):
            set_distractors(
                self.objects,
                self.distractors,
                self._task_objects[self.cfg["distractors"]["target"]],
                self.cfg["distractors"],
                cfgs,
            )

    def _get_container_center(self):
        container_name = self.cfg["fluid"]["container_name"]
        container = self.objects[container_name]
        container_trans, _ = container.get_world_pose()

        return container_trans

    def _create_colored_material(
        self, stage, material_path, color=(1.0, 0.0, 0.0), emissiveColor=(0.0, 0.0, 0.0), opacity=1.0
    ):
        material_prim = stage.DefinePrim(material_path, "Material")
        material = UsdShade.Material(material_prim)

        shader_path = f"{material_path}/PreviewSurface"
        shader_prim = stage.DefinePrim(shader_path, "Shader")
        shader = UsdShade.Shader(shader_prim)

        shader.CreateIdAttr("UsdPreviewSurface")

        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*emissiveColor))

        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.4)
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)

        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

        return material
