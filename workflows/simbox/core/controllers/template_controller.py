"""
Template Controller base class for robot motion planning.

Common functionality extracted from FR3, FrankaRobotiq85, Genie1, Lift2, SplitAloha.
Subclasses implement _get_default_ignore_substring() and _configure_joint_indices().
"""

import random
import time
import os
from copy import deepcopy
from typing import List, Optional

import numpy as np
import torch
from core.utils.constants import CUROBO_BATCH_SIZE
from core.utils.plan_utils import (
    filter_paths_by_position_error,
    filter_paths_by_rotation_error,
    sort_by_difference_js,
)
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.sphere_fit import SphereFitType
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState, RobotConfig
from curobo.util.usd_helper import UsdHelper
from curobo.util_file import get_world_configs_path, join_path, load_yaml
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
    PoseCostMetric,
)
from omni.isaac.core import World
from omni.isaac.core.controllers import BaseController
from omni.isaac.core.tasks import BaseTask
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.transformations import (
    get_relative_transform,
    pose_from_tf_matrix,
)
from omni.isaac.core.utils.types import ArticulationAction


# pylint: disable=line-too-long,unused-argument
class TemplateController(BaseController):
    """Base controller for CuRobo-based motion planning. Supports single and batch planning."""

    def __init__(
        self,
        name: str,
        robot_file: str,
        task: BaseTask,
        world: World,
        constrain_grasp_approach: bool = False,
        collision_activation_distance: float = 0.03,
        ignore_substring: Optional[List[str]] = None,
        use_batch: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(name=name)
        self.name = name
        self.world = world
        self.task = task
        self.robot = self.task.robots[name]
        self.ignore_substring = self._get_default_ignore_substring()
        if ignore_substring is not None:
            self.ignore_substring = ignore_substring
        self.ignore_substring.append(name)
        self.use_batch = use_batch
        self.constrain_grasp_approach = constrain_grasp_approach
        self.collision_activation_distance = collision_activation_distance
        self.usd_help = UsdHelper()
        self.tensor_args = TensorDeviceType()
        self.init_curobo = False
        self.robot_file = robot_file
        self.num_plan_failed = 0
        self.raw_js_names = []
        self.cmd_js_names = []
        self.arm_indices = np.array([])
        self.gripper_indices = np.array([])
        self.reference_prim_path = None
        self.lr_name = None
        self._ee_trans = 0.0
        self._ee_ori = 0.0
        self._gripper_state = 1.0
        self._gripper_joint_position = np.array([1.0])
        self.idx_list = None

        self._configure_joint_indices(robot_file)
        self._load_robot(robot_file)
        self._load_kin_model()
        self._load_world()
        self._init_motion_gen()

        self.usd_help.load_stage(self.world.stage)
        self.cmd_plan = None
        self._last_arm_target = None
        self.cmd_idx = 0
        self._step_idx = 0
        self.num_last_cmd = 0
        self.ds_ratio = 1

    def _get_default_ignore_substring(self) -> List[str]:
        return ["material", "Plane", "conveyor", "scene", "table"]

    def _configure_joint_indices(self, robot_file: str) -> None:
        raise NotImplementedError

    def _load_robot(self, robot_file: str) -> None:
        self.robot_cfg = load_yaml(robot_file)["robot_cfg"]
        opening = self.robot.cfg.get("planner_gripper_open_width")
        if opening is not None:
            opening = float(opening)
            if not np.isfinite(opening) or not 0 < opening <= self.robot.cfg["gripper_max_width"]:
                raise ValueError("Invalid planner_gripper_open_width")
            locked = self.robot_cfg["kinematics"].get("lock_joints", {})
            if not {"joint7", "joint8"}.issubset(locked):
                raise ValueError("Gripper opening synchronization requires locked joint7/joint8")
            for name in ("joint7", "joint8"):
                locked[name] = (1. if locked[name] >= 0 else -1.)*opening/2
        attached_spheres = self.robot.cfg.get("attached_object_spheres")
        if attached_spheres is not None:
            if int(attached_spheres) <= 0:
                raise ValueError("attached_object_spheres must be positive")
            self.robot_cfg["kinematics"].setdefault("extra_collision_spheres", {})["attached_object"] = int(attached_spheres)
        # A carried object may touch both jaws. Keep all other collision pairs;
        # the bundled Piper config omits link8 from its attachment exclusions.
        for link in self.robot.cfg.get("attached_object_contact_links", []):
            ignored = self.robot_cfg["kinematics"].setdefault("self_collision_ignore", {}).setdefault(link, [])
            if "attached_object" not in ignored:
                ignored.append("attached_object")
        proxy_file = self.robot.cfg.get("planner_gripper_collision_file")
        proxy_arm = self.robot.cfg.get("planner_gripper_collision_arm")
        if proxy_file and (proxy_arm is None or proxy_arm == self.lr_name):
            proxies = load_yaml(proxy_file)
            kinematics = self.robot_cfg["kinematics"]
            padding = kinematics.get("collision_sphere_buffer", 0.)
            if isinstance(padding, (int, float)):
                padding = {link: float(padding) for link in kinematics["collision_link_names"]}
            self._measured_gripper_spheres = {}
            staged = self.robot.cfg.get("planner_gripper_collision_stage") == "turn_knob"
            for link, spheres in proxies.items():
                original = deepcopy(kinematics["collision_spheres"][link]) if staged else []
                for sphere in original:
                    sphere["radius"] += padding.get(link, 0.)
                disabled = {"center": [0., 0., 0.], "radius": -10.}
                # Reserve the same number of slots in both models. CUDA graphs
                # require a constant sphere count when changing grasp phases.
                self._measured_gripper_spheres[link] = [deepcopy(disabled) for _ in original] + spheres
                kinematics["collision_spheres"][link] = (
                    original + [deepcopy(disabled) for _ in spheres] if staged else spheres)
            self._planning_gripper_measured = not staged
            # Voxel sphere radii already conservatively cover the measured mesh.
            kinematics["collision_sphere_buffer"] = {**padding, **{link: 0. for link in proxies}}
            count = sum(len(kinematics["collision_spheres"].get(link, []))
                        + (kinematics.get("extra_collision_spheres", {}).get(link, 0)
                           if link not in kinematics["collision_spheres"] else 0)
                        for link in kinematics["collision_link_names"])
            if count >= 1024:
                raise ValueError(f"CuRobo requires fewer than 1024 collision spheres; got {count}")
            print("Measured gripper collision proxies:", {k: len(v) for k, v in proxies.items()}, flush=True)

    def _load_kin_model(self) -> None:
        urdf_file = self.robot_cfg["kinematics"]["urdf_path"]
        base_link = self.robot_cfg["kinematics"]["base_link"]
        ee_link = self.robot_cfg["kinematics"]["ee_link"]
        robot_cfg = RobotConfig.from_basic(urdf_file, base_link, ee_link, self.tensor_args)
        self.kin_model = CudaRobotModel(robot_cfg.kinematics)

    def _load_world(self, use_default: bool = True) -> None:
        if use_default:
            self.world_cfg = WorldConfig()
        else:
            world_cfg_table = WorldConfig.from_dict(
                load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
            )
            self._world_cfg_table = world_cfg_table
            self._world_cfg_table.cuboid[0].pose[2] -= 10.5
            world_cfg1 = WorldConfig.from_dict(
                load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
            ).get_mesh_world()
            world_cfg1.mesh[0].name += "_mesh"
            world_cfg1.mesh[0].pose[2] = -10.5
            self.world_cfg = WorldConfig(cuboid=world_cfg_table.cuboid, mesh=world_cfg1.mesh)

    def _get_motion_gen_collision_cache(self):
        """Override in subclasses to use different cache size (e.g. FR3 uses 1000)."""
        return {"obb": 700, "mesh": 700}

    def _get_grasp_approach_linear_axis(self) -> int:
        """Axis for grasp approach constraint (0=x, 1=y, 2=z). Override in subclasses (e.g. Lift2 uses 0)."""
        if self.robot.cfg["ee_axis"] == "x":
            return 0
        elif self.robot.cfg["ee_axis"] == "y":
            return 1
        elif self.robot.cfg["ee_axis"] == "z":
            return 2
        else:
            raise NotImplementedError

    def _get_sort_path_weights(self) -> Optional[List[float]]:
        """Optional per-joint weights for sort_by_difference_js.

        Used when selecting among batch paths. None means equal weights.
        Override in subclasses (e.g. Genie1).
        """
        return None

    def _init_motion_gen(self) -> None:
        pose_metric = None
        if self.constrain_grasp_approach:
            pose_metric = PoseCostMetric.create_grasp_approach_metric(
                offset_position=0.1,
                linear_axis=self._get_grasp_approach_linear_axis(),
            )
        if self.use_batch:
            self.plan_config = MotionGenPlanConfig(
                enable_graph=True,
                enable_opt=True,
                need_graph_success=True,
                enable_graph_attempt=self.robot.cfg.get("planner_graph_fallback_attempt", 4),
                max_attempts=self.robot.cfg.get("planner_max_attempts", 4),
                timeout=self.robot.cfg.get("planner_timeout", 10.0),
                enable_finetune_trajopt=True,
                parallel_finetune=True,
                time_dilation_factor=self.robot.cfg.get("planner_time_dilation_factor", 1.0),
            )
        else:
            self.plan_config = MotionGenPlanConfig(
                enable_graph=False,
                enable_graph_attempt=self.robot.cfg.get("planner_graph_fallback_attempt", 7),
                max_attempts=self.robot.cfg.get("planner_max_attempts", 10),
                timeout=self.robot.cfg.get("planner_timeout", 10.0),
                pose_cost_metric=pose_metric,
                enable_finetune_trajopt=True,
                time_dilation_factor=self.robot.cfg.get("planner_time_dilation_factor", 1.0),
            )
        motion_gen_config = MotionGenConfig.load_from_robot_config(
            self.robot_cfg,
            self.world_cfg,
            self.tensor_args,
            interpolation_dt=0.01,
            collision_activation_distance=self.collision_activation_distance,
            trajopt_tsteps=32,
            collision_checker_type=CollisionCheckerType.MESH,
            use_cuda_graph=True,
            self_collision_check=True,
            collision_cache=self._get_motion_gen_collision_cache(),
            num_ik_seeds=self.robot.cfg.get("planner_num_ik_seeds", 32),
            num_trajopt_seeds=self.robot.cfg.get("planner_num_trajopt_seeds", 12),
            num_graph_seeds=self.robot.cfg.get("planner_num_graph_seeds", 12),
            optimize_dt=True,
            trajopt_dt=None,
            trim_steps=None,
            project_pose_to_goal_frame=False,
        )
        ik_config = IKSolverConfig.load_from_robot_config(
            self.robot_cfg,
            self.world_cfg,
            rotation_threshold=self.robot.cfg.get("planner_ik_rotation_threshold", 0.05),
            position_threshold=self.robot.cfg.get("planner_ik_position_threshold", 0.005),
            num_seeds=self.robot.cfg.get("planner_ik_check_seeds", 20),
            self_collision_check=True,
            self_collision_opt=True,
            tensor_args=self.tensor_args,
            use_cuda_graph=True,
            collision_checker_type=CollisionCheckerType.MESH,
            collision_cache={"obb": 700, "mesh": 700},
        )
        self.ik_solver = IKSolver(ik_config)
        self.motion_gen = MotionGen(motion_gen_config)
        print("warming up..")
        if self.use_batch:
            self.motion_gen.warmup(parallel_finetune=True, batch=CUROBO_BATCH_SIZE)
        else:
            self.motion_gen.warmup(enable_graph=True, warmup_js_trajopt=False)
        self.world_model = self.motion_gen.world_collision
        self.motion_gen.clear_world_cache()
        self.motion_gen.reset(reset_seed=False)
        self.motion_gen.update_world(self.world_cfg)
        print("Planning budget:", {"arm": self.lr_name, "ik_seeds": motion_gen_config.ik_seeds,
                                   "trajopt_seeds": motion_gen_config.trajopt_seeds,
                                   "graph_seeds": motion_gen_config.graph_seeds,
                                   "max_attempts": self.plan_config.max_attempts,
                                   "timeout_s": self.plan_config.timeout,
                                   "graph_fallback_attempt": self.plan_config.enable_graph_attempt}, flush=True)

    def update_pose_cost_metric(self, hold_vec_weight: Optional[List[float]] = None) -> None:
        # reference: https://curobo.org/advanced_examples/3_constrained_planning.html
        # [angular-x, angular-y, angular-z, linear-x, linear-y, linear-z]
        # For example,
        # when hold_vec_weight is None, the corresponding list is [0, 0, 0, 0, 0, 0],
        # there is no cost added in any directions.
        # When hold_vec_weight = [1, 1, 1, 0, 0, 0], the tool orientation is holed.
        # assert hold_vec_weight is None or len(hold_vec_weight) == 6
        if hold_vec_weight:
            pose_cost_metric = PoseCostMetric(
                hold_partial_pose=True,
                hold_vec_weight=self.motion_gen.tensor_args.to_device(hold_vec_weight),
            )
        else:
            pose_cost_metric = None
        self.plan_config.pose_cost_metric = pose_cost_metric

    def update(self) -> None:
        obstacles = self.usd_help.get_obstacles_from_stage(
            ignore_substring=self.ignore_substring, reference_prim_path=self.reference_prim_path,
            ignore_paths=getattr(self, "_attached_object_paths", None),
        ).get_collision_check_world()
        if self.motion_gen is not None:
            self.motion_gen.update_world(obstacles)
        self.world_cfg = obstacles

    def _reset_fixture_geometry_cache(self) -> bool:
        # CuRobo keys Warp mesh vertices by prim name, ignoring a changed scale.
        # Both MotionGen and the independent IK checker must rebuild these meshes.
        previous = getattr(self, "_fixture_geometry_signature", ())
        names = {cfg["name"] for cfg in self.task.cfg.get("arena", {}).get("fixtures", [])
                 if cfg.get("height_randomization")}
        names.update(name for name, _ in previous)
        signature = tuple((name, tuple(self.task.fixtures[name].get_local_scale()))
                          for name in sorted(names) if name in self.task.fixtures)
        if signature == previous:
            return False
        self.motion_gen.clear_world_cache()
        self.ik_solver.world_coll_checker.clear_cache()
        self._fixture_geometry_signature = signature
        return True

    def reset(self, ignore_substring: Optional[str] = None) -> None:
        if ignore_substring:
            self.ignore_substring = ignore_substring
        if getattr(self, "_attached_object_paths", None):
            self.detach_obj()
        self._include_other_arm_obstacles = False
        self._attached_object_paths = []
        self._cup_plate_chain = None
        self._validated_active = None
        geometry_changed = self._reset_fixture_geometry_cache()
        self.update()
        if geometry_changed:
            self.ik_solver.update_world(self.world_cfg)
        self.init_curobo = True
        self.cmd_plan = None
        self._last_arm_target = None
        self.cmd_idx = 0
        self.num_plan_failed = 0
        if hasattr(self, "_planning_gripper_width"):
            locked = self.robot_cfg["kinematics"]["lock_joints"]
            measured = self.robot.cfg.get("planner_gripper_collision_stage") != "turn_knob"
            self.update_gripper_collision(abs(locked["joint7"]) + abs(locked["joint8"]), measured=measured)
        if self.lr_name == "left":
            self._gripper_state = 1.0 if self.robot.left_gripper_state == 1.0 else -1.0
        elif self.lr_name == "right":
            self._gripper_state = 1.0 if self.robot.right_gripper_state == 1.0 else -1.0
        if self.lr_name == "left":
            self.robot_ee_path = self.robot.fl_ee_path
            self.robot_base_path = self.robot.fl_base_path
        else:
            self.robot_ee_path = self.robot.fr_ee_path
            self.robot_base_path = self.robot.fr_base_path
        self.T_base_ee_init = get_relative_transform(
            get_prim_at_path(self.robot_ee_path), get_prim_at_path(self.robot_base_path)
        )
        self.T_world_base_init = get_relative_transform(
            get_prim_at_path(self.robot_base_path), get_prim_at_path(self.task.root_prim_path)
        )
        self.T_world_ee_init = self.T_world_base_init @ self.T_base_ee_init
        self._ee_trans, self._ee_ori = self.get_ee_pose()
        self._ee_trans = self.tensor_args.to_device(self._ee_trans)
        self._ee_ori = self.tensor_args.to_device(self._ee_ori)
        self.update_pose_cost_metric()

    def plan_batch(self, ee_translation_goal_batch, ee_orientation_goal_batch, sim_js, js_names):
        t1 = time.time()
        torch.cuda.synchronize()
        sim_js_positions = (sim_js.positions)[np.newaxis, :]
        ik_goal = Pose(
            position=self.tensor_args.to_device(ee_translation_goal_batch),
            quaternion=self.tensor_args.to_device(ee_orientation_goal_batch),
            batch=CUROBO_BATCH_SIZE,
        )
        cu_js = JointState(
            position=self.tensor_args.to_device(np.tile(sim_js_positions, (CUROBO_BATCH_SIZE, 1))),
            velocity=self.tensor_args.to_device(np.tile(sim_js_positions, (CUROBO_BATCH_SIZE, 1))) * 0.0,
            acceleration=self.tensor_args.to_device(np.tile(sim_js_positions, (CUROBO_BATCH_SIZE, 1))) * 0.0,
            jerk=self.tensor_args.to_device(np.tile(sim_js_positions, (CUROBO_BATCH_SIZE, 1))) * 0.0,
            joint_names=js_names,
        )
        cu_js = cu_js.get_ordered_joint_state(self.cmd_js_names)
        result = self.motion_gen.plan_batch(cu_js, ik_goal, self.plan_config.clone())
        t2 = time.time()
        torch.cuda.synchronize()
        print("plan batch duration :", t2 - t1)
        return result

    def plan(self, ee_translation_goal, ee_orientation_goal, sim_js: JointState, js_names: list):
        if self.use_batch:
            ik_goal = Pose(
                position=self.tensor_args.to_device(ee_translation_goal.unsqueeze(0).expand(CUROBO_BATCH_SIZE, -1)),
                quaternion=self.tensor_args.to_device(ee_orientation_goal.unsqueeze(0).expand(CUROBO_BATCH_SIZE, -1)),
                batch=CUROBO_BATCH_SIZE,
            )
            cu_js = JointState(
                position=self.tensor_args.to_device(np.tile((sim_js.positions)[np.newaxis, :], (CUROBO_BATCH_SIZE, 1))),
                velocity=self.tensor_args.to_device(np.tile((sim_js.positions)[np.newaxis, :], (CUROBO_BATCH_SIZE, 1)))
                * 0.0,
                acceleration=self.tensor_args.to_device(
                    np.tile((sim_js.positions)[np.newaxis, :], (CUROBO_BATCH_SIZE, 1))
                )
                * 0.0,
                jerk=self.tensor_args.to_device(np.tile((sim_js.positions)[np.newaxis, :], (CUROBO_BATCH_SIZE, 1)))
                * 0.0,
                joint_names=js_names,
            )
            cu_js = cu_js.get_ordered_joint_state(self.cmd_js_names)
            return self.motion_gen.plan_batch(cu_js, ik_goal, self.plan_config.clone())
        ik_goal = Pose(
            position=self.tensor_args.to_device(ee_translation_goal),
            quaternion=self.tensor_args.to_device(ee_orientation_goal),
        )
        cu_js = JointState(
            position=self.tensor_args.to_device(sim_js.positions),
            velocity=self.tensor_args.to_device(sim_js.velocities) * 0.0,
            acceleration=self.tensor_args.to_device(sim_js.velocities) * 0.0,
            jerk=self.tensor_args.to_device(sim_js.velocities) * 0.0,
            joint_names=js_names,
        )
        cu_js = cu_js.get_ordered_joint_state(self.cmd_js_names)
        result = self.motion_gen.plan_single(cu_js.unsqueeze(0), ik_goal, self.plan_config.clone())
        if os.environ.get("INTERNDATA_DEBUG_PICK") == "1" and not bool(result.success.any()):
            try:
                print(
                    "[EXECDBG] action:", getattr(self, "_dbg_last_action", "?"),
                    "status:", result.status,
                    "target_p:",
                    np.round(np.asarray(ee_translation_goal.cpu() if hasattr(ee_translation_goal, "cpu")
                                        else ee_translation_goal), 4),
                    "pos_err:", result.position_error,
                    "rot_err:", result.rotation_error,
                    "js:", np.round(np.asarray(sim_js.positions.cpu() if hasattr(sim_js.positions, "cpu")
                                              else sim_js.positions), 3),
                    flush=True,
                )
            except Exception as exc:
                print("[EXECDBG] print failed:", exc, flush=True)
        return result

    def forward(self, manip_cmd, eps=5e-3):
        ee_trans, ee_ori = manip_cmd[0:2]
        gripper_fn = manip_cmd[2]
        self._dbg_last_action = gripper_fn
        params = dict(manip_cmd[3])
        eps = params.pop("_plan_epsilon", eps)
        hold_target = params.pop("_hold_last_target", False)
        assert hasattr(self, gripper_fn)
        method = getattr(self, gripper_fn)
        if gripper_fn in ["in_plane_rotation", "mobile_move", "dummy_forward"]:
            return method(**params)
        elif gripper_fn in ["update_pose_cost_metric", "update_specific", "update_gripper_collision"]:
            method(**params)
            return self.ee_forward(ee_trans, ee_ori, eps=eps, skip_plan=True, hold_last_target=hold_target)
        else:
            method(**params)
            return self.ee_forward(ee_trans, ee_ori, eps, hold_last_target=hold_target)

    def ee_forward(
        self,
        ee_trans: torch.Tensor | np.ndarray,
        ee_ori: torch.Tensor | np.ndarray,
        eps=1e-4,
        skip_plan=False,
        hold_last_target=False,
    ):
        ee_trans = self.tensor_args.to_device(ee_trans)
        ee_ori = self.tensor_args.to_device(ee_ori)
        sim_js = self.robot.get_joints_state()
        js_names = self.robot.dof_names
        plan_flag = torch.logical_or(
            torch.norm(self._ee_trans - ee_trans) > eps,
            torch.norm(self._ee_ori - ee_ori) > eps,
        )
        if not skip_plan:
            if plan_flag:
                # A failed new plan must hold the current pose. Replaying the
                # previous path from index zero can undo a completed skill.
                self.cmd_plan = None
                self._last_arm_target = None
                self.cmd_idx = 0
                self._step_idx = 0
                self.num_last_cmd = 0
                result = self.plan(ee_trans, ee_ori, sim_js, js_names)
                if self.use_batch:
                    if result.success.any():
                        self._ee_trans = ee_trans
                        self._ee_ori = ee_ori
                        paths = result.get_successful_paths()
                        position_filter_res = filter_paths_by_position_error(
                            paths, result.position_error[result.success]
                        )
                        rotation_filter_res = filter_paths_by_rotation_error(
                            paths, result.rotation_error[result.success]
                        )
                        filtered_paths = [
                            p for i, p in enumerate(paths) if position_filter_res[i] and rotation_filter_res[i]
                        ]
                        if len(filtered_paths) == 0:
                            filtered_paths = paths
                        sort_weights = self._get_sort_path_weights()  # pylint: disable=assignment-from-none
                        weights_arg = self.tensor_args.to_device(sort_weights) if sort_weights is not None else None
                        sorted_indices = sort_by_difference_js(filtered_paths, weights=weights_arg)
                        cmd_plan = self.motion_gen.get_full_js(paths[sorted_indices[0]])
                        self.idx_list = list(range(len(self.raw_js_names)))
                        self.cmd_plan = cmd_plan.get_ordered_joint_state(self.raw_js_names)
                        self.num_plan_failed = 0
                    else:
                        print("Plan did not converge to a solution.")
                        self.num_plan_failed += 1
                else:
                    succ = result.success.item()
                    if succ:
                        self._ee_trans = ee_trans
                        self._ee_ori = ee_ori
                        cmd_plan = result.get_interpolated_plan()
                        self.idx_list = list(range(len(self.raw_js_names)))
                        self.cmd_plan = cmd_plan.get_ordered_joint_state(self.raw_js_names)
                        self.num_plan_failed = 0
                    else:
                        print("Plan did not converge to a solution.")
                        self.num_plan_failed += 1
            if self.cmd_plan and self._step_idx % 1 == 0:
                cmd_state = self.cmd_plan[self.cmd_idx]
                art_action = ArticulationAction(
                    cmd_state.position.cpu().numpy(),
                    cmd_state.velocity.cpu().numpy() * 0.0,
                    joint_indices=self.idx_list,
                )
                self._last_arm_target = np.asarray(art_action.joint_positions).copy()
                self.cmd_idx += self.ds_ratio
                if self.cmd_idx >= len(self.cmd_plan):
                    self.cmd_idx = 0
                    self.cmd_plan = None
            else:
                self.num_last_cmd += 1
                target = getattr(self, "_last_arm_target", None) if hold_last_target else None
                art_action = ArticulationAction(joint_positions=(sim_js.positions[self.arm_indices] if target is None else target))
        else:
            target = getattr(self, "_last_arm_target", None) if hold_last_target else None
            art_action = ArticulationAction(joint_positions=(sim_js.positions[self.arm_indices] if target is None else target))
        self._step_idx += 1
        arm_action = art_action.joint_positions
        gripper_action = self.get_gripper_action()
        joint_positions = np.concatenate([arm_action, gripper_action])
        self._action = {
            "joint_positions": joint_positions,
            "joint_indices": np.concatenate([self.arm_indices, self.gripper_indices]),
            "lr_name": self.lr_name,
            "arm_action": arm_action,
            "gripper_action": gripper_action,
        }
        return self._action

    def get_gripper_action(self):
        return np.clip(self._gripper_state * self._gripper_joint_position, 0.0, 0.04)

    def get_ee_pose(self):
        sim_js = self.robot.get_joints_state()
        q_state = torch.tensor(sim_js.positions[self.arm_indices], **self.tensor_args.as_torch_dict()).reshape(1, -1)
        ee_pose = self.kin_model.get_state(q_state)
        return ee_pose.ee_position[0].cpu().numpy(), ee_pose.ee_quaternion[0].cpu().numpy()

    def get_armbase_pose(self):
        armbase_pose = get_relative_transform(
            get_prim_at_path(self.robot_base_path), get_prim_at_path(self.task.root_prim_path)
        )
        return pose_from_tf_matrix(armbase_pose)

    def forward_kinematic(self, q_state: np.ndarray):
        q_state = q_state.reshape(1, -1)
        q_state = self.tensor_args.to_device(q_state)
        out = self.kin_model.get_state(q_state)
        return out.ee_position[0].cpu().numpy(), out.ee_quaternion[0].cpu().numpy()

    def close_gripper(self):
        self._gripper_state = -1.0

    def set_gripper_width(self, width):
        """Command the opening of a symmetric prismatic gripper, in meters."""
        maximum = 2.0 * float(np.max(np.abs(self._gripper_joint_position)))
        self._gripper_state = float(np.clip(float(width) / maximum, 0.0, 1.0))

    def update_gripper_collision(self, width, measured=None):
        """Match the Lift2 planner's locked fingers to a commanded jaw width."""
        locked = self.robot_cfg["kinematics"].get("lock_joints", {})
        if not {"joint7", "joint8"}.issubset(locked):
            raise ValueError("Gripper collision synchronization requires joint7 and joint8")
        width = float(np.clip(width, 0., self.robot.cfg["gripper_max_width"]))
        measured = getattr(self, "_planning_gripper_measured", False) if measured is None else bool(measured)
        key = (width, measured)
        if getattr(self, "_planning_gripper_key", None) == key:
            return
        if not hasattr(self, "_planning_gripper_models"):
            self._planning_gripper_models = {}
        if key not in self._planning_gripper_models:
            config = deepcopy(self.robot_cfg)
            if measured and hasattr(self, "_measured_gripper_spheres"):
                config["kinematics"]["collision_spheres"].update(self._measured_gripper_spheres)
            config["kinematics"]["lock_joints"] = {**locked, "joint7": width / 2., "joint8": width / 2.}
            model = RobotConfig.from_dict(config, self.tensor_args)
            self._planning_gripper_models[key] = model.kinematics.kinematics_config
        self.motion_gen.kinematics.update_kinematics_config(self._planning_gripper_models[key])
        self._planning_gripper_width = width
        self._planning_gripper_measured = measured
        self._planning_gripper_key = key

    def open_gripper(self):
        self._gripper_state = 1.0
        if self.robot.cfg.get("rigid_food_grasp", False):
            from core.utils.grasp_constraint import release_food_grasp

            release_food_grasp(self)

    def attach_obj(self, obj_prim_path, link_name="attached_object"):
        paths = [obj_prim_path] if isinstance(obj_prim_path, str) else list(obj_prim_path)
        paths = list(dict.fromkeys(paths))
        if not paths:
            raise ValueError("Cannot attach an empty set of object meshes")
        missing = [p for p in paths if self.motion_gen.world_model.get_obstacle(p) is None]
        if missing:
            raise ValueError(f"Object meshes missing from planning world: {missing}")
        sim_js = self.robot.get_joints_state()
        cu_js = JointState(
            position=self.tensor_args.to_device(sim_js.positions[self.arm_indices]),
            velocity=self.tensor_args.to_device(sim_js.velocities[self.arm_indices]) * 0.0,
            acceleration=self.tensor_args.to_device(sim_js.velocities[self.arm_indices]) * 0.0,
            jerk=self.tensor_args.to_device(sim_js.velocities[self.arm_indices]) * 0.0,
            # Aloha USD uses fl_/fr_ prefixes; the single-arm model uses raw
            # names. The controller already defines this index/name mapping.
            joint_names=self.raw_js_names,
        )
        # attach_objects_to_robot computes FK directly from position; it does
        # not select an arm from a dual-arm state by joint_names. Match the
        # active planner's order before fitting cup spheres in the EE frame.
        cu_js = cu_js.get_ordered_joint_state(self.motion_gen.joint_names)
        attached = self.motion_gen.attach_objects_to_robot(
            cu_js,
            paths,
            link_name=link_name,
            sphere_fit_type=SphereFitType.VOXEL_VOLUME_SAMPLE_SURFACE,
            world_objects_pose_offset=Pose.from_list([0, 0, 0.01, 1, 0, 0, 0], self.tensor_args),
        )
        if not attached:
            raise RuntimeError(f"Could not attach object meshes: {paths}")
        self._attached_object_paths = paths
        if self.robot.cfg.get("rigid_food_grasp", False):
            from core.utils.grasp_constraint import add_food_grasp

            add_food_grasp(self, paths[0])

    def detach_obj(self):
        self.motion_gen.detach_object_from_robot()
        self._attached_object_paths = []
        if self.robot.cfg.get("rigid_food_grasp", False):
            from core.utils.grasp_constraint import release_food_grasp

            release_food_grasp(self)

    def update_specific(self, ignore_substring, reference_prim_path):
        obstacles = self.usd_help.get_obstacles_from_stage(
            ignore_substring=ignore_substring, reference_prim_path=reference_prim_path,
            ignore_paths=getattr(self, "_attached_object_paths", None),
        ).get_collision_check_world()
        if getattr(self, "_include_other_arm_obstacles", False):
            other_base = self.robot.fl_base_path if self.lr_name == "right" else self.robot.fr_base_path
            other_root = other_base.rsplit("/", 1)[0]
            other = self.usd_help.get_obstacles_from_stage(
                only_paths=[other_root], ignore_substring=["material"],
                reference_prim_path=reference_prim_path).get_collision_check_world()
            if not other.mesh:
                raise RuntimeError("The other arm has no planning collision meshes")
            obstacles.mesh = (obstacles.mesh or []) + other.mesh
            obstacles.objects = list(obstacles.objects) + other.mesh
            print("Other arm collision meshes:", len(other.mesh), other_root, flush=True)
        if self.motion_gen is not None:
            self.motion_gen.update_world(obstacles)
        self.world_cfg = obstacles

    def test_single_ik(self, ee_trans, ee_ori):
        assert not self.use_batch
        ik_goal = Pose(position=self.tensor_args.to_device(ee_trans), quaternion=self.tensor_args.to_device(ee_ori))
        result = self.ik_solver.solve_single(ik_goal)
        succ = result.success.item()
        if succ:  # pylint: disable=simplifiable-if-statement
            return True
        else:
            return False

    def test_batch_forward(self, ee_trans_batch_np, ee_ori_batch_np):
        ee_trans_batch = self.tensor_args.to_device(ee_trans_batch_np)
        ee_ori_batch = self.tensor_args.to_device(ee_ori_batch_np)
        sim_js = self.robot.get_joints_state()
        js_names = self.robot.dof_names
        result = self.plan_batch(ee_trans_batch, ee_ori_batch, sim_js, js_names)

        if os.environ.get("INTERNDATA_DEBUG_PICK") == "1":
            print("[PLANDBG] batch_targets:", ee_trans_batch_np.shape, flush=True)
            print("[PLANDBG] batch_success:", result.success, flush=True)
            print("[PLANDBG] batch_pos_err:", result.position_error, flush=True)
            print("[PLANDBG] batch_rot_err:", result.rotation_error, flush=True)

        return result

    def test_single_forward(self, ee_trans: np.ndarray, ee_ori: np.ndarray):
        assert ee_trans is not None and ee_ori is not None
        sim_js = self.robot.get_joints_state()
        js_names = self.robot.dof_names
        result = self.plan(ee_trans, ee_ori, sim_js, js_names)
        succ = result.success.item()
        if os.environ.get("INTERNDATA_DEBUG_PICK") == "1":
            print("[PLANDBG] target_p:", np.round(ee_trans, 4), flush=True)
            print(
                "[PLANDBG] succ:", succ,
                "status:", result.status,
                "pos_err:", result.position_error,
                "rot_err:", result.rotation_error,
                flush=True,
            )
            try:
                _pos = sim_js.positions
                _pos = _pos.cpu().numpy() if hasattr(_pos, "cpu") else np.asarray(_pos)
                print("[PLANDBG] sim_js_positions:", np.round(_pos, 3), flush=True)
            except Exception as exc:  # 诊断打印本身不能影响规划
                print("[PLANDBG] sim_js print failed:", exc, flush=True)
        if succ:
            print("Success")
            return 1
        print("Plan did not converge to a solution.")
        return 0

    def pre_forward(self, ee_trans: np.ndarray, ee_ori: np.ndarray, expected_js=None, ds_ratio=1):
        assert ee_trans is not None and ee_ori is not None
        ee_trans = self.tensor_args.to_device(ee_trans)
        ee_ori = self.tensor_args.to_device(ee_ori)
        sim_js = self.robot.get_joints_state()
        js_names = self.robot.dof_names
        if expected_js is not None:
            sim_js.positions[self.arm_indices] = expected_js
        result = self.plan(ee_trans, ee_ori, sim_js, js_names)
        if self.use_batch:
            if result.success.any():
                print("Success")
                cmd_plans = result.get_successful_paths()
                cmd_plan = random.choice(cmd_plans)
                cmd_plan = self.motion_gen.get_full_js(cmd_plan)
                cmd_plan = cmd_plan.get_ordered_joint_state(self.raw_js_names)
                N = cmd_plan.shape[0]
                dt = self.motion_gen.interpolation_dt
                self.ds_ratio = ds_ratio
                cmd_time = N * dt / self.plan_config.time_dilation_factor / self.ds_ratio
                return cmd_time, np.array(cmd_plan[-1].position.cpu())
            print("Plan did not converge to a solution.")
            self.num_plan_failed = 1000
            return 0, expected_js
        succ = result.success.item()
        if succ:
            print("Success")
            cmd_plan = result.get_interpolated_plan()
            N = cmd_plan.shape[0]
            dt = self.motion_gen.interpolation_dt
            self.ds_ratio = ds_ratio
            cmd_time = N * dt / self.plan_config.time_dilation_factor / self.ds_ratio
            return cmd_time, np.array(cmd_plan[-1].position.cpu())
        print("Plan did not converge to a solution.")
        self.num_plan_failed = 1000
        return 0, expected_js

    def in_plane_rotation(self, target_rotate: np.ndarray):
        action = deepcopy(self._action)
        last_arm = len(self.arm_indices) - 1
        action["joint_positions"][last_arm] -= target_rotate
        action["arm_action"][last_arm] -= target_rotate
        return action

    def mobile_move(self, target: np.ndarray, joint_indices: np.ndarray = None, initial_position: np.ndarray = None):
        return {
            "joint_positions": initial_position + target,
            "joint_indices": np.array(joint_indices),
            "lr_name": "whole",
        }

    def dummy_forward(self, arm_action, gripper_state, *args, **kwargs):
        if gripper_state == 1.0:
            self.open_gripper()
        elif gripper_state == -1.0:
            self.close_gripper()
        elif 0.0 < gripper_state < 1.0:
            self._gripper_state = float(gripper_state)
        else:
            raise NotImplementedError
        gripper_action = self.get_gripper_action()
        return {
            "joint_positions": np.concatenate([arm_action, gripper_action]),
            "joint_indices": np.concatenate([self.arm_indices, self.gripper_indices]),
            "lr_name": self.lr_name,
            "arm_action": arm_action,
            "gripper_action": gripper_action,
        }
