"""Place a held object at a measured cavity target and verify containment."""
from copy import deepcopy

import numpy as np
from core.skills.base_skill import register_skill
from core.skills.place import Place
from core.utils.usd_geom_utils import compute_bbox
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.transformations import get_relative_transform, pose_from_tf_matrix, tf_matrix_from_pose
from scipy.spatial.transform import Rotation
from pxr import UsdPhysics, UsdShade


def cavity_metrics(food, container, bounds, target, tolerance=0.003, center_tolerance=0.025, require_center=True):
    """Bounds/target are meters in the container's unscaled pose frame."""
    frame = tf_matrix_from_pose(*container.get_world_pose())
    box = compute_bbox(food.prim)
    corners = np.array(np.meshgrid(*zip(box.min, box.max))).T.reshape(-1, 3)
    local = (corners - frame[:3, 3]) @ frame[:3, :3]
    lower, upper = np.asarray(bounds, dtype=float)
    minimum, maximum = local.min(axis=0), local.max(axis=0)
    center = (minimum + maximum) / 2
    error = float(np.linalg.norm(center[:2] - np.asarray(target)[:2]))
    inside = bool(np.all(minimum >= lower - tolerance) and np.all(maximum <= upper + tolerance))
    return {"bounds_min_m": minimum.tolist(), "bounds_max_m": maximum.tolist(),
            "center_m": center.tolist(), "target_m": list(target), "inside": inside,
            "center_error_xy_m": error, "centered": error <= center_tolerance,
            "success": inside and (not require_center or error <= center_tolerance)}


@register_skill
class CavityPlace(Place):
    def __init__(self, robot, controller, task, cfg, **kwargs):
        super().__init__(robot, controller, task, cfg, **kwargs)
        material = UsdShade.Material.Define(task.stage, f"{task.root_prim_path}/FoodGripMaterial")
        physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        physics.CreateStaticFrictionAttr(cfg.get("grip_friction", 1.2))
        physics.CreateDynamicFrictionAttr(cfg.get("grip_friction", 1.2))
        physics.CreateRestitutionAttr(0.0)
        pads = robot.fr_filter_paths_expr if controller.lr_name == "right" else robot.fl_filter_paths_expr
        for prim in [self.pick_obj.prim, *[task.stage.GetPrimAtPath(path) for path in pads]]:
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                material, bindingStrength=UsdShade.Tokens.strongerThanDescendants, materialPurpose="physics")
        self._last_command_pose = None
        self._release_corrections = 0
        self._release_stable_frames = 0

    def simple_generate_manip_cmds(self):
        try:
            super().simple_generate_manip_cmds()
            lift = self.skill_cfg.get("post_place_lift", 0.0)
            if lift > 0:
                # Lift clear of the released pastry before translating out;
                # a horizontal withdrawal can drag its rim along the pads.
                retreat_index = max(i for i, cmd in enumerate(self.manip_list) if cmd[2] == "open_gripper")
                lift_position = self.place_ee_trans + self.T_base_world[:3, :3] @ np.asarray(
                    self.skill_cfg.get("post_place_lift_vector", [0., 0., lift]))
                self.manip_list.insert(retreat_index,
                                       (lift_position, self.manip_list[retreat_index][1], "open_gripper", {}))
        except RuntimeError as error:
            self.planning_failure = str(error)
            self.manip_list = []
            print("Cavity planning rejected:", self.planning_failure, flush=True)

    def sample_gripper_place_traj(self):
        self.T_world_obj = tf_matrix_from_pose(*self.pick_obj.get_world_pose())
        self.T_world_ee = get_relative_transform(
            get_prim_at_path(self.robot_ee_path), get_prim_at_path(self.task.root_prim_path))
        self.T_base_world = get_relative_transform(
            get_prim_at_path(self.task.root_prim_path), get_prim_at_path(self.robot_base_path))
        self.T_obj_ee = np.linalg.inv(self.T_world_obj) @ self.T_world_ee
        self.T_world_container = tf_matrix_from_pose(*self.place_obj.get_world_pose())
        from core.utils.microwave_collision import measured_cavity
        lower, upper = measured_cavity(self.task.stage, self.place_obj)
        target = (lower + upper) / 2
        target[2] = lower[2] + self.skill_cfg.get("food_half_height", 0.0133) + 0.008
        self.skill_cfg["cavity_bounds"] = [lower.tolist(), upper.tolist()]
        self.skill_cfg["cavity_target"] = target.tolist()
        self.task.cfg["cavity_validation"]["bounds"] = self.skill_cfg["cavity_bounds"]
        self.task.cfg["cavity_validation"]["target"] = self.skill_cfg["cavity_target"]
        print("Measured cavity bounds and target m:", lower, upper, target, flush=True)
        target = np.asarray(self.skill_cfg["cavity_target"], dtype=float)
        orientations = []
        current_rotation = self.T_world_container[:3, :3].T @ self.T_world_obj[:3, :3]
        for euler in self.skill_cfg.get("object_euler_candidates", [self.skill_cfg["object_target_euler"]]):
            flat_rotation = Rotation.from_euler("xyz", euler, degrees=True).as_matrix()
            for yaw in self.skill_cfg.get("object_yaw_candidates", [0]):
                rotation = Rotation.from_euler("z", yaw, degrees=True).as_matrix() @ flat_rotation
                distance = Rotation.from_matrix(current_rotation.T @ rotation).magnitude()
                orientations.append((distance, yaw, euler, rotation))
        orientations.sort(key=lambda item: item[0])
        if self.skill_cfg.get("preserve_grasp_orientation", False):
            orientations.insert(0, (0., 0., "current grasp", current_rotation))
        self.controller.update_specific(
            ignore_substring=self.controller.ignore_substring + self.skill_cfg.get("ignore_substring", []),
            reference_prim_path=self.controller.reference_prim_path)
        # The insertion must be feasible from the end of the approach, not only
        # from the current grasp configuration.
        failures = []
        for _, yaw, euler, orientation in orientations:
            candidate_target = target.copy()
            if euler == "current grasp":
                current_box = compute_bbox(self.pick_obj.prim)
                candidate_target[2] = lower[2] + float(current_box.max[2] - current_box.min[2]) / 2 + 0.008
            object_goal = np.eye(4)
            object_goal[:3, :3] = orientation
            object_goal[:3, 3] = candidate_target
            object_pre = object_goal.copy()
            object_pre[:3, 3] += np.asarray(self.skill_cfg.get("cavity_approach", [0, -0.22, 0]))
            pre = self.T_base_world @ self.T_world_container @ object_pre @ self.T_obj_ee
            goal = self.T_base_world @ self.T_world_container @ object_goal @ self.T_obj_ee
            state = deepcopy(self.robot.get_joints_state())
            pre_result = self.controller.plan(*pose_from_tf_matrix(pre), state, self.robot.dof_names)
            if not bool(pre_result.success.any().item()):
                failures.append(f"yaw={yaw} approach={pre_result.status}")
                continue
            path = self.controller.motion_gen.get_full_js(pre_result.get_interpolated_plan())
            path = path.get_ordered_joint_state(self.controller.raw_js_names)
            state.positions[self.controller.arm_indices] = path.position[-1].detach().cpu().numpy()
            result = self.controller.plan(*pose_from_tf_matrix(goal), state, self.robot.dof_names)
            if bool(result.success.any().item()):
                break
            failures.append(f"yaw={yaw} insertion={result.status}")
        else:
            raise RuntimeError("No connected cavity path: " + "; ".join(failures))
        post = goal.copy()
        self.skill_cfg["cavity_target"] = candidate_target.tolist()
        self.task.cfg["cavity_validation"]["target"] = self.skill_cfg["cavity_target"]
        post[:3, 3] += self.T_base_world[:3, :3] @ np.asarray(self.skill_cfg["post_place_vector"])
        print(f"Cavity approach and connected insertion planned, flat-object euler={euler}, yaw={yaw}", flush=True)
        return [list(pose_from_tf_matrix(pose)) for pose in [pre, goal, post]]

    def is_subtask_done(self, t_eps=0.005, o_eps=0.08):
        p, q = self.controller.get_ee_pose()
        target_p, target_q, action = self.manip_list[0][:3]
        distance = float(np.linalg.norm(p - target_p))
        angle = float(2 * np.arccos(np.clip(abs(np.dot(q, target_q)), 0, 1)))
        reached = distance <= self.skill_cfg.get("cavity_pose_tolerance", 0.005) and angle <= 0.08
        if reached:
            key = (action, tuple(target_p))
            if key != self._last_command_pose:
                print("Cavity command reached:", action, "ee_error_m=", distance,
                      "food_world=", self.pick_obj.get_world_pose()[0].tolist(), flush=True)
                self._last_command_pose = key
        elif self.controller.num_last_cmd > 25:
            print("Cavity execution blocked:", action, "ee_error_m=", distance,
                  "orientation_error_rad=", angle, flush=True)
            self.controller.num_plan_failed = 11
        return reached

    def is_success(self, th=0.0):
        self.last_cavity_metrics = cavity_metrics(
            self.pick_obj, self.place_obj, self.skill_cfg["cavity_bounds"], self.skill_cfg["cavity_target"],
            self.skill_cfg.get("cavity_tolerance", 0.003), self.skill_cfg.get("center_tolerance", 0.025),
            self.skill_cfg.get("require_center", True))
        print("Cavity validation:", self.last_cavity_metrics, flush=True)
        return self.last_cavity_metrics["success"]

    def is_done(self):
        if not self.manip_list:
            return True
        if not self.is_subtask_done():
            return False
        before_release = (len(self.manip_list) > 1 and self.manip_list[0][2] == "close_gripper"
                          and self.manip_list[1][2] == "open_gripper")
        if before_release:
            metrics = cavity_metrics(self.pick_obj, self.place_obj, self.skill_cfg["cavity_bounds"],
                                     self.skill_cfg["cavity_target"],
                                     require_center=self.skill_cfg.get("require_center", True))
            if (metrics["center_error_xy_m"] > 0.008
                    and self._release_corrections < self.skill_cfg.get("max_center_corrections", 6)):
                delta = np.asarray(self.skill_cfg["cavity_target"]) - np.asarray(metrics["center_m"])
                delta[2] = 0
                length = np.linalg.norm(delta)
                delta *= min(0.9, 0.04 / max(length, 1e-8))
                container = tf_matrix_from_pose(*self.place_obj.get_world_pose())
                base_world = get_relative_transform(get_prim_at_path(self.task.root_prim_path),
                                                    get_prim_at_path(self.robot_base_path))
                p, q = self.controller.get_ee_pose()
                p = p + base_world[:3, :3] @ container[:3, :3] @ delta
                self.manip_list[0] = (p, q, "close_gripper", {})
                self._release_corrections += 1
                self._release_stable_frames = 0
                print("Cavity physical centering correction:", self._release_corrections,
                      "center_error_m=", metrics["center_error_xy_m"], "delta_local_m=", delta.tolist(), flush=True)
                return False
            self._release_stable_frames += 1
            if self._release_stable_frames < 10:
                return False
            if not metrics["success"]:
                print("Cavity centering did not settle within tolerance", metrics, flush=True)
                self.controller.num_plan_failed = 11
                return False
        self.manip_list.pop(0)
        return not self.manip_list
