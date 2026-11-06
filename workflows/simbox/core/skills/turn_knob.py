"""Turn an existing revolute knob by robot contact, checking its measured angle."""
from copy import deepcopy
from itertools import product

import numpy as np
from core.skills.base_skill import BaseSkill, register_skill
from core.utils.knob_planning import joint_margins, knob_contact_pose, rotate_about_knob, select_knob_path
from curobo.types.math import Pose
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.transformations import get_relative_transform, pose_from_tf_matrix, tf_matrix_from_pose
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade
from scipy.spatial.transform import Rotation


@register_skill
class TurnKnob(BaseSkill):
    def __init__(self, robot, controller, task, cfg, **kwargs):
        super().__init__()
        self.robot, self.controller, self.task, self.skill_cfg = robot, controller, task, cfg
        self.name = cfg["name"]
        self.oven = task.objects[cfg["objects"][0]]
        self.joint_name = cfg.get("joint_name", "contact_link_0001_revolute")
        self.manip_list = []
        self.initial_angle = None
        self._last_reported_angle = None
        self._last_contact_command = None
        self.contact_view = task.artcontact_views[robot.name][controller.lr_name][cfg["objects"][0] + "_knob_fingers"]

    def _angle(self):
        return float(self.oven.get_joint_positions()[self.joint_index])

    def simple_generate_manip_cmds(self):
        self.joint_index = self.oven.dof_names.index(self.joint_name)
        self.door_index = self.oven.dof_names.index("contact_link_revolute")
        door_angle = abs(float(self.oven.get_joint_positions()[self.door_index]))
        door_limit = self.skill_cfg.get("max_door_angle", 0.08)
        if door_angle > door_limit:
            raise RuntimeError(f"Door angle {door_angle:.5f} exceeds knob clearance limit {door_limit:.5f} rad")
        self.initial_angle = self._angle()
        joint = self.task.stage.GetPrimAtPath(f"{self.oven.prim_path}/{self.joint_name}")
        if "gripper_force_limit" in self.skill_cfg:
            prefix = "fr" if self.controller.lr_name == "right" else "fl"
            for number in [7, 8]:
                gripper_joint = self.task.stage.GetPrimAtPath(f"{self.controller.robot_ee_path}/{prefix}_joint{number}")
                drive = UsdPhysics.DriveAPI.Apply(gripper_joint, "linear")
                drive.CreateMaxForceAttr(float(self.skill_cfg["gripper_force_limit"]))
                drive.CreateStiffnessAttr(float(self.skill_cfg.get("gripper_stiffness", 20000.)))
                drive.CreateDampingAttr(float(self.skill_cfg.get("gripper_damping", 100.)))
        if "joint_friction" in self.skill_cfg:
            # Passive resistance and a speed limit tame the source asset's
            # freely spinning dial; neither prescribes a position or velocity.
            joint_physics = PhysxSchema.PhysxJointAPI.Apply(joint)
            joint_physics.CreateJointFrictionAttr(float(self.skill_cfg["joint_friction"]))
            joint_physics.CreateMaxJointVelocityAttr(float(self.skill_cfg.get("max_joint_speed_deg", 90.)))
            print("Passive knob settings:", dict(friction=float(self.skill_cfg["joint_friction"]),
                                                max_speed_deg=float(self.skill_cfg.get("max_joint_speed_deg", 90.))), flush=True)
        body_path = joint.GetRelationship("physics:body0").GetTargets()[0]
        body = self.task.stage.GetPrimAtPath(body_path)
        body_transform = UsdGeom.Xformable(body).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        pivot = np.asarray(body_transform.Transform(Gf.Vec3d(*joint.GetAttribute("physics:localPos0").Get())))
        local_rotation = joint.GetAttribute("physics:localRot0").Get()
        r = Rotation.from_quat([local_rotation.GetReal(), *local_rotation.GetImaginary()], scalar_first=True)
        axis = np.eye(3)[{"X": 0, "Y": 1, "Z": 2}[joint.GetAttribute("physics:axis").Get()]]
        axis = np.asarray(body_transform.TransformDir(Gf.Vec3d(*(r.as_matrix() @ axis))))
        axis /= np.linalg.norm(axis)
        z_axis = np.array([0., 0., 1.])
        z_axis -= np.dot(z_axis, axis) * axis
        z_axis /= np.linalg.norm(z_axis)
        orientation = np.column_stack([axis, np.cross(z_axis, axis), z_axis])
        T_base_world = get_relative_transform(
            get_prim_at_path(self.task.root_prim_path), get_prim_at_path(self.controller.robot_base_path))
        tip_offset = self.skill_cfg.get("tip_offset", 0.133)
        current = np.linalg.inv(T_base_world) @ tf_matrix_from_pose(*self.controller.get_ee_pose())
        clear = current.copy()
        clear[:3, 3] -= axis * self.skill_cfg.get("initial_retreat", 0.16)

        # Give fingertip pads and the knob a shared, explicit grip material.
        # The knob itself remains passive: no position reset or motor target.
        material_path = f"{self.task.root_prim_path}/KnobGripMaterial"
        material = UsdShade.Material.Define(self.task.stage, material_path)
        physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        physics.CreateStaticFrictionAttr(self.skill_cfg.get("grip_friction", 1.2))
        physics.CreateDynamicFrictionAttr(self.skill_cfg.get("grip_friction", 1.2))
        physics.CreateRestitutionAttr(0.0)
        knob_body = joint.GetRelationship("physics:body1").GetTargets()[0]
        pads = self.robot.fl_filter_paths_expr if self.controller.lr_name == "left" else self.robot.fr_filter_paths_expr
        for path in [str(knob_body), *pads]:
            prim = self.task.stage.GetPrimAtPath(path)
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                material, bindingStrength=UsdShade.Tokens.strongerThanDescendants, materialPurpose="physics")

        ignores = deepcopy(self.controller.ignore_substring) + self.skill_cfg.get("ignore_substring", ["microwave"])
        self.controller._include_other_arm_obstacles = self.skill_cfg.get("include_other_arm_obstacles", False)
        self.controller.update_specific(ignore_substring=ignores,
                                        reference_prim_path=self.controller.reference_prim_path)
        # Sample grasps while keeping their fingertip reference at the same
        # spindle. Small tilts still turn around the real spindle, so the wrist
        # follows an arc rather than incorrectly rolling about a tilted axis.
        self.controller.update_pose_cost_metric(hold_vec_weight=None)
        turn_angle = self.skill_cfg.get("turn_angle", np.pi / 4)
        turn_samples = self.skill_cfg.get("turn_validation_steps", 4)
        approaches = self.skill_cfg.get("approach_candidates", [self.skill_cfg.get("approach_distance", 0.03), 0.015, 0.05])
        retreats = self.skill_cfg.get("retreat_candidates", [self.skill_cfg.get("retreat_distance", 0.08), 0.04, 0.02, 0.01])
        open_width = float(self.robot.cfg["gripper_max_width"])
        grasp_width = self.skill_cfg.get("grasp_width", 0.)
        release_width = self.skill_cfg.get("release_width", 0.041)
        phase_widths = {"approach": open_width, "contact": open_width,
                        "turn": grasp_width, "retreat": release_width}

        def prepare_phase(phase):
            self.controller.update_gripper_collision(phase_widths[phase], measured=True)

        limits = self.controller.motion_gen.kinematics.get_joint_limits()
        indices = [limits.joint_names.index(name) for name in self.controller.raw_js_names]
        lower, upper = limits.position.detach().cpu().numpy()[:, indices]

        def ik_margin(world_pose):
            p, q = pose_from_tf_matrix(T_base_world @ world_pose)
            result = self.controller.motion_gen.ik_solver.solve_single(
                Pose(position=self.controller.tensor_args.to_device(p),
                     quaternion=self.controller.tensor_args.to_device(q)), return_seeds=8)
            valid = result.success.detach().cpu().numpy().reshape(-1)
            if not valid.any():
                return None
            solutions = result.solution.detach().cpu().numpy().reshape(-1, len(lower))[valid]
            return max(joint_margins(solution, lower, upper)[0] for solution in solutions)

        candidates = []
        screened = 0
        rejected = dict(contact=0, approach=0, turn=0, retreat=0)
        rolls = self.skill_cfg.get("roll_candidates_deg", list(np.arange(-180., 180., 15.)))
        tilts = self.skill_cfg.get("tilt_candidates_deg", [0.])
        for offset, tilt in product(self.skill_cfg.get("tip_offset_candidates", [tip_offset]), tilts):
            for roll in rolls:
                screened += 1
                if (screened - 1) % len(rolls) == 0:
                    print("Knob IK screening progress:", {"screened": screened - 1,
                          "viable": len(candidates), "rejected": rejected,
                          "next_tip_offset_m": float(offset), "next_tilt_deg": float(tilt)}, flush=True)
                pose = knob_contact_pose(pivot, orientation, offset, roll, tilt)
                prepare_phase("contact")
                contact_margin = ik_margin(pose)
                if contact_margin is None:
                    rejected["contact"] += 1
                    continue
                viable_approaches = []
                for distance in approaches:
                    goal = pose.copy()
                    goal[:3, 3] -= axis * distance
                    if ik_margin(goal) is not None:
                        viable_approaches.append(distance)
                if not viable_approaches:
                    rejected["approach"] += 1
                    continue
                margins = [contact_margin]
                turns = [rotate_about_knob(pose, pivot, axis, turn_angle * fraction)
                         for fraction in np.linspace(0., 1., turn_samples + 1)[1:]]
                prepare_phase("turn")
                for goal in [pose, *turns]:
                    margin = ik_margin(goal)
                    if margin is None:
                        rejected["turn"] += 1
                        break
                    margins.append(margin)
                else:
                    prepare_phase("retreat")
                    viable_retreats = []
                    for distance in retreats:
                        goal = turns[-1].copy()
                        goal[:3, 3] -= axis * distance
                        if ik_margin(goal) is not None:
                            viable_retreats.append(distance)
                    if not viable_retreats:
                        rejected["retreat"] += 1
                        continue
                    candidates.append({"contact": pose, "roll_deg": float(roll), "tilt_deg": float(tilt),
                                       "tip_offset_m": float(offset),
                                       "ik_margin_fraction": min(margins),
                                       "approach_candidates_m": viable_approaches,
                                       "retreat_candidates_m": viable_retreats})
        candidates.sort(key=lambda candidate: candidate["ik_margin_fraction"], reverse=True)
        print("Knob IK screening:", {"sampled": screened, "viable": len(candidates), "rejected": rejected}, flush=True)
        initial_state = deepcopy(self.robot.get_joints_state())

        def plan_path(world_pose, start_q):
            state = deepcopy(initial_state)
            state.positions[self.controller.arm_indices] = start_q
            result = self.controller.plan(*pose_from_tf_matrix(T_base_world @ world_pose), state, self.robot.dof_names)
            if not bool(result.success.any().item()):
                return None
            path = result.get_interpolated_plan().get_ordered_joint_state(self.controller.raw_js_names)
            return path.position.detach().cpu().numpy()

        selected, report = select_knob_path(
            candidates, pivot, axis, approaches, retreats, turn_angle,
            turn_samples, initial_state.positions[self.controller.arm_indices],
            lower, upper, plan_path, self.skill_cfg.get("max_motion_candidates", 24), prepare_phase,
            lambda progress: print("Knob path search progress:", progress, flush=True))
        prepare_phase("approach")
        self.candidate_report = {"sampled_candidates": screened, "gripper_widths_m": phase_widths,
                                 "ik_rejections": rejected, **report}
        if selected is None:
            self.controller.num_plan_failed = 11
            self.manip_list = []
            print("Knob planning rejected: no connected approach/contact/turn/retreat path", self.candidate_report, flush=True)
            return
        self.candidate_report["selected"] = {key: value for key, value in selected.items()
                                              if key not in ["contact", "pre", "retreat"]}
        contact, pre = selected["contact"], selected["pre"]
        retreat_distance = selected["retreat_m"]
        print("Knob candidate summary:", self.candidate_report, flush=True)
        print("Knob connected path selected:", self.candidate_report["selected"], flush=True)
        cmds = []
        p, q = self.controller.get_ee_pose()
        cmds.append((p, q, "update_pose_cost_metric", {"hold_vec_weight": None}))
        cmds.append((p, q, "update_specific", {"ignore_substring": ignores,
                                              "reference_prim_path": self.controller.reference_prim_path}))
        cmds.append((p, q, "update_gripper_collision", {"width": open_width, "measured": True}))

        def move(world_pose, action, count=1, width=None):
            p, q = pose_from_tf_matrix(T_base_world @ world_pose)
            params = {"_plan_epsilon": self.skill_cfg.get("plan_epsilon", 0.0001), "_hold_last_target": True}
            if width is not None:
                params["width"] = width
            cmds.extend([(p, q, action, params)] * count)

        if self.skill_cfg.get("initial_retreat", 0.16) > 0:
            move(clear, "open_gripper")
        move(pre, "open_gripper")
        move(contact, "open_gripper", 10)
        move(contact, "update_gripper_collision", width=grasp_width)
        grasp_action = "set_gripper_width"
        move(contact, grasp_action, self.skill_cfg.get("grasp_steps", 45), width=grasp_width)
        angle = self.skill_cfg.get("turn_angle", np.pi / 4)
        steps = self.skill_cfg.get("turn_steps", 18)
        for step in range(1, steps + 1):
            twist = rotate_about_knob(contact, pivot, axis, angle * step / steps)
            move(twist, grasp_action, 2, width=grasp_width)
        move(twist, grasp_action, 20, width=grasp_width)
        move(twist, "update_gripper_collision", width=release_width)
        move(twist, "set_gripper_width", 10, width=release_width)
        retreat = twist.copy()
        retreat[:3, 3] -= axis * retreat_distance
        move(retreat, "set_gripper_width", width=release_width)
        move(retreat, "open_gripper", 20)
        move(retreat, "update_gripper_collision", width=open_width)
        p, q = pose_from_tf_matrix(T_base_world @ retreat)
        if self.robot.cfg.get("planner_gripper_collision_stage") == "turn_knob":
            cmds.append((p, q, "update_gripper_collision", {"width": open_width, "measured": False}))
        cmds.append((p, q, "update_specific", {"ignore_substring": deepcopy(self.controller.ignore_substring),
                                              "reference_prim_path": self.controller.reference_prim_path}))
        self.manip_list = cmds
        print("Knob contact plan:", {"pivot_world_m": pivot.tolist(), "axis_world": axis.tolist(),
                                     "initial_angle_rad": self.initial_angle, "turn_angle_rad": angle}, flush=True)

    def update(self):
        angle = self._angle()
        if self.manip_list:
            command = (self.manip_list[0][2], tuple(self.manip_list[0][1]))
            if command != self._last_contact_command:
                forces = np.abs(self.contact_view.get_contact_force_matrix()).sum(axis=-1).reshape(-1)
                print("Knob contact diagnostic:", {"action": command[0], "finger_force_sums": forces.tolist(),
                    "gripper_positions": self.robot.get_joint_positions()[self.controller.gripper_indices].tolist(),
                    "angle_rad": angle}, flush=True)
                self._last_contact_command = command
        if self._last_reported_angle is None or abs(angle - self._last_reported_angle) > 0.08:
            print(f"Measured knob angle: {angle:.5f} rad", flush=True)
            self._last_reported_angle = angle

    def is_feasible(self, th=10):
        return self.controller.num_plan_failed <= th

    def is_done(self):
        if not self.manip_list:
            return True
        p, q = self.controller.get_ee_pose()
        target_p, target_q = self.manip_list[0][:2]
        distance = np.linalg.norm(p - target_p)
        angle = 2 * np.arccos(np.clip(abs(np.dot(q, target_q)), 0, 1))
        if (distance <= self.skill_cfg.get("position_tolerance", 0.004)
                and angle <= self.skill_cfg.get("orientation_tolerance", 0.03)):
            self.manip_list.pop(0)
        elif self.controller.num_last_cmd > 40:
            print("Knob motion blocked:", distance, angle, flush=True)
            self.controller.num_plan_failed = 11
        return not self.manip_list

    def is_success(self):
        if self.initial_angle is None:
            return False
        delta = self._angle() - self.initial_angle
        door = float(self.oven.get_joint_positions()[self.door_index])
        return delta >= self.skill_cfg.get("success_angle", 0.55) and abs(door) <= 0.08
