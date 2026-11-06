"""SplitAloha robot implementation - Dual-arm manipulator."""
import numpy as np

from core.robots.base_robot import register_robot
from core.robots.template_robot import TemplateRobot


# pylint: disable=line-too-long,unused-argument
@register_robot
class SplitAloha(TemplateRobot):
    """SplitAloha dual-arm robot with 6-DOF arms."""

    def _setup_joint_indices(self):
        self.left_joint_indices = self.cfg["left_joint_indices"]
        self.right_joint_indices = self.cfg["right_joint_indices"]
        self.left_gripper_indices = self.cfg["left_gripper_indices"]
        self.right_gripper_indices = self.cfg["right_gripper_indices"]
        self.body_indices = []
        self.head_indices = []
        self.lift_indices = []

    def _setup_paths(self):
        fl_ee_path = self.cfg["fl_ee_path"]
        fr_ee_path = self.cfg["fr_ee_path"]
        self.fl_ee_path = f"{self.robot_prim_path}/{fl_ee_path}"
        self.fr_ee_path = f"{self.robot_prim_path}/{fr_ee_path}"
        self.fl_base_path = f"{self.robot_prim_path}/{self.cfg['fl_base_path']}"
        self.fr_base_path = f"{self.robot_prim_path}/{self.cfg['fr_base_path']}"
        self.fl_hand_path = self.fl_ee_path
        self.fr_hand_path = self.fr_ee_path

    def _setup_gripper_keypoints(self):
        self.fl_gripper_keypoints = self.cfg["fl_gripper_keypoints"]
        self.fr_gripper_keypoints = self.cfg["fr_gripper_keypoints"]

    def _setup_collision_paths(self):
        self.fl_filter_paths_expr = [f"{self.robot_prim_path}/{p}" for p in self.cfg["fl_filter_paths"]]
        self.fr_filter_paths_expr = [f"{self.robot_prim_path}/{p}" for p in self.cfg["fr_filter_paths"]]
        self.fl_forbid_collision_paths = [f"{self.robot_prim_path}/{p}" for p in self.cfg["fl_forbid_collision_paths"]]
        self.fr_forbid_collision_paths = [f"{self.robot_prim_path}/{p}" for p in self.cfg["fr_forbid_collision_paths"]]

    def _get_gripper_state(self, gripper_home):
        return 1.0 if gripper_home and gripper_home[0] >= 0.05 else -1.0

    def _setup_joint_velocities(self):
        # SplitAloha has 12 joints for velocity control
        all_joint_indices = self.left_joint_indices + self.right_joint_indices
        if all_joint_indices:
            self._articulation_view.set_max_joint_velocities(
                [500.0] * 12,
                joint_indices=all_joint_indices,
            )

    def _set_initial_positions(self):
        positions = self.left_joint_home + self.right_joint_home + self.left_gripper_home + self.right_gripper_home
        indices = (
            self.left_joint_indices + self.right_joint_indices + self.left_gripper_indices + self.right_gripper_indices
        )
        if positions and indices:
            self._articulation_view.set_joint_positions(positions, joint_indices=indices)
            self._articulation_view.set_joint_position_targets(positions, joint_indices=indices)
        # Optional named joints avoid relying on a drive moving to its target during warmup.
        initial = getattr(self, "_episode_initial_joint_positions", self.cfg.get("initial_joint_positions", {}))
        if initial:
            extra_indices = [self.get_dof_index(name) for name in initial]
            values = list(initial.values())
            self._articulation_view.set_joint_positions(values, joint_indices=extra_indices)
            self._articulation_view.set_joint_position_targets(values, joint_indices=extra_indices)
        height_cfg = getattr(self, "_episode_base_height_cfg", None)
        if height_cfg:
            lift_index = self.get_dof_index(height_cfg["joint_name"])
            if not hasattr(self, "_base_height_original_gains"):
                kp, kd = self._articulation_view.get_gains(joint_indices=[lift_index])
                self._base_height_original_gains = (lift_index, kp.copy(), kd.copy())
            # Apply to the live physics view after EACH initialize/reset; changing
            # USD drive attributes alone can be overwritten by cached defaults.
            self._articulation_view.set_gains(
                kps=np.array([[height_cfg["drive_stiffness"]]], dtype=np.float32),
                kds=np.array([[height_cfg["drive_damping"]]], dtype=np.float32), joint_indices=[lift_index])
        elif hasattr(self, "_base_height_original_gains"):
            index, kp, kd = self._base_height_original_gains
            self._articulation_view.set_gains(kps=kp, kds=kd, joint_indices=[index])
