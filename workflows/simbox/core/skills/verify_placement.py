"""Hold the arm and verify placement continuously after retreat/home has finished."""

import numpy as np
from core.skills.base_skill import BaseSkill, register_skill
from core.utils.placement_validation import placement_is_stable
from omni.isaac.core.utils.transformations import tf_matrix_from_pose


@register_skill
class VerifyPlacement(BaseSkill):
    def __init__(self, robot, controller, task, cfg, *args, **kwargs):
        super().__init__()
        self.robot, self.controller, self.skill_cfg = robot, controller, cfg
        self.pick_obj, self.place_obj = [task.objects[name] for name in cfg["objects"]]
        self.manip_list = []
        self.stable_count = 0
        self.stable_steps = int(cfg.get("stable_steps", 15))
        self.wait_steps = int(cfg.get("wait_steps", 60))
        if not 0 < self.stable_steps <= self.wait_steps:
            raise ValueError("Require 0 < stable_steps <= wait_steps")

    def simple_generate_manip_cmds(self):
        p, q = self.controller.get_ee_pose()
        joints = np.array(self.robot.get_joint_positions()[self.controller.arm_indices], copy=True)
        cmd = (p, q, "dummy_forward", {"arm_action": joints,
                                     "gripper_state": float(self.skill_cfg.get('gripper_state', 1.))})
        self.manip_list = [cmd] * self.wait_steps
        self.stable_count = 0

    def update(self):
        objects = (self.pick_obj, self.place_obj)
        stable = placement_is_stable(
            *(tf_matrix_from_pose(*obj.get_local_pose()) for obj in objects),
            *((obj.get_linear_velocity(), obj.get_angular_velocity()) for obj in objects),
            self.skill_cfg,
        )
        self.stable_count = self.stable_count + 1 if stable else 0

    def is_done(self):
        if self.manip_list:
            self.manip_list.pop(0)
        return not self.manip_list

    def is_success(self):
        success = not self.manip_list and self.stable_count >= self.stable_steps
        print(f"Final placement: stable_frames={self.stable_count}/{self.stable_steps}, success={success}")
        return success
