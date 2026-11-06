"""Collision-checked return to the measured starting state of the cup task."""
import numpy as np
from core.skills.base_skill import BaseSkill, register_skill
from core.utils.cup_plate_planning import prepare_return_home


@register_skill
class ReturnHome(BaseSkill):
    def __init__(self, robot, controller, task, cfg, *args, **kwargs):
        super().__init__()
        self.robot, self.controller, self.task, self.skill_cfg = robot, controller, task, cfg
        self.manip_list = []
        self.target_joints = None
        self.failed = False
        self.stable_count = 0
        self.tolerance = np.deg2rad(float(cfg.get('joint_tolerance_deg', .5)))
        self.required = int(cfg.get('stable_steps', 3))
        if self.tolerance <= 0 or self.required <= 0:
            raise ValueError('Return-home tolerance and stable steps must be positive')

    def simple_generate_manip_cmds(self):
        self.stable_count = 0
        prepare_return_home(self)

    def joint_error(self):
        actual = self.robot.get_joint_positions()[self.controller.arm_indices]
        return float(np.max(np.abs(actual-self.target_joints)))

    def is_done(self):
        if not self.manip_list:
            return True
        segment = self.manip_list[0][3]['segment']
        ready = segment.get('complete', False) and self.joint_error() <= self.tolerance
        self.stable_count = self.stable_count+1 if ready else 0
        if self.stable_count >= self.required:
            self.manip_list.clear()
        return not self.manip_list

    def is_feasible(self):
        return not self.failed and self.controller.num_plan_failed <= 5

    def is_success(self):
        success = (not self.failed and self.target_joints is not None and
                   self.stable_count >= self.required and self.joint_error() <= self.tolerance)
        print('Cup return home verification:', {'success': bool(success),
            'max_joint_error_deg': None if self.target_joints is None else float(np.rad2deg(self.joint_error())),
            'stable_frames': self.stable_count}, flush=True)
        return success
