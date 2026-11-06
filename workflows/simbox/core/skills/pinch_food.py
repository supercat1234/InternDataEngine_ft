"""Centered side pinch for the thin, flat egg-tart asset."""
import numpy as np
from core.skills.base_skill import register_skill
from core.skills.pick import Pick
from core.utils.usd_geom_utils import compute_bbox
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.transformations import get_relative_transform
from scipy.spatial.transform import Rotation


@register_skill
class PinchFood(Pick):
    def simple_generate_manip_cmds(self):
        super().simple_generate_manip_cmds()
        self.manip_list.extend([self.manip_list[-1]] * self.skill_cfg.get("settle_steps", 40))

    def sample_ee_pose(self, max_length=20):
        box = compute_bbox(self.pick_obj.prim)
        center = (np.asarray(box.min) + np.asarray(box.max)) / 2
        base_world = get_relative_transform(get_prim_at_path(self.task.root_prim_path),
                                            get_prim_at_path(self.controller.robot_base_path))
        center_base = base_world[:3, :3] @ center + base_world[:3, 3]
        poses = []
        for yaw in [0., 10., -10.]:
            for pitch in [16., 22., 10., 28.]:
                pose = np.eye(4)
                pose[:3, :3] = Rotation.from_euler("z", yaw, degrees=True).as_matrix() @ Rotation.from_euler(
                    "y", pitch, degrees=True).as_matrix()
                pose[:3, 3] = center_base - pose[:3, 0] * self.skill_cfg.get("pinch_tip_offset", 0.125)
                poses.append(pose)
        return np.asarray(poses)
