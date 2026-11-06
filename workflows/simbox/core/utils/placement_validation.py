"""Geometric and velocity checks for a cup supported by a circular plate (meters)."""

import numpy as np


def placement_is_stable(object_pose, support_pose, object_velocity, support_velocity, cfg):
    """Check the entire footprint, bottom height, upright axes and settling velocities.

    Poses are rigid 4x4 world transforms; velocities are (linear, angular) world vectors.
    The object origin is at its bottom unless object_bottom_offset is configured.
    """
    object_pose = np.asarray(object_pose)
    support_pose = np.asarray(support_pose)
    if not np.isfinite(object_pose).all() or not np.isfinite(support_pose).all():
        return False
    relative = np.linalg.inv(support_pose) @ object_pose
    usable_radius = float(cfg["support_radius"]) - float(cfg["object_radius"])
    if usable_radius <= 0 or np.linalg.norm(relative[:2, 3]) > usable_radius:
        return False
    bottom = relative[2, 3] + float(cfg.get("object_bottom_offset", 0.))
    if abs(bottom - float(cfg["support_surface_z"])) > float(cfg.get("height_tolerance", .004)):
        return False
    cos_tilt = np.cos(np.deg2rad(float(cfg.get("max_tilt_deg", 5.))))
    cos_support_tilt = np.cos(np.deg2rad(float(cfg.get('max_support_tilt_deg', cfg.get('max_tilt_deg', 5.)))))
    if relative[2, 2] < cos_tilt or support_pose[2, 2] < cos_support_tilt:
        return False
    for linear, angular in (object_velocity, support_velocity):
        if not np.isfinite(linear).all() or not np.isfinite(angular).all():
            return False
        if np.linalg.norm(linear) > float(cfg.get("max_linear_speed", .015)):
            return False
        if np.linalg.norm(angular) > float(cfg.get("max_angular_speed", .15)):
            return False
    return True
