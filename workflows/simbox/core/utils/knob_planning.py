"""Geometry and joint-limit scoring for contact-driven knob motion."""
import numpy as np
from scipy.spatial.transform import Rotation


def knob_contact_pose(pivot, orientation, tip_offset, roll_deg, tilt_deg=0.):
    pose = np.eye(4)
    pose[:3, :3] = orientation @ Rotation.from_euler("x", roll_deg, degrees=True).as_matrix() @ Rotation.from_euler(
        "y", tilt_deg, degrees=True).as_matrix()
    pose[:3, 3] = pivot - tip_offset * pose[:3, 0]
    return pose


def rotate_about_knob(pose, pivot, axis, angle):
    """Rotate both wrist position and orientation around the real spindle."""
    rotation = Rotation.from_rotvec(np.asarray(axis) * angle).as_matrix()
    result = pose.copy()
    result[:3, :3] = rotation @ pose[:3, :3]
    result[:3, 3] = pivot + rotation @ (pose[:3, 3] - pivot)
    return result


def joint_margins(positions, lower, upper):
    positions = np.asarray(positions)
    lower, upper = np.asarray(lower), np.asarray(upper)
    clearance = np.minimum(positions - lower, upper - positions)
    normalized = clearance / (upper - lower)
    return float(normalized.min()), float(clearance.min()), float(normalized.mean())


def select_knob_path(candidates, pivot, axis, approaches, retreats, turn_angle, turn_samples,
                     initial_q, lower, upper, plan_path, max_motion_candidates=24, prepare_phase=None,
                     on_candidate=None):
    """Rank connected paths, not just independently reachable goal poses.

    plan_path(world_pose, start_q) returns a joint trajectory or None. Candidates
    arrive sorted by their IK margin. Check the budgeted prefix; if none works,
    expand to the remaining candidates instead of dropping reachable layouts.
    """
    best, best_score = None, None
    evaluated, connected = 0, 0
    for candidate in candidates:
        if evaluated >= max_motion_candidates and best is not None:
            break
        evaluated += 1
        contact = candidate["contact"]
        turns = [rotate_about_knob(contact, pivot, axis, turn_angle * fraction)
                 for fraction in np.linspace(0., 1., turn_samples + 1)[1:]]
        for approach in candidate.get("approach_candidates_m", approaches):
            pre = contact.copy()
            pre[:3, 3] -= np.asarray(axis) * approach
            q, paths, keyposes = np.asarray(initial_q).copy(), [], []
            phases = [("approach", pre), ("contact", contact), *[("turn", pose) for pose in turns]]
            for phase, pose in phases:
                if prepare_phase is not None:
                    prepare_phase(phase)
                path = plan_path(pose, q)
                if path is None:
                    break
                paths.append(path)
                q = path[-1].copy()
                keyposes.append(q)
            else:
                for distance in candidate.get("retreat_candidates_m", retreats):
                    if prepare_phase is not None:
                        prepare_phase("retreat")
                    retreat = turns[-1].copy()
                    retreat[:3, 3] -= np.asarray(axis) * distance
                    path = plan_path(retreat, q)
                    if path is None:
                        continue
                    connected += 1
                    key_margin, key_rad, key_mean = joint_margins([*keyposes, path[-1]], lower, upper)
                    full_path = np.concatenate([*paths, path])
                    path_margin, path_rad, _ = joint_margins(full_path, lower, upper)
                    travel = float(np.abs(np.diff(full_path, axis=0)).sum())
                    # Initial home joints are shared by every path and may be
                    # near a limit. Prioritize task keyposes, then the full path.
                    score = (key_margin, path_margin, key_mean, -travel)
                    if best_score is None or score > best_score:
                        best_score = score
                        best = {**candidate, "pre": pre, "retreat": retreat,
                                "approach_m": approach, "retreat_m": distance,
                                "keypose_min_margin_fraction": key_margin,
                                "keypose_min_margin_rad": key_rad,
                                "trajectory_min_margin_fraction": path_margin,
                                "trajectory_min_margin_rad": path_rad}
        if on_candidate is not None:
            on_candidate({"evaluated": evaluated, "ik_viable": len(candidates), "connected_paths": connected,
                          "best_keypose_margin_rad": None if best is None else best["keypose_min_margin_rad"]})
    return best, {"ik_viable_candidates": len(candidates), "motion_candidates_evaluated": evaluated,
                  "connected_paths": connected}
