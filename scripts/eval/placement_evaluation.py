"""Evaluation-only cup-bottom support checks and interpretable failure metrics."""
import numpy as np


DEFAULT_CRITERIA = {
    'plate_outer_radius': .077,
    'support_radius': .048,
    'object_bottom_radius': .0265,
    'support_surface_z': .005,
    'height_tolerance': .004,
    'max_tilt_deg': 8.,
    'success_max_tilt_deg': 15.,
    'max_support_tilt_deg': 5.,
    'max_linear_speed': .015,
    'max_angular_speed': .15,
    'tcp_clearance': .12,
    'stable_seconds': .5,
}


def evaluate_placement(cup_pose, plate_pose, cup_velocity, plate_velocity, tcp_position, cfg):
    cup_pose, plate_pose = np.asarray(cup_pose), np.asarray(plate_pose)
    vectors = [np.asarray(v) for v in (*cup_velocity, *plate_velocity, tcp_position)]
    if not all(np.isfinite(v).all() for v in (cup_pose, plate_pose, *vectors)):
        return {'placed': False, 'released': False, 'candidate': False,
                'strict_placed': False, 'strict_candidate': False,
                'failed_checks': ['nonfinite'], 'strict_failed_checks': ['nonfinite'],
                'metrics': {}, 'checks': {'finite': False}, 'strict_checks': {'finite': False}}
    relative = np.linalg.inv(plate_pose) @ cup_pose
    radius = cfg['object_bottom_radius']
    # Cup origin is the center of its bottom. When tilted, the lowest bottom
    # edge is below that origin; check the edge against the plate surface.
    cosine = float(np.clip(relative[2, 2], -1., 1.))
    lowest = float(relative[2, 3] - radius * np.sqrt(max(0., 1. - cosine**2)))
    metrics = {
        'center_distance_m': float(np.linalg.norm(relative[:2, 3])),
        'allowed_center_distance_m': cfg['plate_outer_radius'],
        'strict_allowed_center_distance_m': cfg['support_radius'] - radius,
        'bottom_center_z_m': float(relative[2, 3]),
        'lowest_bottom_z_m': lowest,
        'height_error_m': abs(lowest - cfg['support_surface_z']),
        'cup_tilt_deg': float(np.rad2deg(np.arccos(cosine))),
        'plate_tilt_deg': float(np.rad2deg(np.arccos(np.clip(plate_pose[2, 2], -1., 1.)))),
        'cup_linear_speed_m_s': float(np.linalg.norm(vectors[0])),
        'cup_angular_speed_rad_s': float(np.linalg.norm(vectors[1])),
        'plate_linear_speed_m_s': float(np.linalg.norm(vectors[2])),
        'plate_angular_speed_rad_s': float(np.linalg.norm(vectors[3])),
        'tcp_distance_m': float(np.linalg.norm(vectors[4] - cup_pose[:3, 3])),
    }
    checks = {
        'center_in_plate': metrics['center_distance_m'] <= metrics['allowed_center_distance_m'],
        'height': metrics['height_error_m'] <= cfg['height_tolerance'],
        'cup_tilt': metrics['cup_tilt_deg'] <= cfg['success_max_tilt_deg'],
        'plate_tilt': metrics['plate_tilt_deg'] <= cfg['max_support_tilt_deg'],
        'cup_linear_speed': metrics['cup_linear_speed_m_s'] <= cfg['max_linear_speed'],
        'cup_angular_speed': metrics['cup_angular_speed_rad_s'] <= cfg['max_angular_speed'],
        'plate_linear_speed': metrics['plate_linear_speed_m_s'] <= cfg['max_linear_speed'],
        'plate_angular_speed': metrics['plate_angular_speed_rad_s'] <= cfg['max_angular_speed'],
    }
    strict_checks = {**checks,
                     'cup_tilt': metrics['cup_tilt_deg'] <= min(cfg['max_tilt_deg'], cfg['success_max_tilt_deg']),
                     'footprint': metrics['center_distance_m'] <= metrics['strict_allowed_center_distance_m']}
    placed, strict_placed = all(checks.values()), all(strict_checks.values())
    released = metrics['tcp_distance_m'] > cfg['tcp_clearance']
    checks['tcp_clearance'] = released
    strict_checks['tcp_clearance'] = released
    return {'placed': placed, 'released': released, 'candidate': placed and released,
            'strict_placed': strict_placed, 'strict_candidate': strict_placed and released,
            'failed_checks': [name for name, passed in checks.items() if not passed],
            'strict_failed_checks': [name for name, passed in strict_checks.items() if not passed],
            'metrics': metrics, 'checks': checks, 'strict_checks': strict_checks}


def tcp_from_ee_world(ee_world, axis, offset):
    """EE and TCP positions must use the same world frame as cup/plate poses."""
    ee_world = np.asarray(ee_world)
    return ee_world[:3, 3] + ee_world[:3, {'x': 0, 'y': 1, 'z': 2}[axis]] * offset
