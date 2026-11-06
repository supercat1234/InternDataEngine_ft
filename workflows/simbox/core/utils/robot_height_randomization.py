"""Shared arm-platform height, sampled once and held through robot resets."""
import math
import random


def sample_robot_base_height(robot_cfg, rng=None):
    cfg = robot_cfg['base_height_randomization']
    nominal = float(cfg['nominal_offset'])
    reference = float(robot_cfg['initial_joint_positions'][cfg['joint_name']])
    low, high = map(float, cfg['offset_jitter_range'])
    stiffness, damping = float(cfg['drive_stiffness']), float(cfg['drive_damping'])
    if (not all(math.isfinite(v) for v in (nominal, reference, low, high, stiffness, damping))
            or low > high or stiffness <= 0 or damping <= 0):
        raise ValueError('Robot height bounds must be finite and ordered; drive gains must be positive')
    jitter = (rng or random).uniform(low, high) if cfg.get('enabled', False) else 0.
    return {'nominal_offset_m': nominal, 'jitter_m': jitter, 'height_offset_m': nominal + jitter,
            'reference_joint_position_m': reference, 'joint_name': cfg['joint_name'],
            'joint_position_m': reference + nominal + jitter}


def episode_initial_joint_positions(robot_cfg, use_mem=False):
    initial = dict(robot_cfg.get('initial_joint_positions', {}))
    cfg = robot_cfg.get('base_height_randomization')
    if cfg:
        state = cfg.get('mem_height') if use_mem and cfg.get('enabled', False) else None
        state = state or sample_robot_base_height(robot_cfg)
        cfg['mem_height'] = state
        initial[state['joint_name']] = float(state['joint_position_m'])
        initial.update({name: float(value) for name, value in cfg.get('fixed_base_joint_positions', {}).items()})
    return initial


def set_robot_height_limits(stage, robot, robot_cfg, initial):
    """Author only held-platform joints before physics rebuild; leave arm/wheel limits untouched."""
    from pxr import Usd, UsdPhysics
    cfg = robot_cfg.get('base_height_randomization')
    defaults = getattr(robot, '_base_height_joint_defaults', {})
    for attribute, value in defaults.values():
        attribute.Set(value)
    if not cfg:
        return
    held = {cfg['joint_name']: initial[cfg['joint_name']], **cfg.get('fixed_base_joint_positions', {})}
    joints = {p.GetName(): p for p in Usd.PrimRange(stage.GetPrimAtPath(robot.robot_prim_path))
              if p.IsA(UsdPhysics.Joint)}
    for name, target in held.items():
        prim = joints[name]
        angular = prim.IsA(UsdPhysics.RevoluteJoint)
        factor = 180. / math.pi if angular else 1.
        drive = 'angular' if angular else 'linear'
        values = {'physics:lowerLimit': (target-.0001)*factor,
                  'physics:upperLimit': (target+.0001)*factor,
                  f'drive:{drive}:physics:targetPosition': target*factor,
                  f'state:{drive}:physics:position': target*factor}
        for key, value in values.items():
            attr = prim.GetAttribute(key)
            if not attr or attr.Get() is None:
                continue
            path = str(attr.GetPath())
            if path not in defaults:
                defaults[path] = (attr, attr.Get())
            attr.Set(value)
    robot._base_height_joint_defaults = defaults
