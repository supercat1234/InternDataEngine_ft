"""Opt-in cup/plate planning: body geometry, IK ranking, chained paths and replay.

Pure geometry functions are usable without Isaac/CUDA. CuRobo imports are local
to the adapter. Planning never teleports the simulated robot or the objects.
"""
from collections import Counter
import time

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp
from scipy.interpolate import PchipInterpolator


def pose_matrix(position, quaternion_wxyz):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(np.asarray(quaternion_wxyz)[[1, 2, 3, 0]]).as_matrix()
    result[:3, 3] = position
    return result


def pose_parts(transform):
    xyzw = Rotation.from_matrix(transform[:3, :3]).as_quat()
    return transform[:3, 3].copy(), xyzw[[3, 0, 1, 2]]


def joint_margins(joints, lower, upper):
    return np.minimum(np.asarray(joints) - lower, upper - np.asarray(joints))


def curve_poses(start, goal, up, bow=.01, count=25):
    """A short bowed Cartesian guide; curved motion remains explicitly allowed."""
    t = np.linspace(0., 1., count)
    poses = np.repeat(np.eye(4)[None], count, axis=0)
    poses[:, :3, 3] = ((1 - t[:, None]) * start[:3, 3] + t[:, None] * goal[:3, 3]
                      + (4 * t * (1 - t))[:, None] * np.asarray(up) * bow)
    poses[:, :3, :3] = Slerp([0., 1.], Rotation.from_matrix(np.stack([start[:3, :3], goal[:3, :3]])))(t).as_matrix()
    return poses


class TrajectoryResamplingError(ValueError):
    """This candidate cannot be resampled within the configured bounds."""


class PlanningBudgetExceeded(RuntimeError):
    """This bounded search ran out of time; not a proof of infeasibility."""


class PlanningBudget:
    def __init__(self, seconds, clock=None):
        self.clock = clock or time.monotonic
        if not np.isfinite(seconds) or seconds <= 0:
            raise ValueError('Planning time budget must be positive and finite')
        self.started = self.clock()
        self.deadline = self.started + seconds

    def remaining(self):
        return max(0., self.deadline-self.clock())

    def check(self, stage):
        if self.clock() >= self.deadline:
            raise PlanningBudgetExceeded(f'{stage}: elapsed={self.clock()-self.started:.2f}s')


def resample_joints(knots, max_step=.025, max_points=10000):
    """Smooth time scaling, with bounded joint increments and no endpoint overshoot."""
    knots = np.asarray(knots, dtype=float)
    if knots.ndim != 2 or len(knots) < 2 or knots.shape[1] < 1:
        raise ValueError("Joint knots must have shape (at least 2, at least 1)")
    if not np.isfinite(max_step) or max_step <= 0:
        raise ValueError("max_step must be finite and positive")
    if isinstance(max_points, bool) or not isinstance(max_points, int) or max_points < 12:
        raise ValueError("max_points must be an integer >= 12")
    if not np.isfinite(knots).all():
        raise TrajectoryResamplingError("Joint knots contain nonfinite positions")
    interpolator = PchipInterpolator(np.linspace(0., 1., len(knots)), knots, axis=0)
    # Each cubic's derivative is quadratic. Its maximum absolute value is at
    # an endpoint or its interior vertex; sparse/near-stationary IK knots can
    # have a sharp local derivative that total path length does not predict.
    a, b, c = 3 * interpolator.c[0], 2 * interpolator.c[1], interpolator.c[2]
    widths = np.diff(interpolator.x)[:, None]
    vertex = np.divide(-b, 2 * a, out=np.zeros_like(a), where=a != 0)
    vertex = np.clip(vertex, 0., widths)
    derivative_peak = max(float(np.max(np.abs(value))) for value in
                          [c, a * widths ** 2 + b * widths + c,
                           a * vertex ** 2 + b * vertex + c])
    # s(t)=3t²-2t³ has max s'(t)=1.5. By the mean value theorem this uniform
    # step count bounds every output joint increment without iterative guesses.
    required_intervals = 1.5 * derivative_peak / max_step
    if not np.isfinite(required_intervals) or required_intervals > max_points - 1:
        raise TrajectoryResamplingError(f"Joint resampling exceeds {max_points} points")
    count = max(12, int(np.ceil(required_intervals)) + 1)
    t = np.linspace(0., 1., count)
    path = interpolator(t * t * (3 - 2 * t))
    path[[0, -1]] = knots[[0, -1]]
    if not np.isfinite(path).all() or np.max(np.abs(np.diff(path, axis=0))) > max_step * (1 + 1e-9):
        raise TrajectoryResamplingError("Joint resampling failed its final increment check")
    return path


def trajectory_quality(poses, up):
    positions = poses[:, :3, 3]
    length = np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()
    direct = np.linalg.norm(positions[-1] - positions[0])
    height = positions @ up
    excess = max(0., float(height.max() - max(height[0], height[-1])))
    reversals = max(0., float(np.abs(np.diff(height)).sum() - abs(height[-1] - height[0])))
    rotation = Rotation.from_matrix(poses[:-1, :3, :3].transpose(0, 2, 1) @ poses[1:, :3, :3]).magnitude().sum()
    return {"length_m": float(length), "direct_m": float(direct), "height_excess_m": excess,
            "height_reversal_m": reversals, "rotation_rad": float(rotation),
            "cost": float(length + excess * 4 + reversals * 2 + rotation * .03)}


def cylinder_points(radius, height):
    angles = np.linspace(0., 2 * np.pi, 48, endpoint=False)
    return np.array([[r * np.cos(a), r * np.sin(a), z]
                     for z in np.linspace(0., height, 9) for r in [0., radius / 2, radius] for a in angles])


def palm_clear(ee_poses, base_object, points, bounds, clearance=.002):
    """Conservative measured-palm guard, independent of allowed finger contact."""
    base_points = points @ base_object[:3, :3].T + base_object[:3, 3]
    lower, upper = np.asarray(bounds)
    for pose in ee_poses:
        local = (base_points - pose[:3, 3]) @ pose[:3, :3]
        if np.any(np.all((local >= lower - clearance) & (local <= upper + clearance), axis=1)):
            return False
    return True


def branch_scores(joints, reference, lower, upper, preferred_margin):
    """Prefer interior IK branches, with extra weight on the constrained wrist J5."""
    margins = joint_margins(joints, lower, upper)
    shortage = np.maximum(preferred_margin - margins, 0.) / preferred_margin
    weights = np.ones(len(lower))
    weights[4] = 3.
    movement = np.mean(np.abs(np.asarray(joints) - reference) / (upper - lower), axis=-1)
    return movement + np.sum(shortage * shortage * weights, axis=-1)


def downward_mask(poses, up, minimum=1., maximum=30.):
    # EE +Z is the approach direction. Pregrasp = grasp - approach * offset.
    downward = -np.einsum("ni,i->n", poses[:, :3, 2], up)
    angles = np.rad2deg(np.arcsin(np.clip(downward, -1., 1.)))
    return (angles >= minimum - .01) & (angles <= maximum + .01)


def pregrasp_variants(grasps, offsets, up=None):
    offsets = np.asarray(offsets, dtype=float)
    if offsets.ndim != 1 or not len(offsets) or np.any(offsets <= 0):
        raise ValueError("Pregrasp offsets must be positive")
    poses = np.repeat(grasps, len(offsets), axis=0)
    distances = np.tile(offsets, len(grasps))
    if up is None:
        poses[:, :3, 3] -= poses[:, :3, 2] * distances[:, None]
    else:
        poses[:, :3, 3] += np.asarray(up)*distances[:, None]
    return poses, np.repeat(np.arange(len(grasps)), len(offsets)), distances


def vertical_entry_clear(poses, up, tcp_offset, lateral_tolerance=.002):
    """Keep the fingertip/TCP above one wall during the final vertical insertion."""
    up = np.asarray(up)/np.linalg.norm(up)
    tcp = poses[:, :3, 3]+poses[:, :3, 2]*tcp_offset
    relative = tcp-tcp[-1]
    heights = relative@up
    sideways = relative-heights[:, None]*up
    return bool(np.max(np.linalg.norm(sideways, axis=1)) <= lateral_tolerance
                and np.min(heights) >= -.001 and np.max(np.diff(heights), initial=0.) <= .001)


def pivot_tcp(pose, rotation, tcp_offset):
    """Change EE attitude without sweeping its TCP around the wrist origin."""
    result = pose.copy()
    tcp = pose[:3, 3] + pose[:3, 2]*tcp_offset
    result[:3, :3] = rotation
    result[:3, 3] = tcp - rotation[:, 2]*tcp_offset
    return result


def flexible_pregrasp_variants(grasps, offsets, up, tcp_offset, rotations_deg):
    poses, ids, distances = pregrasp_variants(grasps, offsets, up=up)
    rows, indices, heights = [], [], []
    for pose, index, distance in zip(poses, ids, distances):
        for angle in rotations_deg:
            rotation = pose[:3, :3] @ Rotation.from_euler('x', angle, degrees=True).as_matrix()
            rows.append(pivot_tcp(pose, rotation, tcp_offset))
            indices.append(index)
            heights.append(distance)
    return np.asarray(rows), np.asarray(indices), np.asarray(heights)


def lift_pose_variants(grasp, base_object, up, height, shifts, tilts_deg):
    """Move the cup base up/toward the arm and gently tilt about the jaw axis."""
    relative = np.linalg.inv(base_object) @ grasp
    toward = -base_object[:3, 3]
    toward -= np.dot(toward, up)*up
    toward /= max(np.linalg.norm(toward), 1e-9)
    axis = grasp[:3, 0]  # Piper's radial jaw-closing axis.
    poses = []
    for shift in shifts:
        for angle in tilts_deg:
            obj = base_object.copy()
            obj[:3, :3] = Rotation.from_rotvec(axis*np.deg2rad(angle)).as_matrix() @ base_object[:3, :3]
            obj[:3, 3] += up*height + toward*shift
            poses.append(obj @ relative)
    return np.asarray(poses)


def grasp_candidate_order(scores, mode="best", top_k=8, rng=None):
    """Randomize only the first K feasible IK pairs; retain ranked fallbacks."""
    scores = np.asarray(scores)
    order = np.argsort(scores, kind="stable")
    order = order[np.isfinite(scores[order])]
    if mode == "best":
        return order
    if mode != "random_top_k" or int(top_k) < 1:
        raise ValueError("Grasp selection must be best or random_top_k with positive top_k")
    count = min(int(top_k), len(order))
    order[:count] = (rng or np.random).permutation(order[:count])
    return order


def upright_place_targets(base_from_world, world_plate, object_from_ee, world_object, cfg):
    """Prefer upright, optionally sample small cup tilts, and search yaw."""
    relative = world_plate[:3, :3].T @ world_object[:3, :3]
    current_yaw = np.arctan2(relative[1, 0], relative[0, 0])
    step = float(cfg.get("chain_place_yaw_step_deg", 15.))
    if not 0 < step <= 180:
        raise ValueError("Placement yaw step must be in (0, 180]")
    offsets = sorted(np.arange(-180., 180., step), key=lambda a: (abs(a), a))
    triples = []
    up = base_from_world[:3, :3] @ np.array([0., 0., 1.])
    tilts = [np.eye(3)]
    tilt_bound = float(cfg.get('max_tilt_deg', 5.))
    if not np.isfinite(tilt_bound) or not 0 <= tilt_bound < 90:
        raise ValueError('Placement success tilt bound must be within [0, 90) degrees')
    for angle in cfg.get('chain_place_tilts_deg', []):
        if not 0 <= angle <= tilt_bound:
            raise ValueError('Placement tilt must be within the configured success tilt bound')
        if angle:
            tilts.extend(Rotation.from_rotvec(np.deg2rad(angle)*np.array([np.cos(a), np.sin(a), 0])).as_matrix()
                         for a in np.linspace(0, 2*np.pi, 4, endpoint=False))
    yaws = []
    for offset in offsets:
        for tilt in tilts:
            local = np.eye(4)
            local[:3, :3] = Rotation.from_euler("z", current_yaw + np.deg2rad(offset)).as_matrix() @ tilt
            # A tilted bottom disc needs height compensation to avoid table penetration.
            sin_tilt = np.linalg.norm(local[:2, 2])
            local[2, 3] = cfg["support_surface_z"] + cfg["place_z_offset"] + float(cfg.get('object_bottom_radius', .0265))*sin_tilt
            place = base_from_world @ world_plate @ local @ object_from_ee
            pre, retreat = place.copy(), place.copy()
            distance = cfg.get('chain_final_place_distance', .012) if cfg.get('chain_flexible_placement', False) else cfg['pre_place_z_offset']
            pre[:3, 3] += up * distance
            retreat[:3, 3] += base_from_world[:3, :3] @ np.asarray(cfg["post_place_vector"])
            triples.append((pre, place, retreat))
            yaws.append(offset)
    return np.asarray(triples), np.asarray(yaws)


def segment_finished(controller, segment, pose_flag):
    if segment.get("label") == "grasp" and segment.get("complete"):
        cfg = segment["planner"].cfg
        required = int(cfg.get("chain_grasp_settle_steps", 0))
        if required:
            # High-gain drives can report nonzero solver velocities while the
            # observed pose is stationary. Use consecutive measured positions
            # over physics steps to decide whether the approach has settled.
            joints = np.asarray(controller.robot.get_joints_state().positions[controller.arm_indices]).copy()
            previous = segment.get("settle_joints")
            speed = np.inf if previous is None else np.max(np.abs(joints - previous)) / controller.world.get_physics_dt()
            segment["settle_joints"] = joints
            stable = pose_flag and speed < float(cfg.get("chain_grasp_settle_speed", .05))
            segment["stable_frames"] = segment.get("stable_frames", 0) + 1 if stable else 0
            return segment["stable_frames"] >= required
    return bool(segment.get("complete", False) and pose_flag)


class CupPlateChainPlanner:
    def __init__(self, pick):
        from curobo.types.math import Pose
        from curobo.types.robot import JointState
        from omni.isaac.core.utils.prims import get_prim_at_path
        from omni.isaac.core.utils.transformations import get_relative_transform

        self.Pose, self.JointState = Pose, JointState
        self.pick, self.ctrl, self.cfg = pick, pick.controller, pick.skill_cfg
        self.open_gripper_state = float(self.cfg.get("pre_grasp_width", self.ctrl.robot.cfg["gripper_max_width"])) / float(self.ctrl.robot.cfg["gripper_max_width"])
        if not 0 < self.open_gripper_state <= 1:
            raise ValueError("pre_grasp_width must be positive and fit the robot gripper")
        self.task = pick.task
        self.place_cfg = None
        for group in self.task.cfg["skills"]:
            for sequence in group.get(self.ctrl.name, []):
                for cfg in sequence.get(self.ctrl.lr_name, []):
                    if cfg["name"] == "place" and cfg["objects"][0] == pick.pick_obj.name:
                        self.place_cfg = cfg
                        break
        if self.place_cfg is None:
            raise ValueError("Joint chain planning needs a following place skill for the same object")
        self.plate = self.task.objects[self.place_cfg["objects"][1]]
        self.base_from_world = get_relative_transform(
            get_prim_at_path(self.task.root_prim_path), get_prim_at_path(self.ctrl.robot_base_path))
        self.up = self.base_from_world[:3, :3] @ np.array([0., 0., 1.])
        self.world_object = pose_matrix(*pick.pick_obj.get_local_pose())
        self.world_plate = pose_matrix(*self.plate.get_local_pose())
        print('Cup layout snapshot:', {
            'cup_pose_wxyz': [part.tolist() for part in pose_parts(self.world_object)],
            'plate_pose_wxyz': [part.tolist() for part in pose_parts(self.world_plate)]}, flush=True)
        self.base_object = self.base_from_world @ self.world_object
        self.cup_points = cylinder_points(float(self.cfg.get("cup_radius", .0375)), float(self.cfg.get("cup_height", .08)))
        self.palm_bounds = np.asarray(self.cfg.get("palm_collision_bounds",
            [[-.0829, -.0334, -.0001], [.0828, .0921, .0733]]))
        limits = self.ctrl.motion_gen.kinematics.get_joint_limits().position.cpu().numpy()
        self.lower, self.upper = limits[0], limits[1]
        self.minimum_margin = np.deg2rad(float(self.cfg.get("chain_min_margin_deg", 1.)))
        self.preferred_margin = np.deg2rad(float(self.cfg.get("chain_preferred_margin_deg", 5.)))
        self.stats = Counter()
        self.virtual_attachment = False
        self._attachment_generation = 0
        self._place_endpoint_cache = {}
        self.begin_budget('initial_chain', float(self.cfg.get('chain_planning_timeout_s', 45.)))
        self.ctrl.update_pose_cost_metric()
        # Obtain all mesh geometry once, then exclude only the cup during contact.
        self.ctrl.update_specific(self.ctrl.ignore_substring, self.ctrl.reference_prim_path)
        self.meshes = [self.ctrl.motion_gen.world_model.get_obstacle(path)
                       for path in pick.pick_obj.mesh_prim_paths]
        if any(mesh is None for mesh in self.meshes):
            raise ValueError("Missing cup meshes in planning world")
        self.ctrl.update_specific(self.ctrl.ignore_substring + [pick.pick_obj.name], self.ctrl.reference_prim_path)
        self.ctrl.ik_solver.update_world(self.ctrl.world_cfg)
        self._ik_world_cfg = self.ctrl.world_cfg

    def state(self, joints):
        return self.JointState.from_position(self.ctrl.tensor_args.to_device(np.asarray(joints).reshape(1, -1)),
                                            joint_names=self.ctrl.motion_gen.joint_names)

    def current_joints(self):
        return np.asarray(self.ctrl.robot.get_joints_state().positions[self.ctrl.arm_indices]).copy()

    def begin_budget(self, phase, seconds):
        self._planning_budget = PlanningBudget(seconds)
        self._budget_phase = phase
        self._last_budget_log = self._planning_budget.started
        print('Cup planning started:', {'phase': phase, 'budget_s': seconds}, flush=True)

    def check_budget(self, stage):
        budget = getattr(self, '_planning_budget', None)
        if budget is None:
            return
        budget.check(stage)
        now = budget.clock()
        if now-self._last_budget_log >= float(self.cfg.get('chain_progress_interval_s', 5.)):
            self._last_budget_log = now
            print('Cup planning progress:', {'phase': self._budget_phase, 'stage': stage,
                'elapsed_s': round(now-budget.started, 1), 'remaining_s': round(budget.remaining(), 1),
                'ik_targets_solved': self.stats['ik_targets_solved'],
                'endpoint_cache_hits': self.stats['endpoint_cache_hits']}, flush=True)

    def fk(self, joints):
        position, orientation = self.ctrl.forward_kinematic(np.asarray(joints))
        return pose_matrix(position, orientation)

    def ik_solutions(self, poses, references):
        """Fixed-size batches preserve the IK solver's CUDA graph shapes."""
        batch = int(self.cfg.get("chain_ik_batch_size", 64))
        branches = int(self.cfg.get("chain_ik_branches", 4))
        solutions_list, valid_list = [], []
        self.sync_ik_world()
        for start in range(0, len(poses), batch):
            self.check_budget('ik_batch')
            chunk = poses[start:start + batch]
            padded = np.concatenate([chunk, np.repeat(chunk[-1: ], batch - len(chunk), axis=0)])
            positions, quaternions = zip(*(pose_parts(pose) for pose in padded))
            goal = self.Pose(position=self.ctrl.tensor_args.to_device(np.asarray(positions)),
                             quaternion=self.ctrl.tensor_args.to_device(np.asarray(quaternions)))
            result = self.ctrl.ik_solver.solve_batch(goal, return_seeds=branches,
                retract_config=self.ctrl.tensor_args.to_device(np.concatenate([
                    references[start:start + len(chunk)],
                    np.repeat(references[start + len(chunk) - 1:start + len(chunk)], batch - len(chunk), axis=0)])))
            self.stats['ik_targets_solved'] += len(chunk)
            self.check_budget('ik_batch_complete')
            solutions = result.solution.cpu().numpy()[:len(chunk)]
            valid = result.success.cpu().numpy()[:len(chunk)]
            for field in ("position_error", "rotation_error"):
                value = float(getattr(result, field).min().item())
                key = "ik_"+field+"_min"
                self.stats[key] = min(self.stats.get(key, float("inf")), value)
            margin_ok = np.min(joint_margins(solutions, self.lower, self.upper), axis=-1) >= self.minimum_margin
            self.stats["ik_solver_success"] += int(valid.sum())
            self.stats["ik_margin_rejected"] += int((valid & ~margin_ok).sum())
            valid &= margin_ok
            solutions_list.append(solutions)
            valid_list.append(valid)
        return np.concatenate(solutions_list), np.concatenate(valid_list)

    def sync_ik_world(self):
        # update_specific/update replace WorldConfig after a real stage update.
        # Reusing it for every IK batch only reloads the same table meshes.
        if getattr(self, '_ik_world_cfg', None) is not self.ctrl.world_cfg:
            self.ctrl.ik_solver.update_world(self.ctrl.world_cfg)
            self._ik_world_cfg = self.ctrl.world_cfg

    def ik(self, poses, reference):
        references = np.broadcast_to(reference, (len(poses), len(self.lower))).copy()
        solutions, valid = self.ik_solutions(poses, references)
        return self.rank_ik(solutions, valid, references)

    def rank_ik(self, solutions, valid, references):
        values = branch_scores(solutions, references[:, None, :], self.lower, self.upper, self.preferred_margin)
        values[~valid] = np.inf
        indices = np.argmin(values, axis=1)
        selected = solutions[np.arange(len(solutions)), indices]
        scores = values[np.arange(len(solutions)), indices]
        self.stats["ik_targets"] += len(solutions)
        self.stats["ik_valid"] += int(np.isfinite(scores).sum())
        return selected, scores

    def placement_ik(self, poses, reference):
        """Reuse raw branches while world/attachment/targets are unchanged.

        Failed branches are a bounded search result, not a global no-IK claim.
        Every reused branch is rescored for the new start and paths still get
        the full collision checks. World/attachment changes invalidate cache.
        """
        if not hasattr(self, '_place_endpoint_cache'):
            return self.ik(poses, reference)
        self.check_budget('placement_endpoints')
        context = getattr(self, '_place_cache_context', (None, None))
        generation = self._attachment_generation
        if context[0] is not self.ctrl.world_cfg or context[1] != generation:
            self._place_endpoint_cache.clear()
            self._place_cache_context = (self.ctrl.world_cfg, generation)
        key = (poses.shape, poses.tobytes())
        references = np.broadcast_to(reference, (len(poses), len(self.lower))).copy()
        if key not in self._place_endpoint_cache:
            self._place_endpoint_cache[key] = self.ik_solutions(poses, references)
        else:
            self.stats['endpoint_cache_hits'] += 1
            self.stats['ik_targets_reused'] += len(poses)
        return self.rank_ik(*self._place_endpoint_cache[key], references)

    def fk_path(self, path):
        state = self.ctrl.kin_model.get_state(self.ctrl.tensor_args.to_device(path))
        positions, quaternions = state.ee_position.cpu().numpy(), state.ee_quaternion.cpu().numpy()
        return np.array([pose_matrix(p, q) for p, q in zip(positions, quaternions)])

    def valid_path(self, path, label):
        if np.min(joint_margins(path, self.lower, self.upper)) < self.minimum_margin * .5:
            return False
        # This rechecks every resampled curve against the existing CuRobo
        # collision and bound constraints; it does not bypass collision planning.
        state = self.JointState.from_position(self.ctrl.tensor_args.to_device(path),
                                              joint_names=self.ctrl.motion_gen.joint_names)
        if not bool(self.ctrl.motion_gen.check_constraints(state).feasible.all()):
            return False
        base_label = label.split("_refresh")[0]
        if base_label in ["pregrasp", "approach", "grasp"]:
            if not palm_clear(self.fk_path(path), self.base_object, self.cup_points, self.palm_bounds,
                              float(self.cfg.get("palm_clearance", .002))):
                return False
        if base_label == "grasp" and self.cfg.get("grasp_mode") == "rim_pinch":
            if not vertical_entry_clear(self.fk_path(path), self.up,
                    float(self.cfg.get("tcp_offset", self.ctrl.robot.tcp_offset)),
                    float(self.cfg.get("chain_entry_lateral_tolerance", .002))):
                self.stats["grasp:nonvertical_entry"] += 1
                return False
        relative = getattr(self, 'carry_object_from_ee', None)
        if relative is not None and base_label in ['lift', 'preplace', 'place']:
            objects = self.fk_path(path) @ np.linalg.inv(relative)
            cosines = objects[:, :3, 2] @ self.up
            if base_label == 'lift':
                limit = float(self.cfg.get('chain_lift_max_tilt_deg', 10.))
            elif base_label == 'place':
                limit = float((getattr(self, 'place_cfg', None) or {}).get('max_tilt_deg', 5.))
            else:
                limit = float(self.cfg.get('chain_carry_max_tilt_deg', 60.))
            if not np.isfinite(limit) or not 0 <= limit <= 180:
                raise ValueError('Cup path tilt limit must be within 0..180 degrees')
            if np.any(cosines < np.cos(np.deg2rad(limit))):
                self.stats[base_label+':cup_tilt'] += 1
                return False
        return True

    def guided_path(self, start, goal, label, bow):
        self.check_budget(label+':guide')
        start_pose, goal_pose = self.fk(start), self.fk(goal)
        poses = curve_poses(start_pose, goal_pose, self.up, bow)
        references = np.linspace(start, goal, len(poses))
        solutions, valid = self.ik_solutions(poses, references)
        knots = [np.asarray(start)]
        for i in range(1, len(poses) - 1):
            costs = branch_scores(solutions[i], knots[-1], self.lower, self.upper, self.preferred_margin)
            costs[~valid[i]] = np.inf
            branch = int(np.argmin(costs))
            if not np.isfinite(costs[branch]) or np.max(np.abs(solutions[i, branch] - knots[-1])) > .3:
                return None
            knots.append(solutions[i, branch])
        if np.max(np.abs(goal - knots[-1])) > .3:
            return None
        knots.append(np.asarray(goal))
        try:
            path = resample_joints(np.array(knots), float(self.cfg.get("chain_max_joint_step", .025)))
        except TrajectoryResamplingError as error:
            self.stats[f"{label}:resample_rejected"] += 1
            print(f"Cup guide rejected ({label}): {error}", flush=True)
            return None
        if not self.valid_path(path, label):
            return None
        return path

    def plan_segment(self, start, goal, label):
        self.check_budget(label)
        # Prefer short, shallow curves. Curves are allowed; excessive height
        # reversals and approach-orientation excursions are not.
        base_label = label.split("_refresh")[0]
        candidates = []
        for bow in self.cfg.get("chain_curve_bows", [.0, .01]):
            path = self.guided_path(start, goal, label, float(bow))
            if path is not None:
                poses = self.fk_path(path)
                quality = trajectory_quality(poses, self.up)
                if base_label != "pregrasp" and quality["height_excess_m"] > float(self.cfg.get("chain_max_height_excess", .025)):
                    continue
                if base_label == "grasp":
                    excursion = Rotation.from_matrix(poses[-1, :3, :3].T @ poses[:, :3, :3]).magnitude().max()
                    if excursion > np.deg2rad(float(self.cfg.get("chain_approach_rotation_deg", .75))):
                        continue
                candidates.append((quality["cost"], path, quality))
        if candidates:
            _, path, quality = min(candidates, key=lambda candidate: candidate[0])
            return {"path": path, "start": np.asarray(start).copy(), "goal": self.fk(path[-1]),
                    "label": label, "complete": False, "planner": self, "quality": quality}
        config = self.ctrl.plan_config.clone()
        config.max_attempts = int(self.cfg.get("chain_plan_attempts", 4))
        config.enable_graph_attempt = min(config.max_attempts, 2)
        config.timeout = float(self.cfg.get("chain_plan_timeout", 5.))
        budget = getattr(self, '_planning_budget', None)
        if budget is not None:
            self.check_budget(label+':fallback')
            config.timeout = min(config.timeout, budget.remaining())
        try:
            result = self.ctrl.motion_gen.plan_single_js(self.state(start), self.state(goal), config)
        finally:
            # Upstream check_start_state can return on world collision before
            # restoring self-collision cost. Never let that leak to later paths.
            self.ctrl.motion_gen.rollout_fn.robot_self_collision_constraint.enable_cost()
        self.check_budget(label+':fallback_complete')
        if not bool(result.success.any()):
            self.stats[f"{label}:{result.status}"] += 1
            return None
        path = result.get_interpolated_plan().position.cpu().numpy().copy()
        if path.ndim == 3:
            path = path[0]
        # Check every point, not just the IK goal. This is candidate rejection,
        # never a modification of USD/URDF limits or a clamp of a real state.
        if not self.valid_path(path, label):
            self.stats[f"{label}:path_margin"] += 1
            return None
        poses = self.fk_path(path)
        quality = trajectory_quality(poses, self.up)
        if base_label != "pregrasp" and quality["height_excess_m"] > float(self.cfg.get("chain_max_height_excess", .025)):
            self.stats[f"{label}:height_excess"] += 1
            return None
        if base_label == "grasp":
            excursion = Rotation.from_matrix(poses[-1, :3, :3].T @ poses[:, :3, :3]).magnitude().max()
            if excursion > np.deg2rad(float(self.cfg.get("chain_approach_rotation_deg", .75))):
                self.stats[f"{label}:orientation_excursion"] += 1
                return None
        return {"path": path, "start": np.asarray(start).copy(), "goal": self.fk(path[-1]),
                "label": label, "complete": False, "planner": self, "quality": quality}

    def attach_virtual(self, grasp_joints):
        from curobo.geom.sphere_fit import SphereFitType

        attached = self.ctrl.motion_gen.attach_external_objects_to_robot(
            self.state(grasp_joints), self.meshes,
            sphere_fit_type=SphereFitType.VOXEL_VOLUME_SAMPLE_SURFACE,
            world_objects_pose_offset=self.Pose.from_list([0, 0, .01, 1, 0, 0, 0], self.ctrl.tensor_args))
        if not attached:
            raise RuntimeError("Virtual carried-cup geometry could not be attached")
        self.virtual_attachment = True
        self._attachment_generation += 1

    def clear_virtual(self):
        if self.virtual_attachment:
            self.ctrl.motion_gen.detach_object_from_robot()
            self.virtual_attachment = False
            self._attachment_generation += 1

    def place_chain(self, start, object_from_ee, world_object=None, grasp_joints=None):
        self.check_budget('place_endpoints')
        self.carry_object_from_ee = object_from_ee.copy()
        if world_object is not None:
            # A physical slip/tracking error can invalidate the actual start.
            # Trying many goals cannot repair a colliding start configuration.
            try:
                start_ok, start_status = self.ctrl.motion_gen.check_start_state(self.state(start))
            finally:
                self.ctrl.motion_gen.rollout_fn.robot_self_collision_constraint.enable_cost()
            if not start_ok:
                self.stats['preplace:'+str(start_status)] += 1
                print('Cup actual carry start rejected:', str(start_status), flush=True)
                return None
        targets, yaws = upright_place_targets(self.base_from_world, self.world_plate, object_from_ee,
                                             self.world_object if world_object is None else world_object,
                                             self.place_cfg)
        # Screen the two held-cup endpoints before spending work on the empty
        # wrist. A failed preplace/place pair cannot be rescued by retreat IK.
        if self.place_cfg.get('chain_flexible_placement', False):
            pair_q, pair_scores = self.placement_ik(targets[:, :2].reshape(-1, 4, 4), start)
            qs = np.zeros((len(targets), 3, len(start)))
            scores = np.full((len(targets), 3), np.inf)
            qs[:, :2] = pair_q.reshape(len(targets), 2, -1)
            scores[:, :2] = pair_scores.reshape(len(targets), 2)
            eligible = np.flatnonzero(np.isfinite(scores[:, :2]).all(1))
            angles = self.place_cfg.get('chain_retreat_rotations_deg', [0, -5, 5, -10, 10, -20, 20])
            if len(eligible):
                retreat_poses = np.array([pivot_tcp(triple[2], triple[2, :3, :3] @
                    Rotation.from_euler('x', angle, degrees=True).as_matrix(), self.ctrl.robot.tcp_offset)
                    for triple in targets[eligible] for angle in angles])
                rq, rs = self.placement_ik(retreat_poses, np.repeat(qs[eligible, 1], len(angles), axis=0))
                rq, rs = rq.reshape(len(eligible), len(angles), -1), rs.reshape(len(eligible), len(angles))
                selected = np.argmin(rs, axis=1)
                rows = np.arange(len(eligible))
                qs[eligible, 2] = rq[rows, selected]
                scores[eligible, 2] = rs[rows, selected]
                targets[eligible, 2] = retreat_poses.reshape(len(eligible), len(angles), 4, 4)[rows, selected]
        else:
            qs, scores = self.placement_ik(targets.reshape(-1, 4, 4), start)
            qs, scores = qs.reshape(-1, 3, len(start)), scores.reshape(-1, 3)
        endpoint_ok = np.isfinite(scores)
        print('Cup place IK screening:', {
            'phase': 'actual_grasp' if world_object is not None else 'predicted_grasp',
            'target_groups': len(targets), 'preplace_valid': int(endpoint_ok[:, 0].sum()),
            'place_valid': int(endpoint_ok[:, 1].sum()), 'retreat_valid': int(endpoint_ok[:, 2].sum()),
            'retreat_tested_groups': len(eligible) if self.place_cfg.get('chain_flexible_placement', False) else len(targets),
            'complete_groups': int(endpoint_ok.all(1).sum())}, flush=True)
        objects = targets[:, 1] @ np.linalg.inv(object_from_ee)
        tilts = np.rad2deg(np.arccos(np.clip(objects[:, :3, 2] @ (self.base_from_world @ self.world_plate)[:3, 2], -1, 1)))
        costs = scores.sum(axis=1) + np.abs(yaws) / 180. * .1 + tilts*.02
        order = np.argsort(costs, kind="stable")
        for index in order[:int(self.cfg.get("chain_place_candidates", 12))]:
            self.check_budget('place_candidate')
            if not np.isfinite(costs[index]):
                continue
            if grasp_joints is not None and not self.virtual_attachment:
                self.attach_virtual(grasp_joints)
            pre = self.plan_segment(start, qs[index, 0], "preplace")
            if pre is None:
                continue
            place = self.plan_segment(pre["path"][-1], qs[index, 1], "place")
            if place is None:
                continue
            # Restore the exact carry geometry even if retreat validation raises.
            # Re-fitting meshes here would use the wrong object pose after lifting.
            spheres = self.ctrl.motion_gen.kinematics.kinematics_config.get_link_spheres("attached_object").clone()
            try:
                self.ctrl.motion_gen.detach_object_from_robot()
                retreat = self.plan_segment(place["path"][-1], qs[index, 2], "retreat")
            finally:
                self.ctrl.motion_gen.attach_spheres_to_robot(sphere_tensor=spheres)
            if retreat is not None:
                return {"preplace": pre, "place": place, "retreat": retreat,
                        "object_from_ee": object_from_ee.copy(), "yaw_offset_deg": float(yaws[index]),
                        "place_tilt_deg": float(tilts[index])}
        return None

    def choose_chain(self):
        grasps = self.pick.get_ee_poses("armbase")
        lo, hi = self.cfg.get("downward_tilt_range_deg", [1., 30.])
        grasps = grasps[downward_mask(grasps, self.up, lo, hi)]
        grasps = np.array([pose for pose in grasps if palm_clear(pose[None], self.base_object,
            self.cup_points, self.palm_bounds, float(self.cfg.get("palm_clearance", .002)))])
        if not len(grasps):
            raise ValueError("No cup-wall grasp has the requested downward tilt")
        start = self.current_joints()
        grasp_q, grasp_scores = self.ik(grasps, start)
        valid = np.flatnonzero(np.isfinite(grasp_scores))
        if not len(valid):
            print("Cup chain: no valid grasp IK", {"grasps": len(grasps), "stats": dict(self.stats),
                "grasp_ee_xyz_min": grasps[:, :3, 3].min(0).tolist(),
                "grasp_ee_xyz_max": grasps[:, :3, 3].max(0).tolist()}, flush=True)
            return None
        flexible = bool(self.cfg.get('chain_flexible_approach', False))
        if flexible:
            entryposes, _, _ = pregrasp_variants(grasps[valid], [float(self.cfg.get('chain_stable_entry_distance', .012))], up=self.up)
            entry_q, entry_scores = self.ik(entryposes, grasp_q[valid])
            preposes, ids, distances = flexible_pregrasp_variants(grasps[valid], self.cfg['pre_grasp_offsets'],
                self.up, float(self.ctrl.robot.tcp_offset), self.cfg.get('chain_pregrasp_rotations_deg', [0, -5, 5, -10, 10]))
        else:
            preposes, ids, distances = pregrasp_variants(grasps[valid], self.cfg["pre_grasp_offsets"],
                up=self.up if self.cfg.get("grasp_mode") == "rim_pinch" else None)
        if not len(preposes):
            return None
        pre_q, pre_scores = self.ik(preposes, entry_q[ids] if flexible else start)
        scores = pre_scores + grasp_scores[valid[ids]]
        if flexible:
            scores += entry_scores[ids]
        budget = int(self.cfg.get("chain_max_candidates", 16))
        print("Cup chain IK screening:", {"grasps": len(grasps), "valid_grasps": len(valid),
            "entry_targets": int(np.isfinite(entry_scores).sum()) if flexible else None,
            "pregrasp_pairs": int(np.isfinite(scores).sum())}, flush=True)
        selection = self.cfg.get("chain_grasp_selection", "best")
        order = grasp_candidate_order(scores, selection, self.cfg.get("chain_grasp_top_k", 8))
        for index in order[:budget]:
            self.check_budget('grasp_candidate')
            if not np.isfinite(scores[index]):
                break
            self.stats["full_chain_candidates"] += 1
            self.clear_virtual()
            pre = self.plan_segment(start, pre_q[index], "pregrasp")
            if pre is None:
                continue
            if flexible:
                approach = self.plan_segment(pre['path'][-1], entry_q[ids[index]], 'approach')
                if approach is None:
                    continue
                # Keep this as one cached pregrasp command: no extra pause/nod.
                pre['path'] = np.concatenate([pre['path'], approach['path'][1:]])
                pre['goal'] = approach['goal']
                pre['quality'] = trajectory_quality(self.fk_path(pre['path']), self.up)
            grasp = self.plan_segment(pre["path"][-1], grasp_q[valid[ids[index]]], "grasp")
            if grasp is None:
                continue
            grasp_joints = grasp["path"][-1]
            grasp_pose = self.fk(grasp_joints)
            relative = np.linalg.inv(self.base_object) @ grasp_pose
            self.carry_object_from_ee = relative.copy()
            lift_poses = lift_pose_variants(grasp_pose, self.base_object, self.up,
                float(self.cfg.get('chain_lift_height', .11)),
                self.cfg.get('chain_lift_shifts', [0.]), self.cfg.get('chain_lift_tilts_deg', [0.]))
            lift_q, lift_scores = self.ik(lift_poses, grasp_joints)
            self.attach_virtual(grasp_joints)
            placement = None
            for lift_index in np.argsort(lift_scores)[:int(self.cfg.get('chain_lift_candidates', 6))]:
                self.check_budget('lift_candidate')
                if not np.isfinite(lift_scores[lift_index]):
                    continue
                lift = self.plan_segment(grasp_joints, lift_q[lift_index], "lift")
                if lift is None:
                    continue
                placement = self.place_chain(lift["path"][-1], relative, grasp_joints=grasp_joints)
                if placement is not None:
                    break
            if placement is None:
                continue
            tilt = np.rad2deg(np.arcsin(-np.dot(grasps[valid[ids[index]], :3, 2], self.up)))
            chain = {"pregrasp": pre, "grasp": grasp, "lift": lift, **placement,
                     "initial_joints": start.copy(),
                     "pregrasp_offset": float(distances[index]), "tilt_deg": float(tilt),
                     "world_plate": self.world_plate.copy(), "planner": self}
            self.pick.task.cfg["mem_grasp_selection"] = {
                "mode": selection, "candidate_index": int(valid[ids[index]]),
                "grasp_mode": self.cfg.get("grasp_mode", "exterior_side"),
                "pre_grasp_width_m": float(self.open_gripper_state*self.ctrl.robot.cfg["gripper_max_width"]),
                "flexible_approach": flexible, "place_tilt_deg": placement['place_tilt_deg'],
                "stable_entry_distance_m": float(self.cfg.get('chain_stable_entry_distance', 0.)) if flexible else float(distances[index]),
                "tilt_deg": float(tilt), "pregrasp_offset_m": float(distances[index]),
                "inclination_from_vertical_deg": float(90-tilt),
                "object_from_ee": relative.tolist()}
            print("Cup chain accepted:", {"tilt_deg": round(float(tilt), 3),
                "pregrasp_offset_m": float(distances[index]), "place_yaw_offset_deg": placement["yaw_offset_deg"],
                "stats": dict(self.stats),
                "segments": {label: chain[label]["quality"] for label in
                             ["pregrasp", "grasp", "lift", "preplace", "place", "retreat"]}}, flush=True)
            return chain
        print("No complete cup/plate chain:", dict(self.stats), flush=True)
        return None


def path_command(segment, gripper_state):
    position, quaternion = pose_parts(segment["goal"])
    return (position, quaternion, "validated_forward", {"segment": segment, "gripper_state": gripper_state})


def follow_validated_segment(controller, segment, gripper_state):
    """Replay a checked trajectory; replan only if its actual start has drifted."""
    planner = segment["planner"]
    if gripper_state > 0 and segment["label"] in ["pregrasp", "grasp"]:
        current_cup = planner.pick.pick_obj.get_local_pose()[0]
        displacement = np.linalg.norm(np.asarray(current_cup)[:2] - planner.world_object[:2, 3])
        if displacement > float(planner.cfg.get("chain_max_preclose_cup_shift", .003)):
            print("Approach rejected: cup moved before closing:", float(displacement), flush=True)
            controller.num_plan_failed = 1000
            return controller.dummy_forward(planner.current_joints(), gripper_state)
    if getattr(controller, "_validated_active", None) is not segment:
        actual = planner.current_joints()
        tolerance = np.deg2rad(float(planner.cfg.get("chain_reuse_start_tolerance_deg", 3.)))
        if np.max(np.abs(actual - segment["start"])) > tolerance:
            try:
                if hasattr(planner, 'begin_budget'):
                    planner.begin_budget('segment_refresh', float(planner.cfg.get('chain_refresh_timeout_s', 10.)))
                replacement = planner.plan_segment(actual, segment["path"][-1], segment["label"] + "_refresh")
            except PlanningBudgetExceeded as error:
                print('Cup planning budget exhausted:', str(error), flush=True)
                replacement = None
            if replacement is None:
                print("Validated segment start refresh failed:", segment["label"],
                      "max_start_error_rad=", float(np.max(np.abs(actual - segment["start"]))), flush=True)
                controller.num_plan_failed = 1000
                return controller.dummy_forward(actual, gripper_state)
            segment.update(replacement)
        segment["index"] = 0
        segment["complete"] = False
        controller._validated_active = segment
        position, quaternion = pose_parts(segment["goal"])
        controller._ee_trans = controller.tensor_args.to_device(position)
        controller._ee_ori = controller.tensor_args.to_device(quaternion)
        controller.num_last_cmd = 0
        controller.num_plan_failed = 0
    index = segment["index"]
    target = segment["path"][min(index, len(segment["path"]) - 1)]
    segment["index"] += 1
    segment["complete"] = segment["index"] >= len(segment["path"])
    if segment["complete"]:
        controller.num_last_cmd += 1
        if controller.num_last_cmd > 90:
            if controller.num_last_cmd == 91:
                measured = pose_matrix(*controller.get_ee_pose())
                difference = np.linalg.inv(segment["goal"]) @ measured
                print("Validated segment tracking timeout:", segment["label"],
                      "joint_error_rad=", (planner.current_joints() - segment["path"][-1]).tolist(),
                      "pose_error_m=", float(np.linalg.norm(difference[:3, 3])),
                      "pose_error_deg=", float(np.rad2deg(Rotation.from_matrix(difference[:3, :3]).magnitude())), flush=True)
            controller.num_plan_failed = 1000
    return controller.dummy_forward(target, gripper_state)


def prepare_pick(pick):
    planner = CupPlateChainPlanner(pick)
    pick.controller._cup_plate_chain = None
    pick.controller._cup_plate_place_succeeded = False
    try:
        chain = planner.choose_chain()
    except PlanningBudgetExceeded as error:
        print('Cup planning budget exhausted:', {'phase': 'initial_chain', 'detail': str(error),
            'stats': dict(planner.stats)}, flush=True)
        chain = None
    finally:
        planner.clear_virtual()
    if chain is None:
        pick.manip_list = []
        return
    pick.controller._cup_plate_chain = chain
    grasp_pose = chain["grasp"]["goal"]
    position, quaternion = pose_parts(grasp_pose)
    hold = (position, quaternion, "dummy_forward", {
        "arm_action": chain["grasp"]["path"][-1], "gripper_state": -1.})
    paths = pick.pick_obj.mesh_prim_paths
    pick.manip_list = [path_command(chain["pregrasp"], planner.open_gripper_state),
                      path_command(chain["grasp"], planner.open_gripper_state)]
    pick.manip_list.extend([hold] * int(pick.skill_cfg.get("gripper_change_steps", 10)))
    pick.manip_list.extend([
        (position, quaternion, "update_specific", {"ignore_substring": pick.controller.ignore_substring,
                                                  "reference_prim_path": pick.controller.reference_prim_path}),
        (position, quaternion, "attach_obj", {"obj_prim_path": paths}),
        path_command(chain["lift"], -1.),
    ])


def prepare_place(place):
    controller = place.controller
    chain = getattr(controller, "_cup_plate_chain", None)
    if chain is None:
        place.manip_list = []
        return
    planner = chain["planner"]
    controller.update_specific(controller.ignore_substring + [place.pick_obj.name], controller.reference_prim_path)
    actual_object = pose_matrix(*place.pick_obj.get_local_pose())
    actual_ee = planner.fk(planner.current_joints())
    relative = np.linalg.inv(planner.base_from_world @ actual_object) @ actual_ee
    difference = np.linalg.inv(chain["object_from_ee"]) @ relative
    plate_pose = pose_matrix(*place.place_obj.get_local_pose())
    drift_m = np.linalg.norm(difference[:3, 3])
    drift_rad = Rotation.from_matrix(difference[:3, :3]).magnitude()
    # Re-estimate grasp after physical contact/slip. No object pose is forced.
    if drift_m > .003 or drift_rad > np.deg2rad(2.) or not np.allclose(plate_pose, chain["world_plate"], atol=.001):
        planner.world_plate = plate_pose
        try:
            planner.begin_budget('actual_place', float(planner.cfg.get('chain_place_timeout_s', 20.)))
            replacement = planner.place_chain(planner.current_joints(), relative, world_object=actual_object)
        except PlanningBudgetExceeded as error:
            print('Cup planning budget exhausted:', {'phase': 'actual_place', 'detail': str(error),
                'stats': dict(planner.stats)}, flush=True)
            replacement = None
        if replacement is None:
            actual_tilt = np.rad2deg(np.arccos(np.clip(actual_object[2, 2], -1, 1)))
            print('Cup placement refresh failed after physical grasp:', {
                'grasp_drift_m': float(drift_m), 'grasp_drift_deg': float(np.rad2deg(drift_rad)),
                'actual_cup_tilt_deg': float(actual_tilt), 'stats': dict(planner.stats)}, flush=True)
            place.manip_list = []
            return
        chain.update(replacement)
        selection = planner.task.cfg.get('mem_grasp_selection', {})
        selection['place_tilt_deg'] = replacement['place_tilt_deg']
        selection['actual_object_from_ee'] = relative.tolist()
        selection['grasp_drift_m'] = float(drift_m)
        selection['grasp_drift_deg'] = float(np.rad2deg(drift_rad))
        print("Cup placement refreshed after grasp drift:", {"translation_m": float(drift_m),
                                                             "rotation_deg": float(np.rad2deg(drift_rad))}, flush=True)
    position, quaternion = pose_parts(chain["place"]["goal"])
    hold = (position, quaternion, "dummy_forward", {
        "arm_action": chain["place"]["path"][-1], "gripper_state": planner.open_gripper_state})
    place.manip_list = [path_command(chain["preplace"], -1.), path_command(chain["place"], -1.)]
    place.manip_list.extend([hold] * int(place.skill_cfg.get("gripper_change_steps", 10)))
    place.manip_list.extend([
        (position, quaternion, "detach_obj", {}),
        (position, quaternion, "update_specific", {
            "ignore_substring": controller.ignore_substring + [place.pick_obj.name],
            "reference_prim_path": controller.reference_prim_path}),
        path_command(chain["retreat"], planner.open_gripper_state),
    ])
    retreat_pos, retreat_quat = pose_parts(chain["retreat"]["goal"])
    settle = (retreat_pos, retreat_quat, "dummy_forward", {
        "arm_action": chain["retreat"]["path"][-1], "gripper_state": planner.open_gripper_state})
    place.manip_list.extend([settle] * int(place.skill_cfg.get("settle_steps", 30)))
    place.manip_list.append((retreat_pos, retreat_quat, "update_specific", {
        "ignore_substring": controller.ignore_substring,
        "reference_prim_path": controller.reference_prim_path}))
    place.place_ee_trans = position


def prepare_return_home(skill):
    """Return the empty arm to this episode's measured initial joint state."""
    controller = skill.controller
    chain = getattr(controller, '_cup_plate_chain', None)
    skill.manip_list = []
    skill.failed = True
    if chain is None or not getattr(controller, '_cup_plate_place_succeeded', False):
        return
    planner = chain['planner']
    skill.target_joints = np.asarray(chain['initial_joints']).copy()
    # detach_obj already released the planning cup during place; reload the
    # actual placed cup as a world obstacle before checking the return path.
    controller.update_specific(controller.ignore_substring, controller.reference_prim_path)
    planner.begin_budget('return_home', float(skill.skill_cfg.get('planning_timeout_s', 20.)))
    try:
        planner.check_budget('return_home')
        start = planner.current_joints()
        path = resample_joints(np.stack([start, skill.target_joints]),
                               float(planner.cfg.get('chain_max_joint_step', .025)))
        if planner.valid_path(path, 'return_home'):
            segment = {'path': path, 'start': start.copy(), 'goal': planner.fk(path[-1]),
                       'label': 'return_home', 'complete': False, 'planner': planner,
                       'quality': trajectory_quality(planner.fk_path(path), planner.up)}
        else:
            segment = planner.plan_segment(start, skill.target_joints, 'return_home')
        planner.check_budget('return_home_complete')
    except PlanningBudgetExceeded as error:
        print('Cup return home budget exhausted:', str(error), flush=True)
        return
    if segment is None:
        print('Cup return home planning failed', flush=True)
        return
    skill.failed = False
    skill.manip_list = [path_command(segment, planner.open_gripper_state)]
    print('Cup return home planned:', {'target_joints_rad': skill.target_joints.tolist(),
        'waypoints': len(segment['path'])}, flush=True)
