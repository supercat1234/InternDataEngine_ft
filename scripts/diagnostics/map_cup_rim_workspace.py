#!/usr/bin/env python3
"""Sample rim-grasp endpoint IK on the table, without running Isaac or recording data.

This is a fixed-attitude endpoint reference map, not a success-rate or path guarantee.
The active flexible approach/lift search is not evaluated by this coarse map.
The arm-base frame is reconstructed from a recorded wrist-camera extrinsic and
the recorded base-to-EE transform, avoiding nominal CAD mounting discrepancies.
"""
import argparse
import csv
import json
from pathlib import Path
import pickle
import sys

import lmdb
import numpy as np
from scipy.spatial.transform import Rotation
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"workflows/simbox/core/utils"))
from cup_plate_planning import joint_margins, pose_matrix, palm_clear, cylinder_points


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episode", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--step", type=float, default=.05)
    ap.add_argument("--x-range", nargs=2, type=float, default=[-.65, .65])
    ap.add_argument("--y-range", nargs=2, type=float, default=[-.25, .25])
    args = ap.parse_args()
    task = yaml.safe_load((ROOT/"local_configs/cup_plate_desk/task.yaml").read_text())["tasks"][0]
    robot = yaml.safe_load((ROOT/task["robots"][0]["robot_config_file"]).read_text())
    robot.update(task["robots"][0])
    pick = task["skills"][0][robot["name"]][0]["left"][0]
    place = task["skills"][0][robot["name"]][0]["left"][1]
    env = lmdb.open(str(args.episode/"lmdb"), readonly=True, lock=False)
    with env.begin() as txn:
        cameras = np.asarray(pickle.loads(txn.get(b"camera2env_pose.hand_left")))
        base_ee = np.asarray(pickle.loads(txn.get(b"T_base_ee_fl")))
    camera = task["cameras"][0]
    ee_camera = pose_matrix(camera["translation"], camera["orientation"])
    world_base = cameras[0]@np.linalg.inv(ee_camera)@np.linalg.inv(base_ee[0])
    base_world = np.linalg.inv(world_base)

    from curobo.geom.types import WorldConfig
    from curobo.types.base import TensorDeviceType
    from curobo.types.math import Pose
    from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
    from curobo.util_file import load_yaml
    tensor = TensorDeviceType()
    config = load_yaml(robot["robot_file"][0])["robot_cfg"]
    opening = pick["pre_grasp_width"]
    for key, value in config["kinematics"]["lock_joints"].items():
        config["kinematics"]["lock_joints"][key] = np.sign(value)*opening/2
    table = base_world@pose_matrix([0, 0, .7325], [1, 0, 0, 0])
    quat = Rotation.from_matrix(table[:3, :3]).as_quat(scalar_first=True)
    world = WorldConfig.from_dict({"cuboid": {"desktop": {
        "dims": [1.4, .6, .035], "pose": [*table[:3, 3].tolist(), *quat.tolist()]}}})
    solver = IKSolver(IKSolverConfig.load_from_robot_config(config, world,
        num_seeds=20, rotation_threshold=robot["planner_ik_rotation_threshold"],
        position_threshold=robot["planner_ik_position_threshold"],
        self_collision_check=True, self_collision_opt=True, tensor_args=tensor, use_cuda_graph=True))
    limits = solver.kinematics.get_joint_limits().position.cpu().numpy()
    # Keep all active heights/tilts, thinning only every other cup azimuth.
    # Do not assume a fixed label count: the task's grasp set is configurable.
    raw = np.load(ROOT/pick["grasp_pose_file"])
    rotations = raw[:, 4:13].reshape(-1, 3, 3)
    azimuths = np.round(np.arctan2(rotations[:, 1, 1], rotations[:, 0, 1]), 5)
    _, azimuth_ids = np.unique(azimuths, return_inverse=True)
    raw = raw[azimuth_ids % 2 == 0]
    ee = np.repeat(np.eye(4)[None], len(raw), axis=0)
    ee[:, :3, :3] = raw[:, 4:13].reshape(-1, 3, 3)@np.asarray(robot["R_ee_graspnet"]).T
    ee[:, :3, 3] = raw[:, 13:16]-ee[:, :3, 2]*robot["tcp_offset"]
    symmetry = np.diag([-1, -1, 1, 1])
    ee = np.concatenate([ee, ee@symmetry])
    # Apply the same measured palm guard as task planning before mapping.
    mask = [palm_clear(t[None], np.eye(4), cylinder_points(.0375, .08),
                       np.asarray(pick["palm_collision_bounds"]), .002) for t in ee]
    ee = ee[mask]
    xs = np.round(np.arange(args.x_range[0], args.x_range[1]+1e-6, args.step), 6)
    ys = np.round(np.arange(args.y_range[0], args.y_range[1]+1e-6, args.step), 6)
    rows = []
    batch = 64
    for x in xs:
        for y in ys:
            row = {"x": float(x), "y": float(y)}
            for label, z, offset in (("pick", .7509, min(pick["pre_grasp_offsets"])),
                                      ("place", .7509+place["support_surface_z"]+place["place_z_offset"], place["pre_place_z_offset"])):
                if label == "place" and (abs(x) > .7-.077 or abs(y) > .3-.077):
                    row[label+"_candidate_pairs"] = row[label+"_preferred_pairs"] = 0
                    continue
                poses = pose_matrix([x, y, z], [1, 0, 0, 0])[None]@ee
                upper = poses.copy()
                upper[:, 2, 3] += offset
                targets = base_world[None]@np.stack([poses, upper], axis=1).reshape(-1, 4, 4)
                valid, safe = [], []
                for start in range(0, len(targets), batch):
                    chunk = targets[start:start+batch]
                    padded = np.concatenate([chunk, np.repeat(chunk[-1:], batch-len(chunk), axis=0)])
                    goal = Pose(position=tensor.to_device(padded[:, :3, 3]),
                                quaternion=tensor.to_device(Rotation.from_matrix(padded[:, :3, :3]).as_quat(scalar_first=True)))
                    result = solver.solve_batch(goal, return_seeds=4)
                    joints = result.solution.cpu().numpy()[:len(chunk)]
                    good = result.success.cpu().numpy()[:len(chunk)]
                    margins = joint_margins(joints, limits[0], limits[1]).min(-1)
                    valid.extend((good & (margins >= np.deg2rad(1))).any(-1).tolist())
                    safe.extend((good & (margins >= np.deg2rad(5))).any(-1).tolist())
                row[label+"_candidate_pairs"] = int(np.asarray(valid).reshape(-1, 2).all(1).sum())
                row[label+"_preferred_pairs"] = int(np.asarray(safe).reshape(-1, 2).all(1).sum())
            rows.append(row)
        print("WORKSPACE_COLUMN", x, sum(r["pick_candidate_pairs"] > 0 for r in rows if r["x"] == x), flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/"endpoint_ik.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = {"coordinate_frame": "task", "units": "meters", "grid_step_m": args.step,
               "world_from_arm_base": world_base.tolist(), "candidate_count": len(ee),
               "meaning": "Fixed-attitude endpoint IK with desktop collision and joint margins; active flexible approach/lift, full paths and grasp contact are not tested by this map."}
    for label in ("pick", "place"):
        pts = np.array([[r["x"], r["y"]] for r in rows if r[label+"_candidate_pairs"]])
        summary[label] = {"passed_points": len(pts), "sampled_points": len(rows),
                          "bbox_xy_min": pts.min(0).tolist() if len(pts) else None,
                          "bbox_xy_max": pts.max(0).tolist() if len(pts) else None}
    (args.output/"endpoint_ik.json").write_text(json.dumps(summary, indent=2)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True)
    for ax, label, title in zip(axes, ("pick", "place"), ("Cup: grasp + 40 mm lift", "Plate: upright place + retreat")):
        for r in rows:
            color = "#2a9d8f" if r[label+"_preferred_pairs"] else ("#f4a261" if r[label+"_candidate_pairs"] else "#d9d9d9")
            ax.scatter(r["x"], r["y"], color=color, s=70, marker="s")
        ax.scatter(world_base[0, 3], world_base[1, 3], marker="^", s=90, color="#264653", label="Left arm base")
        ax.set_title(title); ax.set_xlabel("Task X (m)"); ax.set_aspect("equal"); ax.grid(alpha=.15)
        ax.legend(loc="lower left", fontsize=8)
    axes[0].set_ylabel("Task Y (m)")
    fig.suptitle("Rim pinch endpoint IK: green >=5 deg joint margin; orange >=1 deg; gray no candidate\nSampled endpoints only; not a full-task success guarantee", fontsize=10)
    fig.tight_layout()
    fig.savefig(args.output/"endpoint_ik.png", dpi=160)
    print("WORKSPACE_RESULT", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
