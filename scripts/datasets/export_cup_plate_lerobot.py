#!/usr/bin/env python3
"""Export trusted local SimBox LMDB episodes to LeRobot v3 with body-frame TCP actions.

No Isaac Sim, CUDA, Pinocchio, robot connection or Hub upload is required.
The source contains pickle; only use recordings you trust.
"""

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import pickle
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "local_configs/cup_plate_desk/lerobot_v3.json"
ACTION_NAMES = ["tcp.dx", "tcp.dy", "tcp.dz", "tcp.rotvec_x", "tcp.rotvec_y",
                "tcp.rotvec_z", "gripper.openness"]
STATE_NAMES = ["tcp.x", "tcp.y", "tcp.z", "tcp.qx", "tcp.qy", "tcp.qz", "tcp.qw",
               "gripper.openness"]


def validate_transforms(transforms):
    transforms = np.asarray(transforms, dtype=np.float64)
    if transforms.shape[-2:] != (4, 4) or not np.isfinite(transforms).all():
        raise ValueError("Expected finite SE(3) transforms (..., 4, 4)")
    rot = transforms[..., :3, :3]
    if not (np.allclose(transforms[..., 3, :], [0, 0, 0, 1], atol=1e-5)
            and np.allclose(np.swapaxes(rot, -1, -2) @ rot, np.eye(3), atol=1e-4)
            and np.allclose(np.linalg.det(rot), 1, atol=1e-4)):
        raise ValueError("Invalid rigid transform (rotation or homogeneous row)")
    return transforms


def relative_tcp_actions(current, target, openness):
    """Return [R_cur.T @ (p_tgt-p_cur), Log(R_cur.T @ R_tgt), absolute gripper]."""
    current, target = validate_transforms(current), validate_transforms(target)
    if current.shape != target.shape or current.ndim != 3:
        raise ValueError("Current and target must both have shape (N, 4, 4)")
    openness = np.asarray(openness, dtype=np.float64).reshape(-1, 1)
    if len(openness) != len(current) or not np.isfinite(openness).all():
        raise ValueError("Invalid gripper command length or values")
    if np.any((openness < 0) | (openness > 1)):
        raise ValueError("Gripper command must be in [0, 1]")
    inverse_rot = current[:, :3, :3].transpose(0, 2, 1)
    translation = np.einsum("nij,nj->ni", inverse_rot, target[:, :3, 3] - current[:, :3, 3])
    rotvec = Rotation.from_matrix(inverse_rot @ target[:, :3, :3]).as_rotvec()
    return np.concatenate([translation, rotvec, openness], axis=1).astype(np.float32)


def apply_tcp_action(current, action):
    """Decode a single body-frame action: T_target = T_current @ DeltaT."""
    current = validate_transforms(current)
    action = np.asarray(action)
    if current.shape != (4, 4) or action.shape != (7,) or not np.isfinite(action).all():
        raise ValueError("Expected a (4, 4) pose and finite (7,) action")
    delta = np.eye(4)
    delta[:3, 3] = action[:3]
    delta[:3, :3] = Rotation.from_rotvec(action[3:6]).as_matrix()
    return current @ delta


def poses_xyzw(transforms):
    """Absolute pose in the arm base, with continuous quaternion signs."""
    transforms = validate_transforms(transforms)
    quat = Rotation.from_matrix(transforms[:, :3, :3]).as_quat()
    for idx in range(1, len(quat)):
        if np.dot(quat[idx - 1], quat[idx]) < 0:
            quat[idx] *= -1
    return np.concatenate([transforms[:, :3, 3], quat], axis=1).astype(np.float32)


class UrdfFK:
    """CPU FK along the configured base-to-link6 chain; URDF uses metres/radians."""

    def __init__(self, path, base_link, ee_link, joint_names):
        robot = ET.parse(path).getroot()
        parents = {joint.find("child").get("link"): joint for joint in robot.findall("joint")}
        chain, link = [], ee_link
        while link != base_link:
            if link not in parents or len(chain) >= len(parents):
                raise ValueError(f"Cannot find URDF chain {base_link} -> {ee_link}")
            joint = parents[link]
            chain.append(joint)
            link = joint.find("parent").get("link")
        self.chain = []
        self.joint_names = list(joint_names)
        active = []
        for joint in reversed(chain):
            kind, name = joint.get("type"), joint.get("name")
            if kind not in ("fixed", "revolute", "continuous"):
                raise ValueError(f"Unsupported joint type: {kind}")
            origin = joint.find("origin")
            xyz = np.fromstring(origin.get("xyz", "0 0 0") if origin is not None else "0 0 0", sep=" ")
            rpy = np.fromstring(origin.get("rpy", "0 0 0") if origin is not None else "0 0 0", sep=" ")
            transform = np.eye(4)
            transform[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
            transform[:3, 3] = xyz
            axis_node = joint.find("axis")
            axis = np.fromstring(axis_node.get("xyz", "1 0 0") if axis_node is not None else "1 0 0", sep=" ")
            axis /= np.linalg.norm(axis)
            index = None if kind == "fixed" else self.joint_names.index(name)
            if index is not None:
                active.append(name)
            self.chain.append((transform, axis, index))
        if set(active) != set(self.joint_names) or len(active) != len(self.joint_names):
            raise ValueError("Configured joint names do not match the URDF chain")

    def __call__(self, joints):
        joints = np.asarray(joints, dtype=np.float64)
        if joints.ndim != 2 or joints.shape[1] != len(self.joint_names) or not np.isfinite(joints).all():
            raise ValueError("Invalid joint array")
        result = np.repeat(np.eye(4)[None], len(joints), axis=0)
        for origin, axis, index in self.chain:
            result = result @ origin
            if index is not None:
                rotation = np.repeat(np.eye(4)[None], len(joints), axis=0)
                rotation[:, :3, :3] = Rotation.from_rotvec(joints[:, index, None] * axis).as_matrix()
                result = result @ rotation
        return result


def load_config(path):
    cfg = json.loads(Path(path).read_text())
    if cfg["arm"] != "left":
        raise ValueError("This task exporter currently supports the active left arm only")
    if not isinstance(cfg["fps"], int) or cfg["fps"] <= 0 or cfg["gripper_max_width_m"] <= 0:
        raise ValueError("FPS and gripper width must be positive")
    if not cfg["cameras"] or len(set(cfg["cameras"])) != len(cfg["cameras"]):
        raise ValueError("Camera names must be nonempty and unique")
    cfg["T_link6_tcp"] = validate_transforms(cfg["T_link6_tcp"]).tolist()
    cfg["urdf"] = str((ROOT / cfg["urdf"]).resolve())
    return cfg


def read_array(txn, key, length, trailing_shape):
    raw = txn.get(key.encode())
    if raw is None:
        raise ValueError(f"Missing LMDB array: {key}")
    values = np.asarray(pickle.loads(raw), dtype=np.float64)
    if trailing_shape == (1,) and values.shape == (length,):
        values = values[:, None]
    if values.shape != (length, *trailing_shape) or not np.isfinite(values).all():
        raise ValueError(f"Invalid {key}: expected {(length, *trailing_shape)}, got {values.shape}")
    return values


def prepare_episode(txn, meta, cfg, fk):
    count = int(meta["num_steps"])
    if count < 1:
        raise ValueError("Empty episode")
    for camera in cfg["cameras"]:
        key = f"images.rgb.{camera}"
        ids = meta.get("image_valid_step_ids", {}).get(key)
        if ids != list(range(count)):
            raise ValueError(f"{camera}: missing/noncontiguous frame IDs; resampling is required")
        for idx in ids:
            if txn.get(f"{key}/{idx:04d}".encode()) is None:
                raise ValueError(f"Missing image {key}/{idx:04d}")
    measured_q = read_array(txn, "states.left_joint.position", count, (6,))
    target_q = read_array(txn, "master_actions.left_joint.position", count, (6,))
    measured_ee = validate_transforms(read_array(txn, "T_base_ee_fl", count, (4, 4)))
    measured_fk = fk(measured_q)
    pos_error = np.linalg.norm(measured_ee[:, :3, 3] - measured_fk[:, :3, 3], axis=1)
    rot_error = Rotation.from_matrix(measured_ee[:, :3, :3].transpose(0, 2, 1)
                                    @ measured_fk[:, :3, :3]).magnitude()
    if pos_error.max() > cfg["max_fk_position_error_m"] or rot_error.max() > cfg["max_fk_rotation_error_rad"]:
        raise ValueError(f"USD/URDF frames differ: position={pos_error.max():.6g} m, rotation={rot_error.max():.6g} rad")
    tool = np.asarray(cfg["T_link6_tcp"])
    current_tcp = measured_ee @ tool
    target_tcp = fk(target_q) @ tool
    openness = read_array(txn, "master_actions.left_gripper.openness", count, (1,))
    width = read_array(txn, "states.left_gripper.position", count, (1,))
    max_width = cfg["gripper_max_width_m"]
    if width.min() < -0.002 or width.max() > max_width + 0.002:
        raise ValueError("Measured gripper width is inconsistent with configured range")
    state = np.concatenate([poses_xyzw(current_tcp), np.clip(width / max_width, 0, 1)], axis=1).astype(np.float32)
    action = relative_tcp_actions(current_tcp, target_tcp, openness)
    # Check the complete SE(3) reconstruction, including offset-induced TCP translation.
    decoded = np.stack([apply_tcp_action(pose, cmd) for pose, cmd in zip(current_tcp, action)])
    if not np.allclose(decoded, target_tcp, atol=1e-6):
        raise ValueError("Relative TCP action reconstruction failed")
    # Keep diagnostics outside observation/action prefixes so policy feature
    # auto-discovery does not mistake joint targets for another action head.
    frames = {"observation.state": state, "action": action,
              "auxiliary.joint_positions": measured_q.astype(np.float32),
              "auxiliary.joint_targets": target_q.astype(np.float32)}
    report = {"frames": count, "max_fk_position_error_m": float(pos_error.max()),
              "max_fk_rotation_error_rad": float(rot_error.max()),
              "max_action_translation_m": float(np.linalg.norm(action[:, :3], axis=1).max()),
              "max_action_rotation_rad": float(np.linalg.norm(action[:, 3:6], axis=1).max()),
              "max_reconstruction_error": float(np.abs(decoded - target_tcp).max())}
    return frames, report


def decode_image(txn, camera, idx):
    import cv2
    raw = txn.get(f"images.rgb.{camera}/{idx:04d}".encode())
    image = cv2.imdecode(np.asarray(pickle.loads(raw), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode {camera} frame {idx}")
    # SimBox imencode receives RGB directly. imdecode restores those original
    # array channels; another BGR->RGB conversion here would swap red and blue.
    return image


def discover_episodes(source):
    source = Path(source).resolve()
    files = [source / "meta_info.pkl"] if (source / "meta_info.pkl").is_file() else sorted(source.rglob("meta_info.pkl"))
    episodes = [path.parent for path in files if (path.parent / "lmdb/data.mdb").is_file()]
    if not episodes:
        raise ValueError(f"No complete LMDB episodes found in {source}")
    return episodes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repo-id", default="local/cup_plate_tcp_relative")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--max-episodes", type=int,
                        help="Export at most this many episodes, sorted by source path")
    parser.add_argument("--streaming-encoding", action="store_true",
                        help="Encode directly to video without temporary PNG files (requires compatible LeRobot)")
    parser.add_argument("--inspect-only", action="store_true", help="Validate all data without creating a dataset")
    args = parser.parse_args()
    if args.max_episodes is not None and args.max_episodes <= 0:
        parser.error("--max-episodes must be positive")
    if not args.inspect_only and args.output is None:
        parser.error("--output is required for export")
    if args.output is not None and args.output.exists() and not args.inspect_only:
        parser.error("Output already exists; choose a new directory")
    import lmdb
    cfg = load_config(args.config)
    fk = UrdfFK(cfg["urdf"], cfg["base_link"], cfg["ee_link"], cfg["joint_names"])
    episodes = discover_episodes(args.source)
    if args.max_episodes is not None:
        episodes = episodes[:args.max_episodes]
    reports, shapes = [], {}
    # Preflight every episode before creating output files.
    for path in episodes:
        meta = pickle.loads((path / "meta_info.pkl").read_bytes())
        with lmdb.open(str(path / "lmdb"), readonly=True, lock=False, readahead=False) as env:
            with env.begin() as txn:
                _, report = prepare_episode(txn, meta, cfg, fk)
                for camera in cfg["cameras"]:
                    shape = decode_image(txn, camera, 0).shape
                    if camera in shapes and shapes[camera] != shape:
                        raise ValueError(f"Camera resolution changes between episodes: {camera}")
                    shapes[camera] = shape
        reports.append({"source": str(path), **report})
    summary = {"config": cfg, "episodes": reports, "total_frames": sum(r["frames"] for r in reports)}
    print(json.dumps(summary, indent=2), flush=True)
    if args.inspect_only:
        return
    from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset
    if CODEBASE_VERSION != "v3.0":
        raise RuntimeError(f"LeRobot v3 required; installed dataset version is {CODEBASE_VERSION}")
    features = {"observation.state": {"dtype": "float32", "shape": (8,), "names": STATE_NAMES},
                "action": {"dtype": "float32", "shape": (7,), "names": ACTION_NAMES}}
    for key in ["auxiliary.joint_positions", "auxiliary.joint_targets"]:
        features[key] = {"dtype": "float32", "shape": (6,), "names": cfg["joint_names"]}
    for camera, shape in shapes.items():
        features[f"observation.images.{camera}"] = {"dtype": "video", "shape": shape,
                                                    "names": ["height", "width", "channels"]}
    # LeRobot 0.6 moved codec options into RGBEncoderConfig; 0.5 uses vcodec.
    create_parameters = inspect.signature(LeRobotDataset.create).parameters
    if "rgb_encoder" in create_parameters:
        from lerobot.configs.video import RGBEncoderConfig
        encoder = (RGBEncoderConfig(vcodec="h264", g=cfg["fps"], crf=23, preset="veryfast")
                   if args.streaming_encoding else RGBEncoderConfig(vcodec="h264"))
        encoder_options = {"rgb_encoder": encoder}
    else:
        encoder_options = {"vcodec": "h264"}
    writer_options = {"image_writer_threads": 2}
    if args.streaming_encoding:
        if "streaming_encoding" not in create_parameters or "rgb_encoder" not in create_parameters:
            raise RuntimeError("Installed LeRobot does not support this streaming encoder configuration")
        writer_options = {"streaming_encoding": True, "encoder_threads": 2,
                          "encoder_queue_maxsize": 30, "image_writer_threads": 0}
    dataset = LeRobotDataset.create(repo_id=args.repo_id, root=args.output.resolve(), fps=cfg["fps"],
                                   features=features, robot_type="piper_left_tcp_relative",
                                   video_backend="pyav", **writer_options, **encoder_options)
    try:
        for path in episodes:
            meta = pickle.loads((path / "meta_info.pkl").read_bytes())
            with lmdb.open(str(path / "lmdb"), readonly=True, lock=False, readahead=False) as env:
                with env.begin() as txn:
                    frames, report = prepare_episode(txn, meta, cfg, fk)
                    for idx in range(report["frames"]):
                        frame = {key: value[idx] for key, value in frames.items()}
                        frame["task"] = meta["language_instruction"]
                        for camera in cfg["cameras"]:
                            image = decode_image(txn, camera, idx)
                            if image.shape != shapes[camera]:
                                raise ValueError(f"Camera resolution changes at {camera}/{idx}")
                            frame[f"observation.images.{camera}"] = image
                        dataset.add_frame(frame)
            dataset.save_episode(parallel_encoding=False)
            print(f"Saved {path.name}: {report['frames']} frames", flush=True)
    finally:
        dataset.finalize()
    summary.update({"dataset_version": CODEBASE_VERSION,
                    "urdf_sha256": hashlib.sha256(Path(cfg["urdf"]).read_bytes()).hexdigest(),
                    "action_formula": "DeltaT = inverse(T_base_tcp_measured[t]) @ T_base_tcp_command[t]",
                    "action_decoding": "T_base_tcp_target = T_base_tcp_current @ DeltaT",
                    "rotation": "rotation vector (axis * angle), radians, expressed in current TCP",
                    "translation": "metres, expressed in current TCP",
                    "gripper": "absolute command: 0 closed, 1 open; not a delta",
                    "state": "absolute measured TCP pose in arm_base: xyz metres, quaternion xyzw, gripper width/max_width",
                    "target_source": "FK(master_actions.left_joint.position), not next measured state",
                    "source_image_channels": "SimBox RGB array -> OpenCV imencode -> imdecode (no extra channel swap)",
                    "force_labels": False})
    (args.output / "meta/tcp_action_convention.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Dataset ready: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
