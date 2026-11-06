#!/usr/bin/env python3
"""Convert SimBox episodes to the RobotCooking LeRobot v3 joint-action schema.

Actions are future measured feedback, matching convert_robotcooking.py, rather
than the TCP deltas produced by export_cup_plate_lerobot.py. Sources are trusted
local pickle/LMDB recordings. No robot connection or Hub upload is performed.
"""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import pickle

import cv2
import lmdb
import numpy as np

from export_cup_plate_lerobot import discover_episodes, read_array

CAMERAS = {"TOP_CAM": "external", "LEFT_WRIST_CAM": "hand_left", "RIGHT_WRIST_CAM": "hand_right"}
STATE_NAMES = [f"{side}_{part}" for side in ("left", "right")
               for part in ("joint_1.pos", "joint_2.pos", "joint_3.pos", "joint_4.pos",
                            "joint_5.pos", "joint_6.pos", "gripper.pos")]
COLLECTOR = "simbox_scripted_v0.7_top_grasp"


def features_for_shape(height=256, width=256):
    vector = {"dtype": "float32", "shape": (14,), "names": STATE_NAMES}
    features = {"observation.state": dict(vector), "action": dict(vector),
                "complementary_info.policy_action": dict(vector),
                "complementary_info.is_intervention": {"dtype": "float32", "shape": (1,), "names": ["is_intervention"]},
                "complementary_info.state": {"dtype": "float32", "shape": (1,), "names": ["intervention_state"]},
                "complementary_info.collector_policy_id": {"dtype": "string", "shape": (1,), "names": ["collector_policy_id"]}}
    for camera in CAMERAS:
        features[f"observation.images.{camera}"] = {"dtype": "video", "shape": (height, width, 3),
                                                  "names": ["height", "width", "channels"],
                                                  "info": {"is_depth_map": False}}
    return features


def letterbox(image, height=256, width=256):
    source_h, source_w = image.shape[:2]
    scale = min(width / source_w, height / source_h)
    new_w, new_h = max(1, round(source_w * scale)), max(1, round(source_h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    top, left = (height - new_h) // 2, (width - new_w) // 2
    canvas[top:top+new_h, left:left+new_w] = resized
    return np.ascontiguousarray(canvas)


def prepare_joint_episode(txn, meta, future_offset=2, gripper_width_m=0.1):
    count = int(meta["num_steps"])
    if future_offset < 1 or count <= future_offset:
        raise ValueError("Need a positive future offset and more source frames than the offset")
    if not np.isfinite(gripper_width_m) or gripper_width_m <= 0:
        raise ValueError("Gripper maximum width must be positive")
    for source_camera in CAMERAS.values():
        ids = meta.get("image_valid_step_ids", {}).get(f"images.rgb.{source_camera}")
        if ids != list(range(count)):
            raise ValueError(f"{source_camera}: missing or noncontiguous image step IDs")
    parts = []
    for arm in ("left", "right"):
        q = read_array(txn, f"states.{arm}_joint.position", count, (6,))
        width = read_array(txn, f"states.{arm}_gripper.position", count, (1,))
        if width.min() < -0.002 or width.max() > gripper_width_m + 0.002:
            raise ValueError(f"{arm}: measured gripper width is outside configured range")
        parts.extend([q, np.clip(width / gripper_width_m, 0, 1)])
    full_state = np.concatenate(parts, axis=1).astype(np.float32)
    state, action = full_state[:-future_offset], full_state[future_offset:]
    return state, action


def decode_source(txn, camera, index):
    raw = txn.get(f"images.rgb.{camera}/{index:04d}".encode())
    if raw is None:
        raise ValueError(f"Missing {camera} image {index}")
    encoded = np.asarray(pickle.loads(raw), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Invalid {camera} image {index}")
    # Source SimBox passes RGB arrays directly into imencode. imdecode returns
    # those original channels; do not apply another BGR/RGB swap here.
    return image, raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-session", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/cup_plate_v07_robotcooking")
    parser.add_argument("--future-offset-frames", type=int, default=2)
    parser.add_argument("--image-width", type=int, default=256)
    parser.add_argument("--image-height", type=int, default=256)
    parser.add_argument("--gripper-max-width-m", type=float, default=.1)
    parser.add_argument("--max-episodes", type=int)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new directory")
    if args.future_offset_frames < 1 or min(args.image_width, args.image_height) < 1:
        parser.error("Future offset and image dimensions must be positive")
    if args.max_episodes is not None and args.max_episodes < 1:
        parser.error("--max-episodes must be positive")
    reference_episodes = sorted(args.reference_session.glob("episode_*/metadata.json"))
    if not reference_episodes:
        parser.error("Reference session has no episode metadata")
    instructions = {json.loads(p.read_text())["instruction"] for p in reference_episodes}
    if len(instructions) != 1:
        parser.error("Reference session has multiple task texts; explicit task mapping is required")
    task = instructions.pop()
    episodes = discover_episodes(args.source)
    if args.max_episodes:
        episodes = episodes[:args.max_episodes]
    reports, source_shapes = [], {}
    for index, path in enumerate(episodes):
        meta = pickle.loads((path / "meta_info.pkl").read_bytes())
        with lmdb.open(str(path / "lmdb"), readonly=True, lock=False, readahead=False) as env:
            with env.begin() as txn:
                state, action = prepare_joint_episode(txn, meta, args.future_offset_frames, args.gripper_max_width_m)
                for source_camera in CAMERAS.values():
                    image, _ = decode_source(txn, source_camera, 0)
                    if source_camera in source_shapes and source_shapes[source_camera] != image.shape:
                        raise ValueError(f"Resolution changes across source episodes: {source_camera}")
                    source_shapes[source_camera] = image.shape
        reports.append({"episode_index": index, "source": str(path), "source_frames": int(meta["num_steps"]),
                        "frames": len(state), "dropped_tail_frames": args.future_offset_frames,
                        "source_language_instruction": meta["language_instruction"],
                        "source_detailed_language_instruction": meta.get("detailed_language_instruction"),
                        "source_version": meta.get("version"),
                        "numeric_sha256": hashlib.sha256(state.tobytes() + action.tobytes()).hexdigest()})
    print(json.dumps({"episodes": len(reports), "source_frames": sum(r["source_frames"] for r in reports),
                      "output_frames": sum(r["frames"] for r in reports), "source_shapes": source_shapes}), flush=True)

    from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset
    from lerobot.configs.video import RGBEncoderConfig
    if CODEBASE_VERSION != "v3.0":
        raise RuntimeError(f"Expected LeRobot v3.0, got {CODEBASE_VERSION}")
    dataset = LeRobotDataset.create(repo_id=args.repo_id, root=args.output.resolve(), fps=30,
                                   robot_type="bi_piper_follower", features=features_for_shape(args.image_height, args.image_width),
                                   use_videos=True, video_backend="pyav", streaming_encoding=True,
                                   rgb_encoder=RGBEncoderConfig(vcodec="h264", pix_fmt="yuv420p", g=30, crf=23, preset="veryfast"),
                                   encoder_threads=2, encoder_queue_maxsize=30, metadata_buffer_size=10)
    zero_policy = np.zeros(14, dtype=np.float32)
    # Code 3 is assigned as the offline-expert training category, not read as
    # a human intervention event. Collector and manifest preserve synthetic origin.
    offline_expert_code = np.array([3], dtype=np.float32)
    no_intervention = np.array([0], dtype=np.float32)
    try:
        for report, path in zip(reports, episodes):
            meta = pickle.loads((path / "meta_info.pkl").read_bytes())
            with lmdb.open(str(path / "lmdb"), readonly=True, lock=False, readahead=False) as env:
                with env.begin() as txn:
                    state, action = prepare_joint_episode(txn, meta, args.future_offset_frames, args.gripper_max_width_m)
                    image_report = {camera: {"decoded_frames": 0, "source_identical_encoded_pairs": 0,
                                             "identical_pairs_during_motion": 0} for camera in CAMERAS}
                    poses = read_array(txn, "T_base_ee_fl", meta["num_steps"], (4, 4))
                    from scipy.spatial.transform import Rotation
                    distance = np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1)
                    angle = Rotation.from_matrix(poses[:-1, :3, :3].transpose(0, 2, 1) @ poses[1:, :3, :3]).magnitude()
                    motion = (distance > .0005) | (angle > np.deg2rad(.5))
                    previous = {}
                    for idx in range(len(state)):
                        frame = {"observation.state": state[idx], "action": action[idx],
                                 "complementary_info.policy_action": zero_policy,
                                 "complementary_info.is_intervention": no_intervention,
                                 "complementary_info.state": offline_expert_code,
                                 "complementary_info.collector_policy_id": COLLECTOR, "task": task}
                        for camera, source_camera in CAMERAS.items():
                            image, encoded = decode_source(txn, source_camera, idx)
                            if image.shape != source_shapes[source_camera]:
                                raise ValueError(f"Resolution changes at {path}/{source_camera}/{idx}")
                            image_report[camera]["decoded_frames"] += 1
                            if previous.get(camera) == encoded:
                                image_report[camera]["source_identical_encoded_pairs"] += 1
                                if motion[idx-1]:
                                    image_report[camera]["identical_pairs_during_motion"] += 1
                            previous[camera] = encoded
                            frame[f"observation.images.{camera}"] = letterbox(image, args.image_height, args.image_width)
                        dataset.add_frame(frame)
                    report["cameras"] = image_report
            dataset.save_episode(parallel_encoding=True)
            print(f"Saved {report['episode_index']+1}/{len(reports)}: {path.name}: {report['frames']} frames", flush=True)
    finally:
        dataset.finalize()
    manifest = {"format": "LeRobotDataset v3.0", "lerobot_version": importlib.metadata.version("lerobot"),
                "source_root": str(args.source.resolve()), "reference_session": str(args.reference_session.resolve()),
                "repo_id": args.repo_id, "robot_type": "bi_piper_follower", "source_type": "SIMULATION_SCRIPTED_EXPERT",
                "fps": 30, "episodes": len(reports), "source_frames": sum(r["source_frames"] for r in reports),
                "frames": sum(r["frames"] for r in reports), "state_dim": 14, "action_dim": 14,
                "state_names": STATE_NAMES, "state": "measured left joints + normalized gripper, then measured right joints + normalized gripper at source frame t",
                "action": f"measured feedback at source frame t+{args.future_offset_frames}; absolute joint positions, not TCP deltas or controller commands",
                "action_future_offset_frames": args.future_offset_frames,
                "action_future_offset_seconds": args.future_offset_frames / 30,
                "joint_unit_output": "radian", "gripper_unit_output": "normalized",
                "gripper_max_width_m": args.gripper_max_width_m,
                "camera_mapping": CAMERAS, "image_shape_hwc": [args.image_height, args.image_width, 3],
                "image_preprocessing": "aspect-ratio-preserving resize and centered zero padding; RGB source channels preserved",
                "task": task, "collector_policy_id": COLLECTOR,
                "control_code_assignment": {"value": 3, "meaning": "assigned offline-expert training category", "source_control_code_present": False,
                                            "human_demonstration": False, "is_intervention": 0, "policy_action": "zero placeholder"},
                "source_timing": "nominal simulation steps at 30 Hz; no camera acquisition timestamps; not asynchronous real-stream nearest-neighbor alignment",
                "image_state_synchronization_validated": False, "episode_reports": reports}
    (args.output / "conversion_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"Dataset ready: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
