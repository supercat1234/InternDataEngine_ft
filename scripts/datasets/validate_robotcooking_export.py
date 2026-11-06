#!/usr/bin/env python3
"""Verify a SimBox-to-RobotCooking export against its sources and real converter."""

import argparse
from collections import defaultdict
from datetime import datetime
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import pickle
import sys
from zoneinfo import ZoneInfo

import av
import lmdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from export_cup_plate_robotcooking import decode_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--reference-converter", type=Path,
                        default=Path('/home/agilex/RL/robotcooking/sdk/lerobotv3_sdk/convert_robotcooking.py'))
    args = parser.parse_args()
    root = args.dataset.resolve()
    manifest = json.loads((root / "conversion_manifest.json").read_text())
    info = json.loads((root / "meta/info.json").read_text())
    reference_root = Path(manifest["reference_session"])
    tasks = {json.loads(p.read_text())["instruction"] for p in reference_root.glob('episode_*/metadata.json')}
    assert tasks == {manifest["task"]}
    assert info["codebase_version"] == "v3.0" and info["fps"] == manifest["fps"] == 30
    assert info["total_episodes"] == manifest["episodes"]
    assert info["total_frames"] == manifest["frames"]
    assert manifest["action_future_offset_frames"] == 2
    assert manifest["joint_unit_output"] == "radian" and manifest["gripper_unit_output"] == "normalized"
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("robotcooking_reference_validation", args.reference_converter)
    reference = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = reference
    spec.loader.exec_module(reference)
    height, width, _ = manifest["image_shape_hwc"]
    reference_features = reference._features_for_selection((height, width, 3), ("left", "right"), tuple(reference.CAMERA_SOURCE_KEYS))
    automatic = {"timestamp", "frame_index", "episode_index", "index", "task_index"}
    assert set(info["features"]) - automatic == set(reference_features)
    for key, spec in reference_features.items():
        actual = info["features"][key]
        assert actual["dtype"] == spec["dtype"]
        assert tuple(actual["shape"]) == tuple(spec["shape"])
        assert actual["names"] == spec["names"]
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(repo_id=manifest["repo_id"], root=root, video_backend="pyav")
    assert len(ds) == manifest["frames"] and ds.num_episodes == manifest["episodes"]
    table = pa.concat_tables([pq.read_table(p) for p in sorted((root/'meta/episodes').rglob('*.parquet'))])
    episode_metadata = {row["episode_index"]: row for row in table.to_pylist()}
    assert len(episode_metadata) == manifest["episodes"]
    source_episodes = sorted(p.parent for p in Path(manifest["source_root"]).rglob('meta_info.pkl'))
    selected_sources = [r["source"] for r in manifest["episode_reports"]]
    # The exporter can limit the episode count, and a collector can append new
    # episodes after conversion starts. Validate the recorded selection itself.
    assert len(selected_sources) == len(set(selected_sources)) == manifest["episodes"]
    assert selected_sources == sorted(selected_sources)
    assert set(selected_sources).issubset({str(p) for p in source_episodes})
    video_counts = defaultdict(int)
    sample_errors = {key: [] for key in manifest["camera_mapping"]}
    checks, offset, vector_min, vector_max = [], 0, np.full(14,np.inf), np.full(14,-np.inf)
    for report in manifest["episode_reports"]:
        episode = report["episode_index"]
        path, n = Path(report["source"]), report["frames"]
        meta = pickle.loads((path/'meta_info.pkl').read_bytes())
        assert meta["num_steps"] - 2 == n
        with lmdb.open(str(path/'lmdb'),readonly=True,lock=False,readahead=False) as env:
            with env.begin() as txn:
                source_parts = []
                for side in ("left", "right"):
                    q = np.asarray(pickle.loads(txn.get(f"states.{side}_joint.position".encode())),dtype=np.float64)
                    grip = np.asarray(pickle.loads(txn.get(f"states.{side}_gripper.position".encode())),dtype=np.float64).reshape(-1,1)
                    source_parts.extend([q, np.clip(grip/manifest["gripper_max_width_m"],0,1)])
                full_state = np.concatenate(source_parts,axis=1).astype(np.float32)
                assert full_state.shape == (meta["num_steps"],14) and np.isfinite(full_state).all()
                vector_min = np.minimum(vector_min,full_state.min(axis=0))
                vector_max = np.maximum(vector_max,full_state.max(axis=0))
                rows = ds.hf_dataset[offset:offset+n]
                np.testing.assert_allclose(np.asarray(rows['observation.state']),full_state[:-2],atol=1e-7)
                np.testing.assert_allclose(np.asarray(rows['action']),full_state[2:],atol=1e-7)
                np.testing.assert_array_equal(np.asarray(rows['complementary_info.policy_action']),np.zeros((n,14)))
                np.testing.assert_array_equal(np.asarray(rows['complementary_info.state']).reshape(-1),np.full(n,3))
                np.testing.assert_array_equal(np.asarray(rows['complementary_info.is_intervention']).reshape(-1),np.zeros(n))
                assert set(rows['complementary_info.collector_policy_id']) == {manifest['collector_policy_id']}
                for key, values in [('frame_index',np.arange(n)),('episode_index',np.full(n,episode)),
                                    ('index',np.arange(offset,offset+n)),('task_index',np.zeros(n))]:
                    np.testing.assert_array_equal(np.asarray(rows[key]).reshape(-1),values)
                np.testing.assert_allclose(np.asarray(rows['timestamp']).reshape(-1),np.arange(n)/30,atol=3e-6)
                for local in [0,n//2,n-1]:
                    sample = ds[offset+local]
                    assert sample['task'] == manifest['task']
                    assert tuple(sample['observation.state'].shape) == tuple(sample['action'].shape) == (14,)
                    # Independently compare to the reference converter's vector packer.
                    for shift, key in [(0,'observation.state'),(2,'action')]:
                        source_i = local+shift
                        packed = []
                        for start in [0,7]:
                            packed.append(reference._arm_vector({'JOINT':full_state[source_i,start:start+6].tolist(),
                                                                 'GRIPPER':float(full_state[source_i,start+6])},
                                                                'radian','normalized',100.0,'SimBox validation'))
                        np.testing.assert_allclose(sample[key].numpy(),np.concatenate(packed),atol=1e-7)
                    for camera, source_camera in manifest['camera_mapping'].items():
                        original, _ = decode_source(txn,source_camera,local)
                        expected = reference._letterbox(original,(height,width),pad_value=0)
                        actual = sample['observation.images.'+camera].numpy().transpose(1,2,0)*255
                        assert actual.shape == expected.shape == (height,width,3)
                        error = float(np.abs(actual-expected.astype(np.float32)).mean())
                        assert error < 12,(episode,local,camera,error)
                        sample_errors[camera].append(error)
        em = episode_metadata[episode]
        assert em['length'] == n and em['dataset_from_index'] == offset and em['dataset_to_index'] == offset+n
        for camera in manifest['camera_mapping']:
            key = 'observation.images.'+camera
            prefix = 'videos/'+key+'/'
            start, end = em[prefix+'from_timestamp'], em[prefix+'to_timestamp']
            np.testing.assert_allclose(end-start,n/30,atol=2e-4)
            relative = info['video_path'].format(video_key=key,chunk_index=em[prefix+'chunk_index'],file_index=em[prefix+'file_index'])
            video_counts[relative] += n
        checks.append({'episode_index':episode,'source':str(path),'source_frames':meta['num_steps'],
                       'frames':n,'dataset_from_index':offset,'dataset_to_index':offset+n})
        offset += n
        print(f'NUMERIC {episode+1}/{manifest["episodes"]}: {n} frames',flush=True)
    assert offset == len(ds)
    videos = []
    for relative, expected_frames in sorted(video_counts.items()):
        path = root/relative
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            assert stream.codec_context.name == 'h264' and float(stream.average_rate) == 30
            assert (stream.height,stream.width) == (height,width)
            timestamps = np.array([float(f.pts*f.time_base) for f in container.decode(stream)])
        assert len(timestamps) == expected_frames,(relative,len(timestamps),expected_frames)
        np.testing.assert_allclose(timestamps,np.arange(expected_frames)/30,atol=1e-5)
        videos.append({'path':relative,'frames':len(timestamps),'fps':30,'duration_s':len(timestamps)/30,'size_bytes':path.stat().st_size})
        print(f'VIDEO {relative}: {len(timestamps)} frames',flush=True)
    assert sum(v['frames'] for v in videos) == len(ds)*3
    stats = json.loads((root/'meta/stats.json').read_text())
    for key in ['observation.state','action']:
        for name in ['mean','std','q01','q99']:
            assert np.asarray(stats[key][name]).shape == (14,) and np.isfinite(stats[key][name]).all()
    repeat_totals = {camera:sum(r['cameras'][camera]['identical_pairs_during_motion'] for r in manifest['episode_reports'])
                     for camera in manifest['camera_mapping']}
    summary = {'checked_date_local':datetime.now(ZoneInfo('Asia/Shanghai')).date().isoformat(),'dataset_version':'v3.0','lerobot_version':importlib.metadata.version('lerobot'),
               'episodes':ds.num_episodes,'frames':len(ds),'fps':30,'duration_s':len(ds)/30,
               'reference_session':str(reference_root),'reference_converter':str(args.reference_converter.resolve()),
               'reference_feature_schema_matches':True,'action_dimension':14,'state_dimension':14,
               'future_offset_frames':2,'all_numeric_rows_match_raw_sources':True,'all_episode_boundaries_and_timestamps_valid':True,
               'task_matches_reference':True,'task':manifest['task'],
               'official_reader_samples_per_episode':3,'all_videos_fully_decoded':True,'stats_include_q01_q99':True,
               'image_sample_max_mean_absolute_pixel_error':{k:max(v) for k,v in sample_errors.items()},
               'image_shape_hwc':[height,width,3],'unique_numeric_episodes':len({e['numeric_sha256'] for e in manifest['episode_reports']}),
               'source_identical_encoded_images_during_motion':repeat_totals,
               'image_state_synchronization_validated':False,'state_min':vector_min.tolist(),'state_max':vector_max.tolist(),
               'provenance_difference':'scripted simulation, not human teleoperation; offline-expert code 3 assigned, collector_policy_id identifies simulation',
               'reference_storage_difference':'reference session is raw asynchronous JSON/MP4; compatibility is against its LeRobot converter output schema',
               'videos':videos,'episode_checks':checks}
    (root/'meta/validation_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    print('SUMMARY '+json.dumps({k:v for k,v in summary.items() if k not in ['episode_checks']},indent=2),flush=True)


if __name__ == '__main__':
    main()
