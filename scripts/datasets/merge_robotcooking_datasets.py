#!/usr/bin/env python3
"""Merge compatible simulation/real LeRobot v3 datasets and preserve provenance.

Every input episode is retained once. Numeric normalization statistics, including
quantiles, are recomputed from the combined rows rather than averaged from sources.
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from lerobot.datasets.dataset_tools import merge_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--simulation', type=Path, required=True)
    parser.add_argument('--real', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repo-id', required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        parser.error('Output already exists; choose a new directory')
    definitions = [('simulation', args.simulation.resolve()), ('real', args.real.resolve())]
    datasets, manifests = [], []
    for domain, root in definitions:
        manifest = json.loads((root/'conversion_manifest.json').read_text())
        ds = LeRobotDataset(repo_id=manifest['repo_id'], root=root, video_backend='pyav')
        if ds.fps != 30 or tuple(ds.features['action']['shape']) != (14,) or tuple(ds.features['observation.state']['shape']) != (14,):
            raise ValueError(f'{root}: expected 30 Hz and 14-dimensional joint states/actions')
        if manifest['action_future_offset_frames'] != 2:
            raise ValueError(f'{root}: expected future-offset=2')
        if manifest['joint_unit_output'] != 'radian' or manifest['gripper_unit_output'] != 'normalized':
            raise ValueError(f'{root}: inconsistent units')
        datasets.append(ds)
        manifests.append(manifest)
        print('SOURCE', domain, ds.num_episodes, len(ds), flush=True)
    tasks = list(datasets[0].meta.tasks.index)
    if tasks != list(datasets[1].meta.tasks.index) or len(tasks) != 1:
        raise ValueError('Input tasks must match exactly')
    merged = merge_datasets(datasets, output_repo_id=args.repo_id, output_dir=output,
                            concatenate_videos=True, concatenate_data=True)
    assert merged.num_episodes == sum(ds.num_episodes for ds in datasets)
    assert len(merged) == sum(len(ds) for ds in datasets)
    print('MERGED', merged.num_episodes, len(merged), flush=True)
    table = pa.concat_tables([pq.read_table(p) for p in sorted((output/'data').rglob('*.parquet'))])
    stats_path = output/'meta/stats.json'
    stats = json.loads(stats_path.read_text())
    numeric_keys = []
    for key, feature in merged.features.items():
        if feature['dtype'] not in ('float32', 'float64', 'int64', 'int32', 'bool'):
            continue
        values = np.asarray(table[key].to_pylist(), dtype=np.float64).reshape(len(merged), -1)
        if not np.isfinite(values).all():
            raise ValueError(f'Non-finite values in {key}')
        entry = {'min': values.min(axis=0).tolist(), 'max': values.max(axis=0).tolist(),
                 'mean': values.mean(axis=0).tolist(), 'std': values.std(axis=0).tolist(), 'count': [len(merged)]}
        quantiles = np.quantile(values, [.01, .1, .5, .9, .99], axis=0)
        entry.update({name: q.tolist() for name, q in zip(['q01', 'q10', 'q50', 'q90', 'q99'], quantiles)})
        stats[key] = entry
        numeric_keys.append(key)
    stats_path.write_text(json.dumps(stats, indent=2)+'\n')
    source_rows, source_reports = [], []
    episode_offset, frame_offset = 0, 0
    for (domain, root), ds, source_manifest in zip(definitions, datasets, manifests):
        episodes = pa.concat_tables([pq.read_table(p) for p in sorted((root/'meta/episodes').rglob('*.parquet'))]).to_pylist()
        episodes.sort(key=lambda row: row['episode_index'])
        for row in episodes:
            i = int(row['episode_index'])
            source_episode = source_manifest['episode_reports'][i]
            raw_source = (source_episode['source'] if domain == 'simulation' else
                          str(Path(source_manifest['source_root'])/source_episode['episode']))
            source_rows.append({'episode_index': episode_offset+i, 'source_domain': domain,
                                'source_dataset_root': str(root), 'source_episode_index': i,
                                'raw_episode_path': raw_source, 'frames': int(row['length']),
                                'dataset_from_index': frame_offset+int(row['dataset_from_index']),
                                'dataset_to_index': frame_offset+int(row['dataset_to_index'])})
        source_reports.append({'domain': domain, 'dataset_root': str(root), 'repo_id': ds.repo_id,
                               'episodes': ds.num_episodes, 'frames': len(ds), 'frame_fraction': len(ds)/len(merged),
                               'episode_index_start': episode_offset, 'episode_index_end_exclusive': episode_offset+ds.num_episodes,
                               'frame_index_start': frame_offset, 'frame_index_end_exclusive': frame_offset+len(ds)})
        episode_offset += ds.num_episodes
        frame_offset += len(ds)
        destination = output/'meta/sources'/domain
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root/'conversion_manifest.json', destination/'conversion_manifest.json')
        if (root/'meta/validation_summary.json').is_file():
            shutil.copyfile(root/'meta/validation_summary.json', destination/'validation_summary.json')
    pq.write_table(pa.Table.from_pylist(source_rows), output/'meta/episode_sources.parquet')
    mixture = {'created_date_local': datetime.now(ZoneInfo('Asia/Shanghai')).date().isoformat(),
               'dataset_version': 'v3.0', 'repo_id': args.repo_id, 'episodes': merged.num_episodes,
               'frames': len(merged), 'fps': 30, 'duration_s': len(merged)/30, 'task': tasks[0],
               'state_dim': 14, 'action_dim': 14, 'future_offset_frames': 2,
               'mixing_strategy': 'all episodes retained once; simulation then real; no resampling or artificial balancing',
               'concatenate_videos': True, 'concatenate_data': True,
               'numeric_statistics': 'recomputed exactly from all merged rows, including q01/q99',
               'visual_statistics': 'aggregated from input datasets', 'recomputed_numeric_features': numeric_keys,
               'source_provenance_feature': 'complementary_info.collector_policy_id', 'sources': source_reports}
    (output/'mixture_manifest.json').write_text(json.dumps(mixture, indent=2, ensure_ascii=False)+'\n')
    print('READY', output, flush=True)


if __name__ == '__main__':
    main()
