#!/usr/bin/env python3
"""Verify a merged RobotCooking dataset against its input datasets."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def episode_table(root):
    return pa.concat_tables([pq.read_table(p) for p in sorted((root/'meta/episodes').rglob('*.parquet'))])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    args = parser.parse_args()
    root = args.dataset.resolve()
    manifest = json.loads((root/'mixture_manifest.json').read_text())
    merged = LeRobotDataset(repo_id=manifest['repo_id'], root=root, video_backend='pyav')
    total = manifest['frames']
    assert len(merged) == total and merged.num_episodes == manifest['episodes'] and merged.fps == manifest['fps'] == 30
    assert tuple(merged.features['observation.state']['shape']) == tuple(merged.features['action']['shape']) == (14,)
    assert list(merged.meta.tasks.index) == [manifest['task']]
    meta = {int(row['episode_index']): row for row in episode_table(root).to_pylist()}
    sources = pq.read_table(root/'meta/episode_sources.parquet').to_pylist()
    assert len(meta) == len(sources) == manifest['episodes']
    videos_expected = defaultdict(list)
    expected_collectors = Counter()
    comparisons, maximum_pixel_error = [], 0.0
    height, width, _ = merged.features[merged.meta.video_keys[0]]['shape']
    for source in manifest['sources']:
        ds = LeRobotDataset(repo_id=source['repo_id'], root=Path(source['dataset_root']), video_backend='pyav')
        assert len(ds) == source['frames'] and ds.num_episodes == source['episodes']
        frame_offset, episode_offset = source['frame_index_start'], source['episode_index_start']
        source_meta = episode_table(ds.root).to_pylist()
        source_meta.sort(key=lambda row: row['episode_index'])
        for row in source_meta:
            i, n = int(row['episode_index']), int(row['length'])
            j = i + episode_offset
            src_start = int(row['dataset_from_index'])
            dst_start = frame_offset + src_start
            original_rows = ds.hf_dataset[src_start:src_start+n]
            mixed_rows = merged.hf_dataset[dst_start:dst_start+n]
            assert set(original_rows) == set(mixed_rows)
            for key, values in original_rows.items():
                actual = mixed_rows[key]
                if key == 'index':
                    np.testing.assert_array_equal(np.asarray(actual).reshape(-1), np.asarray(values).reshape(-1)+frame_offset)
                elif key == 'episode_index':
                    np.testing.assert_array_equal(np.asarray(actual).reshape(-1), np.asarray(values).reshape(-1)+episode_offset)
                elif key == 'complementary_info.collector_policy_id':
                    assert list(actual) == list(values)
                    expected_collectors.update(values)
                else:
                    np.testing.assert_array_equal(np.asarray(actual), np.asarray(values))
            em = meta[j]
            assert em['length'] == n and em['dataset_from_index'] == dst_start and em['dataset_to_index'] == dst_start+n
            assert sources[j]['episode_index'] == j and sources[j]['source_domain'] == source['domain']
            assert sources[j]['source_episode_index'] == i and sources[j]['frames'] == n
            assert sources[j]['dataset_from_index'] == dst_start and sources[j]['dataset_to_index'] == dst_start+n
            # Preserve optional source outcome labels, including unknown simulation outcomes.
            assert em.get('episode_success') == row.get('episode_success')
            for local in [0, n//2, n-1]:
                original, mixed = ds[src_start+local], merged[dst_start+local]
                assert original['task'] == mixed['task'] == manifest['task']
                for key in merged.meta.camera_keys:
                    assert tuple(mixed[key].shape) == (3, height, width)
                    error = float(np.max(np.abs(original[key].numpy()-mixed[key].numpy())))
                    assert error < 1e-6, (j, local, key, error)
                    maximum_pixel_error = max(maximum_pixel_error, error)
            for key in merged.meta.video_keys:
                prefix = 'videos/'+key+'/'
                relative = merged.meta.info.video_path.format(video_key=key, chunk_index=em[prefix+'chunk_index'], file_index=em[prefix+'file_index'])
                videos_expected[relative].append((em[prefix+'from_timestamp'], n))
                np.testing.assert_allclose(em[prefix+'to_timestamp']-em[prefix+'from_timestamp'], n/30, atol=2e-4)
            print(f'EPISODE {j+1}/{manifest["episodes"]} {source["domain"]}: {n} rows match', flush=True)
        comparisons.append({'domain': source['domain'], 'episodes': ds.num_episodes, 'frames': len(ds),
                            'all_columns_match_source': True})
    videos = []
    for relative, segments in sorted(videos_expected.items()):
        segments.sort(key=lambda item: item[0])
        expected = np.concatenate([start+np.arange(n)/30 for start, n in segments])
        path = root/relative
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            fps = float(stream.average_rate)
            assert stream.codec_context.name == 'h264' and abs(fps-30) < 1e-3
            assert (stream.height, stream.width) == (height, width)
            actual = np.asarray([float(f.pts*f.time_base) for f in container.decode(stream)])
        assert len(actual) == len(expected)
        max_error = float(np.max(np.abs(actual-expected)))
        assert max_error <= 1e-4, (relative, max_error)
        videos.append({'path': relative, 'frames': len(actual), 'fps': fps,
                       'max_timestamp_error_s': max_error, 'size_bytes': path.stat().st_size})
        print('VIDEO', relative, len(actual), 'timestamp error', max_error, flush=True)
    assert sum(video['frames'] for video in videos) == total*3
    table = pa.concat_tables([pq.read_table(p) for p in sorted((root/'data').rglob('*.parquet'))])
    stats = json.loads((root/'meta/stats.json').read_text())
    for key in manifest['recomputed_numeric_features']:
        values = np.asarray(table[key].to_pylist(), dtype=np.float64).reshape(total, -1)
        assert np.isfinite(values).all()
        for name, expected in [('mean', values.mean(axis=0)), ('std', values.std(axis=0)),
                               ('min', values.min(axis=0)), ('max', values.max(axis=0))]:
            np.testing.assert_allclose(stats[key][name], expected, atol=1e-12)
        for name, q in [('q01', .01), ('q10', .1), ('q50', .5), ('q90', .9), ('q99', .99)]:
            np.testing.assert_allclose(stats[key][name], np.quantile(values, q, axis=0), atol=1e-12)
        assert stats[key]['count'] == [total]
    collectors = Counter(table['complementary_info.collector_policy_id'].to_pylist())
    assert collectors == expected_collectors
    np.testing.assert_array_equal(np.asarray(table['index'].to_pylist()), np.arange(total))
    summary = {'checked_date_local': datetime.now(ZoneInfo('Asia/Shanghai')).date().isoformat(),
               'dataset_version': 'v3.0', 'episodes': merged.num_episodes, 'frames': total, 'fps': 30,
               'duration_s': total/30, 'task': manifest['task'], 'state_dim': 14, 'action_dim': 14,
               'future_offset_frames': manifest['future_offset_frames'],
               'all_merged_numeric_and_string_columns_match_sources': True,
               'episode_boundaries_and_provenance_valid': True, 'source_outcome_labels_preserved': True,
               'official_reader_image_samples_per_episode': 3, 'merged_images_match_source_images_exactly': True,
               'maximum_sample_pixel_error': maximum_pixel_error, 'all_videos_fully_decoded': True,
               'numeric_statistics_recomputed_and_exactly_verified': True, 'collectors': dict(collectors),
               'mixing_strategy': manifest['mixing_strategy'], 'sources': comparisons, 'videos': videos,
               'source_simulation_image_state_synchronization_validated': False}
    (root/'meta/validation_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False)+'\n')
    print('SUMMARY', json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
