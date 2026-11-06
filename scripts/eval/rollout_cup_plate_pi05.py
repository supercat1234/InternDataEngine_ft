#!/usr/bin/env python3
"""One fixed or randomized scene episode. No expert controller, planner, logger or dataset writer."""
import argparse
import json
import os
import signal
import shutil
from pathlib import Path
import sys
import time
import urllib.request
import numpy as np
import yaml
from eval_reporting import allocate_output, capture_log, finalize_videos, save_report, ABORT_REASONS
from video_recording import StreamingVideoWriter
from rollout_termination import PlacementProgress
from scene_random_stream import capture_state, restore_state, write_state
from placement_evaluation import DEFAULT_CRITERIA, evaluate_placement, tcp_from_ee_world
from ood_scene import configure_ood, apply_ood_visuals
from cup_plate_adapter import ACTION_NAMES, CAMERAS, TASK, observation, indices, targets

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / 'workflows/simbox'), str(REPO / 'scripts/deploy')]
from pi05_plain_server import infer_remote


def fixed_config(settings, output):
    config = yaml.safe_load((REPO / settings['source_task']).read_text())
    task = config['tasks'][0]
    task['skills'] = []  # Scene contact setup expects this key; no skills instantiated.
    arena = yaml.safe_load((REPO / task['arena_file']).read_text())
    configure_ood(task, arena, settings, REPO)
    flags = settings.get('randomization', {})
    if not flags.get('layout', False) and task.get('shared_region_sampling'):
        task['shared_region_sampling']['enabled'] = False
    for fixture in arena['fixtures']:
        for key in ('appearance_randomization', 'texture'):
            if key in fixture:
                group = 'table' if key == 'appearance_randomization' else 'floor'
                fixture[key]['apply_randomization'] = bool(flags.get(group, False))
    arena_path = output / 'arena.yaml'
    arena_path.write_text(yaml.safe_dump(arena, sort_keys=False))
    task['arena_file'] = str(arena_path)
    light = task['env_map']['photometric_randomization']
    if not flags.get('lighting', False):
        light.update(enabled=True, intensity_scale_range=[1, 1],
                     color_temperature_ranges={'white': [6000, 6000]}, rotation_jitter_deg=[0, 0, 0])
    task['env_map']['apply_randomization'] = False
    for camera in task['cameras']:
        camera['apply_randomization'] = bool(flags.get('camera', False) and camera['name'] == 'split_aloha_external')
    for region in task['regions']:
        if flags.get('layout', False):
            continue
        pos = settings['cup_position' if region['object'] == 'paper_cup' else 'plate_position']
        region['random_config'].update(pos_range=[pos, pos], yaw_rotation=[settings['yaw_deg']] * 2)
    path = output / 'task.yaml'
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def reproduction_config(source, output):
    """Use archived ranges/material/camera config without touching the original run."""
    config = yaml.safe_load((source / 'task.yaml').read_text())
    shutil.copyfile(source / 'arena.yaml', output / 'arena.yaml')
    config['tasks'][0]['arena_file'] = str(output / 'arena.yaml')
    config['tasks'][0]['skills'] = []
    path = output / 'task.yaml'
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='local_configs/cup_plate_pi05_eval/eval.yaml')
    parser.add_argument('--url', default='http://127.0.0.1:8097')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--max-steps', type=int)
    parser.add_argument('--execute-steps', type=int)
    parser.add_argument('--replay-actions', type=Path, help='Wiring smoke test only: replay a saved [T,14] chunk; no live policy')
    parser.add_argument('--observe-only', action='store_true', help='Save initial inputs, do not infer or actuate')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--reserved-output', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--seed', type=int, help='Override scene random seed')
    parser.add_argument('--episode-index', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--scene-rng-state', type=Path, help='Continue a batch worker scene random stream from JSON')
    parser.add_argument('--randomize', action='store_true', help='Enable layout, lighting, wood, floor, external camera randomization')
    parser.add_argument('--save-debug-arrays', action='store_true', help='Opt in to NPZ/NPY inference debugging files')
    parser.add_argument('--reproduce', type=Path, help='Restore the archived scene/config/RNG of a completed run')
    parser.add_argument('--replay-saved-actions', action='store_true', help='With --reproduce: replay trajectory predictions, without the model')
    args = parser.parse_args()
    os.chdir(REPO)
    settings = yaml.safe_load(Path(args.config).read_text())
    if args.replay_saved_actions and not args.reproduce:
        parser.error('--replay-saved-actions requires --reproduce')
    if args.reproduce:
        if args.seed is not None or args.randomize or args.scene_rng_state:
            parser.error('--reproduce restores its saved seed/randomization; omit seed/randomize/scene-rng-state')
        args.reproduce = args.reproduce.resolve()
        archived = json.loads((args.reproduce / 'run.json').read_text())
        settings = archived['settings']
        if archived.get('args', {}).get('execute_steps') is not None:
            settings['execute_steps'] = archived['args']['execute_steps']
    replay_data = None
    if args.replay_saved_actions:
        replay_data = np.asarray([x['predicted'] for x in json.loads((args.reproduce / 'trajectory.json').read_text())])
    elif args.replay_actions:
        replay_data = np.load(args.replay_actions, allow_pickle=False)
    if args.seed is not None:
        settings['seed'] = args.seed
    if args.randomize:
        settings['randomization'] = dict.fromkeys(('layout', 'lighting', 'table', 'floor', 'camera'), True)
    steps = args.max_steps if args.max_steps is not None else min(settings['max_steps'], round(45 * settings['fps']))
    settings['max_steps'] = steps
    execute = args.execute_steps if args.execute_steps is not None else settings['execute_steps']
    settings['execute_steps'] = execute
    settings['post_success_seconds'] = 5.0
    criteria = DEFAULT_CRITERIA | settings.get('success_criteria', {})
    settings['success_criteria'] = criteria
    progress = PlacementProgress(steps, settings['fps'], criteria['stable_seconds'])
    stable_required = progress.required
    stop_reason = None
    if steps <= 0 or execute <= 0:
        parser.error('Step counts must be positive')
    index_root = Path(settings.get('output_root', '/home/agilex/qhx/evaluation/sim/test')).resolve()
    if args.output is None:
        output = allocate_output(index_root)
    else:
        output = args.output.resolve()
        if args.reserved_output:
            if not output.is_dir() or any(output.iterdir()):
                raise ValueError('Reserved batch output must be an empty directory')
        else:
            output.mkdir(parents=True, exist_ok=False)
    writers = {}
    records = []
    result = {'mode': 'replay' if replay_data is not None else ('observe_only' if args.observe_only else 'closed_loop'),
              'success': False, 'strict_success': False, 'steps': 0, 'reason': 'max_steps', 'stable_frames': 0,
              'max_stable_frames': 0, 'placement_criteria_version': 'dual_success_v4',
              'success_criteria': criteria, 'required_stable_frames': stable_required,
              'episode': args.episode_index, 'failed_check_frames': {}}
    app = None
    durations = []
    run_started = time.perf_counter()
    def interrupt(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupt)
    with capture_log(output / 'rollout.log'):
        try:
            (output / 'run.json').write_text(json.dumps({'settings': settings, 'args': vars(args), 'task': TASK}, default=str, indent=2))
            task_path = reproduction_config(args.reproduce, output) if args.reproduce else fixed_config(settings, output)
            health = None
            if not args.observe_only and replay_data is None:
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                with opener.open(args.url.rstrip('/') + '/health', timeout=10) as response:
                    health = json.load(response)
                if health.get('action_feature_names') != ACTION_NAMES or not health.get('absolute_actions'):
                    raise ValueError('Inference service must expose absolute 14D cup/plate contract')
                if Path(health['checkpoint']).resolve() != Path(settings['checkpoint']).resolve():
                    raise ValueError('Inference checkpoint differs from eval config')
            (output / 'run.json').write_text(json.dumps({'settings': settings, 'args': vars(args) | {'output': str(output)},
                                                        'service': health, 'task': TASK}, default=str, indent=2))
            from isaacsim import SimulationApp
            app = SimulationApp({'headless': args.headless, 'anti_aliasing': 0, 'renderer': 'RayTracedLighting'})
            import cv2
            import carb.settings
            import omni.metrics.assembler.core.bindings._metricsAssembler as metrics
            from omni.isaac.core import World
            from workflows import import_extensions
            from workflows.base import NimbusWorkFlow
            from core.utils.utils import set_random_seed
            from omni.isaac.core.utils.transformations import tf_matrix_from_pose
            from omni.isaac.core.utils.prims import get_prim_at_path
            from pxr import UsdGeom
            carb.settings.get_settings().set_int(metrics.SETTINGS_METRICS_ASSEMBLER_OPERATION_MODE, 0)
            import_extensions('SimBoxDualWorkFlow')
            set_random_seed(settings['seed'])
            rng_path = args.scene_rng_state
            if args.reproduce and (args.reproduce / 'scene_rng_before.json').exists():
                rng_path = args.reproduce / 'scene_rng_before.json'
            if rng_path:
                scene_state = json.loads(rng_path.read_text())
                restore_state(scene_state, settings['seed'])
                write_state(output / 'scene_rng_before.json', scene_state)
            try:
                world = World(physics_dt=1/settings['fps'], rendering_dt=1/settings['fps'], stage_units_in_meters=1.)
                workflow = NimbusWorkFlow.workflows['SimBoxDualWorkFlow'](world, str(task_path), random_seed=settings['seed'], preview_only=True)
                workflow.init_task(0)
                if apply_ood_visuals(workflow.task, task_path, output):
                    # Allow RTX to update material resources before the first
                    # policy observation, without advancing physics or RNG.
                    for _ in range(20):
                        world.render()
            finally:
                # Persist immediately after scene sampling, before policy inference.
                # Failures after initialization still advance this worker's stream.
                if rng_path:
                    next_state = capture_state(settings['seed'], scene_state['scenes_started'] + 1)
                    write_state(output / 'scene_rng_after.json', next_state)
                    if args.scene_rng_state:
                        write_state(args.scene_rng_state, next_state)
            assert not any(hasattr(workflow, key) for key in ('controllers', 'skills', 'logger', 'world_recorder'))
            task = workflow.task
            robot = task.robots['split_aloha']
            selected = indices(robot)
            limits = np.asarray(robot._articulation_view.get_dof_limits()).reshape(-1, 2)[selected]
            if args.save_debug_arrays:
                np.save(output / 'selected_joint_limits.npy', limits)
            cup, plate = task.objects['paper_cup'], task.objects['ceramic_plate']
            ee_prim = get_prim_at_path(robot.fl_ee_path)
            initial = None
    
            def read():
                # Refresh camera parent transforms before rendering the frame used by the policy.
                task.get_observations()
                for _ in range(2):
                    world.render()
                return observation(task.get_observations())
    
            def record(obs):
                for name in CAMERAS:
                    frame = obs[f'observation.images.{name}']
                    if name not in writers:
                        writers[name] = StreamingVideoWriter(output / f'{name}.mp4', settings['fps'])
                    writers[name].write(frame)
    
            obs = read()
            initial = obs['observation.state'].copy()
            if args.save_debug_arrays:
                np.savez_compressed(output / 'initial_observation.npz', **obs)
            if args.observe_only:
                images = {}
                for name in CAMERAS:
                    path = output / f'{name}.png'
                    if not cv2.imwrite(str(path), cv2.cvtColor(obs[f'observation.images.{name}'], cv2.COLOR_RGB2BGR)):
                        raise RuntimeError(f'Cannot save {path}')
                    images[name] = str(path)
                result.update(reason='observe_only', images=images)
                return
            print(f'ROLLOUT_READY output={output}', flush=True)
            trace = (output / 'trajectory.jsonl').open('a', buffering=1)
            while app.is_running() and result['steps'] < progress.limit and stop_reason is None:
                if not world.is_playing():
                    result['reason'] = 'interrupted'
                    break
                started = time.perf_counter()
                if replay_data is None:
                    actions, duration = infer_remote(args.url, obs, TASK, timeout=180)
                else:
                    actions = replay_data[result['steps']:]
                    duration = 0.
                    if len(actions) == 0:
                        stop_reason = 'replay_complete'
                        break
                if replay_data is None:
                    durations.append(float(duration))
                if actions.shape[1:] != (14,) or (replay_data is None and len(actions) < execute):
                    raise ValueError('Invalid action chunk shape/length')
                if args.save_debug_arrays:
                    np.savez_compressed(output / f'chunk_{result["steps"]:06d}.npz', actions=actions, **obs)
                print(f'chunk step={result["steps"]} inference={duration:.3f}s wall={time.perf_counter()-started:.3f}s', flush=True)
                # Physics is deliberately paused in wall time while synchronous inference runs.
                for action in actions[:min(execute, progress.limit-result['steps'])]:
                    applied = targets(action, limits)
                    robot.apply_action(joint_positions=applied, joint_indices=selected)
                    world.step(render=False)
                    if not app.is_running() or not world.is_playing():
                        result['reason'] = 'interrupted'
                        stop_reason = 'interrupted'
                        break
                    obs = read()
                    record(obs)
                    cup_pose = tf_matrix_from_pose(*cup.get_world_pose())
                    plate_pose = tf_matrix_from_pose(*plate.get_world_pose())
                    # Require TCP clearance; opening need not persist after the arm retreats.
                    tcp = robot.get_observations()
                    # Robot-root pose is not the left-arm base pose. Read the EE's
                    # actual world transform to avoid mixing these two frames.
                    ee_world = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(ee_prim)).T
                    tcp_position = tcp_from_ee_world(ee_world, robot.cfg['ee_axis'], robot.tcp_offset)
                    placement = evaluate_placement(cup_pose, plate_pose,
                        (cup.get_linear_velocity(), cup.get_angular_velocity()),
                        (plate.get_linear_velocity(), plate.get_angular_velocity()),
                        tcp_position, criteria)
                    placed, released = placement['placed'], placement['released']
                    result['steps'] += 1
                    first_success = progress.success_step is None
                    stop_reason = progress.update(result['steps'], placement)
                    result.update(stable_frames=progress.stable, strict_stable_frames=progress.strict_stable,
                                  max_stable_frames=progress.max_stable, max_strict_stable_frames=progress.max_strict_stable)
                    result['final_placement'] = placement
                    for name in placement['failed_checks']:
                        counts = result['failed_check_frames']
                        counts[name] = counts.get(name, 0) + 1
                    home = bool(np.max(np.abs(obs['observation.state'][[0,1,2,3,4,5,7,8,9,10,11,12]] - initial[[0,1,2,3,4,5,7,8,9,10,11,12]])) < np.deg2rad(.5))
                    records.append({'step': result['steps'], 'state': obs['observation.state'].tolist(),
                                    'predicted': action.tolist(), 'applied': applied.tolist(), 'cup_pose': cup_pose.tolist(),
                                    'plate_pose': plate_pose.tolist(), 'placed': bool(placed), 'released': bool(released), 'home': home,
                                    'placement': placement,
                                    'tcp_position': tcp_position.tolist(),
                                    'joint_velocities': np.asarray(tcp['qvel'])[selected].tolist()})
                    trace.write(json.dumps(records[-1]) + '\n')
                    if progress.strict_success_step is not None:
                        result.update(strict_success=True, strict_success_step=progress.strict_success_step,
                                      strict_success_simulation_seconds=progress.strict_success_step/settings['fps'])
                    if first_success and progress.success_step is not None:
                        result.update(success=True, reason='stable_placement', placement_reached=True,
                                      success_step=result['steps'], success_simulation_seconds=result['steps']/settings['fps'])
                        print(f"PLACEMENT_SUCCESS step={result['steps']}; continue recording for 5 simulation seconds", flush=True)
                        result['stop_at_step'] = progress.limit
                        result['expected_video_seconds'] = progress.limit / settings['fps']
                    (output / 'progress.json').write_text(json.dumps({**result, 'status': 'running', 'recorded_frames': {name: writer.frames for name, writer in writers.items()}}, indent=2))
                    if stop_reason is not None:
                        break
            result['stop_reason'] = stop_reason or ('window_closed' if not app.is_running() else 'max_steps')
            result['post_success_steps'] = (result['steps'] - progress.success_step
                                           if progress.success_step is not None else 0)
            result['post_success_seconds'] = result['post_success_steps'] / settings['fps']
            result['placement_stable_at_end'] = bool(records and records[-1]['placed'] and records[-1]['released'])
            result['strict_placement_stable_at_end'] = bool(records and records[-1]['placement']['strict_candidate'])
            result.update(progress.outcome(result.get('final_placement')))
            if progress.success_step is not None and not result['placement_stable_at_end']:
                result.update(success=False, reason='placement_lost')
            if not app.is_running():
                result['reason'] = 'window_closed'
        except KeyboardInterrupt:
            result.update(reason='interrupted')
            raise
        except Exception as exc:
            result.update(reason='error', error=f'{type(exc).__name__}: {exc}')
            import traceback
            traceback.print_exc()
            raise
        finally:
            # Complete cleanup once; repeated Ctrl+C/SIGTERM must not leave a
            # half-written result or interrupt removal of a cancelled attempt.
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            result.setdefault('stop_reason', result['reason'])
            if progress.success_step is not None:
                result['post_success_steps'] = result['steps'] - progress.success_step
                result['post_success_seconds'] = result['post_success_steps'] / settings['fps']
            if 'trace' in locals():
                trace.close()
            for writer in writers.values():
                try:
                    writer.release()
                except Exception as exc:
                    result['recording_error'] = str(exc)
            if result['reason'] not in ABORT_REASONS:
                try:
                    result['video_metadata'] = finalize_videos(output, writers)
                except Exception as exc:
                    result['recording_error'] = f'{type(exc).__name__}: {exc}'
                    print(f'VIDEO_ERROR {exc}', flush=True)
            report = save_report(output, result, records, settings, durations, run_started, index_root)
            if report is None:
                print('ATTEMPT_DISCARDED: incomplete evaluation; excluded from success rate', flush=True)
            else:
                print(json.dumps(report), flush=True)
                print(f'RESULT_SAVED {output / "result.json"}', flush=True)
            if app is not None:
                app.close()

if __name__ == '__main__':
    main()
