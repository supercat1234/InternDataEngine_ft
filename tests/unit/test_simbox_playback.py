"""Exercise the real render loop without starting Isaac Sim or CuRobo."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("playback", ROOT / "workflows/simbox/core/utils/playback.py")
playback = importlib.util.module_from_spec(spec)
spec.loader.exec_module(playback)
tree = ast.parse((ROOT / "workflows/simbox_dual_workflow.py").read_text())
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SimBoxDualWorkFlow")
method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "plan_with_render")


def log_observation(logger, obs, action, controllers, step_idx):
    logger.append(step_idx)


namespace = {"PlaybackGate": playback.PlaybackGate, "np": np, "log_dual_obs": log_observation}
exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(ROOT), "exec"), namespace)


class World:
    def __init__(self, pauses=(), initially_paused=False, exit_on_pause=None):
        self.playing = not initially_paused
        self.stopped = False
        self.running = True
        self.contacts_fresh = self.playing
        self.current_time_step_index = 0
        self.step_calls = 0
        self.ui_frames = 0
        self.pauses = dict(pauses)
        self.exit_on_pause = exit_on_pause
        self.paused_snapshot = None
        self.app = SimpleNamespace(is_running=lambda: self.running)

    def is_playing(self):
        return self.playing

    def is_stopped(self):
        return self.stopped

    def get_observations(self):
        assert self.playing and self.contacts_fresh
        return {}

    def render(self):
        assert not self.playing
        snapshot = self.workflow.snapshot()
        if self.paused_snapshot is None:
            self.paused_snapshot = snapshot
        assert snapshot == self.paused_snapshot, "task advanced during paused UI rendering"
        self.ui_frames += 1
        if self.ui_frames % 3 == 0:
            if self.exit_on_pause == "stop":
                self.stopped = True
            elif self.exit_on_pause == "close":
                self.running = False
            else:
                self.playing = True
                # Contact cache stays invalid until the next physics step.

    def step(self, render=True):
        assert self.playing, "World.step called while paused"
        self.step_calls += 1
        event = self.pauses.get(self.step_calls)
        if event != "before":
            self.current_time_step_index += 1
            self.contacts_fresh = True
        if event:
            self.playing = False
            self.contacts_fresh = False
            self.paused_snapshot = None


class Workflow:
    plan_with_render = namespace["plan_with_render"]

    def __init__(self, world):
        self.world = world
        world.workflow = self
        self.logger = []
        self.images = []
        self.forward_calls = self.updates = self.plans = self.actions = 0
        self.controllers = {}
        self.task_cfg = {"data": {"max_episode_length": 100}}
        self.task = SimpleNamespace(apply_action=self.apply_action)
        self.world_recorder = SimpleNamespace(record=lambda: None)
        skill = SimpleNamespace(
            manip_list=[None], controller=SimpleNamespace(forward=self.forward),
            is_ready=lambda: True, is_feasible=lambda: True, is_record=lambda: True,
        )
        self.skills = [{"robot": [[[skill]]]}]

    def snapshot(self):
        return self.forward_calls, self.updates, self.plans, self.actions, len(self.logger), len(self.images)

    def plan_first_skill(self, skills, should_continue):
        assert self.world.is_playing() and self.world.contacts_fresh
        self.plans += 1
        return should_continue

    def forward(self, command):
        assert self.world.is_playing() and self.world.contacts_fresh, "read stale contacts"
        self.forward_calls += 1
        return {"joint_positions": np.zeros(1), "joint_indices": np.zeros(1, dtype=int)}

    def apply_action(self, action):
        assert self.world.is_playing()
        self.actions += 1

    def _record_rgb_depth(self, step):
        assert self.world.is_playing() and self.world.contacts_fresh
        self.images.append(step)

    def update_skill_states(self, skills, success, should_continue):
        assert self.world.is_playing() and self.world.contacts_fresh
        self.updates += 1
        if self.updates == 3:
            self.skills = []
        return success, should_continue


class PlaybackTests(unittest.TestCase):
    def test_pause_preserves_commands_and_frame_counts(self):
        baseline = Workflow(World())
        result = baseline.plan_with_render()
        # Initial pause, warmup, controller step, interrupted refresh, and
        # final success frames all use the same gate.
        scenarios = [([], True), ([(1, "before")], False), ([(11, "before")], False),
                     ([(11, "after")], False), ([(11, "before"), (12, "before")], False),
                     ([(15, "before")], False), ([(19, "after")], False)]
        for pauses, initially_paused in scenarios:
            with self.subTest(pauses=pauses, initially_paused=initially_paused):
                wf = Workflow(World(pauses, initially_paused))
                self.assertEqual(wf.plan_with_render(), result)
                self.assertEqual(wf.snapshot(), baseline.snapshot())
                self.assertEqual(wf.logger, baseline.logger)
                self.assertEqual(wf.images, baseline.images)
                self.assertGreater(wf.world.ui_frames, 0)

    def test_success_tail_has_contiguous_frames_and_matches_reported_length(self):
        wf = Workflow(World())
        length = wf.plan_with_render()
        self.assertEqual(wf.images, list(range(length)))
        self.assertEqual(wf.logger, list(range(length)))

    def test_stop_or_close_cancels_without_advancing_skills(self):
        for exit_kind in ["stop", "close"]:
            with self.subTest(exit_kind=exit_kind):
                wf = Workflow(World([(11, "before")], exit_on_pause=exit_kind))
                self.assertEqual(wf.plan_with_render(), 0)
                self.assertEqual(wf.length, 0)
                self.assertEqual(wf.forward_calls, 1)
                self.assertEqual(wf.updates, 0)


if __name__ == "__main__":
    unittest.main()
