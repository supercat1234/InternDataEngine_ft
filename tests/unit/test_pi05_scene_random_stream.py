import json
from pathlib import Path
import random
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/eval'))
from scene_random_stream import capture_state, initial_state, restore_state


class SceneStreamTests(unittest.TestCase):
    def test_isolated_episodes_continue_the_same_scene_sequence(self):
        def sample_scene():
            return [random.uniform(.8, 1.2), random.choice(['wood1', 'wood2']),
                    np.random.uniform(-.35, -.2), np.random.uniform(-.15, .1)]

        restore_state(initial_state(42), 42)
        expected = [sample_scene() for _ in range(5)]
        state = initial_state(42)
        actual = []
        for episode in range(5):
            # Simulate a fresh process and JSON disk roundtrip each episode.
            random.seed(999)
            np.random.seed(999)
            restore_state(json.loads(json.dumps(state)), 42)
            actual.append(sample_scene())
            state = capture_state(42, episode + 1)
        self.assertEqual(actual, expected)
        self.assertNotEqual(actual[0], actual[1])
        self.assertEqual(state['scenes_started'], 5)

    def test_worker_streams_are_repeatable_and_independent(self):
        values = []
        for seed in (42,43,44,42):
            restore_state(initial_state(seed), seed)
            values.append([random.random(), np.random.random()])
        self.assertEqual(values[0],values[3])
        self.assertNotEqual(values[0],values[1])
        with self.assertRaises(ValueError):
            restore_state(initial_state(42),43)


if __name__ == '__main__':
    unittest.main()
