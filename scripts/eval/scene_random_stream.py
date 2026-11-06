"""Continue Python/NumPy scene sampling across isolated evaluation processes."""
import json
import random
from pathlib import Path

import numpy as np


def capture_state(seed, scenes_started):
    algorithm, keys, position, has_gauss, cached_gaussian = np.random.get_state()
    return {'version': 1, 'seed': seed, 'scenes_started': scenes_started,
            'python': random.getstate(),
            'numpy': [algorithm, keys.tolist(), position, has_gauss, cached_gaussian]}


def initial_state(seed):
    # Local generators keep concurrent worker initialization independent.
    rng = np.random.RandomState(seed)
    algorithm, keys, position, has_gauss, cached_gaussian = rng.get_state()
    return {'version': 1, 'seed': seed, 'scenes_started': 0,
            'python': random.Random(seed).getstate(),
            'numpy': [algorithm, keys.tolist(), position, has_gauss, cached_gaussian]}


def restore_state(state, seed):
    if state['version'] != 1 or state['seed'] != seed:
        raise ValueError('Scene stream version/seed differs from this evaluation')
    python_version, python_keys, python_gauss = state['python']
    random.setstate((python_version, tuple(python_keys), python_gauss))
    algorithm, keys, position, has_gauss, cached_gaussian = state['numpy']
    np.random.set_state((algorithm, np.asarray(keys, dtype=np.uint32),
                         position, has_gauss, cached_gaussian))


def write_state(path, state):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(state))
    temporary.replace(path)
