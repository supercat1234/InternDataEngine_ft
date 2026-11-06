"""Joint uniform sampling of object footprints in one shared tabletop region."""
import numpy as np


def sample_separated_disks(bounds, radii, clearance=.01, max_attempts=10000, rng=None):
    """Reject the entire proposal to avoid biasing whichever object is placed last."""
    rng = rng if rng is not None else np.random
    bounds, radii = np.asarray(bounds, dtype=float), np.asarray(radii, dtype=float)
    if bounds.shape != (2, 2) or not np.isfinite(bounds).all() or np.any(bounds[1] <= bounds[0]):
        raise ValueError('Shared XY bounds must be finite [[xmin, ymin], [xmax, ymax]]')
    if radii.ndim != 1 or len(radii) < 2 or not np.isfinite(radii).all() or np.any(radii <= 0):
        raise ValueError('Shared-region object radii must be finite and positive')
    if not np.isfinite(clearance) or clearance < 0 or isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts <= 0:
        raise ValueError('Shared-region clearance must be nonnegative and max_attempts a positive integer')
    low, high = bounds[0] + radii[:, None], bounds[1] - radii[:, None]
    if np.any(high < low):
        raise ValueError('An object footprint is larger than the shared region')
    for attempt in range(1, max_attempts + 1):
        positions = rng.uniform(low, high)
        if all(np.linalg.norm(positions[i]-positions[j]) >= radii[i]+radii[j]+clearance
               for i in range(len(radii)) for j in range(i)):
            return positions, attempt
    raise ValueError(f'Cannot place nonoverlapping objects in shared region after {max_attempts} attempts; enlarge bounds or reduce clearance')
