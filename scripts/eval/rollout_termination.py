"""Simulation-time stopping: placement success followed by a fixed five-second tail."""
import math


class RolloutTermination:
    def __init__(self, max_steps, fps):
        if max_steps < 1 or not math.isfinite(fps) or fps <= 0:
            raise ValueError('Step limit and FPS must be positive')
        self.max_steps = max_steps
        self.extra_frames = max(1, round(5. * fps))
        self.success_step = None

    @property
    def limit(self):
        return self.max_steps if self.success_step is None else self.success_step + self.extra_frames

    def update(self, step, placement_success):
        if self.success_step is None:
            if placement_success:
                self.success_step = step
            return None
        return 'post_success_timeout' if step >= self.limit else None


class PlacementProgress:
    """Track two success levels; only ordinary success starts the fixed tail."""
    def __init__(self, max_steps, fps, stable_seconds):
        self.termination = RolloutTermination(max_steps, fps)
        self.required = max(1, round(stable_seconds * fps))
        self.stable = self.strict_stable = 0
        self.max_stable = self.max_strict_stable = 0
        self.strict_success_step = None

    @property
    def limit(self):
        return self.termination.limit

    @property
    def success_step(self):
        return self.termination.success_step

    def update(self, step, placement):
        self.stable = self.stable + 1 if placement['candidate'] else 0
        self.strict_stable = self.strict_stable + 1 if placement['strict_candidate'] else 0
        self.max_stable = max(self.max_stable, self.stable)
        self.max_strict_stable = max(self.max_strict_stable, self.strict_stable)
        if self.strict_success_step is None and self.strict_stable >= self.required:
            self.strict_success_step = step
        return self.termination.update(step, self.stable >= self.required)

    def outcome(self, placement):
        success = self.success_step is not None and bool(placement and placement['candidate'])
        strict = success and self.strict_success_step is not None and bool(placement['strict_candidate'])
        return {'success': success, 'strict_success': strict}
