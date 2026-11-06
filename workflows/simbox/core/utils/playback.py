"""Keep UI rendering separate from task execution while the timeline is paused."""


class PlaybackGate:
    def __init__(self, world):
        self.world = world
        self._refresh_contact = False

    def wait(self):
        """Wait for Play; after a pause refresh physics before reading contacts.

        The refresh holds the last applied controls and does not advance skills
        or write a dataset frame. Stop or application shutdown cancels the run.
        """
        while self.world.app.is_running() and not self.world.is_stopped():
            if not self.world.is_playing():
                if not self._refresh_contact:
                    print("Simulation paused: task execution and recording suspended", flush=True)
                self._refresh_contact = True
                self.world.render()
                continue
            if not self._refresh_contact:
                return True
            before = self.world.current_time_step_index
            self.world.step(render=True)
            if self.world.is_playing() and self.world.current_time_step_index > before:
                self._refresh_contact = False
                print("Simulation resumed: physics contacts refreshed", flush=True)
                return True
        return False

    def step(self):
        """Return only after a playing physics step, including mid-render pauses."""
        before = self.world.current_time_step_index
        while self.wait():
            # wait() may already have advanced physics to refresh on resume.
            if self.world.current_time_step_index > before:
                return True
            self.world.step(render=True)
            if not self.world.is_playing():
                self._refresh_contact = True
            elif self.world.current_time_step_index > before:
                return True
        return False
