import sys
import time
from abc import abstractmethod
from typing import Optional

from nimbus.components.data.iterator import Iterator
from nimbus.components.data.scene import Scene
from nimbus.daemon.decorators import status_monitor


class StrictRetryLimitError(RuntimeError):
    """A strict-mode target exhausted its configured retry budget."""


class LayoutRandomizer(Iterator):
    """
    Base class for layout randomization in a scene. This class defines the structure for randomizing scenes and
    tracking the randomization process. It manages the current scene, randomization count, and provides hooks for
    subclasses to implement specific randomization logic.

    Args:
        scene_iter (Iterator): An iterator that provides scenes to be randomized.
        random_num (int): The number of randomizations to perform for each scene before moving to the next one.
        strict_mode (bool): If True, the randomizer will check the generation status of the current scene and retry
                            randomization if it was not successful. This ensures that only successfully generated
                            scenes are counted towards the randomization limit.
        max_retries (int): Extra retries for one target after its initial attempt,
                           reset after success. -1 means unlimited. Exhaustion
                           raises StrictRetryLimitError instead of skipping a target.
    """

    def __init__(self, scene_iter: Iterator, random_num: int, strict_mode: bool = False, max_retries: int = -1):
        super().__init__()
        if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < -1:
            raise ValueError("max_retries must be an integer >= -1 (-1 means unlimited)")
        self.scene_iter = scene_iter
        self.random_num = random_num
        self.strict_mode = strict_mode
        self.max_retries = max_retries
        self.strict_retry_count = 0
        self.cur_index = sys.maxsize
        self.scene: Optional[Scene] = None

    def reset(self, scene):
        self.cur_index = 0
        self.strict_retry_count = 0
        self.scene = scene

    def _fetch_next_scene(self):
        scene = next(self.scene_iter)
        self.reset(scene)

    @status_monitor()
    def _randomize_with_status(self, scene) -> Scene:
        scene = self.randomize_scene(self.scene)
        return scene

    def _next(self) -> Scene:
        try:
            if self.strict_mode and self.scene is not None:
                if not self.scene.get_generate_status():
                    if self.max_retries >= 0 and self.strict_retry_count >= self.max_retries:
                        raise StrictRetryLimitError(
                            f"Strict retry limit reached: scene={self.scene.name}, "
                            f"target={self.cur_index}/{self.random_num}, "
                            f"max_retries={self.max_retries}, "
                            f"attempts_for_target={self.strict_retry_count + 1}. "
                            "Stop this worker; completed episodes remain saved."
                        )
                    self.strict_retry_count += 1
                    limit = str(self.max_retries) if self.max_retries >= 0 else "unlimited"
                    self.logger.info(
                        f"strict_mode retry {self.strict_retry_count}/{limit}, "
                        f"target={self.cur_index}/{self.random_num}"
                    )
                    st = time.time()
                    scene = self._randomize_with_status(self.scene)
                    self.collect_seq_info(1, time.time() - st)
                    return scene
            self.strict_retry_count = 0
            if self.cur_index >= self.random_num:
                self._fetch_next_scene()
            if self.cur_index < self.random_num:
                st = time.time()
                scene = self._randomize_with_status(self.scene)
                self.collect_seq_info(1, time.time() - st)
                self.cur_index += 1
            return scene
        except StopIteration:
            raise StopIteration("No more scenes to randomize.")
        except StrictRetryLimitError:
            self.logger.error("Strict retry budget exhausted; stopping generation.")
            raise
        except Exception as e:
            self.logger.exception(f"Error during scene idx {self.cur_index} randomization: {e}")
            self.cur_index += 1
            raise e

    @abstractmethod
    def randomize_scene(self, scene) -> Scene:
        raise NotImplementedError("This method should be overridden by subclasses")
