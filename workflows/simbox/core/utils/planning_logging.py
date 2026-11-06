"""Keep expected cuRobo mesh-cache hits from obscuring planning diagnostics."""
import logging


class MeshCacheHitFilter(logging.Filter):
    def filter(self, record):
        # This exact upstream message means reuse, not failure. Keep all other
        # collision, cache capacity, geometry and convergence warnings visible.
        return not record.getMessage().startswith(
            'Object already in warp cache, using existing instance for: ')


def configure_planning_logging():
    logger = logging.getLogger('curobo')
    if not any(isinstance(item, MeshCacheHitFilter) for item in logger.filters):
        logger.addFilter(MeshCacheHitFilter())
