"""Launch InternDataEngine with the bundled assets' explicit unit scales on Isaac 5.1."""
from pathlib import Path
import os
import runpy
import sys

repo = Path(__file__).resolve().parent.parent
os.chdir(repo)
sys.path[:0] = [str(repo), str(repo / "workflows/simbox")]
import workflows

original_import = workflows.import_extensions

def import_with_asset_units(workflow_type):
    # The task YAML already scales its assets. Isaac 5.1's automatic unit
    # correction would scale centimeter metadata again and corrupt placement.
    import carb.settings
    import omni.metrics.assembler.core.bindings._metricsAssembler as metrics
    carb.settings.get_settings().set_int(metrics.SETTINGS_METRICS_ASSEMBLER_OPERATION_MODE, 0)
    print("InternDataEngine: preserving asset scales from the task configuration", flush=True)
    original_import(workflow_type)
    from core.utils.planning_logging import configure_planning_logging
    configure_planning_logging()

workflows.import_extensions = import_with_asset_units
runpy.run_path(str(repo / "launcher.py"), run_name="__main__")
