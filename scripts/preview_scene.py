"""Inspect a SimBox scene without creating controllers, skills or a data writer."""
import argparse
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="local_configs/cup_plate_desk/task.yaml")
    parser.add_argument("--camera", choices=["external", "hand_left", "hand_right", "perspective"], default="external")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--frames", type=int, default=0, help="Exit after N preview frames; 0 keeps the window open")
    args = parser.parse_args()
    if args.frames < 0 or (args.headless and args.frames == 0):
        parser.error("headless preview requires --frames N > 0; frames cannot be negative")

    repo = Path(__file__).resolve().parent.parent
    os.chdir(repo)
    sys.path[:0] = [str(repo), str(repo / "workflows/simbox")]

    from isaacsim import SimulationApp
    app = SimulationApp({"headless": args.headless, "anti_aliasing": 0, "renderer": "RayTracedLighting"})
    try:
        import carb.settings
        import omni.metrics.assembler.core.bindings._metricsAssembler as metrics
        from omni.isaac.core import World
        from omni.isaac.core.utils.viewports import set_camera_view
        from omni.kit.viewport.utility import get_active_viewport
        from workflows import import_extensions
        from workflows.base import NimbusWorkFlow

        # Match production: the task YAML explicitly supplies asset unit scales.
        carb.settings.get_settings().set_int(metrics.SETTINGS_METRICS_ASSEMBLER_OPERATION_MODE, 0)
        import_extensions("SimBoxDualWorkFlow")
        from core.utils.utils import set_random_seed
        set_random_seed(args.seed)
        world = World(physics_dt=1 / 30, rendering_dt=1 / 30, stage_units_in_meters=1.0)
        workflow = NimbusWorkFlow.workflows["SimBoxDualWorkFlow"](
            world, args.task, random_seed=args.seed, preview_only=True
        )
        workflow.init_task(0)
        # Refresh mounted cameras once after the initial physics settling, then
        # freeze the timeline. No observation is logged or written to disk.
        world.get_observations()
        world.pause()
        assert not any(hasattr(workflow, name) for name in ("controllers", "skills", "logger", "world_recorder"))

        viewport = get_active_viewport()
        camera_paths = {name.removeprefix("split_aloha_"): camera.prim_path
                        for name, camera in workflow.task.cameras.items()}
        if args.camera == "perspective":
            set_camera_view(eye=[1.3, -1.8, 1.8], target=[0, 0, 0.8],
                            camera_prim_path="/OmniverseKit_Persp")
            if viewport:
                viewport.set_active_camera("/OmniverseKit_Persp")
        elif viewport:
            viewport.set_active_camera(camera_paths[args.camera])

        print("PREVIEW_READY: timeline paused; no planning, task execution or dataset recording", flush=True)
        print("Viewport camera menu: Perspective / split_aloha_external / split_aloha_hand_left / split_aloha_hand_right", flush=True)
        for name, path in camera_paths.items():
            print(f"  {name}: {path}", flush=True)
        frames = 0
        while app.is_running() and (not args.frames or frames < args.frames):
            # Keep the UI and renderer alive without advancing task physics,
            # even if Play is clicked. Free-camera navigation still works.
            if world.is_playing():
                world.pause()
            world.render()
            frames += 1
        print(f"PREVIEW_CLOSED: {frames} render frames; no dataset written", flush=True)
    finally:
        app.close()


if __name__ == "__main__":
    main()
