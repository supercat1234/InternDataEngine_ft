#!/usr/bin/env python3
"""Inspect and test the elastic cup in a standalone scene (no dataset production)."""
import argparse
import json
from pathlib import Path
import sys
from datetime import datetime

REPO = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--asset", type=Path, default=REPO/"workflows/simbox/assets/custom/objects/paper_cup_elastic_001/Aligned_obj.usd")
    ap.add_argument("--output-dir", type=Path)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--validate", action="store_true", help="Run one squeeze/release and exit with a checked result")
    ap.add_argument("--youngs-modulus", type=float, help="Override stiffness for an experiment; does not edit the asset")
    ap.add_argument("--lift-test", action="store_true", help="Also lift 30 mm using frictional finger contact, then lower")
    ap.add_argument("--grip-gap", type=float, default=0.0625, help="Closed inner finger gap in meters")
    ap.add_argument("--cup-friction", type=float, help="Override effective paper/pad friction for calibration")
    ap.add_argument("--physics-hz", type=int, default=480)
    args = ap.parse_args()
    if args.output_dir is None:
        args.output_dir = REPO/"output/cup_plate_diagnostics/elastic_cup"/datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if not args.asset.is_file():
        ap.error(f"Build the asset first: {args.asset}")
    if args.headless and not args.validate:
        ap.error("Use --validate with --headless")
    sys.path[:0] = [str(REPO), str(REPO/"workflows/simbox"), str(REPO/"workflows/simbox/core/utils")]
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": args.headless, "width": 960, "height": 720,
                         "anti_aliasing": 0, "renderer": "RayTracedLighting", "fast_shutdown": False,
                         "extra_args": ["--/persistent/physics/enableDeformableBeta=true"]})
    try:
        import carb.settings
        import imageio.v2 as imageio
        import numpy as np
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleRigidPrim
        import omni.physics.tensors
        from isaacsim.core.utils.viewports import set_camera_view
        from isaacsim.sensors.camera import Camera
        from omni.physx.bindings import _physx as bindings
        from pxr import Gf, PhysxSchema, UsdGeom, UsdLux, UsdPhysics, UsdShade
        from elastic_cup import cup_deformation
        from core.objects.deformable_object import DeformableObject

        settings = carb.settings.get_settings()
        settings.set_bool(bindings.SETTING_ENABLE_DEFORMABLE_BETA, True)
        if args.physics_hz < 480:
            raise ValueError("Use at least 480 Hz for this thin-wall FEM asset")
        world = World(physics_dt=1/args.physics_hz, rendering_dt=1/args.physics_hz, stage_units_in_meters=1,
                      backend="torch", device="cuda:0")
        ctx = world.get_physics_context()
        ctx.enable_gpu_dynamics(True)
        ctx.set_broadphase_type("GPU")
        ctx.set_solver_type("TGS")
        stage = world.stage
        world_prim = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(world_prim.GetPrim())
        light = UsdLux.DomeLight.Define(stage, "/World/Light")
        light.CreateIntensityAttr(1200)
        pad_material = UsdShade.Material.Define(stage, "/World/PadMaterial")
        pad_physics = UsdPhysics.MaterialAPI.Apply(pad_material.GetPrim())
        pad_physics.CreateStaticFrictionAttr(1.2)
        pad_physics.CreateDynamicFrictionAttr(1.0)
        pad_physics.CreateRestitutionAttr(0)

        finger_drives = []

        def box(path, size, position, color, actuated=False):
            cube = UsdGeom.Cube.Define(stage, path)
            cube.CreateSizeAttr(1.0)
            xf = UsdGeom.Xformable(cube)
            xf.AddTranslateOp().Set(Gf.Vec3d(*position))
            xf.AddScaleOp().Set(Gf.Vec3f(*size))
            cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
            col = PhysxSchema.PhysxCollisionAPI.Apply(cube.GetPrim())
            col.CreateContactOffsetAttr(0.0004)
            col.CreateRestOffsetAttr(0.0)
            if actuated:
                UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(pad_material, materialPurpose="physics")
                UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
                UsdPhysics.MassAPI.Apply(cube.GetPrim()).CreateMassAttr(0.05)
                PhysxSchema.PhysxRigidBodyAPI.Apply(cube.GetPrim()).CreateDisableGravityAttr(True)
                # Three driven translation axes and locked rotation, like a
                # small Cartesian fixture. Dynamic contact conveys friction.
                joint = UsdPhysics.Joint.Define(stage, path+"_drive")
                joint.CreateBody1Rel().SetTargets([path])
                joint.CreateLocalPos0Attr(Gf.Vec3f(*position))
                joint.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
                for axis in ("rotX", "rotY", "rotZ"):
                    limit = UsdPhysics.LimitAPI.Apply(joint.GetPrim(), axis)
                    limit.CreateLowAttr(0)
                    limit.CreateHighAttr(0)
                drives = []
                for axis in ("transX", "transY", "transZ"):
                    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), axis)
                    drive.CreateTypeAttr("force")
                    drive.CreateStiffnessAttr(10000)
                    drive.CreateDampingAttr(50)
                    drive.CreateMaxForceAttr(20)
                    drive.CreateTargetPositionAttr(0)
                    drives.append(drive)
                finger_drives.append((position[0], drives))
                obj = SingleRigidPrim(path, name=path.rsplit("/", 1)[-1])
                world.scene.add(obj)
                return obj

        box("/World/Table", [0.5, 0.5, 0.03], [0, 0, -0.015], [0.28, 0.26, 0.24])
        cup = DeformableObject(str(args.asset.resolve().parent), "/World", {
            "name": "Cup", "path": args.asset.name, "prim_path_child": "Aligned", "scale": [1, 1, 1]})
        mesh_path = "/World/Cup/Aligned/ElasticMesh"
        mesh = stage.GetPrimAtPath(mesh_path)
        assert stage.GetPrimAtPath("/World/Cup/Aligned").HasAPI("OmniPhysicsDeformableBodyAPI"), "The USD must carry a real FEM deformable body"
        assert not mesh.HasAPI(UsdPhysics.RigidBodyAPI)
        if args.youngs_modulus is not None:
            if not np.isfinite(args.youngs_modulus) or args.youngs_modulus <= 0:
                raise ValueError("Invalid stiffness")
            stage.GetPrimAtPath("/World/Cup/Looks/ElasticPaper").GetAttribute(
                "omniphysics:youngsModulus").Set(args.youngs_modulus)
        if args.cup_friction is not None:
            if not np.isfinite(args.cup_friction) or args.cup_friction < 0:
                raise ValueError("Invalid friction")
            for key in ("dynamicFriction", "staticFriction"):
                stage.GetPrimAtPath("/World/Cup/Looks/ElasticPaper").GetAttribute("omniphysics:"+key).Set(args.cup_friction)
        finger_size = [0.006, 0.014, 0.020]
        initial_gap, final_gap = 0.080, args.grip_gap
        if not 0.058 <= final_gap < initial_gap:
            raise ValueError("grip-gap must be between 0.058 and 0.080 meters")
        fingers = [box(f"/World/Finger_{i}", finger_size,
                       [sign*(initial_gap+finger_size[0])/2, 0, 0.05], [0.2, 0.35, 0.65], True)
                   for i, sign in enumerate((-1, 1))]
        set_camera_view(eye=[0.16, -0.22, 0.15], target=[0, 0, 0.04], camera_prim_path="/OmniverseKit_Persp")
        from core.utils.camera_utils import look_at_usd_orientation
        camera = Camera("/World/Camera", resolution=(960, 720), frequency=30)
        world.reset()
        cup.initialize()
        physics_view = omni.physics.tensors.create_simulation_view("torch")
        physics_view.set_subspace_roots("/")
        camera.initialize()
        camera.set_clipping_range(0.005, 10.0)
        camera.set_world_pose(position=np.array([0.19, -0.27, 0.17]),
                              orientation=look_at_usd_orientation([0.19, -0.27, 0.17], [0, 0, 0.04]),
                              camera_axes="usd")
        rest = cup.rest_nodes
        print("ELASTIC_CUP mesh:", len(rest), "nodes", flush=True)
        mat_view = physics_view.create_deformable_material_view("/World/Cup/Looks/ElasticPaper")
        print("ELASTIC_CUP material:", mat_view.count, mat_view.get_youngs_modulus(), flush=True)
        print("ELASTIC_CUP fingers:", [f.get_world_pose()[0] for f in fingers], flush=True)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        # Save a small portable scene referencing the asset, BEFORE runtime deformation.
        stage.GetRootLayer().Export(str(args.output_dir/"scene.usda"))

        def positions():
            return cup.get_deformable_state()["nodal_positions"]

        def snapshot(name):
            for _ in range(4):
                world.render()
            rgba = camera.get_rgba()
            if rgba.size:
                imageio.imwrite(args.output_dir/f"{name}.png", rgba[:, :, :3].astype(np.uint8))
            np.savez_compressed(args.output_dir/f"{name}.npz", positions=positions())
            metrics = cup_deformation(rest, positions())
            print(f"ELASTIC_CUP {name}: {json.dumps(metrics)}", flush=True)
            return metrics

        def step(gap, count, lift=0):
            for _ in range(count):
                for x_initial, drives in finger_drives:
                    sign = np.sign(x_initial)
                    drives[0].GetTargetPositionAttr().Set(float(sign*(gap+finger_size[0])/2-x_initial))
                    drives[2].GetTargetPositionAttr().Set(float(lift))
                world.step(render=(world.current_time_step_index % 8 == 0))
                if not np.isfinite(positions()).all():
                    raise RuntimeError("Non-finite FEM nodes")

        step(initial_gap, 240)
        baseline = snapshot("01_rest")
        for t in np.linspace(0, 1, 240):
            step(initial_gap+(final_gap-initial_gap)*(3*t*t-2*t*t*t), 1)
        step(final_gap, 120)
        squeezed = snapshot("02_squeezed")
        squeezed_state = cup.get_deformable_state()
        lifted_height = None
        if args.lift_test:
            start_height = cup.get_world_pose()[0][2]
            for t in np.linspace(0, 1, 240):
                step(final_gap, 1, lift=0.03*(3*t*t-2*t*t*t))
            step(final_gap, 120, lift=0.03)
            snapshot("02b_lifted")
            lifted_height = float(cup.get_world_pose()[0][2]-start_height)
            for t in np.linspace(0, 1, 240):
                step(final_gap, 1, lift=0.03*(1-3*t*t+2*t*t*t))
            step(final_gap, 120)
        for t in np.linspace(0, 1, 240):
            step(final_gap+(initial_gap-final_gap)*(3*t*t-2*t*t*t), 1)
        step(initial_gap, 480)
        recovered = snapshot("03_recovered")
        # Verify the adapter on real GPU state: exact restore, teleport with
        # deformation preserved, rest-shape reset at a rotated/transferred pose.
        from scipy.spatial.transform import Rotation
        cup.set_deformable_state(squeezed_state)
        restore_error = float(np.max(np.abs(positions()-squeezed_state["nodal_positions"])))
        before = cup.get_deformation_metrics()["rms_displacement_m"]
        cup.set_local_pose([0.06, 0.02, 0.01], Rotation.from_euler("z", 37, degrees=True).as_quat(scalar_first=True))
        teleport_error = abs(cup.get_deformation_metrics()["rms_displacement_m"]-before)
        target = np.array([0.04, 0.01, 0.02])
        quat = Rotation.from_euler("xyz", [5, -7, 41], degrees=True).as_quat(scalar_first=True)
        cup.reset_deformation(target, quat)
        reset_error = float(np.max(np.abs(positions()-(Rotation.from_quat(quat, scalar_first=True).apply(rest)+target))))
        velocity_error = float(np.max(np.abs(cup.get_deformable_state()["nodal_velocities"])))
        try:
            cup.set_deformable_state({**squeezed_state, "topology_id": "wrong-mesh"})
        except ValueError:
            wrong_mesh_rejected = True
        else:
            wrong_mesh_rejected = False
        cup.reset_deformation([0, 0, 0], [1, 0, 0, 0])
        step(initial_gap, 120)
        compression = baseline["grasp_band_width_x_m"]-squeezed["grasp_band_width_x_m"]
        residual = abs(baseline["grasp_band_width_x_m"]-recovered["grasp_band_width_x_m"])
        result = {"baseline": baseline, "squeezed": squeezed, "recovered": recovered,
                  "compression_m": compression, "residual_width_error_m": residual,
                  "lifted_height_m": lifted_height,
                  "adapter": dict(restore_error_m=restore_error, teleport_deformation_error_m=teleport_error,
                                  reset_error_m=reset_error, reset_velocity_error_mps=velocity_error,
                                  wrong_mesh_rejected=wrong_mesh_rejected),
                  "passed": bool(0.001 <= compression <= 0.006 and residual <= 0.0005
                                 and baseline["rms_displacement_m"] <= 0.0005
                                 and squeezed["max_displacement_m"] <= 0.004
                                 and recovered["rms_displacement_m"] <= 0.0005
                                 and restore_error < 1e-6 and teleport_error < 1e-6
                                 and reset_error < 1e-6 and velocity_error == 0 and wrong_mesh_rejected
                                 and (lifted_height is None or lifted_height >= 0.025))}
        (args.output_dir/"validation.json").write_text(json.dumps(result, indent=2)+"\n")
        print("ELASTIC_CUP_RESULT "+json.dumps(result), flush=True)
        if args.validate:
            if not result["passed"]:
                raise RuntimeError("Elastic squeeze/recovery validation failed; inspect validation.json")
        else:
            world.pause()
            print("ELASTIC_CUP_PREVIEW: squeeze/release completed; scene paused; no training data generated", flush=True)
            while app.is_running():
                world.render()
    except Exception:
        # Kit replaces Python's excepthook during shutdown; print while it is alive.
        import traceback
        traceback.print_exc()
        # Explicitly request a failing process status before Kit's shutdown.
        import carb
        carb.settings.get_settings().set_int("/app/returnCode", 1)
        raise
    finally:
        app.close()


if __name__ == "__main__":
    main()
