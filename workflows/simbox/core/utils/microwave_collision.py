"""Opt-in hollow collision approximation for the audited microwave_0013 asset."""
import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics


RAW_LOWER = np.array([-27.22, -17.215, -12.113])
RAW_UPPER = np.array([14.027, 18.467, 13.126])


def measured_cavity(stage, oven):
    """Measure the authored inner surfaces using the live base transform."""
    from omni.isaac.core.utils.transformations import tf_matrix_from_pose

    transform = UsdGeom.Xformable(stage.GetPrimAtPath(oven.object_base_path)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    corners = np.array(np.meshgrid(*zip(RAW_LOWER, RAW_UPPER))).T.reshape(-1, 3)
    world = np.array([transform.Transform(Gf.Vec3d(*p)) for p in corners])
    frame = tf_matrix_from_pose(*oven.get_world_pose())
    local = (world - frame[:3, 3]) @ frame[:3, :3]
    return local.min(axis=0), local.max(axis=0)


def build_hollow_microwave(stage, oven):
    base = stage.GetPrimAtPath(oven.object_base_path)
    # These visual shells/rings contain an opening; convex decomposition can
    # seal that opening. Keep the panel, door and rotary knob colliders.
    for name in ["Component_5", "Component_6", "Component_9", "Component_10"]:
        prim = stage.GetPrimAtPath(f"{oven.object_base_path}/{name}")
        UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(False)
    transform = UsdGeom.Xformable(base).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    # Author in the same raw base coordinates as the visual shell, so that
    # reset/physics initialization cannot change relative collision sizes.
    lower, upper = RAW_LOWER, RAW_UPPER
    center, size = (lower + upper) / 2, upper - lower
    thickness = 1.0
    walls = []
    for axis, side, name in [(0, -1, "left"), (0, 1, "right"),
                             (2, -1, "floor"), (2, 1, "ceiling"), (1, 1, "back")]:
        position, dimensions = center.copy(), size.copy()
        position[axis] = (lower[axis] - thickness / 2) if side < 0 else (upper[axis] + thickness / 2)
        dimensions[axis] = thickness
        cube = UsdGeom.Cube.Define(stage, f"{oven.object_base_path}/CavityColliders/{name}")
        cube.CreateSizeAttr(1.)
        xform = UsdGeom.Xformable(cube)
        xform.AddTranslateOp().Set(Gf.Vec3d(*position))
        xform.AddScaleOp().Set(Gf.Vec3f(*dimensions))
        cube.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim()).CreateCollisionEnabledAttr(True)
        walls.append(name)
    print("Hollow microwave collision walls:", walls, "inner bounds in asset coordinates:", lower, upper, flush=True)
