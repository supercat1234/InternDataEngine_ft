#!/usr/bin/env python3
"""Build a desktop-fixed dual-PiPER robot asset from the existing split_aloha USD.

Why an override layer instead of rebuilding the robot
----------------------------------------------------
The arm joint chain cannot be edited in place: PhysX rebuilds joint frames from
the prim transforms, so moving/rewriting arm links makes the articulation snap
back to the chassis at the first physics step. The safe move is to *reference*
the original asset and only downgrade the chassis:

  * chassis visual meshes  -> purpose=guide   (invisible, and excluded from the
                                               'default' purpose bbox query used by
                                               region_sampler.compute_bbox)
  * chassis colliders      -> collisionEnabled=false (no physics contact)
  * all links and joints   -> untouched, so DOF order / indices stay identical

Because the chassis drops out of the bbox, the A_on_B_region_sampler aligns the
robot's lowest *visible* geometry (the arm_base mounting plate) onto the table
top automatically -- i.e. "arm base flush with the desktop" without hand-tuned
z offsets.
"""
import os
import sys

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

REPO = "/home/agilex/qhx/InternDataEngine"
SRC = os.path.join(REPO, "workflows/simbox/example_assets/split_aloha_mid_360/robot.usd")
DST = os.path.join(
    REPO, "workflows/simbox/assets/custom/robot/piper_desktop_dual/piper_desktop_dual.usd"
)
ROBOT_PRIM = "/Root/split_aloha_mid_360_with_piper/split_aloha_mid_360_with_piper"

# Everything below belongs to the mobile chassis and must disappear on a desk.
CHASSIS_LINKS = [
    "box_link",
    "base_link",
    "lifting_link",
    "front_camera_link",
    "lidar_link",
    "top_camera_link",
    "fl_steering_wheel_link",
    "fl_wheel_link",
    "fr_steering_wheel_link",
    "fr_wheel_link",
    "rl_steering_wheel_link",
    "rl_wheel_link",
    "rr_steering_wheel_link",
    "rr_wheel_link",
]


def main():
    os.makedirs(os.path.dirname(DST), exist_ok=True)
    if os.path.exists(DST):
        os.remove(DST)

    src_stage = Usd.Stage.Open(SRC)
    stage = Usd.Stage.CreateNew(DST)
    UsdGeom.SetStageMetersPerUnit(stage, UsdGeom.GetStageMetersPerUnit(src_stage))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.GetStageUpAxis(src_stage))

    root = UsdGeom.Xform.Define(stage, "/Root").GetPrim()
    stage.SetDefaultPrim(root)
    root.GetReferences().AddReference(os.path.relpath(SRC, os.path.dirname(DST)))

    def touchable(path):
        """Author on the composed prim so the override lands on the right layer."""
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            return None
        return prim

    stats = {"visual": 0, "collider": 0, "missing": []}
    for link in CHASSIS_LINKS:
        for kind in ("visuals", "collisions"):
            path = f"{ROBOT_PRIM}/{link}/{kind}"
            prim = touchable(path)
            if prim is None:
                stats["missing"].append(path)
                continue
            # guide purpose => not rendered, not in the 'default' purpose bbox
            UsdGeom.Imageable(prim).CreatePurposeAttr().Set(UsdGeom.Tokens.guide)
            UsdGeom.Imageable(prim).CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
            if kind == "visuals":
                stats["visual"] += 1
            else:
                stats["collider"] += 1
            # Kill physics contact on every descendant collider
            for desc in Usd.PrimRange(prim):
                if desc.HasAPI(UsdPhysics.CollisionAPI) or desc.IsA(UsdGeom.Gprim):
                    UsdPhysics.CollisionAPI.Apply(desc).CreateCollisionEnabledAttr().Set(False)
                if desc.IsA(UsdGeom.Gprim):
                    UsdGeom.Imageable(desc).CreatePurposeAttr().Set(UsdGeom.Tokens.guide)
                    UsdGeom.Imageable(desc).CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)

    stage.GetRootLayer().Save()

    # ---- report ----
    chk = Usd.Stage.Open(DST)
    cache = UsdGeom.XformCache()
    for side in ("fl", "fr"):
        p = chk.GetPrimAtPath(f"{ROBOT_PRIM}/{side}/arm_base")
        t = cache.GetLocalToWorldTransform(p).ExtractTranslation()
        print(f"{side}/arm_base world = {[round(v, 4) for v in t]}")
    img = UsdGeom.Imageable(chk.GetPrimAtPath(ROBOT_PRIM))
    rng = img.ComputeWorldBound(Usd.TimeCode.Default(), UsdGeom.Tokens.default_).ComputeAlignedBox()
    print(f"robot default-purpose bbox min = {[round(v, 4) for v in rng.GetMin()]}")
    print(f"robot default-purpose bbox max = {[round(v, 4) for v in rng.GetMax()]}")
    print(f"hidden visuals={stats['visual']} colliders={stats['collider']} missing={len(stats['missing'])}")
    for m in stats["missing"]:
        print("  MISSING", m)
    print("saved:", DST)


if __name__ == "__main__":
    sys.exit(main())
