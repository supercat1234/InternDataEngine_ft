"""Optional rigid-grasp approximation after fingers physically close on food."""
import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
from scipy.spatial.transform import Rotation


def add_food_grasp(controller, mesh_path):
    stage = controller.task.stage
    food = stage.GetPrimAtPath(mesh_path)
    while food and not food.HasAPI(UsdPhysics.RigidBodyAPI):
        food = food.GetParent()
    if not food or not food.HasAPI(UsdPhysics.RigidBodyAPI):
        raise RuntimeError(f"No rigid body for grasp: {mesh_path}")
    name = next(name for name, obj in controller.task.objects.items() if obj.prim_path == str(food.GetPath()))
    contact = controller.task.pickcontact_views[controller.robot.name][controller.lr_name][name]
    force = float(np.abs(contact.get_contact_force_matrix()).sum())
    if force <= 0.001:
        raise RuntimeError("Rigid food grasp requires measured fingertip contact")
    hand = stage.GetPrimAtPath(controller.robot_ee_path)
    food_world = UsdGeom.Xformable(food).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    hand_world = UsdGeom.Xformable(hand).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    point = food_world.ExtractTranslation()
    hand_rotation = np.asarray(hand_world.RemoveScaleShear())[:3, :3].T
    food_rotation = np.asarray(food_world.RemoveScaleShear())[:3, :3].T
    q = Rotation.from_matrix(hand_rotation.T @ food_rotation).as_quat(scalar_first=True)
    path = f"{controller.task.root_prim_path}/FoodGrasp_{controller.lr_name}"
    joint = UsdPhysics.FixedJoint.Define(stage, path)
    joint.CreateBody0Rel().SetTargets([hand.GetPath()])
    joint.CreateBody1Rel().SetTargets([food.GetPath()])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*hand_world.GetInverse().Transform(point)))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(float(q[0]), Gf.Vec3f(*q[1:])))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1))
    joint.CreateExcludeFromArticulationAttr(True)
    joint.CreateCollisionEnabledAttr(False)
    joint.CreateBreakForceAttr(50.)
    joint.CreateBreakTorqueAttr(5.)
    controller._food_grasp_path = path
    print("Food rigid-grasp constraint enabled:", path, "measured contact force sum=", force,
          "without moving the object", flush=True)


def release_food_grasp(controller):
    path = getattr(controller, "_food_grasp_path", None)
    if path:
        controller.task.stage.RemovePrim(path)
        controller._food_grasp_path = None
        print("Food rigid-grasp constraint released", flush=True)
