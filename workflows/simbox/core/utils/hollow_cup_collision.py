"""Session-local, volumetric convex pieces for the original rigid paper cup.

Geometry uses the original asset's millimeter coordinates. Task scale remains
0.001. No rigid body, mass, render mesh, fixed joint or source file is changed.
"""
import numpy as np
from scipy.spatial import ConvexHull
from pxr import Sdf, UsdGeom, UsdPhysics, Vt


def cup_convex_pieces(sectors=48):
    if sectors < 24:
        raise ValueError("Use at least 24 wall segments")
    pieces = []
    angles = np.linspace(0, 2*np.pi, sectors+1)
    for i, (a, b) in enumerate(zip(angles[:-1], angles[1:])):
        # The real inner and outer visual surfaces have different base z.
        points = [[r*np.cos(t), r*np.sin(t), z]
                  for z, inner, outer in [(1.2, 25.3, 26.5+9.7*1.2/78.75), (78.75, 35.0, 36.2)]
                  for r in (inner, outer) for t in (a, b)]
        pieces.append((f"Wall_{i:02d}", np.asarray(points)))
        points = [[(36.25+1.25*np.cos(phi))*np.cos(t),
                   (36.25+1.25*np.cos(phi))*np.sin(t), 78.75+1.25*np.sin(phi)]
                  for t in (a, b) for phi in np.linspace(0, 2*np.pi, 8, endpoint=False)]
        pieces.append((f"Rim_{i:02d}", np.asarray(points)))
    points = [[26.5*np.cos(t), 26.5*np.sin(t), z]
              for z in (0., 1.2) for t in angles[:-1]]
    pieces.append(("Bottom", np.asarray(points)))
    return pieces


def build_hollow_cup(stage, rigid_prim, visual_paths):
    root = str(rigid_prim.GetPath())+"/CupCollisionProxy"
    if stage.GetPrimAtPath(root):
        return
    for path in visual_paths:
        UsdPhysics.CollisionAPI(stage.GetPrimAtPath(path)).CreateCollisionEnabledAttr(False)
    pieces = cup_convex_pieces()
    for name, points in pieces:
        mesh = UsdGeom.Mesh.Define(stage, root+"/"+name)
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
        hull = ConvexHull(points)
        mesh.CreateFaceVertexCountsAttr([3]*len(hull.simplices))
        mesh.CreateFaceVertexIndicesAttr(hull.simplices.ravel().tolist())
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateVisibilityAttr("invisible")
        mesh.CreatePurposeAttr("guide")
        prim = mesh.GetPrim()
        UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("convexHull")
        schemas = prim.GetMetadata("apiSchemas").GetAppliedItems()
        prim.SetMetadata("apiSchemas", Sdf.TokenListOp.CreateExplicit(schemas+["PhysxCollisionAPI"]))
        prim.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float, custom=False).Set(.0005)
        prim.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float, custom=False).Set(0.)
    print(f"Rigid cup hollow collision: {len(pieces)} convex pieces; cavity preserved; mass unchanged", flush=True)
