"""Find mesh geometry without relying on USD child order or material scopes."""

from pxr import Usd, UsdGeom


def collect_mesh_paths(prim):
    if not prim.IsValid():
        raise ValueError("Cannot collect meshes from an invalid prim")
    paths = [str(p.GetPath()) for p in Usd.PrimRange(prim, Usd.TraverseInstanceProxies())
             if p.IsA(UsdGeom.Mesh)]
    if not paths:
        raise ValueError(f"No mesh geometry below {prim.GetPath()}")
    return paths
