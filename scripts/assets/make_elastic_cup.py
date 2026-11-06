#!/usr/bin/env python3
"""Build the standalone elastic cup asset, using explicit hollow FEM tetrahedra.

Run with scripts/isaac_python.sh; this builder needs USD but does not start Isaac.
The USD authors the volume FEM schema (Deformable beta) in Isaac Sim 5.1.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "workflows/simbox/core/utils"))
from elastic_cup import make_elastic_cup


def attribute(prim, name, kind, value):
    prim.CreateAttribute(name, kind, custom=False).Set(value)


def visual_material(stage, path, texture=None, color=(0.96, 0.92, 0.81)):
    mat = UsdShade.Material.Define(stage, path)
    surface = UsdShade.Shader.Define(stage, path+"/Surface")
    surface.CreateIdAttr("UsdPreviewSurface")
    surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.84)
    surface.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    diffuse = surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
    if texture:
        reader = UsdShade.Shader.Define(stage, path+"/UVReader")
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
        tex = UsdShade.Shader.Define(stage, path+"/Texture")
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(texture))
        tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
        tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        diffuse.ConnectToSource(tex.ConnectableAPI(), "rgb")
    else:
        diffuse.Set(Gf.Vec3f(*color))
    mat.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")
    return mat


def build(output, youngs_modulus=1e8, mass=0.008, force=False):
    output = Path(output).resolve()
    if output.exists() and not force:
        raise FileExistsError(f"Asset exists; pass --force to rebuild: {output}")
    if not np.isfinite(youngs_modulus) or youngs_modulus <= 0 or not np.isfinite(mass) or mass <= 0:
        raise ValueError("Mass and Young's modulus must be finite and positive")
    data = make_elastic_cup()
    output.parent.mkdir(parents=True, exist_ok=True)
    textures = output.parent / "textures"
    textures.mkdir(exist_ok=True)
    source = REPO / "workflows/simbox/assets/custom/objects/paper_cup/textures"
    for name in ("cup_paper.png", "cup_inside.png"):
        shutil.copy2(source / name, textures / name)
    stage = Usd.Stage.CreateNew(str(output)) if not output.exists() else Usd.Stage.Open(str(output))
    if stage.GetPrimAtPath("/World"):
        stage.RemovePrim("/World")
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, "Z")
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.Xform.Define(stage, "/World/Aligned")
    path = "/World/Aligned/ElasticMesh"
    mesh = UsdGeom.Mesh.Define(stage, path)
    points, faces = data["points"][data["skin_ids"]], data["skin_faces"]
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
    mesh.CreateFaceVertexCountsAttr([3]*len(faces))
    mesh.CreateFaceVertexIndicesAttr(faces.ravel().tolist())
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateExtentAttr([Gf.Vec3f(*points.min(0)), Gf.Vec3f(*points.max(0))])
    # Texture each face without a UV seam crossing the cup print.
    triangles = points[faces]
    normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    centers = triangles.mean(1)
    radial_dot = np.einsum("ij,ij->i", centers[:, :2], normals[:, :2])
    material_ids = np.where(radial_dot > 0, 0, 1)
    material_ids[centers[:, 2] > 0.0774] = 2
    uv = np.stack((np.arctan2(triangles[:, :, 1], triangles[:, :, 0])/(2*np.pi)+0.5,
                   triangles[:, :, 2]/0.080), axis=-1)
    seam = np.ptp(uv[:, :, 0], axis=1) > 0.5
    uv[seam, :, 0] += (uv[seam, :, 0] < 0.5)
    bottom = np.abs(normals[:, 2]) > 0.9
    uv[bottom] = triangles[bottom, :, :2]/0.053+0.5
    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray,
                                           UsdGeom.Tokens.faceVarying).Set(Vt.Vec2fArray.FromNumpy(uv.reshape(-1, 2).astype(np.float32)))
    # Leave normals un-authored so shading can update along with the deformed skin.
    mats = [visual_material(stage, "/World/Looks/OuterPaper", "textures/cup_paper.png"),
            visual_material(stage, "/World/Looks/InnerPaper", "textures/cup_inside.png"),
            visual_material(stage, "/World/Looks/RimPaper")]
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mats[0])
    for i, mat in enumerate(mats):
        subset = UsdGeom.Subset.Define(stage, path+f"/Material_{i}")
        subset.CreateElementTypeAttr("face")
        subset.CreateFamilyNameAttr("materialBind")
        subset.CreateIndicesAttr(np.flatnonzero(material_ids == i).tolist())
        UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(mat)
    UsdGeom.Subset.SetFamilyType(mesh, "materialBind", "partition")
    prim = mesh.GetPrim()
    # Explicit native tetrahedra preserve the cavity. No voxel/remesh cooking.
    body = stage.GetPrimAtPath("/World/Aligned")
    body.SetMetadata("apiSchemas", Sdf.TokenListOp.CreateExplicit([
        "OmniPhysicsDeformableBodyAPI", "PhysxBaseDeformableBodyAPI", "MaterialBindingAPI"]))
    tet = UsdGeom.TetMesh.Define(stage, "/World/Aligned/SimulationMesh")
    tet.CreatePurposeAttr("guide")
    rest = Vt.Vec3fArray.FromNumpy(data["points"].astype(np.float32))
    physx_tets = data["tets"]
    tet.CreatePointsAttr(rest)
    tet.CreateTetVertexIndicesAttr(Vt.Vec4iArray.FromNumpy(physx_tets))
    tet.CreateSurfaceFaceVertexIndicesAttr(Vt.Vec3iArray.FromNumpy(data["skin_ids"][faces].astype(np.int32)))
    tet.GetPrim().SetMetadata("apiSchemas", Sdf.TokenListOp.CreateExplicit([
        "OmniPhysicsVolumeDeformableSimAPI", "PhysicsCollisionAPI", "PhysxCollisionAPI"]))
    attribute(tet.GetPrim(), "omniphysics:restShapePoints", Sdf.ValueTypeNames.Point3fArray, rest)
    attribute(tet.GetPrim(), "omniphysics:restTetVtxIndices", Sdf.ValueTypeNames.Int4Array,
              Vt.Vec4iArray.FromNumpy(physx_tets))
    attribute(tet.GetPrim(), "physxCollision:contactOffset", Sdf.ValueTypeNames.Float, 0.0004)
    attribute(tet.GetPrim(), "physxCollision:restOffset", Sdf.ValueTypeNames.Float, 0.0)
    # Bind-pose data lets the engine skin the textured visual mesh from FEM nodes.
    for p, pts in ((prim, points), (tet.GetPrim(), data["points"])):
        schemas = p.GetMetadata("apiSchemas").GetAppliedItems()
        p.SetMetadata("apiSchemas", Sdf.TokenListOp.CreateExplicit(schemas+["OmniPhysicsDeformablePoseAPI:default"]))
        attribute(p, "deformablePose:default:omniphysics:purposes", Sdf.ValueTypeNames.TokenArray, ["bindPose"])
        attribute(p, "deformablePose:default:omniphysics:points", Sdf.ValueTypeNames.Point3fArray,
                  Vt.Vec3fArray.FromNumpy(pts.astype(np.float32)))
    for name, kind, value in (
        ("omniphysics:deformableBodyEnabled", Sdf.ValueTypeNames.Bool, True),
        ("physxDeformableBody:selfCollision", Sdf.ValueTypeNames.Bool, False),
        ("physxDeformableBody:enableSpeculativeCCD", Sdf.ValueTypeNames.Bool, True),
        ("physxDeformableBody:solverPositionIterationCount", Sdf.ValueTypeNames.UInt, 255),
        ("physxDeformableBody:linearDamping", Sdf.ValueTypeNames.Float, 1.0),
        ("physxDeformableBody:sleepThreshold", Sdf.ValueTypeNames.Float, 0.0001),
        ("physxDeformableBody:settlingThreshold", Sdf.ValueTypeNames.Float, 0.0001),
        ("omniphysics:mass", Sdf.ValueTypeNames.Float, mass),
    ):
        attribute(body, name, kind, value)
    physics = UsdShade.Material.Define(stage, "/World/Looks/ElasticPaper")
    physics.GetPrim().SetMetadata("apiSchemas", Sdf.TokenListOp.CreateExplicit([
        "OmniPhysicsDeformableMaterialAPI", "PhysxDeformableMaterialAPI"]))
    for key, value in dict(youngsModulus=youngs_modulus, poissonsRatio=0.3,
                           density=mass/data["material_volume_m3"], dynamicFriction=0.7, staticFriction=0.8).items():
        attribute(physics.GetPrim(), "omniphysics:"+key, Sdf.ValueTypeNames.Float, value)
    attribute(physics.GetPrim(), "physxDeformableMaterial:elasticityDamping", Sdf.ValueTypeNames.Float, 0.005)
    UsdShade.MaterialBindingAPI(body).Bind(physics, bindingStrength="strongerThanDescendants", materialPurpose="physics")
    UsdShade.MaterialBindingAPI.Apply(tet.GetPrim()).Bind(physics, materialPurpose="physics")
    body.SetCustomDataByKey("elastic_cup", {"schema": "physx_beta_volume_fem", "units": "meters", "state": "elastic_only"})
    stage.GetRootLayer().Save()
    metadata = {"schema": "physx_beta_volume_fem", "spec": data["spec"],
                "nodes": len(data["points"]), "tetrahedra": len(data["tets"]),
                "skin_triangles": len(faces), "mass_kg": mass,
                "youngs_modulus_pa": youngs_modulus, "poissons_ratio": 0.3,
                "recommended_physics_hz": 480, "solver_position_iterations": 255,
                "material_volume_m3": data["material_volume_m3"],
                "description": "Elastic surrogate; material parameters are not measured real-paper calibration."}
    np.savez_compressed(output.parent/"rest_state.npz", points=data["points"].astype(np.float32),
                        tets=data["tets"], physx_tets=physx_tets, skin_ids=data["skin_ids"])
    (output.parent/"model.json").write_text(json.dumps(metadata, indent=2)+"\n", encoding="utf-8")
    return metadata


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=REPO/"workflows/simbox/assets/custom/objects/paper_cup_elastic_001/Aligned_obj.usd")
    ap.add_argument("--youngs-modulus", type=float, default=1e8)
    ap.add_argument("--mass", type=float, default=0.008)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    print(json.dumps(build(args.output, args.youngs_modulus, args.mass, args.force), indent=2))
