import glob
import math
import os
import random

import numpy as np

from core.objects.base_object import register_object
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils.prims import is_prim_path_valid
from omni.isaac.core.utils.stage import get_current_stage

try:
    from omni.isaac.core.materials.omni_pbr import OmniPBR  # Isaac Sim 4.1.0 / 4.2.0
except ImportError:
    from isaacsim.core.api.materials import OmniPBR  # Isaac Sim 4.5.0

from pxr import Sdf, UsdGeom, UsdShade


@register_object
class PlaneObject(XFormPrim):
    def __init__(self, asset_root, root_prim_path, cfg, *args, **kwargs):
        """
        Args:
            asset_root: Asset root path
            root_prim_path: Root prim path in USD stage
            cfg: Config dict with required keys:
                - name: Object name
                - size: [width, length] of the plane
        """
        # ===== From cfg =====
        self.asset_root = asset_root
        prim_path = os.path.join(root_prim_path, cfg["name"])
        self.cfg = cfg

        # ===== Initialize =====
        stage = get_current_stage()
        if cfg.get("tile_size"):
            UsdGeom.Xform.Define(stage, prim_path)
            self._create_tiles(stage, prim_path, cfg)
        else:
            plane_geom = UsdGeom.Plane.Define(stage, prim_path)
            plane_geom.CreateWidthAttr().Set(cfg["size"][0])
            plane_geom.CreateLengthAttr().Set(cfg["size"][1])
        super().__init__(prim_path=prim_path, name=cfg["name"], *args, **kwargs)

    def _create_tiles(self, stage, prim_path, cfg):
        """Visual tile quads with per-tile UVs and a flat grout underlay."""
        width, length = map(float, cfg["size"])
        tw, th = map(float, cfg["tile_size"])
        gap = float(cfg.get("grout_width", .002))
        if min(width, length, tw, th) <= 0 or not 0 <= gap < min(tw, th):
            raise ValueError("Floor/tile dimensions must be positive; grout must be smaller than a tile")

        def mesh(path, points, uv=None):
            geom = UsdGeom.Mesh.Define(stage, path)
            geom.CreatePointsAttr(points)
            geom.CreateFaceVertexCountsAttr([4] * (len(points) // 4))
            geom.CreateFaceVertexIndicesAttr(list(range(len(points))))
            geom.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
            geom.CreateDoubleSidedAttr(True)
            if uv is not None:
                UsdGeom.PrimvarsAPI(geom).CreatePrimvar(
                    "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying).Set(uv)
            return geom

        points, uv = [], []
        for i in range(math.ceil(width / tw)):
            for j in range(math.ceil(length / th)):
                x, y = -width / 2 + i * tw, -length / 2 + j * th
                x0, y0 = x + gap / 2, y + gap / 2
                x1, y1 = min(x + tw - gap / 2, width / 2), min(y + th - gap / 2, length / 2)
                if x1 <= x0 or y1 <= y0:
                    continue
                points.extend([(x0, y0, 0), (x1, y0, 0), (x1, y1, 0), (x0, y1, 0)])
                # Cut edge tiles retain their physical texture scale.
                u0, v0, u1, v1 = gap / (2 * tw), gap / (2 * th), (x1 - x) / tw, (y1 - y) / th
                uv.extend([(u0, v0), (u1, v0), (u1, v1), (u0, v1)])
        mesh(f"{prim_path}/Tiles", points, uv)
        self._grout_path = f"{prim_path}/Grout"
        mesh(self._grout_path, [(-width / 2, -length / 2, -.0001),
             (width / 2, -length / 2, -.0001), (width / 2, length / 2, -.0001),
             (-width / 2, length / 2, -.0001)])

    def get_observations(self):
        raise NotImplementedError

    def apply_texture(self, asset_root, cfg, use_mem=False):
        if cfg.get("texture_options"):
            from core.utils.appearance_randomization import sample_texture_option
            state = cfg.get("mem_texture") if use_mem else None
            state = state or sample_texture_option(cfg["texture_options"], asset_root,
                                                   cfg.get("apply_randomization", False))
            cfg["mem_texture"] = state
            texture_path = state["texture_file"]
            print(f"Floor texture {self.name}: {state}", flush=True)
        else:
            texture_name = cfg["texture_lib"]
            texture_path_list = glob.glob(os.path.join(asset_root, texture_name, "*"))
            texture_path_list.sort()
            if cfg["apply_randomization"]:
                texture_id = random.randint(0, len(texture_path_list) - 1)
            else:
                texture_id = cfg["texture_id"]
            texture_path = texture_path_list[texture_id]
        mat_prim_path = f"{self.prim_path}/Looks/Material"
        if not is_prim_path_valid(mat_prim_path):
            self.mat = OmniPBR(
                prim_path=mat_prim_path,
                name="Material",
                texture_path=texture_path,
                texture_scale=cfg.get("texture_scale"),
            )
            self.apply_visual_material(self.mat, weaker_than_descendants=bool(self.cfg.get("tile_size")))
        else:
            self.mat.set_texture(
                texture_path,
            )
        if self.cfg.get("tile_size"):
            # Each tile has its own 0..1 UVs; world projection would ignore them.
            self.mat.set_project_uvw(False)
            if not hasattr(self, "_grout_mat"):
                self._grout_mat = OmniPBR(
                    prim_path=f"{self.prim_path}/Looks/GroutMaterial",
                    name=f"{self.name}_grout",
                    color=np.asarray(self.cfg.get("grout_color", [.12, .12, .12])))
                UsdShade.MaterialBindingAPI.Apply(get_current_stage().GetPrimAtPath(self._grout_path)).Bind(
                    self._grout_mat.material)
        if "roughness" in cfg:
            self.mat.set_reflection_roughness(float(cfg["roughness"]))
