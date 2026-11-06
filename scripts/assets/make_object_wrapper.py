#!/usr/bin/env python3
"""把「刚体挂在默认根节点上」的 USD 包成引擎期望的 /World/Aligned 结构。

背景：
    rigid_object.py 加载物体时会做
        rigid_prim_path = <任务里的 prim_path> / <config 的 prim_path_child>
    Isaac 的 create_prim(usd_path=...) 会把 USD **默认根节点的子节点**铺到目标 prim 下，
    所以官方资产的写法是：
        /World  (defaultPrim)
          └── Aligned   ← 带 RigidBodyAPI，prim_path_child 写 "Aligned"
    OpenAI/OmniObject3D 转出来的资产经常是「默认根节点自己就是刚体」：
        /PaperCup (defaultPrim, 带 RigidBodyAPI)
    直接在 config 里写 prim_path_child: PaperCup 是找不到的（那一层被剥掉了），
    必须包一层。

用法：
    PYTHONPATH=$PXR LD_LIBRARY_PATH=... python make-object-wrapper.py \
        --src custom/objects/paper_cup/paper_cup.usd \
        --out custom/objects/paper_cup_001/Aligned_obj.usd
"""

import argparse
import os
import sys
import uuid
from pathlib import Path

from pxr import Sdf, Usd, UsdGeom, UsdPhysics

REPO = str(Path(__file__).resolve().parents[2])
ASSETS = os.path.join(REPO, "workflows/simbox/assets")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="源 USD（相对 asset_root 或绝对路径）")
    ap.add_argument("--out", required=True, help="输出 USD（相对 asset_root 或绝对路径）")
    ap.add_argument("--child", default="Aligned", help="承载刚体的子节点名，与 prim_path_child 对应")
    args = ap.parse_args()

    src = args.src if os.path.isabs(args.src) else os.path.join(ASSETS, args.src)
    out = args.out if os.path.isabs(args.out) else os.path.join(ASSETS, args.out)

    sst = Usd.Stage.Open(src)
    if not sst:
        sys.exit(f"打不开源文件: {src}")
    src_default = sst.GetDefaultPrim()
    if not src_default:
        sys.exit(f"源文件没有 defaultPrim: {src}")

    # 顺手检查刚体到底在哪
    rb = [p.GetPath().pathString for p in Usd.PrimRange(src_default) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    if not src_default.HasAPI(UsdPhysics.RigidBodyAPI):
        sys.exit(f"源文件的 defaultPrim 必须具有 RigidBodyAPI: {src}")
    if not any(p.IsA(UsdGeom.Mesh) for p in Usd.PrimRange(src_default)):
        sys.exit(f"源文件没有网格: {src}")
    if os.path.realpath(src) == os.path.realpath(out) and not os.path.islink(out):
        sys.exit("源文件和输出文件不能相同")

    os.makedirs(os.path.dirname(out), exist_ok=True)
    rel = os.path.relpath(src, os.path.dirname(out))
    # Export beside the destination, then atomically replace it; never write through a symlink.
    temp_out = os.path.join(os.path.dirname(out), f".wrapper-{uuid.uuid4().hex}.usd")
    dst = Usd.Stage.CreateNew(temp_out)
    UsdGeom.SetStageUpAxis(dst, UsdGeom.GetStageUpAxis(sst))
    # 声明成米制：网格数值原样引用，缩放由任务 YAML 的 scale 控制
    UsdGeom.SetStageMetersPerUnit(dst, 1.0)

    world = UsdGeom.Xform.Define(dst, Sdf.Path("/World")).GetPrim()
    dst.SetDefaultPrim(world)

    child = UsdGeom.Xform.Define(dst, Sdf.Path(f"/World/{args.child}")).GetPrim()
    child.GetReferences().AddReference(rel, src_default.GetPath())
    dst.GetRootLayer().Save()
    os.replace(temp_out, out)

    # 校验
    chk = Usd.Stage.Open(out)
    c = chk.GetPrimAtPath(f"/World/{args.child}")
    kids = [p.GetName() for p in c.GetChildren()]
    has_rb = c.HasAPI(UsdPhysics.RigidBodyAPI) or any(
        p.HasAPI(UsdPhysics.RigidBodyAPI) for p in Usd.PrimRange(c)
    )
    print(f"输出   : {out}")
    print(f"  defaultPrim      : {chk.GetDefaultPrim().GetPath().pathString}")
    print(f"  /World/{args.child} 子节点: {kids[:6]}")
    print(f"  含 RigidBodyAPI  : {has_rb}   （源里: {rb[:2]}）")
    meshes = [p for p in Usd.PrimRange(c) if p.IsA(UsdGeom.Mesh)]
    print(f"  mesh count       : {len(meshes)}")
    if not meshes or not c.HasAPI(UsdPhysics.RigidBodyAPI):
        sys.exit("包装层必须在 Aligned 节点具有刚体属性，并包含网格")


if __name__ == "__main__":
    main()
