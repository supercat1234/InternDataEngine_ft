#!/usr/bin/env python3
"""Generate cup grasp labels in meters and GraspNet row-major format.

The center describes the midpoint between the jaws. Rim mode pinches one wall
between an inner and outer finger; the original side mode is kept for old data.
"""

import argparse
from pathlib import Path

import numpy as np


def make_side_grasps(diameter=.075, height=.08, n_yaw=24, grasp_heights=(.045, .05, .055),
                     downward_tilts=(1., 15., 25., 30.)):
    if not 0 < diameter <= .1 or height <= 0 or n_yaw < 1:
        raise ValueError("Cup diameter must fit a 0.1 m gripper; dimensions/yaw count must be positive")
    tilts = np.asarray(downward_tilts, dtype=float)
    if not len(tilts) or not np.isfinite(tilts).all() or np.any((tilts <= 0) | (tilts > 30)):
        raise ValueError("Downward approach tilts must be in (0, 30] degrees")
    rows = []
    for z in grasp_heights:
        if not .02 < z < height - .015:
            raise ValueError("Side grasp must leave clearance from the cup base and rim")
        for theta in np.linspace(0, 2 * np.pi, n_yaw, endpoint=False):
            closing = np.array([-np.sin(theta), np.cos(theta), 0.])
            for tilt in np.deg2rad(tilts):
                approach = np.array([-np.cos(theta) * np.cos(tilt),
                                     -np.sin(theta) * np.cos(tilt), -np.sin(tilt)])
                rotation = np.column_stack([approach, closing, np.cross(approach, closing)])
                row = np.zeros(17, dtype=np.float32)
                # The position is already the jaw midpoint. Extra insertion
                # would move the palm into the tilted cup before closing.
                row[:4] = [.1 + abs(z - height * .625), diameter, .02, 0.]
                row[4:13] = rotation.reshape(-1)
                row[13:16] = [0., 0., z]
                row[16] = -1
                rows.append(row)
    if not rows:
        raise ValueError("At least one grasp height is required")
    return np.stack(rows)


def make_rim_grasps(n_yaw=24, grasp_heights=(.065, .068, .071, .074), opening=.050,
                    tilts_from_vertical=(0., 2., 5., 10., 15., -2., -5., -10.)):
    """Vertical entry: one finger inside, one outside the SAME cup wall.

    The target is the local wall mid-surface, not the cup axis. Dimensions are
    read from the existing rigid cup: radii 26.5 -> 36.2 mm over 78.75 mm,
    1.2 mm wall, with a rolled lip reaching radius 37.5 mm at z=78.75 mm.
    Heights specify the TCP center, not the fingertip contact. Upper-rim
    candidates require physical validation; the cone is not extended past its lip.
    """
    if not isinstance(n_yaw, int) or n_yaw < 1 or not .012 <= opening <= .05:
        raise ValueError("Invalid yaw count or rim-entry opening")
    tilts = np.asarray(tilts_from_vertical, dtype=float)
    if tilts.ndim != 1 or not len(tilts) or not np.isfinite(tilts).all() or np.any(np.abs(tilts) > 20):
        raise ValueError("Signed rim gripper inclination must be between -20 and 20 degrees")
    rows = []
    for z in grasp_heights:
        if not np.isfinite(z) or not .060 <= z <= .080:
            raise ValueError("Rim grasp TCP height must be between 60 and 80 mm")
        radius = .0265+(.0362-.0265)*min(z, .07875)/.07875-.0012/2
        for theta in np.linspace(0, 2*np.pi, n_yaw, endpoint=False):
            radial = np.array([np.cos(theta), np.sin(theta), 0.])
            tangent = np.array([-np.sin(theta), np.cos(theta), 0.])
            for tilt in np.deg2rad(tilts):
                # Incline tangentially so radial jaw closure stays horizontal.
                approach = tangent*np.sin(tilt)+np.array([0., 0., -np.cos(tilt)])
                rotation = np.column_stack([approach, radial, np.cross(approach, radial)])
                row = np.zeros(17, dtype=np.float32)
                row[:4] = [.1+abs(z-.068), opening, .02, 0.]
                row[4:13] = rotation.ravel()
                row[13:16] = radius*radial+np.array([0., 0., z])
                row[16] = -1
                rows.append(row)
    if not rows:
        raise ValueError("At least one grasp height is required")
    return np.stack(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["rim", "side"], default="rim")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--diameter", type=float, default=.075)
    parser.add_argument("--height", type=float, default=.08)
    parser.add_argument("--n-yaw", type=int, default=24)
    parser.add_argument("--grasp-heights", type=float, nargs="+")
    parser.add_argument("--downward-tilts", type=float, nargs="+", default=[1., 15., 25., 30.])
    parser.add_argument("--rim-tilts", type=float, nargs="+", default=[0., 2., 5., 10., 15., -2., -5., -10.])
    parser.add_argument('--rim-opening', type=float, default=.050, help='Total jaw opening in rim mode, meters')
    args = parser.parse_args()
    if args.mode == "rim":
        grasps = make_rim_grasps(args.n_yaw, args.grasp_heights or [.065, .068, .071, .074],
                                 opening=args.rim_opening, tilts_from_vertical=args.rim_tilts)
    else:
        grasps = make_side_grasps(args.diameter, args.height, args.n_yaw, args.grasp_heights or [.045, .05, .055], args.downward_tilts)
    if args.out is None:
        args.out = Path(__file__).resolve().parents[2]/f"local_configs/cup_plate_desk/grasps/paper_cup_{args.mode}.npy"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, grasps)
    print(f"Saved {len(grasps)} {args.mode} grasps: {args.out}")


if __name__ == "__main__":
    main()
