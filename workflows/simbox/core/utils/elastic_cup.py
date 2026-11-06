"""Geometry and state math for the opt-in elastic cup (SI units, no Isaac imports)."""
from dataclasses import asdict, dataclass

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class ElasticCupSpec:
    # Dimensions measured from the existing paper_cup USD, after its 0.001 scale.
    height: float = 0.080
    wall_top: float = 0.07875
    bottom_radius: float = 0.0265
    top_wall_radius: float = 0.0362
    wall_thickness: float = 0.0012
    bottom_thickness: float = 0.0012
    rim_radius: float = 0.0375
    sectors: int = 32
    wall_segments: int = 12
    bottom_rings: int = 4

    def validate(self):
        vals = np.asarray([self.height, self.wall_top, self.bottom_radius,
                           self.top_wall_radius, self.wall_thickness,
                           self.bottom_thickness, self.rim_radius])
        if not np.isfinite(vals).all() or (vals <= 0).any():
            raise ValueError("Cup dimensions must be finite and positive")
        if not self.bottom_thickness < self.wall_top < self.height:
            raise ValueError("Expected bottom_thickness < wall_top < height")
        if self.wall_thickness >= self.bottom_radius or self.rim_radius < self.top_wall_radius:
            raise ValueError("Invalid wall thickness or rim radius")
        for key, minimum in (("sectors", 12), ("wall_segments", 4), ("bottom_rings", 2)):
            val = getattr(self, key)
            if isinstance(val, bool) or not isinstance(val, int) or val < minimum:
                raise ValueError(f"{key} must be an integer >= {minimum}")


def signed_tet_volumes(points, tets):
    q = np.asarray(points, dtype=np.float64)[np.asarray(tets)]
    return np.einsum("ij,ij->i", q[:, 1]-q[:, 0],
                     np.cross(q[:, 2]-q[:, 0], q[:, 3]-q[:, 0])) / 6


def boundary_triangles(tets):
    """Outward faces for positively oriented tetrahedra; reject non-manifold cells."""
    faces = np.asarray(tets)[:, [[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]]].reshape(-1, 3)
    _, first, counts = np.unique(np.sort(faces, axis=1), axis=0,
                                 return_index=True, return_counts=True)
    if np.any(counts > 2):
        raise ValueError("Non-manifold tetrahedral mesh")
    return faces[first[counts == 1]]


def make_elastic_cup(spec=ElasticCupSpec()):
    """A connected, closed MATERIAL volume: annular walls plus a disk bottom.

    Explicit tets avoid voxel cooking that could bridge/fill the cup cavity.
    The small rounded lip is approximated by several annular cross sections.
    Every prism uses the same vertex-order diagonal rule, including the seam.
    """
    spec.validate()
    n, rings = spec.sectors, spec.bottom_rings
    # A planar disk topology with an extra outer annulus for the wall.
    ring_ids = [np.arange(1 + k*n, 1 + (k+1)*n) for k in range(rings+1)]
    disk_triangles, wall_triangles = [], []
    for j in range(n):
        j1 = (j+1) % n
        disk_triangles.append([0, ring_ids[0][j], ring_ids[0][j1]])
        for k in range(rings):
            a, b, c, d = ring_ids[k][j], ring_ids[k+1][j], ring_ids[k+1][j1], ring_ids[k][j1]
            tris = [[a, b, c], [a, c, d]]
            (wall_triangles if k == rings-1 else disk_triangles).extend(tris)
    wall_ids = np.concatenate(ring_ids[-2:])
    all_ids = np.arange(1+(rings+1)*n)
    levels = list(np.linspace(spec.bottom_thickness, 0.0775, spec.wall_segments+1))
    levels = [0.0] + levels + [0.0780, 0.07875, 0.0795, spec.height]
    points, maps = [], []
    for level, z in enumerate(levels):
        outer = spec.bottom_radius + (spec.top_wall_radius-spec.bottom_radius)*z/spec.wall_top
        inner = outer-spec.wall_thickness
        if z >= 0.0780:
            outer, inner = {0.0780: (0.03725, 0.0349), 0.07875: (spec.rim_radius, 0.035),
                            0.0795: (0.03725, 0.03525), spec.height: (0.03665, 0.03585)}[z]
        mapping = {}
        for idx in (all_ids if level < 2 else wall_ids):
            if idx == 0:
                p = [0.0, 0.0, z]
            else:
                ring, sector = divmod(int(idx)-1, n)
                radius = inner*(ring+1)/rings if ring < rings else outer
                theta = 2*np.pi*sector/n
                p = [radius*np.cos(theta), radius*np.sin(theta), z]
            mapping[int(idx)] = len(points)
            points.append(p)
        maps.append(mapping)
    tets = []
    for level in range(len(levels)-1):
        triangles = disk_triangles+wall_triangles if level == 0 else wall_triangles
        lo, hi = maps[level], maps[level+1]
        for tri in triangles:
            i, j, k = sorted(tri)
            a, b, c, A, B, C = lo[i], lo[j], lo[k], hi[i], hi[j], hi[k]
            tets.extend([[a, b, c, C], [a, b, B, C], [a, A, B, C]])
    points, tets = np.asarray(points, dtype=np.float64), np.asarray(tets, dtype=np.int32)
    volumes = signed_tet_volumes(points, tets)
    neg = volumes < 0
    tets[neg, :2] = tets[neg, 1::-1]
    volumes = signed_tet_volumes(points, tets)
    # PhysX's inverse-volume setup rejects determinants <= 1e-9 in SI units.
    # An overly fine disk or a nearly closed lip creates tiny positive-volume tets.
    if np.any(volumes <= 1e-9/6):
        raise ValueError("Tetrahedra are too small for PhysX; coarsen the mesh or thicken the lip cap")
    faces = boundary_triangles(tets)
    skin_ids, skin_faces = np.unique(faces, return_inverse=True)
    return dict(points=points, tets=tets, skin_ids=skin_ids,
                skin_faces=skin_faces.reshape(-1, 3), spec=asdict(spec),
                material_volume_m3=float(volumes.sum()))


def fit_cup_frame(rest, positions):
    """Least-squares rigid reference frame, NOT a claim that the cup is rigid."""
    rest, positions = np.asarray(rest), np.asarray(positions)
    if rest.shape != positions.shape or rest.ndim != 2 or rest.shape[1] != 3:
        raise ValueError("Expected matching (N, 3) reference and current nodes")
    if len(rest) < 3 or not np.isfinite(rest).all() or not np.isfinite(positions).all():
        raise ValueError("Invalid cup nodes")
    a, b = rest.mean(0), positions.mean(0)
    u, _, vt = np.linalg.svd((rest-a).T @ (positions-b))
    correction = np.diag([1, 1, np.linalg.det(vt.T @ u.T)])
    rotation = vt.T @ correction @ u.T
    return b-rotation@a, Rotation.from_matrix(rotation).as_quat(scalar_first=True)


def cup_deformation(rest, positions):
    translation, orientation = fit_cup_frame(rest, positions)
    local = (np.asarray(positions)-translation) @ Rotation.from_quat(orientation, scalar_first=True).as_matrix()
    displacement = np.linalg.norm(local-rest, axis=1)
    band = (np.asarray(rest)[:, 2] >= 0.040) & (np.asarray(rest)[:, 2] <= 0.060)
    return dict(max_displacement_m=float(displacement.max()),
                rms_displacement_m=float(np.sqrt(np.mean(displacement**2))),
                grasp_band_width_x_m=float(np.ptp(local[band, 0])),
                grasp_band_width_y_m=float(np.ptp(local[band, 1])))
