"""Opt-in Isaac 5.1 volume FEM object with explicit nodal state management.

This adapter handles the object lifecycle. Existing rigid-object contact skills
and pose-only replay still need their own deformable-aware integration.
"""
import hashlib
import os

import numpy as np
from scipy.spatial.transform import Rotation
from core.objects.base_object import register_object
from core.utils.elastic_cup import cup_deformation, fit_cup_frame
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils.prims import create_prim, get_prim_at_path
from pxr import UsdGeom


def _numpy(value):
    return value.detach().cpu().numpy().copy() if hasattr(value, "detach") else np.asarray(value).copy()


@register_object
class DeformableObject(XFormPrim):
    """SI-scale volume FEM object; its estimated frame is fitted to bottom nodes.

    Call initialize() AFTER world.reset(). Before that, normal USD transforms
    place the object. During simulation, pose setters transform the complete
    nodal state and preserve deformation; reset_deformation() restores rest shape.
    """
    is_deformable = True

    def __init__(self, asset_root, root_prim_path, cfg, *args, **kwargs):
        import carb.settings
        from omni.physx.bindings import _physx as bindings
        if not carb.settings.get_settings().get_as_bool(bindings.SETTING_ENABLE_DEFORMABLE_BETA):
            raise RuntimeError("DeformableObject requires Isaac 5.1 Deformable beta; enable it before loading the scene")
        if not np.allclose(cfg.get("scale", [1, 1, 1]), [1, 1, 1], atol=1e-8, rtol=0):
            raise ValueError("DeformableObject assets are in meters; use scale [1, 1, 1]")
        self.asset_root = os.path.abspath(asset_root)
        self.base_prim_path = f"{root_prim_path}/{cfg['name']}"
        self.init_translation = cfg.get("init_translation")
        self.init_orientation = cfg.get("init_orientation")
        self.init_parent = cfg.get("init_parent")
        self.gap = cfg.get("gap")
        self._deform_view = None
        self._simulation_view = None
        create_prim(self.base_prim_path, usd_path=os.path.join(self.asset_root, cfg["path"]))
        path = self.base_prim_path+"/"+cfg.get("prim_path_child", "Aligned")
        prim = get_prim_at_path(path)
        if not prim.IsValid() or not prim.HasAPI("OmniPhysicsDeformableBodyAPI"):
            raise ValueError(f"Expected a native deformable body at {path}")
        self.simulation_mesh_path = path+"/"+cfg.get("simulation_mesh", "SimulationMesh")
        self.mesh_prim_path = path+"/"+cfg.get("visual_mesh", "ElasticMesh")
        self.mesh_prim_paths = [self.mesh_prim_path]
        sim = UsdGeom.TetMesh(get_prim_at_path(self.simulation_mesh_path))
        if not sim:
            raise ValueError("DeformableObject requires an explicit tetrahedral SimulationMesh")
        self.rest_nodes = np.asarray(sim.GetPointsAttr().Get(), dtype=np.float32).copy()
        tets = np.asarray(sim.GetTetVertexIndicesAttr().Get(), dtype=np.int32)
        self.topology_id = hashlib.sha256(self.rest_nodes.tobytes()+tets.tobytes()).hexdigest()
        self._reference_ids = np.flatnonzero(self.rest_nodes[:, 2] <= self.rest_nodes[:, 2].min()+0.0013)
        self.mass = prim.GetAttribute("omniphysics:mass").Get()
        super().__init__(prim_path=path, name=cfg["name"], *args, **kwargs)
        import omni.timeline
        self._stop_subscription = omni.timeline.get_timeline_interface().get_timeline_event_stream().create_subscription_to_pop_by_type(
            int(omni.timeline.TimelineEventType.STOP), self._invalidate_view)

    def _invalidate_view(self, event):
        self._deform_view = None
        self._simulation_view = None

    def initialize(self, physics_sim_view=None):
        import carb.settings
        import omni.physics.tensors
        if not carb.settings.get_settings().get_as_bool("/physics/suppressReadback"):
            raise RuntimeError("DeformableObject requires a CUDA World and GPU dynamics")
        # An explicit tensor frontend avoids depending on the rest of the task's backend.
        self._simulation_view = omni.physics.tensors.create_simulation_view("torch")
        self._simulation_view.set_subspace_roots("/")
        self._deform_view = self._simulation_view.create_volume_deformable_body_view(self.prim_path)
        if self._deform_view.count != 1 or self._deform_view.max_simulation_nodes_per_body != len(self.rest_nodes):
            self._deform_view = None
            raise RuntimeError("FEM body is not ready or its topology changed; initialize after world.reset()")

    def get_deformable_state(self):
        if self._deform_view is None:
            raise RuntimeError("Initialize the deformable object after world.reset() before reading nodes")
        return {"topology_id": self.topology_id,
                "nodal_positions": _numpy(self._deform_view.get_simulation_nodal_positions()[0]),
                "nodal_velocities": _numpy(self._deform_view.get_simulation_nodal_velocities()[0])}

    def set_deformable_state(self, state):
        import torch
        if self._deform_view is None:
            raise RuntimeError("Initialize the deformable object before restoring nodes")
        if state.get("topology_id") != self.topology_id:
            raise ValueError("Cannot restore a state from a different FEM mesh")
        positions = np.asarray(state["nodal_positions"], dtype=np.float32)
        velocities = np.asarray(state["nodal_velocities"], dtype=np.float32)
        if positions.shape != self.rest_nodes.shape or velocities.shape != self.rest_nodes.shape:
            raise ValueError("Nodal state shape does not match the FEM mesh")
        if not np.isfinite(positions).all() or not np.isfinite(velocities).all():
            raise ValueError("Nodal state contains non-finite values")
        target = self._deform_view.get_simulation_nodal_positions()
        device = target.device
        indices = torch.tensor([0], dtype=torch.int32, device=device)
        self._deform_view.set_simulation_nodal_positions(torch.as_tensor(positions[None], device=device), indices)
        self._deform_view.set_simulation_nodal_velocities(torch.as_tensor(velocities[None], device=device), indices)

    def _parent_transform(self):
        matrix = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(self.prim.GetParent())).T
        if (not np.allclose(matrix[:3, :3].T@matrix[:3, :3], np.eye(3), atol=1e-5)
                or np.linalg.det(matrix[:3, :3]) < 0):
            raise ValueError("DeformableObject parent must have unit scale")
        return matrix[:3, :3], matrix[:3, 3]

    def get_world_pose(self):
        if self._deform_view is None:
            return tuple(_numpy(x) for x in super().get_world_pose())
        positions = self.get_deformable_state()["nodal_positions"]
        return fit_cup_frame(self.rest_nodes[self._reference_ids], positions[self._reference_ids])

    def get_local_pose(self):
        if self._deform_view is None:
            return tuple(_numpy(x) for x in super().get_local_pose())
        p, q = self.get_world_pose()
        r_parent, p_parent = self._parent_transform()
        r_local = r_parent.T@Rotation.from_quat(q, scalar_first=True).as_matrix()
        return r_parent.T@(p-p_parent), Rotation.from_matrix(r_local).as_quat(scalar_first=True)

    def set_world_pose(self, position=None, orientation=None):
        if self._deform_view is None:
            return super().set_world_pose(position, orientation)
        p, q = self.get_world_pose()
        p_new = p if position is None else np.asarray(position)
        q_new = q if orientation is None else np.asarray(orientation)
        delta = Rotation.from_quat(q_new, scalar_first=True)*Rotation.from_quat(q, scalar_first=True).inv()
        state = self.get_deformable_state()
        state["nodal_positions"] = delta.apply(state["nodal_positions"]-p)+p_new
        state["nodal_velocities"] = delta.apply(state["nodal_velocities"])
        self.set_deformable_state(state)

    def set_local_pose(self, translation=None, orientation=None):
        if self._deform_view is None:
            return super().set_local_pose(translation, orientation)
        old_p, old_q = self.get_local_pose()
        r_parent, p_parent = self._parent_transform()
        p = old_p if translation is None else np.asarray(translation)
        q = old_q if orientation is None else np.asarray(orientation)
        self.set_world_pose(r_parent@p+p_parent,
                            Rotation.from_matrix(r_parent@Rotation.from_quat(q, scalar_first=True).as_matrix()).as_quat(scalar_first=True))

    def set_local_scale(self, scale):
        if not np.allclose(scale, [1, 1, 1], atol=1e-8, rtol=0):
            raise ValueError("Elastic FEM mesh must retain scale [1, 1, 1]; rebuild the asset to resize it")
        return super().set_local_scale(scale)

    def reset_deformation(self, position=None, orientation=None):
        """Restore rest shape at a world pose and clear ALL nodal velocities."""
        if self._deform_view is None:
            raise RuntimeError("Initialize the deformable object before resetting its shape")
        old_p, old_q = self.get_world_pose()
        p = old_p if position is None else np.asarray(position)
        q = old_q if orientation is None else np.asarray(orientation)
        positions = Rotation.from_quat(q, scalar_first=True).apply(self.rest_nodes)+p
        self.set_deformable_state({"topology_id": self.topology_id,
                                   "nodal_positions": positions,
                                   "nodal_velocities": np.zeros_like(self.rest_nodes)})

    def get_linear_velocity(self):
        return self.get_deformable_state()["nodal_velocities"].mean(0)

    def get_angular_velocity(self):
        state = self.get_deformable_state()
        r = state["nodal_positions"]-state["nodal_positions"].mean(0)
        velocity = state["nodal_velocities"]-state["nodal_velocities"].mean(0)
        eye = np.broadcast_to(np.eye(3), (len(r), 3, 3))
        matrices = np.cross(eye, r[:, None, :]).transpose(0, 2, 1)
        return np.linalg.lstsq(matrices.reshape(-1, 3), velocity.ravel(), rcond=None)[0]

    def get_observations(self):
        translation, orientation = self.get_local_pose()
        state = self.get_deformable_state()
        return {"translation": translation, "orientation": orientation,
                "scale": np.ones(3, dtype=np.float32),
                "nodal_positions_world": state["nodal_positions"],
                "nodal_velocities_world": state["nodal_velocities"]}

    def get_deformation_metrics(self):
        return cup_deformation(self.rest_nodes, self.get_deformable_state()["nodal_positions"])
