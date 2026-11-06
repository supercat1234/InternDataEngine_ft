import numpy as np
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.transformations import get_relative_transform
from omni.isaac.sensor import Camera


def look_at_usd_orientation(position, target, up=(0., 0., 1.)):
    """Return wxyz: USD camera looks along -Z, with +Y pointing upward."""
    from scipy.spatial.transform import Rotation

    forward = np.asarray(target, dtype=float) - np.asarray(position, dtype=float)
    if np.linalg.norm(forward) < 1e-9:
        raise ValueError("Camera position and look_at target must differ")
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray(up, dtype=float))
    if np.linalg.norm(right) < 1e-9:
        raise ValueError("Camera look direction must not be parallel to look_at_up")
    right /= np.linalg.norm(right)
    rotation = np.column_stack([right, np.cross(right, forward), -forward])
    return Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]]


def sample_camera_pose(cfg, max_translation_noise=.05, max_orientation_noise=10., rng=None):
    """Sample about the configured pose, never the previously perturbed pose.

    Scalar translation noise preserves the legacy sphere sampling. A three-
    element vector gives independent bounds in parent-frame X/Y/Z. Rotation
    uses a random parent-frame axis and a bounded total axis-angle rotation.
    """
    from scipy.spatial.transform import Rotation
    rng = rng if rng is not None else np.random
    position = np.asarray(cfg["translation"], dtype=float)
    orientation = np.asarray(cfg["orientation"], dtype=float)
    if cfg.get("look_at") is not None:
        if cfg["camera_axes"] != "usd":
            raise ValueError("look_at currently requires camera_axes: usd")
        orientation = look_at_usd_orientation(position, cfg["look_at"], cfg.get("look_at_up", [0., 0., 1.]))
    limits = np.asarray(max_translation_noise, dtype=float)
    if limits.shape not in [(), (3,)] or not np.isfinite(limits).all() or np.any(limits < 0):
        raise ValueError("Camera translation noise must be a nonnegative scalar or XYZ vector")
    angle_limit = float(max_orientation_noise)
    if not np.isfinite(angle_limit) or angle_limit < 0:
        raise ValueError("Camera rotation noise must be finite and nonnegative")
    if limits.ndim == 0:
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        offset = direction * rng.uniform(0., float(limits))
    else:
        offset = rng.uniform(-limits, limits)
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = float(rng.uniform(-angle_limit, angle_limit))
    delta = Rotation.from_rotvec(axis * np.deg2rad(angle))
    perturbed = delta * Rotation.from_quat(orientation[[1, 2, 3, 0]])
    quaternion = perturbed.as_quat()[[3, 0, 1, 2]]
    return {"translation": (position + offset).tolist(), "orientation_wxyz": quaternion.tolist(),
            "translation_offset_m": offset.tolist(), "rotation_angle_deg": angle,
            "rotation_frame": "parent", "camera_axes": cfg["camera_axes"]}


def _get_annotator(camera: Camera, annotator_name: str):
    custom_annotators = getattr(camera, "_custom_annotators", None)
    if not isinstance(custom_annotators, dict):
        return None
    return custom_annotators.get(annotator_name)


def _get_frame(frame):
    if isinstance(frame, np.ndarray) and frame.size > 0:
        return frame[:, :, :3]
    return None


def _get_depth(depth):
    if isinstance(depth, np.ndarray) and depth.size > 0:
        return depth
    return None


def _get_rgb_image(camera: Camera):
    output_mode = getattr(camera, "output_mode", "rgb")
    if output_mode == "rgb":
        return _get_frame(camera.get_rgba())
    if output_mode == "diffuse_albedo":
        annotator = _get_annotator(camera, "DiffuseAlbedo")
        if annotator is None:
            return None
        return _get_frame(annotator.get_data())
    raise NotImplementedError(f"Unsupported output mode: {output_mode}")


def _get_depth_image(camera: Camera):
    annotator = _get_annotator(camera, "distance_to_image_plane")
    if annotator is None:
        return None
    return _get_depth(annotator.get_data())


def _get_object_mask(camera: Camera):
    annotator = _get_annotator(camera, "semantic_segmentation")
    if annotator is None:
        return None
    annotation_data = annotator.get_data()
    if (
        not isinstance(annotation_data, dict)
        or "data" not in annotation_data
        or "info" not in annotation_data
    ):
        return None
    info = annotation_data["info"]
    if not isinstance(info, dict) or "idToLabels" not in info:
        return None
    mask = annotation_data["data"]
    if isinstance(mask, np.ndarray) and mask.size > 0:
        return {"mask": mask, "id2labels": info["idToLabels"]}
    return None


def _get_bbox(camera: Camera, bbox_type: str):
    annotator = _get_annotator(camera, bbox_type)
    if annotator is None:
        return None
    annotation_data = annotator.get_data()
    if (
        not isinstance(annotation_data, dict)
        or "data" not in annotation_data
        or "info" not in annotation_data
    ):
        return None
    info = annotation_data["info"]
    if not isinstance(info, dict) or "idToLabels" not in info:
        return None
    return annotation_data["data"], info["idToLabels"]


def _get_motion_vectors(camera: Camera):
    annotator = _get_annotator(camera, "motion_vectors")
    if annotator is None:
        return None
    annotation_data = annotator.get_data()
    if isinstance(annotation_data, np.ndarray) and annotation_data.size > 0:
        return annotation_data
    return None


def _get_camera2env_pose(camera: Camera):
    prim_path = getattr(camera, "prim_path", None)
    root_prim_path = getattr(camera, "root_prim_path", None)
    if not prim_path or not root_prim_path:
        return None
    return get_relative_transform(get_prim_at_path(prim_path), get_prim_at_path(root_prim_path))


def _get_camera_params(camera: Camera):
    camera_matrix = getattr(camera, "is_camera_matrix", None)
    if camera_matrix is None:
        return None
    if isinstance(camera_matrix, np.ndarray):
        return camera_matrix.tolist()
    return camera_matrix


def get_src(camera: Camera, data_type: str):
    if data_type == "rgb":
        return _get_rgb_image(camera)
    if data_type == "depth":
        return _get_depth_image(camera)
    if data_type == "seg":
        return _get_object_mask(camera)
    if data_type == "bbox2d_tight":
        return _get_bbox(camera, "bounding_box_2d_tight")
    if data_type == "bbox2d_loose":
        return _get_bbox(camera, "bounding_box_2d_loose")
    if data_type == "bbox3d":
        return _get_bbox(camera, "bounding_box_3d")
    if data_type == "motion_vectors":
        return _get_motion_vectors(camera)
    if data_type == "camera2env_pose":
        return _get_camera2env_pose(camera)
    if data_type == "camera_params":
        return _get_camera_params(camera)
    raise NotImplementedError(f"Unsupported source type: {data_type}")
