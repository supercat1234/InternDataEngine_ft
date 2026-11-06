"""Pure, simulator-independent PI05 joint/RGB adapter."""
import numpy as np

CAMERAS = {'TOP_CAM': 'split_aloha_external', 'LEFT_WRIST_CAM': 'split_aloha_hand_left',
           'RIGHT_WRIST_CAM': 'split_aloha_hand_right'}
TASK = 'pick up the paper cup and put it at the plate'
ACTION_NAMES = [f'{side}_{name}.pos' for side in ('left', 'right')
                for name in [*(f'joint_{i}' for i in range(1, 7)), 'gripper']]


def letterbox(image):
    import cv2
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8 or not image.size:
        raise ValueError('Expected nonempty RGB uint8 HWC image')
    h, w = image.shape[:2]
    scale = min(256 / h, 256 / w)
    nh, nw = round(h * scale), round(w * scale)
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    result = np.zeros((256, 256, 3), np.uint8)
    y, x = (256 - nh) // 2, (256 - nw) // 2
    result[y:y+nh, x:x+nw] = resized
    return result


def observation(raw):
    robot = raw['robots']['split_aloha']
    state = np.concatenate([np.ravel(robot[f'states.{side}_{part}.position']) / divisor
                            for side in ('left', 'right')
                            for part, divisor in [('joint', 1), ('gripper', .1)]]).astype(np.float32)
    if state.shape != (14,) or not np.isfinite(state).all():
        raise ValueError('Invalid measured joint state')
    return {'observation.state': state, **{f'observation.images.{name}': letterbox(raw['cameras'][camera]['color_image'])
                                         for name, camera in CAMERAS.items()}}


def indices(robot):
    return np.array([*robot.left_joint_indices, *robot.left_gripper_indices,
                     *robot.right_joint_indices, *robot.right_gripper_indices], dtype=int)


def targets(action, limits):
    action = np.asarray(action, dtype=np.float32)
    if action.shape != (14,) or not np.isfinite(action).all():
        raise ValueError('Expected 14 finite absolute joint/gripper targets')
    applied = action.copy()
    applied[[6, 13]] = np.clip(applied[[6, 13]], 0, 1)
    applied[[6, 13]] *= .05  # Total width .1m; active finger moves half.
    limits = np.asarray(limits)
    if limits.shape != (14, 2):
        raise ValueError('Expected selected USD limits [14,2]')
    bounded = np.clip(applied, limits[:, 0], limits[:, 1])
    tolerance = np.full(14, np.deg2rad(5))
    tolerance[[6, 13]] = .001
    if np.any(np.abs(bounded - applied) > tolerance):
        raise ValueError(f'Predicted action substantially exceeds USD limits: targets={applied.tolist()}, limits={limits.tolist()}')
    return bounded
