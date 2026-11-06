# 杯盘任务：LeRobot v3 与相对 TCP 动作

若需对齐 `session_102914` 的 **14 维双臂关节状态/动作**，请使用另一份[RobotCooking 格式导出说明](cup_plate_robotcooking_lerobot_v3.zh-CN.md)。本文及 `export_cup_plate_lerobot.py` 仍对应 7 维 TCP 增量动作，二者不能混用。

当前 v0.6 采集输出为 `output/cup_plate_desk/v0.6/`，可将下述 `--source` 改为该目录。v0.6 增加每条轨迹固定的光照和桌面外观随机化；v0.5 起外部相机为实机 D435i 的 640×480 模式，v0.4 及更早外部图像为 1280×720，应按版本分别导出，或先明确统一图像预处理。文末 916 帧 LeRobot 示例是此前 v0.2 数据，保留作格式验证；当前仍只采集原始 LMDB，按需离线转换。

2026-10-28 检查了 `v0.6/.../desk_edge` 的 10 条轨迹、4,753 帧，并导出到 `output/lerobot_v3/cup_plate_v06_tcp_relative/`。核心字段完整，但发现末端运动时相机图像重复，尚不能确认视觉与状态严格同步。训练前请先阅读[本批数据的训练就绪检查](cup_plate_v06_training_readiness.zh-CN.md)，不要将格式转换通过视为数据质量验收通过。导出脚本现兼容 LeRobot 0.5 的 `vcodec` 及 0.6 的 `RGBEncoderConfig` 接口。

当前导出采用左臂、LeRobotDataset v3.0，使用官方 `LeRobotDataset.create/add_frame/save_episode/finalize` 写入。只导出到本地，不上传 Hub，不连接实机，也不改变机器人关节或工具参数。J6 保留约 ±180°。

## 动作与状态

设 `T_base_tcp` 把 TCP 局部坐标转换到单臂 `arm_base`，每个记录帧 t 的动作是：

```text
T_current = 记录的 T_base_ee_fl[t] @ T_link6_tcp
T_target  = FK(master_actions.left_joint.position[t]) @ T_link6_tcp
DeltaT    = inverse(T_current) @ T_target

action = [dx, dy, dz, rotvec_x, rotvec_y, rotvec_z, gripper_openness]
```

- `dx/dy/dz`：当前 TCP 坐标系下的平移，单位米。
- `rotvec`：`DeltaT` 的旋转向量，方向为旋转轴、模长为弧度角；不是欧拉角差，也不是角速度。
- `gripper_openness`：绝对开合指令，0 关、1 开；不是夹爪增量，也不是接触力。本任务源指令只有 0/1。
- 动作针对当前帧发出的关节位置目标，不使用下一帧实际状态代替控制命令，不对指令额外错位一帧。
- `DeltaT` 是相对位姿，不是 SE(3) 六维 twist 的指数坐标：前三维直接取相对变换的平移，不经过左雅可比换算。

| 字段 | 维数 | 定义 |
| --- | --- | --- |
| `observation.state` | 8 | 当前实际 TCP 在 `arm_base` 下的绝对 xyz（米）、四元数 xyzw、实际夹爪总宽/最大宽 |
| `action` | 7 | 当前 TCP 下的相对平移、旋转向量、绝对夹爪指令 |
| `auxiliary.joint_positions` | 6 | 当前实际 J1～J6，弧度，辅助检查或重新导出 |
| `auxiliary.joint_targets` | 6 | 同帧控制器 J1～J6 目标，弧度，辅助检查或重新导出 |
| `observation.images.external` | RGB 视频 | 外部相机 |
| `observation.images.hand_left` | RGB 视频 | 左手相机 |
| `observation.images.hand_right` | RGB 视频 | 右手相机，保留原始视角；训练时可按配置选取 |
| `timestamp/frame_index/episode_index/task_index` | 官方字段 | 由 LeRobot 生成；自然语言任务来自 LMDB 元数据 |

状态和动作维数不同是有意的：状态保存无欧拉角奇异点的绝对姿态；动作保存局部旋转增量。训练数据适配器应将 `action` 选为策略输出，另外两个关节字段只作辅助。辅助字段采用独立前缀，使官方策略特征自动发现不会把它们识别为另一个动作输出或状态输入。

## TCP 坐标约定与实机

配置为 `local_configs/cup_plate_desk/lerobot_v3.json`。目前 `T_link6_tcp` 沿仿真 `link6` 的 +Z 偏移 0.12 m、旋转为单位矩阵，与本任务抓取代码中的 `tcp_offset` 对应；最大夹爪总宽 0.10 m。**这是仿真工具定义，尚未实机标定，不代表已确认的实机抓取中心。**

仿真内部继续使用现有 `link6`，不直接修改已验证的 USD、URDF 或抓取旋转。导出时先将实际状态和目标共同变换到配置的 TCP，再计算相对动作。以后确定实机工具轴和 TCP 后，更新完整的 4×4 变换并重新导出；不要给 J6 加 90°，也不要仅交换最终动作的两个分量。

即使采用相对动作，工具轴也不能忽略。如果新工具坐标与当前工具坐标之间的固定变换为 C，则新相对动作矩阵为 `inverse(C) @ DeltaT @ C`；包含工具原点偏移时，旋转还会影响相对平移。

实机执行时，先获得同一定义下的当前 TCP 位姿，再右乘预测的增量：

```python
T_base_tcp_target = apply_tcp_action(T_base_tcp_current, action)
# 如果底层接口接受 link6/法兰目标：
T_base_link6_target = T_base_tcp_target @ inverse(T_link6_tcp)
```

`apply_tcp_action` 可从导出脚本导入，仅进行位姿换算，不向机器人发送命令。实机还需将位姿转换为对应控制接口要求的单位、姿态表示，并经过 IK/限位处理。动作块中的每一步分别相对其对应时刻的 TCP；不能把整段所有增量都相对同一个起始 TCP 解码，也不能对旋转向量直接相加。当前仓库的旧 OpenPI 数据配置不能直接视作这套 7 维动作的部署适配器。

## 导出方法

在独立 Python 环境安装 `scripts/datasets/requirements-lerobot-v3.txt`，不要混入 Isaac Sim 环境。已用本机 Python 3.12、LeRobot 0.5.1 验证。

```bash
python -m pip install -r scripts/datasets/requirements-lerobot-v3.txt

python scripts/datasets/export_cup_plate_lerobot.py \
  --source output/cup_plate_desk \
  --inspect-only

python scripts/datasets/export_cup_plate_lerobot.py \
  --source output/cup_plate_desk \
  --output output/lerobot_v3/cup_plate_tcp_relative \
  --repo-id local/cup_plate_tcp_relative
```

从仓库根目录运行。`--source` 可指向单条 episode，或递归包含多条 episode 的目录。输出目录必须不存在，以免覆盖已有数据。仅输入可信的本地 LMDB：源格式含 pickle。

批量目录可追加 `--max-episodes 10`，仅处理按源路径排序的前十条完整 LMDB 轨迹。时间戳命名的同级目录会按时间先后排序；不传此参数则处理发现的全部轨迹。

默认按任务记录频率 30 Hz 输出，时间戳为 `frame_index / 30`。从 LMDB 中的逐帧图像重新编码视频，不复制旧的 15 fps 预览视频。每个相机必须覆盖连续的所有步；有缺帧或抽帧的数据会拒绝导出，需显式实现重采样后再处理，不能直接改 FPS 或截断长度。

源 logger 把 RGB 数组直接传给 OpenCV `imencode`。此导出器用 `imdecode` 还原原数组，随后直接交给 LeRobot，避免多做一次 BGR/RGB 交换。若更换源 logger 的通道处理，也需相应更新解码逻辑。

目录包括官方 `data/`、`videos/`、`meta/info.json`、统计及 episode 元数据。附加的 `meta/tcp_action_convention.json` 保存完整 TCP 配置、URDF 哈希、来源路径、动作定义及检查结果。原始 LMDB 中的物体真值和相机标定仍保留在源数据，本次不作为策略输入。

导出前逐帧检查 USD 实际末端与 URDF FK 一致性；超阈值直接报错，防止混入先前讨论的不同末端坐标约定。还检查数组长度、图像步号、夹爪范围，以及 `T_current @ DeltaT` 能否重建目标。

官方格式说明：[LeRobotDataset v3.0](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)。LeRobot 是存储格式，不会自动统一动作坐标或执行控制约定。

## 本次导出验证

已导出 `output/lerobot_v3/cup_plate_tcp_relative/`：源目录两条轨迹分别为 457、459 帧，共 916 帧，三个相机的视频均为 916 帧、30 Hz。官方 LeRobot 读取器已成功加载，两条 episode 的数值字段逐帧与源数据一致；抽查每条轨迹首、中、末帧的图像和任务文本，并核对全部视频时间戳与 episode 边界。31 项单元测试通过，其中 9 项覆盖本次坐标转换、TCP 偏移和指令对齐。

当前两条轨迹的 USD/URDF 末端最大位置差约 0.014 mm、姿态差约 0.0029°；这些是仿真内部一致性结果，不是实机精度。检查报告在导出目录的 `meta/validation_summary.json`。

```python
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset(
    repo_id="local/cup_plate_tcp_relative",
    root=Path("output/lerobot_v3/cup_plate_tcp_relative").resolve(),
    video_backend="pyav",
)
sample = dataset[0]
assert sample["action"].shape == (7,)
assert sample["observation.state"].shape == (8,)
```
