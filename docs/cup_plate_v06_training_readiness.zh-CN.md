# v0.6 杯盘数据：LeRobot v3 / π0.5 训练检查

检查日期：2026-10-28（Asia/Shanghai）。源目录：

```text
output/cup_plate_desk/v0.6/BananaBaseTask/split_aloha/custom/pick_cup_to_plate_desk/desk_edge
```

**结论：左臂杯盘任务的 VLA 监督字段齐全，支持转换为 LeRobot v3.0 和末端 TCP 增量动作。但发现运动期间图像重复，尚不能确认图像与状态严格同步，不建议直接将这批数据认定为正式微调训练集。** 10 条轨迹适合先验证训练流程；训练效果和泛化能力仍须用独立轨迹评测。

## 转换脚本

- 项目已跟踪的通用转换脚本为 `policy/lmdb2lerobotv21/lmdb2lerobot_split_aloha_a1.py`（LMDB → v2.1）和 `policy/lmdb2lerobotv21/convertv21_to_v30.py`（v2.1 → v3.0）。旧 split_aloha 脚本使用 `images.rgb.head` 等固定字段，不能直接认为适配本批 `external` 相机与 TCP 定义。
- 本工作区已有杯盘专用 `scripts/datasets/export_cup_plate_lerobot.py`，直接从原始 LMDB 导出 v3.0，配套 `local_configs/cup_plate_desk/lerobot_v3.json`。该脚本和配置在检查开始时已存在，但尚未被 Git 跟踪；不能据此认定上游发行版自带这份专用脚本。
- 本次为专用脚本补上 LeRobot 0.6 的 `RGBEncoderConfig` 接口兼容，保留 0.5 的 `vcodec` 分支。原依赖文件固定 `lerobot==0.5.1`，本机实际转换使用 LeRobot 0.6.2，数据格式版本仍为 `v3.0`。

## 原始数据逐帧检查

| 项目 | 结果 |
| --- | --- |
| Episode 数 | 10，数值轨迹各不相同 |
| 总步数 | 4,753 |
| 标称控制频率 / 总时长 | 30 Hz / 158.43 s |
| 各轨迹帧数（按目录时间排序） | 441、525、465、425、436、387、554、688、471、361 |
| RGB | external / hand_left / hand_right，全部 640×480×3 |
| 图像有效性 | 14,259 张全部解码成功；步号均覆盖 0 到 N−1 |
| 数值有效性 | 已检查的本体、动作、物体和相机外参数组均为有限值；时间序列长度一致 |
| 语言指令 | 全部包含 `Pick up the paper cup and put it on the plate.` |
| 当前实际状态 | 双臂关节、夹爪实际宽度、末端矩阵、机器人基座矩阵、关节速度 |
| 控制目标 | 双臂 `master_actions.*.position` 与夹爪 `openness` |
| 相机几何 | 三路内参、逐帧 `camera2env_pose.*` |
| 其他 | 杯盘位姿、尺度，光照/桌面/抓取随机化参数 |

左臂执行任务；右臂关节整条轨迹的最大跨度约 0.000059 rad，基本静止。夹爪指令每条轨迹均为开→关→开，源值为 0/1；实际总宽约 69.5–100 mm，夹住杯子时不会实际闭合到零。

末帧杯盘模型原点的平面距离约 0.24–2.26 mm，首/中/末图像抽查与放杯流程一致。这些观察支持轨迹结果，但本次没有重新运行物理接触、倾斜和稳定性成功判定，不能用原点距离单独证明放置成功。

## 需要优先处理：运动期间出现重复图像

对相邻源 JPEG 解码后的完整像素数组进行相等比较，同时计算实际末端的帧间运动。以位移 > 0.5 mm 或转角 > 0.5° 为“明显运动”阈值：

| 相机 | 全部相邻重复次数 | 其中末端明显运动的次数 |
| --- | ---: | ---: |
| external | 595 | 375 |
| hand_left | 600 | 378 |
| hand_right | 598 | 377 |

每路相机共有 4,743 个 episode 内相邻帧对。左手相机约 7.97% 的帧对在末端明显运动时仍像素完全相同；其中对应的末端单步平移最大约 8.66 mm。

例如 `2026-10-28_18_37_08_146340` 中，hand_left 的第 48、49 帧图像相同，记录的末端位置却变化约 6.414 mm。三路相机在大量相同位置同时重复，提示渲染帧可能未随每次记录刷新。**这是同步/图像新鲜度疑点；仅靠离线文件无法确定确切延迟及根因。** 静止阶段也会自然重复，以上运动阈值统计已单独区分。

源码中的 `plan_with_render()` 分别获取机器人观测、记录相机图像并推进仿真；`CustomCamera.get_observations()` 直接读取 `get_rgba()`，没有将相机渲染时间戳/帧号保存进本批数据。原始数据也没有可用于重建严格同步的逐帧采集时间戳。导出器只能按 `frame_index / 30` 生成标称时间戳，不能恢复缺失的视觉更新。

建议先在采集端记录 camera frame ID、camera timestamp、simulation step/time，确认每次记录取到对应的渲染帧，再重采或从保留的完整仿真回放信息重渲染。直接删除重复图像会改变动作时距；把 FPS 改成 15 也不能修复这种不规则重复。本次导出保留全部帧，并附带本报告的检查信息，没有静默过滤。

## 末端动作的准确含义

配置中 TCP 是 `link6` 沿自身 +Z 偏移 0.12 m，最大夹爪总宽 0.10 m。对每帧 t：

```text
T_measured_tcp[t] = T_base_ee_fl[t] @ T_link6_tcp
T_command_tcp[t]  = FK(master_actions.left_joint.position[t]) @ T_link6_tcp
DeltaT[t]         = inverse(T_measured_tcp[t]) @ T_command_tcp[t]
action[t]         = [DeltaT.translation, rotvec(DeltaT.rotation), gripper_openness]
```

- 动作共 7 维：当前 TCP 坐标系内的 dx/dy/dz（米）、局部旋转向量（弧度）、绝对夹爪指令（0 关 / 1 开）。这是相对位姿目标，不是速度，也不是直接相减的欧拉角。
- 状态共 8 维：单臂 `arm_base` 下实际 TCP xyz、姿态四元数 xyzw、实际宽度/最大宽度。
- 使用同帧控制目标 `master_actions`，避免把 logger 生成的“下一帧实际状态”字段 `actions.*` 混作真实控制命令。
- 每步动作相对该步实际 TCP；它既不是相邻目标之间的纯轨迹差分，也不是整块共用起始状态的相对动作。

检查结果：USD 实际末端与 URDF FK 最大位置差约 0.01352 mm、旋转差约 0.00290°。`T_measured_tcp @ DeltaT` 能重建目标，最大矩阵元素误差约 9.67e−10。动作平移范数中位数 1.012 mm、P99 9.875 mm、最大 11.802 mm；旋转范数最大 0.02775 rad（1.590°）。这些证明数值和坐标换算一致，不能证明图像同步或实机 TCP 已标定。

## π0.5 微调与执行适配

图像、任务文本、当前状态和监督动作是行为克隆所需的核心数据。常规 VLA 微调不要求额外具备深度、奖励或力反馈。这里 state=8/action=7 的维度可被 π0.5 适配；官方 LeRobot 文档也给出相同维数的 LIBERO 示例，但动作语义必须单独匹配。

1. 本地 `policy/openpi-InternData-A1/src/openpi/training/data_loader.py` 和 `mixture_dataset.py` 仍导入旧 `lerobot.common.datasets`，后者还假定 `meta.episodes` 是可 `.values()` 迭代的字典；现有数据配置使用另一套字段。需要升级/适配 v3 读取器及字段映射。模型文件包含 `pi05` 支持，但当前 `training/config.py` 没有现成的本批 7 维 TCP 动作 π0.5 训练配置。
2. 直接训练已经导出的 TCP 增量时，禁用再次执行 `DeltaActions`、`DeltaActionsPose` 或通用 `use_relative_actions=true` 所触发的减状态转换；部署端也不能套普通逐元素“加回 state”处理器。尤其 state 是 8 维四元数姿态，action 是 7 维旋转向量，不能按同维坐标处理。
3. 若按本格式训练动作块，执行第 k 步时应读取该时刻实际 TCP，再按 `T_target = T_actual @ DeltaT[k]` 解码；夹爪独立执行绝对指令。不要把整块每步都乘到推理时同一个 TCP。可以先用每次只执行一步的闭环验证，再评估块长度和重规划频率。若需要标准“整块相对起始状态”动作，应从绝对目标轨迹重新构造，重新计算对应统计量。
4. 正确映射 external/hand_left/hand_right 图像、任务 prompt、8 维 state、7 维 action，并按所选训练框架补齐 padding、图像处理和归一化配置。π0.5 的 quantile 模式需要 state/action 的 q01/q99；应检查实际导出结果，而非只检查存在 stats.json。
5. 实机执行需要使用相同的 TCP 定义和单位、旋转表示、夹爪范围及 IK。当前 0.12 m TCP 明确是仿真定义，未实机标定。若控制接口采用基坐标系增量或另一工具坐标系，需要显式变换。
6. 10 条轨迹约 2.64 分钟，适合训练链路验证；不能据此保证微调成功或实机泛化。训练/验证应按 episode 划分，避免相邻帧泄漏。

参考：[LeRobot v3.0 格式](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)、[π0.5 的维数与归一化要求](https://huggingface.co/docs/lerobot/pi05)、[相对动作与逐步增量的区别](https://huggingface.co/docs/lerobot/action_representations)。

## 检查产物与复现

- 已实际导出：`output/lerobot_v3/cup_plate_v06_tcp_relative/`，约 51 MB，LeRobotDataset `v3.0`，10 条 episode / 4,753 帧 / 30 Hz。
- 官方 LeRobot 0.6.2 读取器回读成功；全部数值行与源数据转换结果一致，episode 边界、frame_index 和标称 timestamp 全部通过核对。
- 三路视频分别完整解码 4,753 帧，PTS 均按 30 Hz 连续；各 episode 首、中、末帧共 90 张图像与源图像抽查匹配，平均绝对像素误差最大分别约 4.68 / 3.87 / 3.59（0–255，来自 H.264 有损压缩）。
- `meta/stats.json` 的 state/action 已包含 q01/q99；策略特征自动识别仅将 `action` 作为动作输出，辅助关节字段没有被混作策略输入输出。
- `meta/validation_summary.json` 记录格式验证结果，并明确 `training_ready=false`、`image_state_synchronization_validated=false`；`meta/source_quality_audit.json` 和 `meta/training_readiness.zh-CN.md` 随导出数据附带本次质量检查。格式验证通过不消除源图像重复问题。
- `docs/cup_plate_v06_data_audit.json`：全部 episode 的字段形状、有效性、数值误差、图像重复帧索引和运动统计。
- `docs/images/cup_plate_v06_data_audit.jpg`：10 条轨迹的外部/左手相机首、中、末帧拼图。
- 数值检查可运行：

```bash
python scripts/datasets/export_cup_plate_lerobot.py \
  --source output/cup_plate_desk/v0.6/BananaBaseTask/split_aloha/custom/pick_cup_to_plate_desk/desk_edge \
  --inspect-only
```

注意：`--inspect-only` 检查数值、图像键和首帧分辨率；本报告另行解码了全部图像并检查重复帧，不能把普通 inspect-only 通过等同于视觉时序合格。

重新导出可使用同一脚本的 `--output` 和 `--repo-id` 参数，输出目录须不存在。应从 LMDB 图像重编码；原目录 `demo.mp4` 是 logger 固定生成的 15 fps 预览，不宜直接替代 30 Hz 训练图像。

本次 9 项 TCP 坐标/对齐单元测试在 SciPy 1.15.2 和 1.18.1 下均通过；其中单轴欧拉角批量输入的测试改为显式 `(N,1)` 形状以兼容新版 SciPy。
