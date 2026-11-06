# 杯盘任务原始数据与规划失败诊断

本文记录 v0.3 实施前的诊断。前四项优化已实施，当前代码与验证见 [倾斜抓取和整链规划验证](cup_plate_chain_validation.zh-CN.md)。

2026-10-28。此次没有修改机器人限位、速度上限、质量、工具轴或任务布局。保留运行时 500 rad/s 上限；实机工具轴差异在接口层转换。LeRobot 是采集后的离线转换步骤。

## 当前实际输出

以下对应本任务的 `plan_with_render` 加 LMDB writer，不代表引擎所有工作流的可选输出。

| 数据 | 当前字段或文件 |
| --- | --- |
| 双臂实际六关节位置 | `states.left_joint.position`、`states.right_joint.position`，弧度 |
| 双臂实际夹爪总宽 | `states.left_gripper.position`、`states.right_gripper.position`，米 |
| articulation 关节速度 | `qvel`，包含整机 DOF，需按机器人索引解释 |
| 双臂实际 link6 相对各自臂基座的变换 | `T_base_ee_fl`、`T_base_ee_fr` |
| 机器人根位姿 | `T_world_base` |
| 同帧关节位置控制目标 | `master_actions.left_joint.position`、`master_actions.right_joint.position` |
| 同帧夹爪开合指令 | `master_actions.left_gripper.openness`、`master_actions.right_gripper.openness`，0 关、1 开 |
| 下一帧实际状态形式的动作标签 | `actions.*`；与 `master_actions.*` 语义不同 |
| 杯、盘真值 | 各物体的 `translation`、`orientation`、`scale` |
| 三视角逐帧 RGB | LMDB 的 `images.rgb.{external,hand_left,hand_right}/步号`，JPEG 编码 |
| 三视角预览视频 | 对应相机目录的 `demo.mp4`，现 logger 写为 15 fps |
| 相机内外参 | `json_data` 中内参；`camera2env_pose.*` 中逐帧外参 |
| 任务与索引元数据 | `meta_info.pkl`：自然语言、详细描述、步数、字段索引、图像步号、采集版本等 |

当前未启用深度、分割、包围盒和运动向量输出；这些属于可选相机标注。当前也没有接触力/力矩训练标签。夹爪接触用于成功判定不等于已保存力标签。

当前原始输出足以离线生成 RGB 加相对 TCP 动作数据。转换时用实际末端、同帧关节控制目标的 FK、工具变换计算动作，不把 `actions.*` 自动视作原始控制器命令。采集归档还应保留任务/相机/机器人配置、URDF 版本及采样间隔。现任务仿真与图像记录为 30 Hz，不能从 15 fps 预览视频反推控制周期。没有必要为了每次导出格式或 TCP 定义变化重新运行仿真。

## J2/J3 限位来源

本地 `piper100.urdf` 的 Git blob 为 `e1ab7b064ee03f34855ecf94376e6e3e78db6c94`，与 `/home/agilex/qhx/.interndata/minimal-assets-manifest.json` 下载记录一致。下载脚本指向 `InternRobotics/InternData-A1` 的 `InternDataAssets/curobo/`。这说明当前 URDF 中 J2 下限 −0.1 rad、J3 上限 +0.1 rad 来自 InternData 资产包，并非本次新增修改。

资产包配置与厂家标准 ROS 描述不是同一来源。厂家 ROS 的 J2 下限、J3 上限为 0。[厂家 URDF](https://raw.githubusercontent.com/agilexrobotics/piper_ros/noetic/src/piper_description/urdf/piper_description.urdf)

没有找到解释 0.1 rad 余量意图的注释；可能用于数值容差或求解余量，但不能认定作者意图。当前 `position_limit_clip` 默认 0，没有发现额外抵消该余量的裁剪。此前仅凭限位不同就认定必须收紧，判断过早。现按原配置保留，若轨迹实际利用 USD 以外的范围，再针对实际偏差处理。

## 本次复现

打开现有 `INTERNDATA_DEBUG_PICK=1`，分别运行默认单条任务和额外两次布局。所有结果位于 `output/cup_plate_diagnostics/`，未混入正式数据目录。

默认单条任务成功，但打印 36 次候选规划失败，全部为 `MotionGenStatus.IK_FAIL`。候选是为寻找有效抓取而测试的多个目标，不是 36 条任务失败。现有通用打印把它们都写成 `Plan did not converge to a solution.`，应区分候选淘汰与执行中断。

对本次首个失败的预抓取目标（臂基座下约 `[0.4285, -0.0643, 0.1338] m`）做不含环境碰撞的 CPU 多初值逆解对照：

- 原限位下数值解贴住 J5 = −69.90085°，该次测试最好位置误差约 18.6 mm、姿态误差约 0.0889 rad。
- 仅在离线逆解计算中放宽 J5 时，求得满足目标的解，J5 约 −77.33°；其余关节在原范围内。
- 这没有修改仿真或实机限位，也不是全局不可达的数学证明，但支持腕部姿态与限位是该候选的重要瓶颈。其 J2/J3 远离争议的零度边界。

额外两次调试布局均未完成：

1. 抓取与抬升通过，盘上方的预放置目标能求解，但所有采样的最终放置目标报 IK_FAIL，最终出现 `No collision-free preplace/place pair`。该通用文字不能直接解读为已确认碰撞。
2. 预抓取通过，继续接近抓取目标时连续 6 次 `INVALID_START_STATE_JOINT_LIMITS`。日志中左臂约 `[0.507, 2.206, -1.186, 1.147, -1.220, -0.574] rad`，仅 J5 接近规划边界。由于现有打印仅保留三位小数，尚不能精确确认越界量；需高精度记录实际关节值与运行时边界。

额外 IK 调试调用可能改变求解器随机序列；上述测试用于定位失败类型，不能与之前 2/3 结果直接合并成严格成功率估计。已有少量成功轨迹不足以证明任务已稳定。

完整计数及数值对照见 `output/cup_plate_diagnostics/planning_audit_summary.json`。日志为 `/tmp/cup_plate_planning_audit.log` 和 `/tmp/cup_plate_planning_audit_batch.log`。

## 后续改进重点

1. 先做 IK 可达性筛选，并按关节边界余量、姿态变化和路径代价排序。现有抓取分数主要来自抓取高度，没有反映机械臂可达性，导致许多方向不合适的候选被完整规划。
2. 联合验证预抓取、抓取、抬升、预放置、落盘、撤离，避免仅检查前两点就接受一个对后续放置不合适的抓取。现有预抓/抓取测试都从当前起点独立调用规划，不等于整条动作链已经通过。
3. 让预接近距离、杯盘位置与工具姿态保留腕部余量。杯子在位置工作空间内，不保证其严格水平侧抓姿态和所有中间姿态都可达。
4. 起点限位失败时记录高精度关节值、边界及越界量，区分数值级偏差与真正越界，采用有明确容差的处理。当前没有证据需要通过修改 500 rad/s 上限或 J2/J3 的零度附近范围来解决这些失败。
5. 统计每布局结果和失败阶段，候选失败只在调试日志中汇总。分别评估任务成功率、候选通过率与耗时。

CuRobo 对笛卡尔目标先求逆运动学，再做轨迹优化及必要的图搜索；场景无额外遮挡不能保证 IK 成功。[CuRobo MotionGen](https://nvlabs.github.io/curobo/v0.7.6/_api/curobo.wrap.reacher.motion_gen.html)
