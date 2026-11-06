# 杯盘数据：对齐 RobotCooking 的 LeRobot v3 格式

当目标是与 `session_102914` 的实机关节动作格式一致时，使用 `scripts/datasets/export_cup_plate_robotcooking.py`。此前的 `export_cup_plate_lerobot.py` 导出的是另一种 7 维 TCP 增量格式，不能替代本脚本。

在独立的 LeRobot Python 环境中运行，例如：

```bash
python scripts/datasets/export_cup_plate_robotcooking.py \
  --source output/cup_plate_desk/v0.7_top_grasp/parallel \
  --output output/lerobot_v3/cup_plate_v07_top_grasp_robotcooking \
  --reference-session /home/agilex/RL/robotcooking/files/data/session_102914 \
  --repo-id local/cup_plate_v07_top_grasp_robotcooking
```

输出目录必须尚不存在；再次转换请指定新的目录。默认处理源目录递归发现的全部完整 LMDB episode；可用 `--max-episodes` 限制条数。

- `observation.state`、`action` 均为 float32 的 14 维向量：左臂 6 关节＋夹爪，接右臂 6 关节＋夹爪。
- 状态使用实际反馈。动作使用该 episode 内未来第 2 帧的实际反馈，删除缺少未来标签的最后 2 个观测步。它不是 master_actions 命令，也不是 TCP 或关节增量。
- 关节单位 rad；仿真夹爪实际总宽除以 0.10 m 后归一化。不会按本批观测最大值再缩放。
- 三路相机命名为 TOP_CAM、LEFT_WRIST_CAM、RIGHT_WRIST_CAM，默认和实机转换器一样，等比例缩放、居中补黑边至 256×256。
- 任务文本从 `--reference-session` 的 metadata.json 自动读取；多种任务文本时拒绝自动合并，需要明确任务映射。
- 直接流式编码 H.264、30 fps，避免写出大量临时 PNG。已在本机 LeRobot 0.6.2 验证接口。

为对齐实机辅助字段结构，离线专家类别设为 3、接管标志设为 0、policy_action 使用零占位；采集者设为 `simbox_scripted_v0.7_top_grasp`。类别 3 是派生训练标签，不能据此认为仿真源来自人类遥操作；`conversion_manifest.json` 明确记录其合成来源与赋值规则。

转换后可进行独立核对：

```bash
python scripts/datasets/validate_robotcooking_export.py \
  --dataset output/lerobot_v3/cup_plate_v07_top_grasp_robotcooking
```

验证器比较实机转换器的字段定义，逐行核对原始反馈、未来标签和 episode 边界，逐轨迹抽查图像与任务，完整解码所有视频，并检查归一化统计。实机转换器的位置可通过 `--reference-converter` 修改。

文件格式、单位和时间标签兼容不代表物理标定或图像采集同步已验证。仿真与实机具有不同的数据来源、关节停放姿态、相机外参及动力学。详细比较见导出目录中的 `meta/session_102914_compatibility.zh-CN.md`。
