# 杯盘任务 v0.4：抓取与运动验证

日期：2026-10-28。对应 `local_configs/cup_plate_desk/task.yaml`。本次保留原机器人关节限位、500 rad/s 运行时上限、质量、相机和地面设置。

## 已实施的修正

1. 原抓取标注在两指中点之外又添加 25 mm 插入深度，倾斜接近时可能让掌部先碰杯。现在插入深度为 0，再后退 5 mm；保留 1°、15°、25°、30° 从斜上方向下的杯壁抓取。根据 USD 掌部碰撞网格的保守包围盒，加 2 mm 余量检查端点及完整接近路径。两指仍允许接触杯壁。
2. SplitAloha 总开口 100 mm 对应主动单指行程 50 mm。修正原先把单指目标发到 100 mm、超出行程的指令。
3. 优先比较直线与浅弧线引导的 IK 路径；连续选择关节分支，平滑重采样并逐点检查原规划世界的碰撞和关节限位。按路径长度、多余抬高/回落、旋转量评分。无合适引导路径时允许关节空间规划回退；持杯后的额外高度最多 25 mm。初始就位允许必要的弧线。
4. 最后接近杯子时姿态偏离终点不得超过 0.75°；位姿到位且连续实测位置变化估计的关节速度小于 0.05 rad/s 连续 3 帧才闭爪。该仿真高增益驱动的瞬时速度反馈在位置已稳定时仍可能非零，不能用它直接等待静止。闭爪前杯子 XY 移动超过 3 mm 则失败退出。
5. 抬杯、盘上方预放置、松爪后撤离高度由 11/9/12 cm 统一缩短为 6 cm。取消本任务录制末尾的回家动作；保持撤离姿态验证杯子稳定，下一条 episode 重置机器人。保留实抓关系更新和物理接触抓取。

主要实现：`scripts/assets/make_cup_grasp_labels.py`、`workflows/simbox/core/utils/cup_plate_planning.py`、`workflows/simbox/core/skills/pick.py`、`workflows/simbox/core/controllers/splitaloha_controller.py`。本任务另行收紧批量 IK 的位姿误差阈值，其他任务仍使用原默认值。

## 验证结果

47 项单元回归测试通过。使用 seed 42 的连续三个随机布局，完整规划与物理任务均成功，保存 3 条、1364 帧；三个视角 RGB 和各状态/动作数组长度均核对完整。

| Episode 时间 | 帧数 | 闭爪前最大杯子 XY 位移 | 杯子最大抬升 | 闭爪前最后 15 帧实测姿态变化 | 最终杯盘中心偏差 | 最终倾斜 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 14:26:12 | 569 | 0.0052 mm | 67.68 mm | 0.00059° | 0.59 mm | 0.69° |
| 14:26:50 | 363 | 0.0036 mm | 67.86 mm | 0.00080° | 1.86 mm | 0.89° |
| 14:27:28 | 432 | 0.0045 mm | 67.77 mm | 0.00074° | 0.11 mm | 0.67° |

另一次同布局接触诊断成功完成：闭爪前 114 个开爪帧中，掌部接触杯子 **0 帧**。历史 v0.3 的探针记录为 14 帧掌部接触、约 14.8 mm 推杯。以上是小样本仿真验证，不代表任意布局均保证成功。

第一条新路径的抓取接近、抬升、运输、下降和撤离分别约 60/60/132/60/60 mm；这些段没有规划高度超越端点最高值。曲线仍是允许的候选形式，未强制所有轨迹必须直线，也未承诺全局最短。

实际杯子比名义 60 mm 多抬高约 8 mm，包含接触抓取后杯子相对夹爪的小幅变化与随后放置关系修正。最终杯底高度误差 2.60～2.70 mm，处于 4 mm 成功阈值内。

原始数据和指标在 `output/cup_plate_diagnostics/v04_validation/`：`run.log`、`raw_validation_summary.json`、`motion_summary.json`；接触证据在 `output/cup_plate_diagnostics/v04_contact/`。本次没有执行 LeRobot 格式转换。

后续正常运行：

```bash
bash scripts/run_cup_plate_desk.sh 3 --load_stage.scene_loader.args.simulator.headless=false
```

默认输出为 `output/cup_plate_desk/v0.4/`，旧版数据保留。`OUT_DIR` 可覆盖输出根目录。

## 相机对齐建议

相机型号可以不同。优先匹配图像缩放/裁剪之后的有效内参、视野、安装位姿与 RGB 处理流程。fx/fy 改变物体在图像中的尺度，cx/cy 改变投影位置；外参改变观察方向和遮挡。图像 resize 时内参需同步缩放。[OpenCV 投影模型](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)

对于当前 VLA 用法，建议先以实机 RGB 分辨率下的标定内参和安装位姿作为仿真中心设置，再围绕它们做适度随机化，并用实机数据验证迁移。相对 TCP 动作不会自动抵消图像域差异。训练和部署也需要对应的相机名称、图像处理与分辨率约定。[LeRobot 相机配置说明](https://huggingface.co/docs/lerobot/main/il_robots)

当前相机加载器使用 fx/fy 平均值设置焦距，cx/cy 仅记录在矩阵中，没有实际设置成像偏心；所以换成仓库 D435i YAML 仍只是近似参数模拟。当前任务没有切换相机，也没有宣称已经严格匹配实机。

## 仅修改地面纹理

编辑 `local_configs/cup_plate_desk/arena.yaml` 的 `floor.texture`。当前资产中 `texture_id` 的 0、1、2 分别对应 `floor_001.jpg`、`floor_002.jpg`、`floor_003.jpg`。`apply_randomization: true` 会在这些纹理之间随机选择，`texture_scale` 控制平铺。

也可将自己的 JPG/PNG 放入一个仓库内专用目录，将 `texture_lib` 指向该目录的绝对路径，`texture_id: 0` 选第一个文件。加载器按文件名排序选择。

这里只改变 5×5 m 地面平面材质，不改变 HDR 背景照片中的地板、墙面或环境照明。本次没有替用户选择或更换具体纹理。
