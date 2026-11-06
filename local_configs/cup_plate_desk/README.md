# Aloha：从杯口上方夹持同一侧杯壁的内外面，并直立放入盘子

从仓库根目录运行：

```bash
bash scripts/run_cup_plate_desk.sh 1
# 打开仿真窗口
bash scripts/run_cup_plate_desk.sh 1 --load_stage.scene_loader.args.simulator.headless=false
# 多个布局；数字是尝试次数，不保证全部成功
SEED=123 OUT_DIR=output/cup_plate_seed123/ bash scripts/run_cup_plate_desk.sh 10
```

`scripts/isaac_python.sh` 默认使用仓库旁 `.interndata/` 中的运行环境；可设置 `INTERNDATA_HOME` 与 `ISAAC_SIM_PATH`。
启动适配器 `scripts/launch_isaac.py` 沿用现有 Isaac 5.1 显式单位设置。资产仍通过 `workflows/simbox/assets` 软链接加载。

## 多实例并行采集

```bash
# 默认2个独立实例，每个尝试5个布局
bash scripts/run_cup_plate_parallel.sh
# 2个实例，每个尝试10个布局，使用种子42、43
INSTANCES=2 PER_INSTANCE=10 BASE_SEED=42 bash scripts/run_cup_plate_parallel.sh
# 只打印将启动的命令，不创建输出或启动Isaac
INSTANCES=2 PER_INSTANCE=10 bash scripts/run_cup_plate_parallel.sh --dry-run
# 指定新的输出目录，关闭显存采样
OUT_ROOT=output/cup_plate_batch01 WATCH_VRAM=0 bash scripts/run_cup_plate_parallel.sh
# strict模式：每实例目标60条成功数据，每条目标最多额外重试20次
INSTANCES=3 PER_INSTANCE=60 BASE_SEED=42 STRICT_MODE=true MAX_RETRIES=20 bash scripts/run_cup_plate_parallel.sh
```

每个实例调用当前 `run_cup_plate_desk.sh`，沿用本任务的光照、桌面、地砖和相机随机化；默认无窗口运行。额外配置参数会原样传给各实例。种子、输出目录和布局数分别用 `BASE_SEED`、`OUT_ROOT`、`PER_INSTANCE` 控制，避免配置覆盖让实例写到同一个目录。种子从 `BASE_SEED` 开始逐个加一，表示使用不同随机流，不保证所有物体位置都互不重复。

`STRICT_MODE=false` 时，`PER_INSTANCE` 为每实例尝试次数；`STRICT_MODE=true` 时为成功数据目标。`MAX_RETRIES` 是 strict 模式每条目标样本在首次失败后的额外重试上限，成功后重新计数；默认 `-1` 不限次数，`0` 表示首次失败立即停止。设置20时，每条目标最多执行首次尝试加20次重试；连续21次失败后抛出 `StrictRetryLimitError`，该实例以非零退出，已保存的数据保留，其他并行实例继续。每次重试均重新随机布局，不是整批数据共享20次重试预算。该参数不重试进程崩溃、CUDA OOM 或数据写入异常。非strict模式忽略重试预算。

单实例也可用 `STRICT_MODE=true MAX_RETRIES=20 bash scripts/run_cup_plate_desk.sh 60`。等效的CLI覆盖为 `--load_stage.layout_random_generator.args.strict_mode=true --load_stage.layout_random_generator.args.max_retries=20`，单实例和并行入口均支持。`summary.csv` 的 `requested_layouts` 是传入的布局数参数，strict模式下不是实际总尝试数。

默认输出在 `output/cup_plate_desk/v0.7/parallel/<时间_进程号>/`：`inst_1/`、`inst_2/` 是独立数据目录，各自的 `engine/` 保存引擎配置和性能日志，`inst_1.log` 等捕获进程输出，`instances.tsv` 记录进程号、种子及路径，`summary.csv` 汇总配置的布局数、任务成功数、实际保存的 episode 数和进程退出码。每个保存的 `meta_info.pkl` 计为一条本任务的单机器人 episode。已有实例数据或日志的目录会被拒绝，防止覆盖。离线 LeRobot 转换可把本次输出根目录作为 `--source` 递归读取。

默认相邻实例间隔3秒启动，可用 `LAUNCH_DELAY=0` 立即启动。`WATCH_VRAM=1` 时每5秒写 `gpu_watch.csv`；汇总基于实际检测到的GPU容量，包含其他进程占用，不据此推算增加实例数。不同实例共享GPU显存，实例数量需按当前可用显存设置。

一个实例异常不会主动终止其他实例；脚本等待全部实例后汇总，有进程异常则退出码为1，否则为0。尝试失败但进程正常结束不算进程异常，成功条数需看汇总。Ctrl+C / TERM 会终止本次启动的进程组（包括Python子进程），最多等待5秒后强制清理；不会停止其他运行任务。INT退出码130，TERM退出码143。

调度验证位于 `tests/unit/test_cup_parallel_runner.py`：用临时模拟工作进程检查并发启动、独立种子/路径和带空格参数、已有输出保护、单实例失败、INT/TERM清理及显存监视退出；同时用实际 ConfigProcessor 验证实例引擎日志目录覆盖。验证未启动真实多实例Isaac数据生产。

重试预算验证位于 `tests/unit/test_strict_retry_limit.py`：覆盖0/1/2/20次额外重试、成功和新任务清零、无限和非strict旧行为、CLI参数，以及非debug模式下规划组件仍传播上限异常。

2026-10-29 修复长批次中的 `Could not bound joint increments` 异常：该异常会在触发strict预算前中断进程。现在直接按插值导数上界保证采样步长，候选重采样失败会继续筛选，全部候选失败才计入正常重试。原有0.025 rad步长、关节限位和碰撞检查保持不变。原批次的配置和日志证据、复现与验证见 [重采样异常修复](../../docs/cup_plate_retry_resampling_fix.zh-CN.md)。

## 只查看场景和相机

```bash
# 默认打开外部 D435i 视角
bash scripts/preview_cup_plate_desk.sh
# 自由视角查看整个布置
bash scripts/preview_cup_plate_desk.sh --camera perspective
# 查看左腕相机；右腕用 hand_right
bash scripts/preview_cup_plate_desk.sh --camera hand_left
# 用另一随机种子查看杯盘初始布局
bash scripts/preview_cup_plate_desk.sh --seed 123
```

预览只加载任务并短暂运行物理让物体落稳，然后暂停。不会创建规划控制器、技能、数据记录器或输出数据集。窗口保持响应，可在 Viewport 相机菜单中切换 Perspective、外部相机和左右腕相机；观察场景时使用 Perspective，避免拖动任务相机。调整 YAML 后关闭并重新启动预览；界面中临时编辑的位姿不会自动写回 YAML。此入口始终保持暂停，点击 Play 也不会执行任务。关闭窗口即可退出。

无窗口检查可用 `bash scripts/preview_cup_plate_desk.sh --headless --frames 30`；此模式只渲染指定帧数并退出，不保存图像或任务数据。

## 文件职责

| 文件 | 用途 |
| --- | --- |
| `task.yaml` | 机器人、杯盘位置、相机、技能和成功标准 |
| `arena.yaml` | 静态桌子与地面 |
| `grasps/paper_cup_rim.npy` | 768 个从上方进入杯口、夹持同一侧杯壁内外面的候选，单位为米 |
| `grasps/paper_cup_side.npy` | 保留旧版 288 个外侧抓取候选，当前任务不使用 |
| `scripts/assets/make_cup_grasp_labels.py` | 可重建上述抓取候选 |
| `scripts/assets/make_object_wrapper.py` | 将源 USD 包装为 `/World/Aligned` 刚体结构 |
| `workflows/simbox/assets/custom/materials/floor_tiles/` | 本任务使用的五张地砖图片，配置使用相对资产根目录的路径 |
| `workflows/simbox/assets/custom/objects/wood_table/textures/` | 桌子模型专用的木纹、金属纹理 |

旧的 `local_configs/pick_cup_to_plate_desk.yaml` 和 `cup_plate_desk_arena.yaml` 是兼容软链接。因此原有 `.interndata/run-cup-plate-desk.sh` 也会加载修正后的任务。后续维护以上新路径。

材质目录约定：公共自定义材质放入资产库 `custom/materials/<材质用途>/`，模型专用纹理保留在 `custom/objects/<模型>/textures/`，任务目录只维护配置。五张地砖已统一整理为 `black_marble.png`、`tile_02.jpg`、`tile_03.jpg`、`tile_04.jpg`、`tile_05.png`，分别对应原 `tile.png`、`地砖2.jpg`、`地砖3.jpg`、`地砖4.jpg`、`地砖5.png`。资产根目录不再散放这四张图片，任务目录原 `textures/black_marble/` 已移除。未向旧的 `floor_textures/` 添加文件，以保持旧任务的目录枚举和纹理序号不变；木纹仍在桌子模型目录内，以保留 USD 中的相对纹理引用。`workflows/simbox/assets` 是指向外部资产库的软链接，迁移或备份工程时需同时保留该资产库。

## 物理与抓取约定

当前执行顺序为抓取、放置并撤退、左臂回到本条任务的起始关节位置、最终放置复核。回位路径在更新后的碰撞世界中检查，盘中的杯子作为障碍物保留；回位动作也记录到原始数据。`return_home` 的 `joint_tolerance_deg: 0.5` 和 `stable_steps: 3` 控制实际回位判定，规划预算默认 20 秒。当前杯子落盘上限已放宽到 8°，详见 [8° 放置与回位验证](../../docs/cup_plate_eight_deg_return_home.zh-CN.md)。

最新两阶段接近、协同抬升、小倾角落盘配置及物理验证记录见 [姿态与平移协同优化](../../docs/cup_plate_flexible_chain.zh-CN.md)。搬运/放置端点、双臂附加物体关节映射与缓存刷屏修复见 [搬运放置故障修复](../../docs/cup_plate_transfer_failure_20261030.zh-CN.md)。

批量耗时修复见 [重复 IK 与规划预算](../../docs/cup_plate_bounded_search.zh-CN.md)：初始整段规划默认 45 秒、实抓后放置重规划 20 秒、起点漂移单段刷新 10 秒，每 5 秒打印进度。时间预算在各自阶段内累计，超时记为正常布局失败，不强杀进程；启动、物理执行和保存时间另计。Pick 段的 `chain_planning_timeout_s`、`chain_place_timeout_s`、`chain_refresh_timeout_s`、`chain_progress_interval_s` 可调整这些值，改后需要重启实例。strict 的重试预算独立计算。

v0.7 仍加载 `paper_cup_001/Aligned_obj.usd` 的刚体杯，不加载独立柔性杯实验。左夹爪初始闭合（`left_gripper_home: [0.0]`），预抓和松爪总开度为 50 mm；回位途中保持 50 mm，最终稳定复核阶段恢复闭合。预抓取与松爪总开度为 50 mm，规划器锁定指关节同步到相应开度并保留 Piper 两指关节符号；闭爪仍由物理接触形成夹持。

顶抓的固定成功布局、候选可达区域和验证限制见 [顶抓工作空间记录](../../docs/cup_plate_rim_workspace.zh-CN.md)。当前夹爪允许偏离竖直 0°/±2°/±5°/±10°/+15°，随机位置范围由用户在 `regions` 中自行设置。

- 桌子尺寸 1.4 × 0.6 × 0.75 m，杯子直径 75 mm、高 80 mm，盘子直径 154 mm。
- 源模型网格以毫米计，YAML 缩放为 `0.001`。抓取标注已经是米，不再重复缩放。
- `robots.translation` 是整机在任务坐标系中的位姿，整机 Z=0。当前通过共用升降平台将双臂相对旧实测高度上调 80±20 mm，两侧共同采样；详见下方高度配置。桌高保持 0.75 m。
- 当前 v0.7 使用原刚体纸杯，从杯口上方竖直下降，一指进入杯内、另一指留在杯外，沿径向夹持同一侧杯壁。TCP 中心沿同侧杯壁径向布置，局部 Z 为 65/68/71/74 mm，杯周方向间隔 15°，夹爪偏离竖直 0°/±2°/±5°/±10°/+15°。768 个标注经夹爪对称变换得到最多 1536 个工具姿态。当前已移除 77/80 mm 的近杯口候选；TCP 高度不是实际指尖接触高度。上方接近可调整姿态，只有最后 12 mm 入杯段固定姿态并竖直下降，以满足 J5 原有行程。旧版 1°/15°/25°/30° 外侧抓取候选仍保留，但不被当前任务读取。
- 抓取成功需要双指接触并抬升至少 2 cm；杯盘初始区域继续按当前配置采样。
- v0.4 抓取标注插入深度为 0，另沿接近方向后退 5 mm，避免原来多伸入 25 mm。使用实测掌部包围盒加 2 mm 余量筛选抓取点及完整接近轨迹；允许两指接触杯壁。闭爪前杯子水平移动超过 3 mm 的片段失败退出，不进入成功数据。
- 规划器附加杯子的全部网格，排除 `Looks` 等非几何节点；持杯时刷新规划世界不会重新把已附加网格当作静态障碍物。附加只影响规划模型，实际抓取仍依赖 PhysX 夹爪接触，没有为杯子添加固定关节。
- 本任务启用 `joint_chain_planning`：先批量 IK，以关节余量和关节变化量排序；优选 5° 余量，候选端点至少 1°，轨迹中间点至少 0.5°。这是候选筛选，不修改机器人限位。上方预抓距离搜索 4/5.5/7 cm，绕 TCP 保持位置并相对抓取姿态搜索 0°/±5°/±10° 调整；最后 12 mm 稳定入杯段的 TCP 横向偏移不超过 2 mm，不接受上下往复的插入路径。
- 抬升搜索上移 50 mm（`chain_lift_height: 0.05`，比原来增加 10 mm），同时向臂基座侧平移 0/20/40 mm 并使杯子绕闭合轴转动 0°/±5°/±10°；抬升段杯轴偏离竖直不超过 10°，搬运到预放置点途中允许最多 60°，最后落盘整段最多 8°，各段保持碰撞检查。预放置前利用引导曲线的姿态插值逐渐回正；落盘姿态优先直立，也搜索 3°/4°/6°/7° 倾斜，并按杯底半径补偿高度，杯子最终倾角上限为 8°，盘子倾角上限仍为 5°。
- 接受抓取前依次验证预抓、抓取、抬升、预放置、落盘和撤离，每段从上段实际规划终点开始。持杯段附加预测的杯子碰撞几何，撤离段暂时移除并恢复携带几何。仅本任务补齐杯子与两指的合法接触排除，不关闭环境碰撞检查。
- 完整轨迹通过后按关节目标执行缓存路径。实际段起点偏差超过 3° 时重新规划；实抓后若抓取关系变化超过 3 mm/2°，或盘子位姿改变，则依据实际抓取关系重算放置段。没有改变杯子实际位姿，也没有添加物理焊接。
- v0.4 优先比较直线与浅弧线引导的连续 IK 轨迹，按路径长度、多余抬高/回落和旋转量评分；每个重采样点重新检查 CuRobo 碰撞与关节限位。引导曲线无解时可回退到关节空间规划，但持杯后的多余高度受限，接近杯子的姿态摆动不超过 0.75°。这些是本任务的轨迹筛选参数，不修改机器人 500 rad/s 上限。
- 接近完成后，位姿达标且由连续实测关节位置差估计的速度低于 0.05 rad/s 连续 3 帧才闭爪。避免高增益驱动下瞬时求解器速度读数造成误判。夹爪全开修正为主动单指行程 0.05 m，对应两指总开口 0.10 m。
- 盘子中央承托面相对盘子原点高 5 mm，盘沿最高 16 mm。`support_surface_z` 指定中央承托面，`place_z_offset=0.003` 让杯底从面上 3 mm 处释放。
- 放置优先杯子直立，也搜索相对盘子倾斜 3°/4°/6°/7°，绕竖直轴以 15° 步长搜索一周，并结合关节余量、旋转幅度及倾角评分。夹爪姿态由实际抓取关系换算，不强制水平。
- `chain_flexible_placement=true` 时，搬运衔接到最后 12 mm 的固定小倾角落盘段，`pre_place_z_offset` 不再作为本模式的预放置高度。松爪后的空夹爪按 0°/±5°/±10°/±20° 搜索独立撤退姿态；两指总开度保持 50 mm，最终杯子稳定性仍按原判据检查。
- 当前顶抓的抬杯目标为 5 cm，松爪后空夹爪上撤 4 cm；路径允许浅弧线，最后 12 mm 落盘段保持小倾角。放置并撤离后回到初始关节位置，再进行最终稳定性复核。
- 三个相机名均带 `split_aloha_` 前缀，以关联引擎的数据记录分组；外部相机的 `parent` 仍为空，保持世界坐标固定。输出视角名为 `hand_left`、`hand_right`、`external`。
- 成功要求：杯子整个保守圆形投影在半径 48 mm 的承托区内、杯底高度误差不超过 4 mm、杯子相对盘子倾斜不超过 8°、盘子倾斜不超过 5°，杯盘速度不超过 0.015 m/s 与 0.15 rad/s。杯子投影半径采用 37.5 mm，因此判定偏保守。

更换杯盘资产时必须重新核实尺寸、原点、承托面高度、投影半径和抓取候选，不能直接沿用这些数值。

重新生成当前顶向内外壁夹持候选（旧侧抓可添加 `--mode side`）：

当前角度为预生成的离散值。也可以在每个布局初始化时，从连续角度区间随机采样有限个候选，并使用统一随机种子和元信息记录；当前尚未启用这种动态采样。仅修改 `downward_tilt_range_deg` 只会筛选已有标注，不会自动生成中间角度。倾斜沿所选杯周位置的切线正反方向：0°、±2°、±5°、±10°，保留单向 +15°。

```bash
bash scripts/isaac_python.sh scripts/assets/make_cup_grasp_labels.py --mode rim
# 显式指定高度（米）和偏离竖直的倾角（度），与上述默认值一致
bash scripts/isaac_python.sh scripts/assets/make_cup_grasp_labels.py --mode rim \
  --grasp-heights 0.065 0.068 0.071 0.074 --rim-tilts 0 2 5 10 15 -2 -5 -10 --rim-opening 0.050
```

运行针对本任务的回归测试：

```bash
bash scripts/isaac_python.sh tests/unit/test_cup_plate_task.py
```

当前 v0.7 原始输出位于 `output/cup_plate_desk/v0.7/`，外部图像为 D435i 640×480；保留旧版数据。核心回归测试位于 `tests/unit/test_cup_plate_task.py` 和 `tests/unit/test_cup_chain_planning.py`。

v0.4 的推杯修正、运动优化及 3 个随机布局验证结果见 [运动验证](../../docs/cup_plate_motion_validation.zh-CN.md)。

历史 v0.2 的水平抓取验证为 457 帧、3 次尝试成功 2 次，见 `docs/cup_plate_desk_validation.zh-CN.md`；这些数值不代表 v0.3 的结果。倾斜抓取与整链规划验证见 `docs/cup_plate_chain_validation.zh-CN.md`。

## LeRobot v3 训练数据

采用左臂相对当前 TCP 的 7 维动作（局部平移、旋转向量、绝对夹爪开合），配置在本目录的 `lerobot_v3.json`。导出脚本为 `scripts/datasets/export_cup_plate_lerobot.py`，需要独立的 LeRobot v3 环境。完整字段、坐标定义、导出命令和实机对接约定见 [数据说明](../../docs/cup_plate_lerobot_v3.zh-CN.md)。当前 TCP 是仿真定义，尚未实机标定。

## 相机与地面外观

当前桌面固定 750 mm，`arena.yaml` 的 `table.height_randomization.enabled: false`，不再随机桌高。此前的桌高随机化功能保留供旧场景重放使用，下述桌高验证是历史记录。

双臂高度在 `task.yaml` 的 `robots.base_height_randomization` 中设置：`nominal_offset: 0.08` 表示相对旧配置的实测稳定高度上调 80 mm；`offset_jitter_range: [-0.02, 0.02]` 表示每次布局（含失败重试）在标称值上独立均匀采样 ±20 mm，最终相对参考高度增加 60～100 mm。双臂共用同一次采样，轨迹内不逐帧采样，也不因重复初始化累加。`enabled: false` 固定在上调 80 mm 后的高度；改后重启实例生效。机器人根配置、桌子、地面和立柱相机基准位置保持原值，腕相机跟随各自手臂。物体 XY 范围不变。

旧升降关节目标 -0.600 m 的实测稳定位置约为 -0.5862 m，对应 `arm_base` 距地面约 666.5 mm，因此 `initial_joint_positions.lifting_joint: -0.5862` 用作参考值；本条实际目标为参考值 +0.08 m + 扰动，即 -0.5262～-0.4862 m，标称 -0.5062 m。该参考值不能误当成最终高度。原驱动无阻尼，直接调整升降目标会振荡；当前每次机器人初始化后向物理接口写入本条位置目标，以及仅作用于升降关节的 `drive_stiffness: 1000000`、`drive_damping: 10000`，双臂六关节、夹爪、质量和源 USD 均不修改。控制器重置后按新的实际单臂基座提取规划碰撞世界。原始 `episode_randomization.robot_base_height.split_aloha` 记录参考关节位置、标称偏移、本次扰动及本条目标位置；保存的场景配置包含 `mem_height`，重放时恢复该值。新增随机采样会改变相同 seed 的后续随机序列。

本任务没有移动底座或升降动作：每条轨迹将升降 DOF 运行时限位收紧到采样目标 ±0.1 mm，并将 `fixed_base_joint_positions` 中的底座平移和转向保持在参考位置（各自限位 ±0.1 mm / ±0.0001 rad），防止碰撞受力后额外横移。参考值为局部 X=-0.03685 m、局部 Y=0、转向=0，对应此前左臂基座实际 XY 约 [-0.306,-0.33687] m；局部 X 直接归零会使手臂相对旧场景前移约 36.85 mm，因此不能用零代替。限位在下一布局按新采样值更新；重放旧的无高度随机化配置时恢复原驱动和限位。双臂 J1–J6 的限位不改。

2026-11-05 贴桌穿插修正：上段 XY 是旧机器人根位置下的参考。当前将 `robots.translation.y` 从 -0.66850 改为 -0.68813 m，整机向远离桌子的方向后移 19.63 mm，左臂基座 Y 相应约为 -0.35650 m。升降支架凸包在桌板高度范围内原有约 18 mm 穿插；后移后，在双臂升高 60/80/100 mm 时都实测得到约 2.00 mm 净间距。桌子、杯盘区域、外部相机和双臂高度随机化参数保持原配置。详情见 [贴桌与失败诊断](../../docs/cup_plate_table_gap_20261105.zh-CN.md)。

2026-11-04 双臂高度验证：90 项相关测试通过；物理场景依次验证参考高度和上调 60/80/100 mm，基座实测 Z 约 666.50/726.40/746.40/766.40 mm，随机高度重放一致，桌面和外部相机固定。结果与日志在 `output/cup_plate_diagnostics/robot_base_height_randomization/`。使用此前成功的固定杯盘 XY 补跑一条完整抓放，本次随机抬升 85.58 mm，整链规划通过，但接近时杯子在闭爪前移动 10.54 mm，超过现有 3 mm 保护阈值，执行失败且未保存成功 episode。日志在 `output/cup_plate_diagnostics/raised_robot_chain_final_20261104/validation.log`。不能据此声称新高度范围已通过完整抓放验证；需要进一步调整入杯候选或接近路径，本次保留原抓取配置和保护阈值。

桌高变化时，在物理世界重置前调整桌子，并在停止物理时摆放杯盘，再重建物理世界，避免碰撞面滞后或升高桌面与旧物体初态重叠。MotionGen 和独立 IK 求解器的碰撞网格缓存同步刷新，避免规划继续使用上一条轨迹的桌高。实际高度和 Z 缩放记录在原始 `episode_randomization.fixture_geometry.table` 中，已保存的场景配置通过 `mem_height` 恢复相同桌高。新增高度采样会改变相同 seed 后续的随机序列；原有数据不会被修改。

2026-11-04 验证：83 项相关测试通过。使用正式场景重置函数依次切换 730、770、750 mm，每档运行 90 个物理步，杯盘均稳定支撑于新桌面；验证结果在 `output/cup_plate_diagnostics/table_height_randomization/physics_validation.json`。本次未执行完整抓放规划，不据此判断新增高度范围的抓放成功率。

### 每条轨迹的光照、桌面、地砖、相机与抓取随机化（v0.6）

`task.yaml` 的 `env_map.photometric_randomization` 控制光照。`nominal_intensity: 1000` 对应此前 `apply_randomization: false` 时实际使用的强度；原来的 `intensity_range: [4000, 7000]` 当时未生效。`intensity_scale_range: [0.8, 1.2]` 将强度设为 800–1200，并非实测照度 lux 的约束。白光 5500–6500 K 与暖光 3000–4000 K 两档等概率选择，各档内均匀采样。方向相对于 `base_rotation_deg` 扰动，XYZ 欧拉角的幅度分别为 ±5°、±5°、±20°；这是各轴范围，不是总旋转角的上限。`enabled: false` 恢复原固定光照。

本任务继续固定 HDR 文件（外层 `env_map.apply_randomization: false`），只随机其强度、色温与小角度朝向。旋转 DomeLight 会同时旋转 HDR 背景图案；如果要求背景完全静止，应使用独立的主灯方向随机化。每次 episode 初始化采样一次，整条轨迹保持固定，避免帧间闪烁。改变随机种子可预览另一组布置、光照与材质；新增随机采样会改变相同 seed 后续的物体位置随机序列。

桌面在 `arena.yaml` 的 `table.appearance_randomization` 中配置。`texture_options` 按 `weight` 归一化采样；有 `file` 的条目选择木纹，再从 `color_options` 等概率选择标准/浅色。当前 `wood_1` 权重 0.4，标准/浅色各约 20%；用户提供的 `wood_2` 权重保留为 0.3，标准/浅色各约 15%，合计约 30%。新增 `solid_wood` 和 `solid_white` 各占 15%，不再细分标准/浅色，共计六种外观。这是每次场景采样的概率，有限次运行和仅保留成功轨迹后的实际比例可能偏离该值。可调整 `weight` 改变占比，权重必须为非负有限数且总和大于零。图片路径可用绝对路径或相对 `asset_root` 的路径，`enabled: false` 可移除某个条目；启用图片不存在时会报出具体缺失项，不静默回退。

纯色条目通过 `color` 指定线性 RGB：无纹理木色默认 `[0.55,0.32,0.16]`，白色 `[1,1,1]`。纯色直接断开 `Looks/LightOak/Surface.inputs:diffuseColor` 的木纹贴图连接并设置固定颜色，保留粗糙度、光照、阴影、几何、碰撞和金属桌腿材质；切回木纹或重放旧场景时恢复贴图连接。每条轨迹采样一次，轨迹内固定，元信息仍保存在 `episode_randomization.fixtures.table`；纯色记录 `surface_type: solid`、`color`、`texture_file: null`。更改配置后重启实例生效。

2026-11-05 纯色桌面验证：14 项外观测试通过，包括六种外观的采样比例、纯色断开贴图、恢复旧木纹连接、重放以及几何和桌腿材质保持。OOD 材质替换和归档重放两项检查通过；另一个既有位置范围检查失败，因为当前 OOD 位置范围与 ID 范围有重叠，此次未改动杯盘范围。

标准色的纹理 `scale=[1,1,1,1]`、`bias=[0,0,0,0]` 保留原色；当前浅色档为用户调整后的 `C_out = 0.75*C_texture + 0.25`，保留纹理并向白色混合 25%，处理发生在材质采样阶段。可减小 bias、相应增大 scale 以减弱变浅幅度。这只覆盖 `Looks/LightOak/WoodTexture` 的纹理输入，沿用原 UV、粗糙度、物理材质与几何，金属桌腿不变。`apply_randomization: false` 固定选择列表中第一个启用的木纹和第一档颜色，默认就是原木纹标准色。

地砖在 `arena.yaml` 的 `floor.texture.texture_options` 中配置，全部位于资产库 `custom/materials/floor_tiles/`：原黑色大理石 `black_marble.png` 权重 0.4；用户提供的另外四张图片重命名为 `tile_02.jpg`、`tile_03.jpg`、`tile_04.jpg`、`tile_05.png`，各权重 0.15。每条轨迹选择一种纹理铺满整个地面，原图占约 40%，另外四图合计约 60%；不会逐砖或逐帧更换图片。铺砖尺寸、砖缝、粗糙度和几何均沿用现有配置。`apply_randomization: false` 固定选择列表第一项，即原黑色大理石。这里使用显式文件列表，不混用目录内的其他图片。

外部相机在 `task.yaml` 的 `split_aloha_external` 中启用 `apply_randomization: true`。`max_translation_noise: [0.01,0.01,0.01]` 是任务坐标系 XYZ 各轴独立均匀扰动 ±10 mm，`max_orientation_noise: 2.0` 是随机旋转轴上的总旋转角不超过 2°（不是每个欧拉角都取 ±2°）。当前标称位置 `[0,-0.48,1.44]` 对应 X 范围 [-0.01,0.01]、Y 范围 [-0.49,-0.47]、Z 范围 [1.43,1.45] m，光心距桌面在 680–700 mm。扰动以标称位置和 `look_at` 计算出的标称朝向为基准，平移与旋转独立采样，不随 episode 重置累积偏移；平移后不会强制重新对准桌面中心。左右腕相机保持固定，相机内参、分辨率和帧率不变。新入口同时兼容其他任务原来的标量平移扰动（球形范围）。

杯盘位置当前使用 `task.yaml` 的 `shared_region_sampling` 联合采样：`xy_bounds: [[-0.35,-0.15],[0.00,0.10]]` 定义相对桌面中心的同一块 350×250 mm 区域，两个物体完整轮廓都须在其中。半径采用杯口 37.5 mm、盘外缘 77 mm，`clearance: 0.01` 要求两者边缘至少相隔 10 mm，因此中心距离至少 124.5 mm。先按各自半径收缩可采样的中心范围，再独立均匀生成整对位置；碰到重叠就重新采样整对，不通过固定左右顺序或物理挤开排重。杯子可位于盘子左右、前后各侧；矩形几何和物体尺寸使相对方位角并非均匀分布。

之后只改 `shared_region_sampling.xy_bounds` 即可放大或缩小共同区域；范围必须在桌面内且能容纳完整物体。当前杯中心范围为 X=[-0.3125,-0.0375]、Y=[-0.1125,0.0625] m，盘中心范围为 X=[-0.273,-0.077]、Y=[-0.073,0.023] m，随后还须满足相互间距。`max_attempts: 10000` 是每个布局的几何拒绝采样上限，与 strict 的规划重试次数无关。区域太小或太拥挤时明确报错，不放入重叠物体。启用共享区域时 `regions.pos_range` 不控制这两个物体的 XY；`regions` 的支撑面和 `yaw_rotation: [-180,180]` 仍生效。关闭 `shared_region_sampling.enabled` 可恢复原各自范围。场景重放恢复 `mem_shared_layout` 中保存的位姿，实际布局和半径、间距随 `episode_randomization.layout` 保存。

共享区域采样合法并不代表整段抓放可达；扩大区域需重新检查 IK、碰撞、抓取和放置成功率。只保存成功轨迹时，实际数据中左右/前后比例仍会受到左臂可达性影响。物体的 `apply_randomization: false` 只表示不随机替换模型，不会禁用位置或 yaw 采样。当前抓取标注覆盖杯周 24 个方向（间隔 15°）、杯底以上 65/68/71/74 mm 四个 TCP 高度，只有最后 12 mm 入杯段固定姿态竖直向下，夹爪抓取倾角偏离竖直 0°/±2°/±5°/±10°/+15°。当前 `chain_grasp_selection: best` 仍由规划器优选，不能称为对这些姿态均匀随机。

2026-11-05 共享区域验证：100 项相关检查通过（共享采样 3、外观及布局元数据 14、高度 14、杯盘/并行 64、策略评测配置 4、OOD 配置隔离 1）。几何测试检查 3000 对样本的完整轮廓边界、间距及四象限覆盖；八组真实物理摆放和一次保存布局重放也通过，结果在 `output/cup_plate_diagnostics/shared_region_20261105/validation.json`。此次未跑完整抓放成功率统计。策略评测关闭位置随机化时禁用共享采样，使用指定固定位置；OOD 位置配置使用其独立区域，防止被专家共享区域覆盖。

需要增加抓取姿态多样性时，将 pick 配置改为 `chain_grasp_selection: random_top_k`，用 `chain_grasp_top_k: 8` 控制随机池大小。它在抓取和预抓取 IK 均通过、按评分排序的前 K 对候选中打乱尝试顺序，随后回退到其余评分顺序；仍受 `chain_max_candidates` 总尝试预算限制。每对包含一个抓取姿态和一个预抓距离，因此并非对抓取角度/高度均匀采样。最终接受的候选仍须通过掌部间距、关节余量、完整抓放轨迹与碰撞检查，不给已验证轨迹叠加执行噪声。当前插入平移固定竖直向下，改变种子不能保证每次抓取方位或高度不同。

原始 episode 的 `lmdb/info.json` 中新增 `episode_randomization`，记录本条轨迹的光照参数、木纹/颜色档、地砖图片、外部相机实际位姿和接受的物体相对夹爪位姿；同样随 `json_data` 存入 LMDB。外部相机状态使用配置指定的 USD 轴、wxyz 四元数，图像对应的实际 `camera2env_pose` 仍按原字段记录。地砖和相机扰动在每条轨迹初始化时采样并保持固定，从已保存配置重放时恢复所采样值。数据仍由离线脚本转换成 LeRobot，不自动导出。

2026-10-28 随机化验证：58 项回归测试通过，包含强度/色温/角度范围、种子复现、纹理档组合、标准色复原、桌腿材质与几何不变、候选随机顺序和保存元数据。GPU 预览完成白光/暖光与标准/浅色四组参数切换和渲染检查；等待更长着色器更新时间的复核在第四组因显存不足退出。已检查的截图为 `docs/images/cup_plate_white_standard.png`、`cup_plate_white_light.png`、`cup_plate_warm_standard.png`。完整抓放回归在规划器初始化阶段遇到 CUDA OOM，未进入抓取规划，不能据此报告新的抓放成功率；当时其他两个进程合计占用约 19 GB 显存。运行日志为 `/tmp/cup_appearance_render_validation.log` 和 `/tmp/cup_appearance_chain_validation.log`。

随后加入地砖和外部相机扰动的验证：66 项回归测试通过，包括五种地砖图片完整性与采样比例、铺砖/砖缝材质保持、重放恢复、相机 XYZ 和旋转总角度范围、look_at 基准、不累积偏移及元信息保存。GPU 预览依次切换五张图片，并检查四次相机采样和已采样位姿恢复、内参不变，正常退出（未执行抓放）。日志在 `output/cup_plate_diagnostics/floor_camera_validation/`，截图为 `docs/images/cup_plate_floor_tile2.png` 和 `cup_plate_floor_tile5.png`。

相机品牌可以不同。用于实机迁移时，应尽量对齐预处理之后的有效内参、分辨率、相机位置和朝向、RGB 处理；相对 TCP 动作不会自动消除视角差异。

v0.5 外部相机位于立柱中央前侧，`translation: [0, -0.43, 1.44]`，单位米。桌面高 0.75 m，光心高 1.44 m，因此距桌面 690 mm；XY 根据模型估计，尚非实机外参标定。`look_at: [0, 0, 0.75]` 指向桌面中心，俯角约为相对水平面向下 58.07°。修改 `translation` 可移动相机，修改 `look_at` 可调整瞄准位置；启用 `look_at` 后自动计算 USD 相机的 wxyz 四元数，覆盖 `orientation`。二者均使用当前父坐标系；本外部相机 `parent: ''`，即任务坐标系。

相机参数文件是 `cameras/realsense_d435i_109122071725_rgb.yaml`，直接从实际连接的序列号 `109122071725` 读取 RGB8 640×480、30 Hz 出厂内参：`[607.267578125, 607.4727172851562, 315.7395935058594, 239.52183532714844]`。采用此分辨率的依据是本机实机采集项目对同序列号的配置。原始 SDK 读数保存在同目录的 `*_source.json`。畸变模型为 inverse Brown Conrady，但本模式五个系数全为 0，等价于无畸变针孔。

本相机启用 `use_calibrated_intrinsics`，通过 Isaac 5.1 原生 OpenCV pinhole 模型实际设置 fx、fy、cx、cy 并记录相同 K；旧相机默认路径仍保留原设置。非零 inverse/modified Brown 畸变系数不能直接套用 OpenCV 正向畸变公式，加载器会拒绝这种配置。像元大小、光圈与对焦距离是外观近似值，尚非实机测量值。这里只模拟 RGB 成像，没有额外启用深度噪声或 IMU。

重新读取实机内参（使用安装了 `pyrealsense2` 的 Python，无需启动视频流）：

```bash
python3 scripts/cameras/read_realsense_intrinsics.py --serial 109122071725 --width 640 --height 480 --fps 30
# 也可以查看 librealsense 的全部标定信息
rs-enumerate-devices -c
```

更换分辨率时必须读取该模式的内参并同步修改相机 YAML，不能把 640×480 的参数直接用于 1280×720。原来的 D455 配置为 1280×720，因此 v0.5 使用新的输出目录，后续转换 LeRobot 时按版本分别导出。预览见 `docs/images/cup_plate_d435i_mast.png`。

2026-10-28 验证：49 项单元测试通过；实际渲染回读确认四个内参与 30 Hz 配置一致。随机种子 42 的一次完整抓杯放盘成功，保存 466 帧，三路相机图像编号与状态帧数一致；最终杯底距承托面约 2.60 mm、杯身倾斜约 0.59°。该单次结果用于确认新相机配置可运行，不代表多布局成功率。验证日志和原始数据检查摘要保存在 `output/cup_plate_desk/v0.5/`。

当前地面从上述五种图片中随机选择，原黑色大理石图保存在资产库 `custom/materials/floor_tiles/black_marble.png`。`arena.yaml` 中 `tile_size: [0.6, 0.6]` 表示每块砖 60×60 cm，`grout_width: 0.002` 为 2 mm 砖缝；`grout_color` 设置缝色。每块砖使用独立 UV，`texture_scale: [1, 1]` 对应每砖完整一张图，边缘切砖保持纹理尺寸。`texture.roughness: 0.25` 控制表面光泽，数值越大越粗糙。可视地面扩大到 20×20 m，以覆盖当前外部相机视野；砖块和砖缝仅为可视网格，未添加物理碰撞，`/floor/` 从规划世界中排除以延续旧 Plane 的处理。原黑色大理石截图见 `docs/images/cup_plate_marble_floor.png`。

更换图片时，修改 `texture_options` 对应的 `file` 和 `weight`，路径可以是绝对路径或相对资产根目录的路径。旧任务不配置 `texture_options` 时，仍兼容 `texture_lib` 目录和 `texture_id` 的方式。删除 `tile_size` 等铺砖参数可恢复普通地面。此操作只改变场景地面网格的材质，不改变 HDR 背景照片中的地板、墙面和照明。
