# 纸杯放到盘子任务检查报告

> 本文保留修改前的审查记录。后续已按“杯壁外侧水平侧抓”修正代码并整理目录，当前配置、运行方法和成功标准见 `local_configs/cup_plate_desk/README.md`。下面的旧路径可通过兼容软链接访问。

检查日期：2026-10-28。依据本机当前代码、YAML、USD 和 NPY 文件；未修改现有代码、配置或资产，未启动 Isaac Sim 完整轨迹生成。

## 结论

任务组织方式总体符合 InternDataEngine 的设计，但当前有一个明确的抓取阻断问题，以及放置姿态、成功判定上的缺口，暂不能认定已实现可靠的“拿起纸杯并直立放到盘子里”。

用户给出的路径省略了后缀：实际入口为 `/home/agilex/qhx/.interndata/run-cup-plate-desk.sh`，不是一个任务目录。

## 实际加载链

```text
.interndata/run-cup-plate-desk.sh
  → .interndata/python.sh（Isaac Sim 与 Python 依赖环境）
  → .interndata/launch.py（禁用自动单位修正，使用 YAML 显式缩放）
  → InternDataEngine/launcher.py
  → configs/simbox/de_plan_with_render_template.yaml（流水线）
  → local_configs/pick_cup_to_plate_desk.yaml（任务）
      → local_configs/cup_plate_desk_arena.yaml（桌子、地面）
      → core/configs/robots/split_aloha.yaml（机器人及 CuRobo 参数）
      → workflows/simbox/assets/custom/...（资产）
      → 左臂 pick → place → heuristic__skill/home
  → output/cup_plate_desk/（成功生成的数据）
```

`BananaBaseTask` 是通用任务实现，名字不限制操作对象，当前不需要为纸杯另写 Task 类。机器人基础 YAML 先加载，再由任务中的同名字段覆盖。

## 发现的问题

### P1：抓取附加的路径指向材质 Scope，会导致 CuRobo 找不到物体

位置：`workflows/simbox/core/objects/rigid_object.py:49`、`core/skills/pick.py:195`、`core/controllers/template_controller.py:591`。

`RigidObject` 直接把 `Aligned.GetChildren()[0]` 记录为 `mesh_prim_path`。实际读取当前 USD 得到：

```text
paper_cup /World/Aligned
  Looks            ← 第一个子节点，类型 Scope，不是 Mesh
  OuterWall
  InnerWall
  TopAnnulus
  BottomOutside
  BottomInside
  RolledRim

ceramic_plate /World/Aligned
  Looks            ← 同样不是 Mesh
  PlateBody
```

抓取闭爪后执行 `attach_obj`，因此传给规划器的是任务中纸杯的 `.../Aligned/Looks`。CuRobo 的 USD 提取器只生成几何障碍物，不会为 Scope 创建障碍物；`attach_objects_to_robot()` 按名称查找失败后会进入 `Object not found in world` 错误分支。这条失败路径由资产结构与调用代码确认，尚未用完整仿真复现。

修复应递归收集实际几何网格，支持一次附加全部组成网格，同时正确更新世界中的这些障碍物。仅将第一个子节点换成 `OuterWall` 仍会遗漏杯底、内壁和杯沿。另一种办法是预处理成单一规划网格，但需要保留材质并验证物理碰撞形状。

### P2：放置只限制夹爪方向，没有约束纸杯保持直立

位置：`local_configs/pick_cup_to_plate_desk.yaml:180`；实现：`core/skills/place.py:298`。

`filter_z_dir: [downward, 150]` 对夹爪 Z 轴施加约束，允许其偏离竖直向下约 30°。`place_direction: vertical` 决定目标位置的生成方式，不代表保持杯口朝上。放置技能随机采样姿态，配置未启用物体轴对齐。

杯子和盘子当前均为 Z-up，可在 place 技能中增加：

```yaml
align_pick_obj_axis: [0, 0, 1]
align_place_obj_axis: [0, 0, 1]
align_obj_tol: 5
```

上述配置是待仿真验证的建议：应同时确认旋转候选数量，当前实现对“没有满足约束的姿态”会回退到未过滤姿态。若要求严格，需让该分支明确失败并重试。

### P2：成功判定过宽，而且没有检查回家动作后的最终状态

位置：任务 YAML 第 188 行；实现：`core/skills/place.py` 的 `xybbox` 分支，以及 `simbox_dual_workflow.py:update_skill_states()`。

当前只检查杯子原点的 XY 是否进入盘子轴对齐包围盒、并距边界大于 15 mm。没有检查杯底高度、倾斜角、盘面实际轮廓、接触或速度。杯子悬空、倾倒，甚至位于圆盘外但仍在包围盒角落，都可能通过。

成功是在 place 技能结束时检查的。之后还有 home 动作；杯子在撤离时被碰倒，不会自动重新检查“杯子稳定放在盘中”。

建议增加任务专用成功条件：杯底投影位于有效盘面、杯底与承托面高度接近、杯轴近似竖直、线速度和角速度足够小，并在松爪、撤离及最终等待后检查。不要把所有通用 place 的行为直接改成纸杯专用语义。

### P2：放置高度的注释与实际承托面不一致

位置：任务 YAML 第 186 行；实现：`core/skills/place.py:189`。

`position_constraint: object` 把物体原点放到盘子包围盒最高点加 `place_z_offset`。当前纸杯原点位于杯底；盘子局部中央顶面约 5 mm，盘沿最高 16 mm。因此 `place_z_offset: 0.005` 实际让杯底距离盘子中央约 16 mm，即 `16 + 5 - 5` mm，而不是注释中的 5 mm。

这不必然使放置失败，但会增加下落和弹跳。建议按实际承托面或指定盘面子网格计算高度，配合直立约束和释放后的等待验证。若临时用负偏移补偿，必须复核整杯、夹爪与盘沿的碰撞，不能只改数值。

## 需要澄清的配置与注释

- YAML 开头说使用 `custom/robots/split_aloha_desk/robot.usd`，实际第 49 行加载原始 `split_aloha_mid_360/robot.usd`。`make-desk-robot.py` 和该自定义机器人目前不在任务加载链中。
- 机器人升降关节 USD 的初始状态为 `state:linear:physics:position=0`，驱动目标为 `-0.6`，下限也为 `-0.6`。不能把“驱动目标”解释成“初始位置天然就是下限”。当前 SplitAloha 不显式初始化升降关节，臂基座达到 0.75 m 依赖驱动运动后的实际状态，应在预热结束后读取验证。整机抬高 0.0973 m 也意味着底盘相对地面抬起，应区分仿真摆位与实机安装方式。
- 两个 `orientation_mode: random` 目前不生效，因为 `apply_randomization: False` 阻止资产随机更新分支；实际只由 regions 绕 Z 轴随机旋转。建议改成 `keep` 表达直立意图，避免以后开启资产随机化时杯盘翻倒。
- `A_on_B_region_sampler` 会把 `pos_range` 的 Z 偏移加到自动计算的高度上。当前 Z=0 正确，但“Z 不参与摆放高度”的注释不准确。
- `Aligned_obj.usd` 当前是引用原始资产的包装层，不是注释所称的软链接。
- 杯盘最小中心距为 0.14 m，以最大半径 0.0375 m、0.077 m 计算，边缘最小净距约 25.5 mm，并非注释所说的 7 cm。
- `strict_mode=false` 时脚本参数 `20` 表示尝试 20 个随机布局，不保证输出 20 条成功数据。建议保留它用于调试，并明确参数含义。
- `t_eps=0.025`、`o_eps=1` 分别允许 25 mm 和约 57° 的抓取路点误差；对高 80 mm 的杯子偏宽松。应结合执行轨迹收紧，不建议用宽容差掩盖定位误差。

## 已通过的静态检查

- 任务 YAML 和引用路径有效：arena、机器人 YAML、三个相机配置、两个 CuRobo 配置及杯盘机器人 USD 均存在。
- 杯盘包装层 `/World/Aligned` 存在，具有 RigidBodyAPI；物体加载所需的 prim 子路径有效。
- 杯、盘、桌子的 USD 依赖解析无缺失引用或贴图。
- 显式缩放后尺寸：桌子 1.4 × 0.6 × 0.75 m；杯子直径 0.075 m、高 0.08 m；盘子直径 0.154 m、高 0.016 m。
- 桌子是带碰撞的静态几何，杯盘为动态刚体；杯子质量约 8 g，盘子质量约 200 g。
- 抓取标注存在，形状为 `(16, 17)`，数值全部有限，旋转矩阵行列式约为 1，坐标为米；无需再把抓取平移乘以 YAML 的 0.001。
- `neglect_collision_names: []` 保留物理碰撞；机器人 `ignore_substring: [material]` 没有把桌子排除出规划。两者作用于不同碰撞系统。
- 运行入口及环境脚本通过 `bash -n`；辅助 Python 脚本和相关模块通过 AST 语法检查。

以上不证明 IK 可达、真实夹持接触、抓取成功率、盘子稳定性或最终视频正确。当前尚无本次任务完整运行验证。

## 目录评估与建议

当前 `local_configs` 保存小型本地任务、资产统一从 `workflows/simbox/assets` 访问，这种方式合理。外部资产根目录通过软链接挂载，也适合避免把大资产提交到 Git。源资产和 `Aligned` 包装资产分开保存是合理的引用分层，应成套保留。

主要维护问题是：入口、包装和抓取生成脚本位于仓库外的 `.interndata`，代码与配置分散；仓库自身缺少可直接复现该任务的一套受版本管理入口。`custom/robot` 与 `custom/robots` 并存、旧桌面机器人与当前机器人并存，也容易混用。

推荐逐步组织为：

```text
InternDataEngine/
  local_configs/cup_plate_desk/
    task.yaml
    arena.yaml
    README.md                 # 尺寸、坐标、生成步骤、运行方法
  scripts/
    run_cup_plate_desk.sh
    assets/
      make_object_wrapper.py
      make_cup_grasp_labels.py
  workflows/simbox/assets -> 外部资产根目录
  output/cup_plate_desk/

外部资产根目录/custom/
  objects/
    paper_cup/                # 原始 USD 和贴图
    paper_cup_001/            # 包装 USD + Aligned_grasp_sparse.npy
    ceramic_plate/            # 原始 USD 和贴图
    ceramic_plate_001/        # 包装 USD；当前不抓盘子，无需抓取 NPY
    wood_table/
  robots/                     # 统一使用复数命名

.interndata/
  venv/
  curobo/
  assets-kitchen/
  logs/                       # 环境、下载缓存和运行日志
```

移动任务 YAML 后需要同步修改入口的 `cfg_path` 和任务的 `arena_file`。本项目这些路径通常相对仓库工作目录解析，不会自动相对 YAML 文件目录解析。

建议先修复完整物体的规划附加，再补直立放置和最终稳定性判定，之后跑单条可视化验证与多个随机种子的回归，最后再整理目录。当前未执行目录迁移，避免影响现有引用。
