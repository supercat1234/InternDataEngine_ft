# 自定义场景 / 任务修改手册

面向 `local_configs/custom_pick_bread.yaml` 这类自定义配置。所有数值和路径都基于本机实测，不是推测。

---

## 0. 先认清三条路径的分工

| 文件 | 管什么 | 路径规则 |
|---|---|---|
| `local_configs/<task>.yaml` | **任务**：机器人、物体、摆放区域、相机、技能、语言指令 | `asset_root` / `arena_file` 都相对**仓库根目录**，不是相对本文件 |
| `workflows/simbox/core/configs/arenas/<name>.yaml` | **场景**：桌子、地面、墙、纹理库 | fixture 的 `path` 相对 `asset_root` |
| `configs/simbox/de_*.yaml` | **执行**：模式、采样次数、输出目录 | 通常不改，用命令行覆盖 |

启动命令（把 `TASK_CFG` 换成你的文件即可）：

```bash
cd /home/agilex/qhx/InternDataEngine
bash scripts/run_custom_scene.sh 1 --load_stage.scene_loader.args.simulator.headless=false
```

调试时始终加诊断开关：

```bash
INTERNDATA_DEBUG_PICK=1 bash scripts/run_custom_scene.sh 1 2>&1 | tee output/diag/debug_$(date +%H%M).log
```

---

## 1. 采样数量 / 严格模式 / 真并行

### 1.1 一个场景重复采样 N 次（最常用）

```bash
bash scripts/run_custom_scene.sh 10 \
  --store_stage.writer.args.output_dir=output/batch1/
```

- 第 1 个位置参数 = `load_stage.layout_random_generator.args.random_num`
- 这是**串行**的：同一个 Isaac 进程里，每次重新随机布局再跑一遍

### 1.2 失败也计数 vs 必须成功

```bash
--load_stage.layout_random_generator.args.strict_mode=false   # 默认：失败也消耗次数
--load_stage.layout_random_generator.args.strict_mode=true    # 必须攒够 N 条成功才结束
```

想要 N 条**合格数据**，用 `true`；只想快速看成功率，用 `false`。

### 1.3 真正的多进程并行（两条路，先看区别）

| 方式 | 形态 | 优点 | 代价 |
|---|---|---|---|
| `scripts/run_parallel.sh` | N 个**完全独立**的进程，各自跑完整流水线 | 简单、容错好，一个崩了不影响其它；失败能单独重跑 | 每个实例都完整加载场景，显存占用叠加 |
| `de_pipe_*.yaml`（pipe 模式） | 1 个主进程 + 若干 worker，**规划段与渲染段分段并发** | 同一进程内调度，队列自动平衡 | 内部逻辑复杂，本机未跑通验证；一处卡住整条流水线受影响 |

**推荐先用 `run_parallel.sh`。**

```bash
bash scripts/run_parallel.sh                            # 默认 3 实例 × 每实例 5 条
INSTANCES=2 PER_INSTANCE=10 bash scripts/run_parallel.sh
INSTANCES=4 bash scripts/run_parallel.sh                 # 显存够就往上加
```

它做的事：第 i 个实例用 `SEED=42+i`、写 `output/parallel/inst_i/`、日志落 `inst_i.log`；
后台每 5 s 采样一次显存到 `gpu_watch.csv`，跑完自动汇总成功率并给出"还能加几个实例"。

两件事必须注意，否则并行等于白跑：

1. **种子必须不同**（脚本已处理）。同种子会生成完全一致的数据。
2. **输出目录必须不同**（脚本已处理）。共用目录虽然靠微秒级时间戳不会覆盖，但数据混在一起无法分辨来源。

pipe 模式（**本机尚未验证**）：

```bash
CONFIG=configs/simbox/de_pipe_custom_pick.yaml bash scripts/run_custom_scene.sh 6
```

`de_pipe_custom_pick.yaml` 已按本机任务配好。三个机制要点：

- `stage_num: [3, 3]` 是**按 stage 对象切分**，不是按进程数。本配置共 6 个 stage（load / plan / dump / dedump / render / store），切成 `[load, plan, dump]` 和 `[dedump, render, store]` 两段。
- `worker_num: [2, 1]` 的**和就是常驻 Isaac 实例数**（这里 3 个），显存按这个数算。
- 每个 worker 申请的 GPU 份额 = `1 / sum(worker_num)`（`make_pipe()` 里 `gpu_num = 1.0 / processes_per_gpu`，而 `ray.init(num_gpus=1)` 只声明了 1 张卡）。所以加 worker 不会自动分卡，**全挤在同一张 4090 上**。

`run_custom_scene.sh` 已自动把 `dedump_stage.scene_loader.args.cfg_path` 同步成 `TASK_CFG`（仅当 `CONFIG` 含 `pipe` 时追加），不用再手动传。

#### 单卡 4090 能开几个

本机实测（Isaac 启动横幅）：`RTX 4090 / 24564 MB`、`i7-13700 / 16 核 32 线程 / 可用 24 核`、**系统内存仅 32 GB**。

- **显存**：这个场景是 3 路 640×480 相机 + `anti_aliasing: 0` + headless，单实例大约 5~6 GB。24 GB 卡按留 4 GB 余量，**3 个实例（约 18 GB）稳妥，4 个是上限，5 个必炸**。
- **系统内存往往是先撞的墙**：32 GB 内存在跑 Isaac 时已经用掉一半，每个实例还要几 GB 主机内存，4 个实例容易触发 OOM Killer。
- **CPU 是第三道墙**：24 个可用核，每个实例要吃好几个线程，3 个实例就接近打满。

**结论：默认 3，最多试 4。** 真实数字必须自己测：

```bash
INSTANCES=3 bash scripts/run_parallel.sh          # 跑完看汇总里的"单实例均摊"
```

如果脚本报的均摊是 6 GB，那就最多 3 个；如果是 4 GB，可以试 4 个。**先 1 个、再 2 个、再 3 个**往上加，不要一次上 4。

#### 并行前顺手做两件事

```bash
sudo cpupower frequency-set -g performance   # 日志显示 CPU 处于 powersave，多实例会明显拖慢
free -g                                       # 确认可用内存，少于 12 GB 别开 3 个
```

### 1.4 单个任务重复多遍

`load_stage.scene_loader.args.task_repeat: N`（默认 -1 = 不重复）。

⚠️ 这个字段**不在** `de_*.yaml` 模板里，命令行覆盖会被配置校验拒绝（`ConfigProcessor` 只允许覆盖已存在的字段）。要用就先在 `de_plan_with_render_template.yaml` 的 `scene_loader.args` 下加一行，再传参。

---

## 2. 随机化：分五层，逐层开关

随机化不是一个大开关，而是五个独立层。**只开你需要的那层**，否则排查问题时会互相干扰。

| 层 | 字段 | 关闭方法 |
|---|---|---|
| 物体模型替换 | `objects[].apply_randomization` | 设 `false` 就固定用 `path` 那个模型 |
| 物体初始朝向 | `objects[].orientation_mode` | `keep`（用 `euler`）/ `random` / `suggested`（按类别查表） |
| 物体落点 | `regions[].random_config.pos_range` | 两个角点写成同一个点 |
| 机器人初始关节 | `robots[].left_joint_home_std` | 全 0 |
| 相机位姿 | `cameras[].apply_randomization` + `max_translation_noise` / `max_orientation_noise` | `false` |
| 光照 | `env_map.apply_randomization` | `false` |
| 纹理工件 | `arena.fixtures[].texture.apply_randomization` | `false`，再用 `texture_id` 指定 |

### 2.1 只改位置、其它全固定（做单变量实验的标准配置）

```yaml
env_map:
  apply_randomization: False
objects:
  - name: pick_0
    apply_randomization: False
    orientation_mode: keep
    euler: [0.0, 0.0, 0.0]
robots:
  - name: split_aloha
    left_joint_home_std: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    right_joint_home_std: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
cameras:
  - name: split_aloha_head
    apply_randomization: False
arena:
  fixtures:  # 纹理
    - name: table
      texture: {apply_randomization: False, texture_id: 0}
```

### 2.2 物体随机换模型的粒度

```yaml
apply_randomization: True
randomization_scope: category   # 默认：同类别目录下随机挑
# randomization_scope: full     # 所有类别混着挑
# randomization_scope: ["bottle", "cup"]   # 只在指定类别里挑
```

`category` 模式的实现是 `glob(asset_root/<上两级目录>/*/*.usd)`，所以**同类目录里只有 1 个模型时等价于不随机**。

### 2.3 随机种子

- `--random_seed=42` 决定整条随机序列
- 同 seed + 同次数 = 可复现；换 seed = 换一条独立序列
- 注意 `update_scene_pair` 里另外用了 `random.seed(os.getpid() + 时间戳)`，**换桌子这一步是不受 `--random_seed` 控制的**

### 2.4 换桌子的频率

```yaml
arena:
  involved_scenes: dining_room_scene_info
  update_freq: 5000     # 每 N 条样本换一次桌/房间组合，见 banana.py individual_reset
```

---

## 3. 改桌面高度

桌子高度由 arena 里 table fixture 的 **`translation[2]` + `scale[2]`** 共同决定。

```yaml
# workflows/simbox/core/configs/arenas/custom_home_arena.yaml
fixtures:
  - name: table
    path: table0/instance.usd
    translation: [0.0, 0.0, 0.375]     # ← 桌子整体上移/下移看这里
    scale: [0.001, 0.001053, 0.001056] # ← 桌子整体变大变小看这里
```

关键点：

1. **物体和机器人都是相对桌面自动摆放的**，不是绝对高度。`A_on_B_region_sampler` 里：
   ```python
   place_pos[2] = tgt_z_max + (物体原点高度 - 物体最低点高度) + 0.001 + shift_z
   ```
   所以把桌子抬高 10 cm，机器人和物体**会跟着一起抬高**，相对关系不变。
2. 想知道桌子现在到底多高，跑一次读日志里的 `REGIONDBG`：
   ```
   [REGIONDBG] pick_0: tgt_center=(0.056, 0.169, ...) tgt_z_max=1.2968 shift=[...] place_pos=[...]
   ```
   `tgt_z_max` 就是**桌面顶面的世界高度**。本机当前这套 = **1.2968 m**（偏高，普通书桌约 0.75 m）。
3. 想让桌子变矮：优先只改 `translation[2]`（等比缩小会连桌面尺寸一起变）。

---

## 4. 改机器人高度 / 位置 / 朝向

机器人的位置由 **`regions` 里针对机器人的那条**决定，不是 `robots[].translation`（后者会被覆盖）。

```yaml
regions:
  - object: split_aloha        # 名字必须和 robots[].name 一致
    target: table
    random_type: A_on_B_region_sampler
    random_config:
      pos_range: [
        [0.0, -0.86, -0.765],
        [0.0, -0.86, -0.765]
      ]
      yaw_rotation: [0.0, 0.0]
```

逐字段含义：

| 字段 | 含义 | 常见调整 |
|---|---|---|
| `pos_range[0/1][2]` | **相对桌面顶面**的高度偏移 | 负数越小 = 机器人越低；`-0.765 ~ -0.715` 是官方常用区间 |
| `pos_range[*][1]` | 机器人离桌子的远近 | `-0.86` → 改到 `-0.75` 可让机器人**靠近桌子 11 cm**（够不到时的第一手段） |
| `pos_range[*][0]` | 左右居中 | 双机械臂一般保持 0 |
| `yaw_rotation` | 朝向 | 面向桌子通常是配合 `robots[].euler: [0,0,90]` |
| 两个角点不同 | 高度随机化 | 单点 = 固定高度 |

朝向在 `robots[].euler`：

```yaml
robots:
  - name: split_aloha
    euler: [0.0, 0.0, 90.0]   # 绕 Z 转 90°，让机器人正面朝桌子
```

> 机器人如果在 `regions` 里**没有**对应条目，才会用 `robots[].translation` 的绝对坐标。

---

## 5. 改物体摆放范围

```yaml
regions:
  - object: pick_0
    target: table
    random_type: A_on_B_region_sampler
    random_config:
      pos_range: [
        [-0.30, -0.20, 0.0],
        [-0.025, 0.10, 0.0]
      ]
      yaw_rotation: [-180.0, 180.0]
```

### 5.1 坐标语义（务必记住这三条）

1. 两个数组是矩形的**两个对角点**，不是尺寸
2. **x/y 是相对 `target`（桌子）包围盒中心的偏移**，不是世界坐标
3. **z 是相对桌面顶面的偏移**，0 就是贴桌面；写成 0.05 就是悬空 5 cm 再自由落体

### 5.2 六种采样器

| `random_type` | 用途 | 需要的参数 |
|---|---|---|
| `A_on_B_region_sampler` | 在 B 的顶面上撒 A（最常用） | `pos_range`, `yaw_rotation` |
| `A_in_B_region_sampler` | 把 A 放进 B 的容器里（如箱子内） | `x_bias`, `y_bias`, `z_bias` |
| `A_by_B_circle_sampler` | 绕 B 中心按半径+角度撒 | `r_range`, `theta_range`, `yaw_rotation` |
| `A_by_B_region_sampler` | 在 B 侧面/某面撒 | `pos_range`, `yaw_rotation` |
| `A_face_B_circle_sampler` | 贴着 B 的某个面按圆周撒 | `r_range`, `yaw_rotation` |
| `A_along_B_C_circle_sampler` | 绕 B→C 连线按圆周撒 | `r_range`, `yaw_rotation`（配置需 `target2`） |

### 5.3 多个物体如何避免互相重叠

```yaml
objects:
  - name: pick_0
    optimize_2d_layout: True    # 打开：在区域内反复试放，直到不与已放置物体相交
```

它会读同一 `regions` 条目里的 `pos_range` 构造多边形，做 10 次试放（`layout.py: optimize_2d_manip_layout`）。**它只解决物体之间重叠，不会帮你把物体挪进机械臂可达范围。**

### 5.4 想按概率选用几套预设区域

任务级可加 `random_region_list`，区域条目用 `priority` 选：

```yaml
random_region_list:
  - random_config: {pos_range: [[-0.3,-0.2,0],[-0.05,0.1,0]], yaw_rotation: [0,0]}
  - random_config: {pos_range: [[0.05,-0.2,0],[0.3,0.1,0]], yaw_rotation: [0,0]}

regions:
  - object: pick_0
    target: table
    random_type: A_on_B_region_sampler
    priority: [0]        # 固定用第 0 套；写 [] = 每次随机挑一套
```

---

## 6. 换物体

### 6.1 需要改的三行

```yaml
objects:
  - name: pick_0
    path: art/heat_the_food_in_the_microwave/pick_objs/omniobject3d-bread_090/Aligned_obj.usd
    target_class: RigidObject
    prim_path_child: Aligned
    scale: [0.001, 0.001, 0.001]
```

### 6.2 四个必须满足的条件

| 条件 | 检查方法 | 不满足的后果 |
|---|---|---|
| 同目录下存在 `Aligned_grasp_sparse.npy` | `ls` 一下 | 直接 `FileNotFoundError` |
| `prim_path_child` 与 USD 内的刚体 prim 名一致 | 通常是 `Aligned` | 物体加载但抓不到，包围盒异常 |
| `scale` 与 USD 网格单位匹配 | 见 6.3 校准法 | 物体过大/过小，抓取点悬空 |
| `target_class` 选对 | 刚体 `RigidObject`；铰接 `ArticulatedObject` | 加载报错或行为错误 |

### 6.3 scale 校准法（本次踩坑总结，强烈建议照做）

**不要**用抓取 `npy` 的数值去推 USD 的 `scale`——两者单位体系不同（实测：`npy` 是米制，USD 网格是毫米制）。用包围盒实测：

```bash
# 1) 先把 scale 设成 1.0，跑一次
INTERNDATA_DEBUG_PICK=1 bash scripts/run_custom_scene.sh 1 2>&1 | grep PICKDBG | head -3
```

```
[PICKDBG] obj_bbox_min: [-31.912 -31.5774 1.2978] max: [31.4621 31.4851 115.5372]
```

2) 读 `max - min` 得到物体的原始尺寸，再算比例：

| 物体 | 期望尺寸 | scale |
|---|---|---|
| 面包 | 约 0.06 × 0.06 × 0.11 m | `1.0` 时是 63×63×114 m → 需要 `0.001` |
| 桌子 | 桌面约 0.8 × 0.8 m | `1.0` 时是几百米 → 需要 `0.001` 量级 |

原则：**`scale = 期望尺寸 / 实测尺寸`**，一般就是 `0.001`（mm）或 `0.01`（cm）或 `1.0`（m）三选一。

### 6.4 换成铰接物体（微波炉、抽屉）

```yaml
- name: microwave
  target_class: ArticulatedObject
  path: ""                      # 由 art_cat 自动展开，留空
  art_cat: "art/microwave_gr"   # 类别目录
  info_name: "close_v"          # 关键点组名，决定初始开合角
  joint_position_range: [0.78, 0.875]
  scale: [0.001, 0.001, 0.001]
```

要求目录里有 `Kps/<info_name>/info.json`（提供尺度与关键点），配合技能 `open` / `close` / `artpreplan` 使用。

### 6.5 换物体后必须复查

- `regions` 里针对它的 `pos_range` 要重调（新物体大小/重心不同）
- 技能里 `objects: [pick_0]` 的名字没写错
- `data.language_instruction` 里的物体名（可用 `${tasks.0.objects.0.name}` 自动取）

---

## 7. 换桌子

### 7.1 换同类型桌子

```yaml
fixtures:
  - name: table
    path: table0/instance.usd        # ← 换成新桌子的 USD
    target_class: GeometryObject
    translation: [0.0, 0.0, 0.375]
    scale: [0.001, 0.001053, 0.001056]
```

新桌子 `scale` 同样用包围盒校准：跑一次看 `REGIONDBG` 的 `tgt_z_max`，桌面顶面高度应该落在你期望的范围（0.75 m 左右）。**`tgt_z_max` 变了，机器人的相对高度基准也变了**，所以换桌后先单独跑一次只做加载验证。

### 7.2 换成真实房间（带墙、地板贴图的整屋）

改走 `involved_scenes` 机制，需要在**仓库根目录**放一个 `xxx_scene_info.json`：

```json
[
  {"001": {
     "table": {"name": "table", "path": "home_scenes/tables/dining_room/xxx.usd",
               "target_class": "GeometryObject", "translation": [0,0,0.455], "scale": [0.01,0.01,0.01]},
     "scene": {"name": "scene", "path": "home_scenes/rooms/dining_room/yyy.usd",
               "target_class": "GeometryObject", "translation": [2.449,1.016,0.0], "scale": [0.01,0.01,0.01]}
  }},
  {"002": { "...": "..." }}
]
```

arena 里：

```yaml
name: scene_arena
involved_scenes: dining_room_scene_info   # 多个用逗号分隔
update_freq: 5000
fixtures:
  - {name: table, ...apply_randomization: True}
  - {name: scene, ...apply_randomization: True}
```

三条硬约束（`utils/dr.py: update_scene_pair`）：

1. **必须正好 2 个 fixture**
2. 两个都要 `apply_randomization: True`
3. 第一个名字含 `table`，第二个名字含 `scene`；顺序不能反

> 只要有一个物体名字里含 `hearth`，就会改走 `update_hearths` 分支，`involved_scenes` 失效。

---

## 8. 诊断开关与判读

```bash
INTERNDATA_DEBUG_PICK=1 bash scripts/run_custom_scene.sh 1 2>&1 | tee output/diag/dbg.log
```

三类前缀：

| 前缀 | 来源 | 关键信息 |
|---|---|---|
| `[REGIONDBG]` | `region_sampler.py` | `tgt_center` / `tgt_z_max`（桌面高度）/ `shift` / `place_pos` |
| `[PICKDBG]` | `pick.py` | 物体位姿与包围盒、过滤后候选数、当前末端位置、前 5 个目标点、`T_world_base` |
| `[PLANDBG]` | `template_controller.py` | 每个目标点、`succ`、当前关节角 |

判读表：

| 现象 | 结论 | 处理 |
|---|---|---|
| `pick plan success` 出现 | 有候选可解 | 等 `Task is successful` |
| 全是 `Plan did not converge` + 距臂基座 > 0.47 m | **超出可靠带** | 物体挪近 / 机器人 `pos_range[1]` 改小 |
| IK `grasp=True` 但仍全灭 | 路径或构型问题，不是够不到 | 看方向（纯前向最差）、看 `filter_*_dir` 候选数 |
| `pick plan success` 但没有 `Task is successful` | 抓到了没抬起（`lift_th` 20 mm 判定） | 降 `post_grasp_offset`、查物体尺寸与手指接触 |
| `candidates_after_filter: 0` | 方向过滤太严 | 放宽 `filter_*_dir` 的角度 |
| `obj_bbox` 尺寸离谱 | `scale` 错 | 按 6.3 校准 |
| `obj_local_pose` z 为负或几米高 | 物体没落在桌面 | 查 `pos_range` z 与物体包围盒 |
| `pos_err/rot_err` 是 `None` | 单候选分支不返回误差 | 只能用候选数 + 目标坐标判断 |
| 没有 `pick plan success` 但也没报错 | 可能卡在物体接触判定 | 看是否 `attach_obj` 失败 |

### 可达性红线（本机实测，2026-10-27 更正）

- `piper100` 单臂 `arm_base → link6` 运动学臂展约 **0.75 m**，但这只是"单个位姿能解出来"的上限，**不是成功率上限**
- 判断方法：`d = sqrt((y+0.4915)^2 + (x+0.306)^2)`，左臂基座世界坐标实测 `(-0.306, -0.4915)`（右臂 x 取正）
- **可靠带 0.30 ~ 0.44 m**。5 次采样实测：0.385 成、0.343 成、0.474（斜向）成；0.471（纯前向）、0.498 全灭

**方向比距离更关键**：前向（y 方向）最费行程。同样 0.47 m，纯前向全灭、斜向能成。所以收区域要压 y 跨度，x 可以放宽一些。

**IK ≠ 可规划**：`[PICKDBG] IK grasp=True` 只说明姿态能解（`test_single_ik` 不算碰撞和路径）。IK 全 True 但仍全是 `Plan did not converge` → 问题在路径或构型，不是"够不到"。

---

### 碰撞：两个互不相干的开关

| 开关 | 位置 | 管的是哪一层 |
|---|---|---|
| `neglect_collision_names` | 任务 YAML 顶层 | **PhysX 物理碰撞组**。写进列表的名字被拉进"互不碰撞组"，机器人可以真的穿过它 |
| `ignore_substring` | `robots[]` 或 skill 里 | **CuRobo 规划世界**。匹配到的 prim 不参与轨迹规划的碰撞检测 |

四条要点：

1. **桌面默认就在 CuRobo 的规划世界里**（日志里 `Object already in warp cache ... table` 就是它被加进去），规划一直在避桌子，`neglect_collision_names` 并不影响这一点。
2. `neglect_collision_names: ["table"]` 只放开了**物理**穿模，手臂真撞上桌子不会被挡。要出可用于迁移的数据，写 `[]`。
3. 打开桌面碰撞前先确认 home 不穿模：`piper100` home 姿态下各 link 原点相对臂基座 z ≥ 0.123 m，桌面在 −0.209 m，安全。
4. 想反查某物体是否被物理忽略，看它在不在 `neglect_collision_names` 里，**不要**去看 `ignore_substring`。

---

## 9. 改完必查清单

1. `yaml.safe_load` 能过（缩进、列表短横线）
2. `asset_root` 下 `path` 真实存在
3. 新物体有 `Aligned_grasp_sparse.npy`
4. 目标点到臂基座距离 ≤ 0.70 m
5. 技能里的 `objects: [...]` 名字与 `objects[].name` 一致
6. 多个任务时 `task_id` 递增（决定 `/World/task_{id}` prim 路径，重复会撞）
7. 首次改动先用 `headless=false` 看画面对不对，再批量跑

---

## 10. 常用改法速查

| 我想… | 改这里 |
|---|---|
| 一次出 20 条数据 | `bash scripts/run_custom_scene.sh 20` |
| 只要有成功的 20 条 | 追加 `--load_stage.layout_random_generator.args.strict_mode=true` |
| 换输出目录 | 追加 `--store_stage.writer.args.output_dir=output/xxx/` 或 `OUT_DIR=... bash ...` |
| 桌子矮 10 cm | arena 里 table 的 `translation[2]` 减 0.1 |
| 机器人离桌子近 10 cm | 机器人 region 的 `pos_range[*][1]` 从 -0.86 改 -0.76 |
| 机器人更矮 | 机器人 region 的 `pos_range[*][2]` 更负 |
| 物体撒得更散/更集中 | `pos_range` 两个角点拉开/收窄 |
| 物体随机转圈 | `yaw_rotation: [-180, 180]`；固定则写 `[0, 0]` |
| 物体每次换型号 | `apply_randomization: true` + 同类目录下有多个模型 |
| 关掉光照随机 | `env_map.apply_randomization: false` |
| 关掉纹理随机 | arena 里每个 fixture 的 `texture.apply_randomization: false` |
| 固定随机结果 | `--random_seed=42` 且 `random_num` 不变 |
| 让桌面参与真实物理碰撞 | 任务 YAML 顶层 `neglect_collision_names: []` |
| 并行多实例出数据 | `INSTANCES=3 PER_INSTANCE=5 bash scripts/run_parallel.sh` |
| 换执行配置（如 pipe 模式） | `CONFIG=configs/simbox/de_pipe_custom_pick.yaml bash scripts/run_custom_scene.sh 6` |
| 换随机种子 | `SEED=7 bash scripts/run_custom_scene.sh 5` |
| 多进程生产 | `--config configs/simbox/de_pipe_template.yaml` + `stage_pipe.worker_num` |
