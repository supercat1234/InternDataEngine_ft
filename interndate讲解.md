# InternDataEngine 使用笔记

## 简介

InternDataEngine 基于 Isaac Sim，通过任务配置、场景随机化、运动规划和图像渲染，自动生成机器人操作演示数据。

## 使用方法

 **1. Split ALOHA 垃圾分类**示例:无需激活 Conda，直接使用本机启动脚本：

```bash
# 不显示窗口，尝试生成一条演示
bash /home/agilex/qhx/.interndata/run-example.sh 1

# 显示仿真窗口，观察机器人动作
bash /home/agilex/qhx/.interndata/run-example.sh 1 \
  --load_stage.scene_loader.args.simulator.headless=false
```

将 `1` 改为 `10` 可尝试 10 次；失败场景默认跳过，因此成功条数可能较少。看到 `Task is successful` 后，等待保存完成。启动脚本已包含 Isaac Sim 5.1 的单位兼容适配。

 **2. lift2食物放入微波炉并关门任务**(/home/agilex/qhx/InternDataEngine/workflows/simbox/core/configs/tasks/art/lift2/heat_the_food_in_the_microwave/heat_the_food_in_the_microwave.yaml)

完成:抓食物 → 放进微波炉内腔 → 关门

运行命令:

```
bash /home/agilex/qhx/.interndata/run-lift2-microwave.sh 3 \
  --load_stage.scene_loader.args.simulator.headless=false \
  --load_stage.layout_random_generator.args.strict_mode=true \
  --random_seed=43 
```

 --load_stage.layout_random_generator.args.strict_mode=true \代表开启严格模式,成功才算数,默认strict_mode=false,不成功也算进次数

...sh 1 \ --random_seed=42  代表按照seed=42的序列跑1次

...sh 10 \  \--random_seed=42  按照seed=42的序列跑10次,注意每次都做了随即化,但是每次的顺序是一样的,如果更换random_seed=123,就是另一个序列

--store_stage.writer.args.output_dir=output/lift2_batch/ 代表数据存到哪,默认是 `output/${name}/`（`${name}` 取自执行配置里的 `name`，即 `simbox_plan_with_render`）

这个示例写死了只有位置随机化,纹理\光照等不变:

| 会随种子变化                                                 | 不会变                                                       |
| :----------------------------------------------------------- | :----------------------------------------------------------- |
| 微波炉在桌面的 **x/y 位置**（±5cm）                          | 微波炉型号（固定 `microwave_0013`）                          |
| 面包落点 **x 0.15~0.25 / y -0.30~-0.23**                     | 面包模型（固定 `bread_090`）、朝向（yaw 全 0）               |
| lift2 基座 **x ±2.5cm、z -0.7~-0.6**（相当于桌子相对机器人的高度变化 10cm） | 关节初始角（`std` 全 0）、相机噪声（关闭）                   |
| 一条 run 内多条样本的排列顺序                                | 光照/背景（`env_map.apply_randomization: false`，arena 也没写纹理） |

### 3.定制任务

自定义程度分三档

| 档位               | 场景                                                 | 需要动什么                                                   |
| :----------------- | :--------------------------------------------------- | :----------------------------------------------------------- |
| **A 档（最常见）** | 换物体、换摆放区域、换相机位、换动作顺序             | 只改任务 YAML。arena 能复用就不新建                          |
| **B 档**           | 换机器人（如换 Franka、lift2），或换纹理库、新资产库 | 任务 YAML + `robots/xxx.yaml`；纹理库只需往 `assets/` 加目录 |
| **C 档**           | 要现有技能表达不了的动作（如按按钮、倒水的新逻辑）   | 必须改 Python：`core/skills/` 下加技能类，并在任务类里注册   |

**先讨论a挡情况,主要修改三个yaml文件+.sh 启动脚本,三个文件的角色:**

例如抓面包简单任务:

`bash scripts/run_custom_scene.sh 1 --load_stage.scene_loader.args.simulator.headless=false`

| 文件                                                         | 作用                                                       |
| :----------------------------------------------------------- | :--------------------------------------------------------- |
| `local_configs/custom_pick_bread.yaml`                       | 任务配置：机器人、物体、摆放、相机、技能、指令             |
| `workflows/simbox/core/configs/arenas/custom_home_arena.yaml` | 场景：桌子、地面、墙、纹理                                 |
| `configs/simbox/de_plan_with_render_template.yaml`           | 执行：采样次数、模式、输出（多数用命令行覆盖，不用改文件） |

> **一、任务配置 `custom_pick_bread.yaml`到对话**
>
> | 修改需求           | 行号     | 字段                                                         | 怎么改                                                      |
> | :----------------- | :------- | :----------------------------------------------------------- | :---------------------------------------------------------- |
> | 换场景             | 21       | `arena_file`                                                 | 指向另一个 arena YAML 路径                                  |
> | 光照随机化         | 23–27    | `env_map` / `apply_randomization` / `intensity_range` / `rotation_range` | `apply_randomization: False` 关闭；改强度/旋转范围          |
> | 机器人朝向         | 34       | `robots[].euler`                                             | 面朝桌子 `[0,0,90]`                                         |
> | 机器人初始关节     | 35–36    | `left_joint_home` / `right_joint_home`                       | 各 6 个角                                                   |
> | 关节随机噪声       | 37–38    | `left_joint_home_std` / `right_joint_home_std`               | std 越大越抖                                                |
> | 换物体             | 46–48    | `name` / `path` / `target_class`                             | `path` 换模型路径，`target_class` 换类型                    |
> | 物体刚体 prim      | 51       | `prim_path_child`                                            | 抓取失败先查这里                                            |
> | 物体缩放           | 54       | `scale`                                                      | 毫米制 `[0.001,0.001,0.001]`                                |
> | 物体随机化         | 55–56    | `apply_randomization` / `orientation_mode`                   | 换模型开关 + 朝向模式                                       |
> | **物体摆放范围**   | 63–71    | `object: pick_0` 的 `pos_range` / `yaw_rotation`             | x/y/z 范围和朝向角度                                        |
> | **机器人底座位置** | 73–81    | `object: split_aloha` 的 `pos_range`                         | y 离桌距离、z 相对桌面高度                                  |
> | 相机视角           | 85–109   | `cameras[]`                                                  | 每个相机的 `parent` / `translation` / `apply_randomization` |
> | 语言指令           | 114–115  | `language_instruction` / `detailed_language_instruction`     | 换任务要同步改                                              |
> | 动作序列           | 123–130+ | `skills`                                                     | 各臂技能顺序、参数                                          |
>
> **二、场景配置 `arenas/custom_home_arena.yaml`**
>
> | 修改需求       | 行号                   | 字段                                      | 怎么改                                   |
> | :------------- | :--------------------- | :---------------------------------------- | :--------------------------------------- |
> | **桌面高度**   | 11                     | `translation: [0,0,0.375]`                | 改 z（同时重算机器人 region 的 z）       |
> | **换桌子模型** | 9                      | `path`                                    | 换成别的 USD                             |
> | 桌子尺寸       | 12                     | `scale`                                   | 按新模型尺寸调，别乱改                   |
> | 桌子纹理       | 13–17                  | `texture` / `texture_lib` / `texture_id`  | `texture_id` 0~7，或换 `texture_lib`     |
> | 地面/墙位置    | 24 / 36 / 47 / 58 / 69 | `floor`、`background0~3` 的 `translation` | 围出房间的四墙与地面                     |
> | 地面/墙纹理    | 26–29 等               | 各 `texture.texture_lib`                  | `floor_textures` / `background_textures` |
>
> > 想换成真实房间（带墙贴图的整屋 USD），用文件末尾 80–91 行注释掉的那套「table + scene 两件套」写法。
>
> 
>
> **三、执行配置 `configs/simbox/de_plan_with_render_template.yaml**
>
> | 修改需求     | 行号 | 字段                | 说明                               |
> | :----------- | :--- | :------------------ | :--------------------------------- |
> | **采样次数** | 17   | `random_num: 6`     | 每任务随机重采样次数（= 数据条数） |
> | 严格模式     | 18   | `strict_mode: true` | 失败是否计入次数                   |
> | 异步写盘     | 26   | `batch_async`       | 生产可开 true 提速                 |
> | 输出目录     | 27   | `output_dir`        | 数据保存位置                       |
>
> **注意**：这三个文件里，执行配置（三）的字段通常**不用改文件**，用命令行覆盖更灵活：
>
> bash
>
> ```bash
> bash scripts/run_custom_scene.sh 10 \
>   --load_stage.layout_random_generator.args.strict_mode=true \
>   --store_stage.writer.args.output_dir=output/my_batch/
> ```
>
> ## 关键提醒（位置之间的联动）
>
> 1. **改第 11 行桌高** → 必须同步改**第 73–81 行**机器人 region 的 z。
> 2. **改第 9 行换桌子** → 重算桌子 scale + 机器人 z + 物体区域（第 63–71 行）。
> 3. **换物体（46–48 行）** → 新物体目录必须有 `Aligned_grasp_sparse.npy`，否则 `pick` 报错。
> 4. **物体 x<0 用左臂、x>0 用右臂**（第 123 行 skills 里的 arm 名要对应）。



**自定义任务1：拿起纸杯放到盘子里**

```
bash /home/agilex/qhx/InternDataEngine/scripts/run_cup_plate_desk.sh 1 \
  --load_stage.scene_loader.args.simulator.headless=false \
  --load_stage.layout_random_generator.args.strict_mode=true \
  --random_seed=43 
```

按当前 [任务配置](/home/agilex/qhx/InternDataEngine/local_configs/cup_plate_desk/task.yaml) 和 [材质配置](/home/agilex/qhx/InternDataEngine/local_configs/cup_plate_desk/arena.yaml)，已启用以下随机化：

| 类别               | 当前设置                                           |
| ------------------ | -------------------------------------------------- |
| **杯子位置**       | 中心 X：−0.3125～0.0625 m；Y：−0.1625～0.0625 m    |
| **盘子位置**       | 中心 X：−0.273～0.023 m；Y：−0.123～0.023 m        |
| **杯盘初始朝向**   | 分别绕竖直轴随机−180°～180°，保持直立              |
| **双臂安装高度**   | 相对旧高度共同上调80 mm，再扰动±20 mm，即上调60～100 mm |
| **光照强度**       | 标称值1000的0.8～1.2倍，即800～1200                |
| **光照色温**       | 白光5500～6500 K、暖光3000～4000 K，各占50%        |
| **光照方向**       | XYZ欧拉角分别扰动±5°、±5°、±20°                    |
| **桌面木纹与颜色** | 木纹一标准/浅色各20%；木纹二标准/浅色各15%         |
| **桌面纯色**       | 无纹理暖木色15%、纯白色15%，不再叠加标准/浅色档    |
| **桌面浅色处理**   | 当前为原纹理颜色×0.75＋白色×0.25                   |
| **地砖纹理**       | 原黑色大理石40%；另外四张各15%                     |
| **外部相机位置**   | 围绕 `[0, -0.48, 1.44]` m，各轴独立扰动 **±20 mm** |
| **外部相机朝向**   | 围绕标称朝向，绕随机轴旋转，总角度不超过2°         |

杯盘共用 `shared_region_sampling.xy_bounds` 指定的区域：X 为−0.35～0.10 m，Y 为−0.20～0.10 m。按杯半径37.5 mm、盘半径77 mm向内留边，得到表中的中心范围；整对位置重新采样，保证完整轮廓在区域内、边缘间距至少10 mm。当前 `regions.pos_range` 不控制杯盘XY，`yaw_rotation` 仍生效。

随机化每次布局（含失败重试）采样一次，轨迹内固定。双臂高度由 `robots.base_height_randomization` 控制，关闭 `enabled` 后固定在上调80 mm的标称高度；桌高固定750 mm，腕部相机不额外扰动、随手臂运动。抓取位姿当前由规划器优选（`chain_grasp_selection: best`），不是均匀随机采样。

打开仿真检查场景：

> “只预览”入口，默认打开外部 D435i 视角：
>
> ```
> bash /home/agilex/qhx/InternDataEngine/scripts/preview_cup_plate_desk.sh
> ```
>
> 查看整个场景，用自由视角：
>
> ```
> bash /home/agilex/qhx/InternDataEngine/scripts/preview_cup_plate_desk.sh --camera perspective
> ```
>
> 查看腕部相机：
>
> ```
> # 左腕；右腕改为 hand_right
> bash /home/agilex/qhx/InternDataEngine/scripts/preview_cup_plate_desk.sh --camera hand_left
> ```
>
> 窗口中也可以通过 **Viewport 的相机菜单**切换视角。预览会短暂运行物理让物体落稳，然后保持暂停，**不执行抓取、不生成数据集**。关闭窗口即可退出。
>
> 修改 [task.yaml](/home/agilex/qhx/InternDataEngine/local_configs/cup_plate_desk/task.yaml) 中的场景或相机配置后，重启预览即可查看效果。界面里的临时调整不会自动写回 YAML。

多实例并行

```
INSTANCES=3 PER_INSTANCE=66 BASE_SEED=42 \
bash /home/agilex/qhx/InternDataEngine/scripts/run_cup_plate_parallel.sh \
  --load_stage.layout_random_generator.args.strict_mode=true \
  --load_stage.layout_random_generator.args.max_retries=40
```

strict_mode=true严格模式，失败后重新随机布局并重试直到攒够次数

  --load_stage.layout_random_generator.args.max_retries=20最大额外重试次数

两个实例分别使用种子42、43，输出在：

```
output/cup_plate_desk/v0.6/parallel/<时间_进程号>/
├── inst_1/        # 数据及独立引擎日志
├── inst_2/
├── inst_1.log
├── inst_2.log
├── summary.csv   # 成功数、实际保存数、退出码
└── gpu_watch.csv # 显存采样
```

## 输入

路径均相对于 `InternDataEngine/`：

| 输入 | 内容 |
| --- | --- |
| `configs/simbox/de_plan_with_render_template.yaml` | 执行模式、采样次数、渲染与保存设置 |
| `workflows/simbox/core/configs/tasks/example/sort_the_rubbish.yaml` | 机器人、物体、相机、摆放区域和技能序列 |
| 模型与规划资源 | USD 模型、材质、抓取标注、机器人规划配置等 |

加热蛋挞任务:

```
cd /home/agilex/qhx
bash .interndata/run-egg-tart-microwave-knob.sh 1 \
  --load_stage.scene_loader.args.simulator.headless=false \
  --random_seed=43 
```



## 输出与数据内容

默认保存到 `InternDataEngine/output/minimal_example/`，每个时间戳目录是一条成功演示：

```text
<演示目录>/
├── images.rgb.head/demo.mp4        # 头部视角
├── images.rgb.hand_left/demo.mp4   # 左腕视角
├── images.rgb.hand_right/demo.mp4  # 右腕视角
├── lmdb/data.mdb
├── lmdb/info.json
└── meta_info.pkl
```

| 文件 | 保存的信息 |
| --- | --- |
| `data.mdb` | 各帧 RGB 图像、双臂关节状态/速度、夹爪状态、动作标签、物体位姿、相机外参等 |
| `info.json` | 当前样例的三个相机内参矩阵，也存于 LMDB 的 `json_data` 键 |
| `meta_info.pkl` | 观测步数、任务指令、版本、LMDB 字段索引和图像步编号 |

一条演示包含多帧观测；已验证样例为 **647 帧、三个 RGB 视角，约 125 MiB**。当前保存代码将下一步状态作为 `actions.*.position` 标签，并另存控制器动作 `master_actions.*`。

[详细使用指南](QUICKSTART_LOCAL.zh-CN.md) · [安装记录](MINIMAL_INSTALL_REPORT.md) · [官方 Quick Start](https://internrobotics.github.io/InternDataEngine-Docs/guides/quickstart.html)
