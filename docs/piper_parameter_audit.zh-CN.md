# 当前任务目录与 Piper 参数核对

日期：2026-10-28。本次为文件与参数审查，没有改动任务或机器人参数。图片为 AgileX 六轴 PIPER 产品铭牌，未列出硬件/固件版本、连杆尺寸、限位或夹爪开度，不能单凭铭牌确认整套参数一致。

## 当前任务相关目录

```text
/home/agilex/qhx/InternDataEngine/
├── local_configs/
│   ├── cup_plate_desk/
│   │   ├── task.yaml                  # 任务：机器人、杯盘、相机、侧抓与放置
│   │   ├── arena.yaml                 # 桌子和地面
│   │   ├── README.md                  # 使用和参数说明
│   │   └── grasps/paper_cup_side.npy   # 72 个杯壁外侧侧抓
│   ├── pick_cup_to_plate_desk.yaml -> cup_plate_desk/task.yaml
│   └── cup_plate_desk_arena.yaml -> cup_plate_desk/arena.yaml
├── scripts/
│   ├── run_cup_plate_desk.sh           # 当前推荐入口
│   ├── isaac_python.sh                # 运行环境
│   ├── launch_isaac.py                # 单位与启动适配
│   └── assets/
│       ├── make_cup_grasp_labels.py
│       └── make_object_wrapper.py
├── configs/simbox/de_plan_with_render_template.yaml
├── workflows/simbox/
│   ├── assets -> /home/agilex/qhx/.interndata/assets-kitchen/InternDataAssets/assets
│   ├── curobo -> /home/agilex/qhx/.interndata/curobo
│   └── core/
│       ├── configs/robots/split_aloha.yaml
│       ├── robots/split_aloha.py
│       ├── controllers/{splitaloha_controller,template_controller}.py
│       ├── objects/rigid_object.py
│       ├── skills/{pick,place,verify_placement}.py
│       └── utils/{rigid_geometry,placement_validation}.py
├── workflows/simbox_dual_workflow.py
├── tests/unit/test_cup_plate_task.py
├── docs/
│   ├── cup_plate_desk_review.zh-CN.md
│   ├── cup_plate_desk_validation.zh-CN.md
│   └── piper_parameter_audit.zh-CN.md
└── output/
    ├── cup_plate_desk/                # 最终验证数据、截图、核验 JSON
    └── cup_plate_diagnostics/         # 之前调试数据
```

外部资产根目录中，任务实际引用：

```text
split_aloha_mid_360/robot.usd
custom/objects/
├── wood_table/wood_table_v2.usd
├── paper_cup/paper_cup.usd             # 原始源文件，附贴图
├── paper_cup_001/Aligned_obj.usd       # 引用源文件的包装层
├── ceramic_plate/ceramic_plate.usd     # 原始源文件，附贴图
└── ceramic_plate_001/Aligned_obj.usd   # 引用源文件的包装层
```

`custom/robots/split_aloha_desk/robot.usd` 和 `custom/robot/piper_desktop_dual/piper_desktop_dual.usd` 均不是当前任务引用的机器人。旧的 `/home/agilex/qhx/.interndata/run-cup-plate-desk.sh` 仍通过兼容 YAML 加载新任务。

## 比较依据与证据层级

1. 真正参与物理仿真的模型：`workflows/simbox/assets/split_aloha_mid_360/robot.usd`，已直接读取 USD 中的关节、质量和驱动属性。
2. 规划器模型：`workflows/simbox/curobo/src/curobo/content/configs/robot/piper100_{left,right}_arm.yml`，引用 `content/assets/robot/piper100/piper100.urdf`。
3. 本机 ROS 参考：`/home/agilex/piper_ros/src/piper_description/urdf/piper_description.urdf`。
4. 本机 SDK 参考：`/home/agilex/qhx/RL/robotcooking/sdk/piper_sdk/piper_param/piper_param_manager.py`。
5. 本机实机项目设置：`/home/agilex/qhx/RL/piper_teleop_ws/src/piper_master_slave/config/policy_validation.yaml`，以及 `robotcooking/device_manager/arm/piper.py`。

本地 SDK/ROS 中的默认限制不等于已从图片中这台机器的控制器读取的限制。实机项目配置也是软件设置证据，不替代该序列号设备的实测或反馈。

## 关节限位比较

单位为度，四舍五入显示；USD 实际属性单位为度，URDF 限位单位为弧度，已转换。

| 关节 | PhysX USD | CuRobo URDF | 本机 ROS/SDK 参考 |
| --- | --- | --- | --- |
| J1 | −150～150 | −150～150 | −150～150 |
| J2 | 0～179.91 | **−5.73～179.91** | 0～179.91 |
| J3 | −154.53～0 | **−154.53～5.73** | −170～0 |
| J4 | −104.97～104.97 | −104.97～104.97 | −100～100 |
| J5 | −69.90～69.90 | −69.90～69.90 | −69.90～69.90 |
| J6 | −179.91～179.91 | −179.91～179.91 | −120～120 |

观察到的内部差异：规划器允许 J2 小于 0、J3 大于 0，但物理模型禁止这些范围。2026-10-28 进一步核对发现，本地规划 URDF 的 Git blob 哈希 `e1ab7b064ee03f34855ecf94376e6e3e78db6c94` 与 InternData 官方资产下载清单一致，限位并非本次修改。文件没有说明放宽 0.1 rad 的设计意图，数值容差/求解余量是可能解释，不能把这个猜测当作已确认设计。CuRobo 此配置的 `position_limit_clip` 默认为 0，没有发现自动抵消该余量的设置。

此前直接认定必须收紧的判断过早；当前先保留原配置。应关注规划结果是否真正使用了 USD 范围外的关节角，以及是否造成执行偏差。零位附近已观察到约 0.03° 的 PhysX 数值越界，因此将规划边界直接设为严格 0 也可能影响起点有效性检查。

J3、J4、J6 与本机参考也不同，不能认为整套限位已对齐实机。J6 尤其需要区分版本：官方当前 SDK 说明旧默认保留 ±120°，较新预设可能允许 ±180°，因此不能仅凭旧软件默认就判定本机硬件只能转 ±120°。[官方 SDK 默认值及版本说明](https://raw.githubusercontent.com/agilexrobotics/piper_sdk/master/piper_sdk/piper_param/piper_param_manager.py)

用户随后确认实机 J6 应允许 ±180°，本任务按此保留当前 J6 范围。

本机遥操作配置中的全部关节 ±180° 有明确“宽泛默认、应替换”的注释，不能用它作为六个关节的真实物理限位。

## 连杆尺寸与坐标系

两套本地 URDF 的主要关节间距为：

| 间距 | 仿真规划模型 | 本机 ROS 模型 |
| --- | --- | --- |
| J1 原点高度 | 123.000 mm | 123.000 mm |
| J2→J3 原点距离 | 285.031 mm | 285.030 mm |
| J3→J4 原点距离（含横向偏置） | 251.714 mm | 251.712 mm |
| J5→J6 原点距离 | 91.011 mm | 91.000 mm |

因此主要几何尺寸基本吻合。J3→J4 的 251.7 mm 是向量长度，不应与官方 MDH 的单一轴向参数 250.75 mm 混为一谈。

两套模型的各级坐标表达不同；零位工具坐标约相差绕末端 Z 轴 90°。零位末端位置差约 1.42 mm。在 20 组共同允许范围的随机关节角中，扣除零位估计的固定工具变换后，最大位置残差约 2.23 mm、姿态残差约 0.0048°。这些是模型间数值比较，不是实机标定精度；不能直接把两个模型的末端坐标值混用。

固件还会影响零位：官方说明 S-V1.6-3 前后的 J2/J3 坐标约定有 2° 偏移，需要匹配相应 URDF。[官方 ROS 版本说明](https://github.com/agilexrobotics/piper_ros#0-注意urdf版本)

## 夹爪

- 当前任务 `gripper_max_width=0.10 m`，USD 两指各 50 mm，规划 URDF 同样各 50 mm，总开度为 100 mm。
- 本机实机遥操作 YAML 明确写了物理行程 100 mm，并设置 `gripper_max_range_config: 100`、`external_command_gripper_max_mm: 100.0`；robotcooking 也默认 `gripper_range=0.1`。因此当前仿真开度与本机实机项目配置一致。
- 旧 ROS/SDK 默认对应 70 mm，不能不区分硬件配置就把当前任务改成 70 mm。但软件配置仍不足以证明图片中该设备实际行程，最后应通过控制器反馈或测量确认。
- 任务 `tcp_offset=0.12 m` 是抓取姿态换算用的工具偏移，不是臂展或夹爪总长。指关节原点距 `link6` 约 135.03 mm；两者不能直接作同一量比较。实机相机、指尖、安装板和 TCP 坐标需单独标定。

## 动力学与速度

- 当前 USD 一条臂（arm_base、link1～8）质量合计约 **1.362 kg**；本机标准 ROS 描述包括夹爪的质量合计约 **4.67 kg**。部件与工具配置不同，但如此大的差别说明不能认为质量、惯量已经对齐。官方产品页标称标准臂本体约 4.2 kg、负载 1.5 kg、工作半径 626 mm，这些指标也不能仅凭任务成功来证明已在模型中正确复现。[官方产品参数](https://global.agilex.ai/products/piper)
- 规划 URDF 和原始 USD 关节速度约为 **3 rad/s**。`workflows/simbox/core/robots/split_aloha.py:48` 在初始化时调用 `set_max_joint_velocities([500.0] * 12)`，将运行时允许上限设为 **500 rad/s**，不是 500°/s。允许上限不等于当前实际运行速度；已检查的轨迹实际速度较低，按用户要求保留此上限，当前不作为修改项。
- USD 转动关节刚度为 `1e7`、阻尼为 0；左指 J7 驱动 `maxForce=400`，J8 为 10。这里是仿真驱动参数，不能当作实机伺服参数或标称夹持力。
- 仿真包含 Aloha 双臂支架、升降和安装位姿；图片仅能识别单条 Piper，无法确认当前双臂间距、台面相对高度与真实安装一致。

## 当前成功轨迹的边界检查

对已保存的 457 帧左臂轨迹作离线检查：

| 关节 | 最小角度 | 最大角度 |
| --- | --- | --- |
| J1 | −9.507° | 26.952° |
| J2 | −0.028° | 129.415° |
| J3 | −79.140° | 0.030° |
| J4 | −0.001° | 60.516° |
| J5 | −69.551° | 0° |
| J6 | −45.799° | 0° |

J2/J3 在零位附近有约 0.03° 的仿真数值越界；其余范围未超过本机旧参考限位。J5 已较接近其 ±70° 参考边界。以 30 Hz 位置差分估计，六关节最大速度约为 `[0.432, 1.410, 0.903, 0.585, 0.546, 0.415] rad/s`，实际这条轨迹没有用到 500 rad/s。

上述结果不构成这条轨迹可直接在实机执行的确认：固件、零位、工具坐标、场景安装及控制器限位尚未逐项与实机反馈核对。

## 建议处理顺序

1. 当前优先定位抓取候选与执行阶段的规划失败，不因文件参数不同直接修改机器人限位。
2. 保留运行时 500 rad/s 上限、J6 约 ±180°、现有质量及有本机项目证据的 100 mm 开度。
3. 工具轴表达差异在实机接口中转换；部署前核对单臂基坐标、关节零位、TCP 及真实桌面位置。
4. 采集时保留原始 LMDB、相机与动作步号、采样间隔及模型/任务配置，后续离线转换 LeRobot。当前末端动作转换已独立于仿真采集流程。
5. 如果以后需要真实接触力、负载和动力学迁移，再对齐质量、惯量、摩擦与驱动响应。

本次仅给出审查结果，尚未更改这些机器人参数，避免依据不完整的铭牌信息覆盖已验证的任务配置。
