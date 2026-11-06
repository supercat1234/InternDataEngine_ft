# π0.5 杯子放盘子任务：仿真闭环测试说明

单实例连续测试和多实例并行测试的完整操作见第15节。当前45秒上限、中断丢弃、精简报告与单条复现命令见第18节。

本文对应 2026-11-03 已跑通的固定场景测试。项目根目录为 `/home/agilex/qhx/InternDataEngine`，以下命令均从该目录执行。

## 1. 测试目标与当前范围

将微调后的 π0.5 接入 Isaac Sim：读取三路相机与机器人反馈状态，模型预测动作，仿真执行一小段，再读取新观测并重新推理。

默认单轮配置为固定杯盘位置、光照、材质和相机安装位姿；可开启随机化，也支持单实例/多实例连续测试，见第14～15节。腕部相机仍随机械臂运动。左右臂均由模型的14维动作控制；测试不创建专家控制器、技能、规划器或训练数据记录器，不改动专家数据生成入口。

这是同步闭环：等待推理时不推进仿真物理时间。当前验证了动作接线与一轮固定布局任务，尚未评估随机布局成功率、异步推理或实机实时表现。

## 2. 项目相关目录

```text
/home/agilex/qhx/InternDataEngine/
├── local_configs/
│   ├── cup_plate_pi05_eval/
│   │   └── eval.yaml                    # 测试专用参数
│   └── cup_plate_desk/
│       ├── task.yaml                    # 复用的基础任务/机器人/相机配置
│       ├── arena.yaml                   # 复用的桌面、地面和材质配置
│       └── cameras/                     # 外部相机内参配置
├── scripts/
│   ├── run_cup_plate_pi05_server.sh      # 推理服务启动入口
│   ├── run_cup_plate_pi05_eval.sh        # 仿真测试启动入口
│   ├── run_cup_plate_pi05_batch.sh       # 每路固定seed、连续多轮的批量入口
│   ├── isaac_python.sh                  # Isaac Python 环境包装器
│   ├── eval/
│   │   ├── serve_pi05.py                # 本地 HTTP 模型服务
│   │   ├── rollout_cup_plate_pi05.py     # 场景加载与闭环执行
│   │   ├── batch_cup_plate_pi05.py       # 多实例调度、超时及批次统计
│   │   ├── scene_random_stream.py       # 跨进程延续各路场景随机数状态
│   │   ├── placement_evaluation.py      # 测试专用杯底支撑判据及失败诊断
│   │   ├── cup_plate_adapter.py         # 状态、图像、动作适配
│   │   ├── eval_reporting.py            # 编号、H.264 转码、日志与汇总表
│   │   ├── video_recording.py           # 实时分段 H.264 编码
│   │   └── repair_interrupted_video.py  # 恢复旧版未收尾的 mp4v 视频
│   ├── deploy/
│   │   └── pi05_plain_server.py         # 复用的模型加载/前后处理/HTTP 客户端
│   ├── preview_cup_plate_desk.sh         # 只查看场景的入口
│   ├── preview_scene.py                 # 暂停物理的场景与相机预览
│   ├── run_cup_plate_desk.sh            # 原专家数据生成入口
│   └── run_cup_plate_parallel.sh        # 原专家数据并行生成入口
├── workflows/
│   ├── simbox_dual_workflow.py          # 复用 preview_only 场景初始化路径
│   └── simbox/
│       ├── assets/                     # 场景和机器人资产
│       └── core/utils/
│           └── placement_validation.py # 专家数据生成原有判据，评测不再调用
├── tests/unit/
│   └── test_pi05_eval_adapter.py        # 动作映射、边界、图像/状态测试
├── docs/
│   └── cup_plate_pi05_eval.zh-CN.md     # 本说明
└── output/
    ├── pi05_eval/                       # 旧测试历史结果（保留）
    └── lerobot_v3/
        └── cup_plate_v07_and_102914/    # 微调数据集，rollout 不直接读取
```

模型文件位于仓库外：

```text
/home/agilex/qhx/output/pi05_simandreal/checkpoints/004000/pretrained_model/
├── config.json
├── model.safetensors
├── policy_preprocessor.json
├── policy_postprocessor.json
├── policy_preprocessor_step_3_normalizer_processor.safetensors
├── policy_postprocessor_step_0_unnormalizer_processor.safetensors
└── tokenizer/
```

测试加载 `pretrained_model`，不是 checkpoint 根目录。训练数据集不参与在线推理；输入规范与该数据集对齐，归一化参数从 checkpoint 的处理器文件恢复。

## 3. 主要脚本及作用

| 文件 | 作用 |
|---|---|
| `run_cup_plate_pi05_server.sh` | 使用 LeRobot 环境启动服务，默认 CUDA、端口 8097，启用 Hugging Face 离线模式 |
| `serve_pi05.py` | 检查绝对 14 维动作顺序，严格加载权重与处理器，提供 `/health` 和 `/infer` |
| `run_cup_plate_pi05_eval.sh` | 切换到项目目录，使用 Isaac Python 启动测试 |
| `run_cup_plate_pi05_batch.sh` / `batch_cup_plate_pi05.py` | 每实例测试 TOTAL 次；每路固定 seed 并延续随机数序列，共用模型服务 |
| `scene_random_stream.py` | 保存/恢复 Python 和 NumPy 场景随机数状态，每路独立推进 |
| `placement_evaluation.py` | 按实际杯底尺寸判断支撑，输出距离、角度、速度和未通过条件 |
| `rollout_cup_plate_pi05.py` | 从基础场景生成固定配置，采集观测、请求推理、施加关节位置目标、判断成功并保存结果 |
| `cup_plate_adapter.py` | 将 RGB 图像补边至 256×256，组装 14 维反馈，将动作映射至实际关节索引和夹爪位移 |
| `pi05_plain_server.py` | 加载 checkpoint 自带 tokenizer/归一化和反归一化处理器；提供 HTTP 推理客户端 |
| `eval.yaml` | 设置 checkpoint、固定杯盘位置、种子、执行帧数和最大测试步数 |

模型服务和 Isaac Sim 使用不同的 Python 环境，通过本机 HTTP 通信。默认推理 Python 为 `/home/agilex/miniforge3/envs/lerobot/bin/python`，当前本地 LeRobot 为 0.6.2；仿真环境由 `scripts/isaac_python.sh` 配置。

## 4. 输入、输出与控制约定

输入：

- `observation.state`：14 维实际反馈状态。
- `observation.images.TOP_CAM`：外部相机。
- `observation.images.LEFT_WRIST_CAM`、`RIGHT_WRIST_CAM`：左右腕部相机。
- 指令：`pick up the paper cup and put it at the plate`。

图像为 RGB，按长宽比缩放并居中补黑边到 256×256。客户端发送原始状态和图像，服务执行 checkpoint 配套前处理，输出反归一化后的动作。

动作顺序：
```te
[left_J1, left_J2, left_J3, left_J4, left_J5, left_J6, left_gripper,
 right_J1, right_J2, right_J3, right_J4, right_J5, right_J6, right_gripper]
```
六个关节使用绝对角度，单位 rad。夹爪为 0～1，对应总开度 0～100 mm；仿真主动单指位置目标为 `gripper × 0.05 m`。动作通过位置目标驱动，不逐帧传送关节位置。
当前 checkpoint 每次预测 50 帧，默认只执行前 10 帧，然后根据新观测重新推理。物理步长为 1/30 秒。没有额外插值和平滑；小于等于 5°的关节限位越界裁剪到 USD 边界，更大的越界停止。夹爪归一化输出先裁剪至 [0,1]，再检查物理限位；超过物理限位 1 mm 以上停止。原始预测与实际目标均保留。

## 5. 测试流程与命令

### 5.1 检查 GPU
```b
cd /home/agilex/qhx/InternDataEngine
nvidia-smi
```
模型与仿真共用 GPU。

### 5.2 启动模型服务：终端 A

```bash
cd /home/agilex/qhx/InternDataEngine
bash scripts/run_cup_plate_pi05_server.sh
```

等待权重严格加载完成，以及：

```text
PI05_READY checkpoint=.../004000/pretrained_model port=8097
```

检查服务：

```bash
curl --noproxy '*' http://127.0.0.1:8097/health
```

响应应包含 `action_dim: 14`、`chunk_size: 50`、`absolute_actions: true` 及正确 checkpoint 路径。

### 5.3 短程接线检查：终端 B

```bash
cd /home/agilex/qhx/InternDataEngine
bash scripts/run_cup_plate_pi05_eval.sh --headless --max-steps 30
```

这会运行 30 帧、通常触发 3 次在线推理。正常到达步数上限时，`reason` 为 `max_steps`；这只是短程检查结束，不能据此认为策略任务失败。

### 5.4 完整固定场景测试

显示仿真窗口：

```bash
bash scripts/run_cup_plate_pi05_eval.sh
```

不显示窗口：

```bash
bash scripts/run_cup_plate_pi05_eval.sh --headless
```

放置成功前默认最多运行1350帧，即45秒仿真时间；放置成功后继续执行模型动作和录像，固定追加 5 秒后结束，不再等待静止。成功后的录制阶段可以超过原 `max_steps` 上限，但固定增加 150 帧。实际墙钟时间还包含加载、渲染、推理及文件保存。

需要显式设置测试长度和闭环间隔时：

```bash
bash scripts/run_cup_plate_pi05_eval.sh \
  --headless \
  --max-steps 1350 \
  --execute-steps 10
```

`execute-steps` 必须为 1～50；数值越小，重新观测和推理越频繁，但推理开销也越大。

### 5.5 指定输出位置与保存控制台日志

默认自动分配日期编号；如显式传入 `--output`，目录必须尚不存在，父目录可存在（此参数会覆盖默认命名）：

```bash
set -o pipefail
bash scripts/run_cup_plate_pi05_eval.sh \
  --headless \
  --output /home/agilex/qhx/evaluation/sim/test/manual_test_001 \
  2>&1 | tee /tmp/pi05_manual_test_001.log
```

主脚本自动保存三路视频、轨迹、结果 JSON 和 `rollout.log` 控制台日志，并追加到 `/home/agilex/qhx/evaluation/sim/test/summary.csv`。上面的 `tee` 只是额外保存一份日志，可以省略。

### 5.6 结束测试

测试达到成功判据或最大步数后，仿真自动退出。在终端 A 按 `Ctrl+C` 关闭模型服务、释放其显存。模型服务不会随着仿真退出而自动关闭。

## 6. 修改测试配置

`local_configs/cup_plate_pi05_eval/eval.yaml` 当前内容：

```yaml
source_task: local_configs/cup_plate_desk/task.yaml
checkpoint: /home/agilex/qhx/output/pi05_simandreal/checkpoints/004000/pretrained_model
cup_position: [-0.27, -0.07, 0.0]
plate_position: [-0.06, 0.06, 0.0]
yaw_deg: 0
seed: 42
fps: 30
execute_steps: 10
max_steps: 1350
output_root: /home/agilex/qhx/evaluation/sim/test
```

杯盘位置为桌面中心的偏移，单位米，支撑采样器负责放到桌面上，第三项 0 不表示世界高度为零。`yaw_deg` 同时固定杯盘的区域 yaw。保持 `fps: 30` 与训练数据的时间尺度一致。

可复制该文件另建测试配置：

```bash
bash scripts/run_cup_plate_pi05_eval.sh \
  --config local_configs/cup_plate_pi05_eval/eval.yaml
```

切换模型时，必须同时修改测试配置里的 `checkpoint`，并重启服务：

```bash
bash scripts/run_cup_plate_pi05_server.sh \
  --checkpoint /绝对路径/新checkpoint/pretrained_model
```

测试入口会检查服务返回的 checkpoint 与配置一致。新模型仍须满足相同的输入、绝对 14 维动作顺序和图像规范。

修改服务端口时，两端保持一致：

```bash
# 终端 A
bash scripts/run_cup_plate_pi05_server.sh --port 8098

# 终端 B
bash scripts/run_cup_plate_pi05_eval.sh --url http://127.0.0.1:8098
```

如需更换推理 Python：

```bash
PI05_PYTHON=/绝对路径/python bash scripts/run_cup_plate_pi05_server.sh
```

固定场景在运行时由基础任务配置派生：光强 1000、白光 6000 K、光照方向 0°、第一种标准木纹、黑色大理石地砖，关闭相机安装位姿扰动。因此未来对基础任务文件的其他修改仍会被测试继承；每次输出的 `task.yaml` 与 `arena.yaml` 可用于核对实际加载场景。

## 7. 其他调试模式

仅采集初始观测，不推理、不施加策略动作，不需要模型服务：

```bash
bash scripts/run_cup_plate_pi05_eval.sh --headless --observe-only
```

该模式保存三路 PNG 图片（添加 `--save-debug-arrays` 才额外保存初始输入数组）后退出，不再生成只有一帧的 MP4，不会一直停留在交互预览窗口。如果只想查看本轮固定场景和相机，可使用该轮生成的任务文件：

```bash
bash scripts/preview_cup_plate_desk.sh \
  --task output/pi05_eval/20261103_165617_325797/task.yaml \
  --camera external
```

预览窗口暂停物理，不运行策略。`--camera` 可选 `external`、`hand_left`、`hand_right` 或 `perspective`。

回放已保存的模型预测，仅验证动作应用、不重新推理：

```bash
bash scripts/run_cup_plate_pi05_eval.sh \
  --headless --max-steps 10 \
  --replay-actions output/pi05_eval/20261103_164450_548797/smoke_predicted_actions.npy
```

回放文件为 `[T,14]` 的 NumPy 数组；应有足够帧数覆盖测试，末次分块也须满足执行块长度。结果标记为 `mode: replay`，不能代替真实闭环评估。

## 8. 结果文件与判定

```text
/home/agilex/qhx/evaluation/sim/test/20261103_001/
├── run.json                    # 参数、服务信息、checkpoint、语言指令
├── task.yaml                   # 本轮固定任务配置
├── arena.yaml                  # 本轮固定桌面/地面配置
├── initial_observation.npz     # 仅 --save-debug-arrays：初始状态和三路图像
├── selected_joint_limits.npy   # 仅 --save-debug-arrays：USD 限位
├── chunk_000000.npz            # 仅 --save-debug-arrays：观测和预测块
├── chunk_000010.npz
├── ...
├── TOP_CAM.mp4                 # 外部相机
├── LEFT_WRIST_CAM.mp4          # 左腕相机
├── RIGHT_WRIST_CAM.mp4         # 右腕相机
├── trajectory.json             # 每帧反馈、原始预测、实际目标、杯盘位姿和判定
├── rollout.log                 # 自动捕获 Python 与原生仿真标准输出/错误
└── result.json                 # 精简结果：seed、两级成功、原因、仿真时长、失败条件
```

视频按 30 fps 的仿真时间保存，并转为 H.264/yuv420p，启用 faststart；保存时检查帧数与解码。仅观测模式以三路同名 PNG 代替 MP4。`chunk_*.npz` 中的图像是该次推理输入，`actions` 是该次完整预测；`trajectory.json` 中的 `applied` 是映射到主动关节后的实际目标，夹爪单位为单指米制位移，不能直接与归一化夹爪预测相减。

查看某轮结果：

```bash
cat output/pi05_eval/20261103_165617_325797/result.json
```

常见字段：

| 字段/值 | 含义 |
|---|---|
| `mode: closed_loop` | 使用新观测持续调用模型服务 |
| `mode: replay` | 离线动作回放 |
| `success: true` | 杯底中心位于盘子外轮廓内，且满足其余稳定放置判据；总成功率按此统计 |
| `strict_success: true` | 完整杯底位于盘内平坦底部，且满足其余相同判据 |
| `reason: stable_placement` | 已达成稳定放置，结束时仍满足放置条件 |
| `stop_reason: post_success_timeout` | 成功后额外录像达到 5 秒 |
| `reason: max_steps` | 达到本轮最大步数 |
| `reason: error` | 出现异常，查看 `error` 和控制台日志 |
| `returned_home` | 成功时是否已回到初始双臂关节角，仅在成功结果中写入 |

当前为两级成功，见第16节：普通success要求杯底中心距盘中心≤77 mm；strict_success要求完整杯底位于平坦支撑圆内，对应中心偏移≤21.5 mm。两级都检查倾斜后的杯底最低点高度、杯子相对盘面倾角：普通≤15°、严格≤8°、盘子自身倾角≤5°、杯盘线速度≤0.015 m/s、角速度≤0.15 rad/s，并要求TCP离杯底超过120 mm，连续满足15帧。释放条件是几何代理判据，没有使用接触传感器证明。该评测判据与原专家数据生成判据分开。

成功后继续在线推理、执行和录像，不调用专家回零。无论机械臂静止与否，固定追加 5 秒仿真时间后结束；30 Hz 下为 150 帧。若模型自行回零，会被录制。`returned_home` 检查最终双臂关节角是否回到初始角度，不包含夹爪开度。若后续动作破坏放置，结束时不再满足放置条件，会记录 `placement_lost`，并保留首次成功时刻。

## 9. 已完成验证与复现结果

2026-11-03 已完成：4 项适配器单元测试、权重严格加载、真实仿真观测前向推理、离线动作应用、30 帧实时闭环和一轮完整稳定放置测试。

完整闭环输出：`output/pi05_eval/20261103_165617_325797/`。

| 指标 | 本轮结果 |
|---|---|
| 模式 | 实时同步闭环 |
| 放置结果 | 成功 |
| 执行步数 | 224 |
| 仿真时间 | 约 7.47 秒 |
| 在线推理次数 | 23 |
| 平均模型推理耗时 | 约 0.148 秒 |
| 杯子距盘心水平偏移 | 约 3.28 mm |
| 杯子相对盘面倾角 | 约 0.78° |
| 连续稳定帧数 | 15 |
| 成功时是否回零 | 否 |

该历史测试额外保存了 `validation.json`（人工整理的统计）。更新后的脚本会自动生成 `rollout.log`，并将耗时和推理次数等常用统计直接写入 `result.json`；不另行自动生成 `validation.json`。一次固定布局成功不能代表多次测试或随机布局的成功率。

适配器单元测试命令：

```bash
/home/agilex/miniforge3/envs/lerobot/bin/python \
  -m unittest discover -s tests/unit -p test_pi05_eval_adapter.py
```


## 10. 每轮自动保存（更新后的默认行为）

正常成功或达到45秒上限才保留本轮结果。Ctrl+C、关窗、停止仿真、运行异常及墙钟超时的当前目录删除，不写入summary.csv、不计入成功率；已完成结果保留。日期编号采用Asia/Shanghai，原子创建支持并发；中断条目的编号可能被后续测试重新使用。

`result.json` 已精简，字段见第18节。checkpoint和配置见run.json，逐帧位姿和判据见trajectory.jsonl/trajectory.json，三路视频固定命名；耗时、推理统计及回零状态仍可在根目录summary.csv查阅。

所有新测试另追加一行至 `/home/agilex/qhx/evaluation/sim/test/summary.csv`，可用表格软件打开。`mode` 区分在线闭环、回放和仅观测；`status` 为 `success`、`not_completed`、`error`、`interrupted` 或 `diagnostic`。统计策略成功率时应筛选 `mode=closed_loop`，并排除专门的短程接线检查，不能把回放和仅观测计入策略评估。旧测试不会自动补入索引。


## 11. NPZ 与零秒视频说明

`.npz` 是 NumPy 的压缩数组容器，可以在同一个文件里保存多个具名数组；它不是视频，也不是模型 checkpoint。这些调试数组不是 rollout 所必需的，现已默认关闭；历史测试目录中的 109 个 NPZ 调试文件已按要求删除。若特意需要保存，添加 `--save-debug-arrays`。

- `initial_observation.npz`：初始 `observation.state`（14 维反馈状态）和三路 RGB 图像（256×256×3）。
- `chunk_000000.npz` 等：该次在线推理的输入状态、三路图像，以及 `actions`（当前模型为 50×14 的完整预测块）。
- `selected_joint_limits.npy`：单个 14×2 数组，对应各主动关节的 USD 上下限；`.npy` 保存单个数组，`.npz` 可以保存多个数组。

使用推理环境查看数组名称和形状（替换实际结果目录）：

```bash
/home/agilex/miniforge3/envs/lerobot/bin/python - <<'PYCODE'
import numpy as np
with np.load('/home/agilex/qhx/evaluation/sim/test/20261103_001/chunk_000000.npz', allow_pickle=False) as data:
    for key in data.files:
        print(key, data[key].shape, data[key].dtype)
PYCODE
```

检查旧记录后发现，最近一次 `observe_only` 只有 1 帧（约 0.033 秒），播放器可能显示 0 秒；它只有初始画面，没有动作过程。旧完整闭环记录则有 224 帧、7.466667 秒。更新后仅观测保存 PNG，闭环与回放使用兼容性更好的 H.264 视频。短程测试仍保留真实时长，例如 10 帧仅 0.33 秒，不通过重复帧人为拉长。

新位置仅用于后续运行；旧 `output/pi05_eval/` 数据未迁移。新目录分配与转码已经在临时目录测试；本轮向新根目录运行仿真的验证因自动权限审核两次超时未执行，并非仿真报错。


## 12. 放置成功后继续录制

旧的 224 帧视频在第 224 帧判定放置成功后立即停止，因此时长约 7.47 秒，未录到后续可能的撤退或回零。这是旧终止逻辑，不是视频截断故障。

现在成功判据与停止判据分开：成功后仍持续模型闭环、执行和录像，固定再录满 5 秒后结束。`result.json` 保存 `success_step`、`success_simulation_seconds`、`stop_reason`、`post_success_steps`、`post_success_seconds`、`placement_stable_at_end` 和最终 `returned_home`，以区分“何时放好”和“何时结束录像”。未成功的测试仍按原最大步数结束。

对应 `eval.yaml` 参数：

```yaml
post_success_seconds: 5.0  # 统一规则，程序固定为5秒
```

固定五秒尾录、成功时刻锁定及边界帧数已通过单元测试。本次未重新运行完整 GPU 闭环；已有七秒视频不包含后续动作，无法补回历史中未录制的帧。


## 13. 中断录像修复（20261103_001）

这轮实际保存了数百帧视频数据，但三路原 MP4 缺少 `moov` 索引，`result.json` 也没有生成，日志停在第 1070 步附近。可判断正常收尾未完成，不能仅凭这些文件确定是关窗、崩溃还是强制退出。与此前单帧 `observe_only` 是不同问题。

已从原 MPEG4 数据恢复可完整解码的画面并重新编码为 H.264，保留原文件，在原目录新增：

| 文件 | 恢复帧数 | 30 fps 时长 |
|---|---:|---:|
| `TOP_CAM_recovered.mp4` | 792 | 26.40 秒 |
| `LEFT_WRIST_CAM_recovered.mp4` | 985 | 32.83 秒 |
| `RIGHT_WRIST_CAM_recovered.mp4` | 893 | 29.77 秒 |

`video_recovery.json` 保存恢复信息。最后不完整的视频包已丢弃；各路未刷盘帧数不同，恢复后长度不同，不能伪造为完整运行录像或补写成功结论。

后续视频改用独立 FFmpeg 进程实时写 H.264 fragmented MP4，录制时就带有初始化索引并按片段写入，而非依赖最后一次 `VideoWriter.release()` 才生成整个文件的索引。正常结束后仍转换为常规可播放 MP4 并验证。异常中断可能丢失最后片段，但避免旧方式整个视频不可读的问题。

另外每帧追加 `trajectory.jsonl` 并更新 `progress.json`（帧数、当前判定、运行状态），正常结束仍生成完整 `trajectory.json` 和 `result.json`。没有 `result.json` 的历史运行不应当作已正常完成。新增真实编码和父进程未经清理退出的测试，均验证保留了 65 帧可解码视频；全部 13 项相关测试通过。


## 14. 五秒尾录、简化保存与开启随机化（当前默认）

成功后固定额外录制 **5 秒仿真时间**（30 Hz 下 150 帧），不再以静止作为结束条件。已有视频不会因此被截短。

默认不生成 `.npz` 或 `.npy`，保留三路视频、结果、配置、日志、轨迹和进度。数组文件用于精确检查/重放某次推理输入与完整预测块，并不是测试必须数据。如果调试时需要，可运行 `bash scripts/run_cup_plate_pi05_eval.sh --save-debug-arrays`。

开启全部随机化：先照常启动模型服务并等待 `PI05_READY`，再运行：

```bash
cd /home/agilex/qhx/InternDataEngine
bash scripts/run_cup_plate_pi05_eval.sh --randomize --seed 42
```

无界面运行加 `--headless`。多轮测试使用不同种子，模型服务保持开启：

```bash
for seed in 42 43 44; do
  bash scripts/run_cup_plate_pi05_eval.sh --headless --randomize --seed "$seed"
done
```

这是依次运行多个独立实例，每轮自动创建新的日期编号目录；不是多实例并行。相同种子用于复现同一场景采样，不代表模型采样轨迹也完全相同。

如果只开启某些随机化，修改 `local_configs/cup_plate_pi05_eval/eval.yaml`：

```yaml
randomization:
  layout: false
  lighting: true
  table: true
  floor: true
  camera: false
```

然后不加 `--randomize`，让配置按分项生效：

```bash
bash scripts/run_cup_plate_pi05_eval.sh --seed 43
```

`--randomize` 会把所有五项强制开启；`--seed` 只改变采样种子，本身不打开任何随机化。默认种子固定为 42，不换种子不会自动得到新的布局。

| 配置项 | 当前启用后的范围 | 范围来源 |
|---|---|---|
| `layout` | 杯子 XY：[-0.35,-0.15] 到 [-0.20,0.10] m；盘子 XY：[-0.10,-0.10] 到 [0,0.10] m；yaw ±180° | `local_configs/cup_plate_desk/task.yaml` 的 `regions` |
| `lighting` | 光强 0.8～1.2 倍，白/暖光，方向扰动 ±5°/±5°/±20° | 同文件 `env_map.photometric_randomization` |
| `table` | 木纹1占70%、木纹2占30%；各自标准/浅色等概率 | `local_configs/cup_plate_desk/arena.yaml` |
| `floor` | 黑色大理石40%，其余四种各15% | 同上 `floor.texture` |
| `camera` | 外部相机 XYZ 各 ±20 mm、旋转角最大2°；腕部安装位姿固定 | 基础任务外部相机配置 |

每轮只在初始化时采样一次，过程中保持该轮设置；机械臂运动仍会带动腕部相机。修改范围在对应基础任务/场景配置中完成，测试入口不会调用专家规划筛选随机布局。默认 HDR 贴图和机器人初始姿态不随机。

结果位置继续使用 `/home/agilex/qhx/evaluation/sim/test/YYYYMMDD_编号/`，`run.json` 会记录生效的分项开关和种子，`task.yaml`/`arena.yaml` 保存本轮使用的范围，采样光照和材质名称可在 `rollout.log` 中查看。配置生成及16项相关单元测试通过，本次未重新执行 GPU 随机布局测试。


## 15. 不停止的原因与批量测试（最新）

有头和无头模式使用相同的停止规则。检查 `20261103_005` 发现，机械臂撤退/回零后夹爪闭合，旧判据要求“夹爪持续张开至少35 mm”，因此即使杯子在盘内也未触发成功，日志没有 `PLACEMENT_SUCCESS`。该轮后来出现 `joint_state is None` 错误退出，不能归因为录像模式。

现在取消“撤退后夹爪必须保持张开”的限制，以TCP已远离杯子、杯子稳定在盘面作为几何释放判据。首次连续15帧普通success后固定再录制5秒，然后退出；strict_success不阻止退出。未成功仍最多1350帧。旧版脚本已启动的进程不会热更新，需要下次启动才使用新逻辑。`PLACEMENT_SUCCESS` 表示开始五秒尾录，`RESULT_SAVED` 表示视频和报告已保存。

### 15.1 公共准备：启动一个模型服务

单实例连续测试和多实例并行测试统一使用 `scripts/run_cup_plate_pi05_batch.sh`，调度实现为 `scripts/eval/batch_cup_plate_pi05.py`。两种模式都只需启动**一个**模型服务，并在整个批次运行期间保持开启。

终端A：

```bash
cd /home/agilex/qhx/InternDataEngine
bash scripts/run_cup_plate_pi05_server.sh
```

等待 `PI05_READY` 后，在终端B选择下列一种测试方式。如果该模型服务已经在运行，直接使用，不要重复启动同端口的服务。

### 15.2 单实例多次测试：同一路种子连续10次

```bash
cd /home/agilex/qhx/InternDataEngine
INSTANCES=1 TOTAL=10 BASE_SEED=100 \
bash scripts/run_cup_plate_pi05_batch.sh --randomize
```

含义：同一时间仅有一个仿真实例，共10次尝试，全部使用主种子100。每次退出后启动下一次，并恢复该路上次场景初始化后的随机数状态，继续采样新的杯盘位置、光照等；结果单独保存，无需手动重复启动模型服务。要连续100次，将 `TOTAL=10` 改为 `TOTAL=100`。

种子用于初始化一条随机数序列，不是每轮重新从序列开头采样。批量入口通过保存Python/NumPy的状态延续序列，因此每次不用换seed，也不会每轮重复同一个场景。如果关闭全部随机化，则仍是固定场景。

只预览单实例的任务和种子分配：

```bash
INSTANCES=1 TOTAL=10 BASE_SEED=100 \
bash scripts/run_cup_plate_pi05_batch.sh --dry-run
```

### 15.3 多实例多次测试：3路并行、每路100次

```bash
cd /home/agilex/qhx/InternDataEngine
INSTANCES=3 TOTAL=100 BASE_SEED=42 \
bash scripts/run_cup_plate_pi05_batch.sh --randomize
```

含义：最多同时3个无头仿真实例，每路100次，共300次尝试。各自完成一轮后立即开始下一轮，不必等待其他worker。

| worker | 固定主种子 | 尝试次数 |
|---|---|---:|
| 1 | 42 | 100 |
| 2 | 43 | 100 |
| 3 | 44 | 100 |

同一批次内各路主种子不同，各自延续独立随机数序列。单路第n轮的场景取决于该路前n−1轮的场景采样，不受其他worker完成先后顺序影响。不同序列仍可能偶然采样到相近布局，这与重复使用同一随机数状态不同。

多实例中日期编号目录的创建顺序由实际启动顺序决定；用批次 `summary.csv` 的 `worker`、`episode`、`seed` 字段对应每条结果。全局 `attempt` 编号按worker分段：worker1为1～100，worker2为101～200，worker3为201～300，因此日志编号不一定按完成顺序递增。

多个客户端共用同一模型服务，推理请求由服务顺序处理，避免加载三份模型。并行3路的实际显存需求尚未验证；如遇显存不足，将 `INSTANCES` 降为2或1即可。

只预览并行任务分配，不启动仿真、不写结果、不要求模型服务：

```bash
INSTANCES=3 TOTAL=100 BASE_SEED=42 \
bash scripts/run_cup_plate_pi05_batch.sh --dry-run
```

### 15.4 TOTAL 的新含义与重复批次

```bash
INSTANCES=3 TOTAL=10 BASE_SEED=42 \
bash scripts/run_cup_plate_pi05_batch.sh --randomize
```

以上是每路10次、三路共30次。`TOTAL` 现在表示**每实例的尝试次数**，失败也计入，不是成功目标。总尝试数为 `INSTANCES × TOTAL`。这改变了旧脚本的含义：`INSTANCES=3 TOTAL=100` 现在是300次，`TOTAL=300` 则是900次。

重新启动相同命令会从各路主种子的序列开头重新测试，便于复现实验；模型采样动作不保证完全相同。另开一个独立批次若不想使用相同主种子，需要自行换一组 `BASE_SEED`（如三路42～44之后使用45）。程序不跨独立批次自动分配种子。

### 15.5 参数、随机化与短程检查

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `INSTANCES` | 3 | 并行worker数；单实例连续测试设为1 |
| `TOTAL` | 100 | 每个worker的尝试次数，失败也计入 |
| `BASE_SEED` | 42 | 第一路固定种子；其余路分别加1，各轮不换种子 |
| `EPISODE_TIMEOUT` | 600 | 每次尝试的墙钟超时秒数，包含初始化、录制及退出 |
| `--randomize` | 不启用 | 强制开启全部五项场景随机化 |
| `--config` | 默认eval.yaml | 使用指定测试配置 |
| `--max-steps` | 默认1350 | 成功前最大物理步数，成功后仍固定追加5秒 |
| `--dry-run` | 不启用 | 仅输出调度计划，无需模型服务 |

如果只按 `eval.yaml` 中的分项开关启用随机化，去掉 `--randomize`。如果五项开关均为false，随机数序列继续推进也不会自动开启随机化。场景种子用于复现场景采样，不保证模型随机采样出的动作完全一致。

批量入口统一使用无头模式，不需要添加 `--headless`。需要显示窗口查看动作时，使用单轮入口：

```bash
bash scripts/run_cup_plate_pi05_eval.sh --randomize --seed 42
```

例如先做3轮、每轮30帧的单实例接线检查：

```bash
INSTANCES=1 TOTAL=3 BASE_SEED=42 \
bash scripts/run_cup_plate_pi05_batch.sh --randomize --max-steps 30
```

短程检查不是完整任务测试，不能将其成功率作为策略表现。每轮默认墙钟超时600秒，确有需要可用 `EPISODE_TIMEOUT=900` 调整。

### 15.6 单次与批次结果位置

两种批量模式使用相同的保存规则。每次尝试单独保存到：

```text
/home/agilex/qhx/evaluation/sim/test/YYYYMMDD_编号/
```

其中包含三路视频、`result.json`、轨迹、配置、日志和进度；默认不保存NPZ。批量测试另保存 `scene_rng_before.json` / `scene_rng_after.json`，记录该轮场景采样前后的Python/NumPy状态，便于定位、复现场景。所有正常完成报告的尝试还会追加到根目录的 `summary.csv`。

每个批次额外创建：

```text
/home/agilex/qhx/evaluation/sim/test/batches/<日期时间_进程号>/
├── plan.json           # 每路次数、批次总数、模型checkpoint、任务和种子分配
├── worker_1_scene_rng.json # 第1路最新场景随机数状态；其余路各有一份
├── attempt_001.log     # 第1次尝试的启动/退出日志
├── attempt_002.log
├── ...
├── summary.csv         # worker、episode、种子、success、strict_success、原因、退出码及目录
└── result.json         # 有效完成数、丢弃数、两级成功数与成功率；逐次明细见CSV
```

正常任务失败保留并继续下一轮；异常终止的当前目录和该条启动日志删除，不写入CSV、不计入分母。批次只记录丢弃数量discarded；success_rate=successes/finished，finished只含正常结束的成功和失败。无有效结果时成功率为null。没有自动补跑，故有效完成数可能少于请求次数；已持久化的场景随机数状态继续推进。

按 `Ctrl+C` 停止本批次创建的进程组，保留已完成结果，不终止外部启动的模型服务。批次完成后在终端A按 `Ctrl+C` 关闭模型服务即可。

已完成30项单元/集成测试，包括每路固定种子、跨进程随机数序列延续、两级成功及统计、旧CSV兼容、固定5秒停止，以及真实编码出的12秒回归视频；整批GPU测试需按上述命令运行。

## 16. 成功判据、误判原因与诊断

评测配置位于 `local_configs/cup_plate_pi05_eval/eval.yaml` 的 `success_criteria`，实现为 `scripts/eval/placement_evaluation.py`。当前版本标记为 `dual_success_v4`，成功分为两级：

| 结果字段 | 位置条件 | 用途 |
|---|---|---|
| `success` | 杯底中心位于盘子外轮廓内，即盘子坐标系XY距离≤77 mm | 总成功率和成功后5秒停止 |
| `strict_success` | 完整杯底位于盘内平坦底部，即中心偏移+杯底半径26.5 mm≤平坦支撑半径48 mm | 额外的严格成功率，不控制退出 |

外半径77 mm已从当前盘子USD模型顶点核对，外径为154 mm。严格成功是普通成功的子集。

两级都保留以下共同条件，各自必须连续满足0.5秒（30 Hz下15帧）：

| 检查 | 默认要求 |
|---|---|
| 杯底高度 | 杯底最低点距盘子原点的高度为5±4 mm；对倾斜杯底做最低边缘补偿 |
| 杯子倾角 | 普通success相对盘面≤15°；strict_success≤8° |
| 盘子倾角 | 相对世界竖直 ≤5° |
| 稳定速度 | 杯子、盘子各自线速度 ≤0.015 m/s，角速度 ≤0.15 rad/s |
| 机械臂撤离 | 左臂TCP距杯底中心 >120 mm；不限制回零和夹爪最终开度 |

首次普通success确认后继续执行模型动作并录制5秒，然后结束，即使strict_success从未满足，也不再等待。两级分别记录首次连续稳定时刻；结束时重新检查各级当前条件。若杯子离开平坦区但仍满足普通success，则 `success=true, strict_success=false`；若普通放置条件也丢失，则 `success=false, strict_success=false, reason=placement_lost`。

未成功为 `reason=max_steps`，默认最多1350帧/45秒；运行异常、墙钟超时、用户中断的当前条目丢弃。普通success倾角上限为15°（success_max_tilt_deg）；strict_success保留8°（max_tilt_deg）。位置、高度、速度、撤离和连续稳定等条件继续检查。

旧判据误用了杯口半径37.5 mm，中心偏移上限只有10.5 mm；倾斜时还直接检查杯底中心高度，可能把已接触盘面的杯底边缘误判为悬空。现在仅修正评测，不改专家生产的规划及校验。

以下为旧记录的末帧几何核查（仅比较位置、倾角、高度，未重算速度与连续稳定）：

| 记录 | 中心偏移 | 杯子倾角 | 新两级末帧几何结果 |
|---|---:|---:|---|
| `20261103_011` | 13.55 mm | 3.72° | 普通及严格均符合 |
| `20261103_012` | 19.53 mm | 3.09° | 普通及严格均符合 |
| `20261103_010` | 29.04 mm | 8.022° | 按当前15°普通判据几何符合，严格不符合 |
| `20261103_013` | 25.94 mm | 6.10° | 普通符合，严格不符合 |

旧轨迹没有记录杯盘速度，因此以上不能直接当作一次完整测试的成功结论；不回写旧报告、不直接重算历史成功率。新启动的测试记录两级结果；根目录summary.csv首次新增记录时会兼容升级列，旧记录的strict_success留空，表示未知。

以下为此前完整报告的诊断字段；当前精简result.json只保留failed_checks/strict_failed_checks，完整逐帧诊断仍在轨迹文件中：

- `placement_criteria_version: dual_success_v3`：用于区分两级判据与旧版。
- `success_criteria`：本轮生效的全部阈值。
- `final_placement.failed_checks`：普通success未通过条件，如 `center_in_plate`、`height`、`cup_tilt`、`tcp_clearance`。
- `final_placement.strict_failed_checks`：严格success未通过条件，额外包含 `footprint`。
- `final_placement.metrics`：实际中心偏移、高度误差、倾角、杯盘速度、TCP距离。
- `max_stable_frames` / `required_stable_frames`：最长连续满足帧数及要求。
- `strict_success_step`、`strict_success_simulation_seconds`、`max_strict_stable_frames`：严格成功的时刻与连续帧数。
- `stop_at_step`、`expected_video_seconds`：普通success确认时确定的停止帧及预计视频时长。
- `failed_check_frames`：整轮每项条件不满足的累计帧数，包含抓取前阶段，不能单独当最终失败原因。
- `trajectory.jsonl` / `trajectory.json` 每帧的 `placement`：完整逐帧诊断。

已经运行的进程继续使用启动时的旧代码；下一批启动后使用新语义和新判据。

## 17. 为什么视频是60秒，以及怎样确认修复

`20261103_010`～`20261103_015` 的报告均为 `success=false, reason=max_steps`，1800帧恰好是60秒。它们没有通过当时的放置条件，因此未进入成功后的5秒倒计时。`_007`～`_009` 已通过成功条件，其视频分别约12.8、12.87、11.5秒，说明编码器没有强制把所有视频扩展到60秒。`_016` 在16.27秒确认成功，21.27秒结束。

现在停止倒计时只依赖普通success，不再受完整杯底是否进入平坦区影响。以第210帧（7秒）确认普通success为例，第360帧结束，视频12秒；若第225帧（7.5秒）才完成0.5秒连续稳定确认，则第375帧结束，视频12.5秒。起点是全部条件连续满足后的确认时刻，不能按肉眼首次接近盘子的时刻计时。

撤离距离的计算也修正为末端USD真实世界变换加TCP偏移；此前把机器人根坐标与左臂基座坐标混用，会使TCP距离失真。此改动不调整120 mm阈值，也不改变模型输入、动作或专家流程。

日志中应出现：

```text
PLACEMENT_SUCCESS step=...; continue recording for 5 simulation seconds
RESULT_SAVED .../result.json
```

报告应满足 `post_success_steps=150`、`post_success_seconds=5.0`、`stop_reason=post_success_timeout`。若仍然为60秒，先看 `placement_criteria_version` 是否为 `dual_success_v3`，再看 `final_placement.failed_checks` 和逐帧诊断；仅改短max_steps会截断未成功的尝试，不能替代成功触发逻辑。

本次已完成两轮真实GPU闭环验证，均自动退出、正常保存；三路视频逐个经ffprobe读取实际帧数：

| 结果目录（位于 `/home/agilex/qhx/evaluation/sim/test/`） | success | strict_success | 成功确认时刻 | 三路视频各自时长/帧数 |
|---|---|---|---:|---:|
| `20261103_017`，seed104 | true | true | 9.03 s，第271帧 | 14.03 s，421帧 |
| `20261103_018`，seed106 | true | false | 11.10 s，第333帧 | 16.10 s，483帧 |

第二轮杯底中心偏移25.13 mm，超过严格上限21.5 mm，但普通success成立，仍按时结束，验证了strict_success不阻止退出。该轮第9秒杯子已稳定，但TCP仅撤离104.5 mm；第319帧（10.63秒）才超过120 mm，随后连续15帧满足条件，在11.10秒确认成功。因此保留现有撤离判据时，该轮不会在12秒结束；它确实是确认成功后再录5秒。

另有真实FFmpeg编码回归测试：普通success在第210帧/7秒确认、strict_success始终false，最终MP4为360帧/12秒。该测试验证计时与编码，不替代上面两轮真实策略闭环。

## 18. 当前规则：45秒、中断丢弃、单条复现和精简报告

### 18.1 时长及统计

未成功默认最多1350帧/45秒；在45秒以内确认success后，仍完整追加5秒，因此最迟约50秒结束。显式 `--max-steps` 可以覆盖上限；例如希望保持45秒请使用1350。批量的 `EPISODE_TIMEOUT` 是包含初始化的墙钟保护时间，与45秒仿真时间不同。

正常成功、正常到达45秒、成功后又丢失放置的条目保留并参与统计。Ctrl+C、关窗、GUI停止仿真、SIGTERM、运行异常、批量墙钟超时或子进程崩溃的当前条目丢弃：删除本条目录及批量中本条启动日志，不写入summary.csv，不计入成功率分母。已完成条目保留；不会自动补跑。

批次result.json只保留requested、finished（有效完成数）、discarded、两级成功数和两级成功率。success_rate=successes/finished；无有效结果时为null。单实例直接SIGKILL或断电无法执行进程内清理；批量管理器仍存活时会兜底清理被杀子进程的目录。请优先用Ctrl+C或GUI停止。

### 18.2 恢复某条场景并重新测试模型

保持模型服务运行，使用原结果目录，例如：

```bash
cd /home/agilex/qhx/InternDataEngine
bash scripts/run_cup_plate_pi05_eval.sh \
  --reproduce /home/agilex/qhx/evaluation/sim/test/20261103_013
```

默认显示窗口，加 `--headless` 则无头。读取原run.json中的有效参数、原task.yaml/arena.yaml；批量结果有scene_rng_before.json时恢复该轮随机数状态，没有时使用原seed。不要同时传 `--randomize`、`--seed` 或 `--scene-rng-state`。

原目录只读，新测试另建日期编号目录。由此无需回退到每条换seed；同一主seed的第n轮也可以直接恢复。复现依赖原资产未被替换；模型随机采样动作可能不同，因此恢复相同场景不保证重新推理仍重现同一次失败。

### 18.3 重放当时的失败动作

若要检查原失败过程，而不重新采样模型动作：

```bash
bash scripts/run_cup_plate_pi05_eval.sh \
  --reproduce /home/agilex/qhx/evaluation/sim/test/20261103_013 \
  --replay-saved-actions
```

读取trajectory.json中逐帧保存的predicted动作，无需模型服务，不生成NPZ。mode=replay、status=diagnostic，不能作为新的策略成功率样本。同一环境下物理可能存在数值差异；历史60秒失败默认只播放前45秒，如确需全长可显式加 `--max-steps 1800`。当前成功判据和成功后5秒停止仍生效。

真实验证：原018→新020，均483帧/16.1秒；逐帧杯子位姿、盘子位姿、14维状态和预测动作的最大绝对差均为0。

### 18.4 精简的单条result.json

```json
{
  "mode": "closed_loop",
  "status": "not_completed",
  "seed": 42,
  "episode": 3,
  "success": false,
  "strict_success": false,
  "reason": "max_steps",
  "simulation_seconds": 30.0,
  "post_success_seconds": 0.0,
  "failed_checks": ["cup_tilt"],
  "strict_failed_checks": ["cup_tilt", "footprint"]
}
```

成功时增加success_simulation_seconds（首次确认成功的仿真时刻）。每路主seed、episode保留，精确场景恢复使用旁边保存的RNG状态和配置。删除了报告中的大段阈值、位姿、视频元数据和重复路径。

需要细节时：run.json记录checkpoint、有效设置与复现来源；task.yaml/arena.yaml记录场景；trajectory.jsonl/trajectory.json记录每帧动作、状态、杯盘位姿和placement诊断；三路MP4固定命名。旧报告不改写，历史章节里的完整报告字段只适用于旧文件。32项单元/集成测试通过，包含被SIGTERM终止的子进程清理、剔除分母和45秒上限。

### 18.5 修改结果保存地址

编辑 `local_configs/cup_plate_pi05_eval/eval.yaml`：

```yaml
output_root: /你的绝对路径/pi05_test
```

单轮和批量入口共用此配置；新测试会在该路径下创建日期编号目录、summary.csv，批量汇总放在batches/中，不迁移已有结果。修改后重新启动测试，模型服务无需重启。

也可以复制eval.yaml为独立配置并添加 `--config /绝对路径/my_eval.yaml`；两个测试入口都支持。单轮 `--output /绝对路径/某一条目录` 只覆盖这一次的结果文件夹，目录必须尚不存在，根目录汇总仍使用配置中的output_root。复现模式恢复原run.json中的设置，故沿用原保存根目录；可用单轮--output明确指定新的单条目录。

## 普通success倾角放宽（当前）

`in_500/20261103_011` 的末帧倾角8.70°，仅因超过旧8°被判普通失败。现在普通success允许15°，strict_success仍保留8°和完整杯底支撑限制；其余条件保持原设置。

```yaml
success_criteria:
  success_max_tilt_deg: 15.0  # 普通success
  max_tilt_deg: 8.0           # strict_success
```

使用该条已保存的逐帧位姿、速度模长和TCP位置离线重新检查：普通成功在9.63秒确认，按5秒尾录规则将在14.63秒结束，strict_success为false。该核查不是重新运行模型，也未修改历史报告。33项测试通过。修改对新启动条目生效；正式对比统计应避免在同一批次混用新旧判据。
