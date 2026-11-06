# session_102914 原始数据与 π0.5 / LeRobot 训练就绪检查

检查日期：2026-10-29（Asia/Shanghai）。源目录：`/home/agilex/RL/robotcooking/files/data/session_102914`。

结论：数据完整，适合转换后用于“左臂抓纸杯放到盘子上”的小规模 π0.5 微调验证。它没有右臂运动示范，不能用这批数据验证双臂协同能力。19 条同场景示范不足以证明更广泛的摆放、光照和背景泛化能力。

本次读取了全部 38 个 JSON 文件，完整解码了全部 57 个视频，共 49,721 帧；视觉人工检查覆盖所有 19 个 episode 的顶部视角五个阶段，以及第 1、10、19 个 episode 的三相机五个阶段。未修改原始数据，未转换数据集，未安装依赖，未启动训练或连接机器人。

## 数据总览

| 项目 | 结果 |
| --- | --- |
| episode / 文件数量 | 19 / 95 |
| 总大小 | 291,440,223 bytes，约 278 MiB |
| 各轨迹顶部相机跨度 | 25.68–33.22 秒 |
| 顶部相机时间跨度合计 | 553.26 秒，约 9.22 分钟 |
| RGB 相机 | TOP_CAM、LEFT_WRIST_CAM、RIGHT_WRIST_CAM |
| 实际视频 | 全部 640×480、30 fps；没有深度视频 |
| 相机 JSON 样本数 | 顶部 16,605；左腕 16,554；右腕 16,562 |
| 机器人样本数 | 左臂 55,374；右臂 55,376，约 100 Hz |
| instruction | `pick up the paper cup and put it at the plate`，19 条一致 |
| outcome | 19 条均标记 success |
| 控制来源 | 两臂全部 `IS_INTERVENTION=3`，即独立人工专家示范 |

抽查图中 19 个 episode 的最终纸杯均在盘子上，和 success 标注一致；这并不等同于逐帧人工验证每次接触和抓取质量。物体起始位置有一定变化，但场景、相机和物体种类基本相同。

## 完整性与数值检查

- 57 个视频均可完整解码，各自解码帧数与其 JSON 数组长度一致；`frame_index` 从 0 连续递增。
- 五路数据各自时间戳均有限且严格递增；两臂 JOINT 都是 6 维，POSE 都是 7 维；JOINT、GRIPPER、POSE 未发现 NaN/Inf。
- 姿态四元数范数约为 1。该检查不包含与机器人 FK 的独立比对。
- 逐帧缩略图均值未发现全黑/全白帧；32×24 缩略图精确相等检查未发现超过 0.5 秒的连续重复图像。这不能排除局部遮挡或运动模糊。
- 左臂最大相邻 100 Hz 关节变化约 0.0332 rad，没有明显的大幅关节跳变；没有执行硬件关节限位和碰撞验证。
- 当前 Piper 驱动确认关节反馈单位是 rad，夹爪反馈是按物理量程归一化的 [0,1]。本批左夹爪为 0.002–0.694，右夹爪恒为 0。不要再把它们按米或毫米转换，也不要按本批最大值 0.694 重新缩放成 1。

## 影响训练的关键点

### 1. 实际上是左臂任务

右臂在全部 55,376 条样本中的六关节和夹爪完全不变：

```text
[0, -0.0022863813, -0.0083426738, 0, 0.5444031003, 0, 0]
```

视觉抽查也显示右臂停放不动。这批数据可以让 14 维模型学到“左臂操作、右臂保持”，但没有为右臂操作提供监督。14 维总体 loss 还包含七维恒定右臂，容易让平均 loss 显得更低；应额外关注左臂与夹爪误差及实际任务成功率。

| 选择 | observation.state / action | 相机 | 适用条件 |
| --- | --- | --- | --- |
| 保留现有双臂接口 | 14 维，左 6 关节+夹爪，再右 6 关节+夹爪 | 三路原名 | 继续对接同时挂载两臂的现有客户端，建议优先保留这个接口 |
| 左臂专用 | 7 维，左 6 关节+夹爪 | TOP_CAM、LEFT_WRIST_CAM | 训练与部署均明确切换为左臂；不会因为删除右臂而自动适配现有双臂客户端 |

当前客户端按挂载机器人计算 action 维度，两臂一起挂载时期待 14 维；不要把 7 维 checkpoint 直接接入该配置。π0.5 的 32 维上限是模型内部 padding 上限，不代表需要人为在数据集补到 32 维。

### 2. action 不是原始遥操作命令

原始 JSON 只有 SLAVE_LEFT / SLAVE_RIGHT 反馈，没有 MASTER 或显式控制目标。现有转换器默认：

```text
state(t)  = 最近于 TOP_CAM 时间戳 t 的从臂关节/夹爪反馈
image(t)  = 最近于 t 的对应相机图像
 action(t) = 最近于 TOP_CAM 第 t+2 帧时间戳的从臂关节/夹爪反馈
```

两帧约 66.7 ms。这是用未来达到的状态作为动作代理，可作为先跑通任务的基线；它不是采集到的实际目标指令，也不能由这批数据证明恰好对应控制延迟。保留 `future_offset_frames=2` 时，训练和部署都应理解其含义。

不建议为了“对齐”直接改成 offset=0：这会让当前定义下的 action 和 state 完全相同。若新采集数据，应同时记录发送给从臂的命令及其时间戳、主臂目标和从臂反馈，从而直接确定动作标签和控制延迟。

### 3. 要按 timestamp 对齐，不能按数组下标拼接

相机约 30 Hz、机器人约 100 Hz，各路长度不同属于正常情况。使用 TOP_CAM 作为时间参考，最近邻对齐容差 25 ms：

- 左腕相机每条轨迹的 P99 最近邻误差约 17.45–17.99 ms，右腕约 17.27–18.20 ms。
- 单路最大相邻间隔：顶部 62.85 ms、左腕 67.24 ms、右腕 68.41 ms；不能仅凭 MP4 标称 30 fps 假定每个实际间隔精确为 33.33 ms。
- 部分录制边界存在 80–112 ms 最近邻误差。按转换器默认 offset=2 和 25 ms 容差，19 条合计先去掉 38 个没有未来标签的尾帧，再剔除 6 帧超容差样本，预计得到 **16,561 帧**。
- 这 6 帧来自 145618、145757（2 帧）、150152、150622 的尾部，以及 150744 的首部；没有必要因此删除整个 episode。
- 14 维/三相机与 7 维/顶部+左腕两种配置的对齐模拟，结果均为 16,561 帧。

这些误差反映记录时间戳之间的一致性。当前机器人时间戳来自主机读取循环，未证明所有传感器在曝光/硬件反馈层面同步，也未独立测量相机内部流水线延迟。

### 4. 首尾静止较多，保留必要抓取停顿

左臂多数 episode 起始连续不变约 2–5.6 秒，结束后停放约 2–4.7 秒。它们会增加“不动作”样本比例。可以在派生训练集上适度裁剪首尾等待，保留约 0.5–1 秒上下文和任务完成后的短暂停留。不要全局删除静止帧，否则夹持、接触确认、释放等合理停顿也会被误删。

`audit.json` 给出了完整的逐轨迹静止区间；其中基于首尾值差 0.01 的 `suggested_idle_review` 只是粗略筛查，部分轨迹开头微动会使该指标失真，不可直接当裁剪边界。

### 5. 训练数据量与验证

先用本批做流程验证和同场景任务微调是合理的；不要只看训练 loss 判断成功。按完整 episode 留出约 2 条，避免相邻帧随机切分造成泄漏，并在未用于训练的杯/盘位置上验证成功率。后续优先补充位置、相对距离、朝向、抓取姿态变化，而不是只重复相同的起点。

## 转换命令示例（未执行）

保留当前双臂接口，明确单位和 offset，输出放在当前工作区的派生目录：

```bash
cd /home/agilex/RL/robotcooking
/home/agilex/miniforge3/envs/lerobot/bin/python -m sdk.lerobotv3_sdk.convert_robotcooking \
  /home/agilex/RL/robotcooking/files/data/session_102914 \
  --output-root /home/agilex/qhx/InternDataEngine/artifacts/session_102914_lerobot_v3 \
  --repo-id local/session_102914 \
  --joint-unit radian \
  --gripper-unit normalized \
  --future-offset-frames 2 \
  --alignment-tolerance-sec 0.025 \
  --image-width 256 --image-height 256
```

现有转换器会保持宽高比，640×480 缩到 256×192 后上下补黑边到 256×256；π0.5 在模型内再处理为 224×224。部署端需保持相同的颜色通道和缩放补边约定。原始相机名应在训练和部署之间保持一致。

如已确定单臂部署，在上面命令追加：

```bash
--arms left --cameras TOP_CAM LEFT_WRIST_CAM
```

转换后还应实际用 LeRobotDataset 读取首尾和随机样本，确认 frame/episode 数量、图像通道、state/action 维度，并核对 `meta/stats.json` 包含 state/action 的 q01、q99 分位数。这里只模拟了转换器的对齐规则，尚未生成或读取最终 LeRobot 数据集。

## π0.5 训练起点（未执行）

以下为本地源代码版本可识别的配置示例，需先完成转换、依赖安装和 GPU 验证。它用于小规模基线，不是经过调参验证的最终配方：

```bash
/home/agilex/miniforge3/envs/lerobot/bin/lerobot-train \
  --dataset.repo_id=local/session_102914 \
  --dataset.root=/home/agilex/qhx/InternDataEngine/artifacts/session_102914_lerobot_v3 \
  --dataset.eval_split=0.1 \
  --policy.type=pi05 \
  --policy.pretrained_path=lerobot/pi05_base \
  --policy.use_relative_actions=false \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true \
  --policy.train_expert_only=true \
  --policy.chunk_size=50 \
  --policy.n_action_steps=10 \
  --policy.push_to_hub=false \
  --batch_size=4 \
  --steps=3000 \
  --eval_steps=500 \
  --save_freq=500 \
  --wandb.enable=false \
  --output_dir=/home/agilex/qhx/InternDataEngine/artifacts/pi05_session_102914_baseline \
  --job_name=pi05_session_102914_baseline
```

说明：数据集使用绝对关节目标，先保持 `use_relative_actions=false` 以减少处理差异；如后续切相对动作，应连同 action 统计和后处理一起切换。`chunk_size=50` 在 30 Hz 下约 1.67 秒；`n_action_steps=10` 表示每次执行约 0.33 秒后更新，但还需与实际部署的 RTC/执行器配置一致。使用本地 `policy.type + pretrained_path` 路径时，相机特征由数据集定义；不应凭空套用 LIBERO 的相机命名。

## 当前环境的实际限制

已检查 `/home/agilex/miniforge3/envs/lerobot`：安装元数据为 LeRobot 0.6.2、Torch 2.11.0；实际源码来自 `/home/agilex/RL/lerobot/src/lerobot`。转换器所需的 LeRobot v3 Dataset/Encoder API 可以导入，创建签名包含现有转换器使用的参数。

这个环境目前没有 transformers、accelerate；也没有 peft（只有选择 LoRA 等方案时才需要单独考虑）。π0.5 类能因延迟导入机制被 import，不等于能够实例化训练；其构造函数会检查 transformers。当前本地 pyproject 推荐的相关安装入口为 `.[pi,train]`，不要只凭别处的旧教程固定 transformers 版本。预训练权重和 tokenizer 的可下载性、访问权限、缓存完整性尚未检查。

当前执行环境运行 `nvidia-smi` 无法连接 NVIDIA 驱动，因此未能确认 GPU 型号、可用显存和训练可运行性；这不等于已经证明整台机器没有 GPU。若实际训练在另一台服务器，应在那台服务器检查环境。

额外部署核对：旧版 `online_infer/client.py` 中 `MODEL_ACTION_DT=0.03`，与训练数据 30 Hz 的 `1/30≈0.03333` 秒不同；同文件还会对下发夹爪值额外乘 0.8。该发现针对这条旧客户端代码路径，未确认它就是你当前启用的部署路径；上线前需核对实际使用的客户端，避免动作节拍和夹爪尺度发生变化。

## 逐轨迹结果

默认：offset=2，最近邻容差 25 ms。drop 索引是 TOP_CAM 的 0 起始下标，不含每条固定去掉的末尾 2 帧。

| 索引 | Episode | 时长/秒 | 顶部帧数 | 预计保留 | 超容差 drop 索引 |
| --- | --- | --- | --- | --- | --- |
| 0 | episode_20261029_145332 | 29.39 | 882 | 880 | [] |
| 1 | episode_20261029_145511 | 29.55 | 887 | 885 | [] |
| 2 | episode_20261029_145618 | 28.65 | 860 | 857 | [857] |
| 3 | episode_20261029_145701 | 33.22 | 997 | 995 | [] |
| 4 | episode_20261029_145757 | 32.12 | 964 | 960 | [960, 961] |
| 5 | episode_20261029_145932 | 26.79 | 804 | 802 | [] |
| 6 | episode_20261029_150014 | 28.15 | 845 | 843 | [] |
| 7 | episode_20261029_150056 | 32.46 | 974 | 972 | [] |
| 8 | episode_20261029_150152 | 27.49 | 825 | 822 | [822] |
| 9 | episode_20261029_150237 | 30.09 | 903 | 901 | [] |
| 10 | episode_20261029_150319 | 31.12 | 934 | 932 | [] |
| 11 | episode_20261029_150408 | 28.75 | 863 | 861 | [] |
| 12 | episode_20261029_150457 | 28.25 | 848 | 846 | [] |
| 13 | episode_20261029_150537 | 26.95 | 809 | 807 | [] |
| 14 | episode_20261029_150622 | 29.89 | 897 | 894 | [894] |
| 15 | episode_20261029_150705 | 26.72 | 802 | 800 | [] |
| 16 | episode_20261029_150744 | 29.39 | 882 | 879 | [0] |
| 17 | episode_20261029_150834 | 25.68 | 771 | 769 | [] |
| 18 | episode_20261029_150911 | 28.59 | 858 | 856 | [] |

## 检查材料与依据

- [完整机器可读检查结果](audit.json)
- [前 10 条顶部视角抽帧](top_contact_1.jpg)
- [后 9 条顶部视角抽帧](top_contact_2.jpg)
- [第 1 条三相机抽帧](three_cameras_145332.jpg)
- [第 10 条三相机抽帧](three_cameras_150237.jpg)
- [第 19 条三相机抽帧](three_cameras_150911.jpg)
- [本批数据检查脚本](audit.py)，运行会重新读取原始数据并在脚本所在目录写出检查结果和抽帧。

本地依据：RobotCooking 的 `sdk/lerobotv3_sdk/convert_robotcooking.py`、`device_manager/arm/piper.py`、`device_manager/robot.py`、`data_manager/record.py`、`online_infer/client.py`、`online_infer/client_rtc.py`；LeRobot 的 `docs/source/pi05.mdx`、`policies/pi05/configuration_pi05.py`、`processor_pi05.py`、`configs/default.py`、`pyproject.toml`。外网文档验证尝试失败，因此版本行为以这台机器的本地代码为准。
