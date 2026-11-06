# π0.5 微调模型的普通同步推理部署

本例针对本机 LeRobot **0.5.1** 格式的 π0.5 checkpoint。只加载微调模型和它的预处理/反归一化，不加载 Actor、Critic、RLToken，不使用 RTC、EMA、轨迹插值或异步预取。服务本身不连接机械臂。

## 1. 模型与环境

```bash
source /home/agilex/miniforge3/etc/profile.d/conda.sh
conda activate evo-rlt
```

当前可用模型目录：

```text
/home/agilex/model/pi05/102214_0.5.1/pretrained_model
```

保留整个 `pretrained_model` 目录：`model.safetensors`、`config.json`、`policy_preprocessor.json`、`policy_postprocessor.json`，以及两个 processor 引用的 normalizer/unnormalizer `.safetensors` 文件。只复制模型权重不够。

tokenizer 使用本机 `/home/agilex/model/paligemma_3b`，覆盖 checkpoint 中已失效的 `/datapool/...` 路径。不必修改 checkpoint。

| 输入 | 传入格式 |
|---|---|
| `observation.state` | `float32[7]`，右臂实测 J1–J6 弧度、归一化夹爪开度 |
| `observation.images.RIGHT_WRIST_CAM` | RGB、uint8、HWC、256×256×3 |
| `observation.images.TOP_CAM` | RGB、uint8、HWC、256×256×3 |
| `task` | 当前任务文字，例如 `insert pen` |

输出为 **50×7** 的动作块，已用 checkpoint 的统计量反归一化。本机这套数据约定下是绝对关节目标（弧度）和归一化夹爪开度，不能再次当作增量或再做一次反归一化。新模型必须按其训练数据约定核对这些含义。

## 2. 启动模型服务

先确认 GPU 有足够剩余显存：原有 8096 模型仍在运行时，换一个 HTTP 端口不会释放它占用的显存。需要先在原终端停止不再使用的模型服务，或使用另一块 GPU。

```bash
python /home/agilex/qhx/InternDataEngine/scripts/deploy/pi05_plain_server.py serve \
  --checkpoint /home/agilex/model/pi05/102214_0.5.1/pretrained_model \
  --tokenizer /home/agilex/model/paligemma_3b \
  --port 8097
```

另一个终端检查：

```bash
curl --noproxy '*' http://127.0.0.1:8097/health
```

应显示 `rtc_enabled: false`、`action_dim: 7`、`chunk_size: 50` 和正确相机字段。服务严格检查权重加载是否完成；遇到不匹配会拒绝启动，不应忽略缺失权重提示。

本例默认只监听本机。跨机时可用 SSH 转发机器人端本地 8097 到模型机器 8097，客户端地址仍使用 `http://127.0.0.1:8097`。

## 3. 先验证一次模型请求

客户端可在 `robotcooking` 环境运行，不需要在机器人端安装 PyTorch。

从真实观测生成 NPZ；以下变量应取自同一时刻附近的相机/机械臂反馈：

```python
import numpy as np

observation = {
    "observation.state": np.asarray([*joints_rad, gripper_01], dtype=np.float32),
    "observation.images.RIGHT_WRIST_CAM": wrist_rgb_256,
    "observation.images.TOP_CAM": top_rgb_256,
}
np.savez_compressed("/tmp/pi05_observation.npz", **observation)
```

这里 `joints_rad` 是真实右臂 6 个关节读数；两张图像是 RGB uint8。现有 Orbbec 驱动给的是 BGR，先转 RGB，再按训练部署约定保持纵横比缩放、黑边填充到 256×256；不要直接拉伸，不要将图像预先归一化成浮点数组。

```bash
conda activate robotcooking
python /home/agilex/qhx/InternDataEngine/scripts/deploy/pi05_plain_server.py infer \
  --observation /tmp/pi05_observation.npz \
  --task 'insert pen' \
  --output /tmp/pi05_actions.npy
```

预期打印 `actions=(50, 7)` 与推理耗时。该命令只保存动作数组，不会移动机械臂。黑图或全零状态只能验证接口形状，不应把它们的输出送到真机。

## 4. 接入真机：普通同步循环

模型端只算动作，机器人端读取观测、请求动作并下发。最简单的同步流程是：

```text
读取两路最新 RGB 和右臂状态
          ↓
调用 /infer，等待 50×7 动作块
          ↓
按训练/采集频率执行前 K 个动作
          ↓
舍弃本块剩余动作，重新读取观测
```

可以直接导入本例提供的请求函数：

```python
import sys
import time

sys.path.insert(0, "/home/agilex/qhx/InternDataEngine/scripts/deploy")
from pi05_plain_server import infer_remote

# 以下三个函数接你现有硬件层：
# read_observation(): 返回上节的 observation 字典
# send_checked_action(a): 校验并发送 7 维绝对目标
# should_stop(): 返回停止按钮/退出信号状态
fps = 50.0  # 本机现有模型采用的采集动作频率；其他模型按训练数据确认
execute_steps = 10
while not should_stop():
    observation = read_observation()
    actions, model_seconds = infer_remote(
        "http://127.0.0.1:8097", observation, "insert pen"
    )
    for action in actions[:execute_steps]:
        if should_stop():
            break
        started = time.monotonic()
        send_checked_action(action)
        time.sleep(max(0.0, 1.0 / fps - (time.monotonic() - started)))
```

这是硬件接线示例，需要补全上述三个函数，不是独立真机启动脚本。选择 `K=10` 时一轮动作约 0.2 秒，再加一次推理等待时间；选择 `K=50` 时约 1 秒。无 RTC、无预取时，块间有等待是正常现象。50 Hz 是块内目标下发频率，不是每秒执行 50 次模型推理。

本机现有硬件接口可参考 `robotcooking/device_manager/arm/piper.py`：`get_joint()`、`get_gripper()`、`set_joint()`、`set_gripper()`；后两者可实现下发层。发送前保留动作维度、有限值、机械臂真实关节限位和夹爪范围校验。网络异常、用户停止时终止动作块并保持当前姿态，不要发送全零关节目标。

不要直接切换到现有 `legacy` 客户端来替代上述循环：它还有固定 30 ms 动作间隔、轨迹执行器及夹爪缩放，不能代表这套 50 Hz 模型的原始同步推理。

## 本次验证范围

已检查本机模型配置与 LeRobot 0.5.1 API；通过 meta 模型与 safetensors 文件头核对了全部 813 个权重字段，名称与形状一致；用模拟推理引擎验证了健康接口、NPZ 请求传输、50×7 响应及无效输入拒绝。没有启动第二个 GPU 模型、没有执行真实模型前向，也没有下发真机动作。
